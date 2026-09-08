#!/usr/bin/env python3
"""SO-101 interactive control with Viser 1.1.0 and LeRobot 0.6.1.

Goal, plan, execute, the way a planner front end works: dragging the handle or moving
a joint slider sets the orange goal; Plan trajectory computes the path and runs the
orange model along it; Execute follows it. Model mode (default) moves the model only.
--hardware connects the Follower bus: readings are live from the start, motion is sent
only after the operator arms the arm by typing ENABLE, every command is a small joint
step at a limited rate, the arm is watched for lag, Stop holds the current position, and
EMERGENCY STOP cuts torque. There is no collision detection: the operator watches the
arm and keeps the DC cutoff within reach. Keep so101_cartesian_demo.py in the same directory.
"""
from __future__ import annotations

import argparse
import math
import os
import queue
import socket
import sys
import threading
import time
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

import numpy as np

import so101_cartesian_demo as control

VISER_VERSION = "1.1.0"
LOOPBACK = "127.0.0.1"  # Hardware control must never bind a LAN/public interface.
UPDATE_SECONDS = 0.05  # 20 Hz UI, readback and command loop; not a real-time guarantee.
MOTION_RATE_DEG_S = 10.0  # Joint speed for hardware execution and the model-mode execute.
GRIPPER_RATE_PCT_S = 20.0  # Gripper opening speed; its eased peak stays under LeRobot's 2 %-per-command clip.
ANIMATION_MIN_SECONDS = 2.0  # Execution of even a tiny move takes at least this long.
PREVIEW_RATE_DEG_S = 45.0  # The orange model only: fast enough to watch, nothing physical behind it.
PREVIEW_MIN_SECONDS = 1.0
TRACKING_ABORT_DEG = 8.0  # The arm lags its command by more than this: stop and hold.
SETTLE_TOLERANCE_DEG = 0.8
SETTLE_TIMEOUT_SECONDS = 2.0
REPLAN_TOLERANCE_DEG = 0.5  # The arm moved since planning: plan again.
MAX_REQUEST_AGE_SECONDS = 0.5  # Reject queued execution requests rather than replay them.
SLIDER_STEP_DEG = 0.5
ARM_WORD = "ENABLE"
TRAIL_SAMPLES = 48
TRAIL_COLOR = (255, 156, 31)
PREVIEW_GRIPPER_RAD = 0.0  # Geometry reference only: not mapped from LeRobot percent.
CURRENT_COLOR = (0.18, 0.48, 0.72, 1.0)
TARGET_COLOR = (1.0, 0.61, 0.12, 0.35)
JOINT_LABELS = ("F1 shoulder_pan", "F2 shoulder_lift", "F3 elbow_flex", "F4 wrist_flex", "F5 wrist_roll")


@dataclass(frozen=True)
class Plan:
    start: tuple[float, ...]
    goal: tuple[float, ...]
    gripper_start: float | None
    gripper_goal: float | None
    xyz: tuple[float, ...]
    duration: float


@dataclass(frozen=True)
class Request:
    kind: str
    client_id: int | None
    created: float
    payload: tuple[float, ...] | str = ()


def eased(fraction):
    """Smooth start and stop; still 0 at 0 and 1 at 1."""
    if not math.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError("Invalid animation fraction")
    return 0.5 - 0.5 * math.cos(math.pi * fraction)


def interpolation(start, target, fraction):
    """Pure joint interpolation. Never extrapolate after a delayed UI event."""
    if not math.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError("Invalid interpolation fraction")
    if len(start) != len(control.JOINTS) or len(target) != len(control.JOINTS):
        raise ValueError("Expected five arm joints")
    if not all(math.isfinite(float(v)) for v in (*start, *target)):
        raise ValueError("Non-finite joint value")
    return [float(a + (b - a) * fraction) for a, b in zip(start, target)]


def motion_duration(start, goal, gripper_change=0.0, minimum=0.0):
    """Seconds so that no joint exceeds MOTION_RATE_DEG_S and the gripper its own rate."""
    joints = max(abs(float(b) - float(a)) for a, b in zip(start, goal)) / MOTION_RATE_DEG_S
    gripper = abs(float(gripper_change)) / GRIPPER_RATE_PCT_S
    return max(joints, gripper, minimum, UPDATE_SECONDS)


def preview_duration(start, goal):
    """Seconds the orange model takes to show a plan; independent of the execution speed."""
    return max(max(abs(float(b) - float(a)) for a, b in zip(start, goal)) / PREVIEW_RATE_DEG_S, PREVIEW_MIN_SECONDS, UPDATE_SECONDS)


def trail_points(kinematics, start, goal, samples=TRAIL_SAMPLES):
    """Gripper positions along the joint interpolation a motion will follow."""
    if samples < 2:
        raise ValueError("A trail needs at least two samples")
    return np.array([kinematics.forward_kinematics(np.array(interpolation(start, goal, i / (samples - 1))))[:3, 3]
                     for i in range(samples)])


def motion_steps(duration):
    """Ticks a motion takes; steps, not wall-clock, so a slow tick cannot enlarge a step."""
    return max(1, math.ceil(float(duration) / UPDATE_SECONDS))


def lag(current, commanded):
    """Largest joint distance between the arm and what it was last told."""
    return max(abs(float(a) - float(b)) for a, b in zip(current, commanded))


def intersect_limits(model_limits, reading_limits):
    """Targets must be reachable by the model and inside the arm's recorded travel."""
    return {name: (max(model_limits[name][0], reading_limits[name][0]), min(model_limits[name][1], reading_limits[name][1]))
            for name in control.JOINTS}


def slider_bounds(limits):
    """Slider ends rounded inwards to the slider step, so every slider value is a valid target."""
    factor = 1 / SLIDER_STEP_DEG
    return {name: (math.ceil(low * factor) / factor, math.floor(high * factor) / factor) for name, (low, high) in limits.items()}


def slider_value(value, bounds):
    """Round to the slider step and clamp inside its ends; the servo reading itself is never altered."""
    low, high = bounds
    return float(min(max(round(float(value) / SLIDER_STEP_DEG) * SLIDER_STEP_DEG, low), high))


def ik_seed(current, previous_goal, limits):
    """Where the solver starts: the last goal while dragging, else the arm clamped into the model.

    An arm resting on a mechanical stop sits a few degrees outside the URDF limits; clamping the
    seed lets a goal be set from there, while the plan itself still starts at the real pose.
    """
    if previous_goal is not None:
        return np.asarray(previous_goal, dtype=float)
    lows = np.array([limits[name][0] for name in control.JOINTS])
    highs = np.array([limits[name][1] for name in control.JOINTS])
    return np.clip(np.asarray(current, dtype=float), lows, highs)


def counts_from_degrees(degrees, calibration):
    """Encoder counts the calibration table would read for these degrees: zero is the range middle."""
    return int(round((calibration.range_min + calibration.range_max) / 2 + float(degrees) * 4095 / 360))


def counts_from_percent(percent, calibration):
    """Encoder counts for a gripper opening percentage over its recorded range."""
    return int(round(calibration.range_min + float(percent) / 100 * (calibration.range_max - calibration.range_min)))


def arming_requested(text):
    return text.strip() == ARM_WORD


def check_request(request, owner, now, connected_ids):
    if request.client_id != owner or set(connected_ids) != {owner}:
        raise ValueError("Only the original connected browser may command the arm")
    if not 0 <= now - request.created <= MAX_REQUEST_AGE_SECONDS:
        raise ValueError("Request expired; try again")


def viewer_configuration(names, joints_deg):
    """Map by URDF joint name. Gripper geometry deliberately stays a reference."""
    by_name = dict(zip(control.JOINTS, map(math.radians, joints_deg)))
    by_name["gripper"] = PREVIEW_GRIPPER_RAD
    if set(names) != set(by_name):
        raise ValueError("Unexpected actuated joints in the pinned visual model")
    return [by_name[name] for name in names]


def validate_web_port(port):
    if not 1024 <= port <= 65535:
        raise ValueError("Choose an explicitly assigned unprivileged web port")
    if os.environ.get("_VISER_PORT_OVERRIDE"):
        raise ValueError("Unset _VISER_PORT_OVERRIDE; use the explicit --web-port")
    with socket.socket() as probe:
        # Connections left in TIME_WAIT by the previous run must not block a restart;
        # a live listener on the port still fails the bind and is reported.
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind((LOOPBACK, port))


def run(args):
    import viser
    from viser.extras import ViserUrdf

    if version("viser") != VISER_VERSION:
        raise RuntimeError(f"Use viser[urdf]=={VISER_VERSION}")
    zh = args.locale == "cn"
    copy = lambda chinese, english: chinese if zh else english
    kinematics, model_limits = control.load_kinematics(args.model_dir)
    reading_limits = target_limits = model_limits
    arm = None
    server = None
    armed = False
    estopped = False
    requests: queue.Queue[Request] = queue.Queue(maxsize=64)
    stop_requested = threading.Event()
    owner = None
    owner_lost = False
    lock = threading.Lock()
    latest_drag = None
    latest_joints = None
    goal = None  # (joints, gripper, xyz): where the operator wants to go
    plan = None  # Plan: how to get there, computed on request
    phase = "idle"  # idle | planning (ghost runs the path) | executing
    step = 0
    last_command = None
    trail = []

    def enqueue(kind, client_id=None, payload=()):
        # Callbacks only enqueue immutable input. IK, bus I/O and motion state live on
        # the main thread; Viser callbacks run concurrently.
        nonlocal latest_drag, latest_joints
        if kind == "quit":
            if armed:
                kind = "quit-refused"
            else:
                stop_requested.set()
                return
        if client_id is not None and client_id != owner and kind not in ("connected", "estop"):
            return
        if kind in ("drag", "joints"):
            with lock:  # Coalesce: only the latest slider or drag position matters.
                request = Request(kind, client_id, time.monotonic(), tuple(payload))
                if kind == "drag":
                    latest_drag = request
                else:
                    latest_joints = request
            return
        try:
            requests.put_nowait(Request(kind, client_id, time.monotonic(), payload))
        except queue.Full:
            pass  # Dropped input is safer than a replayed backlog.

    def read_pose():
        observation = arm.get_observation()
        q = np.array([observation[f"{name}.pos"] for name in control.JOINTS], dtype=float)
        control.validate_joints(q, reading_limits)
        opening = float(observation["gripper.pos"])
        if not math.isfinite(opening) or not 0 <= opening <= 100:
            raise ValueError("Gripper readback is outside its calibrated range")
        return q, opening

    def hold_here(current, opening):
        """Command the present position so a stopped arm neither drifts nor drops."""
        action = {f"{name}.pos": float(v) for name, v in zip(control.JOINTS, current)}
        action["gripper.pos"] = float(opening)
        arm.send_action(action)

    try:
        if args.hardware:
            from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
            arm = SO101Follower(SO101FollowerConfig(
                port=args.port, id=args.robot_id, calibration_dir=Path(args.calibration_dir),
                use_degrees=True, max_relative_target=control.MAX_JOINT_STEP_DEG,
                cameras={}, disable_torque_on_disconnect=False,
            ))
            # Do not use Robot.connect(): its generic configuration enables torque.
            arm.bus.connect()
            if not arm.calibration or not arm.is_calibrated:
                raise RuntimeError("Calibration missing or mismatched; return to Communication and Calibration")
            if any(arm.bus.read("Torque_Enable", name, normalize=False) != 0 for name in arm.bus.motors):
                raise RuntimeError("Torque already enabled; cut power and investigate")
            reading_limits = control.calibrated_limits(arm)
            target_limits = intersect_limits(model_limits, reading_limits)
            current, opening = read_pose()
        else:
            current = np.array(control.PREVIEW_JOINTS_DEG, dtype=float)
            opening = None
        control.validate_joints(current, reading_limits)

        validate_web_port(args.web_port)
        server = viser.ViserServer(host=LOOPBACK, port=args.web_port, label="SO-101 · Pose Viewer")
        if server.get_port() != args.web_port:
            raise RuntimeError("Assigned port became unavailable; stopping the fallback listener")
        server.scene.set_up_direction("+z")
        server.initial_camera.position = (0.85, -0.7, 0.6)
        server.initial_camera.look_at = (0.15, 0.0, 0.14)
        server.scene.add_grid("/ground", width=1.2, height=1.2, plane="xy", cell_size=0.05)
        server.scene.add_frame("/base", axes_length=0.12, axes_radius=0.003)
        server.scene.add_frame("/current", show_axes=False)
        ghost_root = server.scene.add_frame("/goal", show_axes=False, visible=False)
        model_path = Path(args.model_dir) / control.URDF_NAME
        actual_model = ViserUrdf(server, model_path, root_node_name="/current", mesh_color_override=CURRENT_COLOR)
        ghost_model = ViserUrdf(server, model_path, root_node_name="/goal", mesh_color_override=TARGET_COLOR)
        names = actual_model.get_actuated_joint_names()
        xyz = kinematics.forward_kinematics(current)[:3, 3].copy()
        # Arrows move along one axis, the plane handles move in two: free dragging, position only.
        gizmo = server.scene.add_transform_controls(
            "/target", position=tuple(xyz), scale=0.13, disable_rotations=True, disable_sliders=False, line_width=4.0,
        )

        def joints_line(joints, gripper):
            """Degrees always; encoder counts beside them when a calibration is loaded, like km/h and mph."""
            if arm:
                parts = [f"F{i + 1} {v:.1f}° ({counts_from_degrees(v, arm.calibration[name])})" for i, (name, v) in enumerate(zip(control.JOINTS, joints))]
                gripper_part = f"{copy('夹爪', 'Gripper')} {gripper:.1f}% ({counts_from_percent(gripper, arm.calibration['gripper'])})"
                return " · ".join(parts) + f"\n\n{gripper_part}"
            return " · ".join(f"F{i + 1} {v:.1f}°" for i, v in enumerate(joints))

        def mode_label():
            if estopped:
                return copy("紧急停止 · 力矩已切断，重开程序才能再启用", "Emergency stop · torque cut, restart the program to enable again")
            if not arm:
                return copy("模型模式 · 未连接硬件", "Model mode · no hardware")
            return copy("实机模式 · 力矩已启用，执行会动真机", "Hardware mode · torque on, Execute moves the arm") if armed \
                else copy("实机模式 · 力矩未启用，只读", "Hardware mode · torque off, reading only")

        mode_text = server.gui.add_markdown(f"### {mode_label()}")
        server.gui.add_markdown(copy(
            "蓝色：当前姿态，实机模式下来自电机回读。橙色：目标姿态。拖动三轴或平面手柄、或拨动关节滑杆设定目标；“计划轨迹”算出路径并让橙色走一遍；“执行运动”才真的过去。旋转视角不改变底座坐标。",
            "Blue: current pose, read from the motors in hardware mode. Orange: goal pose. Drag the axes or plane handles, or move the joint sliders, to set a goal; Plan trajectory computes the path and runs the orange model along it; Execute is what actually moves. Camera rotation does not change the base frame.",
        ))
        readout = server.gui.add_markdown("")
        goal_text = server.gui.add_markdown("")
        status = server.gui.add_markdown(copy("拖动目标或拨动滑杆，先出现橙色目标。", "Drag the target or move a slider; the orange goal appears first."))
        bounds = slider_bounds(target_limits)
        with server.gui.add_folder(copy("目标关节角 / °", "Goal joints / °")):
            sliders = [server.gui.add_slider(label, bounds[name][0], bounds[name][1], SLIDER_STEP_DEG, slider_value(v, bounds[name]))
                       for label, name, v in zip(JOINT_LABELS, control.JOINTS, current)]
            gripper_slider = server.gui.add_slider(copy("夹爪开合 / %", "Gripper / %"), 0, 100, 1, float(opening)) if arm else None
        plan_button = server.gui.add_button(copy("计划轨迹", "Plan trajectory"), disabled=True,
                                            hint=copy("算出从当前到目标的路径，橙色模型走一遍，实物不动", "Compute the path from the current pose to the goal; the orange model runs it, the arm does not move"))
        execute = server.gui.add_button(copy("执行运动", "Execute"), disabled=True,
                                        hint=copy("沿计划好的轨迹过去；实机模式需先启用力矩", "Follow the planned trajectory; hardware mode requires torque on"))
        stop = server.gui.add_button(copy("停止并保持", "Stop and hold"), disabled=True,
                                     hint=copy("停在当前位置，电机继续出力", "Hold the current position with torque on"))
        reset = server.gui.add_button(copy("目标回到当前姿态", "Reset goal to current pose"))
        if arm:
            with server.gui.add_folder(copy("力矩", "Torque")):
                arm_word = server.gui.add_text(copy(f"输入 {ARM_WORD} 再点启用", f"Type {ARM_WORD}, then enable"), "")
                arm_button = server.gui.add_button(copy("启用力矩并保持当前姿态", "Enable torque and hold this pose"))
                release_button = server.gui.add_button(copy("释放力矩", "Release torque"), disabled=True,
                                                       hint=copy("先托住手臂：释放后六个电机不再出力，手臂会下落", "Support the arm first: after release the six motors stop driving and the arm drops"))
        estop_button = server.gui.add_button(copy("紧急停止", "EMERGENCY STOP"), color="red",
                                             hint=copy("立刻停止并切断力矩，手臂会下落；之后要重开程序", "Stop at once and cut torque; the arm drops; restart the program afterwards"))
        quit_button = server.gui.add_button(copy("关闭程序", "Close program"))
        server.gui.add_markdown(copy(
            "三种停法：“停止并保持”停在原地、电机继续出力；“释放力矩”是单独的一步，先托住手臂再点；“紧急停止”立刻切断全部力矩，手臂会掉。执行时每次只发一小步，速度上限每秒 10°，实物落后指令超过 8° 自动停止并保持。力矩开着时不能关闭程序；终端 Ctrl+C 会直接卸力。软件不是物理断电。",
            "Three ways to stop: Stop and hold keeps the motors driving in place; Release torque is a separate step, support the arm first; EMERGENCY STOP cuts all torque at once and the arm drops. Execution sends one small step at a time, at most 10° per second, and stops and holds if the arm lags by more than 8°. The program cannot be closed while torque is on; Ctrl+C in the terminal releases torque at once. Software is not a physical power cutoff.",
        ))

        @server.on_client_connect
        def connected(client):
            enqueue("connected", client.client_id)

        @server.on_client_disconnect
        def disconnected(client):
            nonlocal owner_lost
            if client.client_id == owner:
                owner_lost = True

        @gizmo.on_update
        def dragged(event):
            enqueue("drag", event.client_id, tuple(gizmo.position))

        def slider_goal():
            return tuple(float(s.value) for s in sliders) + ((float(gripper_slider.value),) if gripper_slider else ())

        for slider in sliders + ([gripper_slider] if gripper_slider else []):
            @slider.on_update
            def moved(event):
                enqueue("joints", event.client_id, slider_goal())

        plan_button.on_click(lambda event: enqueue("plan", event.client_id))
        execute.on_click(lambda event: enqueue("execute", event.client_id))
        stop.on_click(lambda event: enqueue("stop", event.client_id))
        reset.on_click(lambda event: enqueue("reset", event.client_id))
        estop_button.on_click(lambda event: enqueue("estop", event.client_id))
        if arm:
            arm_button.on_click(lambda event: enqueue("arm", event.client_id, arm_word.value))
            release_button.on_click(lambda event: enqueue("release", event.client_id))
        quit_button.on_click(lambda event: enqueue("quit", event.client_id))

        def clear_trail():
            for handle in trail:
                handle.remove()
            trail.clear()

        def refresh_buttons():
            plan_button.disabled = estopped or goal is None or phase != "idle"
            execute.disabled = estopped or plan is None or phase != "idle" or (bool(arm) and not armed)
            stop.disabled = phase != "executing"
            if arm:
                arm_button.disabled = estopped or armed
                release_button.disabled = estopped or not armed
            mode_text.content = f"### {mode_label()}"

        def set_goal(goal_q, opening_goal):
            nonlocal goal, plan
            control.validate_joints(goal_q, target_limits)
            goal_xyz = kinematics.forward_kinematics(np.array(goal_q))[:3, 3]
            goal = (tuple(map(float, goal_q)), None if opening is None else float(opening_goal), tuple(map(float, goal_xyz)))
            plan = None  # A new goal makes any earlier plan stale.
            clear_trail()
            ghost_root.visible = True
            ghost_model.update_cfg(np.array(viewer_configuration(names, goal[0])))
            gizmo.position = goal[2]
            for slider, name, value in zip(sliders, control.JOINTS, goal[0]):
                if abs(float(slider.value) - value) >= SLIDER_STEP_DEG / 2:
                    slider.value = slider_value(value, bounds[name])
            goal_text.content = f"**{copy('目标', 'Goal')}**: " + joints_line(goal[0], goal[1])
            refresh_buttons()

        def drop_goal():
            nonlocal goal, plan
            goal = plan = None
            clear_trail()
            ghost_root.visible = False
            goal_text.content = ""
            refresh_buttons()

        def make_plan():
            goal_q, opening_goal, goal_xyz = goal
            gripper_change = 0.0 if opening is None else float(opening_goal) - float(opening)
            duration = motion_duration(current, goal_q, gripper_change, ANIMATION_MIN_SECONDS)
            clear_trail()
            if lag(current, goal_q) > SLIDER_STEP_DEG / 2:  # A gripper-only goal has no path to draw.
                path = trail_points(kinematics, current, goal_q)
                trail.append(server.scene.add_spline_catmull_rom("/trail/path", path, color=TRAIL_COLOR, thickness=0.004))
                trail.append(server.scene.add_point_cloud("/trail/waypoints", path[::6], np.tile(TRAIL_COLOR, (len(path[::6]), 1)),
                                                         point_size=0.008, point_shape="circle"))
            return Plan(tuple(map(float, current)), goal_q, None if opening is None else float(opening),
                        None if opening is None else float(opening_goal), goal_xyz, duration)

        def abort_and_hold(reason):
            nonlocal phase, last_command, plan
            if arm and armed:
                hold_here(current, opening)
            phase = "idle"
            last_command = None
            plan = None
            clear_trail()
            gizmo.visible = True
            refresh_buttons()
            status.content = f"{copy('已停止并保持', 'Stopped and holding')}: {reason}"
            print(f"STOP-HOLD: {reason}", file=sys.stderr)

        def emergency_stop():
            """Cut torque now. The arm drops; the operator restarts the program to continue."""
            nonlocal phase, last_command, armed, estopped, plan
            phase = "idle"
            last_command = None
            plan = None
            estopped = True
            if arm:
                arm.bus.disable_torque()
                armed = False
            clear_trail()
            gizmo.visible = True
            refresh_buttons()
            status.content = copy("紧急停止：力矩已切断。检查手臂，然后重开程序。", "EMERGENCY STOP: torque cut. Check the arm, then restart the program.")
            print("EMERGENCY STOP: torque cut on all motors.", file=sys.stderr)

        print(f"Open http://{LOOPBACK}:{args.web_port}; mode={mode_label()}")
        while True:
            tick = time.monotonic()
            if stop_requested.is_set():
                return
            if owner_lost:
                owner_lost = False
                if phase == "executing":
                    abort_and_hold(copy("浏览器断开", "browser disconnected"))
                print("Browser disconnected; the program keeps holding. Ctrl+C releases torque and exits.", file=sys.stderr)
            if arm:
                current, opening = read_pose()
            actual_model.update_cfg(np.array(viewer_configuration(names, current)))
            xyz = kinematics.forward_kinematics(current)[:3, 3].copy()
            coordinates = ", ".join(f"{axis} {value * 1000:.1f}" for axis, value in zip("XYZ", xyz))
            readout.content = f"**{copy('当前', 'Current')} / mm**: {coordinates}\n\n" + joints_line(current, opening)

            batch = []
            for _ in range(requests.qsize()):
                batch.append(requests.get_nowait())
            with lock:
                for pending in (latest_joints, latest_drag):
                    if pending is not None:
                        batch.append(pending)
                latest_drag = latest_joints = None
            batch.sort(key=lambda r: r.created)
            for request in batch:
                if request.kind == "connected":
                    if owner is None:
                        owner = request.client_id
                    else:
                        print("A second browser connected; only the first one is in control.", file=sys.stderr)
                    continue
                if request.kind == "estop":
                    emergency_stop()
                    continue
                if request.client_id is not None and request.client_id != owner:
                    continue
                try:
                    if request.kind == "quit-refused":
                        status.content = copy("力矩还开着：先托住手臂，点“释放力矩”，再关闭程序。", "Torque is still on: support the arm, press Release torque, then close the program.")
                        continue
                    if request.kind == "stop":
                        if phase == "executing":
                            abort_and_hold(copy("操作者按下停止", "operator pressed Stop"))
                        continue
                    if phase == "executing":
                        status.content = copy("执行中：先按“停止并保持”。", "Executing: press Stop and hold first.")
                        continue
                    if request.kind in ("drag", "joints", "reset"):
                        if phase == "planning":
                            phase = "idle"
                        if request.kind == "drag":
                            seed = ik_seed(current, goal[0] if goal is not None else None, target_limits)
                            try:
                                goal_q, _, error = control.solve_position(kinematics, seed, np.asarray(request.payload, dtype=float), target_limits)
                            except ValueError:
                                # Out of reach or outside the joint range: keep the last reachable goal and snap back.
                                gizmo.position = goal[2] if goal is not None else tuple(xyz)
                                status.content = copy("够不到或超出关节范围，操纵柄退回上一个可达位置。", "Out of reach or outside the joint range; the handle snapped back to the last reachable position.")
                                continue
                            set_goal(goal_q, gripper_slider.value if gripper_slider else None)
                            status.content = copy(f"目标已设定，模型误差 {error:.3f} mm。点“计划轨迹”。", f"Goal set, model residual {error:.3f} mm. Press Plan trajectory.")
                        elif request.kind == "joints":
                            goal_q = np.asarray(request.payload[:5], dtype=float)
                            opening_goal = float(request.payload[5]) if len(request.payload) > 5 else None
                            if goal is not None and lag(goal_q, goal[0]) < SLIDER_STEP_DEG / 2 and (opening_goal is None or abs(opening_goal - goal[1]) < 0.5):
                                continue  # Echo of a slider we just synchronised.
                            set_goal(goal_q, opening_goal)
                            status.content = copy("目标来自滑杆。点“计划轨迹”。", "Goal set from the sliders. Press Plan trajectory.")
                        else:
                            drop_goal()
                            gizmo.position = tuple(xyz)
                            for slider, name, value in zip(sliders, control.JOINTS, current):
                                slider.value = slider_value(value, bounds[name])
                            if gripper_slider:
                                gripper_slider.value = float(round(opening))
                            status.content = copy("目标已回到当前姿态。", "Goal reset to the current pose.")
                    elif request.kind == "plan":
                        if estopped:
                            raise ValueError("Emergency stop is latched; restart the program")
                        if goal is None:
                            raise ValueError("No goal; drag the target or move a slider first")
                        plan = make_plan()
                        phase, step = "planning", 0
                        refresh_buttons()
                        status.content = copy(f"已计划，橙色模型正在走这条路；真正执行约需 {plan.duration:.1f} s。满意就点“执行运动”。", f"Planned; the orange model is running the path. Execution will take about {plan.duration:.1f} s. Press Execute if it looks right.")
                    elif request.kind == "execute":
                        check_request(request, owner, time.monotonic(), server.get_clients())
                        if estopped:
                            raise ValueError("Emergency stop is latched; restart the program")
                        if plan is None:
                            raise ValueError("No plan; press Plan trajectory first")
                        if arm and not armed:
                            raise ValueError("Torque is off; enable and hold first")
                        if lag(current, plan.start) > REPLAN_TOLERANCE_DEG:
                            plan = None
                            refresh_buttons()
                            raise ValueError("The arm moved since planning; plan again")
                        phase, step = "executing", 0
                        last_command = tuple(plan.start)
                        gizmo.visible = False
                        refresh_buttons()
                        status.content = copy("执行中……", "Executing…")
                    elif request.kind == "arm":
                        check_request(request, owner, time.monotonic(), server.get_clients())
                        if estopped:
                            raise ValueError("Emergency stop is latched; restart the program")
                        if not arming_requested(str(request.payload)):
                            raise ValueError(f"Type {ARM_WORD} in the text box first")
                        from lerobot.motors.feetech import OperatingMode
                        control.configure_and_hold(arm, OperatingMode.POSITION.value)
                        armed = True
                        arm_word.value = ""
                        refresh_buttons()
                        status.content = copy("力矩已启用，手臂保持当前姿态。可以松手，但断电开关要在手边。", "Torque on; the arm holds this pose. You may let go, but keep the power cutoff within reach.")
                    elif request.kind == "release":
                        arm.bus.disable_torque()
                        armed = False
                        refresh_buttons()
                        status.content = copy("力矩已释放，手臂现在可以用手搬动。", "Torque released; the arm can be moved by hand.")
                except (ValueError, RuntimeError) as exc:
                    if request.kind == "arm" and not armed:
                        status.content = f"{copy('未启用', 'Not enabled')}: {exc}"
                    elif request.kind == "joints":
                        drop_goal()
                        status.content = copy(f"目标不可用：{exc}", f"Goal unavailable: {exc}")
                    else:
                        status.content = f"{copy('未执行', 'Not executed')}: {exc}"

            if phase in ("planning", "executing") and plan is not None:
                # One fixed step per tick: a slow tick slows the motion, it never enlarges a step.
                seconds = preview_duration(plan.start, plan.goal) if phase == "planning" else plan.duration
                steps = max(1, math.ceil(seconds / UPDATE_SECONDS))
                step = min(step + 1, steps)
                fraction = eased(step / steps)
                waypoint = interpolation(plan.start, plan.goal, fraction)
                control.validate_joints(waypoint, reading_limits)  # The start may sit on a stop outside the model.
                if phase == "planning":
                    ghost_model.update_cfg(np.array(viewer_configuration(names, waypoint)))
                    if step >= steps:
                        phase = "idle"
                        refresh_buttons()
                        status.content = copy("计划演示完毕。要真的过去，点“执行运动”；改目标就要重新计划。", "Plan shown. Press Execute to move; changing the goal requires planning again.")
                elif arm:
                    if lag(current, last_command) > TRACKING_ABORT_DEG:
                        abort_and_hold(copy(f"实物落后指令 {lag(current, last_command):.1f}°", f"arm lags its command by {lag(current, last_command):.1f}°"))
                        continue
                    action = {f"{name}.pos": float(v) for name, v in zip(control.JOINTS, waypoint)}
                    action["gripper.pos"] = float(plan.gripper_start + (plan.gripper_goal - plan.gripper_start) * fraction)
                    sent = arm.send_action(action)
                    if any(abs(sent[key] - value) > control.SENT_TARGET_TOLERANCE_DEG for key, value in action.items()):
                        abort_and_hold(copy("LeRobot 截短了关节目标", "LeRobot clipped the joint target"))
                        continue
                    last_command = tuple(waypoint)
                    if step >= steps:
                        if lag(current, plan.goal) <= SETTLE_TOLERANCE_DEG:
                            phase, last_command, plan = "idle", None, None
                            gizmo.visible = True
                            clear_trail()
                            refresh_buttons()
                            status.content = copy("已到达目标并保持。", "Goal reached and holding.")
                        elif step > steps + SETTLE_TIMEOUT_SECONDS / UPDATE_SECONDS:
                            abort_and_hold(copy("到位超时", "did not settle in time"))
                        else:
                            step += 1  # Keep counting while the arm settles at the final command.
                else:
                    current = np.array(waypoint)
                    if step >= steps:
                        phase, plan = "idle", None
                        gizmo.visible = True
                        clear_trail()
                        refresh_buttons()
                        status.content = copy("模型已到达目标。", "The model reached the goal.")
            time.sleep(max(0.0, UPDATE_SECONDS - (time.monotonic() - tick)))
    finally:
        try:
            if arm and arm.bus.is_connected:
                if armed:
                    print("Releasing torque; support the arm.", file=sys.stderr)
                    arm.bus.disable_torque()
                arm.bus.disconnect(disable_torque=False)
        finally:
            if server:
                server.stop()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--web-port", type=int, required=True, help="Explicitly allocated local web port; no automatic choice")
    parser.add_argument("--locale", choices=("cn", "en"), default="cn", help="Interface language: cn (Chinese, default) or en (English)")
    parser.add_argument("--hardware", action="store_true", help="Connect the Follower bus: live readings; motion only after arming with ENABLE")
    parser.add_argument("--port")
    parser.add_argument("--robot-id")
    parser.add_argument("--calibration-dir")
    parser.add_argument("--confirm-model-match", action="store_true")
    args = parser.parse_args(argv)
    if args.hardware and not all((args.port, args.robot_id, args.calibration_dir, args.confirm_model_match)):
        parser.error("Hardware mode requires port, robot-id, calibration-dir, and confirm-model-match")
    if not args.hardware and any((args.port, args.robot_id, args.calibration_dir, args.confirm_model_match)):
        parser.error("Hardware arguments require explicit --hardware")
    try:
        run(args)
        return 0
    except KeyboardInterrupt:
        print("Stopped; this is not a physical power cutoff.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"STOP: {exc}. Hardware: cut motor power before investigating.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
