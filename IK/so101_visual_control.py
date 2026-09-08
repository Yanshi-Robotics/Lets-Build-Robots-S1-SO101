#!/usr/bin/env python3
"""SO-101 interactive control with Viser 1.1.0 and LeRobot 0.6.1.

Model mode (default) plans and animates on the model only. --hardware connects the
Follower bus: joint readings are live from the start, and motion is sent only after
the operator arms the arm by typing ENABLE. Every command is a small interpolated
joint step at a limited rate, the arm is watched for lag, and Stop holds the current
position. There is no collision detection: the operator watches the arm and keeps
the DC cutoff within reach. Keep so101_cartesian_demo.py in the same directory.
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
MOTION_RATE_DEG_S = 10.0  # Joint speed for previews and for hardware execution.
GRIPPER_RATE_PCT_S = 20.0  # Gripper opening speed; its eased peak stays under LeRobot's 2 %-per-command clip.
ANIMATION_MIN_SECONDS = 2.0  # A preview plays long enough to be watched.
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
    requests: queue.Queue[Request] = queue.Queue(maxsize=64)
    stop_requested = threading.Event()
    owner = None
    owner_lost = False
    lock = threading.Lock()
    latest_drag = None
    latest_joints = None
    plan = None
    phase = "idle"  # idle | previewing | executing
    step = 0  # Motion advances one fixed step per tick, never by wall-clock time.
    last_command = None
    trail = []

    def enqueue(kind, client_id=None, payload=()):
        # Callbacks only enqueue immutable input. IK, bus I/O and motion state live on
        # the main thread; Viser callbacks run concurrently.
        nonlocal latest_drag, latest_joints
        if kind == "quit":
            if armed:
                try:
                    requests.put_nowait(Request("quit-refused", client_id, time.monotonic()))
                except queue.Full:
                    pass
                return
            stop_requested.set()
            return
        if client_id is not None and client_id != owner and kind != "connected":
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
        ghost_root = server.scene.add_frame("/preview", show_axes=False, visible=False)
        model_path = Path(args.model_dir) / control.URDF_NAME
        actual_model = ViserUrdf(server, model_path, root_node_name="/current", mesh_color_override=CURRENT_COLOR)
        ghost_model = ViserUrdf(server, model_path, root_node_name="/preview", mesh_color_override=TARGET_COLOR)
        names = actual_model.get_actuated_joint_names()
        xyz = kinematics.forward_kinematics(current)[:3, 3].copy()
        gizmo = server.scene.add_transform_controls(
            "/target", position=tuple(xyz), scale=0.13, disable_rotations=True, disable_sliders=True, line_width=4.0,
        )

        def mode_label():
            if not arm:
                return copy("模型模式 · 未连接硬件", "Model mode · no hardware")
            return copy("实机模式 · 力矩已启用，执行会动真机", "Hardware mode · torque on, execute moves the arm") if armed \
                else copy("实机模式 · 力矩未启用，只读", "Hardware mode · torque off, reading only")

        mode_text = server.gui.add_markdown(f"### {mode_label()}")
        server.gui.add_markdown(copy(
            "蓝色：当前姿态，实机模式下来自电机回读。橙色：目标姿态。拖动红、绿、蓝轴或拨动关节滑杆设定目标，橙色轨迹线是夹爪将走的路。旋转视角不改变底座坐标。",
            "Blue: current pose, read from the motors in hardware mode. Orange: target pose. Drag the red, green or blue axis or move the joint sliders to set a target; the orange line is the path the gripper will take. Camera rotation does not change the base frame.",
        ))
        readout = server.gui.add_markdown("")
        status = server.gui.add_markdown(copy("拖动目标或拨动滑杆，先出现橙色目标和轨迹。", "Drag the target or move a slider; the orange target and path appear first."))
        bounds = slider_bounds(target_limits)
        with server.gui.add_folder(copy("目标关节角 / °", "Target joints / °")):
            sliders = [server.gui.add_slider(label, bounds[name][0], bounds[name][1], SLIDER_STEP_DEG, slider_value(v, bounds[name]))
                       for label, name, v in zip(JOINT_LABELS, control.JOINTS, current)]
            gripper_slider = server.gui.add_slider(copy("夹爪开合 / %", "Gripper / %"), 0, 100, 1, float(opening)) if arm else None
        preview = server.gui.add_button(copy("预览运动（只动橙色模型）", "Preview motion (orange model only)"), disabled=True)
        execute = server.gui.add_button(copy("执行运动", "Execute motion"), disabled=True)
        stop = server.gui.add_button(copy("停止并保持", "Stop and hold"), disabled=True)
        reset = server.gui.add_button(copy("目标回到当前姿态", "Reset target to current pose"))
        if arm:
            with server.gui.add_folder(copy("力矩", "Torque")):
                arm_word = server.gui.add_text(copy(f"输入 {ARM_WORD} 再点启用", f"Type {ARM_WORD}, then enable"), "")
                arm_button = server.gui.add_button(copy("启用力矩并保持当前姿态", "Enable torque and hold this pose"))
                release_button = server.gui.add_button(copy("释放力矩", "Release torque"), disabled=True,
                                                       hint=copy("先托住手臂：释放后六个电机不再出力，手臂会下落", "Support the arm first: after release the six motors stop driving and the arm drops"))
        quit_button = server.gui.add_button(copy("关闭程序", "Close program"))
        server.gui.add_markdown(copy(
            "执行时每次只发送一小步关节目标，速度上限每秒 10°；实物落后指令超过 8° 即停止并保持。力矩开着时不能关闭程序，要先托住手臂、释放力矩；终端里按 Ctrl+C 会直接释放力矩，手臂会下落。软件不是物理断电。",
            "Execution sends one small joint step at a time, at most 10° per second; if the arm lags its command by more than 8° it stops and holds. The program cannot be closed while torque is on: support the arm and release torque first. Ctrl+C in the terminal releases torque at once and the arm drops. Software is not a physical power cutoff.",
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

        for slider in sliders:
            @slider.on_update
            def moved(event):
                enqueue("joints", event.client_id, tuple(float(s.value) for s in sliders))

        preview.on_click(lambda event: enqueue("preview", event.client_id))
        execute.on_click(lambda event: enqueue("execute", event.client_id))
        stop.on_click(lambda event: enqueue("stop", event.client_id))
        reset.on_click(lambda event: enqueue("reset", event.client_id))
        if arm:
            arm_button.on_click(lambda event: enqueue("arm", event.client_id, arm_word.value))
            release_button.on_click(lambda event: enqueue("release", event.client_id))
        quit_button.on_click(lambda event: enqueue("quit", event.client_id))

        def clear_trail():
            for handle in trail:
                handle.remove()
            trail.clear()

        def show_plan(new_plan):
            nonlocal plan
            plan = new_plan
            clear_trail()
            ghost_root.visible = True
            ghost_model.update_cfg(np.array(viewer_configuration(names, new_plan.goal)))
            gizmo.position = tuple(new_plan.xyz)
            path = trail_points(kinematics, new_plan.start, new_plan.goal)
            trail.append(server.scene.add_spline_catmull_rom("/trail/path", path, color=TRAIL_COLOR, thickness=0.004))
            trail.append(server.scene.add_point_cloud("/trail/waypoints", path[::6], np.tile(TRAIL_COLOR, (len(path[::6]), 1)),
                                                     point_size=0.008, point_shape="circle"))
            for slider, name, value in zip(sliders, control.JOINTS, new_plan.goal):
                if abs(float(slider.value) - value) >= SLIDER_STEP_DEG / 2:
                    slider.value = slider_value(value, bounds[name])
            preview.disabled = False
            execute.disabled = bool(arm) and not armed

        def drop_plan():
            nonlocal plan
            plan = None
            clear_trail()
            ghost_root.visible = False
            preview.disabled = True
            execute.disabled = True

        def make_plan(goal_q, opening_goal):
            control.validate_joints(goal_q, target_limits)
            goal_xyz = kinematics.forward_kinematics(np.array(goal_q))[:3, 3]
            gripper_change = 0.0 if opening is None else float(opening_goal) - float(opening)
            duration = motion_duration(current, goal_q, gripper_change, ANIMATION_MIN_SECONDS)
            return Plan(tuple(map(float, current)), tuple(map(float, goal_q)),
                        None if opening is None else float(opening), None if opening is None else float(opening_goal),
                        tuple(map(float, goal_xyz)), duration)

        def abort_and_hold(reason):
            nonlocal phase, last_command
            if arm and armed:
                hold_here(current, opening)
            phase = "idle"
            last_command = None
            stop.disabled = True
            drop_plan()
            status.content = f"{copy('已停止并保持', 'Stopped and holding')}: {reason}"
            print(f"STOP-HOLD: {reason}", file=sys.stderr)

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
            joints_text = " · ".join(f"F{i + 1} {value:.1f}°" for i, value in enumerate(current))
            readout.content = f"**{copy('当前', 'Current')} / mm**: {coordinates}\n\n{joints_text}" + (
                f"\n\n{copy('夹爪回读', 'Gripper readback')}: {opening:.1f}%" if opening is not None else "")

            batch = []
            for _ in range(requests.qsize()):
                batch.append(requests.get_nowait())
            with lock:
                for pending in (latest_joints, latest_drag):  # A drag after a slider move wins, and vice versa.
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
                if request.client_id is not None and request.client_id != owner:
                    continue
                try:
                    if request.kind == "quit-refused":
                        status.content = copy("力矩还开着：先托住手臂，点“释放力矩”，再关闭程序。", "Torque is still on: support the arm, press Release torque, then close the program.")
                        continue
                    if request.kind == "stop":
                        if phase == "executing":
                            abort_and_hold(copy("操作者按下停止", "operator pressed stop"))
                        continue
                    if phase == "executing":
                        status.content = copy("执行中：先按“停止并保持”，再改目标。", "Executing: press Stop and hold before changing the target.")
                        continue
                    if request.kind in ("drag", "joints", "reset"):
                        if phase == "previewing":
                            phase = "idle"
                        if request.kind == "drag":
                            if plan is not None and np.linalg.norm(np.asarray(request.payload) - np.asarray(plan.xyz)) < 1e-4:
                                continue  # Echo of a gizmo position we just synchronised.
                            goal_q, _, error = control.solve_position(kinematics, current, np.asarray(request.payload, dtype=float), target_limits)
                            show_plan(make_plan(goal_q, gripper_slider.value if gripper_slider else None))
                            status.content = copy(f"已规划：模型误差 {error:.3f} mm，用时约 {plan.duration:.1f} s。橙色是目标，不是实机反馈。",
                                                  f"Planned: model residual {error:.3f} mm, about {plan.duration:.1f} s. Orange is the target, not hardware feedback.")
                        elif request.kind == "joints":
                            goal_q = np.asarray(request.payload, dtype=float)
                            if plan is not None and lag(goal_q, plan.goal) < SLIDER_STEP_DEG / 2:
                                continue  # Echo of a slider we just synchronised.
                            show_plan(make_plan(goal_q, gripper_slider.value if gripper_slider else None))
                            status.content = copy(f"已规划：关节目标来自滑杆，用时约 {plan.duration:.1f} s。", f"Planned from the sliders, about {plan.duration:.1f} s.")
                        else:
                            drop_plan()
                            gizmo.position = tuple(xyz)
                            for slider, name, value in zip(sliders, control.JOINTS, current):
                                slider.value = slider_value(value, bounds[name])
                            if gripper_slider:
                                gripper_slider.value = float(round(opening))
                            status.content = copy("目标已回到当前姿态。", "Target reset to the current pose.")
                    elif request.kind == "preview":
                        if plan is None:
                            raise ValueError("No plan; set a target first")
                        if lag(current, plan.start) > REPLAN_TOLERANCE_DEG:
                            raise ValueError("The arm moved since planning; set the target again")
                        phase, step = "previewing", 0
                        status.content = copy("预览中：橙色模型沿轨迹走一遍，实物不动。", "Previewing: the orange model runs the path; the arm does not move.")
                    elif request.kind == "execute":
                        check_request(request, owner, time.monotonic(), server.get_clients())
                        if plan is None:
                            raise ValueError("No plan; set a target first")
                        if arm and not armed:
                            raise ValueError("Torque is off; enable and hold first")
                        if lag(current, plan.start) > REPLAN_TOLERANCE_DEG:
                            raise ValueError("The arm moved since planning; set the target again")
                        phase, step = "executing", 0
                        last_command = tuple(plan.start)
                        stop.disabled = False
                        gizmo.visible = False
                        status.content = copy("执行中……", "Executing…")
                    elif request.kind == "arm":
                        check_request(request, owner, time.monotonic(), server.get_clients())
                        if not arming_requested(str(request.payload)):
                            raise ValueError(f"Type {ARM_WORD} in the text box first")
                        control.validate_joints(current, target_limits)  # Off a stop, inside the model: motion can start from here.
                        from lerobot.motors.feetech import OperatingMode
                        control.configure_and_hold(arm, OperatingMode.POSITION.value)
                        armed = True
                        arm_word.value = ""
                        release_button.disabled = False
                        mode_text.content = f"### {mode_label()}"
                        if plan is not None:
                            execute.disabled = False
                        status.content = copy("力矩已启用，手臂保持当前姿态。可以松手，但断电开关要在手边。", "Torque on; the arm holds this pose. You may let go, but keep the power cutoff within reach.")
                    elif request.kind == "release":
                        if phase == "executing":
                            abort_and_hold(copy("释放力矩前先停止", "stopped before releasing torque"))
                        arm.bus.disable_torque()
                        armed = False
                        release_button.disabled = True
                        execute.disabled = True
                        mode_text.content = f"### {mode_label()}"
                        status.content = copy("力矩已释放，手臂现在可以用手搬动。", "Torque released; the arm can be moved by hand.")
                except (ValueError, RuntimeError) as exc:
                    if request.kind == "arm" and not armed:
                        status.content = f"{copy('未启用', 'Not enabled')}: {exc}"
                    elif request.kind in ("drag", "joints"):
                        drop_plan()
                        status.content = f"{copy('目标不可用', 'Target unavailable')}: {exc}"
                    else:
                        status.content = f"{copy('未执行', 'Not executed')}: {exc}"

            if phase == "previewing" and plan is not None:
                step = min(step + 1, motion_steps(plan.duration))
                fraction = eased(step / motion_steps(plan.duration))
                ghost_model.update_cfg(np.array(viewer_configuration(names, interpolation(plan.start, plan.goal, fraction))))
                if fraction >= 1.0:
                    phase = "idle"
                    status.content = copy("预览结束。要让实物或模型真的过去，点“执行运动”。", "Preview finished. Press Execute motion to move the model or the arm.")
            elif phase == "executing" and plan is not None:
                step = min(step + 1, motion_steps(plan.duration) + int(SETTLE_TIMEOUT_SECONDS / UPDATE_SECONDS))
                fraction = eased(min(step / motion_steps(plan.duration), 1.0))
                waypoint = interpolation(plan.start, plan.goal, fraction)
                control.validate_joints(waypoint, target_limits)
                if arm:
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
                    if fraction >= 1.0:
                        if lag(current, plan.goal) <= SETTLE_TOLERANCE_DEG:
                            phase, last_command = "idle", None
                            stop.disabled, gizmo.visible = True, True
                            drop_plan()
                            status.content = copy("已到达目标并保持。", "Target reached and holding.")
                        elif step >= motion_steps(plan.duration) + int(SETTLE_TIMEOUT_SECONDS / UPDATE_SECONDS):
                            abort_and_hold(copy("到位超时", "did not settle in time"))
                else:
                    current = np.array(waypoint)
                    if fraction >= 1.0:
                        phase = "idle"
                        stop.disabled, gizmo.visible = True, True
                        drop_plan()
                        status.content = copy("模型已到达目标。", "The model reached the target.")
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
