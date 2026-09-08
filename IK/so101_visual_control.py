#!/usr/bin/env python3
"""SO-101 interactive control with Viser 1.1.0 and LeRobot 0.6.1.

Goal, plan, execute, the way a planner front end works: dragging the handle or moving
a joint slider sets the orange goal; Plan trajectory computes the path and runs the
orange model along it; Execute follows it. Model mode (default) moves the model only.
--hardware connects the Follower bus: readings are live from the start, motion is sent
only after the operator arms the arm by typing ENABLE, every command is a small joint
step at a limited rate, the arm is watched for lag and load, and every stop, the
emergency one included, holds position with torque on; only Release torque switches
the motors off. There is no collision detection: the operator watches the
arm and keeps the DC cutoff within reach. Keep so101_cartesian_demo.py in the same directory.
"""
from __future__ import annotations

import argparse
import logging
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
# Contact stop: a joint that falls behind its command while its motor load climbs has met
# something. Defaults are conservative starting points; tune them on the real arm with
# --contact-error-deg and --contact-load-pct. The gripper is excluded: grasping stalls by design.
CONTACT_ERROR_DEG = 4.0
CONTACT_LOAD_PCT = 60.0
LOAD_READ_RETRIES = 2
SETTLE_TOLERANCE_DEG = 0.8
SETTLE_TIMEOUT_SECONDS = 2.0
REPLAN_TOLERANCE_DEG = 1.0  # The arm moved since planning: plan again. Holding sag stays well under this.
READ_RETRIES = 3  # Bus reads retry before a failure counts; one bad packet must not drop the arm.
COMM_FAILURE_LIMIT = 3  # Consecutive failed cycles during execution before stopping and holding.
MAX_REQUEST_AGE_SECONDS = 0.5  # Reject queued execution requests rather than replay them.
SLIDER_STEP_DEG = 0.5
ARM_WORD = "ENABLE"
TRAIL_SAMPLES = 48
TRAIL_COLOR = (255, 156, 31)
PREVIEW_GRIPPER_RAD = 0.0  # Geometry reference only: not mapped from LeRobot percent.
CURRENT_COLOR = (0.18, 0.48, 0.72, 1.0)
TARGET_COLOR = (1.0, 0.61, 0.12, 0.35)
JOINT_LABELS = ("F1 shoulder_pan", "F2 shoulder_lift", "F3 elbow_flex", "F4 wrist_flex", "F5 wrist_roll")
# Every run writes one debug log next to the program; the folder is git-ignored and pruned.
LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_KEEP = 20
HEARTBEAT_SECONDS = 1.0  # Idle hardware telemetry rate in the log; execution logs every cycle.
LOG = logging.getLogger("so101_visual_control")


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


def load_percent(raw_signed):
    """Feetech Present_Load after LeRobot's sign decoding: tenths of a percent of full torque, signed."""
    return abs(float(raw_signed)) / 10.0


def blocked_joints(current, commanded, loads_pct, error_deg, load_pct):
    """Arm joints that lag their command by more than error_deg while loaded above load_pct."""
    return [(name, abs(float(c) - float(g)), float(load))
            for name, c, g, load in zip(control.JOINTS, current, commanded, loads_pct)
            if abs(float(c) - float(g)) > error_deg and float(load) > load_pct]


def settle_state(residual_deg, settle_ticks):
    """After the last command: done when close enough, otherwise wait, then accept with a note."""
    if residual_deg <= SETTLE_TOLERANCE_DEG:
        return "done"
    return "timeout" if settle_ticks * UPDATE_SECONDS >= SETTLE_TIMEOUT_SECONDS else "wait"


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


def start_log(args, directory=LOG_DIR, keep=LOG_KEEP):
    """Open this run's debug log, prune old ones, record the configuration. Returns the log path."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"so101_visual_control_{time.strftime('%Y%m%d_%H%M%S')}.log"
    for handler in list(LOG.handlers):
        LOG.removeHandler(handler)
        handler.close()
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s.%(msecs)03d %(levelname)s %(message)s", "%H:%M:%S"))
    LOG.addHandler(handler)
    LOG.setLevel(logging.DEBUG)
    LOG.propagate = False
    for old in sorted(directory.glob("so101_visual_control_*.log"))[:-keep]:
        old.unlink()
    LOG.info("start %s", " ".join(sys.argv))
    LOG.info("args %s", vars(args))
    LOG.info("python %s lerobot %s viser %s", sys.version.split()[0], version("lerobot"), version("viser"))
    return path


def help_markdown(zh, args):
    e, l = f"{args.contact_error_deg:g}", f"{args.contact_load_pct:g}"
    if zh:
        return f"""#### 颜色与手柄
- **蓝色手臂**：当前姿态。实机模式下来自电机回读；每个关节显示度数，括号里是编码器格数，就是校准表里那种读数。
- **橙色手臂**：目标姿态，尚未执行。橙色线和圆点是计划好的夹爪路径。
- **操纵柄**：三根箭头沿底座 X、Y、Z 单轴拖，三个平面手柄在两轴平面里拖。拖到够不到的地方会弹回上一个可达位置。
- **滑杆**：直接给五个关节和夹爪设目标；和操纵柄是同一个目标的两种给法。

#### 三步走
1. **目标**：拖操纵柄或拨滑杆，橙色手臂出现。
2. **计划轨迹**：算出从当前到目标的路径并画出来，橙色模型走一遍演示（每秒 {PREVIEW_RATE_DEG_S:g}°，只是演示）。改了目标就要重新计划。
3. **执行运动**：沿计划的轨迹过去。模型模式动蓝色模型；实机模式动真机，速度上限每秒 {MOTION_RATE_DEG_S:g}°，每个周期只发一小步。

#### 力矩（实机模式）
- **启用力矩并保持当前姿态**：先输入 {ARM_WORD}。程序核对读数在校准范围内，把六个电机的目标设成当前位置，再开力矩。可以在休息姿态直接启用。
- **释放力矩**：六个电机同时停止出力，手臂会失去支撑，先托住再点。这是唯一让电机松开的按钮。

#### 四种停
- **停止并保持**：执行中随时可按，停在原地，电机继续出力。
- **自动停止**：执行中某关节落后指令超过 {e}° 且负载超过 {l}%（判定为碰到东西）、落后超过 {TRACKING_ABORT_DEG:g}°、LeRobot 截短了目标、浏览器断开，程序都自动停止并保持，状态行写明原因。
- **紧急停止**：任何时候都能按，比如执行中看到手臂快撞到东西。立刻停在原地并保持力矩，绝不卸力；之后界面锁住，检查完点“解除紧急停止”继续，或托住手臂后点“释放力矩”。
- **终端 Ctrl+C**：程序退出，退出前释放力矩，手臂会掉。只作最后手段。

#### 边界
- 程序不识别障碍物，也不规划避障；碰撞检测是碰上之后才停，不是提前避开。
- “负载”一行实时显示六个电机的负载百分比，用它来调 `--contact-error-deg` 和 `--contact-load-pct`。
- 力矩开着时不能关闭程序，先释放。软件停止不是物理断电，直流电源开关必须在手边。"""
    return f"""#### Colours and handles
- **Blue arm**: current pose. In hardware mode it comes from the motors; each joint shows degrees with the encoder count in brackets, the same number as in the calibration table.
- **Orange arm**: goal pose, not yet executed. The orange line and dots are the planned gripper path.
- **Handle**: three arrows drag along base X, Y or Z; three plane handles drag in a two-axis plane. Dragged out of reach it snaps back to the last reachable position.
- **Sliders**: set the five joints and the gripper directly; the handle and the sliders are two ways of giving the same goal.

#### Three steps
1. **Goal**: drag the handle or move a slider; the orange arm appears.
2. **Plan trajectory**: computes and draws the path from the current pose to the goal and runs the orange model along it ({PREVIEW_RATE_DEG_S:g}° per second, a demonstration only). Changing the goal requires planning again.
3. **Execute**: follows the planned trajectory. Model mode moves the blue model; hardware mode moves the arm at no more than {MOTION_RATE_DEG_S:g}° per second, one small step per cycle.

#### Torque (hardware mode)
- **Enable torque and hold this pose**: type {ARM_WORD} first. The program checks the readings are inside the calibrated range, sets all six motor targets to the present position, then switches torque on. It may be done in the rest pose.
- **Release torque**: all six motors stop driving and the arm loses its support, so hold it before pressing. This is the only button that lets the motors go.

#### Four kinds of stop
- **Stop and hold**: available at any moment during execution; the arm stops in place with the motors still driving.
- **Automatic stop**: during execution a joint more than {e}° behind its command at over {l}% load (taken as contact), a lag over {TRACKING_ABORT_DEG:g}°, a target clipped by LeRobot, or a browser disconnect all stop and hold, and the status line states why.
- **EMERGENCY STOP**: available at any moment, for example when you see the arm about to hit something during execution. It stops in place at once with torque kept on and never releases; the interface then locks until you press Clear emergency stop after checking, or Release torque with the arm supported.
- **Ctrl+C in the terminal**: the program exits and releases torque first, so the arm drops. Last resort only.

#### Boundaries
- The program does not recognise obstacles or plan around them; contact detection stops after contact, it does not avoid it.
- The Load line shows the six motor loads live; use it to tune `--contact-error-deg` and `--contact-load-pct`.
- The program cannot be closed while torque is on; release first. A software stop is not a physical power cutoff; keep the DC switch within reach."""


def run(args):
    import viser
    from viser.extras import ViserUrdf

    if version("viser") != VISER_VERSION:
        raise RuntimeError(f"Use viser[urdf]=={VISER_VERSION}")
    log_path = start_log(args)
    print(f"Log: {log_path}")

    def note(message):
        """Operator-facing terminal line, also kept in the log."""
        print(message, file=sys.stderr)
        LOG.warning(message)

    zh = args.locale == "cn"
    copy = lambda chinese, english: chinese if zh else english
    kinematics, model_limits = control.load_kinematics(args.model_dir)
    LOG.info("model limits deg %s", {k: (round(a, 2), round(b, 2)) for k, (a, b) in model_limits.items()})
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
    settle_ticks = 0
    comm_failures = 0
    read_failures = 0
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
        # Same normalisation as get_observation, but retried: a single bad packet must not end the program.
        positions = arm.bus.sync_read("Present_Position", num_retry=READ_RETRIES)
        q = np.array([positions[name] for name in control.JOINTS], dtype=float)
        control.validate_joints(q, reading_limits)
        opening = float(positions["gripper"])
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
            LOG.info("calibration %s", {name: (c.range_min, c.range_max, c.homing_offset) for name, c in arm.calibration.items()})
            LOG.info("reading limits deg %s", {k: (round(a, 2), round(b, 2)) for k, (a, b) in reading_limits.items()})
            LOG.info("target limits deg %s", {k: (round(a, 2), round(b, 2)) for k, (a, b) in target_limits.items()})
            current, opening = read_pose()
            LOG.info("initial pose deg %s gripper %.1f%%", np.round(current, 2).tolist(), opening)
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
                return copy("紧急停止 · 停在原地并保持力矩，解除后才能继续", "Emergency stop · holding in place with torque on; clear it to continue")
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
        load_text = server.gui.add_markdown("")
        goal_text = server.gui.add_markdown("")
        status = server.gui.add_markdown(copy("拖动目标或拨动滑杆，先出现橙色目标。", "Drag the target or move a slider; the orange goal appears first."))

        def say(text):
            """Status line for the operator, also kept in the log."""
            status.content = text
            LOG.info("status: %s", text)

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
                                             hint=copy("任何时候可按：立刻停在原地并保持力矩，绝不卸力；之后界面锁住，检查完再解除", "Press at any moment: stops in place with torque kept on, never releases; the interface then locks until you clear it"))
        clear_estop_button = server.gui.add_button(copy("解除紧急停止", "Clear emergency stop"), disabled=True,
                                                   hint=copy("检查过手臂和周围之后再点；力矩保持不变", "Press after checking the arm and its surroundings; torque stays as it is"))
        help_button = server.gui.add_button(copy("说明", "Help"), hint=copy("展开或收起完整说明", "Show or hide the full explanation"))
        quit_button = server.gui.add_button(copy("关闭程序", "Close program"))
        server.gui.add_markdown(copy("完整说明在“说明”按钮里。软件停止不是物理断电，直流电源开关要在手边。",
                                     "The full explanation is behind the Help button. A software stop is not a physical power cutoff; keep the DC switch within reach."))
        help_text = server.gui.add_markdown(help_markdown(zh, args), visible=False)

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
        clear_estop_button.on_click(lambda event: enqueue("clear-estop", event.client_id))
        help_button.on_click(lambda event: enqueue("help", event.client_id))
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
            clear_estop_button.disabled = not estopped
            if arm:
                arm_button.disabled = estopped or armed
                release_button.disabled = not armed  # Always a way out, latched or not: support the arm first.
            mode_text.content = f"### {mode_label()}"

        def set_goal(goal_q, opening_goal):
            nonlocal goal, plan
            control.validate_joints(goal_q, target_limits)
            goal_xyz = kinematics.forward_kinematics(np.array(goal_q))[:3, 3]
            goal = (tuple(map(float, goal_q)), None if opening is None else float(opening_goal), tuple(map(float, goal_xyz)))
            LOG.info("goal joints=%s gripper=%s xyz_mm=%s", np.round(goal[0], 2).tolist(), goal[1], np.round(np.array(goal[2]) * 1000, 1).tolist())
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
            LOG.info("plan start=%s goal=%s gripper %s->%s duration=%.2fs", np.round(current, 2).tolist(), np.round(goal_q, 2).tolist(),
                     None if opening is None else round(float(opening), 1), None if opening is None else round(float(opening_goal), 1), duration)
            return Plan(tuple(map(float, current)), goal_q, None if opening is None else float(opening),
                        None if opening is None else float(opening_goal), goal_xyz, duration)

        def abort_and_hold(reason):
            nonlocal phase, last_command, plan
            if arm and armed:
                try:
                    hold_here(current, opening)
                except (ConnectionError, RuntimeError, OSError) as exc:
                    note(f"hold command failed ({exc}); the servos keep their last goal")
            phase = "idle"
            last_command = None
            plan = None
            clear_trail()
            gizmo.visible = True
            refresh_buttons()
            say(f"{copy('已停止并保持', 'Stopped and holding')}: {reason}")
            note(f"STOP-HOLD: {reason}")

        def emergency_stop():
            """Freeze in place with torque kept on, then latch. Releasing torque here would drop the arm."""
            nonlocal phase, last_command, estopped, plan
            if arm and armed:
                try:
                    hold_here(current, opening)
                except (ConnectionError, RuntimeError, OSError) as exc:
                    note(f"hold command failed ({exc}); the servos keep their last goal")
            phase = "idle"
            last_command = None
            plan = None
            estopped = True
            clear_trail()
            gizmo.visible = True
            refresh_buttons()
            say(copy("紧急停止：已停在原地并保持力矩。检查手臂和周围，然后点“解除紧急停止”继续，或托住手臂后“释放力矩”。",
                                  "EMERGENCY STOP: holding in place with torque on. Check the arm and its surroundings, then press Clear emergency stop to continue, or Release torque with the arm supported."))
            note("EMERGENCY STOP: holding in place; torque unchanged.")

        def clear_emergency_stop():
            nonlocal estopped
            estopped = False
            refresh_buttons()
            say(copy("紧急停止已解除。目标和计划需要重新给。", "Emergency stop cleared. Set the goal and plan again."))

        print(f"Open http://{LOOPBACK}:{args.web_port}; mode={mode_label()}")
        LOG.info("serving http://%s:%d mode=%s", LOOPBACK, args.web_port, mode_label())
        last_heartbeat = 0.0
        while True:
            tick = time.monotonic()
            if stop_requested.is_set():
                return
            if owner_lost:
                owner_lost = False
                owner = None  # The next page to connect, a refresh included, takes over.
                if phase == "executing":
                    abort_and_hold(copy("浏览器断开", "browser disconnected"))
                note("Controlling browser disconnected; the next page to connect takes control. Ctrl+C releases torque and exits.")
            if arm:
                try:
                    current, opening = read_pose()
                    raw_loads = arm.bus.sync_read("Present_Load", normalize=False, num_retry=LOAD_READ_RETRIES)
                    read_failures = 0
                except ValueError as exc:
                    # A reading outside the calibrated range. Once running this is never fatal: exiting
                    # would release torque and drop the arm. Stop, hold, report, keep the last good pose.
                    if phase == "executing":
                        abort_and_hold(copy("读数超出校准范围", "reading outside the calibrated range"))
                    say(copy(f"读数超出校准范围：{exc}。反复出现就检查标定文件是否属于这只臂。", f"Reading outside the calibrated range: {exc}. If it persists, check that the calibration file belongs to this arm."))
                    time.sleep(UPDATE_SECONDS)
                    continue
                except (ConnectionError, RuntimeError, OSError) as exc:
                    read_failures += 1
                    note(f"bus read failed ({read_failures}): {exc}")
                    if phase == "executing" and read_failures >= COMM_FAILURE_LIMIT:
                        abort_and_hold(copy("总线通信连续失败", "bus communication failed repeatedly"))
                    say(copy(f"总线读取失败 {read_failures} 次，正在重试；电机仍保持上一个目标。", f"Bus read failed {read_failures} times, retrying; the motors still hold their last target."))
                    time.sleep(UPDATE_SECONDS)
                    continue
                loads = [load_percent(raw_loads[name]) for name in control.JOINTS]
                if phase != "executing" and tick - last_heartbeat >= HEARTBEAT_SECONDS:
                    last_heartbeat = tick
                    LOG.debug("pose=%s gripper=%.1f loads=%s phase=%s armed=%s", np.round(current, 2).tolist(), opening, [round(v) for v in loads], phase, armed)
                load_text.content = f"**{copy('负载', 'Load')} / %**: " + " · ".join(f"F{i + 1} {v:.0f}" for i, v in enumerate(loads)) + f" · {copy('夹爪', 'gripper')} {load_percent(raw_loads['gripper']):.0f}"
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
            batch.sort(key=lambda r: (r.kind != "estop", r.created))  # An emergency stop goes first.
            for request in batch:
                LOG.debug("request %s client=%s payload=%s phase=%s armed=%s", request.kind, request.client_id,
                          np.round(request.payload, 3).tolist() if isinstance(request.payload, tuple) and request.payload else request.payload, phase, armed)
                if request.kind == "connected":
                    if owner is None:
                        owner = request.client_id
                        LOG.info("owner -> client %s", owner)
                        say(copy("这个页面已接管控制。", "This page is now in control."))
                    else:
                        note("A second browser connected; only the first one is in control.")
                    continue
                if request.kind == "estop":
                    emergency_stop()
                    continue
                if request.client_id is not None and owner is None:
                    owner = request.client_id  # Input arrived before the connect notice was processed.
                if request.client_id is not None and request.client_id != owner:
                    say(copy("输入被忽略：另一个浏览器页面先连上了。关掉那个页面，或刷新本页接管。", "Input ignored: another browser page connected first. Close that page, or refresh this one to take over."))
                    continue
                if request.kind == "help":
                    help_text.visible = not help_text.visible
                    continue
                if request.kind == "clear-estop":
                    if estopped:
                        clear_emergency_stop()
                    continue
                try:
                    if request.kind == "quit-refused":
                        say(copy("力矩还开着：先托住手臂，点“释放力矩”，再关闭程序。", "Torque is still on: support the arm, press Release torque, then close the program."))
                        continue
                    if request.kind == "stop":
                        if phase == "executing":
                            abort_and_hold(copy("操作者按下停止", "operator pressed Stop"))
                        continue
                    if phase == "executing":
                        say(copy("执行中：先按“停止并保持”。", "Executing: press Stop and hold first."))
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
                                say(copy("够不到或超出关节范围，操纵柄退回上一个可达位置。", "Out of reach or outside the joint range; the handle snapped back to the last reachable position."))
                                continue
                            set_goal(goal_q, gripper_slider.value if gripper_slider else None)
                            say(copy(f"目标已设定，模型误差 {error:.3f} mm。点“计划轨迹”。", f"Goal set, model residual {error:.3f} mm. Press Plan trajectory."))
                        elif request.kind == "joints":
                            goal_q = np.asarray(request.payload[:5], dtype=float)
                            opening_goal = float(request.payload[5]) if len(request.payload) > 5 else None
                            if goal is not None and lag(goal_q, goal[0]) < SLIDER_STEP_DEG / 2 and (opening_goal is None or abs(opening_goal - goal[1]) < 0.5):
                                continue  # Echo of a slider we just synchronised.
                            set_goal(goal_q, opening_goal)
                            say(copy("目标来自滑杆。点“计划轨迹”。", "Goal set from the sliders. Press Plan trajectory."))
                        else:
                            drop_goal()
                            gizmo.position = tuple(xyz)
                            for slider, name, value in zip(sliders, control.JOINTS, current):
                                slider.value = slider_value(value, bounds[name])
                            if gripper_slider:
                                gripper_slider.value = float(round(opening))
                            say(copy("目标已回到当前姿态。", "Goal reset to the current pose."))
                    elif request.kind == "plan":
                        if estopped:
                            raise ValueError("Emergency stop is latched; clear it first")
                        if goal is None:
                            raise ValueError("No goal; drag the target or move a slider first")
                        plan = make_plan()
                        phase, step = "planning", 0
                        refresh_buttons()
                        say(copy(f"已计划，橙色模型正在走这条路；真正执行约需 {plan.duration:.1f} s。满意就点“执行运动”。", f"Planned; the orange model is running the path. Execution will take about {plan.duration:.1f} s. Press Execute if it looks right."))
                    elif request.kind == "execute":
                        if len(server.get_clients()) > 1:
                            raise ValueError(copy("有第二个浏览器页面连着，关掉它再执行", "A second browser page is connected; close it before executing"))
                        check_request(request, owner, time.monotonic(), server.get_clients())
                        if estopped:
                            raise ValueError("Emergency stop is latched; clear it first")
                        if plan is None:
                            raise ValueError("No plan; press Plan trajectory first")
                        if arm and not armed:
                            raise ValueError("Torque is off; enable and hold first")
                        if lag(current, plan.start) > REPLAN_TOLERANCE_DEG:
                            plan = None
                            refresh_buttons()
                            raise ValueError("The arm moved since planning; plan again")
                        phase, step, settle_ticks, comm_failures = "executing", 0, 0, 0
                        last_command = tuple(plan.start)
                        LOG.info("execute begins hardware=%s", bool(arm))
                        gizmo.visible = False
                        refresh_buttons()
                        say(copy("执行中……", "Executing…"))
                    elif request.kind == "arm":
                        if len(server.get_clients()) > 1:
                            raise ValueError(copy("有第二个浏览器页面连着，关掉它再启用", "A second browser page is connected; close it before enabling torque"))
                        check_request(request, owner, time.monotonic(), server.get_clients())
                        if estopped:
                            raise ValueError("Emergency stop is latched; clear it first")
                        if not arming_requested(str(request.payload)):
                            raise ValueError(f"Type {ARM_WORD} in the text box first")
                        from lerobot.motors.feetech import OperatingMode
                        LOG.info("arming: pose %s", np.round(current, 2).tolist())
                        control.configure_and_hold(arm, OperatingMode.POSITION.value)
                        armed = True
                        LOG.info("armed; torque on")
                        arm_word.value = ""
                        refresh_buttons()
                        say(copy("力矩已启用，手臂保持当前姿态。可以松手，但断电开关要在手边。", "Torque on; the arm holds this pose. You may let go, but keep the power cutoff within reach."))
                    elif request.kind == "release":
                        arm.bus.disable_torque()
                        armed = False
                        LOG.info("torque released by operator")
                        refresh_buttons()
                        say(copy("力矩已释放，手臂现在可以用手搬动。", "Torque released; the arm can be moved by hand."))
                except (ValueError, RuntimeError) as exc:
                    LOG.warning("request %s rejected: %s", request.kind, exc)
                    if request.kind == "arm" and not armed:
                        say(f"{copy('未启用', 'Not enabled')}: {exc}")
                    elif request.kind == "joints":
                        drop_goal()
                        say(copy(f"目标不可用：{exc}", f"Goal unavailable: {exc}"))
                    else:
                        say(f"{copy('未执行', 'Not executed')}: {exc}")

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
                        say(copy("计划演示完毕。要真的过去，点“执行运动”；改目标就要重新计划。", "Plan shown. Press Execute to move; changing the goal requires planning again."))
                elif arm:
                    blocked = blocked_joints(current, last_command, loads, args.contact_error_deg, args.contact_load_pct)
                    if blocked:
                        name, error, load = blocked[0]
                        joint = f"F{control.JOINTS.index(name) + 1}"
                        abort_and_hold(copy(f"{joint} 受阻：落后指令 {error:.1f}°，负载 {load:.0f}%，像是碰到了东西", f"{joint} blocked: {error:.1f}° behind its command at {load:.0f}% load; it has probably met something"))
                        continue
                    if lag(current, last_command) > TRACKING_ABORT_DEG:
                        abort_and_hold(copy(f"实物落后指令 {lag(current, last_command):.1f}°", f"arm lags its command by {lag(current, last_command):.1f}°"))
                        continue
                    action = {f"{name}.pos": float(v) for name, v in zip(control.JOINTS, waypoint)}
                    action["gripper.pos"] = float(plan.gripper_start + (plan.gripper_goal - plan.gripper_start) * fraction)
                    try:
                        sent = arm.send_action(action)
                        comm_failures = 0
                    except (ConnectionError, RuntimeError, OSError) as exc:
                        comm_failures += 1  # Skip this cycle; the previous command is still held by the servos.
                        note(f"bus error during execution ({comm_failures}/{COMM_FAILURE_LIMIT}): {exc}")
                        if comm_failures >= COMM_FAILURE_LIMIT:
                            abort_and_hold(copy("总线通信连续失败", "bus communication failed repeatedly"))
                        continue
                    if any(abs(sent[key] - value) > control.SENT_TARGET_TOLERANCE_DEG for key, value in action.items()):
                        abort_and_hold(copy("LeRobot 截短了关节目标", "LeRobot clipped the joint target"))
                        continue
                    last_command = tuple(waypoint)
                    LOG.debug("exec %d/%d f=%.3f cmd=%s cur=%s lag=%.2f loads=%s", step, steps, fraction, np.round(waypoint, 2).tolist(),
                              np.round(current, 2).tolist(), lag(current, waypoint), [round(v) for v in loads])
                    if step >= steps:
                        residual = lag(current, plan.goal)
                        state = settle_state(residual, settle_ticks)
                        settle_ticks += 1
                        if state != "wait":
                            phase, last_command, plan = "idle", None, None
                            gizmo.visible = True
                            clear_trail()
                            refresh_buttons()
                            say(copy("已到达目标并保持。", "Goal reached and holding.") if state == "done" else
                                copy(f"已发出最终目标并保持，剩余误差 {residual:.1f}°。", f"Final target sent and holding; residual error {residual:.1f}°."))
                else:
                    current = np.array(waypoint)
                    if step >= steps:
                        phase, plan = "idle", None
                        gizmo.visible = True
                        clear_trail()
                        refresh_buttons()
                        say(copy("模型已到达目标。", "The model reached the goal."))
            time.sleep(max(0.0, UPDATE_SECONDS - (time.monotonic() - tick)))
    finally:
        LOG.info("shutting down: armed=%s phase=%s estopped=%s", armed, phase, estopped)
        try:
            if arm and arm.bus.is_connected:
                if armed:
                    note("Releasing torque; support the arm.")
                    try:
                        arm.bus.disable_torque()
                    except Exception as exc:  # noqa: BLE001 - report, then still disconnect
                        note(f"Torque release failed ({exc}); the motors may still be holding. Cut DC power to release.")
                arm.bus.disconnect(disable_torque=False)
        finally:
            if server:
                server.stop()
            LOG.info("exit")


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
    parser.add_argument("--contact-error-deg", type=float, default=CONTACT_ERROR_DEG, help="Contact stop: a joint this far behind its command, while loaded, stops the motion")
    parser.add_argument("--contact-load-pct", type=float, default=CONTACT_LOAD_PCT, help="Contact stop: motor load in percent that counts as pushing against something")
    args = parser.parse_args(argv)
    if not (0 < args.contact_error_deg <= TRACKING_ABORT_DEG) or not (0 < args.contact_load_pct <= 100):
        parser.error(f"contact thresholds must be within (0, {TRACKING_ABORT_DEG}] deg and (0, 100] percent")
    if args.hardware and not all((args.port, args.robot_id, args.calibration_dir, args.confirm_model_match)):
        parser.error("Hardware mode requires port, robot-id, calibration-dir, and confirm-model-match")
    if not args.hardware and any((args.port, args.robot_id, args.calibration_dir, args.confirm_model_match)):
        parser.error("Hardware arguments require explicit --hardware")
    try:
        run(args)
        return 0
    except KeyboardInterrupt:
        LOG.info("keyboard interrupt")
        print("Stopped; this is not a physical power cutoff.", file=sys.stderr)
        return 130
    except Exception as exc:
        LOG.exception("fatal: %s", exc)
        print(f"STOP: {exc}. Hardware: cut motor power before investigating.", file=sys.stderr)
        if LOG.handlers:
            print(f"Details in {LOG.handlers[0].baseFilename}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
