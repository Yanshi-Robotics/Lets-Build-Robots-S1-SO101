#!/usr/bin/env python3
"""SO-101 interactive position IK with Viser 1.1.0 and LeRobot 0.6.1.

Default mode is model-only. --readback opens the Follower bus read-only for an
operator to compare measured joint positions with the model. This program never
enables torque or sends motor targets. Execute animates only the model-only mode.
Animation draws the gripper path first, then moves the model along it, the way a
planner preview does; there is no collision detection or hardware control in this viewer.
Keep so101_cartesian_demo.py in the same directory.
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

import so101_cartesian_demo as control

VISER_VERSION = "1.1.0"
LOOPBACK = "127.0.0.1"  # Hardware control must never bind a LAN/public interface.
UPDATE_SECONDS = 0.05  # 20 Hz UI/readback loop; not a real-time safety guarantee.
JOINT_TARGET_RATE_DEG_S = 10.0  # Model animation speed, not measured motor speed.
ANIMATION_MIN_SECONDS = 2.0  # Even a small move plays long enough to be watched.
TRAIL_SAMPLES = 48  # Gripper path samples drawn before the model moves.
TRAIL_COLOR = (255, 156, 31)
# Model-only bounds: nothing physical stands behind them, so a reader can drag far
# enough to see the arm move. Readback mode keeps the hardware teaching bounds.
MODEL_STEP_MM = 20.0
MODEL_JOINT_ENVELOPE_DEG = 60.0
MODEL_XYZ_ENVELOPE_MM = 150.0
MAX_REQUEST_AGE_SECONDS = 0.5  # Reject queued execution requests, rather than replay them.
MAX_PREVIEW_SEED_CHANGE_DEG = 0.5  # A moved arm requires a newly inspected preview.
PREVIEW_GRIPPER_RAD = 0.0  # Geometry reference only: not mapped from LeRobot percent.
CURRENT_COLOR = (0.18, 0.48, 0.72, 1.0)
TARGET_COLOR = (1.0, 0.61, 0.12, 0.35)


@dataclass(frozen=True)
class Preview:
    seed: tuple[float, ...]
    joints: tuple[float, ...]
    xyz: tuple[float, ...]


@dataclass(frozen=True)
class Request:
    kind: str
    client_id: int | None
    created: float
    target: tuple[float, ...] = ()
    preview: Preview | None = None
    revision: int = 0


def check_preview(preview, current):
    if preview is None:
        raise ValueError("No confirmed preview; move the target and inspect it first")
    for values in (current, preview.seed, preview.joints):
        if len(values) != len(control.JOINTS) or not all(math.isfinite(float(v)) for v in values):
            raise ValueError("Invalid current or preview pose")
    if len(preview.xyz) != 3 or not all(math.isfinite(float(v)) for v in preview.xyz):
        raise ValueError("Invalid target position")
    if max(abs(a - b) for a, b in zip(preview.seed, current)) > MAX_PREVIEW_SEED_CHANGE_DEG:
        raise ValueError("Arm moved since preview; preview and confirm again")


def eased(fraction):
    """Smooth start and stop of the model animation; still 0 at 0 and 1 at 1."""
    if not math.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError("Invalid animation fraction")
    return 0.5 - 0.5 * math.cos(math.pi * fraction)


def trail_points(kinematics, start, goal, samples=TRAIL_SAMPLES):
    """Gripper positions along the joint interpolation the animation will follow."""
    if samples < 2:
        raise ValueError("A trail needs at least two samples")
    import numpy as np
    return np.array([kinematics.forward_kinematics(np.array(interpolation(start, goal, i / (samples - 1))))[:3, 3]
                     for i in range(samples)])


def interpolation(start, target, fraction):
    """Pure joint interpolation. Never extrapolate after a delayed UI event."""
    if not math.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError("Invalid interpolation fraction")
    if len(start) != len(control.JOINTS) or len(target) != len(control.JOINTS):
        raise ValueError("Expected five arm joints")
    if not all(math.isfinite(float(v)) for v in (*start, *target)):
        raise ValueError("Non-finite joint value")
    return [float(a + (b - a) * fraction) for a, b in zip(start, target)]


def check_request(request, owner, now, connected_ids):
    if request.client_id != owner or set(connected_ids) != {owner}:
        raise ValueError("Only the original connected browser may execute")
    if not 0 <= now - request.created <= MAX_REQUEST_AGE_SECONDS:
        raise ValueError("Execution request expired; preview and confirm again")


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
    import numpy as np
    import viser
    from viser.extras import ViserUrdf

    if version("viser") != VISER_VERSION:
        raise RuntimeError(f"Use viser[urdf]=={VISER_VERSION}")
    kinematics, model_limits = control.load_kinematics(args.model_dir)
    limits = model_limits
    arm = None
    server = None
    requests: queue.Queue[Request] = queue.Queue(maxsize=64)
    lost_owner = threading.Event()
    stop_requested = threading.Event()
    owner = None
    preview_lock = threading.Lock()
    displayed_preview = None
    latest_target = None
    target_revision = 0
    motion_active = False
    trail = []

    def clear_trail():
        for handle in trail:
            handle.remove()
        trail.clear()

    def enqueue(kind, client_id=None, target=()):
        # Callbacks only enqueue immutable input. All IK, bus I/O and motion state
        # live on this one main thread; Viser callbacks can run concurrently.
        nonlocal displayed_preview, latest_target, target_revision
        if client_id is not None and client_id != owner and kind != "connected":
            return
        if kind == "quit":
            stop_requested.set()
            return
        with preview_lock:
            if motion_active and kind in ("target", "nudge", "reset", "execute"):
                return
            snapshot = displayed_preview if kind == "execute" else None
            if kind in ("target", "nudge", "reset"):
                displayed_preview = None
                target_revision += 1
            if kind == "target":
                # Coalesce drag updates; never let an old drag backlog delay I/O.
                latest_target = Request(kind, client_id, time.monotonic(), tuple(target), revision=target_revision)
                return
            if kind == "execute":
                displayed_preview = None  # One confirmation can be consumed once.
            revision = target_revision
        try:
            requests.put_nowait(Request(kind, client_id, time.monotonic(), tuple(target), snapshot, revision))
        except queue.Full:
            lost_owner.set()  # Fail closed rather than execute an old backlog.

    def require_connected_operator():
        if stop_requested.is_set() or lost_owner.is_set() or set(server.get_clients()) != {owner}:
            raise RuntimeError("Operator disconnected or requested stop")

    def read_pose():
        observation = arm.get_observation()
        q = np.array([observation[f"{name}.pos"] for name in control.JOINTS], dtype=float)
        control.validate_joints(q, limits)
        opening = float(observation["gripper.pos"])
        if not math.isfinite(opening) or not 0 <= opening <= 100:
            raise ValueError("Gripper readback is outside its calibrated range")
        return q, opening

    try:
        if args.readback:
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
            limits = control.calibrated_limits(arm)
            current, opening = read_pose()
        else:
            current = np.array(control.PREVIEW_JOINTS_DEG)
            opening = None
        control.validate_joints(current, limits)
        startup = current.copy()
        startup_xyz = kinematics.forward_kinematics(startup)[:3, 3].copy()
        target_xyz = startup_xyz.copy()
        target_q = current.copy()
        job = None

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
        target_root = server.scene.add_frame("/preview", show_axes=False, visible=False)
        model_path = Path(args.model_dir) / control.URDF_NAME
        actual_model = ViserUrdf(server, model_path, root_node_name="/current", mesh_color_override=CURRENT_COLOR)
        goal_model = ViserUrdf(server, model_path, root_node_name="/preview", mesh_color_override=TARGET_COLOR)
        names = actual_model.get_actuated_joint_names()
        gizmo = server.scene.add_transform_controls(
            "/target", position=tuple(target_xyz), scale=0.13,
            disable_rotations=True, disable_sliders=True, line_width=4.0,
        )
        zh = args.locale == "zh"
        copy = lambda chinese, english: chinese if zh else english
        mode_label = copy("实机只读 · 不发送动作", "Hardware readback only · no motor commands") if arm else copy("模型预览 · 未连接硬件", "Model preview · no hardware")
        mode_text = server.gui.add_markdown(f"### {mode_label}")
        server.gui.add_markdown(copy(
            "蓝色：当前五轴姿态。橙色：待执行目标。预览运动时先画出夹爪轨迹，蓝色模型再沿轨迹移动。拖动红、绿、蓝轴改变 X、Y、Z；旋转视角不会改变底座坐标。",
            "Blue: current five-joint pose. Orange: target preview. Animation draws the gripper path first, then moves the blue model along it. Drag the red, green, or blue axis for X, Y, Z. Camera rotation does not change the base frame.",
        ))
        feedback = server.gui.add_markdown("")
        nudge_mm = control.STEP_MM if arm else MODEL_STEP_MM
        status = server.gui.add_markdown(copy(f"拖动目标，或用下方按钮沿 Z 轴增加 {nudge_mm:g} mm。", f"Drag the target, or use the button for a {nudge_mm:g} mm Z offset."))
        nudge = server.gui.add_button(copy(f"目标 Z +{nudge_mm:g} mm", f"Target Z +{nudge_mm:g} mm"))
        # Enabled only once a preview has been solved and inspected.
        execute = server.gui.add_button(copy("预览模型运动", "Animate model target"), disabled=True)
        reset = server.gui.add_button(copy("目标回到当前姿态", "Reset target to current pose"))
        quit_button = server.gui.add_button(copy("关闭预览", "Close preview"))
        server.gui.add_markdown(copy(
            "夹爪外形保持参考开度，实机开合百分比单独显示。只读模式不发送目标、不切换力矩；关闭窗口只会断开读取。实机操作仍需稳定支撑和物理断电措施。",
            "Gripper geometry uses a fixed reference opening; hardware opening percentage is shown separately. Readback mode sends no targets and never changes torque. Closing the viewer only disconnects readback. Hardware still requires stable support and physical power cutoff.",
        ))

        @server.on_client_connect
        def connected(client):
            enqueue("connected", client.client_id)

        @server.on_client_disconnect
        def disconnected(client):
            if client.client_id == owner:
                lost_owner.set()

        @gizmo.on_update
        def changed(event):
            # No execution on drag end (Viser can synthesize it on disconnect).
            enqueue("target", event.client_id, gizmo.position)

        nudge.on_click(lambda event: enqueue("nudge", event.client_id))
        execute.on_click(lambda event: enqueue("execute", event.client_id))
        reset.on_click(lambda event: enqueue("reset", event.client_id))
        quit_button.on_click(lambda event: enqueue("quit", event.client_id))

        print(f"Open http://{LOOPBACK}:{args.web_port}; mode={mode_label}")
        while True:
            tick = time.monotonic()
            if stop_requested.is_set():
                return
            if lost_owner.is_set():
                raise RuntimeError("Browser disconnected or input queue overflowed; restart explicitly")
            if arm:
                current, opening = read_pose()
            actual_model.update_cfg(np.array(viewer_configuration(names, current)))
            xyz = kinematics.forward_kinematics(current)[:3, 3].copy()
            coordinates = ", ".join(f"{axis} {value * 1000:.1f}" for axis, value in zip("XYZ", xyz))
            joints_text = " · ".join(f"F{i + 1} {value:.1f}°" for i, value in enumerate(current))
            feedback.content = f"**{copy('当前', 'Current')} / mm**: {coordinates}\n\n{joints_text}" + (
                f"\n\n{copy('夹爪回读', 'Gripper readback')}: {opening:.1f}%" if opening is not None else "")
            # Process a bounded batch and only the latest dragged position.
            batch = []
            for _ in range(requests.qsize()):
                batch.append(requests.get_nowait())
            with preview_lock:
                if latest_target is not None:
                    batch.append(latest_target)
                    latest_target = None
            for request in batch:
                if request.kind == "connected":
                    if owner is None:
                        owner = request.client_id
                    continue
                if request.client_id is not None and request.client_id != owner:
                    continue
                if request.kind == "quit":
                    return
                if job is not None:
                    continue  # Never queue another move while one is executing.
                if request.kind == "execute":
                    try:
                        with preview_lock:
                            if request.revision != target_revision:
                                raise ValueError("Target changed after confirmation; preview again")
                        check_request(request, owner, time.monotonic(), server.get_clients())
                        require_connected_operator()
                        if arm:
                            raise ValueError("Readback mode cannot execute hardware or model motion")
                        check_preview(request.preview, current)
                        target_q = np.array(request.preview.joints)
                        actual = np.array(request.preview.xyz)
                        control.check_step(current, current, startup, actual, startup_xyz, args)
                        if max(abs(target_q - startup)) > args.session_joint_envelope_deg:
                            raise ValueError("Joint session envelope reached")
                        duration = max(float(max(abs(target_q - current))) / JOINT_TARGET_RATE_DEG_S, ANIMATION_MIN_SECONDS)
                        path = trail_points(kinematics, current, target_q)
                        with preview_lock:
                            # Claim only after the possibly blocking fresh read.
                            # A drag during that read invalidates this request.
                            if request.revision != target_revision:
                                raise ValueError("Target changed during readback; preview again")
                            motion_active = True
                            job = (current.copy(), target_q.copy(), time.monotonic(), duration)
                            displayed_preview = None
                            latest_target = None
                            target_revision += 1
                        execute.disabled = True
                        gizmo.visible = False
                        clear_trail()
                        trail.append(server.scene.add_spline_catmull_rom("/trail/path", path, color=TRAIL_COLOR, thickness=0.004))
                        trail.append(server.scene.add_point_cloud("/trail/waypoints", path[::6], np.tile(TRAIL_COLOR, (len(path[::6]), 1)), point_size=0.008, point_shape="circle"))
                    except (ValueError, RuntimeError) as exc:
                        status.content = f"{copy('目标未执行', 'Target not executed')}: {exc}"
                    continue
                clear_trail()  # A new target makes the drawn path stale.
                if request.kind == "reset":
                    target_xyz = xyz.copy()
                    gizmo.position = tuple(target_xyz)
                elif request.kind == "nudge":
                    target_xyz = xyz + np.array([0, 0, nudge_mm / 1000])
                    gizmo.position = tuple(target_xyz)
                elif request.kind == "target":
                    target_xyz = np.asarray(request.target, dtype=float)
                else:
                    continue
                try:
                    with preview_lock:
                        solving_revision = request.revision
                    execute.disabled = True
                    target_q, actual, error = control.solve_position(kinematics, current, target_xyz, limits)
                    if arm:
                        require_connected_operator()
                        # IK is serialized on the I/O owner thread. Refresh after
                        # solving; never treat its pre-solve reading as fresh.
                        fresh, opening = read_pose()
                        check_preview(Preview(tuple(current), tuple(target_q), tuple(actual)), fresh)
                        current = fresh
                    with preview_lock:
                        if solving_revision == target_revision:
                            target_root.visible = True
                            goal_model.update_cfg(np.array(viewer_configuration(names, target_q)))
                            status.content = copy(f"模型误差 {error:.3f} mm；橙色为目标，不是实机反馈。", f"Model residual {error:.3f} mm; orange is a target, not hardware feedback.")
                            displayed_preview = Preview(tuple(current), tuple(target_q), tuple(actual))
                            execute.disabled = bool(arm)
                except (ValueError, RuntimeError) as exc:
                    target_root.visible = False
                    status.content = f"{copy('目标未执行', 'Target not executed')}: {exc}"
            if job is not None:
                initial, goal, began, duration = job
                proposed = np.array(interpolation(initial, goal, eased(min((time.monotonic() - began) / duration, 1.0))))
                require_connected_operator()
                if arm:
                    current, opening = read_pose()
                control.validate_joints(proposed, limits)
                proposed_xyz = kinematics.forward_kinematics(proposed)[:3, 3].copy()
                control.check_step(current, proposed, startup, proposed_xyz, startup_xyz, args)
                if arm:
                    raise RuntimeError("Readback mode must never have an animation job")
                current = proposed  # Explicitly model-only, never hardware feedback.
                finished = time.monotonic() - began >= duration
                if finished and max(abs(current - goal)) <= control.TRACKING_TOLERANCE_DEG:
                    job = None
                    with preview_lock:
                        motion_active = False
                    execute.disabled = bool(arm)
                    gizmo.visible = True
                elif time.monotonic() - began > duration + control.TRACKING_TIMEOUT_SECONDS:
                    raise RuntimeError("Tracking timed out; cut motor power")
            time.sleep(max(0.0, UPDATE_SECONDS - (time.monotonic() - tick)))
    finally:
        try:
            if arm and arm.bus.is_connected:
                arm.bus.disconnect(disable_torque=False)
        finally:
            if server:
                server.stop()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--web-port", type=int, required=True, help="Explicitly allocated local preview port; no automatic choice")
    parser.add_argument("--locale", choices=("zh", "en"), default="zh")
    parser.add_argument("--readback", action="store_true", help="Operator-only, strict read-only serial mode; no motor commands")
    parser.add_argument("--port")
    parser.add_argument("--robot-id")
    parser.add_argument("--calibration-dir")
    parser.add_argument("--confirm-model-match", action="store_true")
    args = parser.parse_args(argv)
    args.max_joint_step_deg = control.MAX_JOINT_STEP_DEG
    args.session_joint_envelope_deg = control.SESSION_JOINT_ENVELOPE_DEG if args.readback else MODEL_JOINT_ENVELOPE_DEG
    args.session_xyz_envelope_mm = control.SESSION_XYZ_ENVELOPE_MM if args.readback else MODEL_XYZ_ENVELOPE_MM
    if args.readback and not all((args.port, args.robot_id, args.calibration_dir, args.confirm_model_match)):
        parser.error("Readback mode requires port, robot-id, calibration-dir, and confirm-model-match")
    if not args.readback and any((args.port, args.robot_id, args.calibration_dir, args.confirm_model_match)):
        parser.error("Hardware arguments require explicit --readback")
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
