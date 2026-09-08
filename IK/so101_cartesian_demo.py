#!/usr/bin/env python3
"""SO-101 position-only IK demonstration, using LeRobot 0.6.1 and the upstream URDF.

prepare and preview never import robot drivers or open ports. jog is an explicit
operator-only hardware mode. Software limits are not collision detection, a speed
controller, or an emergency stop. Keep a physical motor-power cutoff available.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET
from importlib.metadata import version
from pathlib import Path

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
LEROBOT_VERSION = "0.6.1"
MODEL_COMMIT = "7629d2ad9853d10fb903093a33ef6114099d97e5"
MODEL_BASE = f"https://raw.githubusercontent.com/TheRobotStudio/SO-ARM100/{MODEL_COMMIT}"
URDF_NAME = "so101_new_calib.urdf"
URDF_SHA256 = "3a65d2d35e68a8d2f0c2cc176d19b884506543c93ba72980145b80abe276022c"
# A model-only illustration pose, never a commanded hardware start position.
PREVIEW_JOINTS_DEG = (0.0, -30.0, 60.0, -30.0, 0.0)
# Conservative teaching bounds, not measured hardware safety guarantees.
STEP_MM = 2.0
# A reading taken on a mechanical stop equals the recorded bound up to float rounding
# and a little mechanical slack; without this margin every honest rest pose is refused.
LIMIT_MARGIN_DEG = 1.0
MAX_JOINT_STEP_DEG = 2.0
SESSION_JOINT_ENVELOPE_DEG = 8.0
SESSION_XYZ_ENVELOPE_MM = 20.0
IK_TOLERANCE_MM = 0.1
IK_MAX_ITERATIONS = 100
TRACKING_TOLERANCE_DEG = 0.8
# Floating-point comparison of requested vs returned API targets, not FK error.
SENT_TARGET_TOLERANCE_DEG = 1e-6
TRACKING_TIMEOUT_SECONDS = 2.0
READ_INTERVAL_SECONDS = 0.05
DOWNLOAD_TIMEOUT_SECONDS = 45
DOWNLOAD_ATTEMPTS = 3
# Match the official SOFollower gripper protection configuration in LeRobot 0.6.1.
GRIPPER_PROTECTION_REGISTERS = {"Max_Torque_Limit": 500, "Protection_Current": 250, "Overload_Torque": 25}


def prepare_model(directory: Path):
    """Download only this pinned model and its referenced geometry and license."""
    directory.mkdir(parents=True, exist_ok=True)

    def fetch(relative, destination):
        if destination.exists():
            return
        destination.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(DOWNLOAD_ATTEMPTS):
            try:
                with urllib.request.urlopen(f"{MODEL_BASE}/{relative}", timeout=DOWNLOAD_TIMEOUT_SECONDS) as response:
                    data = response.read()
                destination.write_bytes(data)
                return
            except (OSError, TimeoutError):
                if attempt + 1 == DOWNLOAD_ATTEMPTS:
                    raise

    path = directory / URDF_NAME
    fetch(f"Simulation/SO101/{URDF_NAME}", path)
    robot = load_description(path)
    for mesh in robot.findall(".//mesh"):
        relative = Path(mesh.attrib["filename"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unexpected mesh path in pinned model")
        fetch(f"Simulation/SO101/{relative.as_posix()}", directory / relative)
    fetch("LICENSE", directory / "LICENSE")
    print(f"Model ready: {path}\nSource commit: {MODEL_COMMIT}")


def load_description(path):
    if hashlib.sha256(Path(path).read_bytes()).hexdigest() != URDF_SHA256:
        raise ValueError("URDF does not match the pinned new-calibration SO-101 model")
    return ET.parse(path).getroot()


def load_kinematics(directory):
    if version("lerobot") != LEROBOT_VERSION:
        raise RuntimeError(f"Use lerobot=={LEROBOT_VERSION}")
    path = Path(directory) / URDF_NAME
    root = load_description(path)
    limits = {}
    for name in JOINTS:
        limit = root.find(f"./joint[@name='{name}']/limit")
        if limit is None:
            raise ValueError(f"Missing joint limits: {name}")
        limits[name] = tuple(math.degrees(float(limit.attrib[key])) for key in ("lower", "upper"))
    from lerobot.model.kinematics import RobotKinematics
    return RobotKinematics(str(path), "gripper_frame_link", list(JOINTS)), limits


def validate_joints(joints, limits):
    if len(joints) != len(JOINTS):
        raise ValueError("Expected five arm joints, in the documented order")
    for name, value in zip(JOINTS, joints):
        low, high = limits[name]
        if not math.isfinite(float(value)) or not low <= value <= high:
            raise ValueError(f"Joint outside allowed range: {name}={value}; allowed {low:.2f}..{high:.2f} deg")


def solve_position(kinematics, current, target_xyz, limits):
    """Iterate the official soft IK step, then require an independent FK residual."""
    import numpy as np
    q = np.asarray(current, dtype=float).copy()
    target_xyz = np.asarray(target_xyz, dtype=float)
    validate_joints(q, limits)
    if target_xyz.shape != (3,) or not np.isfinite(target_xyz).all():
        raise ValueError("Target must be three finite coordinates in metres")
    target_pose = kinematics.forward_kinematics(q).copy()
    target_pose[:3, 3] = target_xyz
    for _ in range(IK_MAX_ITERATIONS):
        q = kinematics.inverse_kinematics(q, target_pose, position_weight=1.0, orientation_weight=0.0)
        validate_joints(q, limits)
        actual = kinematics.forward_kinematics(q)[:3, 3]
        residual_mm = float(np.linalg.norm(actual - target_xyz) * 1000)
        if residual_mm <= IK_TOLERANCE_MM:
            return q, actual, residual_mm
    raise ValueError("IK did not reach the target within tolerance; no action is allowed")


def check_step(current, proposed, startup, xyz, startup_xyz, args):
    """Pure checks used before every hardware action; no clamping of bad solutions."""
    if max(abs(float(a) - float(b)) for a, b in zip(proposed, current)) > args.max_joint_step_deg:
        raise ValueError("Joint step is too large; choose a smaller Cartesian step or another pose")
    if max(abs(float(a) - float(b)) for a, b in zip(proposed, startup)) > args.session_joint_envelope_deg:
        raise ValueError("Joint session envelope reached; do not expand it to bypass a rejection")
    if max(abs(float(a) - float(b)) * 1000 for a, b in zip(xyz, startup_xyz)) > args.session_xyz_envelope_mm:
        raise ValueError("Cartesian session envelope reached")


def calibrated_limits(arm):
    """Validate hardware readings against the arm's own recorded travel, not the URDF.

    The pinned URDF stops shoulder_lift at ±100°, while an arm calibrated to its real
    mechanical stops records about ±105°; a resting arm sits on a stop, so checking
    readings against the URDF refused every honest rest pose. Zero is the middle of the
    recorded range, exactly as LeRobot's degree normalisation defines it. IK targets are
    still confined to the URDF: the solver enforces those limits itself.
    """
    limits = {}
    for name in JOINTS:
        calibration = arm.calibration[name]
        resolution = arm.bus.model_resolution_table[arm.bus.motors[name].model] - 1
        half_range_deg = (calibration.range_max - calibration.range_min) * 180 / resolution
        limits[name] = (-half_range_deg - LIMIT_MARGIN_DEG, half_range_deg + LIMIT_MARGIN_DEG)
    return limits


def configure_and_hold(arm, position_mode):
    """Configure without an auto-reenabling context; enable only after all checks."""
    arm.bus.disable_torque()
    arm.bus.configure_motors()
    for name in arm.bus.motors:
        arm.bus.write("Operating_Mode", name, position_mode)
        for register, value in (("P_Coefficient", arm.config.position_p_coefficient),
                                ("I_Coefficient", arm.config.position_i_coefficient),
                                ("D_Coefficient", arm.config.position_d_coefficient)):
            arm.bus.write(register, name, value)
        if name == "gripper":
            for register, value in GRIPPER_PROTECTION_REGISTERS.items():
                arm.bus.write(register, name, value)
    # Phase/mode are now final. Validate the same raw snapshot used for hold.
    raw = arm.bus.sync_read("Present_Position", normalize=False)
    degrees = []
    for name in JOINTS:
        calibration = arm.calibration[name]
        resolution = arm.bus.model_resolution_table[arm.bus.motors[name].model] - 1
        degrees.append((raw[name] - (calibration.range_min + calibration.range_max) / 2) * 360 / resolution)
    validate_joints(degrees, calibrated_limits(arm))
    gripper = arm.calibration["gripper"]
    if not gripper.range_min <= raw["gripper"] <= gripper.range_max:
        raise ValueError("Gripper is outside its calibrated range")
    arm.bus.sync_write("Goal_Position", raw, normalize=False)
    # Broadcast sync_write has no per-motor acknowledgement. Verify every hold
    # target before enabling, so an undelivered packet cannot leave an old goal.
    for name, target in raw.items():
        if arm.bus.read("Goal_Position", name, normalize=False) != target:
            raise RuntimeError(f"Hold target not confirmed for {name}; torque remains disabled")
    arm.bus.enable_torque()


def preview(args):
    import numpy as np
    kinematics, limits = load_kinematics(args.model_dir)
    seed = np.asarray(args.joints_deg, dtype=float)
    validate_joints(seed, limits)
    start = kinematics.forward_kinematics(seed)[:3, 3].copy()
    target = start + np.asarray(args.delta_mm, dtype=float) / 1000
    solved, actual, residual = solve_position(kinematics, seed, target, limits)
    print(json.dumps({"mode": "model-only; no hardware", "joint_names": JOINTS,
        "start_degrees": seed.tolist(), "solved_degrees": solved.tolist(),
        "start_xyz_m": start.tolist(), "target_xyz_m": target.tolist(),
        "actual_xyz_m": actual.tolist(), "position_error_mm": residual}, indent=2))


def jog(args):
    import numpy as np
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
    from lerobot.motors.feetech import OperatingMode

    kinematics, _model_limits = load_kinematics(args.model_dir)

    class HoldCurrentFollower(SO101Follower):
        def configure(self):
            # connect(calibrate=False) still invokes configure(). Check calibration
            # before writes, then replace stale targets before base configure enables torque.
            if not self.calibration or not self.is_calibrated:
                raise RuntimeError("Calibration is missing or does not match motors; return to Communication and Calibration")
            if any(self.bus.read("Torque_Enable", name, normalize=False) != 0 for name in self.bus.motors):
                raise RuntimeError("Torque already enabled. Cut power and investigate before starting")
            observation = self.get_observation()
            q = [observation[f"{name}.pos"] for name in JOINTS]
            validate_joints(q, calibrated_limits(self))
            print("Current joint degrees:", dict(zip(JOINTS, q)))
            print("Clear the workspace. Enabling torque can move the arm; support it without entering pinch points.")
            if input("Type ENABLE to hold the current pose, or anything else to exit: ").strip() != "ENABLE":
                raise KeyboardInterrupt
            configure_and_hold(self, OperatingMode.POSITION.value)

    arm = HoldCurrentFollower(SO101FollowerConfig(
        port=args.port, id=args.robot_id, calibration_dir=Path(args.calibration_dir),
        use_degrees=True, max_relative_target=args.max_joint_step_deg,
        disable_torque_on_disconnect=True, cameras={},
    ))
    try:
        arm.connect(calibrate=False)
        limits = calibrated_limits(arm)
        def read_joints():
            observation = arm.get_observation()
            q = np.array([observation[f"{name}.pos"] for name in JOINTS], dtype=float)
            validate_joints(q, limits)
            return q
        startup = read_joints()
        startup_xyz = kinematics.forward_kinematics(startup)[:3, 3].copy()
        print("Commands: x+, x-, y+, y-, z+, z-, q. Each command sends one small position target.")
        print("Jog does not change gripper opening; startup holds all six motors. No collision detection.")
        print("q releases torque; support the arm before exiting.")
        while True:
            command = input("jog> ").strip().lower()
            if command == "q":
                break
            if command not in ("x+", "x-", "y+", "y-", "z+", "z-"):
                print("Enter one of the listed commands; nothing sent.")
                continue
            current = read_joints()
            xyz = kinematics.forward_kinematics(current)[:3, 3].copy()
            target = xyz.copy()
            target["xyz".index(command[0])] += (1 if command[1] == "+" else -1) * args.step_mm / 1000
            try:
                proposed, actual, residual = solve_position(kinematics, current, target, limits)
                check_step(current, proposed, startup, actual, startup_xyz, args)
            except (ValueError, RuntimeError) as exc:
                print(f"REJECTED: {exc}. Nothing sent; the previous hold remains active.")
                continue
            # Send only the five arm joints: never convert gripper percentage to radians.
            sent = arm.send_action({f"{name}.pos": float(value) for name, value in zip(JOINTS, proposed)})
            if any(abs(sent[f"{name}.pos"] - proposed[i]) > SENT_TARGET_TOLERANCE_DEG for i, name in enumerate(JOINTS)):
                raise RuntimeError("LeRobot clipped the joint target; stop rather than treating it as reached")
            deadline = time.monotonic() + TRACKING_TIMEOUT_SECONDS
            while max(abs(read_joints() - proposed)) > TRACKING_TOLERANCE_DEG:
                if time.monotonic() > deadline:
                    raise RuntimeError("Motor tracking timed out; cut power and inspect the arm")
                time.sleep(READ_INTERVAL_SECONDS)
            print(f"Model target (m): {target}; model residual: {residual:.4f} mm. Verify real motion visually.")
    finally:
        # Covers partially failed connect as well. Communication faults can prevent
        # torque release; only the operator can physically cut the DC supply.
        if arm.bus.is_connected:
            try:
                arm.bus.disable_torque()
            finally:
                arm.bus.disconnect(disable_torque=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    for mode in ("prepare", "preview", "jog"):
        sub = commands.add_parser(mode)
        sub.add_argument("--model-dir", required=True)
        if mode == "preview":
            sub.add_argument("--joints-deg", nargs=5, type=float, default=PREVIEW_JOINTS_DEG)
            sub.add_argument("--delta-mm", nargs=3, type=float, default=(0.0, 0.0, STEP_MM))
        if mode == "jog":
            sub.add_argument("--port", required=True)
            sub.add_argument("--robot-id", required=True)
            sub.add_argument("--calibration-dir", required=True)
            sub.add_argument("--confirm-model-match", action="store_true", required=True,
                             help="Operator has checked the physical joint layout, directions and model/calibration convention")
            sub.add_argument("--step-mm", type=float, default=STEP_MM)
            sub.add_argument("--max-joint-step-deg", type=float, default=MAX_JOINT_STEP_DEG)
            sub.add_argument("--session-joint-envelope-deg", type=float, default=SESSION_JOINT_ENVELOPE_DEG)
            sub.add_argument("--session-xyz-envelope-mm", type=float, default=SESSION_XYZ_ENVELOPE_MM)
    args = parser.parse_args(argv)
    if args.mode == "jog":
        for key in ("step_mm", "max_joint_step_deg", "session_joint_envelope_deg", "session_xyz_envelope_mm"):
            if not math.isfinite(getattr(args, key)) or getattr(args, key) <= 0:
                parser.error(f"{key} must be finite and positive")
    try:
        if args.mode == "prepare":
            prepare_model(Path(args.model_dir))
        elif args.mode == "preview":
            preview(args)
        else:
            jog(args)
        return 0
    except KeyboardInterrupt:
        print("Stopped. This is not a physical power cutoff.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"STOP: {exc}\nFor hardware: cut motor power before investigating. Do not force a rejected target.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
