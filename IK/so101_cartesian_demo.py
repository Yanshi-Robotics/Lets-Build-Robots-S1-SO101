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
# A reading taken on a mechanical stop can sit past the recorded bound: the range was
# recorded by hand with torque off, and a motor holding against the same stop compresses
# the printed part further. Measured 2026-09-08 on a Follower: 1.7 deg past the recorded
# shoulder_lift minimum with torque on. The margin still catches a wrong calibration file,
# which is tens of degrees off, without refusing an honest rest pose.
LIMIT_MARGIN_DEG = 5.0
# Joints resting this close to the end of their recorded travel are named at power-on, so the
# operator knows which ones the soft start below is protecting. 2026-09-08 incident: torque
# enabled with shoulder_lift and wrist_flex on their stops drove both at 100 % load into the
# stops within 250 ms and tripped the servos' overload protection (output cut to 20 %, error
# bit set on every acknowledged write afterwards) - with the goal verified equal to the present
# position beforehand, so the servos drove at a target that was not the one in their register.
POWER_ON_MARGIN_DEG = 5.0
# Soft start: torque comes on with Torque_Limit lowered, the goal is written once more with torque
# on, and the arm is watched before each motor's own Max_Torque_Limit is restored. Holding a mid
# pose reads 0-5 % load on this arm; a push into a stop saturates at the lowered limit, which is
# far below what compresses a stop or trips overload.
SOFT_START_TORQUE_LIMIT = 300  # of 1000
SOFT_START_SECONDS = 1.0
# What counts as a fault while watching. A gravity-loaded joint sags a little and then holds, because
# a proportional servo only makes force from error; that is not a fault. Running away from the goal
# is, and so is pinning the output at the lowered limit instead of settling.
SOFT_START_RUNAWAY_STEPS = 120  # about 10.5 deg away from the goal it was given
SOFT_START_SATURATED_PCT = 25.0  # against the 30 % limit: pushing as hard as it is allowed to
SOFT_START_SATURATED_SECONDS = 0.5
# Hold and goal commands never point at the very end of the recorded travel.
HOLD_MARGIN_DEG = 2.0
WRIST_ROLL_LOCK_WEIGHT = 1.0  # Soft joints task: position-only IK must not drift the roll axis.
MAX_JOINT_STEP_DEG = 2.0
SESSION_JOINT_ENVELOPE_DEG = 8.0
SESSION_XYZ_ENVELOPE_MM = 20.0
IK_TOLERANCE_MM = 0.1
IK_MAX_ITERATIONS = 100
TRACKING_TOLERANCE_DEG = 0.8
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


def joints_on_a_stop(joints, reading_limits, margin=POWER_ON_MARGIN_DEG):
    """Joints closer than margin to either end of the recorded travel: torque must not be enabled there."""
    blockers = []
    for name, value in zip(JOINTS, joints):
        low, high = reading_limits[name]
        # reading_limits already carry LIMIT_MARGIN_DEG outwards; measure from the recorded ends themselves.
        low_end, high_end = low + LIMIT_MARGIN_DEG, high - LIMIT_MARGIN_DEG
        if float(value) < low_end + margin or float(value) > high_end - margin:
            blockers.append((name, float(value), (low_end, high_end)))
    return blockers


def clamp_for_hold(joints, reading_limits, margin=HOLD_MARGIN_DEG):
    """A hold target inside the recorded travel, never on its ends."""
    out = []
    for name, value in zip(JOINTS, joints):
        low, high = reading_limits[name]
        low_end, high_end = low + LIMIT_MARGIN_DEG, high - LIMIT_MARGIN_DEG
        out.append(float(min(max(float(value), low_end + margin), high_end - margin)))
    return out


def lock_wrist_roll(kinematics, degrees):
    """Keep wrist_roll at the arm's present angle while solving position-only IK.

    The gripper position barely depends on the roll axis, so without this the solver is free
    to walk it towards a limit: a dragged goal once planned a 155-degree roll.
    """
    task = getattr(kinematics, "_wrist_roll_task", None)
    if task is None:
        task = kinematics.solver.add_joints_task()
        task.configure("wrist_roll_lock", "soft", WRIST_ROLL_LOCK_WEIGHT)
        kinematics._wrist_roll_task = task
    task.set_joints({"wrist_roll": math.radians(float(degrees))})


def release_torque(bus):
    """Switch torque off on every motor, tolerating motors that answer with an error status.

    A servo in overload protection acknowledges writes with its error bit set, which the bus
    layer reports as a failure even though the register took the value. Write each motor on
    its own, then broadcast, then read back. Returns the motors that could not be confirmed off.
    """
    problems = {}
    for name in bus.motors:
        try:
            bus.write("Torque_Enable", name, 0, normalize=False, num_retry=2)
        except Exception as exc:  # noqa: BLE001 - keep going: the next motor must still be released
            problems[name] = f"write: {exc}"
    try:
        bus.sync_write("Torque_Enable", {name: 0 for name in bus.motors}, normalize=False)
    except Exception as exc:  # noqa: BLE001
        problems["broadcast"] = str(exc)
    unconfirmed = {}
    for name in bus.motors:
        try:
            if bus.read("Torque_Enable", name, normalize=False, num_retry=2) != 0:
                unconfirmed[name] = "torque bit still set"
        except Exception as exc:  # noqa: BLE001
            unconfirmed[name] = f"no clean answer ({exc})"
    return unconfirmed


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
    """Pure checks used before every hardware action; no clamping of bad solutions.

    The step limit leaves room for the settling tolerance: the loop accepts an arm that stopped
    TRACKING_TOLERANCE_DEG short of its last target, and the next step is measured from there.
    """
    step_limit = args.max_joint_step_deg - TRACKING_TOLERANCE_DEG
    if max(abs(float(a) - float(b)) for a, b in zip(proposed, current)) > step_limit:
        raise ValueError(f"Joint step is too large (limit {step_limit:.2f} deg); choose a smaller Cartesian step or another pose")
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


class SoftStartFailed(RuntimeError):
    """Torque was released again during the soft start. `unconfirmed` names motors whose release could not be confirmed."""

    def __init__(self, message, unconfirmed=None):
        super().__init__(message)
        self.unconfirmed = dict(unconfirmed or {})


def _watch_soft_start(arm, reference, log):
    """Torque is on at the lowered limit. Write the goal once more, then watch it for a second.

    The goal is never rewritten while watching. A proportional servo makes force only from error, so
    a joint carrying its own weight has to sag a few degrees before it holds; rewriting the goal to
    wherever it has sagged sets that error back to zero and it sags again. Doing that a few times in
    a row made a healthy arm look like it was driving itself, and the earlier version released torque
    and dropped it. Sagging and settling is what this arm is supposed to do.

    Torque is released only for the two things that are not that: a joint that runs away from the goal
    it was given, and one that holds its output at the lowered limit instead of settling.
    """
    # A goal written with torque off is what the 2026-09-08 incident showed the servos not to follow.
    arm.bus.sync_write("Goal_Position", reference, normalize=False)
    saturated = dict.fromkeys(arm.bus.motors, 0.0)
    deadline = time.monotonic() + SOFT_START_SECONDS
    while time.monotonic() < deadline:
        time.sleep(READ_INTERVAL_SECONDS)
        present = arm.bus.sync_read("Present_Position", normalize=False, num_retry=2)
        load = arm.bus.sync_read("Present_Load", normalize=False, num_retry=2)
        if log:
            log("soft start present=%s goal2=%s load=%s current=%s",
                present, arm.bus.sync_read("Goal_Position_2", normalize=False, num_retry=2),
                load, arm.bus.sync_read("Present_Current", normalize=False, num_retry=2))
        runaway = {name: present[name] - reference[name] for name in arm.bus.motors
                   if abs(present[name] - reference[name]) > SOFT_START_RUNAWAY_STEPS}
        for name in arm.bus.motors:
            saturated[name] = saturated[name] + READ_INTERVAL_SECONDS if abs(load[name]) / 10 >= SOFT_START_SATURATED_PCT else 0.0
        pinned = [name for name, seconds in saturated.items() if seconds >= SOFT_START_SATURATED_SECONDS]
        if runaway or pinned:
            unconfirmed = release_torque(arm.bus)
            detail = ", ".join(
                [f"{name} ran {steps * 360 / 4095:+.1f} deg from its goal" for name, steps in runaway.items()]
                + [f"{name} held its output at the soft-start limit" for name in pinned])
            raise SoftStartFailed(f"a motor drove itself at power-on; torque released. {detail}", unconfirmed)


def configure_and_hold(arm, position_mode, log=None):
    """Configure, hold the present pose, and enable torque through a soft start.

    Torque comes on with every motor's Torque_Limit lowered to SOFT_START_TORQUE_LIMIT; the goal
    is written once more with torque on and the arm is watched (see _watch_soft_start); only then
    is each motor's own Max_Torque_Limit restored. If anything goes wrong while torque is on at
    the lowered limit, torque is released before the error is raised: the caller never inherits
    an energised arm it does not know about.
    """
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
    # Each motor's own ceiling (the gripper's is lower); Torque_Limit returns to it after the watch.
    normal_limit = arm.bus.sync_read("Max_Torque_Limit", normalize=False)
    arm.bus.sync_write("Torque_Limit", {name: SOFT_START_TORQUE_LIMIT for name in arm.bus.motors}, normalize=False)
    arm.bus.sync_write("Goal_Position", raw, normalize=False)
    # Broadcast sync_write has no per-motor acknowledgement. Verify every hold
    # target before enabling, so an undelivered packet cannot leave an old goal.
    for name, target in raw.items():
        if arm.bus.read("Goal_Position", name, normalize=False) != target:
            raise RuntimeError(f"Hold target not confirmed for {name}; torque remains disabled")
    arm.bus.enable_torque()
    try:
        _watch_soft_start(arm, raw, log)
        arm.bus.sync_write("Torque_Limit", normal_limit, normalize=False)
    except SoftStartFailed:
        raise
    except Exception as exc:  # noqa: BLE001 - a bus fault here would otherwise leave torque on unannounced
        unconfirmed = release_torque(arm.bus)
        raise SoftStartFailed(f"bus fault during the soft start ({exc}); torque released", unconfirmed) from exc


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
        use_degrees=True, max_relative_target=None,
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
            arm.send_action({f"{name}.pos": float(value) for name, value in zip(JOINTS, proposed)})
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
                unconfirmed = release_torque(arm.bus)
                if unconfirmed:
                    print(f"Torque not confirmed off on {unconfirmed}; cut DC power.", file=sys.stderr)
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
        if args.max_joint_step_deg <= TRACKING_TOLERANCE_DEG:
            parser.error(f"max_joint_step_deg must exceed the {TRACKING_TOLERANCE_DEG:g} deg settling tolerance")
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
