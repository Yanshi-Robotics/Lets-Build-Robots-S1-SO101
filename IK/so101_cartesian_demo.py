#!/usr/bin/env python3
"""SO-101 position-only IK, using LeRobot 0.6.1's own solver and the upstream URDF.

Neither mode opens a serial port. `prepare` downloads and verifies the pinned model;
`preview` solves one Cartesian step and prints the joint angles. The two teleoperation
programs in this folder import the model helpers and the pipeline settings from here.

The IK is LeRobot's: `RobotKinematics.inverse_kinematics`, with `orientation_weight=0.0`,
which is what its docstring prescribes for the 5-DOF SO-101. There is no solver of our own.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import urllib.request
import xml.etree.ElementTree as ET
from importlib.metadata import version
from pathlib import Path

# The five positioning joints, in the order LeRobot reports them. The gripper is a 0-100
# opening rather than an angle, so it is outside the kinematic chain but inside the motor list.
JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
MOTORS = (*JOINTS, "gripper")
LEROBOT_VERSION = "0.6.1"
MODEL_COMMIT = "7629d2ad9853d10fb903093a33ef6114099d97e5"
MODEL_BASE = f"https://raw.githubusercontent.com/TheRobotStudio/SO-ARM100/{MODEL_COMMIT}"
URDF_NAME = "so101_new_calib.urdf"
URDF_SHA256 = "3a65d2d35e68a8d2f0c2cc176d19b884506543c93ba72980145b80abe276022c"
# A model-only illustration pose, never a commanded hardware start position.
PREVIEW_JOINTS_DEG = (0.0, -30.0, 60.0, -30.0, 0.0)
STEP_MM = 2.0

# ── Settings for the official end-effector pipeline ──────────────────────────────────────
# One teleoperator unit of delta moves the gripper this far, so a held key travels
# STEP_MM * fps per second: 6 cm/s at the default 30 fps.
EE_STEP_M = STEP_MM / 1000
# EEBoundsAndSafety clips each frame's position change to this. Five nominal steps, so a
# momentary tracking glitch is rate-limited instead of thrown at the arm.
MAX_EE_STEP_M = 5 * EE_STEP_M
# The gripper command is discrete (close / hold / open), which GripperVelocityToJoint turns
# into +-100 before scaling. 0.01 gives one percent of opening per frame, 30 %/s at 30 fps.
GRIPPER_SPEED_FACTOR = 0.01
GRIPPER_MIN_PCT, GRIPPER_MAX_PCT = 0.0, 100.0
# A teaching workspace, not a measured safety guarantee. Sampling the pinned model over its URDF
# joint limits gives a reachable box of x [-0.34, 0.48], y [-0.44, 0.44], z [-0.22, 0.53] metres;
# these bounds sit inside it and above the mounting plane.
# ⚠️ The box must contain the pose the arm is parked in, because EEBoundsAndSafety does not
# reject a pose outside it -- it clips the commanded target to the nearest face, so the arm
# walks to the edge before anyone touches a control. Two measurements from 2026-09-09 set the
# floor and the near wall: a follower parked with the gripper down reads x=0.115, z=0.011, and
# the recorded Rest angles fold it back over the base to x=0.042, z=0.291. z=0 is the base
# mounting plane, which is the table, so the gripper cannot honestly be asked to go below it.
# ⛔ Widening them does not make a rejected target safe. Override only with the arm watched.
DEFAULT_BOUNDS_M = {"min": (0.00, -0.22, 0.00), "max": (0.38, 0.22, 0.42)}
DOWNLOAD_TIMEOUT_SECONDS = 45
DOWNLOAD_ATTEMPTS = 3


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


def model_limits(root):
    """Each positioning joint's travel in degrees, read from the pinned URDF."""
    limits = {}
    for name in JOINTS:
        limit = root.find(f"./joint[@name='{name}']/limit")
        if limit is None:
            raise ValueError(f"Missing joint limits: {name}")
        limits[name] = tuple(math.degrees(float(limit.attrib[key])) for key in ("lower", "upper"))
    return limits


def load_kinematics(directory, lock_wrist_roll=True):
    """LeRobot's own solver on the pinned model, with the roll axis held still.

    Position-only IK leaves wrist_roll free: the gripper point barely moves with it, so the
    solver is at liberty to walk it towards a limit while the position error stays zero.
    `mask_dof` is placo's own way to take a joint out of the problem, and `solver` is a public
    attribute of RobotKinematics -- LeRobot calls `mask_fbase` on it one line after creating it.
    """
    if version("lerobot") != LEROBOT_VERSION:
        raise RuntimeError(f"Use lerobot=={LEROBOT_VERSION}")
    path = Path(directory) / URDF_NAME
    limits = model_limits(load_description(path))
    from lerobot.model.kinematics import RobotKinematics
    kinematics = RobotKinematics(str(path), "gripper_frame_link", list(JOINTS))
    if lock_wrist_roll:
        kinematics.solver.mask_dof("wrist_roll")
    return kinematics, limits


def validate_joints(joints, limits):
    if len(joints) != len(JOINTS):
        raise ValueError("Expected five arm joints, in the documented order")
    for name, value in zip(JOINTS, joints):
        low, high = limits[name]
        if not math.isfinite(float(value)) or not low <= value <= high:
            raise ValueError(f"Joint outside allowed range: {name}={value}; allowed {low:.2f}..{high:.2f} deg")


def bounds_dict(low, high):
    """The end_effector_bounds mapping EEBoundsAndSafety expects, validated."""
    import numpy as np
    low, high = np.asarray(low, dtype=float), np.asarray(high, dtype=float)
    if low.shape != (3,) or high.shape != (3,) or not np.isfinite(low).all() or not np.isfinite(high).all():
        raise ValueError("Workspace bounds must be three finite metres each")
    if not (low < high).all():
        raise ValueError("Every workspace minimum must be below its maximum")
    return {"min": low, "max": high}


def build_robot_action_processor(kinematics, bounds, step_m=EE_STEP_M, max_step_m=MAX_EE_STEP_M):
    """The official end-effector pipeline, assembled and not modified.

    Every step below ships with LeRobot 0.6.1. No command-line entry point wires them up --
    `lerobot-teleoperate` hardcodes the identity processors -- so assembling them is the only
    thing these programs add. `raise_on_jump=False` is the choice that step's own docstring
    calls the safer one for live teleoperation: rate-limit a glitch rather than abandon the arm.
    """
    from lerobot.processor import MapDeltaActionToRobotActionStep, RobotProcessorPipeline
    from lerobot.processor.converters import (
        robot_action_observation_to_transition,
        transition_to_robot_action,
    )
    from lerobot.robots.so_follower.robot_kinematic_processor import (
        EEBoundsAndSafety,
        EEReferenceAndDelta,
        GripperVelocityToJoint,
        InverseKinematicsEEToJoints,
    )

    return RobotProcessorPipeline(
        steps=[
            MapDeltaActionToRobotActionStep(),
            EEReferenceAndDelta(
                kinematics=kinematics,
                end_effector_step_sizes={"x": step_m, "y": step_m, "z": step_m},
                motor_names=list(MOTORS),
                # Follow the arm rather than a pose latched when the key went down, so a held
                # key keeps moving and a released one leaves the arm where it stands.
                use_latched_reference=False,
            ),
            EEBoundsAndSafety(
                end_effector_bounds=bounds,
                max_ee_step_m=max_step_m,
                raise_on_jump=False,
            ),
            GripperVelocityToJoint(
                speed_factor=GRIPPER_SPEED_FACTOR,
                clip_min=GRIPPER_MIN_PCT,
                clip_max=GRIPPER_MAX_PCT,
                discrete_gripper=True,
            ),
            InverseKinematicsEEToJoints(
                kinematics=kinematics,
                motor_names=list(MOTORS),
                # The step's own docstring: 0.0 is position-only IK, for under-actuated arms
                # like the SO-101, which cannot reach an arbitrary orientation anyway.
                orientation_weight=0.0,
            ),
        ],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )


# A joint this close to either end of its modelled travel is treated as sitting on the stop.
STOP_MARGIN_DEG = 3.0


def check_start_pose(observation, kinematics, bounds, limits=None, margin=STOP_MARGIN_DEG):
    """Where the gripper is now, refusing to start from a pose the arm cannot leave.

    This runs once, before the loop, and reads nothing but the observation the caller already
    has. It is not in the control chain. Two things make a pose unusable:

    A pose outside the workspace is not rejected downstream but *corrected*: EEBoundsAndSafety
    clips the first target to the nearest face and the arm walks there on its own, which is the
    one motion nobody asked for.

    A joint resting on its stop has nowhere to go in one direction, and the first command out of
    there is the hardest one the arm will ever be asked for: on 2026-09-08 a follower folded onto
    its shoulder stop drove at full load into that stop and tripped the servo's overload
    protection. Fold the arm to a pose it can hold before powering it.
    """
    import numpy as np
    joints = np.array([float(observation[f"{name}.pos"]) for name in JOINTS])
    if limits:
        on_a_stop = [
            f"{name} at {value:+.1f} deg, against its {low:+.1f}..{high:+.1f} travel"
            for name, value, (low, high) in ((n, float(v), limits[n]) for n, v in zip(JOINTS, joints))
            if value < low + margin or value > high - margin
        ]
        if on_a_stop:
            raise RuntimeError(
                "These joints are resting on a stop, so the arm was never powered:\n    "
                + "\n    ".join(on_a_stop)
                + "\nWith motor power off, move the arm to a pose it can hold clear of its ends, "
                "then run this again. Starting on a stop asks the first command to lift the arm "
                "straight off it."
            )
    here = kinematics.forward_kinematics(joints)[:3, 3]
    low, high = np.asarray(bounds["min"], dtype=float), np.asarray(bounds["max"], dtype=float)
    outside = [axis for index, axis in enumerate("xyz") if not low[index] <= here[index] <= high[index]]
    if outside:
        raise RuntimeError(
            f"The gripper is at x={here[0]:.3f} y={here[1]:.3f} z={here[2]:.3f} m, outside the "
            f"workspace on {', '.join(outside)}.\n"
            f"    workspace  min {tuple(round(float(v), 3) for v in low)}  "
            f"max {tuple(round(float(v), 3) for v in high)}\n"
            "Move the arm into the workspace by hand with motor power off, or pass --bounds-min-m "
            "and --bounds-max-m for a box that contains this pose. Starting here would have the "
            "arm travel to the nearest edge before you touch anything."
        )
    return here


def preview(args):
    """One Cartesian step through LeRobot's solver, reported with its own FK residual."""
    import numpy as np
    kinematics, limits = load_kinematics(args.model_dir)
    seed = np.asarray(args.joints_deg, dtype=float)
    validate_joints(seed, limits)
    start_pose = kinematics.forward_kinematics(seed)
    start = start_pose[:3, 3].copy()
    target = start + np.asarray(args.delta_mm, dtype=float) / 1000
    target_pose = start_pose.copy()
    target_pose[:3, 3] = target
    solved = kinematics.inverse_kinematics(seed, target_pose, position_weight=1.0, orientation_weight=0.0)
    actual = kinematics.forward_kinematics(solved)[:3, 3]
    print(json.dumps({
        "mode": "model-only; no hardware",
        "solver": f"lerobot {LEROBOT_VERSION} RobotKinematics, orientation_weight=0.0, wrist_roll masked",
        "joint_names": JOINTS,
        "start_degrees": seed.tolist(), "solved_degrees": solved.tolist(),
        "start_xyz_m": start.tolist(), "target_xyz_m": target.tolist(),
        "actual_xyz_m": actual.tolist(),
        "position_error_mm": float(np.linalg.norm(actual - target) * 1000),
    }, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    for mode in ("prepare", "preview"):
        sub = commands.add_parser(mode)
        sub.add_argument("--model-dir", required=True)
        if mode == "preview":
            sub.add_argument("--joints-deg", nargs=5, type=float, default=PREVIEW_JOINTS_DEG)
            sub.add_argument("--delta-mm", nargs=3, type=float, default=(0.0, 0.0, STEP_MM))
    args = parser.parse_args(argv)
    try:
        if args.mode == "prepare":
            prepare_model(Path(args.model_dir))
        else:
            preview(args)
        return 0
    except KeyboardInterrupt:
        print("Stopped.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"STOP: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
