"""The SO-101 as a kinematic model: URDF in, poses out.

This is the only file (with `so101_arm.py`) that knows it is an SO-101. It owns:

* the joint list and the tool frame name,
* forward kinematics (FK) through placo / pinocchio,
* the two orientation helpers `compose` and `decompose` that turn the user's
  (pitch, roll) into a rotation matrix and back,
* the joint limits, either from the URDF or narrowed to what this particular arm's
  calibration file recorded.

Facts about the SO-101 that the rest of the pipeline relies on (all measured on the URDF
with placo, see README):

* 5 arm joints + 1 gripper. shoulder_pan turns about the vertical axis; shoulder_lift,
  elbow_flex and wrist_flex are three parallel pitch axes; wrist_roll spins about the
  tool's approach axis. The tool's pitch is therefore the sum of the three pitch joints.
* The tool's yaw is fixed by shoulder_pan. A user may choose x, y, z, pitch and roll;
  yaw is whatever the arm has, so `pose_from_target` takes the current yaw as an input.
* The tool frame `gripper_frame_link`: its z axis is the approach direction (= the
  wrist_roll axis). At the all-zero pose the arm points along +x and the tool z axis is
  exactly +x; the tool x axis points roughly down with a 2.8 degree built-in tilt, which
  is why "roll = 0" is defined as "whatever orientation wrist_roll = 0 gives".
* The tool point sits 7.9 mm off the wrist_roll axis, so roll changes position a little
  and must be solved together with position, not written straight to the joint.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import placo

from target import Target

# Joint order = motor id order (1..6). Angles cross this package as a 6-vector `q`:
# q[0:5] in radians for the five arm joints, q[5] in percent (0..100) for the gripper.
ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
GRIPPER = "gripper"
ALL_JOINTS = (*ARM_JOINTS, GRIPPER)
GRIPPER_INDEX = 5

# Each joint's frame in the URDF is its child link's frame, and every joint axis is the
# local z axis of that frame. The joint rings in the viewer are placed on these frames.
JOINT_CHILD_LINK = {
    "shoulder_pan": "shoulder_link",
    "shoulder_lift": "upper_arm_link",
    "elbow_flex": "lower_arm_link",
    "wrist_flex": "wrist_link",
    "wrist_roll": "gripper_link",
    "gripper": "moving_jaw_so101_v1_link",
}

URDF_NAME = "so101_new_calib.urdf"  # TheRobotStudio/SO-ARM100, the LeRobot "new calibration" zero convention
TOOL_FRAME = "gripper_frame_link"

# STS3215 has 4096 positions per turn; LeRobot's degree normalisation divides by 4095.
MOTOR_RESOLUTION = 4096
# How exactly the tool z axis must equal +x at the zero pose for compose/decompose to be
# valid. The Onshape export gives it to 1.3e-5 (0.0008 degrees); anything worse than this
# tolerance means a different URDF was loaded.
TOOL_AXIS_TOLERANCE = 1e-3
# Pan angle used once at start-up to learn the sign between shoulder_pan and tool yaw.
YAW_PROBE_RAD = 0.5


def rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def wrap_angle(a: float) -> float:
    """Map any angle into (-pi, pi]."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def calibration_half_travel_deg(calibration: dict) -> dict[str, float]:
    """LeRobot's degrees put zero at the middle of the recorded range, so the reachable
    span is +/- half of it: (range_max - range_min) / 2 * 360 / (resolution - 1)."""
    out = {}
    for name, entry in calibration.items():
        span = entry["range_max"] - entry["range_min"]
        out[name] = span / 2.0 * 360.0 / (MOTOR_RESOLUTION - 1)
    return out


class Model:
    def __init__(self, model_dir: Path, calibration_path: Path | None = None):
        self.urdf_path = Path(model_dir) / URDF_NAME
        if not self.urdf_path.exists():
            raise FileNotFoundError(f"{self.urdf_path} not found; see README for where the model comes from")
        self.robot = placo.RobotWrapper(str(self.urdf_path))
        self._q_index = {name: self.robot.get_joint_offset(name) for name in ALL_JOINTS}

        # Joint limits, radians, as (low, high). URDF first, then narrowed by calibration.
        self.limits: dict[str, tuple[float, float]] = {}
        for name in ALL_JOINTS:
            low, high = self.robot.get_joint_limits(name)
            self.limits[name] = (float(low), float(high))
        self.calibration_path = None
        if calibration_path is not None:
            self.calibration_path = Path(calibration_path)
            calibration = json.loads(self.calibration_path.read_text(encoding="utf-8"))
            for name, half_deg in calibration_half_travel_deg(calibration).items():
                if name == GRIPPER or name not in self.limits:
                    continue  # the gripper is not solved; its limits stay as the URDF says
                half = math.radians(half_deg)
                low, high = self.limits[name]
                self.limits[name] = (max(low, -half), min(high, half))
        for name, (low, high) in self.limits.items():
            self.robot.set_joint_limits(name, low, high)

        # Reference orientation: the tool frame at the all-zero pose. compose/decompose
        # are written relative to it so that roll = 0 means wrist_roll = 0.
        self.R_ref = self.fk(np.zeros(6))[:3, :3].copy()
        tool_axis = self.R_ref[:, 2]
        if np.linalg.norm(tool_axis - np.array([1.0, 0.0, 0.0])) > TOOL_AXIS_TOLERANCE:
            raise RuntimeError(
                f"tool z axis at the zero pose is {tool_axis}, expected +x; "
                f"this is not the {URDF_NAME} this module was written for"
            )

        # Tool yaw is exactly a multiple (+1 or -1) of the shoulder_pan angle: the pan axis
        # is vertical and nothing downstream can turn the tool about the vertical axis.
        # Reading yaw from the pan joint instead of from the tool axis keeps it well defined
        # when the tool points straight down (where the tool axis has no yaw of its own).
        probe = np.zeros(6)
        probe[0] = YAW_PROBE_RAD
        yaw_at_probe = self.decompose(self.fk(probe)[:3, :3])[0]
        self.pan_to_yaw = 1.0 if yaw_at_probe > 0 else -1.0
        if abs(abs(yaw_at_probe) - YAW_PROBE_RAD) > TOOL_AXIS_TOLERANCE:
            raise RuntimeError(f"tool yaw {yaw_at_probe} at pan {YAW_PROBE_RAD}: pan axis is not vertical?")

    # ---- joint vectors ---------------------------------------------------------------

    def q_from_dict(self, angles: dict[str, float]) -> np.ndarray:
        """{joint: radians (gripper: percent)} -> q."""
        return np.array([angles[name] for name in ALL_JOINTS], dtype=float)

    def q_to_dict(self, q: np.ndarray) -> dict[str, float]:
        return {name: float(q[i]) for i, name in enumerate(ALL_JOINTS)}

    def clamp(self, q: np.ndarray) -> np.ndarray:
        out = np.array(q, dtype=float)
        for i, name in enumerate(ARM_JOINTS):
            low, high = self.limits[name]
            out[i] = min(max(out[i], low), high)
        out[GRIPPER_INDEX] = min(max(out[GRIPPER_INDEX], 0.0), 100.0)
        return out

    def gripper_angle(self, pct: float) -> float:
        """Gripper percent -> URDF joint angle, for drawing only (0% = URDF low, 100% = high)."""
        low, high = self.limits[GRIPPER]
        return low + (high - low) * min(max(pct, 0.0), 100.0) / 100.0

    # ---- forward kinematics ------------------------------------------------------------

    def _apply(self, q: np.ndarray) -> None:
        for i, name in enumerate(ARM_JOINTS):
            self.robot.set_joint(name, float(q[i]))
        self.robot.set_joint(GRIPPER, self.gripper_angle(float(q[GRIPPER_INDEX])))
        self.robot.update_kinematics()

    def fk(self, q: np.ndarray) -> np.ndarray:
        """Tool frame in the base frame, 4x4, for joint vector q."""
        self._apply(q)
        return np.array(self.robot.get_T_world_frame(TOOL_FRAME))

    def joint_frames(self, q: np.ndarray) -> dict[str, np.ndarray]:
        """Each joint's frame (= its child link frame) in the base frame, 4x4."""
        self._apply(q)
        return {name: np.array(self.robot.get_T_world_frame(link)) for name, link in JOINT_CHILD_LINK.items()}

    def state_q(self) -> np.ndarray:
        """The joint vector placo currently holds (after the solver moved it)."""
        q = np.zeros(6)
        for i, name in enumerate(ARM_JOINTS):
            q[i] = self.robot.get_joint(name)
        q[GRIPPER_INDEX] = 0.0  # gripper is masked in the solver; caller overwrites
        return q

    # ---- orientation <-> (yaw, pitch, roll) ---------------------------------------------

    def compose(self, yaw: float, pitch: float, roll: float) -> np.ndarray:
        """R = Rz(yaw) . Ry(pitch) . R_ref . Rz(-roll).

        Rz(yaw) and Ry(pitch) are world-axis rotations: they are exactly what shoulder_pan
        and the three parallel pitch joints do to the zero-pose tool frame. Rz(-roll) is a
        rotation about the tool's own z axis, which is what wrist_roll does; the sign is
        chosen so that a positive roll is a positive wrist_roll joint angle (the tool frame
        in the URDF is flipped 180 degrees about y relative to gripper_link).
        """
        return rot_z(yaw) @ rot_y(pitch) @ self.R_ref @ rot_z(-roll)

    def decompose(self, R: np.ndarray, yaw: float | None = None) -> tuple[float, float, float]:
        """Inverse of `compose`: (yaw, pitch, roll) from a rotation matrix.

        Pass `yaw` whenever it is known (from the pan joint, see `tool_yaw`). Without it,
        yaw is read off the tool axis, which is fine except when the tool points straight
        up or down: there the axis has no yaw of its own and the answer is arbitrary.
        """
        d = R[:, 2]  # tool z axis = Rz(yaw) Ry(pitch) [1,0,0] = [cos y cos p, sin y cos p, -sin p]
        if yaw is None:
            yaw = math.atan2(d[1], d[0])
        d_local = rot_z(-yaw) @ d  # back into the arm's own plane: [cos p, ~0, -sin p]
        pitch = math.atan2(-d_local[2], d_local[0])
        M = (rot_z(yaw) @ rot_y(pitch) @ self.R_ref).T @ R  # what is left must be Rz(-roll)
        roll = -math.atan2(M[1, 0], M[0, 0])
        return yaw, pitch, roll

    # ---- Target <-> pose ---------------------------------------------------------------

    def target_from_q(self, q: np.ndarray) -> Target:
        T = self.fk(q)
        _, pitch, roll = self.decompose(T[:3, :3], yaw=self.tool_yaw(q))
        return Target(xyz=tuple(float(v) for v in T[:3, 3]), pitch=pitch, roll=roll,
                      gripper_pct=float(q[GRIPPER_INDEX]))

    def pose_from_target(self, target: Target, yaw: float) -> np.ndarray:
        """4x4 target pose. `yaw` is the arm's current tool yaw (it is not the user's to choose)."""
        T = np.eye(4)
        T[:3, :3] = self.compose(yaw, target.pitch, target.roll)
        T[:3, 3] = target.xyz
        return T

    def tool_yaw(self, q: np.ndarray) -> float:
        """Yaw of the tool for joint vector q, straight from the pan joint (see __init__)."""
        return self.pan_to_yaw * float(q[0])

    def current_yaw(self) -> float:
        """Same, but from the joint state placo currently holds (used inside the solver loop)."""
        return self.pan_to_yaw * float(self.robot.get_joint(ARM_JOINTS[0]))
