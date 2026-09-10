#!/usr/bin/env python3
"""SO-101 Cartesian control: the pinned model, the solver, and the settings both programs share.

`prepare` downloads and verifies the pinned model; `preview` solves one Cartesian move and
prints what it found. Neither mode opens a serial port.

⚠️ The solver here is placo's own `KinematicsSolver`, held by this course, and NOT LeRobot's
`RobotKinematics`. That is a deliberate change made 2026-09-10, after the keyboard and IK modes
were found unable to move the arm at all while the leader mode worked. Three findings, each
measured rather than reasoned:

  1. LeRobot 0.6.1's `RobotKinematics.inverse_kinematics` takes a single Newton step and never
     iterates, so a target 10 cm away comes back as joint angles whose gripper is still 149.7 mm
     short of it. It also omits `update_kinematics()` before solving, so it is not a function of
     its arguments: the same inputs gave four different answers in four consecutive calls.
     Upstream fixed only the first of those, after 0.6.1 shipped, and 0.6.1 is still the newest
     release on PyPI.
  2. Feeding those steps `FK(measured) + one small step` as the target asks the solver to travel
     that small step and no further, so the commanded pose sits about one degree ahead of the
     arm for ever, however long a key is held.
  3. ⛔ One degree cannot move this arm. The STS3215 has no torque interface: Feetech's register
     table defines `Present_Load` as "voltage duty cycle of the drive", so position error is the
     only thing that makes force. LeRobot writes P=16 where Feetech ships 32, and upstream issue
     #3400 measures what follows: 63 ticks (5.5 deg) of steady-state error at P=16, 26.7 ticks
     (2.3 deg) at P=32. A degree of lead asks for a fraction of the force needed to hold the arm
     up, never mind move it.

So the arm is driven the way the one project that does this successfully drives it -- Viser
handle to a real SO-101, legalaspro/robokin, Apache-2.0: a velocity-limited differential IK step
every control tick, against the pose the operator actually wants, not one step ahead of where the
arm already is.
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

import numpy as np

# The five positioning joints, in the order LeRobot reports them. The gripper is a 0-100
# opening rather than an angle, so it is outside the kinematic chain but inside the motor list.
JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
MOTORS = (*JOINTS, "gripper")
LEROBOT_VERSION = "0.6.1"
MODEL_COMMIT = "7629d2ad9853d10fb903093a33ef6114099d97e5"
MODEL_BASE = f"https://raw.githubusercontent.com/TheRobotStudio/SO-ARM100/{MODEL_COMMIT}"
URDF_NAME = "so101_new_calib.urdf"
URDF_SHA256 = "3a65d2d35e68a8d2f0c2cc176d19b884506543c93ba72980145b80abe276022c"
EE_FRAME = "gripper_frame_link"
# A model-only illustration pose, never a commanded hardware start position.
PREVIEW_JOINTS_DEG = (0.0, -30.0, 60.0, -30.0, 0.0)
DOWNLOAD_TIMEOUT_SECONDS = 45
DOWNLOAD_ATTEMPTS = 3


# ── The two numbers that decide how the arm behaves ──────────────────────────────────────────
# ⭐ Read this before changing either of them. They look similar and they do opposite jobs.
#
# CONTROL_HZ is the rate of the one loop that owns the serial port. 50 Hz is what robokin's
# real-arm demo runs and sits inside the 50-100 Hz band the community reports for this arm.
CONTROL_HZ = 50
CONTROL_DT = 1 / CONTROL_HZ
#
# MAX_JOINT_SPEED_RAD_S caps how far the solver may move a joint in one tick, so it decides HOW
# HARD the arm pushes: with the command re-seeded from the measurement every tick, the gap
# between what a motor is told and where it is can never exceed MAX_JOINT_SPEED_RAD_S * CONTROL_DT
# -- and on this servo that gap IS the force. Measured 2026-09-10 against the pinned model: a
# stalled arm settles at exactly that gap and stays there, so it is a bound, not a trend.
# ⛔ It must be larger than the joint's own steady-state error, or the arm cannot even hold
# itself up, let alone move: at Feetech's P=32 that error is about 2.3 deg, and at LeRobot's
# default P=16 about 5.5 deg (upstream issue #3400). 3.5 rad/s gives 4.0 deg at 50 Hz, which
# clears 2.3 with room and is why SERVO_P_COEFFICIENT below is not left at LeRobot's default.
# ⚠️ Every arm is different. `so101_visual_control.py --measure-following` measures this one's
# error joint by joint; if it reports more than this allows, raise this rather than guess.
MAX_JOINT_SPEED_RAD_S = 3.5
#
# REF_LINEAR_SPEED_MPS and REF_ANGULAR_SPEED_RADPS cap how fast the target the solver chases
# walks towards the handle, so they decide HOW FAST the arm travels -- and nothing else. Keeping
# the two jobs in two numbers is what lets this be slow and still able to push: measured at
# 0.05 m/s the gripper moves 1 mm per tick with only 0.5-0.7 deg of lead, and the lead only
# grows to its 4.0 deg bound when the arm is actually held back.
REF_LINEAR_SPEED_MPS = 0.05
REF_ANGULAR_SPEED_RADPS = 0.5
#
# How far the chased target may get ahead of the gripper before it stops walking. Without it the
# target runs away from an arm that is merely slow; with it, an unreachable target settles
# quietly instead of the solver hunting (measured over 11 targets: 19 direction changes with it,
# 1290 without).
# ⚠️ It is also the second limit on force, and the one that binds first. The command can only
# lead the arm by as much joint travel as this much Cartesian distance asks for, and how much
# that is depends on where the arm is standing. Measured 2026-09-10 against a stalled arm:
#   lead   folded   preview   reaching out        (the joint budget is 4.01 deg)
#   20 mm   3.30      2.65        1.72
#   60 mm   4.01      4.01        4.01
# ⛔ At 20 mm a reaching-out arm is commanded 1.72 deg, which is below the 2.3 deg of position
# error this servo needs at P=32 -- the arm would simply not move, which is the exact fault this
# rewrite exists to remove, hiding in one pose instead of all of them. At 60 mm the joint budget
# is what binds in every pose, so MAX_JOINT_SPEED_RAD_S means what it says.
MAX_REF_LEAD_MM = 60.0
# The gripper is this close to a reference before it counts as arrived and the target stops.
ARRIVED_MM = 1.0
# The gripper has to make at least this much progress towards the reference over STALL_FRAMES,
# or it is not following. One tick of travel is 1 mm at the speeds above, so this is a twentieth
# of what a healthy arm does in that time.
STALL_PROGRESS_MM = 1.0
# ⛔ Not a jump limit: a whole second of a held-back arm. Long enough that a slow start, a heavy
# reach or a moment of stiction is not called a fault, short enough to end well before the two
# seconds at 80 % load that trip the servo's own overload protection (Feetech register table,
# addresses 35 and 36) -- which is the state the 2026-09-08 incident ended in.
STALL_FRAMES = CONTROL_HZ
# A single tick that moves a joint further than this is the solver changing branch, not the step
# anyone asked for. Position-only IK on a 5-DOF arm has more than one answer, and measured over
# 10069 frames of dragging inside the workspace 91 % of ticks stay under 3 deg while a branch
# flip runs 60 to 220. 8 deg is above every healthy tick seen and far below every flip.
# ⚠️ Must stay above MAX_JOINT_SPEED_RAD_S * CONTROL_DT, or healthy ticks trip it.
SOLVER_BRANCH_DEG = 8.0

# ── Solver weights ───────────────────────────────────────────────────────────────────────────
# The SO-101 places its gripper with five joints, so it cannot reach an arbitrary orientation,
# and a solver told to insist on one gives up position to chase it. This course's handle sets a
# position and says nothing about orientation, so the orientation task is switched off entirely
# rather than merely weighted down -- which is also what LeRobot's own step documents the zero
# for ("set to 0.0 to only constrain position").
# ⚠️ Measured 2026-09-10, asking for (0.20, 0.10, 0.15) m while holding the starting
# orientation: the gripper stops 119 mm short at weight 0.1 (robokin's number, for a handle that
# does carry an orientation), 36 mm short at 0.01, and lands exactly on it at 0.0. ⛔ Do not
# copy a weight across from a project whose handle has a rotation ring on it.
POSITION_WEIGHT = 1.0
ORIENTATION_WEIGHT = 0.0
# Penalises large joint velocities. With the orientation task off and wrist_roll masked, four
# joints are solving three degrees of freedom, and the spare one is free to drift: measured
# 2026-09-10, wrist_flex spends about a fifth of a second at the joint speed limit in the middle
# of an otherwise gentle move, tracking nothing. Damping that costs a little time and buys a
# much quieter arm -- 1e-2 takes the worst tick from 4.01 to 2.42 deg for 6 % more travel time,
# while 5e-2 damps so hard that two of the six test targets are never reached at all.
# ⚠️ robokin runs 1e-4 here; its pose task is full six-axis, so it has no spare joint to damp.
REGULARIZATION = 1e-2

# ── Servo registers ──────────────────────────────────────────────────────────────────────────
# Feetech ships P=32; LeRobot writes 16 on every connect. Upstream knows (issue #3400, open since
# 2026-04-17) and left the default alone "to avoid changing overall stiffness without broader
# hardware validation" (PR #3555), so it is left to us to ask for the factory value.
# ⚠️ LeRobot writes one P to all six motors, and P sits in EPROM where writing needs torque off,
# so a per-joint P cannot be applied after connect() without dropping the arm. One value for the
# arm it is, then: the shoulder needs the gain, and this course does not drive wrist_roll hard
# enough to meet the twitchiness that made issue #1333 lower it.
SERVO_P_COEFFICIENT = 32
# Torque_Limit lives in SRAM, so unlike P it can be set once the arm is live. LeRobot caps the
# gripper at 50 % and leaves the other five motors at 100 %; a teaching arm has no reason to
# have its full stall torque available to a program.
# ⚠️ Percent of stall torque, and the arm has to hold itself up out of this budget.
ARM_TORQUE_LIMIT_PCT = 60
# The gripper's scale is not a choice of ours: SOFollower gives that motor MotorNormMode
# RANGE_0_100, so 0 and 100 are the two ends of whatever range Lesson 6 recorded.
GRIPPER_MIN_PCT, GRIPPER_MAX_PCT = 0.0, 100.0
# How much of its range the gripper closes or opens per tick while a key is held: 1 % a tick is
# a full sweep in two seconds at 50 Hz.
GRIPPER_STEP_PCT = 1.0

# A teaching workspace, not a measured safety guarantee. Sampling the pinned model over its URDF
# joint limits gives a reachable box of x [-0.34, 0.48], y [-0.44, 0.44], z [-0.22, 0.53] metres;
# these bounds sit inside it and above the mounting plane. z=0 is the base mounting plane, which
# is the table, so the gripper cannot honestly be asked to go below it.
# ⛔ Widening these does not make an unreachable target safe. Override only with the arm watched.
DEFAULT_BOUNDS_M = {"min": (0.00, -0.22, 0.00), "max": (0.38, 0.22, 0.42)}

# A joint this close to either end of its own recorded travel counts as sitting on the stop.
STOP_MARGIN_DEG = 3.0
# How much further than the margin the operator is asked to move, so the arm does not come back
# resting exactly on the boundary.
STOP_CLEARANCE_DEG = 3.0
# Same idea for the model's own limits, which are a different fact about a different thing.
MODEL_CLEARANCE_DEG = 3.0
# A goal further than this from where the motor actually is, at the moment torque comes on, is a
# fault: the servo is being asked to travel rather than to hold.
GOAL_MARGIN_DEG = 5.0
# How far outside the parked pose the workspace box is opened when it has to be widened.
WORKSPACE_MARGIN_M = 0.02
# The servo this course uses. LeRobot names it in SOFollower's own motor table; the encoder
# resolution that goes with it is looked up rather than written down (see degrees_per_step).
MOTOR_MODEL = "sts3215"


class FollowingLost(Exception):
    """The gripper stopped making progress towards the pose it was being asked to reach.

    ⛔ Whoever raises this does not release torque, and whoever catches it does not either: an
    arm that is being held up drops when it is let go. This only ends the control loop.
    """


# ══ The pinned model ═════════════════════════════════════════════════════════════════════════

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


def validate_joints(joints, limits):
    if len(joints) != len(JOINTS):
        raise ValueError("Expected five arm joints, in the documented order")
    outside = joints_outside_the_model(dict(zip(JOINTS, joints)), limits)
    if outside:
        raise ValueError(outside[0])


def joints_outside_the_model(degrees, limits, clearance=MODEL_CLEARANCE_DEG):
    """Joints the pinned model cannot represent, named, with how far back each has to come.

    ⚠️ Not the same thing as `joints_on_a_stop`, and the two are not interchangeable: that one
    compares against the travel this arm recorded for itself in Lesson 6, this one against what
    the URDF can express. An arm can sit well inside its own travel and still be outside the
    model -- the follower measured 2026-09-08 rests at shoulder_lift -103.8 against the model's
    +-100, wrist_flex -100.2 against +-95.

    ⛔ It matters because the solver clamps its answer into the model's range. Replaying that
    parked pose through a solve moves four joints by up to 5.8 deg before anyone has asked for
    anything, which is why the solving modes refuse to start from there instead of reporting it.
    """
    complaints = []
    for name in JOINTS:
        if name not in degrees or name not in limits:
            continue
        value = float(degrees[name])
        low, high = limits[name]
        if math.isfinite(value) and low <= value <= high:
            continue
        if not math.isfinite(value):
            complaints.append(f"{name} did not read a finite angle.")
            continue
        back = abs(value - (high if value > high else low)) + clearance
        complaints.append(
            f"{name} reads {value:+.1f} deg, outside the pinned model's {low:.1f}..{high:.1f}.\n"
            f"        Move it at least {back:.1f} deg towards 0 with motor power off: the solver "
            "clamps its answer into that range, so the first solved tick would move the arm there "
            "whether or not anything was asked for.")
    return complaints


def bounds_dict(low, high):
    """A workspace box, validated."""
    import numpy as np
    low, high = np.asarray(low, dtype=float), np.asarray(high, dtype=float)
    if low.shape != (3,) or high.shape != (3,) or not np.isfinite(low).all() or not np.isfinite(high).all():
        raise ValueError("Workspace bounds must be three finite metres each")
    if not (low < high).all():
        raise ValueError("Every workspace minimum must be below its maximum")
    return {"min": low, "max": high}


# ══ The solver ═══════════════════════════════════════════════════════════════════════════════

class PlacoServo:
    """One velocity-limited differential IK step per control tick.

    ⭐ This is not "solve the IK for a pose". It is a servo: each tick it is given where the arm
    actually is and where the gripper is wanted, and it returns the next command, at most one
    tick's worth of joint travel away. Chain enough ticks together and the arm walks there.

    ⛔ Do not iterate it to convergence to "get a better answer". Convergence is what
    `solve_pose` is for, and it belongs to the model-only preview, never to the loop that is
    driving a motor: a converged answer is a whole journey collapsed into one command, which is
    the shape of the 2026-09-08 incident.

    Angles cross this boundary as `{motor_name: degrees}` dictionaries in both directions, on
    purpose. placo orders its joints by the URDF's topology and the bus orders them by motor id,
    the two happen to agree today, and nothing would report it if they stopped: a positional
    vector handed to the wrong order does not fail, it just moves the wrong joints.
    """

    def __init__(self, urdf_path, dt=CONTROL_DT, max_joint_speed=MAX_JOINT_SPEED_RAD_S,
                 ee_frame=EE_FRAME, position_weight=POSITION_WEIGHT,
                 orientation_weight=ORIENTATION_WEIGHT, regularization=REGULARIZATION):
        import numpy as np
        import placo

        if not max_joint_speed > 0 or not dt > 0:
            raise ValueError("The tick length and the joint speed limit must both be positive")
        self.dt = float(dt)
        self.max_joint_speed = float(max_joint_speed)
        self.robot = placo.RobotWrapper(str(urdf_path))
        self.joint_names = tuple(self.robot.joint_names())
        if set(self.joint_names) != set(MOTORS):
            raise ValueError(f"The pinned model actuates {self.joint_names}, not {MOTORS}")

        self.limits = {name: tuple(math.degrees(v) for v in self.robot.get_joint_limits(name))
                       for name in self.joint_names}
        self.solver = placo.KinematicsSolver(self.robot)
        self.solver.mask_fbase(True)
        self.solver.dt = self.dt
        self.task = self.solver.add_frame_task(ee_frame, np.eye(4))
        self.task.configure("ee", "soft", float(position_weight), float(orientation_weight))
        self.solver.add_regularization_task(float(regularization))
        # ⛔ The URDF's own velocity limits are placeholders -- every joint declares 10 rad/s and
        # an effort of 10 -- which at 50 Hz would let a single tick move a joint 11.5 deg. The
        # limit is the force budget, so it is set here and not inherited.
        self.robot.set_velocity_limits(self.max_joint_speed)
        self.solver.enable_velocity_limits(True)
        self.solver.enable_joint_limits(True)
        # The gripper is an opening between 0 and 100, not a way to place the gripper frame.
        self.solver.mask_dof("gripper")
        if orientation_weight == 0.0:
            # ⛔ Position-only IK leaves wrist_roll with nothing to do: rolling the gripper
            # barely moves the frame it is measured at, so the solver is free to walk that joint
            # towards a limit while the position error stays zero. Taking it out of the problem
            # is placo's own way of saying it does not participate.
            self.solver.mask_dof("wrist_roll")

    @property
    def max_joint_step_deg(self):
        """The most one tick can move a joint, which is also the most it can lead the arm by."""
        return math.degrees(self.max_joint_speed * self.dt)

    def _load(self, degrees):
        for name in self.joint_names:
            if name in degrees:
                self.robot.set_joint(name, math.radians(float(degrees[name])))
        self.robot.update_kinematics()

    def _unload(self):
        return {name: math.degrees(self.robot.get_joint(name)) for name in self.joint_names}

    def inside_the_model(self, degrees):
        """The same pose with every joint pulled into the range the solver can represent."""
        pulled = dict(degrees)
        for name, (low, high) in self.limits.items():
            if name in pulled:
                pulled[name] = min(max(float(pulled[name]), low), high)
        return pulled

    def fk(self, degrees):
        """The gripper frame, as a 4x4, for a pose given in degrees per motor name."""
        import numpy as np
        self._load(degrees)
        return np.array(self.robot.get_T_world_frame(EE_FRAME), dtype=float)

    def gripper_xyz(self, degrees):
        return self.fk(degrees)[:3, 3]

    def servo_step(self, measured_deg, target_pose):
        """The next command: where the arm is, plus at most one tick towards where it is wanted.

        ⭐ Seeded from the measurement every tick on purpose. That is safe here because the
        target is an absolute pose the operator chose, so nothing accumulates -- and it is what
        keeps the command honest about where the arm really is. A target built by adding a small
        step to the measurement is the case where this would be wrong, and the programs that do
        that integrate the target from their own last command instead.
        """
        import numpy as np
        pose = np.asarray(target_pose, dtype=float)
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            raise ValueError("A target pose is a finite 4x4 transform")
        # ⛔ Seeded inside the model's own limits, always. The solver's joint limits are hard
        # constraints, so a seed outside them makes the problem infeasible and placo raises
        # `QPError` -- and an unpowered SO-101 rests outside them (measured 2026-09-08:
        # shoulder_lift -103.8 against the model's +-100). Refusing to enable from such a pose
        # is what normally keeps this from happening, but backlash and a hand on the arm can
        # put a joint back outside it while a mode is live, and a solver that throws mid-move
        # would leave the arm held with no loop running.
        # ⚠️ Clamping the seed is a bounded lie about where the arm is, and it is confined to
        # this one call: `fk` never clamps, because reporting where the arm really is is its job.
        self._load(self.inside_the_model(measured_deg))
        self.task.T_world_frame = pose
        self.solver.solve(True)
        self.robot.update_kinematics()
        commanded = self._unload()
        commanded["gripper"] = float(measured_deg.get("gripper", commanded["gripper"]))
        return commanded

    def solve_pose(self, seed_deg, target_pose, iterations=100):
        """Iterate to a converged answer. ⛔ Model-only: never call this while a motor is live."""
        import numpy as np
        pose = np.asarray(target_pose, dtype=float)
        self._load(seed_deg)
        self.task.T_world_frame = pose
        for _ in range(int(iterations)):
            self.solver.solve(True)
            self.robot.update_kinematics()
        return self._unload()


def load_servo(directory, **kwargs):
    """The servo on the pinned model, with the model's own joint limits alongside it."""
    path = Path(directory) / URDF_NAME
    limits = model_limits(load_description(path))
    return PlacoServo(path, **kwargs), limits


# ══ The chased pose ══════════════════════════════════════════════════════════════════════════

def clamp_into_bounds(pose, bounds):
    """The same pose with its position pulled inside the workspace box."""
    import numpy as np
    clamped = np.array(pose, dtype=float, copy=True)
    low = np.asarray(bounds["min"], dtype=float)
    high = np.asarray(bounds["max"], dtype=float)
    clamped[:3, 3] = np.clip(clamped[:3, 3], low, high)
    return clamped


def advance_reference(reference, handle, gripper_xyz, dt=CONTROL_DT,
                      linear_speed=REF_LINEAR_SPEED_MPS, angular_speed=REF_ANGULAR_SPEED_RADPS,
                      max_lead_m=MAX_REF_LEAD_MM / 1000):
    """Walk the chased pose one tick towards the handle, without letting it run off.

    ⭐ This is the whole of the speed setting. The handle can be thrown anywhere at any moment;
    what the solver chases moves at `linear_speed`, so that is how fast the arm travels.

    ⛔ And it stops walking once it is `max_lead_m` ahead of the gripper. Without that, an arm
    that is merely slow gets left behind by a target that keeps going, and the distance between
    the two stops meaning anything -- which would take the stall check below with it.
    """
    import numpy as np
    from scipy.spatial.transform import Rotation

    walked = np.array(reference, dtype=float, copy=True)
    wanted = np.asarray(handle, dtype=float)
    here = np.asarray(gripper_xyz, dtype=float)

    towards = wanted[:3, 3] - walked[:3, 3]
    distance = float(np.linalg.norm(towards))
    if distance > 1e-12:
        step = min(linear_speed * dt, distance)
        lead_now = float(np.linalg.norm(walked[:3, 3] - here))
        # Never widen a lead that is already at its limit; shorten the step instead of refusing
        # it, so a reference that is ahead on one axis can still turn towards another.
        step = min(step, max(0.0, max_lead_m - lead_now))
        walked[:3, 3] = walked[:3, 3] + towards / distance * step

    now = Rotation.from_matrix(walked[:3, :3])
    turn = (Rotation.from_matrix(wanted[:3, :3]) * now.inv()).as_rotvec()
    angle = float(np.linalg.norm(turn))
    if angle > 1e-12:
        walked[:3, :3] = (Rotation.from_rotvec(turn / angle * min(angular_speed * dt, angle))
                          * now).as_matrix()
    return walked


def pose_distance_mm(pose, xyz):
    import numpy as np
    return float(np.linalg.norm(np.asarray(pose, dtype=float)[:3, 3] - np.asarray(xyz, dtype=float)) * 1000)


class StallWatch:
    """Is the gripper still getting closer to what it is being asked to reach?

    ⭐ It has to be asked this way round. The obvious check -- how far a motor's command is from
    its position -- cannot work here: the command is re-seeded from the measurement every tick
    and capped at one tick of travel, so that gap is bounded by construction and reads the same
    whether the arm is moving freely or held in a vice (measured 2026-09-10: 3.44 deg in both).
    Progress towards the target is the thing that actually differs.
    """

    def __init__(self, frames=STALL_FRAMES, progress_mm=STALL_PROGRESS_MM, arrived_mm=ARRIVED_MM):
        self.frames = int(frames)
        self.progress_mm = float(progress_mm)
        self.arrived_mm = float(arrived_mm)
        self.reset()

    def reset(self):
        self._best_mm = None
        self._waited = 0

    def update(self, gripper_xyz, handle):
        """Returns a complaint once the gripper has stopped closing on where it is wanted.

        ⚠️ Measured against the handle the operator set, ⛔ never against the pose the solver is
        chasing this tick. That pose walks along a fixed distance in front of the gripper, so
        while the arm is travelling normally the gap to it barely changes -- ask about progress
        towards *that* and a healthy arm reports itself stuck after a second (measured
        2026-09-10, and it is why this takes the handle instead).
        """
        remaining = pose_distance_mm(handle, gripper_xyz)
        if remaining <= self.arrived_mm:
            self.reset()
            return None
        if self._best_mm is None or remaining <= self._best_mm - self.progress_mm:
            self._best_mm, self._waited = remaining, 0
            return None
        self._waited += 1
        if self._waited < self.frames:
            return None
        return (f"the gripper stopped closing on its target: {remaining:.0f} mm away and no "
                f"progress for {self._waited / CONTROL_HZ:.1f} s. Something is holding the arm, "
                "or the target cannot be reached from here.")


def branch_flip(previous_deg, candidate_deg, margin=SOLVER_BRANCH_DEG):
    """Joints a single tick moved further than any step could have asked for.

    Position-only IK on a five-joint arm has more than one answer, and a solver is free to jump
    between them. ⛔ The offending tick is dropped whole rather than scaled down: a shrunken
    branch flip walks the arm along a path nobody asked for, which is worse than not moving.
    """
    return [f"{name} {float(candidate_deg[name]) - float(previous_deg[name]):+.0f} deg"
            for name in JOINTS
            if name in previous_deg and name in candidate_deg
            and abs(float(candidate_deg[name]) - float(previous_deg[name])) > margin]


# ══ The arm ══════════════════════════════════════════════════════════════════════════════════

def read_pose_before_power(robot):
    """Where the arm is, read over its own bus without enabling a single motor.

    `Robot.connect()` ends in `configure()`, which leaves torque ON -- so any check made after
    connecting is made too late to prevent what it is checking for. `bus.connect()` only opens
    the port and handshakes, so the arm can be read first and the port handed back untouched.

    Returns the pose, the motors that already had torque, and the calibration each motor is
    actually holding, so all of it can be judged before anything is energised.
    """
    robot.bus.connect()
    try:
        reading = robot.bus.sync_read("Present_Position", num_retry=robot.config.num_read_retries)
        powered = [name for name, value in
                   robot.bus.sync_read("Torque_Enable", normalize=False).items() if value]
        stored = {register: robot.bus.sync_read(register, normalize=False)
                  for register in ("Homing_Offset", "Min_Position_Limit", "Max_Position_Limit")}
        in_the_motors = {name: (stored["Homing_Offset"][name], stored["Min_Position_Limit"][name],
                                stored["Max_Position_Limit"][name]) for name in reading}
        return ({f"{motor}.pos": float(value) for motor, value in reading.items()},
                powered, in_the_motors)
    finally:
        robot.bus.disconnect(disable_torque=False)


def limit_arm_torque(robot, percent=ARM_TORQUE_LIMIT_PCT):
    """Cap what the five arm motors are allowed to pull, once the arm is live.

    LeRobot caps the gripper at half its stall torque and leaves the other five at all of it.
    `Torque_Limit` is a SRAM register, so unlike the P coefficient it can be set after connect
    without taking torque off and dropping the arm. Feetech's scale is per mille of stall torque.

    ⚠️ The arm holds itself up out of this budget, so it cannot go very low. Returns what was
    written so the caller can say it out loud rather than change the arm's behaviour quietly.
    """
    if not 0 < percent <= 100:
        raise ValueError("A torque limit is a percentage of stall torque, above zero")
    value = int(round(percent * 10))
    for name in JOINTS:
        robot.bus.write("Torque_Limit", name, value, normalize=False)
    return {name: percent for name in JOINTS}


# Registers worth reading back before trusting anything written down about them.
# ⛔ LeRobot's connect() rewrites Operating_Mode, the three PID coefficients, Acceleration and
# Maximum_Acceleration on every single connection, and says so nowhere. Anything reasoned from
# the factory defaults, or from what this file asked for, is a guess until it has been read.
WATCHED_REGISTERS = ("P_Coefficient", "I_Coefficient", "D_Coefficient", "Torque_Limit",
                     "Max_Torque_Limit", "Acceleration", "Goal_Velocity", "CW_Dead_Zone",
                     "CCW_Dead_Zone", "Minimum_Startup_Force", "Overload_Torque",
                     "Protection_Time", "Operating_Mode")


def dump_servo_registers(robot, registers=WATCHED_REGISTERS):
    """What the motors actually hold right now, register by register and motor by motor."""
    dump = {}
    for register in registers:
        try:
            dump[register] = {name: int(value) for name, value
                              in robot.bus.sync_read(register, normalize=False).items()}
        except Exception as exc:                      # noqa: BLE001 - reported, never swallowed
            dump[register] = f"could not be read ({exc})"
    return dump


def joint_degrees(observation):
    """The `{motor}.pos` observation LeRobot returns, as the `{motor: degrees}` the servo takes."""
    return {name: float(observation[f"{name}.pos"]) for name in MOTORS
            if f"{name}.pos" in observation}


def as_observation(degrees):
    """The other direction: what `robot.send_action` expects."""
    return {f"{name}.pos": float(value) for name, value in degrees.items()}


def recorded_range(entry):
    """(range_min, range_max) from a MotorCalibration or from the same fields read out of JSON."""
    if hasattr(entry, "range_min"):
        return float(entry.range_min), float(entry.range_max)
    return float(entry["range_min"]), float(entry["range_max"])


def degrees_per_step(model_name=MOTOR_MODEL):
    """How many degrees one encoder step is, taken from LeRobot rather than written down here.

    `MotorsBus._normalize` converts a DEGREES joint with `(raw - mid) * 360 / (resolution - 1)`,
    and the resolution comes from the model table. ⛔ Copying the figure into this file would
    make a second source of truth that can stop matching the library without anyone noticing --
    and every stop distance measured below depends on it.
    """
    from lerobot.motors.feetech.tables import MODEL_RESOLUTION
    return 360 / (MODEL_RESOLUTION[model_name] - 1)


def travel_degrees(calibration, per_step=None):
    """Half of each joint's own recorded travel, in degrees: the distance from zero to a stop.

    LeRobot reports a reading in degrees around the midpoint of the range recorded in Lesson 6,
    so the reachable span is symmetric about zero and half of it is the distance to either end.

    ⚠️ This is not the URDF's joint limit, and the two are not interchangeable. On the follower
    measured 2026-09-09 the recorded travel is wider than the model on every joint (wrist_flex
    ±104.75 against the model's ±95). The stop is a physical fact about this arm; the model
    limit is what the solver can represent. Both are checked, separately.
    """
    per_step = degrees_per_step() if per_step is None else per_step
    travel = {}
    for name, entry in calibration.items():
        low, high = recorded_range(entry)
        travel[name] = (high - low) / 2 * per_step
    return travel


def calibration_disagrees(in_the_motors, calibration):
    """Motors whose stored calibration is not the one in the file, named.

    ⚠️ This says *that* they disagree, not why. Twelve disagreements usually mean the two serial
    ports are the other way round rather than a broken calibration, and the Lesson 7 register
    check is what tells the two apart -- so the message points there instead of guessing.
    """
    disagreeing = []
    for name, entry in calibration.items():
        if name not in in_the_motors:
            continue
        low, high = recorded_range(entry)
        offset = float(getattr(entry, "homing_offset", None)
                       if hasattr(entry, "homing_offset") else entry["homing_offset"])
        if tuple(float(v) for v in in_the_motors[name]) != (offset, low, high):
            disagreeing.append(name)
    return disagreeing


def joints_on_a_stop(observation, calibration, margin=STOP_MARGIN_DEG):
    """Joints resting against an end of their own travel, each with how far to move it back.

    A joint on its stop has nowhere to go in one direction, and the first command out of there is
    the hardest one the arm will ever be asked for. Measured 2026-09-08: a follower parked 0.8 deg
    from its shoulder stop drove at 100 % load 2.5 deg deeper into that stop when torque came on,
    and the servo's overload protection cut it to 20 % two seconds later.
    """
    half = travel_degrees(calibration)
    complaints = []
    for name in JOINTS:
        if name not in half or f"{name}.pos" not in observation:
            continue
        value = float(observation[f"{name}.pos"])
        end = -half[name] if value < 0 else half[name]
        room = abs(end - value)
        if room < margin:
            move = margin + STOP_CLEARANCE_DEG - room
            complaints.append(
                f"{name} is {room:.1f} deg from the end of its travel "
                f"({value:+.1f} of {end:+.1f}).\n"
                f"        Move it at least {move:.1f} deg towards 0, with motor power off.")
    return complaints


def gripper_position(observation, servo):
    """Where the gripper is now, from the joints in the observation."""
    return servo.gripper_xyz(joint_degrees(observation))


def outside_the_workspace(here, bounds):
    """Whether the gripper starts outside the box, which is clamped rather than refused."""
    import numpy as np
    low, high = np.asarray(bounds["min"], dtype=float), np.asarray(bounds["max"], dtype=float)
    outside = [axis for index, axis in enumerate("xyz") if not low[index] <= here[index] <= high[index]]
    if not outside:
        return []
    return [f"the gripper is at x={here[0]:.3f} y={here[1]:.3f} z={here[2]:.3f} m, outside the "
            f"workspace on {', '.join(outside)}.\n"
            f"        Workspace min {tuple(round(float(v), 3) for v in low)} "
            f"max {tuple(round(float(v), 3) for v in high)}. Move the arm inside it with motor "
            "power off, or pass --bounds-min-m and --bounds-max-m for a box that contains it."]


def bounds_including(bounds, point, margin=WORKSPACE_MARGIN_M):
    """The workspace box, widened if it does not already contain the pose the arm starts in.

    A target outside the box is clamped to the nearest face, not refused, so an arm parked
    outside would be walked to the edge before anyone touched a control. The arm rests wherever
    gravity leaves it, so the box accommodates that rather than the other way round. The box
    actually in use is printed, so a widened one is never a silent one.
    """
    import numpy as np
    low = np.minimum(np.asarray(bounds["min"], dtype=float), np.asarray(point, dtype=float) - margin)
    high = np.maximum(np.asarray(bounds["max"], dtype=float), np.asarray(point, dtype=float) + margin)
    return {"min": low, "max": high}


def park_the_goal(robot, observation):
    """Write where each motor is as where it is told to go.

    Before torque, this is the whole of the power-on protection: a servo drives towards
    Goal_Position the instant torque comes on, and LeRobot's connect() never writes one, so after
    a DC power cycle every motor reads a goal of 0 -- one end of the travel. Measured 2026-09-08:
    torque enabled with a stale goal drove two joints at 100 % load 2.5 deg into their stops, and
    the servos' overload protection cut them to 20 % two seconds later.

    ⭐ It is also how a live arm is told to stop pushing: writing its present position as its
    goal takes the force out of a motor that is leaning on something, while leaving it held.
    ⛔ That is not the same as releasing torque, and it is not a substitute for it: an arm that
    is holding itself up drops the moment it is let go, so letting go stays a decision a person
    makes with a hand on the arm.
    """
    goal = {name: float(observation[f"{name}.pos"]) for name in MOTORS
            if f"{name}.pos" in observation}
    robot.bus.sync_write("Goal_Position", goal)
    return goal


def goal_diverged(robot, margin=GOAL_MARGIN_DEG):
    """Motors whose goal is not where they are, read back once torque is on.

    The 2026-09-08 incident happened with the goal believed equal to the present position, so
    this reads the servos' own registers rather than trusting the write.
    """
    goal = robot.bus.sync_read("Goal_Position")
    present = robot.bus.sync_read("Present_Position")
    return [f"{name}: goal {float(goal[name]):+.1f} against present {float(present[name]):+.1f}"
            for name in goal if abs(float(goal[name]) - float(present[name])) > margin]


# ══ The modes, and the two ends of a control loop ════════════════════════════════════════════
# ⭐ The loop below is written once and used by both programs in this folder: the browser page
# and the console one. ⛔ Never copy it into either of them -- a control loop kept in two places
# is a control loop that is right in one place.
#
# A "page" is anything that answers these: take_arm_request, take_end_request, handle_xyz,
# gripper_target, set_gripper_target, nudge_gripper, move_handle, armed, disarmed, show.
HANDLE, KEYBOARD, LEADER = "handle", "keyboard", "leader"
MODE_LABELS = {HANDLE: "1 - Drag the handle", KEYBOARD: "2 - Arrow keys",
               LEADER: "3 - Leader arm"}


def bounds_low(bounds):
    import numpy as np
    return np.asarray(bounds["min"], dtype=float)


def bounds_high(bounds):
    import numpy as np
    return np.asarray(bounds["max"], dtype=float)


# ══ The arm, in the two forms this program drives ════════════════════════════════════════════
# ⭐ Both answer the same four questions, so the control loop below is written once and cannot
# drift between "the version in the lesson" and "the version that runs on the bench".

class ModelArm:
    """The arm as the model says it behaves: it goes exactly where it is told, immediately.

    ⛔ Nothing here is evidence about hardware. It exists so the solver, the handle, the speed
    limits and the whole page can be seen and taught without an arm attached -- and so the
    control loop can be tested without one.
    """

    is_live = False

    def __init__(self, degrees):
        self._degrees = dict(degrees)
        self.holding = False

    def read(self):
        return dict(self._degrees)

    def send(self, degrees):
        self._degrees.update({name: float(value) for name, value in degrees.items()})

    def hold(self, degrees):
        self.holding = True

    def stop_pushing(self, degrees):
        """Nothing to stop pushing against. Present so the loop is written once, not twice."""

    def release(self):
        self.holding = False


class LiveArm:
    """The follower on the bench, and the few registers this course sets on it.

    ⚠️ `position_p_coefficient` is passed on purpose. LeRobot writes 16 to every motor on every
    connect where Feetech ships 32, and upstream issue #3400 measures what that costs: 5.5 deg
    of steady-state error instead of 2.3. The arm is asked to hold itself up out of that error,
    so at 16 it cannot -- the commanded lead this program allows is 4.0 deg.
    """

    is_live = True

    def __init__(self, port, robot_id, calibration_dir, p_coefficient=SERVO_P_COEFFICIENT,
                 torque_limit_pct=ARM_TORQUE_LIMIT_PCT):
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
        self.torque_limit_pct = torque_limit_pct
        self.robot = SO101Follower(SO101FollowerConfig(
            port=port, id=robot_id, calibration_dir=Path(calibration_dir),
            use_degrees=True, cameras={},
            # ⭐ A crash must not drop a raised arm. Ending a mode releases the motors
            # explicitly; every other way out deliberately leaves them held.
            disable_torque_on_disconnect=False,
            position_p_coefficient=p_coefficient,
        ))
        self.holding = False

    # -- before power ------------------------------------------------------------------------
    def read_before_power(self):
        return read_pose_before_power(self.robot)

    def open_bus(self):
        self.robot.bus.connect()

    def close_bus(self):
        if self.robot.bus.is_connected:
            self.robot.bus.disconnect(disable_torque=False)

    # -- the four the control loop uses ------------------------------------------------------
    def read(self):
        reading = self.robot.bus.sync_read("Present_Position",
                                           num_retry=self.robot.config.num_read_retries)
        return {name: float(value) for name, value in reading.items()}

    def send(self, degrees):
        self.robot.send_action(as_observation(degrees))

    def hold(self, degrees):
        """Park the goal where the arm already is, then bring torque on. ⛔ In that order."""
        # A servo drives at its Goal_Position the instant torque comes on, and connect() never
        # writes one, so the goal is set while the motors are still free to be moved by hand.
        park_the_goal(self.robot, as_observation(degrees))
        self.close_bus()
        self.robot.connect()          # the one place torque comes on
        self.holding = True
        drifted = goal_diverged(self.robot)
        if drifted:
            raise RuntimeError("Torque came on with motors being told to travel, not to hold:"
                               "\n    " + "\n    ".join(drifted))
        limit_arm_torque(self.robot, self.torque_limit_pct)

    def stop_pushing(self, degrees):
        """Take the force out of a motor leaning on something, ⛔ without letting go of it."""
        park_the_goal(self.robot, as_observation(degrees))

    def release(self):
        self.robot.bus.disable_torque()
        self.holding = False
        self.robot.disconnect()
        self.robot.bus.connect()
# ══ The keyboard ═════════════════════════════════════════════════════════════════════════════

GRIPPER_CLOSE, GRIPPER_HOLD, GRIPPER_OPEN = 0, 1, 2


def keyboard_device(robot_id):
    """LeRobot's own keyboard device, with one upstream bug held off, and nothing else changed.

    ⚠️ `_drain_pressed_keys` records a released key as `current_pressed[key] = False` instead of
    dropping it, and `get_action` then walks that dictionary assigning one axis at a time. So a
    key released a moment ago still gets its turn, and writes its zero over the key that is
    still held: hold Up, tap and release Down, and Up stops working. Deleting the entry is
    exactly what upstream's own unmerged PR #3947 does, done here from the outside because a
    course cannot ask its students to patch their site-packages.
    """
    from lerobot.teleoperators.keyboard import (KeyboardEndEffectorTeleop,
                                                KeyboardEndEffectorTeleopConfig)

    class SteadyKeyboard(KeyboardEndEffectorTeleop):
        def _drain_pressed_keys(self):
            super()._drain_pressed_keys()
            for key in [key for key, down in list(self.current_pressed.items()) if not down]:
                del self.current_pressed[key]

    return SteadyKeyboard(KeyboardEndEffectorTeleopConfig(id=robot_id, use_gripper=True))


def gripper_from_keyboard(action):
    """⚠️ A released left ctrl reports -1, which is not one of the three commands there are."""
    command = action.get("gripper", GRIPPER_HOLD)
    return command if command in (GRIPPER_CLOSE, GRIPPER_HOLD, GRIPPER_OPEN) else GRIPPER_HOLD


def keys_held(action):
    axes = [name for name, value in (("x", action.get("delta_x", 0)),
                                     ("y", action.get("delta_y", 0)),
                                     ("z", action.get("delta_z", 0))) if value]
    jaw = {GRIPPER_CLOSE: "close", GRIPPER_OPEN: "open"}.get(gripper_from_keyboard(action))
    return ", ".join(axes + ([jaw] if jaw else [])) or ""


def handle_after_keys(xyz, action, bounds, step_m):
    """Where the arrow keys have pushed the handle to, kept inside the workspace box."""
    moved = np.asarray(xyz, dtype=float) + step_m * np.array([
        float(action.get("delta_x", 0)), float(action.get("delta_y", 0)),
        float(action.get("delta_z", 0))], dtype=float)
    return np.clip(moved, bounds_low(bounds), bounds_high(bounds))


# ══ The control loop ═════════════════════════════════════════════════════════════════════════

def refusal_to_arm(mode, measured, limits):
    """Why this mode cannot be enabled from where the arm is standing, or nothing.

    ⛔ Only the solving modes are refused. The leader mode has no solver in its path, and an
    unpowered SO-101 falls onto its shoulder stop and rests outside the model -- refusing that
    would refuse the one mode that works from where the arm actually parks.
    """
    if mode == LEADER:
        return ""
    outside = joints_outside_the_model(measured, limits)
    if not outside:
        return ""
    return ("Not enabled, and nothing was powered:\n\n" + "\n\n".join(outside))


def control_loop(page, arm, servo, bounds, limits, leader=None, keyboard=None,
                 hz=CONTROL_HZ):
    """The one loop. It owns the serial port, and it is the only thing that moves the arm.

    ⛔ Nothing else in this program reads or writes the bus, and no part of it runs on a Viser
    callback thread. The page is asked what the operator wants; the arm is asked where it is;
    one command goes out; repeat.
    """
    period = 1 / hz
    step_m = REF_LINEAR_SPEED_MPS * period
    watch = StallWatch()
    armed, reference = None, None
    commanded = None

    while True:
        started = time.perf_counter()
        measured = arm.read()
        if commanded is None:
            commanded = dict(measured)
        note, keys = "", ""
        here = servo.gripper_xyz(measured)

        if armed is None:
            # -- read-only. The arm can be moved by hand and the page follows it. ------------
            commanded = dict(measured)
            page.move_handle(here)
            wanted = page.take_arm_request()
            if wanted is not None:
                note = refusal_to_arm(wanted, measured, limits)
                if note:
                    print("Not enabled:\n    " + note.replace("\n\n", "\n    "), flush=True)
                else:
                    arm.hold(measured)
                    armed, reference = wanted, servo.fk(measured)
                    commanded = dict(measured)
                    # ⚠️ The jaw does not jump when a mode starts: the slider is moved to where
                    # the gripper already is, not the other way round.
                    page.set_gripper_target(measured.get("gripper", GRIPPER_MIN_PCT))
                    watch.reset()
                    page.armed(armed)
                    print(f"{MODE_LABELS[armed]} is live. Press End on the page to stop and "
                          "release.", flush=True)
            status = ("Nothing is powered. Move the arm by hand if you like, then pick a mode "
                      "and press Enable." if arm.is_live else
                      "No arm attached. Drag the handle to see what the solver does with it.")

        elif page.take_end_request():
            # -- the one place torque is released ------------------------------------------
            ended = armed
            arm.release()
            armed, reference = None, None
            commanded = dict(measured)
            page.disarmed(f"{MODE_LABELS[ended]} ended. Motors released; the arm can be moved "
                          "by hand again.")
            print(f"{MODE_LABELS[ended]} ended, motors released.\n", flush=True)
            status = "Released."

        elif armed == LEADER:
            # -- joint for joint, no solver in the path ------------------------------------
            # ⛔ No speed limit and no stall check on this one: the hand on the leader is both.
            # It is Lesson 7's chain, and it has to keep working from wherever the arm parks.
            commanded = joint_degrees(leader.get_action())
            arm.send(commanded)
            status = f"**{MODE_LABELS[LEADER]}** live. Move the leader by hand."

        else:
            # -- handle and keyboard share every line below ---------------------------------
            if armed == KEYBOARD:
                pressed = dict(keyboard.get_action())
                keys = keys_held(pressed)
                page.move_handle(handle_after_keys(page.handle_xyz(), pressed, bounds, step_m))
                jaw = gripper_from_keyboard(pressed)
                if jaw != GRIPPER_HOLD:
                    page.nudge_gripper(GRIPPER_STEP_PCT
                                       * (1 if jaw == GRIPPER_OPEN else -1))

            handle = servo.fk(commanded)
            handle[:3, 3] = np.clip(page.handle_xyz(), bounds_low(bounds), bounds_high(bounds))
            reference = advance_reference(reference, handle, here)

            stalled = watch.update(here, handle)
            if stalled:
                arm.stop_pushing(measured)
                raise FollowingLost(stalled)

            commanded = servo.servo_step(measured, reference)
            commanded["gripper"] = step_towards(commanded.get("gripper", 0.0),
                                                page.gripper_target(), GRIPPER_STEP_PCT)
            arm.send(commanded)
            remaining = pose_distance_mm(handle, here)
            status = (f"**{MODE_LABELS[armed]}** live - `{remaining:6.1f}` mm to the handle, "
                      f"commanding at most `{servo.max_joint_step_deg:.1f}` deg a tick.")

        page.show(measured, commanded, status, note, keys)
        left = period - (time.perf_counter() - started)
        if left > 0:
            time.sleep(left)


def step_towards(now, wanted, step):
    """One tick of the gripper's travel towards where the slider is."""
    gap = float(wanted) - float(now)
    return float(now) + math.copysign(min(abs(gap), step), gap)


# ══ The model-only preview ═══════════════════════════════════════════════════════════════════

def preview(args):
    """One Cartesian move on the model alone: the converged answer, and the walk to it.

    ⭐ Both halves are the point of the lesson. `solved_degrees` is the answer to "what angles
    put the gripper there"; `ticks_to_arrive` is what actually happens on the arm, because the
    program never sends the answer -- it sends one tick of travel at a time, recomputed from
    where the arm really is.
    """
    import numpy as np
    servo, limits = load_servo(args.model_dir)
    seed = dict(zip(JOINTS, (float(v) for v in args.joints_deg)))
    validate_joints([seed[name] for name in JOINTS], limits)
    seed["gripper"] = GRIPPER_MIN_PCT

    start_pose = servo.fk(seed)
    target_pose = start_pose.copy()
    target_pose[:3, 3] = start_pose[:3, 3] + np.asarray(args.delta_mm, dtype=float) / 1000

    solved = servo.solve_pose(seed, target_pose)
    reached = servo.gripper_xyz(solved)

    walking, reference, ticks = dict(seed), start_pose.copy(), 0
    while ticks < args.max_ticks:
        here = servo.gripper_xyz(walking)
        if pose_distance_mm(target_pose, here) <= ARRIVED_MM:
            break
        reference = advance_reference(reference, target_pose, here)
        walking = servo.servo_step(walking, reference)
        ticks += 1

    print(json.dumps({
        "mode": "model-only; no hardware",
        "solver": f"placo KinematicsSolver, position {POSITION_WEIGHT} / orientation "
                  f"{ORIENTATION_WEIGHT} soft, {MAX_JOINT_SPEED_RAD_S} rad/s per joint",
        "joint_names": JOINTS,
        "start_degrees": [seed[name] for name in JOINTS],
        "solved_degrees": [solved[name] for name in JOINTS],
        "start_xyz_m": start_pose[:3, 3].tolist(),
        "target_xyz_m": target_pose[:3, 3].tolist(),
        "reached_xyz_m": reached.tolist(),
        "position_error_mm": float(np.linalg.norm(reached - target_pose[:3, 3]) * 1000),
        "ticks_to_arrive": ticks,
        "seconds_to_arrive": round(ticks * CONTROL_DT, 2),
        "max_joint_step_deg": round(servo.max_joint_step_deg, 2),
    }, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    for mode in ("prepare", "preview"):
        sub = commands.add_parser(mode)
        sub.add_argument("--model-dir", required=True)
        if mode == "preview":
            sub.add_argument("--joints-deg", nargs=5, type=float, default=PREVIEW_JOINTS_DEG)
            sub.add_argument("--delta-mm", nargs=3, type=float, default=(0.0, 0.0, 20.0))
            sub.add_argument("--max-ticks", type=int, default=10 * CONTROL_HZ)
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
