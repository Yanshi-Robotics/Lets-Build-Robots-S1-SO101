#!/usr/bin/env python3
"""One browser page for the SO-101, with three ways to drive it, all of them LeRobot's.

    keyboard   LeRobot's KeyboardEndEffectorTeleop; its deltas go through LeRobot's five
               end-effector steps, the last of which is the IK solver
    leader     LeRobot's SO101Leader, joint for joint, no IK at all -- Lesson 7's chain
    IK         drag the handle, Plan to see what the solver makes of it, Execute to move

How a session goes:

    1. a self-check runs first. If anything fails, nothing is started and nothing is powered:
       the failures are printed with what to do about each, and the program exits.
    2. the page opens showing the arm exactly where it is. Nothing is powered, so the arm can
       still be moved by hand, and the page follows it.
    3. pick a mode and press Enable. Only then are the motors powered and held, and only one
       mode is live at a time.
    4. press End to finish: the mode stops, torque is released, and the page returns to
       read-only, ready for another mode.

⛔ If the program stops any other way -- a crash, or Ctrl+C -- torque is deliberately NOT
released, because letting go of a raised arm drops it. The program says so on its way out, and
the release command is `--release-torque`.

The page is a teleoperator: it subclasses LeRobot's `Teleoperator` and reports what a leader arm
reports, one target per motor, so the loop around it is LeRobot's own `teleop_loop` with the
identity processors `lerobot-teleoperate` itself uses. Nothing here plans a trajectory,
integrates a position, or solves anything of its own.
"""
from __future__ import annotations

import argparse
import math
import socket
import sys
import threading
import time
from dataclasses import dataclass, field
from importlib.metadata import version
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import so101_cartesian_demo as model  # noqa: E402  (same folder, as the lesson downloads it)

VISER_VERSION = "1.1.0"
LOOPBACK = "127.0.0.1"  # Hardware control must never bind a LAN or public interface.
DEFAULT_WEB_PORT = 4602
DEFAULT_FPS = 30

KEYBOARD, LEADER, IK = "keyboard", "leader", "ik"
MODE_LABELS = {KEYBOARD: "1 - Keyboard", LEADER: "2 - Leader arm", IK: "3 - Inverse kinematics"}
# Whether starting this mode has to hold the arm. All three drive the follower, so all three do;
# a mode that only watched would set this False and the rest of the flow would be unchanged.
MODE_HOLDS_THE_ARM = {KEYBOARD: True, LEADER: True, IK: True}

# The handle and the planned target are rate controls, not position commands: however far away
# they are, one frame asks for at most one step.
MAX_UNITS_PER_FRAME = 1.0
# Below this a request counts as noise, matching MapDeltaActionToRobotActionStep's own 1e-3
# threshold on the same units.
NOISE_UNITS = 1e-3
# Execution stops when the gripper is this close to the planned target.
ARRIVED_MM = 1.0
GRIPPER_CLOSE, GRIPPER_HOLD, GRIPPER_OPEN = 0, 1, 2
# Nearer than this to the wanted figure counts as arrived: one frame moves the gripper by about
# one percent, so a tighter tolerance would leave the command oscillating for ever.
GRIPPER_TOLERANCE_PCT = 1.0
GRIPPER_INITIAL_PCT = 50.0

ARM_COLOR = (0.18, 0.48, 0.72, 1.0)
TARGET_COLOR = (1.0, 0.61, 0.12, 0.35)
GRID_SIZE_M = 1.2
GRID_CELL_M = 0.05
HANDLE_SCALE = 0.12

SAFETY_NOTICE = (
    "SAFETY: the program stopped without a mode being ended, so torque was NOT released and the\n"
    "        arm is still held. Releasing a raised arm drops it, which is why nothing let go.\n"
    "        When the arm is supported, release it with:\n"
    "            python IK/so101_visual_control.py --release-torque --model-dir {model_dir} \\\n"
    "                --port {port} --robot-id {robot_id} --calibration-dir {calibration_dir}\n"
    "        Cutting DC power does the same thing instantly."
)


class ModeEnded(Exception):
    """Raised from `get_action()` when the operator presses End.

    LeRobot's teleop_loop runs "until a set duration is reached or it is manually interrupted"
    and catches nothing, so this is how a teleoperator says it is finished. ⛔ Do not replace it
    with a control loop of our own.
    """

    def __init__(self, mode):
        super().__init__(f"{mode} mode ended by the operator")
        self.mode = mode


@dataclass
class ViserTeleopConfig:
    """What Teleoperator's constructor reads, plus this page's own settings."""
    id: str
    model_dir: str
    web_port: int = DEFAULT_WEB_PORT
    step_mm: float = model.STEP_MM
    calibration_dir: Path | None = None
    bounds: dict = field(default_factory=lambda: dict(model.DEFAULT_BOUNDS_M))


def validate_web_port(port):
    if not 1024 <= port <= 65535:
        raise ValueError("Choose an explicitly assigned unprivileged web port")
    with socket.socket() as probe:
        # A port left in TIME_WAIT by the previous run must not block a restart; a live
        # listener on it still fails the bind and is reported.
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind((LOOPBACK, port))


def steps_towards(target_xyz, gripper_xyz, step_m):
    """How many steps this frame should ask for, per axis, to get from here to there.

    A pure function so the rate limit is testable without a browser: the arm is never asked for
    more than one step per frame, however far away the target is.
    """
    if step_m <= 0 or not math.isfinite(step_m):
        raise ValueError("Step size must be positive")
    delta = (np.asarray(target_xyz, dtype=float) - np.asarray(gripper_xyz, dtype=float)) / step_m
    if not np.isfinite(delta).all():
        return np.zeros(3)
    delta = np.clip(delta, -MAX_UNITS_PER_FRAME, MAX_UNITS_PER_FRAME)
    return np.where(np.abs(delta) < NOISE_UNITS, 0.0, delta)


def gripper_command(reported_pct, wanted_pct):
    """Close, hold or open: the three-way command LeRobot's own keyboard device sends.

    ⚠️ Direction comes from LeRobot, not from us. GripperVelocityToJoint maps command 0 (close)
    to a positive step and command 2 (open) to a negative one, because "joint position increases
    on close" -- so a larger reported figure is a more closed gripper.
    """
    if reported_pct is None or not math.isfinite(float(wanted_pct)):
        return GRIPPER_HOLD
    error = float(wanted_pct) - float(reported_pct)
    if abs(error) <= GRIPPER_TOLERANCE_PCT:
        return GRIPPER_HOLD
    return GRIPPER_CLOSE if error > 0 else GRIPPER_OPEN


def gripper_urdf_range(model_dir):
    """The gripper joint's travel in the pinned URDF, read rather than written down.

    ⚠️ Which end is open is not established here and is not claimed anywhere until it has been
    watched on hardware; this only maps LeRobot's 0-100 onto the model's travel so the picture
    moves with the real gripper.
    """
    joint = model.load_description(Path(model_dir) / model.URDF_NAME).find("./joint[@name='gripper']/limit")
    if joint is None:
        raise ValueError("The pinned model has no gripper joint limit")
    return float(joint.attrib["lower"]), float(joint.attrib["upper"])


def held_keys_text(pressed):
    """Which keys LeRobot's keyboard device currently believes are down.

    Shown because a keyboard that reaches nothing looks exactly like a program that does nothing:
    pynput captures keys only on an X11 session, never on Wayland or over a bare ssh connection.
    """
    held = sorted(str(key).replace("Key.", "") for key, down in pressed.items() if down)
    return f"held: `{'` `'.join(held)}`" if held else "no key held"


def following_text(leader_action, observation):
    """Leader reading against follower reading, joint by joint."""
    rows = []
    for name in model.MOTORS:
        leader = leader_action.get(f"{name}.pos")
        follower = observation.get(f"{name}.pos")
        if leader is None or follower is None:
            continue
        rows.append(f"`{name:<14} L {float(leader):+7.1f}  F {float(follower):+7.1f}"
                    f"  diff {float(leader) - float(follower):+6.1f}`")
    return "\n\n".join(rows) if rows else "No leader reading."


def viewer_configuration(names, joints_deg, gripper_pct, gripper_range):
    """Pose a browser model by URDF joint name, gripper included."""
    by_name = dict(zip(model.JOINTS, (math.radians(float(v)) for v in joints_deg)))
    low, high = gripper_range
    by_name["gripper"] = low + (high - low) * float(np.clip(float(gripper_pct), 0.0, 100.0)) / 100
    if set(names) != set(by_name):
        raise ValueError("Unexpected actuated joints in the pinned visual model")
    return [by_name[name] for name in names]


class ViserControlPage:
    """A page that reports one target per motor, exactly as a leader arm does.

    `build` below mixes this with LeRobot's Teleoperator base class, so importing this module
    needs no lerobot. The page runs in two states. Read-only: no motor is powered, the caller
    paints it from the bus, and the operator can move the arm by hand. Armed: one mode is live
    and LeRobot's teleop_loop is asking this object for a frame at a time.
    """

    name = "viser_so101"

    def __init__(self, config: ViserTeleopConfig, kinematics, pipeline, keyboard=None, leader=None):
        self.config = config
        self.id = config.id
        self.kinematics = kinematics
        self.pipeline = pipeline
        self.keyboard = keyboard
        self.leader = leader
        self.step_m = config.step_mm / 1000
        self.gripper_range = None  # read from the URDF in connect(); the constructor touches no file
        self._lock = threading.Lock()
        self._observation = None
        self._panel = KEYBOARD if keyboard is not None else IK
        self._armed = None            # which mode teleop_loop is currently serving
        self._arm_request = None      # set by an Enable button, taken by the read-only loop
        self._end_requested = False   # set by an End button, taken by get_action
        self._plan = None             # (target_xyz, solved_joints, residual_mm)
        self._executing = False
        self._connected = False
        self.server = None

    # -- Teleoperator interface ----------------------------------------------------------
    @property
    def action_features(self) -> dict:
        # The same shape SO101Leader reports, so the identity processors carry it unchanged.
        return {f"{motor}.pos": float for motor in model.MOTORS}

    @property
    def feedback_features(self) -> dict:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        """Nothing to calibrate: a page has no encoders."""

    def configure(self) -> None:
        """Nothing to configure: no registers behind this device."""

    def send_feedback(self, feedback) -> None:
        """No force feedback in a browser."""

    def get_action(self) -> dict:
        """One frame's action, always one target per motor. Only called while a mode is armed."""
        with self._lock:
            if self._end_requested:
                self._end_requested = False
                raise ModeEnded(self._armed)
            armed = self._armed
            observation = dict(self._observation) if self._observation else None
        if observation is None or armed is None:
            return {}
        if armed == LEADER:
            # No IK at all: the leader reports joint positions and they go straight through.
            return self.leader.get_action()
        delta = self.keyboard.get_action() if armed == KEYBOARD else self._ik_delta(observation)
        return self.pipeline((delta, observation))

    # -- Arming ---------------------------------------------------------------------------
    def take_arm_request(self):
        """The mode an Enable button asked for, cleared as it is taken. None while read-only."""
        with self._lock:
            wanted, self._arm_request = self._arm_request, None
        return wanted

    def armed(self, mode) -> None:
        """Called once the motors are powered and teleop_loop is about to run this mode."""
        with self._lock:
            self._armed, self._end_requested, self._plan, self._executing = mode, False, None, False
        self.pipeline.reset()
        self._refresh_controls()

    def disarmed(self, note="") -> None:
        """Called once the mode has stopped and the motors have been released."""
        with self._lock:
            self._armed, self._end_requested, self._plan, self._executing = None, False, None, False
        self._refresh_controls()
        if note:
            self.notice.content = note

    # -- Where each armed mode's frame comes from -----------------------------------------
    def _ik_delta(self, observation):
        """Steps towards the planned target while executing, and nothing otherwise."""
        with self._lock:
            plan, executing = self._plan, self._executing
        wanted = gripper_command(observation["gripper.pos"], self.gripper_slider.value)
        still = {"delta_x": 0.0, "delta_y": 0.0, "delta_z": 0.0, "gripper": wanted}
        if not executing or plan is None:
            return still
        here = model.gripper_position(observation, self.kinematics)
        remaining_mm = float(np.linalg.norm(np.asarray(plan[0]) - here) * 1000)
        if remaining_mm <= ARRIVED_MM:
            self._stop_executing(f"Arrived, {remaining_mm:.1f} mm from the target.")
            return still
        delta = steps_towards(plan[0], here, self.step_m)
        return {"delta_x": float(delta[0]), "delta_y": float(delta[1]), "delta_z": float(delta[2]),
                "gripper": wanted}

    # -- The page --------------------------------------------------------------------------
    def connect(self, calibrate: bool = True) -> None:
        import viser
        from viser.extras import ViserUrdf

        if version("viser") != VISER_VERSION:
            raise RuntimeError(f"Use viser[urdf]=={VISER_VERSION}")
        validate_web_port(self.config.web_port)
        self.gripper_range = gripper_urdf_range(self.config.model_dir)
        self.server = viser.ViserServer(host=LOOPBACK, port=self.config.web_port,
                                        label="SO-101 . Software Teleoperation")
        self.server.scene.add_grid("/ground", width=GRID_SIZE_M, height=GRID_SIZE_M,
                                   plane="xy", cell_size=GRID_CELL_M)
        urdf_path = Path(self.config.model_dir) / model.URDF_NAME
        self.arm = ViserUrdf(self.server, urdf_path, root_node_name="/arm",
                             mesh_color_override=ARM_COLOR)
        self.ghost = ViserUrdf(self.server, urdf_path, root_node_name="/planned",
                               mesh_color_override=TARGET_COLOR)
        self.joint_names = self.arm.get_actuated_joint_names()
        self.handle = self.server.scene.add_transform_controls(
            "/target", scale=HANDLE_SCALE, disable_rotations=True,
            translation_limits=tuple(
                (float(lo), float(hi))
                for lo, hi in zip(self.config.bounds["min"], self.config.bounds["max"])
            ),
        )
        self._build_panels()
        self._show_panel(self._panel)
        self._refresh_controls()
        self._connected = True
        # ⛔ No zero-pose window: whatever the caller already read is painted before anyone can
        # look at the page. 2026-09-09: the page used to sit at the URDF's zero configuration,
        # 384 mm from where the arm actually was, until the control loop started.
        if self._observation:
            self.observe(dict(self._observation))
        print(f"Open http://{LOOPBACK}:{self.config.web_port} to drive the arm.")

    def disconnect(self) -> None:
        if self.server is not None:
            self.server.stop()
            self.server = None
        self._connected = False

    def _build_panels(self):
        gui = self.server.gui
        available = [mode for mode in (KEYBOARD, LEADER, IK)
                     if mode != KEYBOARD or self.keyboard is not None]
        available = [mode for mode in available if mode != LEADER or self.leader is not None]
        self.mode_dropdown = gui.add_dropdown("Mode", [MODE_LABELS[m] for m in available],
                                              initial_value=MODE_LABELS[self._panel])
        self.mode_dropdown.on_update(lambda _: self._show_panel(self._chosen_panel()))
        self.status = gui.add_markdown("Reading the arm.")
        self.notice = gui.add_markdown("")

        self.enable_buttons, self.end_buttons, self.panels = {}, {}, {}
        for mode in available:
            with gui.add_folder(MODE_LABELS[mode]) as folder:
                self.panels[mode] = folder
                self.enable_buttons[mode] = gui.add_button("Enable this mode")
                self.end_buttons[mode] = gui.add_button("End this mode")
                self._build_mode_body(gui, mode)
            self.enable_buttons[mode].on_click(
                lambda _event, mode=mode: self._request_arm(mode))
            self.end_buttons[mode].on_click(lambda _event: self._request_end())

    def _build_mode_body(self, gui, mode):
        if mode == KEYBOARD:
            gui.add_markdown(
                "Arrow keys move the gripper in x and y. Shift and right shift move it down and "
                "up. Left and right ctrl close and open the gripper.\n\n"
                "Keys are captured for the whole desktop, so this page does not need focus.")
            self.keyboard_state = gui.add_markdown("no key held")
        elif mode == LEADER:
            self.leader_state = gui.add_markdown(
                "Move the leader by hand once enabled; the follower tracks it, with no IK.")
        else:
            gui.add_markdown("Drag the handle, then Plan to see what the solver makes of it.")
            self.plan_button = gui.add_button("Plan")
            self.execute_button = gui.add_button("Execute")
            self.stop_button = gui.add_button("Stop")
            self.plan_state = gui.add_markdown("No plan yet.")
            self.gripper_slider = gui.add_slider(
                "Gripper (LeRobot 0-100)", min=model.GRIPPER_MIN_PCT, max=model.GRIPPER_MAX_PCT,
                step=1, initial_value=GRIPPER_INITIAL_PCT)
            self.plan_button.on_click(lambda _: self._make_plan())
            self.execute_button.on_click(lambda _: self._start_executing())
            self.stop_button.on_click(lambda _: self._stop_executing("Stopped by the operator."))

    def _chosen_panel(self):
        return next(m for m, label in MODE_LABELS.items() if label == self.mode_dropdown.value)

    def _show_panel(self, mode):
        with self._lock:
            self._panel = mode
        for name, folder in self.panels.items():
            folder.visible = name == mode
        self.handle.visible = mode == IK
        self.ghost.show_visual = mode == IK

    def _refresh_controls(self):
        """Only one mode can be live, so the others cannot be enabled while it is."""
        with self._lock:
            armed = self._armed
        for mode in self.panels:
            self.enable_buttons[mode].disabled = armed is not None
            self.end_buttons[mode].disabled = armed != mode
        self.mode_dropdown.disabled = armed is not None

    def _request_arm(self, mode):
        with self._lock:
            if self._armed is not None:
                return
            self._arm_request = mode
        self.notice.content = f"Enabling {MODE_LABELS[mode]}..."

    def _request_end(self):
        with self._lock:
            if self._armed is None:
                return
            self._end_requested = True

    # -- Inverse kinematics panel ----------------------------------------------------------
    def _make_plan(self):
        """One call to LeRobot's solver for the dragged target. Nothing is sent."""
        with self._lock:
            observation = dict(self._observation) if self._observation else None
        if observation is None:
            self.plan_state.content = "No arm reading yet."
            return
        joints = np.array([float(observation[f"{name}.pos"]) for name in model.JOINTS])
        here = self.kinematics.forward_kinematics(joints)[:3, 3]
        target = np.asarray(self.handle.position, dtype=float)
        pose = self.kinematics.forward_kinematics(joints).copy()
        pose[:3, 3] = target
        solved = self.kinematics.inverse_kinematics(joints, pose, position_weight=1.0,
                                                    orientation_weight=0.0)
        reached = self.kinematics.forward_kinematics(solved)[:3, 3]
        residual_mm = float(np.linalg.norm(reached - target) * 1000)
        with self._lock:
            self._plan = (target, solved, residual_mm)
        self.ghost.update_cfg(np.asarray(viewer_configuration(
            self.joint_names, solved, observation["gripper.pos"], self.gripper_range)))
        angles = "  ".join(f"`{name} {value:+.1f}`" for name, value in zip(model.JOINTS, solved))
        self.plan_state.content = (
            f"target `{target[0]:+.3f} {target[1]:+.3f} {target[2]:+.3f}` m, "
            f"{np.linalg.norm(target - here) * 1000:.0f} mm away\n\n{angles}\n\n"
            f"solver residual `{residual_mm:.3f}` mm - wrist_roll is out of the solve, so it holds"
            "\n\nExecute walks there one step per frame, the same five steps re-solving each frame."
        )

    def _start_executing(self):
        with self._lock:
            if self._armed != IK:
                self.plan_state.content = "Enable this mode first."
                return
            if self._plan is None:
                self.plan_state.content = "Press Plan first."
                return
            self._executing = True
        self.pipeline.reset()

    def _stop_executing(self, why):
        with self._lock:
            self._executing = False
        self.plan_state.content = why

    # -- Painting ---------------------------------------------------------------------------
    def observe(self, observation) -> None:
        """Show where the arm actually is. Called every frame, read-only and armed alike."""
        joints = [float(observation[f"{name}.pos"]) for name in model.JOINTS]
        gripper_pct = float(observation["gripper.pos"])
        here = self.kinematics.forward_kinematics(np.asarray(joints, dtype=float))[:3, 3]
        self.arm.update_cfg(np.asarray(viewer_configuration(
            self.joint_names, joints, gripper_pct, self.gripper_range)))
        with self._lock:
            self._observation = dict(observation)
            armed, panel, executing, plan = self._armed, self._panel, self._executing, self._plan
        if armed != IK:
            self.handle.position = tuple(float(v) for v in here)
        if panel == KEYBOARD and self.keyboard is not None:
            self.keyboard_state.content = held_keys_text(self.keyboard.current_pressed)
        if panel == LEADER and self.leader is not None and armed == LEADER:
            self.leader_state.content = following_text(self.leader.get_action(), observation)
        state = f"**LIVE - {MODE_LABELS[armed]}**" if armed else "**READ-ONLY** - no torque, move the arm by hand"
        self.status.content = (
            f"{state}\n\ngripper `{here[0]:+.3f} {here[1]:+.3f} {here[2]:+.3f}` m"
            f" - reported opening `{gripper_pct:.0f}`"
            + (f" - executing, {float(np.linalg.norm(np.asarray(plan[0]) - here) * 1000):.0f} mm to go"
               if executing and plan is not None else "")
        )

    def seed(self, observation) -> None:
        """The reading taken during the self-check, so the page never shows the model's zero pose."""
        with self._lock:
            self._observation = dict(observation)


def build(config, kinematics, pipeline, keyboard=None, leader=None):
    """Make the page a real LeRobot Teleoperator subclass, without importing lerobot on import."""
    from lerobot.teleoperators import Teleoperator
    subclass = type("ViserControlPage", (ViserControlPage, Teleoperator), {})
    return subclass(config, kinematics, pipeline, keyboard=keyboard, leader=leader)


def display_tap(page):
    """A pass-through pipeline whose only effect is showing the arm on the page.

    LeRobot's teleoperation loop hands the observation it already read to the teleop action
    pipeline on every frame, and that pipeline is a parameter of the loop precisely so it can be
    replaced. Mirroring the arm here costs no extra bus read. The step returns the action
    untouched and computes nothing.
    """
    from lerobot.processor import RobotActionProcessorStep, RobotProcessorPipeline, TransitionKey
    from lerobot.processor.converters import (
        robot_action_observation_to_transition,
        transition_to_robot_action,
    )

    @dataclass
    class ShowArmOnPage(RobotActionProcessorStep):
        def action(self, action):
            observation = self.transition.get(TransitionKey.OBSERVATION)
            if observation:
                page.observe(observation)
            return action

        def transform_features(self, features):
            return features

    return RobotProcessorPipeline(
        steps=[ShowArmOnPage()],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )


# -- The self-check -----------------------------------------------------------------------
LESSON_7_CHECK = ("Run the Lesson 7 register check to see why:\n"
                  "        python Teleop/so101_teleop_log.py registers "
                  "--leader-port <LEADER_PORT> --follower-port <FOLLOWER_PORT>")


class CheckFailed(Exception):
    """One self-check item that has to be fixed before anything is started."""


def report(name, detail=""):
    print(f"  ok      {name}" + (f"  ({detail})" if detail else ""), flush=True)


def self_check(args, robot, leader, kinematics, bounds, keyboard_available):
    """Everything that must be true before a server is started or a motor is powered.

    ⛔ Nothing here energises anything and nothing here opens the page: a page that comes up
    looking healthy while none of its modes can work is worse than no page at all.

    Returns the follower reading, so the page can be painted with the real pose from the start.
    """
    print("Self-check:", flush=True)
    if version("lerobot") != model.LEROBOT_VERSION or version("viser") != VISER_VERSION:
        raise CheckFailed(f"Install lerobot=={model.LEROBOT_VERSION} and viser=={VISER_VERSION}; "
                          f"found {version('lerobot')} and {version('viser')}.")
    report("versions", f"lerobot {model.LEROBOT_VERSION}, viser {VISER_VERSION}")

    report("pinned model", f"{Path(args.model_dir) / model.URDF_NAME} matches its recorded hash")

    try:
        validate_web_port(args.web_port)
    except OSError as exc:
        raise CheckFailed(f"Web port {args.web_port} cannot be bound ({exc}).\n"
                          "        Close whatever is listening, or pass another --web-port.") from exc
    report("web port", str(args.web_port))

    try:
        observation, powered, in_the_motors = model.read_pose_before_power(robot)
    except Exception as exc:
        raise CheckFailed(f"The follower on {args.port} did not answer ({exc}).\n"
                          "        Check the USB cable, the DC supply, and the port name.") from exc
    report("follower bus", f"{len(observation)} motors answered on {args.port}")

    if powered:
        raise CheckFailed(
            f"Torque is already enabled on {', '.join(powered)}.\n"
            "        A previous run left the arm held. Support the arm, then release it with\n"
            "        --release-torque, or cut DC power.")
    report("follower torque", "off on all six motors")

    disagreeing = model.calibration_disagrees(in_the_motors, robot.calibration)
    if disagreeing:
        raise CheckFailed(
            f"The follower's motors hold a different calibration from {args.calibration_dir}: "
            f"{', '.join(disagreeing)}.\n        " + LESSON_7_CHECK)
    report("follower calibration", "every motor matches its file")

    on_a_stop = model.joints_on_a_stop(observation, robot.calibration)
    if on_a_stop:
        raise CheckFailed("Joints are resting against the end of their travel:\n    "
                          + "\n    ".join(on_a_stop))
    report("joint travel", "no joint is against a stop")

    _kinematics, limits = kinematics
    outside = model.joints_outside_the_model(observation, limits)
    if outside:
        raise CheckFailed("The pose is outside the model the solver works in:\n    "
                          + "\n    ".join(outside))
    report("model range", "every joint is inside the pinned model")

    here = model.gripper_position(observation, _kinematics)
    off_the_box = model.outside_the_workspace(here, bounds)
    if off_the_box:
        raise CheckFailed("\n    ".join(off_the_box))
    report("workspace", f"gripper at x={here[0]:.3f} y={here[1]:.3f} z={here[2]:.3f} m")

    if leader is None:
        report("leader arm", "not requested; that mode is left out")
    else:
        try:
            _reading, leader_powered, leader_motors = model.read_pose_before_power(leader)
        except Exception as exc:
            raise CheckFailed(f"The leader on {args.leader_port} did not answer ({exc}).\n"
                              "        Check its cable and port, or drop --leader-port.") from exc
        leader_disagrees = model.calibration_disagrees(leader_motors, leader.calibration)
        if leader_disagrees:
            raise CheckFailed(
                f"The leader's motors hold a different calibration from "
                f"{args.leader_calibration_dir}: {', '.join(leader_disagrees)}.\n        "
                + LESSON_7_CHECK)
        report("leader arm", f"answered on {args.leader_port}, calibration matches"
               + (f", torque on {', '.join(leader_powered)}" if leader_powered else ""))

    if not keyboard_available:
        raise CheckFailed(
            "LeRobot's keyboard device cannot capture keys here.\n"
            "        pynput needs an X11 session: not Wayland, and not a bare ssh connection.\n"
            "        Check `echo $XDG_SESSION_TYPE` prints x11, or drop the keyboard mode.")
    report("keyboard capture", "pynput can read keys in this session")

    print("Self-check passed.\n", flush=True)
    return observation


def read_only_display(page, robot, fps):
    """Paint the live arm with nothing energised, until an Enable button is pressed.

    ⛔ This is a display loop, not a control loop: it reads over the bus and sends nothing at
    all. The only control loop in this program is LeRobot's teleop_loop.
    """
    period = 1 / fps
    while True:
        reading = robot.bus.sync_read("Present_Position", num_retry=robot.config.num_read_retries)
        observation = {f"{motor}.pos": float(value) for motor, value in reading.items()}
        page.observe(observation)
        wanted = page.take_arm_request()
        if wanted is not None:
            return wanted, observation
        time.sleep(period)


def make_follower(args):
    """The official follower configuration, with one field set deliberately and nothing added.

    ⛔ No max_relative_target: on a proportional servo it clamps the goal-to-present gap, which is
    what makes the force, so it caps torque rather than speed (2026-09-08 logs).
    ⭐ disable_torque_on_disconnect=False so that a crash cannot drop a raised arm. Ending a mode
    releases the motors explicitly; every other way out deliberately leaves them held.
    """
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
    return SO101Follower(SO101FollowerConfig(
        port=args.port, id=args.robot_id, calibration_dir=Path(args.calibration_dir),
        use_degrees=True, cameras={}, disable_torque_on_disconnect=False,
    ))


def release_only(args):
    """Let go of every motor and exit: what to run after the program stopped without ending a mode.

    ⚠️ The arm drops if it is not supported. That is why nothing does this automatically.
    """
    robot = make_follower(args)
    robot.bus.connect()
    try:
        before = robot.bus.sync_read("Present_Position")
        powered = [name for name, value in
                   robot.bus.sync_read("Torque_Enable", normalize=False).items() if value]
        if not powered:
            print("Every motor is already released; nothing to do.")
            return 0
        print(f"Releasing {', '.join(powered)}. Support the arm.", flush=True)
        robot.bus.disable_torque()
        after = robot.bus.sync_read("Present_Position")
        for name in model.MOTORS:
            print(f"    {name:<14} {float(after[name]):+8.2f}   moved "
                  f"{float(after[name]) - float(before[name]):+6.2f}")
        return 0
    finally:
        robot.bus.disconnect(disable_torque=False)


def run(args):
    from lerobot.processor import make_default_processors
    from lerobot.scripts.lerobot_teleoperate import teleop_loop
    from lerobot.teleoperators.keyboard import KeyboardEndEffectorTeleop, KeyboardEndEffectorTeleopConfig
    from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig
    from lerobot.utils.keyboard_input import pynput_can_capture

    kinematics, limits = model.load_kinematics(args.model_dir)
    bounds = model.bounds_dict(args.bounds_min_m, args.bounds_max_m)
    pipeline = model.build_robot_action_processor(kinematics, bounds, step_m=args.step_mm / 1000)
    robot = make_follower(args)
    leader = None
    if args.leader_port:
        leader = SO101Leader(SO101LeaderConfig(
            port=args.leader_port, id=args.leader_id,
            calibration_dir=Path(args.leader_calibration_dir), use_degrees=True))

    # ⛔ Before any server and before any motor: a page that opens looking healthy while none of
    # its modes can work is worse than no page.
    observation = self_check(args, robot, leader, (kinematics, limits), bounds, pynput_can_capture())

    keyboard = KeyboardEndEffectorTeleop(
        KeyboardEndEffectorTeleopConfig(id=args.robot_id, use_gripper=True))
    keyboard.connect()
    if leader is not None:
        leader.connect()  # SOLeader.configure() disables torque: it stays free to move by hand

    page = build(ViserTeleopConfig(
        id=args.robot_id, model_dir=args.model_dir, web_port=args.web_port, step_mm=args.step_mm,
        calibration_dir=Path(args.calibration_dir),
        bounds={"min": bounds["min"], "max": bounds["max"]},
    ), kinematics, pipeline, keyboard=keyboard, leader=leader)
    page.seed(observation)
    page.connect()
    print("Nothing is powered. Move the arm by hand if you like; press Enable in a mode to start.\n",
          flush=True)

    _unused, identity_action, identity_observation = make_default_processors()
    robot.bus.connect()
    holding = False
    try:
        while True:
            mode, observation = read_only_display(page, robot, args.fps)
            try:
                model.check_start_pose(observation, kinematics, bounds, limits, robot.calibration)
            except RuntimeError as exc:
                page.disarmed(f"**Cannot enable {MODE_LABELS[mode]}**\n\n```\n{exc}\n```")
                continue
            robot.bus.disconnect(disable_torque=False)
            robot.connect()  # the one place torque comes on
            holding = True
            page.armed(mode)
            print(f"{MODE_LABELS[mode]} is live. Press End on the page to stop and release.",
                  flush=True)
            try:
                teleop_loop(
                    teleop=page, robot=robot, fps=args.fps,
                    teleop_action_processor=display_tap(page),
                    robot_action_processor=identity_action,
                    robot_observation_processor=identity_observation,
                )
            except ModeEnded as ended:
                robot.bus.disable_torque()
                holding = False
                robot.disconnect()
                robot.bus.connect()
                page.disarmed(f"{MODE_LABELS[ended.mode]} ended. Motors released; the arm can be "
                              "moved by hand again.")
                print(f"{MODE_LABELS[ended.mode]} ended, motors released.\n", flush=True)
    except BaseException:
        if holding:
            print("\n" + SAFETY_NOTICE.format(
                model_dir=args.model_dir, port=args.port, robot_id=args.robot_id,
                calibration_dir=args.calibration_dir), file=sys.stderr, flush=True)
        raise
    finally:
        keyboard.disconnect()
        if leader is not None:
            leader.disconnect()
        page.disconnect()
        if robot.bus.is_connected:
            robot.bus.disconnect(disable_torque=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--port", required=True, help="follower serial port")
    parser.add_argument("--robot-id", required=True)
    parser.add_argument("--calibration-dir", required=True)
    parser.add_argument("--release-torque", action="store_true",
                        help="let go of every motor and exit; for after the program stopped "
                             "without a mode being ended. The arm drops if it is not supported.")
    parser.add_argument("--leader-port", help="add the leader-arm mode; omit to leave it out")
    parser.add_argument("--leader-id")
    parser.add_argument("--leader-calibration-dir", default="calibration/leader")
    parser.add_argument("--web-port", type=int, default=DEFAULT_WEB_PORT)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--step-mm", type=float, default=model.STEP_MM,
                        help="gripper travel per frame in the keyboard and IK modes")
    parser.add_argument("--bounds-min-m", nargs=3, type=float, default=model.DEFAULT_BOUNDS_M["min"])
    parser.add_argument("--bounds-max-m", nargs=3, type=float, default=model.DEFAULT_BOUNDS_M["max"])
    args = parser.parse_args(argv)
    if args.fps <= 0 or args.step_mm <= 0:
        parser.error("fps and step-mm must be positive")
    if args.leader_port and not args.leader_id:
        parser.error("--leader-port needs --leader-id, the name its calibration file is under")
    try:
        return release_only(args) if args.release_torque else run(args)
    except CheckFailed as failed:
        print(f"\nSelf-check FAILED, so nothing was started and nothing was powered:\n"
              f"    {failed}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"STOP: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
