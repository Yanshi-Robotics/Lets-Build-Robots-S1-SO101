#!/usr/bin/env python3
"""One browser page for the SO-101, with three ways to drive it, all of them LeRobot's.

The page is a teleoperator. It subclasses LeRobot's `Teleoperator`, the base class its keyboard,
gamepad and leader-arm devices use, and reports what a leader arm reports: one target per motor.
LeRobot's own teleoperation loop drives it, with the identity processors `lerobot-teleoperate`
itself uses, and LeRobot's SO101Follower executes.

    keyboard   LeRobot's KeyboardEndEffectorTeleop; its deltas go through LeRobot's five
               end-effector steps, the last of which is the IK solver
    leader     LeRobot's SO101Leader, joint for joint, no IK at all -- Lesson 7's chain
    IK         drag the handle, press Plan to see what the solver makes of it, press Execute
               to move; the same five steps carry it there

Nothing here plans a trajectory, integrates a position or solves anything of its own. Choosing a
mode picks which of the three fills in one frame's action; everything after that is identical.

Connecting enables motor torque. The follower is configured to KEEP holding when this program
stops, so the arm does not drop at the end of a session: cut DC power to release it. Read
Lesson 7 first. Software bounds are not a stop and not a collision check.
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
# How long to wait for someone to open the page before giving up, rather than powering an arm
# nobody is watching.
BROWSER_WAIT_SECONDS = 120
BROWSER_POLL_SECONDS = 0.2

KEYBOARD, LEADER, IK = "keyboard", "leader", "ik"
MODE_LABELS = {KEYBOARD: "1 · Keyboard", LEADER: "2 · Leader arm", IK: "3 · Inverse kinematics"}

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
# The gripper's URDF travel is a geometry reference; LeRobot reports a 0-100 figure instead.
GRIPPER_URDF_RANGE_RAD = (-0.174533, 1.74533)

ARM_COLOR = (0.18, 0.48, 0.72, 1.0)
TARGET_COLOR = (1.0, 0.61, 0.12, 0.35)
GRID_SIZE_M = 1.2
GRID_CELL_M = 0.05
HANDLE_SCALE = 0.12


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


def viewer_configuration(names, joints_deg, gripper_pct):
    """Pose a browser model by URDF joint name, gripper included."""
    by_name = dict(zip(model.JOINTS, (math.radians(float(v)) for v in joints_deg)))
    low, high = GRIPPER_URDF_RANGE_RAD
    by_name["gripper"] = low + (high - low) * float(np.clip(float(gripper_pct), 0.0, 100.0)) / 100
    if set(names) != set(by_name):
        raise ValueError("Unexpected actuated joints in the pinned visual model")
    return [by_name[name] for name in names]


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


def gripper_xyz_of(observation, kinematics):
    joints = np.array([float(observation[f"{name}.pos"]) for name in model.JOINTS])
    return joints, kinematics.forward_kinematics(joints)[:3, 3]


class ViserControlPage:
    """A page that reports one target per motor, exactly as a leader arm does.

    `build` below mixes this with LeRobot's Teleoperator base class, so importing this module
    needs no lerobot. The arm's own state reaches the page through `observe`, which the display
    tap calls once per frame with the observation the teleoperation loop already read.
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
        self._lock = threading.Lock()
        self._observation = None
        self._mode = KEYBOARD if keyboard is not None else IK
        self._plan = None          # (target_xyz, solved_joints, residual_mm)
        self._executing = False
        self._connected = False
        self.server = None

    # -- Teleoperator interface ----------------------------------------------------------
    @property
    def action_features(self) -> dict:
        # The same shape SO101Leader reports, so the identity processors can carry it unchanged.
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

    def seed(self, observation) -> None:
        """The first observation, read before the loop starts, so frame one has an arm to read."""
        with self._lock:
            self._observation = dict(observation)

    def connect(self, calibrate: bool = True) -> None:
        import viser
        from viser.extras import ViserUrdf

        if version("viser") != VISER_VERSION:
            raise RuntimeError(f"Use viser[urdf]=={VISER_VERSION}")
        validate_web_port(self.config.web_port)
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
        self._show_mode(self._mode)
        self._connected = True
        print(f"Open http://{LOOPBACK}:{self.config.web_port} to drive the arm.")

    def get_action(self) -> dict:
        """One frame's action, always one target per motor. LeRobot's loop calls this at fps."""
        with self._lock:
            observation = dict(self._observation) if self._observation else None
            mode = self._mode
        if observation is None:
            return {}
        if mode == LEADER:
            # No IK at all: the leader reports joint positions and they go straight through.
            return self.leader.get_action()
        delta = self._keyboard_delta() if mode == KEYBOARD else self._ik_delta(observation)
        return self.pipeline((delta, observation))

    def disconnect(self) -> None:
        if self.server is not None:
            self.server.stop()
            self.server = None
        self._connected = False

    # -- Where each mode's frame comes from ----------------------------------------------
    def _keyboard_delta(self):
        """LeRobot's own keyboard device, verbatim. A gamepad reports the same four names."""
        return self.keyboard.get_action()

    def _ik_delta(self, observation):
        """Steps towards the planned target while executing, and nothing otherwise."""
        with self._lock:
            plan, executing = self._plan, self._executing
        wanted = gripper_command(observation["gripper.pos"], self.gripper_slider.value)
        if not executing or plan is None:
            return {"delta_x": 0.0, "delta_y": 0.0, "delta_z": 0.0, "gripper": wanted}
        _joints, here = gripper_xyz_of(observation, self.kinematics)
        remaining_mm = float(np.linalg.norm(np.asarray(plan[0]) - here) * 1000)
        if remaining_mm <= ARRIVED_MM:
            self._stop_executing(f"Arrived, {remaining_mm:.1f} mm from the target.")
            return {"delta_x": 0.0, "delta_y": 0.0, "delta_z": 0.0, "gripper": wanted}
        delta = steps_towards(plan[0], here, self.step_m)
        return {"delta_x": float(delta[0]), "delta_y": float(delta[1]), "delta_z": float(delta[2]),
                "gripper": wanted}

    # -- The page ------------------------------------------------------------------------
    def _build_panels(self):
        gui = self.server.gui
        available = [IK] + ([KEYBOARD] if self.keyboard is not None else []) \
            + ([LEADER] if self.leader is not None else [])
        self.mode_dropdown = gui.add_dropdown(
            "Mode", [MODE_LABELS[m] for m in sorted(available, key=list(MODE_LABELS).index)],
            initial_value=MODE_LABELS[self._mode])
        self.mode_dropdown.on_update(lambda _: self._choose_mode())
        self.status = gui.add_markdown("Waiting for the first arm reading.")

        with gui.add_folder("Keyboard") as self.keyboard_panel:
            gui.add_markdown(
                "Arrow keys move the gripper in x and y. Shift and right shift move it down and "
                "up. Left and right ctrl close and open the gripper.\n\n"
                "Keys are captured for the whole desktop, so this page does not need focus."
            )
            self.keyboard_state = gui.add_markdown("no key held")
        with gui.add_folder("Leader arm") as self.leader_panel:
            self.leader_state = gui.add_markdown("Move the leader by hand; the follower tracks it.")
        with gui.add_folder("Inverse kinematics") as self.ik_panel:
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

    def _choose_mode(self):
        wanted = next(m for m, label in MODE_LABELS.items() if label == self.mode_dropdown.value)
        with self._lock:
            self._mode, self._executing, self._plan = wanted, False, None
        # EEBoundsAndSafety remembers the last commanded position and EEReferenceAndDelta the
        # last command while disabled; neither should carry across a change of input device.
        self.pipeline.reset()
        self._show_mode(wanted)

    def _show_mode(self, mode):
        self.keyboard_panel.visible = mode == KEYBOARD
        self.leader_panel.visible = mode == LEADER
        self.ik_panel.visible = mode == IK
        self.handle.visible = mode == IK
        self.ghost.show_visual = mode == IK

    def _make_plan(self):
        """One call to LeRobot's solver for the dragged target. Nothing is sent."""
        with self._lock:
            observation = dict(self._observation) if self._observation else None
        if observation is None:
            self.plan_state.content = "No arm reading yet."
            return
        joints, here = gripper_xyz_of(observation, self.kinematics)
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
            self.joint_names, solved, observation["gripper.pos"])))
        angles = "  ".join(f"`{name} {value:+.1f}`" for name, value in zip(model.JOINTS, solved))
        self.plan_state.content = (
            f"target `{target[0]:+.3f} {target[1]:+.3f} {target[2]:+.3f}` m, "
            f"{np.linalg.norm(target - here) * 1000:.0f} mm away\n\n{angles}\n\n"
            f"solver residual `{residual_mm:.3f}` mm · wrist_roll is out of the solve, so it holds"
            "\n\nExecute walks there one step per frame, the same five steps re-solving each frame."
        )

    def _start_executing(self):
        with self._lock:
            if self._plan is None:
                self.plan_state.content = "Press Plan first."
                return
            self._executing = True
        self.pipeline.reset()

    def _stop_executing(self, why):
        with self._lock:
            self._executing = False
        self.plan_state.content = why

    def observe(self, observation) -> None:
        """Show where the arm actually is, and park an idle handle on the gripper."""
        joints, here = gripper_xyz_of(observation, self.kinematics)
        gripper_pct = float(observation["gripper.pos"])
        self.arm.update_cfg(np.asarray(viewer_configuration(self.joint_names, joints, gripper_pct)))
        with self._lock:
            self._observation = dict(observation)
            mode, executing, plan = self._mode, self._executing, self._plan
        if mode != IK:
            self.handle.position = tuple(float(v) for v in here)
        if mode == KEYBOARD and self.keyboard is not None:
            self.keyboard_state.content = held_keys_text(self.keyboard.current_pressed)
        if mode == LEADER and self.leader is not None:
            self.leader_state.content = following_text(self.leader.get_action(), observation)
        self.status.content = (
            f"**{MODE_LABELS[mode]}** · gripper `{here[0]:+.3f} {here[1]:+.3f} {here[2]:+.3f}` m"
            f" · reported opening `{gripper_pct:.0f}` · torque held"
            + (f" · executing, {float(np.linalg.norm(np.asarray(plan[0]) - here) * 1000):.0f} mm to go"
               if executing and plan is not None else "")
        )

    def wait_for_a_browser(self, seconds, poll_seconds) -> None:
        """Return once someone has the page open. Raises if nobody does.

        Runs before the robot is connected, so nothing is powered while it waits: an arm that is
        live with no operator at the controls is the state to avoid.
        """
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if self.server.get_clients():
                print("Browser connected.", flush=True)
                return
            time.sleep(poll_seconds)
        raise RuntimeError(
            f"No browser opened the page in {seconds:.0f} s, so the arm was never powered.\n"
            f"    Open http://{LOOPBACK}:{self.config.web_port} on this machine and run it again.\n"
            "    The page binds to the loopback interface only, so another machine cannot reach it."
        )


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


def run(args):
    from lerobot.processor import make_default_processors
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
    from lerobot.scripts.lerobot_teleoperate import teleop_loop
    from lerobot.teleoperators.keyboard import KeyboardEndEffectorTeleop, KeyboardEndEffectorTeleopConfig
    from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig

    kinematics, limits = model.load_kinematics(args.model_dir)
    bounds = model.bounds_dict(args.bounds_min_m, args.bounds_max_m)
    pipeline = model.build_robot_action_processor(kinematics, bounds, step_m=args.step_mm / 1000)

    keyboard = KeyboardEndEffectorTeleop(
        KeyboardEndEffectorTeleopConfig(id=args.robot_id, use_gripper=True))
    leader = None
    if args.leader_port:
        leader = SO101Leader(SO101LeaderConfig(
            port=args.leader_port, id=args.leader_id or args.robot_id,
            calibration_dir=Path(args.leader_calibration_dir), use_degrees=True))

    # The official follower configuration, with two fields set deliberately and nothing added.
    # ⛔ No max_relative_target: on a proportional servo it clamps the goal-to-present gap, which
    # is what makes the force, so it caps torque rather than speed (2026-09-08 logs).
    # ⭐ disable_torque_on_disconnect=False: the arm keeps holding when this program stops, so it
    # does not drop at the end of a session. Cut DC power to release it.
    robot = SO101Follower(SO101FollowerConfig(
        port=args.port, id=args.robot_id, calibration_dir=Path(args.calibration_dir),
        use_degrees=True, cameras={}, disable_torque_on_disconnect=False,
    ))
    page_config = ViserTeleopConfig(
        id=args.robot_id, model_dir=args.model_dir, web_port=args.web_port, step_mm=args.step_mm,
        calibration_dir=Path(args.calibration_dir),
        bounds={"min": bounds["min"], "max": bounds["max"]},
    )

    # Every input device first, and an operator proven to be at the page, before any motor is
    # powered: connecting the follower enables torque.
    keyboard.connect()
    if not keyboard.is_connected:
        # pynput captures keys only on an X11 session. Offering a mode that silently does
        # nothing is what made the last attempt look like a dead program.
        print("Keyboard capture is unavailable here, so that mode is left out of the page.\n"
              "    pynput needs an X11 session: not Wayland, and not a bare ssh connection.",
              file=sys.stderr)
        keyboard = None
    page = build(page_config, kinematics, pipeline, keyboard=keyboard, leader=leader)
    page.connect()
    try:
        page.wait_for_a_browser(BROWSER_WAIT_SECONDS, BROWSER_POLL_SECONDS)
        # Read the arm and judge the pose over its own bus, with nothing energised. Doing this
        # after connect() would be too late: connect() ends by enabling torque, which is the very
        # thing a joint resting on its stop must not have (2026-09-08 logs).
        observation, already_powered = model.read_pose_before_power(robot)
        here = model.check_start_pose(observation, kinematics, bounds, limits)
        if already_powered:
            print(f"Note: torque was already enabled on {', '.join(already_powered)} before this "
                  "run started.", file=sys.stderr)
        if leader is not None:
            leader.connect()
        robot.connect()
    except Exception:
        # Nothing has been commanded yet, so the arm is still in the self-supporting pose the
        # operator left it in: letting go is safe, and without it a refusal would hold the arm
        # while telling the operator to move it by hand.
        model.release_torque(robot, "startup stopped before any command was sent")
        if keyboard is not None:
            keyboard.disconnect()
        page.disconnect()
        raise

    try:
        page.seed(observation)
        print(f"\nFollower on {args.port} is live. Gripper at "
              f"x={here[0]:.3f} y={here[1]:.3f} z={here[2]:.3f} m, inside the workspace.\n"
              f"Pick a mode on the page. Ctrl+C stops the program; the motors KEEP holding, so "
              f"the arm does not drop.\nCut DC power to release them.\n", flush=True)
        _unused, identity_action, identity_observation = make_default_processors()
        teleop_loop(
            teleop=page,
            robot=robot,
            fps=args.fps,
            teleop_action_processor=display_tap(page),
            robot_action_processor=identity_action,
            robot_observation_processor=identity_observation,
        )
    finally:
        if keyboard is not None:
            keyboard.disconnect()
        if leader is not None:
            leader.disconnect()
        page.disconnect()
        robot.disconnect()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--port", required=True, help="follower serial port")
    parser.add_argument("--robot-id", required=True)
    parser.add_argument("--calibration-dir", required=True)
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
        run(args)
        return 0
    except KeyboardInterrupt:
        print("\nStopped. The motors are still holding; cut DC power to release them.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"STOP: {exc}\nCut motor power before investigating.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
