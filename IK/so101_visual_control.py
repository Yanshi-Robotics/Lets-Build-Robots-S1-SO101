#!/usr/bin/env python3
"""Drag the SO-101 gripper in a browser, through LeRobot 0.6.1's own IK pipeline.

The page is a teleoperator and nothing else. It subclasses LeRobot's `Teleoperator`, the same
base class its keyboard, gamepad and phone devices use, and reports the same four numbers they
do: delta_x, delta_y, delta_z and a gripper command. Everything after that is LeRobot's -- the
five end-effector processor steps, the SO101Follower, and the teleoperation loop. There is no
control loop, no power sequencing and no motion planner of our own.

    drag the handle   move the gripper       release   the handle returns to the arm
    open / close      the two buttons        Ctrl+C    stop -- every motor releases

Connecting enables motor torque. Fold the arm into Rest first, keep the DC cutoff within reach,
and read Lesson 7 before running this. The page binds to 127.0.0.1 only. Software bounds are
not a stop and not a collision check.
"""
from __future__ import annotations

import argparse
import math
import socket
import sys
import threading
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
# The handle is a rate control, not a position command: however far it is dragged, one frame
# asks for at most one step. Releasing it puts it back on the gripper, so the two never diverge.
MAX_UNITS_PER_FRAME = 1.0
# Below this a drag counts as noise, matching MapDeltaActionToRobotActionStep's own 1e-3
# threshold on the same units.
DRAG_NOISE_UNITS = 1e-3
GRIPPER_CLOSE, GRIPPER_HOLD, GRIPPER_OPEN = 0, 1, 2
# The gripper's URDF travel is a geometry reference; LeRobot reports a 0-100 opening instead.
GRIPPER_URDF_RANGE_RAD = (-0.174533, 1.74533)
ARM_COLOR = (0.18, 0.48, 0.72, 1.0)
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


def drag_to_units(handle_xyz, gripper_xyz, step_m):
    """How many steps this frame should ask for, per axis, from where the handle was dragged.

    A pure function so the rate limit is testable without a browser: the arm is never asked for
    more than one step per frame, however far or fast the handle is thrown.
    """
    if step_m <= 0 or not math.isfinite(step_m):
        raise ValueError("Step size must be positive")
    delta = (np.asarray(handle_xyz, dtype=float) - np.asarray(gripper_xyz, dtype=float)) / step_m
    if not np.isfinite(delta).all():
        return np.zeros(3)
    delta = np.clip(delta, -MAX_UNITS_PER_FRAME, MAX_UNITS_PER_FRAME)
    return np.where(np.abs(delta) < DRAG_NOISE_UNITS, 0.0, delta)


def viewer_configuration(names, joints_deg, gripper_pct):
    """Pose the browser model by URDF joint name, gripper opening included."""
    by_name = dict(zip(model.JOINTS, (math.radians(float(v)) for v in joints_deg)))
    low, high = GRIPPER_URDF_RANGE_RAD
    by_name["gripper"] = low + (high - low) * float(np.clip(float(gripper_pct), 0.0, 100.0)) / 100
    if set(names) != set(by_name):
        raise ValueError("Unexpected actuated joints in the pinned visual model")
    return [by_name[name] for name in names]


class ViserEndEffectorTeleop:
    """A browser page that reports the same delta action LeRobot's keyboard device reports.

    `build` below mixes this with LeRobot's Teleoperator base class, so importing this module
    needs no lerobot. The arm's own state reaches the page through `observe`, which the display
    tap calls once per frame with the observation the teleoperation loop already read.
    """

    name = "viser_ee"

    def __init__(self, config: ViserTeleopConfig, kinematics):
        self.config = config
        self.id = config.id
        self.kinematics = kinematics
        self.step_m = config.step_mm / 1000
        self._lock = threading.Lock()
        self._dragging = False
        self._gripper = GRIPPER_HOLD
        self._gripper_xyz = np.zeros(3)
        self._connected = False
        self.server = None

    # -- Teleoperator interface ----------------------------------------------------------
    @property
    def action_features(self) -> dict:
        return {"dtype": "float32", "shape": (4,),
                "names": {"delta_x": 0, "delta_y": 1, "delta_z": 2, "gripper": 3}}

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
        self.arm = ViserUrdf(self.server, Path(self.config.model_dir) / model.URDF_NAME,
                             root_node_name="/arm", mesh_color_override=ARM_COLOR)
        self.joint_names = self.arm.get_actuated_joint_names()
        self.handle = self.server.scene.add_transform_controls(
            "/target", scale=HANDLE_SCALE, disable_rotations=True,
            translation_limits=tuple(
                (float(lo), float(hi))
                for lo, hi in zip(self.config.bounds["min"], self.config.bounds["max"])
            ),
        )
        self.handle.on_drag_start(self._drag_start)
        self.handle.on_drag_end(self._drag_end)
        with self.server.gui.add_folder("Gripper"):
            close_button = self.server.gui.add_button("Close")
            open_button = self.server.gui.add_button("Open")
        close_button.on_click(lambda _: self._set_gripper(GRIPPER_CLOSE))
        open_button.on_click(lambda _: self._set_gripper(GRIPPER_OPEN))
        self.status = self.server.gui.add_markdown("Waiting for the first arm reading.")
        self._connected = True
        print(f"Open http://{LOOPBACK}:{self.config.web_port} and drag the handle to move the gripper.")

    def get_action(self) -> dict:
        """One frame's command. LeRobot's teleoperation loop calls this at the loop rate."""
        with self._lock:
            gripper, dragging = self._gripper, self._dragging
            self._gripper = GRIPPER_HOLD
            gripper_xyz = self._gripper_xyz.copy()
        delta = drag_to_units(self.handle.position, gripper_xyz, self.step_m) if dragging else np.zeros(3)
        return {"delta_x": float(delta[0]), "delta_y": float(delta[1]), "delta_z": float(delta[2]),
                "gripper": int(gripper)}

    def disconnect(self) -> None:
        if self.server is not None:
            self.server.stop()
            self.server = None
        self._connected = False

    # -- The page's own behaviour --------------------------------------------------------
    def observe(self, observation) -> None:
        """Show where the arm actually is, and keep a released handle sitting on the gripper."""
        joints = [float(observation[f"{name}.pos"]) for name in model.JOINTS]
        gripper_pct = float(observation["gripper.pos"])
        gripper_xyz = self.kinematics.forward_kinematics(np.asarray(joints, dtype=float))[:3, 3]
        self.arm.update_cfg(np.asarray(viewer_configuration(self.joint_names, joints, gripper_pct)))
        with self._lock:
            self._gripper_xyz = np.asarray(gripper_xyz, dtype=float).copy()
            dragging = self._dragging
        if not dragging:
            self.handle.position = tuple(float(v) for v in gripper_xyz)
        self.status.content = (
            f"gripper `{gripper_xyz[0]:+.3f} {gripper_xyz[1]:+.3f} {gripper_xyz[2]:+.3f}` m"
            f" - opening `{gripper_pct:.0f}%` -"
            f" {'dragging' if dragging else 'handle parked on the arm'}"
        )

    def _drag_start(self, _event) -> None:
        with self._lock:
            self._dragging = True

    def _drag_end(self, _event) -> None:
        with self._lock:
            self._dragging = False

    def _set_gripper(self, command) -> None:
        with self._lock:
            self._gripper = command


def build(config, kinematics):
    """Make the page a real LeRobot Teleoperator subclass, without importing lerobot on import."""
    from lerobot.teleoperators import Teleoperator
    subclass = type("ViserEndEffectorTeleop", (ViserEndEffectorTeleop, Teleoperator), {})
    return subclass(config, kinematics)


def display_tap(page):
    """A pass-through pipeline whose only effect is showing the arm on the page.

    LeRobot's teleoperation loop hands the observation it already read to the teleop action
    pipeline on every frame, and that pipeline is a parameter of the loop precisely so it can be
    replaced. Mirroring the arm here costs no extra bus read. The step returns the action
    untouched and computes nothing: the control chain stays exactly the official one.
    """
    from lerobot.processor import (
        RobotActionProcessorStep,
        RobotProcessorPipeline,
        TransitionKey,
    )
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

    kinematics, _limits = model.load_kinematics(args.model_dir)
    bounds = model.bounds_dict(args.bounds_min_m, args.bounds_max_m)
    robot_action_processor = model.build_robot_action_processor(
        kinematics, bounds, step_m=args.step_mm / 1000
    )
    *_unused, robot_observation_processor = make_default_processors()

    # The official follower configuration, with nothing added. In particular no
    # max_relative_target: on a proportional servo that clamps the goal-to-present gap, which is
    # what makes the force, so it caps torque rather than speed (2026-09-08 logs).
    robot = SO101Follower(SO101FollowerConfig(
        port=args.port, id=args.robot_id, calibration_dir=Path(args.calibration_dir),
        use_degrees=True, cameras={},
    ))
    page = build(ViserTeleopConfig(
        id=args.robot_id, model_dir=args.model_dir, web_port=args.web_port,
        step_mm=args.step_mm, calibration_dir=Path(args.calibration_dir),
        bounds={"min": bounds["min"], "max": bounds["max"]},
    ), kinematics)

    robot.connect()
    page.connect()
    try:
        teleop_loop(
            teleop=page,
            robot=robot,
            fps=args.fps,
            teleop_action_processor=display_tap(page),
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor,
        )
    finally:
        page.disconnect()
        robot.disconnect()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--port", required=True)
    parser.add_argument("--robot-id", required=True)
    parser.add_argument("--calibration-dir", required=True)
    parser.add_argument("--web-port", type=int, default=DEFAULT_WEB_PORT)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--step-mm", type=float, default=model.STEP_MM,
                        help="gripper travel per frame while the handle is held away from it")
    parser.add_argument("--bounds-min-m", nargs=3, type=float, default=model.DEFAULT_BOUNDS_M["min"])
    parser.add_argument("--bounds-max-m", nargs=3, type=float, default=model.DEFAULT_BOUNDS_M["max"])
    args = parser.parse_args(argv)
    if args.fps <= 0 or args.step_mm <= 0:
        parser.error("fps and step-mm must be positive")
    try:
        run(args)
        return 0
    except KeyboardInterrupt:
        print("\nStopped. Motors are released; this is not a physical power cutoff.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"STOP: {exc}\nCut motor power before investigating.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
