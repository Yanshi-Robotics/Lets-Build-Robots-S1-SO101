#!/usr/bin/env python3
"""Drive the SO-101 gripper with the arrow keys, through LeRobot 0.6.1's own IK pipeline.

Everything that moves the arm here is LeRobot's: its keyboard end-effector teleoperator, its
five end-effector processor steps, its SO101Follower, and its teleoperation loop. This file
only wires them together, because no `lerobot-*` command does: `lerobot-teleoperate` builds
the identity processors and has no flag for the end-effector pipeline.

    arrow keys   move the gripper in x and y      shift / right shift   move it down / up
    left ctrl    close the gripper                right ctrl            open it
    Ctrl+C       stop -- every motor releases, and the arm drops if it is not folded up

Connecting enables motor torque. Fold the arm into Rest first, keep the DC cutoff in reach,
and read Lesson 7 before running this. Software bounds are not a stop and not a collision check.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import so101_cartesian_demo as model  # noqa: E402  (same folder, as the lesson downloads it)

DEFAULT_FPS = 30
# How long to wait for the operator to prove the keyboard is being captured, before any motor
# is powered. pynput needs an X11 session on Linux; on Wayland or over plain ssh it silently
# captures nothing, and without this check that looks exactly like a dead program.
KEY_CHECK_SECONDS = 30
KEY_POLL_SECONDS = 0.05

KEY_MAP = """  arrow keys           move the gripper in x and y
  shift / right shift  move it down / up
  left ctrl / right ctrl  close / open the gripper
  Ctrl+C               stop (every motor releases, and an unfolded arm drops)"""


def wait_for_a_key(teleop):
    """Return once the keyboard is proven to reach this program. Raises if it never does.

    Runs before the robot is connected, so nothing is powered while it waits.
    """
    print("Press any arrow key now, to prove the keyboard reaches this program.", flush=True)
    deadline = time.monotonic() + KEY_CHECK_SECONDS
    while time.monotonic() < deadline:
        action = teleop.get_action()
        if any(abs(float(action.get(f"delta_{axis}", 0.0))) > 0 for axis in "xyz"):
            print("Keyboard confirmed.", flush=True)
            return
        time.sleep(KEY_POLL_SECONDS)
    raise RuntimeError(
        f"No key reached this program in {KEY_CHECK_SECONDS} s, so the arm was never powered.\n"
        "    LeRobot captures keys through pynput, which on Linux only works on an X11 session:\n"
        "    not on Wayland, and not over an ssh connection without a local display.\n"
        "    Check `echo $XDG_SESSION_TYPE` prints x11, and run this from a terminal on that desktop."
    )


def run(args):
    from lerobot.processor import make_default_processors
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
    from lerobot.scripts.lerobot_teleoperate import teleop_loop
    from lerobot.teleoperators.keyboard import KeyboardEndEffectorTeleop, KeyboardEndEffectorTeleopConfig

    kinematics, limits = model.load_kinematics(args.model_dir)
    bounds = model.bounds_dict(args.bounds_min_m, args.bounds_max_m)
    teleop_action_processor, _identity, robot_observation_processor = make_default_processors()

    # The official follower configuration, with nothing added. In particular no
    # max_relative_target: on a proportional servo that clamps the goal-to-present gap, which
    # is what makes the force, so it caps torque rather than speed (2026-09-08 logs).
    robot = SO101Follower(SO101FollowerConfig(
        port=args.port, id=args.robot_id, calibration_dir=Path(args.calibration_dir),
        use_degrees=True, cameras={},
    ))
    teleop = KeyboardEndEffectorTeleop(KeyboardEndEffectorTeleopConfig(id=args.robot_id, use_gripper=True))

    # The input device first, and proven to work, before any motor is powered. Connecting the
    # follower enables torque, so an arm that is live while the operator has no working control
    # is the state to avoid.
    teleop.connect()
    try:
        wait_for_a_key(teleop)
        # Judged with nothing energised: connect() ends by enabling torque, which is exactly what
        # a joint resting on its stop must not have (2026-09-08 logs).
        observation, _powered, _stored = model.read_pose_before_power(robot)
        here = model.gripper_position(observation, kinematics)
        bounds = model.bounds_including(bounds, here)
        # ⛔ Park the goal before torque: a servo drives at its Goal_Position the instant torque
        # comes on, and connect() never writes one (2026-09-08 logs).
        robot.bus.connect()
        try:
            model.park_the_goal(robot, observation)
        finally:
            robot.bus.disconnect(disable_torque=False)
        robot.connect()
        drifted = model.goal_diverged(robot)
        if drifted:
            raise RuntimeError("Torque came on with motors being told to travel, not to hold:\n    "
                               + "\n    ".join(drifted))
    except Exception:
        teleop.disconnect()
        raise

    try:
        print(f"\nFollower on {args.port} is live. Gripper at "
              f"x={here[0]:.3f} y={here[1]:.3f} z={here[2]:.3f} m, inside the workspace.")
        robot_action_processor = model.build_robot_action_processor(
            kinematics, bounds, step_m=args.step_mm / 1000)
        print(f"{KEY_MAP}\n\nA held key moves the gripper {args.step_mm:g} mm per frame, "
              f"{args.step_mm * args.fps:g} mm/s at {args.fps} fps.\n", flush=True)
        teleop_loop(
            teleop=teleop,
            robot=robot,
            fps=args.fps,
            teleop_action_processor=teleop_action_processor,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor,
        )
    finally:
        teleop.disconnect()
        robot.disconnect()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--port", required=True)
    parser.add_argument("--robot-id", required=True)
    parser.add_argument("--calibration-dir", required=True)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--step-mm", type=float, default=model.STEP_MM,
                        help="gripper travel per frame while a direction key is held")
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
