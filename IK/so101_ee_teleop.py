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
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import so101_cartesian_demo as model  # noqa: E402  (same folder, as the lesson downloads it)

DEFAULT_FPS = 30


def run(args):
    from lerobot.processor import make_default_processors
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
    from lerobot.scripts.lerobot_teleoperate import teleop_loop
    from lerobot.teleoperators.keyboard import KeyboardEndEffectorTeleop, KeyboardEndEffectorTeleopConfig

    kinematics, _limits = model.load_kinematics(args.model_dir)
    bounds = model.bounds_dict(args.bounds_min_m, args.bounds_max_m)
    robot_action_processor = model.build_robot_action_processor(
        kinematics, bounds, step_m=args.step_mm / 1000
    )
    teleop_action_processor, _identity, robot_observation_processor = make_default_processors()

    # The official follower configuration, with nothing added. In particular no
    # max_relative_target: on a proportional servo that clamps the goal-to-present gap, which
    # is what makes the force, so it caps torque rather than speed (2026-09-08 logs).
    robot = SO101Follower(SO101FollowerConfig(
        port=args.port, id=args.robot_id, calibration_dir=Path(args.calibration_dir),
        use_degrees=True, cameras={},
    ))
    teleop = KeyboardEndEffectorTeleop(KeyboardEndEffectorTeleopConfig(id=args.robot_id, use_gripper=True))

    robot.connect()
    teleop.connect()
    try:
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
