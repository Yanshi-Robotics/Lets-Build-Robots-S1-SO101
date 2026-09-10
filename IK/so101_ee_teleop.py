#!/usr/bin/env python3
"""Drive the SO-101 gripper with the arrow keys, with no browser in the way.

The smallest complete version of what `so101_visual_control.py` does: the same solver, the same
speed limits, the same stall check, the same one control loop -- just a terminal instead of a
three-dimensional page. ⛔ The loop itself lives in `so101_cartesian_demo.py` and is imported,
never copied: a control loop kept in two files is a control loop that is right in one of them.

    arrow keys   move the target in x and y       shift / right shift   move it down / up
    left ctrl    close the gripper                right ctrl            open it
    Ctrl+C       stop

⚠️ Ctrl+C stops the program and deliberately does NOT release the motors: an arm that is holding
itself up drops when it is let go. The release command is printed on the way out.

⭐ The arrow keys move a target, and the arm follows that target -- they do not nudge the arm
directly. That is the whole difference between this working and not working on hardware: a
command that only ever asks for a millimetre sits inside the position error this servo needs
before it makes any force at all. The reasoning, and the measurements, are in
`so101_cartesian_demo.py`.

Enabling a mode enables motor torque. Keep the DC cutoff in reach, clear the bench, and read
Lesson 7 first. Software bounds are not a stop and not a collision check.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import so101_cartesian_demo as model  # noqa: E402  (same folder, as the lesson downloads it)

# How long to wait for the operator to prove the keyboard is being captured, before any motor
# is powered. pynput needs an X11 session on Linux; on Wayland or over plain ssh it silently
# captures nothing, and without this check that looks exactly like a dead program.
KEY_CHECK_SECONDS = 30
KEY_POLL_SECONDS = 0.05
# How often the status line is redrawn. The loop runs at 50 Hz; a terminal does not need to.
STATUS_EVERY_SECONDS = 0.2

KEY_MAP = """  arrow keys           move the target in x and y
  shift / right shift  move it down / up
  left ctrl / right ctrl  close / open the gripper
  Ctrl+C               stop (the motors stay held; the release command is printed)"""


def wait_for_a_key(teleop):
    """Return once the keyboard is proven to reach this program. Raises if it never does.

    ⛔ Before any motor is powered, on purpose. An input device that silently captures nothing
    looks exactly like a dead program -- and finding that out with the arm already live is how
    an operator ends up reaching for a moving arm.
    """
    print(f"Press and hold any movement key within {KEY_CHECK_SECONDS} s to prove the keyboard "
          "reaches this program.\n" + KEY_MAP, flush=True)
    deadline = time.perf_counter() + KEY_CHECK_SECONDS
    while time.perf_counter() < deadline:
        action = teleop.get_action()
        if any(float(action.get(axis, 0.0)) for axis in ("delta_x", "delta_y", "delta_z")):
            print("Keyboard confirmed.\n", flush=True)
            return
        time.sleep(KEY_POLL_SECONDS)
    raise RuntimeError(
        f"No key reached this program in {KEY_CHECK_SECONDS} s, so the arm was never powered.\n"
        "    pynput needs an X11 session: not Wayland, and not a bare ssh connection.\n"
        "    Check that `echo $XDG_SESSION_TYPE` prints x11.")


class ConsolePage:
    """The same page the shared control loop expects, printed instead of drawn.

    ⛔ Nothing here talks to the arm. It holds what the operator has asked for and hands it to
    the loop, exactly as the browser page does.
    """

    def __init__(self, bounds, mode=None):
        self.bounds = bounds
        self._mode = model.KEYBOARD if mode is None else mode
        self._handle = np.zeros(3)
        self._gripper = model.GRIPPER_MIN_PCT
        self._asked = False
        self._last_drawn = 0.0

    # -- what the loop reads -----------------------------------------------------------------
    def take_arm_request(self):
        if self._asked:
            return None
        self._asked = True
        return self._mode

    def take_end_request(self):
        return False        # Ctrl+C is the only way out, and it leaves the motors held.

    def handle_xyz(self):
        return self._handle

    def gripper_target(self):
        return self._gripper

    # -- what the loop writes ----------------------------------------------------------------
    def set_gripper_target(self, percent):
        self._gripper = float(np.clip(percent, model.GRIPPER_MIN_PCT, model.GRIPPER_MAX_PCT))

    def nudge_gripper(self, percent):
        self.set_gripper_target(self._gripper + percent)

    def move_handle(self, xyz):
        self._handle = np.asarray(xyz, dtype=float)

    def armed(self, mode):
        print(f"{model.MODE_LABELS[mode]} is live. Ctrl+C stops.\n", flush=True)

    def disarmed(self, message=""):
        if message:
            print(message, flush=True)

    def show(self, measured, commanded, status, note="", keys=""):
        if note:
            print(f"\n{note}", flush=True)
        now = time.perf_counter()
        if now - self._last_drawn < STATUS_EVERY_SECONDS:
            return
        self._last_drawn = now
        plain = status.replace("**", "").replace("`", "")
        print(f"\r{plain}   keys: {keys or '-':<20}", end="", flush=True)


def run(args):
    from lerobot.utils.keyboard_input import pynput_can_capture

    servo, limits = model.load_servo(args.model_dir, max_joint_speed=args.max_joint_speed)
    bounds = model.bounds_dict(args.bounds_min_m, args.bounds_max_m)
    if not pynput_can_capture():
        raise RuntimeError("LeRobot's keyboard device cannot capture keys in this session; "
                           "pynput needs X11.")

    arm = model.LiveArm(args.port, args.robot_id, args.calibration_dir,
                        p_coefficient=args.p_coefficient)
    observation, powered, in_the_motors = arm.read_before_power()
    if powered:
        raise RuntimeError(f"Torque is already enabled on {', '.join(powered)}. Support the arm "
                           "and release it with so101_visual_control.py --release-torque.")
    disagreeing = model.calibration_disagrees(in_the_motors, arm.robot.calibration)
    if disagreeing:
        raise RuntimeError(f"Calibration disagrees on {', '.join(disagreeing)}.")
    degrees = model.joint_degrees(observation)
    refused = model.joints_outside_the_model(degrees, limits)
    if refused:
        raise RuntimeError("Nothing was powered:\n    " + "\n    ".join(refused))
    bounds = model.bounds_including(bounds, servo.gripper_xyz(degrees))

    teleop = model.keyboard_device(args.robot_id)
    teleop.connect()                     # ⛔ before the arm: see wait_for_a_key
    try:
        wait_for_a_key(teleop)
        page = ConsolePage(bounds)
        page.move_handle(servo.gripper_xyz(degrees))
        arm.open_bus()
        model.control_loop(page, arm, servo, bounds, limits, keyboard=teleop)
    except model.FollowingLost as lost:
        print(f"\nSTOP-HOLD: {lost}\n"
              "           Stopped, and the arm is still held where it stands.",
              file=sys.stderr, flush=True)
        raise
    finally:
        if arm.holding:
            print("\n" + model_release_notice(args), file=sys.stderr, flush=True)
        teleop.disconnect()
        arm.close_bus()


def model_release_notice(args):
    return ("SAFETY: the motors were NOT released, because letting go of a raised arm drops it.\n"
            "        When the arm is supported, release them with:\n"
            "            python IK/so101_visual_control.py --release-torque "
            f"--model-dir {args.model_dir} \\\n"
            f"                --port {args.port} --robot-id {args.robot_id} "
            f"--calibration-dir {args.calibration_dir}\n"
            "        Cutting DC power does the same thing instantly.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--port", required=True, help="follower serial port")
    parser.add_argument("--robot-id", default="so101-follower")
    parser.add_argument("--calibration-dir", default="calibration/follower")
    parser.add_argument("--max-joint-speed", type=float, default=model.MAX_JOINT_SPEED_RAD_S,
                        help="radians a second, per joint: how far a command may lead the arm")
    parser.add_argument("--p-coefficient", type=int, default=model.SERVO_P_COEFFICIENT)
    parser.add_argument("--bounds-min-m", nargs=3, type=float, default=model.DEFAULT_BOUNDS_M["min"])
    parser.add_argument("--bounds-max-m", nargs=3, type=float, default=model.DEFAULT_BOUNDS_M["max"])
    args = parser.parse_args(argv)
    try:
        run(args)
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"STOP: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
