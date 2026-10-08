#!/usr/bin/env python3
"""Run a LeRobot command against the arms and release torque on both when it exits.

    python Bringup/so101_run.py -- lerobot-teleoperate \
        --robot.type=so101_follower \
        --robot.port=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B8E116415-if00 \
        --robot.id=my_follower \
        --teleop.type=so101_leader \
        --teleop.port=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B8E116437-if00 \
        --teleop.id=my_leader

Everything before the `--` belongs to this program (`--hold`, `--ports`); everything after it is
the command to run, written exactly as it would be written on its own.

Why this exists. LeRobot disables torque on the way out, one motor at a time, and raises on the
first motor that answers with an error bit — so that motor and every motor after it in the list
keep their torque: the arm stays stiff and the motors stay warm after a shutdown that looked
normal. Measured on 2026-10-09 against lerobot 0.6.1, with id 2 reporting Overload: disconnect()
raises, ids 1 and 2 are written, and ids 3 to 6 are never written at all. The same code is still
in LeRobot's main branch; the fix is huggingface/lerobot#4695, open since 2026-09-21.

What this does. It starts the command unchanged and lets it handle Ctrl-C as it always did, then
releases torque itself on both arms, per motor, without raising on a status byte, reading each
one back. Both arms: the Follower's motors are the ones under load, but the Leader's are left
holding too, and releasing it also stops the leader arm from dropping onto the desk under its own
weight mid-session.

Two things worth knowing before it is used in anger:

  * With no --ports it reads the ports out of the command's own arguments (--robot.port and
    --teleop.port). If it cannot find them it stops instead of guessing, because writing to a
    port that is not an arm is not a thing this program gets to do.
  * `lerobot-record` is not teleoperation: a dropped Follower at the end of a recording session
    is a fall for no reason. Pass --hold and torque is left on; release it yourself once the arm
    is supported.

Its exit code answers the one question it is here to answer, whether the arms let go: 0 when every
motor reports its torque off, 1 when one does not and the supply is the thing to cut, 2 when it
stopped before it could tell. A command stopped with Ctrl-C exits non-zero as a matter of course,
so with --hold the command's own code is passed through instead.

Read-only software does not guarantee an unpowered or motionless arm.
"""
from __future__ import annotations

import argparse
import importlib.util
import signal
import subprocess
import sys
from pathlib import Path

# The release logic stays in the program that owns it, so both write the same single register.
# Loaded by path rather than by putting Bringup/ on sys.path, which would shadow a module name
# as ordinary as this one for anything the child imports later.
_spec = importlib.util.spec_from_file_location(
    "so101_release_torque", Path(__file__).resolve().parent / "so101_release_torque.py")
release = importlib.util.module_from_spec(_spec)
sys.modules["so101_release_torque"] = release
_spec.loader.exec_module(release)

# Options LeRobot accepts that state which port an arm is on. The wrapper reads them so the
# operator does not have to write every port twice.
PORT_OPTIONS = ("--robot.port", "--teleop.port")
# The child closes its own port on the way out; give it this long after Ctrl-C before reaching
# for the port ourselves. Waiting costs nothing when the child exits at once.
CHILD_CLEANUP_TIMEOUT_S = 30.0


def split_command(argv):
    """Split the wrapper's own options from the command they wrap, after a bare `--`.

    Everything up to the first standalone `--` is ours; everything after it is the child's, and
    `--` itself is dropped. When there is no `--`, the whole list is the child's command.
    """
    for index, arg in enumerate(argv):
        if arg == "--":
            return list(argv[:index]), list(argv[index + 1:])
    return [], list(argv)


def ports_in_arguments(arguments):
    """Read the arm ports out of a LeRobot command line, in the order they appear.

    Accepts both `--robot.port=X` and `--robot.port X`, which is how these options are written
    in practice; a flag followed by another flag is not a port.
    """
    ports = []
    for index, arg in enumerate(arguments):
        for option in PORT_OPTIONS:
            value = None
            if arg == option and index + 1 < len(arguments) and not arguments[index + 1].startswith("-"):
                value = arguments[index + 1]
            elif arg.startswith(option + "="):
                value = arg[len(option) + 1:]
            if value:
                ports.append(value)
    ordered, seen = [], set()
    for port in ports:
        if port not in seen:
            seen.add(port)
            ordered.append(port)
    return ordered


def run_child(command):
    """Run the wrapped command with this process's terminal, ignoring Ctrl-C while it cleans up.

    Ctrl-C reaches the child directly, because both processes share the foreground process
    group; the wrapper's own handler is what has to wait rather than die, or the release would
    race the child's shutdown instead of following it.
    """
    previous = signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        child = subprocess.Popen(command)  # noqa: S603 - the operator's own command line
        try:
            return child.wait(timeout=CHILD_CLEANUP_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            print(f"\nso101_run: {command[0]} still has not exited after {CHILD_CLEANUP_TIMEOUT_S:.0f}s; "
                  "releasing torque anyway", flush=True)
            # Left running on purpose: killing it here would leave the port open, and the release
            # below is the part that has to happen. A process still holding the port is reported
            # rather than hidden.
            return 0
    finally:
        signal.signal(signal.SIGINT, previous)


def release_every(ports, *, release_port=release.release_port, output=print):
    """Release each arm in turn. Returns the ports whose motors did not all report OFF."""
    still_on = {}
    for port in ports:
        output(f"\nso101_run: releasing torque on {port}")
        try:
            left = release_port(port, output=output)
        except Exception as exc:  # noqa: BLE001 - one line per arm beats a traceback here
            output(f"{port}: could not release ({exc})")
            still_on[port] = ["unknown"]
            continue
        if left:
            still_on[port] = left
    return still_on


def main(argv=None, *, run=run_child, release_ports=release_every, output=print) -> int:
    own, command = split_command(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ports", nargs="+", default=None,
                        help="arm ports to release; default is to read them from the command")
    parser.add_argument("--hold", action="store_true",
                        help="leave torque on (recording: a dropped Follower is a fall for no reason)")
    args = parser.parse_args(own)

    if not command:
        parser.error("no command to run; write it after `--`, such as `-- lerobot-teleoperate ...`")

    ports = args.ports if args.ports is not None else ports_in_arguments(command)
    if not args.hold and not ports:
        output("STOP: no arm port found in the command; pass --ports <PORT> [<PORT> ...] "
               "(or --hold to leave torque on)")
        return 2

    try:
        release.check_lerobot_version()
    except Exception as exc:  # noqa: BLE001 - same one-line style as the release program
        output(f"STOP: {exc}")
        return 2

    output(f"so101_run: running {' '.join(command)}")
    code = run(command)

    if args.hold:
        output("\nso101_run: --hold, torque left on. The arms are still powered; support them, then run:")
        for port in ports:
            output(f"  python Bringup/so101_release_torque.py --port {port}")
        return code

    output("\nso101_run: the command has exited; releasing torque on both arms")
    try:
        still_on = release_ports(ports, output=output)
    except KeyboardInterrupt:
        output("\nSTOP: interrupted while releasing; some motors may still be on. Re-run:\n"
               "  python Bringup/so101_release_torque.py --port <PORT>")
        return 2

    if still_on:
        for port, names in still_on.items():
            output(f"FAIL: {port}: torque still on or no reply: {', '.join(names)} — cut motor power.")
        return 1
    output(f"PASS: all six motors on {len(ports)} arm(s) report Torque_Enable=0.")
    # The exit code answers the question this program is here to answer: did the arms let go? A
    # command stopped with Ctrl-C exits non-zero as a matter of course, so passing its code on
    # would make every clean shutdown look like a failure. A command that failed on its own says
    # so in its own output, which is right above this line.
    if code:
        output(f"note: the command exited {code}; the arms are released, the exit code is about "
               "the command, not the arms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
