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

On POSIX, the child runs in its own process group. Ctrl-C is forwarded once, and the wrapper
waits for shutdown before opening either bus. A cleanup timeout begins only after a stop
request. A child that does not stop is terminated and reaped; a failed shutdown prevents release.

The release writes only Torque_Enable=0 and then checks all motors repeatedly. Exit code 0
means the registers reported OFF, not that physical release was measured. Support the arms
before stopping. If they remain stiff, cut motor power and retain the diagnostic output.

Ports come from --robot.port and --teleop.port unless --ports is supplied. --hold skips this
wrapper's additional release; it cannot prevent the child from performing its own shutdown.
Recording defaults to --hold. --release explicitly requests the additional release for recording.

Read-only software does not guarantee an unpowered or motionless arm.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import signal
import subprocess
import sys
import time
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
# Bound forced shutdown separately from normal operation; neither limits session duration.
CHILD_TERMINATE_TIMEOUT_S = 5.0
CHILD_POLL_INTERVAL_S = 0.1


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


def run_child(command, *, cleanup_timeout=CHILD_CLEANUP_TIMEOUT_S,
              terminate_timeout=CHILD_TERMINATE_TIMEOUT_S):
    """Forward stop requests to an owned POSIX process group and reap before returning.

    A callable parent handler is reset by exec; SIG_IGN would be inherited by Python children.
    The child's separate session avoids delivering terminal Ctrl-C twice during its cleanup.
    Its inherited stdin/stdout remain available for calibration prompts and logs.
    """
    if os.name != "posix":
        raise RuntimeError("This wrapper's shutdown supervision currently requires POSIX.")
    requested = []

    def request_stop(signum, _frame):
        if not requested:
            requested.append(signum)

    previous = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    child = None

    def send_group(sig):
        try:
            os.killpg(child.pid, sig)
        except ProcessLookupError:
            pass

    try:
        child = subprocess.Popen(command, start_new_session=True)  # noqa: S603
        deadline = None
        terminating = False
        while child.poll() is None:
            now = time.monotonic()
            if requested and deadline is None:
                send_group(requested[0])
                deadline = now + cleanup_timeout
            elif deadline is not None and now >= deadline:
                if not terminating:
                    print("so101_run: cleanup timed out; terminating the child group", flush=True)
                    send_group(signal.SIGTERM)
                    terminating = True
                    deadline = now + terminate_timeout
                else:
                    send_group(signal.SIGKILL)
                    child.wait(timeout=terminate_timeout)
                    break
            time.sleep(CHILD_POLL_INTERVAL_S)
        return child.wait()
    finally:
        # Descendants must not survive an exited launcher and continue controlling the bus.
        # Only this newly created process group is targeted, never a name or global pattern.
        try:
            if child is not None:
                send_group(signal.SIGKILL)
                child.wait(timeout=terminate_timeout)
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)


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
                        help="skip additional release; the child still performs its own cleanup")
    parser.add_argument("--release", action="store_true",
                        help="release anyway, for a recording session whose arm is already supported")
    args = parser.parse_args(own)

    if not command:
        parser.error("no command to run; write it after `--`, such as `-- lerobot-teleoperate ...`")

    # Recording holds on by default. A Follower that drops at the end of a recording session is a
    # fall for no reason, and the operator is standing right there to support the arm before
    # releasing it. Nobody has to remember a flag for the safe choice to happen. The command is
    # recognised by its name, whichever of the three ways it is spelled: the console script, or
    # `python -m lerobot.scripts.lerobot_record`.
    recording = any("lerobot_record" in Path(part).name or Path(part).name == "lerobot-record"
                    for part in command[:3])
    hold = (args.hold or recording) and not args.release

    ports = args.ports if args.ports is not None else ports_in_arguments(command)
    if not hold and not ports:
        output("STOP: no arm port found in the command; pass --ports <PORT> [<PORT> ...] "
               "(or --hold to leave torque on)")
        return 2

    try:
        release.check_lerobot_version()
    except Exception as exc:  # noqa: BLE001 - same one-line style as the release program
        output(f"STOP: {exc}")
        return 2

    output(f"so101_run: running {' '.join(command)}")
    try:
        code = run(command)
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
        output(f"STOP: child shutdown not confirmed ({exc}); no release attempted. Cut motor power.")
        return 2

    if hold:
        why = "recording" if recording and not args.hold else "--hold"
        output(f"\nso101_run: {why}, additional release skipped. Torque state is unverified; support the arms, "
               "then run:")
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
    output(f"REGISTER CHECK: all six motors on {len(ports)} arm(s) report Torque_Enable=0. "
           "Physical release is not verified; if still stiff, support the arms and cut motor power.")
    # The exit code answers the question this program is here to answer: did the arms let go? A
    # command stopped with Ctrl-C exits non-zero as a matter of course, so passing its code on
    # would make every clean shutdown look like a failure. A command that failed on its own says
    # so in its own output, which is right above this line.
    if code:
        output(f"note: the command exited {code}; register checks completed, physical release is unverified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
