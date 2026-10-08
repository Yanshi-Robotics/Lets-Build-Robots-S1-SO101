#!/usr/bin/env python3
"""Switch off the torque on all six motors of one SO-101 bus, error flags and all.

    python Bringup/so101_release_torque.py --port "<FOLLOWER_PORT>"

Why this exists. When `lerobot-teleoperate` is stopped with Ctrl-C, LeRobot disconnects by
disabling torque motor by motor. A motor answering with an error bit set makes that raise,
and everything after it in the chain keeps its torque on: the arm stays stiff and the motors
stay warm while looking, from the outside, like a normal shutdown.

This was written on 2026-09-16 while recording episode 6 of Season 1, where the Follower's
shoulder_lift (id 2) came back with the Overload bit set and ids 3 to 6 were never released.

What it does differently. It skips the handshake, because a handshake treats a motor with an
error bit as absent; it writes Torque_Enable=0 to each id in turn without raising on a status
byte; and it reads Torque_Enable and Status back so the result is measured rather than
assumed. Cutting motor power is still the thing to do if any motor does not report OFF.

One reading of a motor is not proof that it let go. A servo that is overloaded does not release
by itself: it clamps its output to Protection_Torque (20% by default) and stays there, which
feels like a stiff arm even with the torque flag at 0. Sweeping the bus again and reading each
motor back is what tells the two apart — and if a motor still reports ON, cut power at the
supply, because Torque_Enable lives in SRAM and power is the one thing that always clears it.

Read-only software does not guarantee an unpowered or motionless arm. This one does write a
single register per motor — the one that releases it — and nothing else.

Upstream tracks the same fault as huggingface/lerobot#4695, open since 2026-09-21: per-motor
failure handling in disconnect(), so that one motor answering with an error bit cannot skip the
rest. Until that ships, `Bringup/so101_run.py` runs this release by itself after every session,
and this program is what you reach for when a command that did not go through that wrapper left
an arm stiff.
"""
from __future__ import annotations

import argparse
import sys
import time
from importlib.metadata import version

JOINT_NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
LEROBOT_VERSION = "0.6.1"
MODEL = "sts3215"
# Matches scservo_sdk's ERRBIT_*: what each bit of a motor's status byte means.
ERROR_BITS = {1: "voltage", 2: "angle-sensor", 4: "overheat", 8: "overcurrent", 32: "overload"}
# Retries per motor. Enough for an occasional dropped serial packet, short enough that a
# genuinely dead motor does not stall the whole release.
RETRIES = 5
# Sweeps of the whole bus. A write whose acknowledgement never arrives leaves the arm's state
# unknown, so the second sweep asks again rather than reporting a motor that a byte of noise can
# account for; a motor seen OFF on any sweep is not asked a third time.
PASSES = 3
PASS_INTERVAL_S = 0.2


def describe_flags(status: int) -> str:
    names = [name for bit, name in ERROR_BITS.items() if status & bit]
    return ",".join(names) if names else "none"


def release_motor(bus, motor, torque_addr, torque_len, status_addr, status_len, output=print):
    """Write Torque_Enable=0 to one motor and report what it says afterwards.

    Returns the joint name when the motor is still on or did not answer, otherwise None. Every
    failure is reported rather than raised: a motor that faults is the reason the other five
    still need their turn, so one bad answer must not end the sweep.
    """
    id_ = bus.motors[motor].id
    bus._write(torque_addr, torque_len, id_, 0, num_retry=RETRIES, raise_on_error=False)
    torque, comm, _ = bus._read(torque_addr, torque_len, id_, num_retry=RETRIES, raise_on_error=False)
    status, comm2, _ = bus._read(status_addr, status_len, id_, num_retry=RETRIES, raise_on_error=False)
    if not (bus._is_comm_success(comm) and bus._is_comm_success(comm2)):
        output(f"{motor} (id {id_}): NO REPLY")
        return motor
    output(f"{motor} (id {id_}): torque={'OFF' if torque == 0 else 'ON'} status_flags={describe_flags(status)}")
    return None if torque == 0 else motor


def release_bus(bus, get_address, *, output=print, passes=PASSES, sleep=time.sleep):
    """Release every motor in turn and report what each one says afterwards.

    Returns the joints that are still on or did not answer, so the caller can decide. The whole
    bus is swept `passes` times and a motor is released as soon as one sweep sees it off, so a
    healthy arm costs one sweep and a faulted one gets another chance instead of a report.
    Separated from main() so the release logic is testable without a serial port.
    """
    torque_addr, torque_len = get_address(bus.model_ctrl_table, MODEL, "Torque_Enable")
    status_addr, status_len = get_address(bus.model_ctrl_table, MODEL, "Status")
    still_on = list(JOINT_NAMES)
    for attempt in range(passes):
        if attempt:
            sleep(PASS_INTERVAL_S)
        if attempt:
            output(f"second look at {', '.join(still_on)}")
        still_on = [motor for motor in still_on
                    if release_motor(bus, motor, torque_addr, torque_len,
                                     status_addr, status_len, output) is not None]
        if not still_on:
            break
    return still_on


def open_bus(port, *, output=print, attempts=3, interval=1.0, sleep=time.sleep):
    """Open one arm's bus without the handshake.

    ⛔ handshake=False on purpose: a handshake treats a motor with an error bit as absent, which
    is exactly the motor this program exists to release. The retries are for the case this runs
    straight after another process let go of the port, where the tty can take a moment to come
    back.
    """
    from lerobot.motors import Motor, MotorNormMode
    from lerobot.motors.feetech import FeetechMotorsBus

    bus = FeetechMotorsBus(port=port, motors={
        name: Motor(index, MODEL, MotorNormMode.RANGE_M100_100)
        for index, name in enumerate(JOINT_NAMES, 1)
    })
    for attempt in range(attempts):
        try:
            bus.connect(handshake=False)
            return bus
        except Exception as exc:  # noqa: BLE001 - the last failure is reported by the caller
            if attempt + 1 == attempts:
                raise
            output(f"{port}: {exc} — trying again")
            sleep(interval)
    raise AssertionError("unreachable")


def release_port(port, *, output=print, attempts=3, sleep=time.sleep):
    """Release one arm and close its port. Returns the joints that are still on."""
    from lerobot.motors.motors_bus import get_address

    bus = open_bus(port, output=output, attempts=attempts, sleep=sleep)
    try:
        return release_bus(bus, get_address, output=output, sleep=sleep)
    finally:
        # disable_torque=False: the releasing is done above, per motor, without raising.
        bus.disconnect(disable_torque=False)


def check_lerobot_version(expected=LEROBOT_VERSION):
    found = version("lerobot")
    if found != expected:
        raise RuntimeError(f"Use lerobot=={expected} in the course environment, found {found}.")
    return found


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", required=True, help="the controller's serial port, such as /dev/ttyACM0 or COM7")
    args = parser.parse_args(argv)

    try:
        check_lerobot_version()
        still_on = release_port(args.port)
    except Exception as exc:  # noqa: BLE001 - one message, no traceback, for a course tool
        print(f"STOP: {exc}")
        return 2

    if still_on:
        print(f"FAIL: torque still on or no reply: {', '.join(still_on)} — cut motor power.")
        return 1
    print("PASS: all six motors report Torque_Enable=0.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
