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

Read-only software does not guarantee an unpowered or motionless arm. This one does write a
single register per motor — the one that releases it — and nothing else.
"""
from __future__ import annotations

import argparse
import sys
from importlib.metadata import version

JOINT_NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
LEROBOT_VERSION = "0.6.1"
MODEL = "sts3215"
# Matches scservo_sdk's ERRBIT_*: what each bit of a motor's status byte means.
ERROR_BITS = {1: "voltage", 2: "angle-sensor", 4: "overheat", 8: "overcurrent", 32: "overload"}
# Retries per motor. Enough for an occasional dropped serial packet, short enough that a
# genuinely dead motor does not stall the whole release.
RETRIES = 5


def describe_flags(status: int) -> str:
    names = [name for bit, name in ERROR_BITS.items() if status & bit]
    return ",".join(names) if names else "none"


def release_bus(bus, get_address, *, output=print):
    """Release every motor in turn and report what each one says afterwards.

    Returns the motors that are still on or did not answer, so the caller can decide.
    Separated from main() so the release logic is testable without a serial port.
    """
    torque_addr, torque_len = get_address(bus.model_ctrl_table, MODEL, "Torque_Enable")
    status_addr, status_len = get_address(bus.model_ctrl_table, MODEL, "Status")
    still_on = []
    for name in JOINT_NAMES:
        id_ = bus.motors[name].id
        bus._write(torque_addr, torque_len, id_, 0, num_retry=RETRIES, raise_on_error=False)
        torque, comm, _ = bus._read(torque_addr, torque_len, id_, num_retry=RETRIES, raise_on_error=False)
        status, comm2, _ = bus._read(status_addr, status_len, id_, num_retry=RETRIES, raise_on_error=False)
        if not (bus._is_comm_success(comm) and bus._is_comm_success(comm2)):
            output(f"{name} (id {id_}): NO REPLY")
            still_on.append(name)
            continue
        if torque != 0:
            still_on.append(name)
        output(f"{name} (id {id_}): torque={'OFF' if torque == 0 else 'ON'} status_flags={describe_flags(status)}")
    return still_on


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", required=True, help="the controller's serial port, such as /dev/ttyACM0 or COM7")
    args = parser.parse_args(argv)

    try:
        if version("lerobot") != LEROBOT_VERSION:
            raise RuntimeError(f"Use lerobot=={LEROBOT_VERSION} in the course environment.")
        from lerobot.motors import Motor, MotorNormMode
        from lerobot.motors.feetech import FeetechMotorsBus
        from lerobot.motors.motors_bus import get_address
        bus = FeetechMotorsBus(port=args.port, motors={
            name: Motor(index, MODEL, MotorNormMode.RANGE_M100_100)
            for index, name in enumerate(JOINT_NAMES, 1)
        })
        # ⛔ handshake=False on purpose: a handshake treats a motor with an error bit as
        # absent, which is exactly the motor this program exists to release.
        bus.connect(handshake=False)
    except Exception as exc:  # noqa: BLE001 - one message, no traceback, for a course tool
        print(f"STOP: {exc}")
        return 2

    try:
        still_on = release_bus(bus, get_address)
    finally:
        # disable_torque=False: the releasing is done above, per motor, without raising.
        bus.disconnect(disable_torque=False)

    if still_on:
        print(f"FAIL: torque still on or no reply: {', '.join(still_on)} — cut motor power.")
        return 1
    print("PASS: all six motors report Torque_Enable=0.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
