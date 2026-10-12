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

Register readback is not proof of physical release. This program checks communication results
and samples all six torque registers again after writing. It cannot measure holding force or
identify a firmware braking state. If the arm remains stiff, support it and cut motor power.

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
import math
import os
import subprocess
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
# Two delayed checks after the initial check catch re-enabling without writing it away.
VERIFY_SAMPLES = 3
VERIFY_INTERVAL_S = 1.0


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
    try:
        write_comm, write_error = bus._write(
            torque_addr, torque_len, id_, 0, num_retry=RETRIES, raise_on_error=False)
        output(f"{motor} (id {id_}): write_comm={write_comm} write_error={write_error}")
        torque, comm, error = bus._read(torque_addr, torque_len, id_, num_retry=RETRIES, raise_on_error=False)
        status, comm2, error2 = bus._read(status_addr, status_len, id_, num_retry=RETRIES, raise_on_error=False)
    except Exception as exc:
        output(f"{motor} (id {id_}): UNKNOWN ({exc})")
        return motor
    output(f"{motor} (id {id_}): torque_raw={torque} read_comm={comm} read_error={error} "
           f"status_raw={status} status_comm={comm2} status_error={error2}")
    if not (bus._is_comm_success(comm) and bus._is_comm_success(comm2)):
        output(f"{motor} (id {id_}): NO REPLY")
        return motor
    output(f"{motor} (id {id_}): torque={'OFF' if torque == 0 else 'ON'} status_flags={describe_flags(status)}")
    return None if torque == 0 else motor


def verify_bus(bus, get_address, *, output=print, samples=VERIFY_SAMPLES,
               interval=VERIFY_INTERVAL_S, sleep=time.sleep):
    """Read every motor repeatedly, without writing away a re-enabled torque flag."""
    if samples < 1 or not math.isfinite(interval) or interval < 0:
        raise ValueError("samples must be positive and interval finite and non-negative")
    addr, size = get_address(bus.model_ctrl_table, MODEL, "Torque_Enable")
    unresolved = set()
    for sample in range(samples):
        if sample:
            sleep(interval)
        for motor in JOINT_NAMES:
            try:
                value, comm, error = bus._read(addr, size, bus.motors[motor].id,
                                               num_retry=RETRIES, raise_on_error=False)
                output(f"verify {sample + 1}/{samples} {motor}: torque_raw={value} "
                       f"comm={comm} packet_error={error}")
                if not bus._is_comm_success(comm) or value != 0:
                    unresolved.add(motor)
            except Exception as exc:
                output(f"verify {sample + 1}/{samples} {motor}: UNKNOWN ({exc})")
                unresolved.add(motor)
    return [motor for motor in JOINT_NAMES if motor in unresolved]


def release_bus(bus, get_address, *, output=print, passes=PASSES, sleep=time.sleep,
                verify_samples=VERIFY_SAMPLES, verify_interval=VERIFY_INTERVAL_S):
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
    unstable = verify_bus(bus, get_address, output=output, samples=verify_samples,
                          interval=verify_interval, sleep=sleep)
    return [motor for motor in JOINT_NAMES if motor in still_on or motor in unstable]


def ensure_port_idle(port):
    """On Linux, refuse a known competing owner before opening a real or pseudo terminal."""
    if sys.platform != "linux":
        return
    device = os.path.realpath(port)
    result = subprocess.run(["fuser", device], capture_output=True, text=True, check=False)
    if result.returncode != 1 or result.stdout.strip() or result.stderr.strip():
        raise RuntimeError(f"Port ownership not clear for {device}: "
                           f"{result.stdout.strip()} {result.stderr.strip()}")


def open_bus(port, *, output=print, attempts=3, interval=1.0, sleep=time.sleep):
    """Open one arm's bus without the handshake.

    ⛔ handshake=False on purpose: a handshake treats a motor with an error bit as absent, which
    is exactly the motor this program exists to release. The retries are for the case this runs
    straight after another process let go of the port, where the tty can take a moment to come
    back.
    """
    from lerobot.motors import Motor, MotorNormMode
    from lerobot.motors.feetech import FeetechMotorsBus

    ensure_port_idle(port)
    bus = FeetechMotorsBus(port=port, motors={
        name: Motor(index, MODEL, MotorNormMode.RANGE_M100_100)
        for index, name in enumerate(JOINT_NAMES, 1)
    })
    for attempt in range(attempts):
        try:
            bus.connect(handshake=False)
            return bus
        except Exception as exc:  # noqa: BLE001 - the last failure is reported by the caller
            if bus.is_connected:
                bus.disconnect(disable_torque=False)
            if attempt + 1 == attempts:
                raise
            output(f"{port}: {exc} — trying again")
            sleep(interval)
    raise AssertionError("unreachable")


def release_port(port, *, output=print, attempts=3, sleep=time.sleep,
                 verify_samples=VERIFY_SAMPLES, verify_interval=VERIFY_INTERVAL_S):
    """Release one arm and close its port. Returns the joints that are still on."""
    from lerobot.motors.motors_bus import get_address

    bus = open_bus(port, output=output, attempts=attempts, sleep=sleep)
    try:
        return release_bus(bus, get_address, output=output, sleep=sleep,
                           verify_samples=verify_samples, verify_interval=verify_interval)
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
    parser.add_argument("--verify-samples", type=int, default=VERIFY_SAMPLES)
    parser.add_argument("--verify-interval", type=float, default=VERIFY_INTERVAL_S,
                        help="seconds between register checks; this is not a physical force measurement")
    args = parser.parse_args(argv)
    if args.verify_samples < 2 or not math.isfinite(args.verify_interval) or args.verify_interval <= 0:
        parser.error("use at least two samples and a finite, positive verification interval")

    try:
        check_lerobot_version()
        still_on = release_port(args.port, verify_samples=args.verify_samples,
                                verify_interval=args.verify_interval)
    except KeyboardInterrupt:
        print("STOP: interrupted; release is unverified. Support the arm and cut motor power.")
        return 130
    except Exception as exc:  # noqa: BLE001 - one message, no traceback, for a course tool
        print(f"STOP: {exc}")
        return 2

    if still_on:
        print(f"FAIL: torque still on or no reply: {', '.join(still_on)} — cut motor power.")
        return 1
    print("REGISTER CHECK: all six motors report Torque_Enable=0 across the verification samples.")
    print("Physical release is NOT verified. If the arm is still stiff, support it and cut motor power.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
