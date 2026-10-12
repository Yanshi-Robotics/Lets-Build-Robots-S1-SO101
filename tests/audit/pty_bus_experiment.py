#!/usr/bin/env python3
"""What a Ctrl-C shutdown leaves behind, and what the release program then does about it.

    .venv/bin/python tests/audit/pty_bus_experiment.py

This is the evidence behind the "Stopping a session" section of the repository README. It is
read-only, takes a couple of minutes, and touches no hardware: a pty stands in for one arm's
serial port and a fake servo bus answers on the other end of it with real Feetech protocol
packets, so the real `lerobot-teleoperate` and the real `Bringup/so101_release_torque.py` run
against it, unmodified.

One motor is made to answer its torque-disable write with the Overload bit set at the moment the
session ends, which is what happened on 2026-09-16 and again in episode 11. Three things are then
measured, all by reading the fake motors' registers back rather than by trusting a log line:

  1  after `lerobot-teleoperate` is stopped with SIGINT, how many motors still hold torque
  2  after `Bringup/so101_release_torque.py` runs on that same bus, how many still do
  3  the same session under `Bringup/so101_run.py`, which should need neither of the above

It is deliberately not part of `tests/test_bringup.py`: it forks processes and opens ptys, and a
suite that hangs is worse than a suite that does not run. `tests/audit/README.md` says who runs it
and when.
"""
from __future__ import annotations

import errno
import json
import os
import select
import signal
import subprocess
import sys
import tempfile
import threading
import time
import tty
from pathlib import Path

BUS_POLL_INTERVAL_S = 0.02  # Pause while no client owns the emulated serial port.

REPO = Path(__file__).resolve().parents[2]
PYTHON = REPO / ".venv/bin/python"
JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
FAULTY_MOTOR = 2                   # shoulder_lift: the motor that latched Overload on 2026-09-16

TORQUE_ENABLE_ADDR = 40
PRESENT_POSITION_ADDR = 56
FIRMWARE_MAJOR_ADDR = 0
FIRMWARE_MINOR_ADDR = 1
MODEL_NUMBER_ADDR = 3
MODEL_NUMBER = 777
FIRMWARE_VERSION = (3, 10)
MIN_POSITION_LIMIT_ADDR = 9
MAX_POSITION_LIMIT_ADDR = 11
HOMING_OFFSET_ADDR = 31
CALIBRATION = {"homing_offset": 0, "range_min": 0, "range_max": 4095}
ERROR_OVERLOAD = 32
SYNC_WRITE = 0x83


class FakeBus:
    """One end of the pty: speaks the Feetech protocol and keeps the registers it is told about.

    `stopping` marks the moment the session ends. The torque-disable write to `FAULTY_MOTOR` is
    then answered with the Overload bit set and the register stays at 1 — the same answer the real
    arm gave — so every motor after it in the list is never written. The fault is a moment: the
    harness clears `stopping` afterwards, so a later, cleaner attempt gets a clean motor.
    """

    def __init__(self, fd):
        self.fd = fd
        self.lock = threading.RLock()
        self.buffer = bytearray()
        self.stopping = False
        self.status = {motor_id: 0 for motor_id in range(1, 7)}
        self.torque_writes = []        # (motor_id, confirmed) per torque-disable write, in order
        self.registers = {motor_id: {TORQUE_ENABLE_ADDR: 1, PRESENT_POSITION_ADDR: 2048,
                                     FIRMWARE_MAJOR_ADDR: FIRMWARE_VERSION[0],
                                     FIRMWARE_MINOR_ADDR: FIRMWARE_VERSION[1],
                                     MODEL_NUMBER_ADDR: MODEL_NUMBER,
                                     MIN_POSITION_LIMIT_ADDR: CALIBRATION["range_min"],
                                     MAX_POSITION_LIMIT_ADDR: CALIBRATION["range_max"],
                                     HOMING_OFFSET_ADDR: CALIBRATION["homing_offset"]}
                          for motor_id in range(1, 7)}

    def serve_until(self, stop):
        """Answer continuously: lerobot gives a sync_read one second, so the bus has to be live for
        the whole session rather than only while the harness happens to be listening."""
        try:
            while not stop.is_set():
                ready, _, _ = select.select([self.fd], [], [], 0.02)
                with self.lock:
                    if ready:
                        try:
                            self.buffer += os.read(self.fd, 4096)
                        except OSError as exc:
                            if exc.errno != errno.EIO:
                                raise
                            stop.wait(BUS_POLL_INTERVAL_S)
                            continue
                    self.consume()
        except BaseException as exc:  # noqa: BLE001 - a dead bus thread is the thing to see
            print(f"  [bus] SERVER DIED: {type(exc).__name__}: {exc}", flush=True)
            raise

    def consume(self):
        while True:
            start = self.buffer.find(b"\xff\xff")
            if start < 0 or len(self.buffer) < start + 4:
                return
            # Instruction packet: FF FF ID LENGTH INSTRUCTION ... CHECKSUM. LENGTH counts
            # INSTRUCTION, the parameters and CHECKSUM, so the packet ends LENGTH + 1 bytes after
            # the LENGTH byte.
            length = self.buffer[start + 3]
            end = start + 3 + length + 1
            if len(self.buffer) < end:
                return
            packet = bytes(self.buffer[start:end])
            del self.buffer[:end]
            instruction = packet[4]
            if instruction == 1:               # PING: the SDK follows it with a read of register 3
                self.reply(packet[2], b"")
            elif instruction == 2:             # READ
                address, size = packet[5], packet[6]
                value = self.registers.get(packet[2], {}).get(address, 0)
                self.reply(packet[2], value.to_bytes(size, "little"))
            elif instruction == 3:             # WRITE
                self.write_value(packet[2], packet[5], int.from_bytes(packet[6:-1], "little"))
                self.reply(packet[2], b"")
            elif instruction == 0x82:          # SYNC_READ: each motor answers on its own
                address, size = packet[5], packet[6]
                for motor_id in packet[7:-1]:
                    value = self.registers.get(motor_id, {}).get(address, 0)
                    self.reply(motor_id, value.to_bytes(size, "little"))
            elif instruction == SYNC_WRITE:    # SYNC_WRITE: never acknowledged, by protocol
                address, size = packet[5], packet[6]
                body = packet[7:-1]
                for offset in range(0, len(body), size + 1):
                    self.write_value(body[offset], address,
                                     int.from_bytes(body[offset + 1:offset + 1 + size], "little"),
                                     sync=True)

    def write_value(self, motor_id, address, value, *, sync=False):
        if address != TORQUE_ENABLE_ADDR or value != 0:
            self.registers.setdefault(motor_id, {})[address] = value
            return
        if sync:
            # A sync write is a goal the motor never acknowledges, so it proves nothing.
            self.torque_writes.append((motor_id, False))
            return
        if self.stopping and motor_id == FAULTY_MOTOR:
            self.status[motor_id] = ERROR_OVERLOAD
            self.torque_writes.append((motor_id, True))
            return
        self.registers.setdefault(motor_id, {})[TORQUE_ENABLE_ADDR] = 0
        self.status[motor_id] = 0
        self.torque_writes.append((motor_id, True))

    def reply(self, motor_id, payload):
        # Status packet: FF FF ID LENGTH ERROR [payload] CHECKSUM.
        body = bytes([motor_id, len(payload) + 2, self.status.get(motor_id, 0)]) + payload
        checksum = (~sum(body)) & 0xFF
        os.write(self.fd, b"\xff\xff" + body + bytes([checksum]))

    def holding(self):
        """Motors whose torque register is not 0. Read out of the registers, not assumed."""
        with self.lock:
            return sorted(motor_id for motor_id in range(1, 7)
                          if self.registers[motor_id][TORQUE_ENABLE_ADDR] != 0)


class Bench:
    """One pty and the fake bus answering on it."""

    def __init__(self, work):
        self.master, self.slave = os.openpty()
        tty.setraw(self.master)
        tty.setraw(self.slave)
        self.port = os.ttyname(self.slave)
        # The emulator owns only the master; the client must be the sole slave owner.
        os.close(self.slave)
        self.bus = FakeBus(self.master)
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.bus.serve_until, args=(self.stop,), daemon=True)
        self.thread.start()
        self.work = work

    def close(self):
        self.stop.set()
        self.thread.join(timeout=5)
        os.close(self.master)

    def fault_once_then_let_go(self, *, settle=0.4):
        """Answer the next torque-disable write with Overload, then stop faulting.

        The real motor does not refuse forever: it answers the acknowledgement with the error bit
        and is responsive again afterwards. The delay gives the command time to finish dealing with
        that first refusal before the bus becomes clean for whatever runs next.
        """
        def watch():
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                if any(confirmed for _, confirmed in list(self.bus.torque_writes)):
                    time.sleep(settle)
                    self.bus.stopping = False
                    return
                time.sleep(0.01)
        threading.Thread(target=watch, daemon=True).start()

    def run(self, name, command, *, timeout, signal_after=None):
        """Run a command against this bus. Returns (exit code, log path)."""
        log = self.work / f"{name}.log"
        with log.open("a", encoding="utf-8") as sink:
            child = subprocess.Popen(command, cwd=REPO, stdout=sink, stderr=subprocess.STDOUT,
                                     start_new_session=True)
            if signal_after is not None:
                # Fault injection must happen after connect/configure, not during startup.
                # Otherwise a startup exception plus a later Ctrl-C can masquerade as success.
                deadline = time.monotonic() + timeout
                while child.poll() is None and time.monotonic() < deadline:
                    if "SOFollower connected." in log.read_text():
                        break
                    time.sleep(BUS_POLL_INTERVAL_S)
                if "SOFollower connected." not in log.read_text():
                    child.kill()
                    child.wait(timeout=10)
                    return "startup-failed", log
                time.sleep(signal_after)
                self.bus.torque_writes.clear()
                self.bus.stopping = True
                self.fault_once_then_let_go()
                os.killpg(child.pid, signal.SIGINT)
            try:
                code = child.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=10)
                code = "timeout"
        return code, log


def main():
    work = Path(tempfile.mkdtemp(prefix="so101-run-e2e-"))
    calibration = {name: {"id": index, "drive_mode": 0, **CALIBRATION}
                   for index, name in enumerate(JOINTS, 1)}
    for role in ("follower", "leader"):
        (work / role).mkdir(parents=True)
        (work / role / f"{role}.json").write_text(json.dumps(calibration), encoding="utf-8")

    teleoperate = [
        str(PYTHON), "-m", "lerobot.scripts.lerobot_teleoperate",
        "--robot.type=so101_follower", "--robot.id=follower",
        f"--robot.calibration_dir={work / 'follower'}",
        "--teleop.type=so101_leader", "--teleop.id=leader",
        f"--teleop.calibration_dir={work / 'leader'}", "--fps=30",
    ]

    # 1. The session, stopped the way an operator stops it, on a bus where one motor faults.
    bench = Bench(work)
    command = [*teleoperate[:3], f"--robot.port={bench.port}", *teleoperate[3:5],
               f"--teleop.port={bench.port}", *teleoperate[5:]]
    print(f"1. lerobot-teleoperate on {bench.port}, stopped with SIGINT after 3s", flush=True)
    code, log = bench.run("session", command, timeout=60, signal_after=3.0)
    after_session = bench.bus.holding()
    writes = [motor_id for motor_id, ok in bench.bus.torque_writes if ok]
    print(f"   exit code {code}")
    print(f"   confirmed torque-disable writes: {writes}")
    print(f"   motors still holding torque: {after_session or 'none'}")
    print(f"   log: {log}")

    # 2. The release program on the same bus, which is what the operator does next.
    print(f"\n2. Bringup/so101_release_torque.py --port {bench.port}", flush=True)
    release = [str(PYTHON), "Bringup/so101_release_torque.py", f"--port={bench.port}"]
    bench.bus.torque_writes.clear()
    code2, log2 = bench.run("release", release, timeout=60)
    after_release = bench.bus.holding()
    print(f"   exit code {code2}")
    print(f"   motors still holding torque: {after_release or 'none'}")
    print(f"   log: {log2}")
    bench.close()

    # 3. The same session under the wrapper, which should need none of that.
    print("\n3. the same session under Bringup/so101_run.py", flush=True)
    bench = Bench(work)
    wrapped = [str(PYTHON), "Bringup/so101_run.py", "--",
               *teleoperate[:3], f"--robot.port={bench.port}", *teleoperate[3:5],
               f"--teleop.port={bench.port}", *teleoperate[5:]]
    code3, log3 = bench.run("wrapped", wrapped, timeout=120, signal_after=3.0)
    time.sleep(1.0)
    after_wrapper = bench.bus.holding()
    print(f"   exit code {code3}")
    print(f"   motors still holding torque: {after_wrapper or 'none'}")
    print(f"   log: {log3}")
    bench.close()

    print("\n===== verdicts =====")
    checks = {
        "the session's own Ctrl-C shutdown stopped at the faulted motor, so later motors were never written":
            writes != list(range(1, 7)),
        "after that session the arm is still held by motors with torque on":
            after_session != [],
        "the release program clears every one of them":
            code2 == 0 and after_release == [],
        "the same session under the wrapper needs no release afterwards":
            code3 == 0 and after_wrapper == [],
    }
    for label, ok in checks.items():
        print(("  ok   " if ok else "  FAIL ") + label)
    print(f"\n  after the session: {after_session or 'none'} holding")
    print(f"  after the release: {after_release or 'none'} holding")
    print(f"  after the wrapper: {after_wrapper or 'none'} holding")
    print(f"\nlogs and calibration: {work}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
