#!/usr/bin/env python3
"""Ctrl-C in a real terminal: does the release happen by itself?

    .venv/bin/python tests/audit/pty_ctrl_c_terminal.py

This is the check `tests/audit/pty_bus_experiment.py` cannot make. There the harness sent SIGINT
with os.killpg; here the byte 0x03 is written to the master side of a pty, so the kernel's terminal
driver delivers SIGINT to the foreground process group exactly as a keyboard Ctrl-C does, to the
wrapper and to its child both.

The child is a stand-in for `lerobot-teleoperate` that behaves the way that program does on the way
out: it installs a SIGINT handler which calls LeRobot's real `FeetechMotorsBus.disconnect()`. One
motor answers that call with the Overload bit set, which raises — the same failure as on the real
arm — and the child dies with a traceback and a non-zero exit code, as the real one does.

Two runs of the same child, three seconds each, on a fake Feetech bus behind a pty:

  bare      the child on its own
  wrapped   the child under Bringup/so101_run.py

What is measured afterwards is the fake motors' torque registers, read back. Read-only, no
hardware; `tests/audit/README.md` says who runs it and how often.
"""
from __future__ import annotations

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

# The stand-in for lerobot-teleoperate. It connects with the handshake, turns torque on, waits for
# Ctrl-C, and then disconnects through LeRobot's own code path so the failure under test is the
# real one rather than a re-implementation of it.
CHILD = r'''
import sys, time
from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus

port = next(a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--robot.port="))
motors = {name: Motor(index, "sts3215", MotorNormMode.RANGE_M100_100)
          for index, name in enumerate(
              ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"), 1)}
bus = FeetechMotorsBus(port=port, motors=motors)
bus.connect()
for name in motors:
    bus.write("Torque_Enable", name, 1)
print("child: torque on, teleoperating", flush=True)

def bye(*_):
    print("child: Ctrl-C, disconnecting the way lerobot-teleoperate does", flush=True)
    try:
        bus.disconnect()          # LeRobot's own shutdown, unmodified
    except Exception as exc:
        print(f"child: disconnect raised {type(exc).__name__}: {exc}", flush=True)
    print("child: exiting", flush=True)
    raise SystemExit(1)

import signal
signal.signal(signal.SIGINT, bye)
while True:
    time.sleep(0.1)
'''


class FakeBus:
    """One end of the pty: speaks the Feetech protocol and keeps the registers it is told about."""

    def __init__(self, fd):
        self.fd = fd
        self.lock = threading.RLock()
        self.buffer = bytearray()
        self.stopping = False
        self.faulted = False
        self.status = {motor_id: 0 for motor_id in range(1, 7)}
        self.registers = {motor_id: {TORQUE_ENABLE_ADDR: 0, PRESENT_POSITION_ADDR: 2048,
                                     FIRMWARE_MAJOR_ADDR: FIRMWARE_VERSION[0],
                                     FIRMWARE_MINOR_ADDR: FIRMWARE_VERSION[1],
                                     MODEL_NUMBER_ADDR: MODEL_NUMBER,
                                     MIN_POSITION_LIMIT_ADDR: CALIBRATION["range_min"],
                                     MAX_POSITION_LIMIT_ADDR: CALIBRATION["range_max"],
                                     HOMING_OFFSET_ADDR: CALIBRATION["homing_offset"]}
                          for motor_id in range(1, 7)}

    def serve_until(self, stop):
        while not stop.is_set():
            ready, _, _ = select.select([self.fd], [], [], 0.02)
            with self.lock:
                if ready:
                    self.buffer += os.read(self.fd, 4096)
                self.consume()

    def consume(self):
        while True:
            start = self.buffer.find(b"\xff\xff")
            if start < 0 or len(self.buffer) < start + 4:
                return
            length = self.buffer[start + 3]
            end = start + 3 + length + 1
            if len(self.buffer) < end:
                return
            packet = bytes(self.buffer[start:end])
            del self.buffer[:end]
            instruction = packet[4]
            if instruction == 1:
                self.reply(packet[2], b"")
            elif instruction == 2:
                address, size = packet[5], packet[6]
                value = self.registers.get(packet[2], {}).get(address, 0)
                self.reply(packet[2], value.to_bytes(size, "little"))
            elif instruction == 3:
                self.write_value(packet[2], packet[5], int.from_bytes(packet[6:-1], "little"))
                self.reply(packet[2], b"")
            elif instruction == 0x82:
                address, size = packet[5], packet[6]
                for motor_id in packet[7:-1]:
                    value = self.registers.get(motor_id, {}).get(address, 0)
                    self.reply(motor_id, value.to_bytes(size, "little"))
            elif instruction == SYNC_WRITE:
                address, size = packet[5], packet[6]
                body = packet[7:-1]
                for offset in range(0, len(body), size + 1):
                    self.write_value(body[offset], address,
                                     int.from_bytes(body[offset + 1:offset + 1 + size], "little"),
                                     sync=True)

    def write_value(self, motor_id, address, value, *, sync=False):
        if address != TORQUE_ENABLE_ADDR:
            self.registers.setdefault(motor_id, {})[address] = value
            return
        if value == 1:
            self.registers.setdefault(motor_id, {})[TORQUE_ENABLE_ADDR] = 1
            return
        if sync:
            return
        if self.stopping and motor_id == FAULTY_MOTOR:
            # The real motor refuses the acknowledgement once and is responsive again afterwards;
            # a bus that refused forever would break the wrapper's release too, which is not what
            # the arm does. The fault is cleared in a moment, from the harness.
            self.status[motor_id] = ERROR_OVERLOAD
            self.faulted = True
            return
        self.registers.setdefault(motor_id, {})[TORQUE_ENABLE_ADDR] = 0
        self.status[motor_id] = 0

    def reply(self, motor_id, payload):
        body = bytes([motor_id, len(payload) + 2, self.status.get(motor_id, 0)]) + payload
        checksum = (~sum(body)) & 0xFF
        os.write(self.fd, b"\xff\xff" + body + bytes([checksum]))

    def let_go_after_the_first_refusal(self, *, settle=0.6):
        def watch():
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                if self.faulted:
                    time.sleep(settle)
                    self.stopping = False
                    return
                time.sleep(0.01)
        threading.Thread(target=watch, daemon=True).start()

    def holding(self):
        with self.lock:
            return sorted(motor_id for motor_id in range(1, 7)
                          if self.registers[motor_id][TORQUE_ENABLE_ADDR] != 0)


class Bench:
    def __init__(self, work):
        self.master, self.slave = os.openpty()
        tty.setraw(self.master)
        tty.setraw(self.slave)
        self.port = os.ttyname(self.slave)
        self.bus = FakeBus(self.master)
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.bus.serve_until, args=(self.stop,), daemon=True)
        self.thread.start()
        self.script = work / "fake_teleoperate.py"
        self.script.write_text(CHILD, encoding="utf-8")

    def close(self):
        self.stop.set()
        self.thread.join(timeout=5)
        os.close(self.slave)
        os.close(self.master)

    def ctrl_c_session(self, wrapper, *, seconds=3.0):
        """Run the child on a controlling tty and press Ctrl-C. Returns (output, exit code)."""
        command = [str(PYTHON), str(self.script), f"--robot.port={self.port}"]
        if wrapper:
            command = [str(PYTHON), "Bringup/so101_run.py", "--", *command]
        pid, master = os.forkpty()
        if pid == 0:                                   # child: the session leader on the new tty
            os.chdir(REPO)
            os.execv(command[0], command)
            os._exit(127)
        os.set_blocking(master, False)
        time.sleep(seconds)
        os.write(master, b"\x03")                      # the Ctrl-C byte: the tty does the rest
        output = bytearray()
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            ready, _, _ = select.select([master], [], [], 0.2)
            if ready:
                try:
                    chunk = os.read(master, 65536)
                except OSError:
                    break
                if not chunk:
                    break
                output += chunk
        try:
            _, status = os.waitpid(pid, 0)
            code = os.waitstatus_to_exitcode(status)
        except ChildProcessError:
            code = "gone"
        # Drain whatever is still buffered, then let go of the tty.
        for _ in range(20):
            ready, _, _ = select.select([master], [], [], 0.05)
            if not ready:
                break
            try:
                output += os.read(master, 65536)
            except OSError:
                break
        os.close(master)
        time.sleep(0.5)
        return output.decode("utf-8", "replace"), code


def main():
    work = Path(tempfile.mkdtemp(prefix="so101-ctrl-c-"))
    results = {}
    for label, wrapper in (("bare", False), ("wrapped", True)):
        bench = Bench(work)
        bench.bus.stopping = True                      # the motor faults when the session ends
        bench.bus.let_go_after_the_first_refusal()
        print(f"\n===== {label}: the child {'under so101_run.py' if wrapper else 'on its own'} "
              f"on {bench.port} =====", flush=True)
        output, code = bench.ctrl_c_session(wrapper)
        time.sleep(0.5)
        holding = bench.bus.holding()
        results[label] = {"code": code, "holding": holding}
        print(f"--- what the terminal showed ({label}) ---")
        print("\n".join(line for line in output.splitlines() if line.strip()))
        print(f"--- exit code {code}; motors still holding torque: {holding or 'none'} ---")
        bench.close()

    print("\n===== verdicts =====")
    checks = {
        "the child's own Ctrl-C shutdown left motors holding torque":
            results["bare"]["holding"] != [],
        "under the wrapper, Ctrl-C released every motor with no extra command":
            results["wrapped"]["holding"] == [],
        "and the wrapper said so (exit code 0)":
            results["wrapped"]["code"] == 0,
    }
    for label, ok in checks.items():
        print(("  ok   " if ok else "  FAIL ") + label)
    print(f"\n  bare    after Ctrl-C: {results['bare']['holding'] or 'none'} holding")
    print(f"  wrapped after Ctrl-C: {results['wrapped']['holding'] or 'none'} holding")
    print(f"\nwork dir: {work}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
