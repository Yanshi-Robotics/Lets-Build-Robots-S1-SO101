#!/usr/bin/env python3
"""Run actual LeRobot on two emulated buses; kernel terminal Ctrl+C, no real hardware.

Created: 2026-10-11. Task: validate shutdown-debug evidence and stop policy.
Artifacts: temp/so101-upgrade-20261011/debug-tests/; retain until the debug version is superseded.
"""
from __future__ import annotations

import errno
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import time
import uuid

from pty_ctrl_c_terminal import Bench, JOINTS, CALIBRATION, TORQUE_ENABLE_ADDR

REPO = Path(__file__).resolve().parents[2]
PYTHON = REPO / ".venv/bin/python"
SCRIPT = REPO / "Teleop/so101_shutdown_debug.py"
ROOT = REPO / "temp/so101-upgrade-20261011/debug-tests" / uuid.uuid4().hex[:8]


def read_events(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def terminal_run(command, *, started=None, press_ctrl_c=False, timeout=90):
    # Establish the controlling tty in a fresh interpreter, avoiding forkpty in a threaded host.
    master, slave = os.openpty()
    helper = "import fcntl,termios,os,sys; fcntl.ioctl(0,termios.TIOCSCTTY,0); os.execv(sys.argv[1],sys.argv[1:])"
    process = subprocess.Popen([str(PYTHON), "-c", helper, str(PYTHON), "-u", *map(str, command)],
                               stdin=slave, stdout=slave, stderr=slave, cwd=REPO, start_new_session=True)
    os.close(slave)
    os.set_blocking(master, False)
    output = bytearray()
    deadline = time.monotonic() + timeout
    triggered = False
    try:
        while time.monotonic() < deadline:
            if not triggered and started and started():
                triggered = True
                if press_ctrl_c:
                    os.write(master, b"\x03")  # The kernel signals the foreground group.
            ready, _, _ = select.select([master], [], [], 0.05)
            if ready:
                try:
                    data = os.read(master, 65536)
                except OSError as exc:
                    if exc.errno == errno.EIO:
                        break
                    raise
                if data:
                    output += data
                else:
                    break
            if process.poll() is not None:
                break
        # EIO at tty close can precede waitpid readiness by a few milliseconds.
        try:
            code = process.wait(timeout=max(0.1, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            raise AssertionError("diagnostic PTY run did not terminate within test deadline")
        return code, output.decode("utf-8", "replace")
    finally:
        os.close(master)


def one_case(case):
    work = ROOT / case
    work.mkdir(parents=True)
    for role in ("leader", "follower"):
        (work / role).mkdir()
        (work / role / f"{role}.json").write_text(json.dumps({name: {"id": index, "drive_mode": 0, **CALIBRATION}
            for index, name in enumerate(JOINTS, 1)}))
    follower, leader = Bench(work), Bench(work)
    original_reply = follower.bus.reply
    try:
        command = [SCRIPT, "teleop", "--output-root", work / "logs", "--", "--robot.type=so101_follower",
                   f"--robot.port={follower.port}", "--robot.id=follower",
                   f"--robot.calibration_dir={work / 'follower'}", "--teleop.type=so101_leader",
                   f"--teleop.port={leader.port}", "--teleop.id=leader",
                   f"--teleop.calibration_dir={work / 'leader'}", "--fps=30"]
        if case == "normal":
            command.append("--teleop_time_s=0.2")
        def started():
            paths = list((work / "logs").glob("*/child.jsonl"))
            if not paths or not any(row["event"] == "loop_begin" for row in read_events(paths[0])):
                return False
            if case == "ctrl_c_cleanup_fault":
                # The leader disconnect fails first; upstream would otherwise skip the follower.
                leader.bus.stopping = True
                leader.bus.let_go_after_the_first_refusal(settle=0.05)
            if case == "runtime_fault":
                # LeRobot sync_read ignores the servo error byte. Drop one reply to provoke
                # a real upstream communication exception, instead of assuming bit 32 raises.
                def drop_one_motor(motor_id, payload):
                    if motor_id != 2:
                        original_reply(motor_id, payload)
                follower.bus.reply = drop_one_motor
            return True
        code, terminal = terminal_run(command, started=started, press_ctrl_c=case.startswith("ctrl_c"))
        (work / "terminal-test.log").write_text(terminal)
        run = next((work / "logs").iterdir())
        summary = json.loads((run / "summary.json").read_text())
        child = read_events(run / "child.jsonl")
        parent = read_events(run / "supervisor.jsonl")
        assert summary["physical_release"] == "UNVERIFIED", summary
        assert any(row["event"] == "snapshot" and row["stage"] == "configured_before_loop" for row in child)
        if case in ("normal", "ctrl_c_cleanup_fault"):
            assert summary["auto_release"], summary
            assert follower.bus.holding() == [] and leader.bus.holding() == []
        if case == "normal":
            assert code == 0 and summary["reason"] == "normal", summary
        if case == "ctrl_c_cleanup_fault":
            assert summary["reason"] == "ctrl_c" and code != 0, summary  # Preserve the cleanup exception.
            assert len(summary["child"]["requests"]) == 1, summary
            assert any(row["event"] == "bus_disconnect_error" for row in child)
            assert any(row["event"] == "packet" and row.get("packet_error") == 32
                       and row["motor_id"] == 2 for row in child)
            # Robot.__del__ may retry the follower cleanup after the leader exception. The
            # log must expose this extra path instead of assuming every later motor stayed ON.
            before = next(row for row in child if row["event"] == "snapshot"
                          and row["stage"] == "before_official_disconnect" and row["port"] == follower.port)
            # Ctrl+C can also leave the SDK's local is_using flag set. Such a snapshot must
            # be UNKNOWN, never six apparently successful zero values.
            assert (not before["complete"] and before.get("sdk_is_using")) or all(
                value["Torque_Enable"]["valid"] and value["Torque_Enable"]["raw"] == 1
                for value in before["rows"].values())
            assert any(row["event"] == "bus_disconnect_begin"
                       and any("__del__" in frame for frame in row["caller"]) for row in child)
        if case == "runtime_fault":
            assert summary["reason"] == "runtime_fault" and not summary["auto_release"] and code != 0, summary
            assert follower.bus.holding() == list(range(1, 7)), follower.bus.holding()
            follower.bus.reply = original_reply  # Restore communication before explicit manual release.
            manual_code, manual_terminal = terminal_run([SCRIPT, "release", "--from-run", run,
                                                         "--output-root", work / "manual"])
            (work / "manual-test.log").write_text(manual_terminal)
            assert manual_code == 0 and follower.bus.holding() == [], manual_terminal
            manual_run = next((work / "manual").iterdir())
            manual_events = read_events(manual_run / "supervisor.jsonl")
            writes = [bytes.fromhex(row["hex"]) for row in manual_events
                      if row["event"] == "packet" and row.get("instruction") == 3]
            assert writes and all(packet[5] == TORQUE_ENABLE_ADDR and packet[6] == 0 for packet in writes)
        return {"case": case, "exit_code": code, "stop_reason": summary["reason"], "evidence": str(run)}
    finally:
        follower.close()
        leader.close()


def main():
    ROOT.mkdir(parents=True)
    results = []
    for case in ("normal", "ctrl_c_cleanup_fault", "runtime_fault"):
        print(f"Testing {case} on emulated ports", flush=True)
        results.append(one_case(case))
        print(f"PASS: {case}", flush=True)
    (ROOT / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
