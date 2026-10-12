#!/usr/bin/env python3
"""Process and register regressions. No test opens a hardware port.

Run with .venv/bin/python -B tests/test_stop_safety.py.
"""
import importlib.util
import os
import pty
from pathlib import Path
import select
import signal
import subprocess
import sys
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "Bringup" / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


release = load("release_regression", "so101_release_torque.py")
wrapper = load("wrapper_regression", "so101_run.py")


class RegisterTests(unittest.TestCase):
    def bus(self):
        return SimpleNamespace(
            model_ctrl_table={},
            motors={name: SimpleNamespace(id=i) for i, name in enumerate(release.JOINT_NAMES, 1)},
            _read=Mock(return_value=(0, 0, 0)), _write=Mock(return_value=(0, 0)),
            _is_comm_success=lambda code: code == 0,
        )

    def verify(self, bus):
        return release.verify_bus(bus, lambda *args: (40, 1), output=lambda _: None,
                                  sleep=lambda _: None)

    def test_all_motors_are_read_in_all_samples(self):
        bus = self.bus()
        self.assertEqual(self.verify(bus), [])
        self.assertEqual([c.args[2] for c in bus._read.call_args_list], list(range(1, 7)) * 3)
        bus._write.assert_not_called()

    def test_transient_reenable_is_not_erased_by_later_off(self):
        bus = self.bus()
        values = [(0, 0, 0)] * 18
        values[6] = (1, 0, 0)
        bus._read.side_effect = values
        self.assertEqual(self.verify(bus), ["shoulder_pan"])
        bus._write.assert_not_called()

    def test_failed_read_returning_zero_is_unknown(self):
        bus = self.bus()
        bus._read.side_effect = [(0, -1, 0)] + [(0, 0, 0)] * 17
        self.assertEqual(self.verify(bus), ["shoulder_pan"])

    def test_one_write_exception_does_not_skip_other_motors(self):
        bus = self.bus()
        bus._write.side_effect = lambda _a, _s, motor, *_args, **_kw: (
            (_ for _ in ()).throw(OSError("transport failed")) if motor == 2 else (0, 0))
        unresolved = release.release_bus(bus, lambda *args: (40, 1), output=lambda _: None,
                                         sleep=lambda _: None)
        self.assertIn("shoulder_lift", unresolved)
        self.assertEqual({c.args[2] for c in bus._write.call_args_list}, set(range(1, 7)))

    def test_linux_busy_port_refuses_open(self):
        with patch.object(release.sys, "platform", "linux"), patch.object(
            release.subprocess, "run", return_value=SimpleNamespace(returncode=0, stdout="123", stderr="")):
            with self.assertRaises(RuntimeError):
                release.ensure_port_idle("test-fixture-not-a-device")

    def test_result_does_not_claim_physical_release(self):
        with patch.object(release, "check_lerobot_version"), patch.object(
            release, "release_port", return_value=[]), patch("builtins.print") as printed:
            self.assertEqual(release.main(["--port", "test-fixture-not-a-device"]), 0)
        self.assertIn("Physical release is NOT verified", str(printed.call_args_list))


@unittest.skipUnless(os.name == "posix", "POSIX process groups")
class ProcessTests(unittest.TestCase):
    def test_real_terminal_ctrl_c_and_child_input(self):
        child_code = ("import time\nprint('PROMPT',flush=True)\n"
                      "print('INPUT',input(),flush=True)\n"
                      "try: time.sleep(60)\n"
                      "except KeyboardInterrupt: print('CLEANUP',flush=True)\n")
        script = ("import sys; sys.path.insert(0," + repr(str(ROOT / "Bringup")) + "); "
                  "from so101_run import run_child; "
                  "print('RESULT',run_child([sys.executable,'-u','-c'," + repr(child_code) +
                  "],cleanup_timeout=0.5,terminate_timeout=0.2),flush=True)")
        pid, fd = pty.fork()
        if pid == 0:
            os.execv(sys.executable, [sys.executable, "-B", "-u", "-c", script])
        output = b""
        def until(marker):
            nonlocal output
            deadline = time.monotonic() + 5
            while marker not in output and time.monotonic() < deadline:
                if select.select([fd], [], [], 0.1)[0]:
                    try:
                        output += os.read(fd, 4096)
                    except OSError:
                        break
            self.assertIn(marker, output)
        try:
            until(b"PROMPT")
            os.write(fd, b"hello\n")
            until(b"INPUT hello")
            os.write(fd, b"\x03")
            until(b"RESULT 0")
            self.assertIn(b"CLEANUP", output)
            _, status = os.waitpid(pid, 0)
            self.assertEqual(os.waitstatus_to_exitcode(status), 0)
        finally:
            try:
                os.kill(pid, signal.SIGTERM)
                os.waitpid(pid, 0)
            except (ProcessLookupError, ChildProcessError):
                pass
            os.close(fd)

    def test_session_duration_is_not_cleanup_timeout(self):
        started = time.monotonic()
        code = wrapper.run_child([sys.executable, "-c", "import time; time.sleep(0.4)"],
                                 cleanup_timeout=0.05, terminate_timeout=0.2)
        self.assertEqual(code, 0)
        self.assertGreaterEqual(time.monotonic() - started, 0.4)

    def run_signalled(self, child_code):
        script = ("import sys; sys.path.insert(0," + repr(str(ROOT / "Bringup")) + "); "
                  "from so101_run import run_child; "
                  "print('RESULT',run_child([sys.executable,'-u','-c'," + repr(child_code) +
                  "],cleanup_timeout=0.2,terminate_timeout=0.2),flush=True)")
        process = subprocess.Popen([sys.executable, "-B", "-u", "-c", script],
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, start_new_session=True)
        child_pid = None
        try:
            self.assertTrue(select.select([process.stdout], [], [], 5)[0])
            line = process.stdout.readline()
            self.assertTrue(line.startswith("READY "), line)
            child_pid = int(line.split()[1])
            process.send_signal(signal.SIGINT)
            output, _ = process.communicate(timeout=5)
            self.assertEqual(process.returncode, 0, output)
            with self.assertRaises(ProcessLookupError):
                os.kill(child_pid, 0)
            return output
        finally:
            if child_pid is not None:
                try:
                    os.killpg(child_pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
            process.stdout.close()

    def test_python_default_sigint_handler_is_preserved(self):
        output = self.run_signalled(
            "import os,signal,time\n"
            "assert signal.getsignal(signal.SIGINT) != signal.SIG_IGN\n"
            "print('READY',os.getpid(),flush=True)\n"
            "try: time.sleep(60)\n"
            "except KeyboardInterrupt: print('CLEANUP',flush=True)\n")
        self.assertIn("CLEANUP", output)
        self.assertIn("RESULT 0", output)

    def test_ignored_signals_escalate_and_child_is_reaped(self):
        output = self.run_signalled(
            "import os,signal,time\n"
            "signal.signal(signal.SIGINT,signal.SIG_IGN)\n"
            "signal.signal(signal.SIGTERM,signal.SIG_IGN)\n"
            "print('READY',os.getpid(),flush=True)\n"
            "time.sleep(60)\n")
        self.assertIn("RESULT -9", output)

    def test_unconfirmed_shutdown_prevents_release(self):
        release_ports = Mock()
        with patch.object(wrapper.release, "check_lerobot_version"):
            code = wrapper.main(["--ports", "test-fixture-not-a-device", "--", "dummy"],
                                run=Mock(side_effect=subprocess.TimeoutExpired("dummy", 1)),
                                release_ports=release_ports, output=lambda _: None)
        self.assertEqual(code, 2)
        release_ports.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
