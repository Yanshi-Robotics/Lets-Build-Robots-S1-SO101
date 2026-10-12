"""Offline evidence semantics and refusal checks. Never opens a real serial device."""
from __future__ import annotations

import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import signal
import types
import unittest
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "Teleop"))
import so101_shutdown_debug as debug
from _shutdown_trace import Trace


class DebugTests(unittest.TestCase):
    def setUp(self):
        artifact_root = REPO / "temp/so101-upgrade-20261011"
        artifact_root.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=artifact_root)
        self.directory = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def read(self, name="test"):
        return [json.loads(line) for line in (self.directory / f"{name}.jsonl").read_text().splitlines()]

    def test_failure_returning_zero_is_not_valid_off(self):
        bus = types.SimpleNamespace(port="FAKE", motors={"joint": types.SimpleNamespace(id=1, model="sts3215")},
                                    model_ctrl_table={}, _read=lambda *a, **k: (0, -3001, 0),
                                    _is_comm_success=lambda value: value == 0)
        trace = Trace(self.directory, "test")
        with patch("lerobot.motors.motors_bus.get_address", return_value=(40, 1)):
            trace.snapshot(bus, "fault")
        trace.close()
        snapshot = next(row for row in self.read() if row["event"] == "snapshot")
        torque = snapshot["rows"]["joint"]["Torque_Enable"]
        self.assertEqual(torque["raw"], 0)
        self.assertFalse(torque["valid"])
        self.assertEqual(torque["comm"], -3001)

    def test_device_error_is_preserved_alongside_valid_value(self):
        bus = types.SimpleNamespace(port="FAKE", motors={"joint": types.SimpleNamespace(id=1, model="sts3215")},
                                    model_ctrl_table={}, _read=lambda *a, **k: (0, 0, 32),
                                    _is_comm_success=lambda value: value == 0)
        trace = Trace(self.directory, "test")
        with patch("lerobot.motors.motors_bus.get_address", return_value=(40, 1)):
            trace.snapshot(bus, "fault")
        trace.close()
        torque = next(row for row in self.read() if row["event"] == "snapshot")["rows"]["joint"]["Torque_Enable"]
        self.assertTrue(torque["valid"])
        self.assertEqual(torque["packet_error"], 32)

    def test_ring_retains_last_packets_and_reports_loss(self):
        trace = Trace(self.directory, "test", tail_packets=2)
        trace.running = True
        port = types.SimpleNamespace(port_name="FAKE")
        for motor in (1, 2, 3):
            trace.wire("rx", port, [255, 255, motor, 2, 0, 0], 0)
        trace.close()
        rows = self.read()
        self.assertEqual([row["motor_id"] for row in rows if row["event"] == "packet"], [2, 3])
        self.assertEqual(next(row for row in rows if row["event"] == "packet_tail")["overwritten"], 1)

    def test_error_flushes_tail_without_waiting_for_exit(self):
        trace = Trace(self.directory, "test")
        trace.running = True
        port = types.SimpleNamespace(port_name="FAKE")
        trace.wire("tx", port, [255, 255, 2, 4, 3, 40, 0, 0], 0)
        trace.wire("rx", port, [255, 255, 2, 2, 32, 0], 0)
        rows = self.read()
        self.assertEqual([row["direction"] for row in rows if row["event"] == "packet"], ["tx", "rx"])
        self.assertEqual(rows[-1]["packet_error"], 32)
        trace.close()

    def test_repeated_port_options_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Repeated"):
            debug.options_dict(["--robot.port=A", "--robot.port", "B"])

    def test_missing_fuser_is_unknown_not_idle(self):
        with patch.object(debug.subprocess, "run", side_effect=FileNotFoundError("fuser")):
            self.assertFalse(debug.ownership("/NO-HARDWARE")["idle"])

    def test_busy_port_never_opens_bus(self):
        trace = Trace(self.directory, "test")
        with patch.object(debug, "ownership", return_value={"idle": False, "port": "FAKE"}), \
                patch.object(debug, "load_release") as load:
            result = debug.collect_ports({"follower": "FAKE"}, trace, release_requested=True)
            load.return_value.open_bus.assert_not_called()
        self.assertFalse(result["follower"]["release_attempted"])
        trace.close()

    def test_operator_note_preserves_unicode_and_does_not_load_bus(self):
        debug.save_json(self.directory / "metadata.json", {})
        with contextlib.redirect_stdout(io.StringIO()), patch.object(debug, "load_release") as load:
            code = debug.main(["note", "--run-dir", str(self.directory), "--text", "仍僵硬；DC 未断电"])
            load.assert_not_called()
        self.assertEqual(code, 0)
        item = json.loads((self.directory / "observations.jsonl").read_text())
        self.assertEqual(item["text"], "仍僵硬；DC 未断电")
        self.assertEqual(item["source"], "operator_observation")

    def test_nonfinite_budget_rejected_before_any_io(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            debug.main(["teleop", "--snapshot-budget", "nan"])

    def test_enable_not_repeated_while_configuration_exception_propagates(self):
        from lerobot.motors.feetech import FeetechMotorsBus
        trace = Trace(self.directory, "test")
        with patch.object(FeetechMotorsBus, "enable_torque") as original:
            trace.install(gate_fault_cleanup=True)
            try:
                raise RuntimeError("configuration failed")
            except RuntimeError:
                FeetechMotorsBus.enable_torque(types.SimpleNamespace(port="FAKE"))
            original.assert_not_called()
            trace.close()
        self.assertTrue(any(row["event"] == "enable_suppressed_during_exception" for row in self.read()))

    def test_snapshot_budget_marks_omitted_joints_unknown(self):
        bus = types.SimpleNamespace(port="FAKE", motors={"a": types.SimpleNamespace(id=1, model="sts3215"),
                                                       "b": types.SimpleNamespace(id=2, model="sts3215")},
                                    model_ctrl_table={}, _read=lambda *a, **k: (0, 0, 0),
                                    _is_comm_success=lambda value: value == 0)
        trace = Trace(self.directory, "test", snapshot_budget=1)
        with patch("lerobot.motors.motors_bus.get_address", return_value=(40, 1)), \
                patch("_shutdown_trace.time.monotonic", side_effect=[0, 0, 2]):
            snapshot = trace.snapshot(bus, "budget")
        self.assertFalse(snapshot["complete"])
        self.assertNotIn("b", snapshot["rows"])
        self.assertIn("UNKNOWN", snapshot["omitted"])
        trace.close()

    def test_interrupted_sdk_transaction_is_unknown_without_added_reads(self):
        trace = Trace(self.directory, "test")
        bus = types.SimpleNamespace(port="FAKE", port_handler=types.SimpleNamespace(is_using=True))
        snapshot = trace.snapshot(bus, "after_interrupt")
        self.assertFalse(snapshot["complete"])
        self.assertTrue(snapshot["sdk_is_using"])
        self.assertEqual(snapshot["rows"], {})
        trace.close()

    def test_hung_child_saves_stack_then_escalates_owned_group(self):
        script = self.directory / "hung_child.py"
        ready = self.directory / "ready"
        script.write_text("import signal,time,faulthandler,sys\n"
                          "faulthandler.register(signal.SIGUSR1,all_threads=True)\n"
                          "signal.signal(signal.SIGINT,signal.SIG_IGN)\n"
                          "signal.signal(signal.SIGTERM,signal.SIG_IGN)\n"
                          f"open({str(ready)!r},'w').close()\n"
                          "while True: time.sleep(0.1)\n")
        def request():
            deadline = time.monotonic() + 3
            while not ready.exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            if ready.exists():
                os.kill(os.getpid(), signal.SIGINT)
        thread = threading.Thread(target=request)
        trace = Trace(self.directory, "test")
        thread.start()
        with contextlib.redirect_stdout(io.StringIO()):
            outcome = debug.supervise([sys.executable, str(script)], self.directory, trace,
                                      cleanup_seconds=0.4, terminate_seconds=0.2)
        thread.join(timeout=3)
        trace.close()
        self.assertTrue(outcome["forced"])
        self.assertTrue(outcome["group_gone"])
        self.assertEqual(outcome["code"], -signal.SIGKILL)
        events = self.read()
        self.assertEqual([row["escalation"] for row in events if row["event"] == "cleanup_timeout"],
                         ["SIGTERM", "SIGKILL"])
        self.assertIn("hung_child.py", (self.directory / "terminal.log").read_text())


if __name__ == "__main__":
    unittest.main()
