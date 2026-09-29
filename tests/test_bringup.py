#!/usr/bin/env python3
"""Software-only tests for the Bringup, Teleop and Cameras programs. Fakes are test fixtures, never hardware evidence.

    python tests/test_bringup.py

The Cartesian programs (Lesson 8) carry their own tests: python Cartesian/test_cartesian.py --model-dir models/so101
"""
import ast
import importlib.util
import json
import math
import sys
import tempfile
import time
import unittest
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from unittest.mock import Mock, patch

# The course programs are loaded from their published paths; keep bytecode out of the tree.
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


scan = load("bus_check", ROOT / "Bringup/so101_bus_check.py")
calibrate = load("calibrate_entry", ROOT / "Bringup/so101_calibrate.py")
release = load("release_torque", ROOT / "Bringup/so101_release_torque.py")
teleop = load("teleop_log", ROOT / "Teleop/so101_teleop_log.py")
cameras = load("camera_check", ROOT / "Cameras/so101_camera_check.py")
sys.modules["so101_camera_check"] = cameras
policy_view = load("policy_view", ROOT / "Cameras/so101_policy_view.py")


class BusReadTests(unittest.TestCase):
    def bus(self):
        return SimpleNamespace(port="test-fixture-not-a-device", is_connected=True,
            connect=Mock(), read=Mock(return_value=0),
            sync_read=Mock(return_value={name: 2048 for name in scan.JOINT_NAMES}), disconnect=Mock())

    def run_scan(self, bus):
        scan.check_bus(bus, role="follower", samples=3, interval=0, output=Mock(), sleep=Mock())

    def test_six_motors_and_readonly_disconnect(self):
        bus = self.bus()
        self.run_scan(bus)
        self.assertEqual(bus.sync_read.call_count, 3)
        bus.disconnect.assert_called_once_with(disable_torque=False)

    def test_no_register_writes_in_scan_source(self):
        tree = ast.parse(Path(scan.__file__).read_text(encoding="utf-8"))
        called = {node.func.attr for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
        self.assertFalse(called & {"write", "sync_write", "enable_torque", "disable_torque", "configure", "write_calibration"})

    def test_failed_handshake_closes_open_port(self):
        bus = self.bus()
        bus.connect.side_effect = RuntimeError("handshake failed")
        with self.assertRaises(RuntimeError):
            self.run_scan(bus)
        bus.disconnect.assert_called_once_with(disable_torque=False)

    def test_enabled_torque_is_rejected(self):
        bus = self.bus()
        bus.read.return_value = 1
        with self.assertRaises(RuntimeError):
            self.run_scan(bus)
        bus.sync_read.assert_not_called()

    def test_missing_motor_is_rejected(self):
        bus = self.bus()
        bus.sync_read.return_value = {"shoulder_pan": 10}
        with self.assertRaises(RuntimeError):
            self.run_scan(bus)

    def test_nonfinite_read_is_rejected(self):
        bus = self.bus()
        bus.sync_read.return_value["gripper"] = math.nan
        with self.assertRaises(RuntimeError):
            self.run_scan(bus)


class CalibrationTests(unittest.TestCase):
    def test_official_calibration_without_robot_connect(self):
        events = []
        bus = SimpleNamespace(is_connected=True, connect=lambda: events.append("bus-connect"),
            disable_torque=lambda: events.append("disable"),
            configure_motors=lambda: events.append("configure-bus"),
            disconnect=lambda **kwargs: events.append(("disconnect", kwargs)))
        device = SimpleNamespace(bus=bus, calibrate=lambda: events.append("calibrate"), connect=Mock())
        calibrate.calibrate_device(device)
        self.assertEqual(events[:4], ["bus-connect", "disable", "configure-bus", "calibrate"])
        self.assertEqual(events[-2:], ["disable", ("disconnect", {"disable_torque": False})])
        device.connect.assert_not_called()

    def test_failure_stays_disabled_and_closes(self):
        bus = SimpleNamespace(is_connected=True, connect=Mock(), disable_torque=Mock(), configure_motors=Mock(), disconnect=Mock())
        device = SimpleNamespace(bus=bus, calibrate=Mock(side_effect=RuntimeError("calibration failed")))
        with self.assertRaises(RuntimeError):
            calibrate.calibrate_device(device)
        self.assertEqual(bus.disable_torque.call_count, 2)
        bus.disconnect.assert_called_once_with(disable_torque=False)

    def test_bus_configuration_failure_skips_calibration(self):
        bus = SimpleNamespace(is_connected=True, connect=Mock(), disable_torque=Mock(),
            configure_motors=Mock(side_effect=RuntimeError("configuration failed")), disconnect=Mock())
        device = SimpleNamespace(bus=bus, calibrate=Mock())
        with self.assertRaises(RuntimeError):
            calibrate.calibrate_device(device)
        device.calibrate.assert_not_called()
        self.assertEqual(bus.disable_torque.call_count, 2)
        bus.disconnect.assert_called_once_with(disable_torque=False)


class TeleopLogTests(unittest.TestCase):
    """The teleoperation diagnostics: three read-only checks beside the official tools."""

    def test_only_wrist_roll_folds_across_the_seam(self):
        # Both arms record wrist_roll as a full turn, so its zero sits opposite a seam and two arms
        # 2 deg apart across it would otherwise read 358 deg apart.
        self.assertAlmostEqual(teleop.wrapped_difference(358), -2)
        self.assertAlmostEqual(teleop.wrapped_difference(-358), 2)
        self.assertAlmostEqual(teleop.wrapped_difference(180), -180)  # half a turn: the sign is arbitrary
        self.assertAlmostEqual(teleop.joint_difference("wrist_roll", 179, -179), -2)
        # Every other joint travels well under a turn; folding one would hide a real disagreement.
        self.assertAlmostEqual(teleop.joint_difference("shoulder_pan", 179, -179), 358)
        with self.assertRaises(ValueError):
            teleop.wrapped_difference(math.nan)

    def test_status_bits_name_the_latched_errors(self):
        self.assertEqual(teleop.decode_status(0), [])
        self.assertEqual(teleop.decode_status(1 << 5), ["overload"])
        self.assertEqual(teleop.decode_status((1 << 0) | (1 << 5)), ["voltage", "overload"])

    def test_agreement_judges_only_ticks_where_both_arms_moved(self):
        rising = [float(i) for i in range(20)]
        self.assertEqual(teleop.agreement(rising, rising), 1.0)
        self.assertEqual(teleop.agreement(rising, [-v for v in rising]), 0.0)
        # 2026-09-09: a joint the operator never touched scored 0.00 and was reported as assembled
        # backwards. An arm that stood still has no direction, so there is nothing to disagree with.
        self.assertIsNone(teleop.agreement(rising, [3.0] * 20))
        self.assertIsNone(teleop.agreement([1.0] * 20, rising))
        # Moving them one at a time leaves too few ticks with both in motion to judge.
        alternating = [0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0]
        self.assertIsNone(teleop.agreement(alternating, [0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0]))
        self.assertEqual(teleop.longest_still_run([3.0] * 7 + [4.0, 5.0]), 7)

    def test_diagnostics_never_write_a_motor_register(self):
        # The whole point of registers/compare is that they cannot move an arm. Teleoperation and
        # recording stay with the official tools, so no action is ever sent either.
        tree = ast.parse(Path(teleop.__file__).read_text(encoding="utf-8"))
        called = {node.func.attr for node in ast.walk(tree)
                  if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
        self.assertFalse(called & {"write", "sync_write", "enable_torque", "disable_torque", "configure",
                                   "configure_motors", "write_calibration", "send_action", "calibrate"})
        self.assertIn("sync_read", called)
        disconnects = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                       and isinstance(node.func, ast.Attribute) and node.func.attr == "disconnect"]
        self.assertTrue(disconnects)
        for node in disconnects:  # the default disconnect writes Torque_Enable
            self.assertTrue(any(keyword.arg == "disable_torque" and keyword.value.value is False
                                for keyword in node.keywords))

    def fake_bus(self, powered=False):
        from types import SimpleNamespace
        events = []
        registers = {name: dict.fromkeys(teleop.REGISTERS, 0) for name in teleop.JOINT_NAMES}
        for name, values in registers.items():
            values.update(Torque_Enable=1 if powered else 0, Torque_Limit=1000, Max_Torque_Limit=1000,
                          Present_Voltage=50, Goal_Position=2048, Present_Position=2048,
                          Homing_Offset=-1380, Min_Position_Limit=635, Max_Position_Limit=3375)
        def read(register, name, normalize=True, num_retry=0):
            events.append(("read", register, name))
            return registers[name][register]
        return SimpleNamespace(is_connected=True, connect=lambda: events.append(("connect",)),
                               disconnect=lambda disable_torque=True: events.append(("disconnect", disable_torque)),
                               read=read), registers, events

    def test_registers_refuses_a_powered_arm_before_reading_anything_else(self):
        bus, _, _ = self.fake_bus(powered=True)
        with self.assertRaises(RuntimeError) as raised:
            teleop.refuse_if_powered(bus, "follower")
        self.assertIn("torque is enabled", str(raised.exception))
        released, _, _ = self.fake_bus(powered=False)
        teleop.refuse_if_powered(released, "follower")  # does not raise

    def test_register_dump_reads_every_motor_and_flags_a_mismatched_file(self):
        from unittest.mock import patch
        bus, registers, events = self.fake_bus()
        registers["elbow_flex"]["Homing_Offset"] = 999  # file and motor disagree
        registers["shoulder_lift"]["Status"] = 1 << 5  # latched overload
        registers["wrist_flex"]["Torque_Limit"] = 300  # left low by an interrupted soft start
        calibration = {name: {"homing_offset": -1380, "range_min": 635, "range_max": 3375} for name in teleop.JOINT_NAMES}
        printed = []
        with patch.object(teleop, "say", printed.append):
            rows = teleop.dump_registers(bus, "follower", calibration)
            problems = teleop.summarise_registers(rows, "follower")
        self.assertEqual(set(rows), set(teleop.JOINT_NAMES))
        self.assertFalse([event for event in events if event[0] != "read"])  # reads only
        self.assertTrue(any("MISMATCH" in line for line in printed))
        self.assertTrue(any("overload" in problem for problem in problems))
        self.assertTrue(any("torque limit 300" in problem for problem in problems))

    def test_swapped_ports_are_named_instead_of_twelve_mismatches(self):
        # Linux reassigns serial device names between sessions, so the arm on a port changes.
        # Calibration lives in the motors, so a swap looks like every motor mismatching its own
        # file while matching the other one exactly. Measured on 2026-09-09.
        leader_file = {name: {"homing_offset": -100 - index, "range_min": 800, "range_max": 3500}
                       for index, name in enumerate(teleop.JOINT_NAMES)}
        follower_file = {name: {"homing_offset": -200 - index, "range_min": 600, "range_max": 3300}
                         for index, name in enumerate(teleop.JOINT_NAMES)}

        def dump(source):
            return {name: {"Homing_Offset": values["homing_offset"], "Min_Position_Limit": values["range_min"],
                           "Max_Position_Limit": values["range_max"]} for name, values in source.items()}

        calibrations = {"leader": leader_file, "follower": follower_file}
        crossed = {"leader": dump(follower_file), "follower": dump(leader_file)}
        straight = {"leader": dump(leader_file), "follower": dump(follower_file)}
        self.assertTrue(teleop.swapped_ports(crossed, calibrations))
        self.assertFalse(teleop.swapped_ports(straight, calibrations))
        # One motor genuinely off is a calibration problem, not a swap.
        partial = {"leader": dump(leader_file), "follower": dump(follower_file)}
        partial["follower"]["elbow_flex"]["Homing_Offset"] = 7
        self.assertFalse(teleop.swapped_ports(partial, calibrations))
        self.assertFalse(teleop.swapped_ports({"leader": dump(leader_file)}, calibrations))

    def test_the_register_dump_ends_with_one_of_three_verdicts(self):
        # Jeff, 2026-09-09: the per-motor MATCH lines are for reading afterwards; before running a
        # teleoperation command the operator needs one judgement. Undecided is a real outcome and
        # must never be dressed up as ready.
        leader_file = {name: {"homing_offset": -100 - index, "range_min": 800, "range_max": 3500}
                       for index, name in enumerate(teleop.JOINT_NAMES)}
        follower_file = {name: {"homing_offset": -200 - index, "range_min": 600, "range_max": 3300}
                         for index, name in enumerate(teleop.JOINT_NAMES)}

        def dump(source):
            return {name: {"Homing_Offset": values["homing_offset"], "Min_Position_Limit": values["range_min"],
                           "Max_Position_Limit": values["range_max"]} for name, values in source.items()}

        calibrations = {"leader": leader_file, "follower": follower_file}
        straight = {"leader": dump(leader_file), "follower": dump(follower_file)}
        crossed = {"leader": dump(follower_file), "follower": dump(leader_file)}
        self.assertEqual(teleop.port_verdict(straight, calibrations)[0], "READY")
        self.assertEqual(teleop.port_verdict(crossed, calibrations)[0], "WRONG PORTS")
        # No file to compare against, one arm not read, and a mismatch that is not a swap:
        # all undecided, never READY.
        self.assertEqual(teleop.port_verdict(straight, {"leader": leader_file, "follower": {}})[0],
                         "CANNOT TELL")
        self.assertEqual(teleop.port_verdict({"leader": dump(leader_file)}, calibrations)[0],
                         "CANNOT TELL")
        one_off = {"leader": dump(leader_file), "follower": dump(follower_file)}
        one_off["follower"]["elbow_flex"]["Homing_Offset"] = 7
        verdict, detail = teleop.port_verdict(one_off, calibrations)
        self.assertEqual(verdict, "CANNOT TELL")
        self.assertIn("follower", detail)

    def test_the_verdict_block_is_ruled_off_and_names_the_ports(self):
        ports = {"leader": "/dev/ttyACM1", "follower": "/dev/ttyACM0"}
        for verdict, detail, expected in (
            ("READY", "everything matches", "command can use them"),
            ("WRONG PORTS", "each arm matches the other", "--leader-port /dev/ttyACM0"),
            ("CANNOT TELL", "no file", "Read the lines above"),
        ):
            printed = []
            with patch.object(teleop, "say", printed.append):
                teleop.say_result(verdict, detail, ports)
            text = "\n".join(printed)
            self.assertEqual(text.count(teleop.RESULT_RULE), 2, "the block must be ruled top and bottom")
            self.assertIn(f"RESULT: {verdict}", text)
            self.assertIn("/dev/ttyACM1", text)
            self.assertIn(expected, text)

    def test_a_power_cycled_arm_is_reported_once_not_six_times(self):
        from unittest.mock import patch
        rows = {name: dict.fromkeys(teleop.REGISTERS, 0) for name in teleop.JOINT_NAMES}
        for values in rows.values():
            values.update(Torque_Limit=1000, Max_Torque_Limit=1000, Goal_Position=0, Present_Position=2048)
        with patch.object(teleop, "say", lambda text: None):
            problems = teleop.summarise_registers(rows, "follower")
        self.assertEqual(len(problems), 1)
        self.assertIn("goal of 0", problems[0])
        self.assertIn("toward step 0", problems[0])

    def test_the_stop_reference_finds_the_moment_both_arms_were_against_one_end(self):
        # A joint's zero is the midpoint of its recorded travel, so the two arms are only comparable
        # when both sit against the same physical end. Finding that moment by eye while both hands
        # are on the arms does not work, so it is found in the recording instead.
        ranges = {"leader": (800, 3200), "follower": (900, 3300)}
        # 2026-09-09: this compared degrees against a travel recorded in encoder steps, so every
        # sample looked like it was against the low stop. The reading and the range must share units.
        far = [{"leader_raw": 2000, "follower_raw": 2000, "difference": 0.0}]
        samples = [
            {"leader_raw": 810, "follower_raw": 1500, "difference": 40.0},  # only the leader is at its end
            {"leader_raw": 815, "follower_raw": 915, "difference": 0.4},    # both against the low end
            {"leader_raw": 805, "follower_raw": 905, "difference": 0.2},    # both closer still: this one wins
            {"leader_raw": 3195, "follower_raw": 910, "difference": 90.0},  # opposite ends: not comparable
        ]
        best = teleop.stop_reference(samples, ranges)
        self.assertIsNotNone(best)
        self.assertEqual(best[1], "low")
        self.assertAlmostEqual(best[2], 0.2)
        # Nothing qualifies when neither arm ever reached an end.
        self.assertIsNone(teleop.stop_reference([{"leader_raw": 2000, "follower_raw": 2000, "difference": 0.0}], ranges))
        self.assertIsNone(teleop.stop_reference([], ranges))
        self.assertIsNone(teleop.stop_reference(far, ranges))

    def test_compare_summary_names_a_reversed_joint_and_an_untested_one(self):
        from unittest.mock import patch
        ranges = {name: {"leader": (800, 3200), "follower": (800, 3200)} for name in teleop.JOINT_NAMES}
        history = {}
        for name in teleop.JOINT_NAMES:
            if name == "elbow_flex":  # both moving, reading opposite ways
                history[name] = [{"leader": float(i), "follower": float(-i), "difference": float(2 * i),
                                  "leader_raw": 2000 + i, "follower_raw": 2000 - i} for i in range(30)]
            elif name == "wrist_flex":  # only the leader was touched
                history[name] = [{"leader": float(i), "follower": 5.0, "difference": float(i) - 5,
                                  "leader_raw": 2000 + i, "follower_raw": 2000} for i in range(30)]
            else:
                history[name] = [{"leader": float(i), "follower": float(i) - 1, "difference": 1.0,
                                  "leader_raw": 2000 + i, "follower_raw": 2000 + i} for i in range(30)]
        printed = []
        with patch.object(teleop, "say", printed.append):
            teleop.summarise_compare(history, ranges)
        text = "\n".join(printed)
        self.assertIn("elbow_flex", text)
        self.assertIn("read opposite ways", text)
        self.assertIn("Direction not tested", text)
        self.assertIn("wrist_flex", text)
        # An arm that was never moved must not be called reversed.
        self.assertNotIn("wrist_flex: the two arms read opposite ways", text)

    def write_recording(self, directory, leader, follower, fps=30):
        """A dataset shaped the way lerobot-record writes one."""
        import pandas as pd
        names = [f"{name}.pos" for name in teleop.JOINT_NAMES]
        root = Path(directory)
        (root / "meta").mkdir(parents=True)
        (root / "data" / "chunk-000").mkdir(parents=True)
        (root / "meta" / "info.json").write_text(json.dumps({
            "fps": fps,
            "features": {"action": {"names": names}, "observation.state": {"names": names}},
        }), encoding="utf-8")
        pd.DataFrame({
            "action": [list(row) for row in leader],
            "observation.state": [list(row) for row in follower],
        }).to_parquet(root / "data" / "chunk-000" / "file-000.parquet")
        return root

    def report_lines(self, leader, follower):
        from types import SimpleNamespace
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory:
            root = self.write_recording(directory, leader, follower)
            printed = []
            with patch.object(teleop, "say", printed.append):
                teleop.run_report(SimpleNamespace(dataset=str(root)))
            return printed

    def test_report_grades_a_following_arm_as_healthy(self):
        leader = [[i * 0.5] * 6 for i in range(60)]
        follower = [[i * 0.5 - 1.0] * 6 for i in range(60)]  # one degree behind, same direction
        lines = self.report_lines(leader, follower)
        self.assertTrue(any("followed the Leader in the same direction" in line for line in lines))

    def test_report_names_a_reversed_joint(self):
        leader = [[i * 0.5] * 6 for i in range(60)]
        follower = [[-i * 0.5 if index == 2 else i * 0.5 - 1.0 for index in range(6)] for i in range(60)]
        lines = self.report_lines(leader, follower)
        self.assertTrue(any("elbow_flex" in line and "opposite" in line for line in lines))

    def test_report_names_a_joint_that_never_moved(self):
        leader = [[i * 0.5] * 6 for i in range(90)]
        follower = [[7.0 if index == 1 else i * 0.5 - 1.0 for index in range(6)] for i in range(90)]
        lines = self.report_lines(leader, follower)
        self.assertTrue(any("shoulder_lift" in line and "did not move at all" in line for line in lines))

    def test_a_log_is_written_per_run_and_old_ones_are_pruned(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            with patch.object(teleop, "LOG_DIR", folder), patch.object(teleop, "LOG_KEEP", 3):
                for _ in range(5):
                    teleop.start_log("compare", SimpleNamespace(mode="compare"))
                    time.sleep(0.01)
            for handler in list(teleop.LOG.handlers):
                teleop.LOG.removeHandler(handler)
                handler.close()
            self.assertLessEqual(len(sorted(folder.glob("so101_teleop_log_*.log"))), 3)



class PolicyViewTests(unittest.TestCase):
    """The panels claim to show what a policy receives, so they must reproduce it exactly.

    Every transform here is checked against the function LeRobot itself calls. An
    approximation would be worse than no picture: it would look authoritative while
    teaching the wrong thing about where the padding goes.
    """

    def frames(self):
        import numpy as np

        generator = np.random.default_rng(0)
        for height, width in ((480, 640), (720, 1280), (640, 480)):
            yield generator.integers(0, 255, (height, width, 3), dtype=np.uint8)

    def test_the_pi0_panel_matches_lerobots_centred_padding(self):
        import numpy as np
        import torch
        from lerobot.policies.common.vla_utils import resize_with_pad_torch

        for frame in self.frames():
            mine = policy_view.resized(frame, (224, 224), "centred")
            tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0)
            theirs = resize_with_pad_torch(tensor, 224, 224)[0].permute(1, 2, 0).numpy()
            self.assertEqual(mine.shape, theirs.shape)
            # Compare where the padding is rather than pixel values: the two use different
            # resamplers, but a bar in the wrong place is the failure that matters.
            np.testing.assert_array_equal(mine.sum(axis=(1, 2)) == 0, theirs.sum(axis=(1, 2)) == 0)
            np.testing.assert_array_equal(mine.sum(axis=(0, 2)) == 0, theirs.sum(axis=(0, 2)) == 0)

    def test_the_smolvla_panel_matches_lerobots_top_left_padding(self):
        import numpy as np
        import torch
        from lerobot.policies.common.vla_utils import resize_with_pad

        for frame in self.frames():
            mine = policy_view.resized(frame, (512, 512), "top-left")
            tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float() / 255
            theirs = resize_with_pad(tensor, 512, 512, pad_value=0)[0].permute(1, 2, 0).numpy()
            self.assertEqual(mine.shape, theirs.shape)
            np.testing.assert_array_equal(mine.sum(axis=(1, 2)) == 0, theirs.sum(axis=(1, 2)) == 0)
            np.testing.assert_array_equal(mine.sum(axis=(0, 2)) == 0, theirs.sum(axis=(0, 2)) == 0)

    def test_the_two_conventions_put_the_bars_in_different_places(self):
        import numpy as np

        # If these ever agreed, one of them would be wrong: that difference is the whole
        # reason the page shows SmolVLA next to the pi0 family.
        frame = np.full((480, 640, 3), 255, dtype=np.uint8)
        centred = policy_view.resized(frame, (512, 512), "centred")
        top_left = policy_view.resized(frame, (512, 512), "top-left")
        self.assertTrue((centred[0] == 0).all() and (centred[-1] == 0).all())  # bars on both edges
        self.assertTrue((top_left[0] == 0).all() and (top_left[-1] != 0).any())  # bar only on top

    def test_act_receives_the_frame_untouched(self):
        import numpy as np

        frame = next(self.frames())
        np.testing.assert_array_equal(policy_view.resized(frame, None, None), frame)

    def test_every_listed_policy_is_one_this_program_can_reproduce(self):
        # A policy whose resizing lives in its backbone's own processor cannot be drawn
        # honestly here, so it is named in the footnote instead of guessed at.
        for policy in policy_view.POLICIES:
            self.assertIn(policy["pad"], (None, "centred", "top-left"), policy["id"])
            self.assertTrue(policy["note"].strip(), policy["id"])
        self.assertIn("GR00T", policy_view.NOT_DRAWN)
        listed = {policy["id"] for policy in policy_view.POLICIES}
        self.assertEqual(listed, {"act", "pi0", "pi05", "smolvla"})


class ReleaseTorqueTests(unittest.TestCase):
    """Releasing torque on a bus where a motor answers with an error bit set.

    This is the case LeRobot's own disconnect cannot handle: it disables torque motor by
    motor and raises on the first error byte, leaving everything after it powered.
    """

    class FakeBus:
        """A six-motor bus. `flags` sets a status byte per id; `silent` ids never answer."""

        def __init__(self, flags=None, silent=(), stuck=()):
            self.model_ctrl_table = {}
            self.motors = {name: SimpleNamespace(id=index)
                           for index, name in enumerate(release.JOINT_NAMES, 1)}
            self.flags = flags or {}
            self.silent = set(silent)
            self.stuck = set(stuck)          # ids whose torque never actually goes off
            self.torque = {index: 1 for index in range(1, 7)}
            self.writes = []
            self.disconnected_with = None

        def _write(self, addr, length, id_, value, num_retry=0, raise_on_error=True):
            self.writes.append((id_, value))
            if id_ not in self.stuck:
                self.torque[id_] = value

        def _read(self, addr, length, id_, num_retry=0, raise_on_error=True):
            if id_ in self.silent:
                return 0, 1, 0                       # non-zero comm code = failure
            return (self.flags.get(id_, 0) if addr == "status" else self.torque[id_]), 0, 0

        def _is_comm_success(self, comm):
            return comm == 0

        def disconnect(self, disable_torque=True):
            self.disconnected_with = disable_torque

    @staticmethod
    def get_address(_table, _model, field):
        return ("status" if field == "Status" else "torque"), 1

    def test_every_motor_is_released_even_when_one_reports_overload(self):
        bus = self.FakeBus(flags={2: 32})            # shoulder_lift latched Overload
        lines = []
        still_on = release.release_bus(bus, self.get_address, output=lines.append)
        self.assertEqual(still_on, [])
        # ⛔ The point of the whole program: ids 3-6 are written too, which is what
        # LeRobot's disconnect fails to do once id 2 raises.
        self.assertEqual([id_ for id_, value in bus.writes], [1, 2, 3, 4, 5, 6])
        self.assertTrue(all(value == 0 for _, value in bus.writes))
        self.assertIn("overload", "\n".join(lines))

    def test_a_motor_that_does_not_answer_is_reported_rather_than_assumed_off(self):
        bus = self.FakeBus(silent={4})
        still_on = release.release_bus(bus, self.get_address, output=lambda _line: None)
        self.assertEqual(still_on, ["wrist_flex"])

    def test_a_motor_still_reporting_torque_on_is_not_counted_as_released(self):
        bus = self.FakeBus(stuck={6})
        still_on = release.release_bus(bus, self.get_address, output=lambda _line: None)
        self.assertEqual(still_on, ["gripper"])

    def test_several_error_bits_are_all_named(self):
        self.assertEqual(release.describe_flags(0), "none")
        self.assertEqual(release.describe_flags(1 | 32), "voltage,overload")

    def test_the_source_writes_only_the_torque_register(self):
        """A course tool that claims to touch one register has to be checked, not trusted."""
        source = (ROOT / "Bringup/so101_release_torque.py").read_text(encoding="utf-8")
        writes = [line for line in source.splitlines() if "._write(" in line and not line.strip().startswith("#")]
        self.assertEqual(len(writes), 1)
        self.assertIn("torque_addr", writes[0])


class CameraCheckTests(unittest.TestCase):
    """The camera program's judgements, exercised without a camera attached.

    Everything tested here decides something an operator would otherwise have to notice by
    eye: which path names a camera, and whether two configured streams are fit to record.
    """

    def configured(self, **overrides):
        # ⛔ The two entries must hold different handles. They used to share one, because on
        # Linux the bus map below is what tells the cameras apart and the handle did not
        # matter. Off Linux there is no bus, so verdict() compares handles — and two equal
        # handles are SAME CAMERA, which is correct behaviour that swallowed four rate tests
        # on the macOS and Windows runners. Distinct handles keep each test on its subject.
        entry = {"width": 1280, "height": 720, "fps": 30, "fourcc": "MJPG", "rotation": 0}
        paths = {"top": "/dev/video0", "wrist": "/dev/video2"}
        return {name: {**entry, "path": paths[name], **overrides.get(name, {})} for name in ("top", "wrist")}

    def measured(self, fps=30.0, identical=0, frames=90):
        return {name: {"fps": fps, "identical": identical, "frames": frames,
                       "width": 1280, "height": 720} for name in ("top", "wrist")}

    # preferred_path only ever runs on Linux, because /dev/v4l/by-id links exist nowhere else.
    # ⛔ Its tests use PurePosixPath rather than Path: on a Windows runner Path("/dev/video9")
    # becomes a WindowsPath and str() gives \dev\video9, which fails an assertion about logic
    # that was never wrong. PurePosixPath keeps POSIX semantics on every host, so the Linux
    # behaviour is what gets tested wherever the suite runs.
    def test_by_id_is_preferred_only_while_it_names_one_camera(self):
        links = {"/dev/video4": [("by-id", PurePosixPath("/dev/v4l/by-id/usb-Model-video-index0")),
                                 ("by-path", PurePosixPath("/dev/v4l/by-path/pci-0-usb-0:4.3:1.0-video-index0"))]}
        alone, _ = cameras.preferred_path(PurePosixPath("/dev/video4"), links, model_is_duplicated=False)
        self.assertEqual(alone.parent.name, "by-id")
        # Two cameras of one model: udev keeps a single by-id link and it points at whichever
        # enumerated last, so the link that exists is the wrong thing to write down.
        duplicated, reason = cameras.preferred_path(PurePosixPath("/dev/video4"), links, model_is_duplicated=True)
        self.assertEqual(duplicated.parent.name, "by-path")
        self.assertIn("serial", reason)

    def test_a_camera_with_no_link_at_all_reports_the_bare_number_as_unstable(self):
        path, reason = cameras.preferred_path(PurePosixPath("/dev/video9"), {}, model_is_duplicated=False)
        self.assertEqual(str(path), "/dev/video9")
        self.assertIn("changes when the camera is replugged", reason)

    def test_two_entries_on_one_camera_are_named_before_anything_is_opened(self):
        # One camera entered twice also fails to open the second time. That failure must not
        # be reported as "could not be opened", which sends the operator to the wrong problem.
        verdict, detail = cameras.verdict(self.configured(), {}, {"top": "usb-1-4.4", "wrist": "usb-1-4.4"})
        self.assertEqual(verdict, "SAME CAMERA")
        self.assertIn("usb-1-4.4", detail)

    def test_one_handle_written_twice_is_caught_without_any_bus(self):
        """The macOS and Windows shape: survey_by_index reports no bus at all.

        With nothing but numbers to compare, the same number written twice is the only
        same-camera case still detectable — and it has to be, because that configuration is
        what a reader produces by labelling one panel twice.

        IS_LINUX is patched rather than skipping this on Linux: the branch would otherwise be
        reachable only on a macOS or Windows machine, and a Linux-only change could break it
        without anything saying so until CI ran.
        """
        doubled = self.configured(wrist={"path": "/dev/video0"})
        with patch.object(cameras, "IS_LINUX", False):
            verdict, detail = cameras.verdict(doubled, self.measured(), {"top": None, "wrist": None})
        self.assertEqual(verdict, "SAME CAMERA")
        self.assertIn("/dev/video0", detail)

    def test_on_linux_a_missing_bus_stays_undecided_rather_than_same_camera(self):
        """The other side of that switch, and the reason it is keyed on the platform.

        On Linux a bus of None means the configured path does not exist. Treating that as
        SAME CAMERA would tell the operator to change a path that is merely absent, so the
        handle comparison must not run here even though the handles are equal.
        """
        doubled = self.configured(wrist={"path": "/dev/video0"})
        with patch.object(cameras, "IS_LINUX", True):
            verdict, _ = cameras.verdict(doubled, self.measured(), {"top": None, "wrist": None})
        self.assertNotEqual(verdict, "SAME CAMERA")

    def test_a_frozen_stream_is_not_ready_even_at_full_rate(self):
        results = self.measured()
        results["wrist"] = {**results["wrist"], "identical": 89}
        verdict, detail = cameras.verdict(self.configured(), results,
                                          {"top": "usb-1-4.4", "wrist": "usb-1-4.3"})
        self.assertEqual(verdict, "NOT READY")
        self.assertIn("wrist", detail)

    def test_a_still_workbench_does_not_fail_on_repeated_frames(self):
        # Identical neighbours are normal when nothing moves; only an entirely unchanging
        # buffer means the stream stopped.
        results = self.measured(identical=88)
        verdict, _ = cameras.verdict(self.configured(), results,
                                     {"top": "usb-1-4.4", "wrist": "usb-1-4.3"})
        self.assertEqual(verdict, "READY")

    def test_a_stream_below_its_configured_rate_reports_the_number(self):
        results = self.measured()
        results["top"] = {**results["top"], "fps": 21.4}
        verdict, detail = cameras.verdict(self.configured(), results,
                                          {"top": "usb-1-4.4", "wrist": "usb-1-4.3"})
        self.assertEqual(verdict, "NOT READY")
        self.assertIn("21.4", detail)
        self.assertIn("30", detail)

    def test_an_unopened_camera_is_undecided_rather_than_failed(self):
        results = {"top": self.measured()["top"]}
        verdict, detail = cameras.verdict(self.configured(), results,
                                          {"top": "usb-1-4.4", "wrist": "usb-1-4.3"})
        self.assertEqual(verdict, "CANNOT TELL")
        self.assertIn("wrist", detail)

    def write_config(self, directory, entries):
        file = Path(directory) / "cameras.json"
        file.write_text(json.dumps(entries), encoding="utf-8")
        return file

    def test_the_configuration_must_carry_both_course_view_names(self):
        with tempfile.TemporaryDirectory() as directory:
            file = self.write_config(directory, {"top": {"type": "opencv", "index_or_path": "/dev/video0"}})
            with self.assertRaises(RuntimeError) as raised:
                cameras.load_cameras_file(file)
            self.assertIn("wrist", str(raised.exception))

    def test_a_rotation_lerobot_cannot_express_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            entry = {"type": "opencv", "index_or_path": "/dev/video0", "rotation": 45}
            file = self.write_config(directory, {"top": entry, "wrist": dict(entry)})
            with self.assertRaises(RuntimeError) as raised:
                cameras.load_cameras_file(file)
            self.assertIn("45", str(raised.exception))

    def test_the_four_rotations_are_exactly_lerobots_own(self):
        from lerobot.cameras.configs import Cv2Rotation
        self.assertEqual(set(cameras.ROTATIONS), {rotation.value for rotation in Cv2Rotation})

    def test_a_quarter_turn_swaps_the_dimensions_that_get_written_down(self):
        # LeRobot validates width and height against the frame after rotation, so the pair
        # written into cameras.json has to follow the quarter turns.
        self.assertEqual(cameras.size_for(1280, 720, 0), (1280, 720))
        self.assertEqual(cameras.size_for(1280, 720, 180), (1280, 720))
        self.assertEqual(cameras.size_for(1280, 720, 90), (720, 1280))
        self.assertEqual(cameras.size_for(1280, 720, -90), (720, 1280))
        # Being its own inverse is what lets one helper both read a rotated pair back to
        # sensor order and write it out again.
        self.assertEqual(cameras.size_for(*cameras.size_for(1280, 720, 90), 90), (1280, 720))

    def page_state(self, rotations, *, save_path, roles=("top", "wrist")):
        streams = [SimpleNamespace(rotation=rotation,
                                   camera={"path": f"/dev/video{index}", "card": "cam", "role": role})
                   for index, (rotation, role) in enumerate(zip(rotations, roles))]
        return cameras.PreviewState(streams, save_path=save_path)

    def test_the_page_saves_a_configuration_the_loader_accepts(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "cameras.json"
            state = self.page_state([90, 180], save_path=target)
            state.save()
            self.assertIn("Saved", state.message)
            reloaded = cameras.load_cameras_file(target)
        self.assertEqual(reloaded["top"]["rotation"], 90)
        self.assertEqual(reloaded["wrist"]["rotation"], 180)
        # Derived from the default capture size, so changing that default does not fail this.
        turned = cameras.size_for(cameras.CAPTURE_WIDTH, cameras.CAPTURE_HEIGHT, 90)
        self.assertEqual((reloaded["top"]["width"], reloaded["top"]["height"]), turned)
        self.assertEqual((reloaded["wrist"]["width"], reloaded["wrist"]["height"]),
                         (cameras.CAPTURE_WIDTH, cameras.CAPTURE_HEIGHT))

    def test_a_cache_busting_query_string_still_routes(self):
        # The page appends ?t=<now> when refreshing the stills. A parser that splits the
        # raw path sees "1?t=1757..." instead of "1", and every refresh 404s: the panels
        # go to a broken-image icon while the rest of the page looks fine.
        self.assertEqual(cameras.route_parts("/model/1?t=1757000000000"), ["model", "1"])
        self.assertEqual(cameras.route_parts("/model/1"), ["model", "1"])
        self.assertEqual(cameras.route_parts("/"), [])
        self.assertEqual(cameras.route_parts("/rotate/2/180"), ["rotate", "2", "180"])

    def test_the_check_page_holds_one_connection_per_camera(self):
        """The check page streams the live views and nothing else.

        Every endless MJPEG response consumes one of the browser's few connections to an
        origin. A second stream per camera consumed them all, and a click on a rotation
        button then had none left to travel on: the page looked frozen rather than broken.
        What a policy receives lives in its own program, partly for this reason.
        """
        with tempfile.TemporaryDirectory() as directory:
            state = self.page_state([0, 0], save_path=Path(directory) / "cameras.json")
            page = cameras.build_page(state, heading="h", hint="", rotatable=True, assignable=True).decode()
        self.assertIn('src="/stream/1"', page)
        self.assertIn('src="/stream/2"', page)
        self.assertNotIn("/model/", page)

    def test_switching_size_releases_the_camera_before_opening_it_again(self):
        """A camera cannot be held open twice.

        Opening the new capture before releasing the old one always fails, and the
        failure reads as the camera refusing the size rather than as the bug it is.
        """
        events = []

        class FakeCapture:
            def __init__(self, tag):
                self.tag = tag

            def release(self):
                events.append(f"release:{self.tag}")

            def read(self):
                return False, None

        def fake_open(path, *, width=None, height=None):
            events.append(f"open:{width}x{height}")
            return FakeCapture(f"{width}x{height}")

        stream = cameras.CameraStream({"path": "/dev/video0", "card": "cam"})
        with patch.object(cameras, "open_for_preview", fake_open):
            self.assertTrue(stream.start(640, 480))
            events.clear()
            self.assertTrue(stream.reopen(1280, 720))
            self.assertEqual(events, ["release:640x480", "open:1280x720"])
            self.assertEqual((stream.width, stream.height), (1280, 720))
            stream.close()

    def test_a_refused_size_leaves_the_camera_open_at_the_previous_one(self):
        # Otherwise a size the hardware will not give turns that panel black for good.
        def refusing_open(path, *, width=None, height=None):
            return None if (width, height) == (9999, 9999) else object()

        stream = cameras.CameraStream({"path": "/dev/video0", "card": "cam"})
        stream.width, stream.height = 640, 480
        with patch.object(cameras, "open_for_preview", refusing_open):
            self.assertFalse(stream.reopen(9999, 9999))
        self.assertIsNotNone(stream.capture)
        self.assertEqual((stream.width, stream.height), (640, 480))

    def test_the_saved_file_matches_the_shape_the_lesson_prints(self):
        """The lesson shows this file so it can be typed by hand; the page writes the same shape.

        A reader comparing the two should not have to decide whether a formatting
        difference means a content difference, so the layout is pinned here: one camera
        per line, and the field order the lesson uses.
        """
        entries = {
            name: {"type": "opencv", "index_or_path": f"<{name.upper()}_CAMERA_PATH>",
                   "width": 640, "height": 480, "fps": 30, "fourcc": "MJPG", "rotation": 0}
            for name in ("top", "wrist")
        }
        rendered = cameras.rendered_configuration(entries)
        self.assertEqual(rendered, (
            '{\n'
            '  "top": {"type": "opencv", "index_or_path": "<TOP_CAMERA_PATH>", "width": 640, '
            '"height": 480, "fps": 30, "fourcc": "MJPG", "rotation": 0},\n'
            '  "wrist": {"type": "opencv", "index_or_path": "<WRIST_CAMERA_PATH>", "width": 640, '
            '"height": 480, "fps": 30, "fourcc": "MJPG", "rotation": 0}\n'
            '}\n'))
        self.assertEqual(json.loads(rendered).keys(), entries.keys())

    def test_saving_before_both_views_are_chosen_writes_nothing(self):
        # A half-filled cameras.json would fail later, at recording time, where it is
        # far more expensive to notice.
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "cameras.json"
            state = self.page_state([0, 0], save_path=target, roles=("top", None))
            state.save()
            self.assertFalse(target.exists())
            self.assertIn("wrist", state.message)

    def test_one_role_cannot_sit_on_two_cameras(self):
        with tempfile.TemporaryDirectory() as directory:
            state = self.page_state([0, 0], save_path=Path(directory) / "cameras.json")
            state.assign(2, "top")  # camera 1 held top; it must let go
            self.assertEqual(state.roles, {2: "top"})
            state.assign(1, "wrist")
            self.assertEqual(state.roles, {2: "top", 1: "wrist"})

    def test_a_saved_rotation_does_not_swap_the_sizes_a_second_time_on_reopening(self):
        # The file stores sizes after rotation. Reading them back as if they were sensor
        # order would swap them again on every visit to the page.
        sensor = cameras.size_for(720, 1280, 90)
        self.assertEqual(sensor, (1280, 720))
        self.assertEqual(cameras.size_for(*sensor, 90), (720, 1280))


if __name__ == "__main__":
    unittest.main(argv=[__file__], verbosity=2)
