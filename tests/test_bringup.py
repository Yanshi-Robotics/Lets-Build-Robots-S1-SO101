#!/usr/bin/env python3
"""Software-only tests for the Bringup and IK programs. Fakes are test fixtures, never hardware evidence.

    python tests/test_bringup.py
    python tests/test_bringup.py --model-dir models/so101   # adds the numerical IK checks
"""
import argparse
import ast
import importlib.util
import json
import math
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

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
demo = load("cartesian_demo", ROOT / "IK/so101_cartesian_demo.py")
sys.modules["so101_cartesian_demo"] = demo
visual = load("visual_control", ROOT / "IK/so101_visual_control.py")
teleop = load("teleop_log", ROOT / "Teleop/so101_teleop_log.py")


class VisualControlTests(unittest.TestCase):
    def test_interpolation_endpoints_and_midpoint(self):
        start, goal = [0] * 5, [2, -2, 4, -4, 0]
        self.assertEqual(visual.interpolation(start, goal, 0), start)
        self.assertEqual(visual.interpolation(start, goal, 1), goal)
        self.assertEqual(visual.interpolation(start, goal, 0.5), [1, -1, 2, -2, 0])

    def test_eased_animation_keeps_endpoints_and_order(self):
        self.assertEqual(visual.eased(0), 0)
        self.assertAlmostEqual(visual.eased(1), 1)
        self.assertAlmostEqual(visual.eased(0.5), 0.5)
        samples = [visual.eased(i / 20) for i in range(21)]
        self.assertEqual(samples, sorted(samples))
        for value in (-0.1, 1.1, math.nan):
            with self.assertRaises(ValueError):
                visual.eased(value)

    def test_trail_follows_the_animated_interpolation(self):
        import numpy as np
        def forward(q):
            pose = np.eye(4)
            pose[:3, 3] = np.asarray(q)[:3] / 100  # a fake FK: the first three joints as metres
            return pose
        kinematics = SimpleNamespace(forward_kinematics=forward)
        path = visual.trail_points(kinematics, [0, 0, 0, 0, 0], [10, -20, 30, 0, 0], samples=11)
        self.assertEqual(path.shape, (11, 3))
        np.testing.assert_allclose(path[0], [0, 0, 0])
        np.testing.assert_allclose(path[-1], [0.1, -0.2, 0.3])
        np.testing.assert_allclose(path[5], [0.05, -0.1, 0.15])
        with self.assertRaises(ValueError):
            visual.trail_points(kinematics, [0] * 5, [1] * 5, samples=1)

    def test_model_preview_is_watchable_and_cli_jog_bounds_untouched(self):
        self.assertGreaterEqual(visual.ANIMATION_MIN_SECONDS, 1.0)
        self.assertLessEqual(visual.MOTION_RATE_DEG_S, 15.0)
        self.assertEqual(demo.STEP_MM, 2.0)
        self.assertEqual(demo.SESSION_JOINT_ENVELOPE_DEG, 8.0)

    def test_no_extrapolation_or_nonfinite_interpolation(self):
        for value in (-0.1, 1.1, math.nan):
            with self.assertRaises(ValueError):
                visual.interpolation([0] * 5, [1] * 5, value)
        with self.assertRaises(ValueError):
            visual.interpolation([math.nan] * 5, [1] * 5, 0.5)

    def test_execute_requires_current_single_owner(self):
        request = visual.Request("execute", 1, 10)
        visual.check_request(request, 1, 10.1, {1})
        for owner, now, connected in [(2, 10.1, {1}), (1, 11, {1}), (1, 9, {1}), (1, 10.1, set()), (1, 10.1, {1, 2})]:
            with self.assertRaises(ValueError):
                visual.check_request(request, owner, now, connected)

    def test_plan_is_immutable(self):
        from dataclasses import FrozenInstanceError
        plan = visual.Plan((0,) * 5, (1,) * 5, None, None, (0, 0, 0), 2.0)
        with self.assertRaises(FrozenInstanceError):
            plan.goal = (2,) * 5

    def test_visual_joint_mapping_uses_names_and_radians(self):
        names = ["gripper", *reversed(demo.JOINTS)]
        values = visual.viewer_configuration(names, [0, 30, 60, 90, 180])
        self.assertEqual(values[0], visual.PREVIEW_GRIPPER_RAD)
        self.assertAlmostEqual(values[1], math.pi)
        with self.assertRaises(ValueError):
            visual.viewer_configuration(["wrong"], [0] * 5)

    def test_hardware_arguments_fail_before_run(self):
        from unittest.mock import patch
        with patch.object(visual, "run") as run, patch("sys.stderr"):
            with self.assertRaises(SystemExit):
                visual.main(["--model-dir", "not-a-device", "--web-port", "4602", "--hardware"])
            with self.assertRaises(SystemExit):
                visual.main(["--model-dir", "not-a-device", "--web-port", "4602", "--port", "x"])
            with self.assertRaises(SystemExit):  # interface languages are cn and en only
                visual.main(["--model-dir", "not-a-device", "--web-port", "4602", "--locale", "zh"])
            run.assert_not_called()
            visual.main(["--model-dir", "not-a-device", "--web-port", "4602", "--locale", "cn"])
            self.assertEqual(run.call_args.args[0].locale, "cn")
            visual.main(["--model-dir", "not-a-device", "--web-port", "4602"])
            self.assertEqual(run.call_args.args[0].locale, "cn")

    def test_visual_torque_only_through_the_checked_hold_and_explicit_release(self):
        # Torque is enabled only by control.configure_and_hold (validated snapshot, hold, then enable);
        # the viewer itself never writes registers or calls enable_torque, and it releases torque
        # explicitly before the single read-only disconnect.
        tree = ast.parse(Path(visual.__file__).read_text())
        called = {node.func.attr for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
        self.assertFalse(called & {"write", "sync_write", "enable_torque", "configure", "write_calibration", "connect_robot"})
        self.assertIn("configure_and_hold", called)
        self.assertIn("release_torque", called)  # the only way the viewer switches torque off
        self.assertNotIn("disable_torque", called)
        disconnects = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "disconnect"]
        self.assertEqual(len(disconnects), 1)
        self.assertTrue(any(keyword.arg == "disable_torque" and isinstance(keyword.value, ast.Constant) and keyword.value.value is False for keyword in disconnects[0].keywords))

    def test_motion_duration_respects_joint_and_gripper_rates(self):
        self.assertAlmostEqual(visual.motion_duration([0] * 5, [30, 0, 0, 0, 0]), 30 / visual.MOTION_RATE_DEG_S)
        self.assertAlmostEqual(visual.motion_duration([0] * 5, [0] * 5, gripper_change=50), 50 / visual.GRIPPER_RATE_PCT_S)
        self.assertEqual(visual.motion_duration([0] * 5, [1] * 5, minimum=2.0), 2.0)
        self.assertGreaterEqual(visual.motion_duration([0] * 5, [0] * 5), visual.UPDATE_SECONDS)

    def test_step_based_motion_never_exceeds_one_lerobot_command_clip(self):
        # The eased profile peaks at pi/2 times the average rate; every per-tick joint and
        # gripper change must stay under MAX_JOINT_STEP_DEG, or LeRobot clips and the viewer aborts.
        start, goal = [0.0] * 5, [90.0, -60.0, 45.0, 30.0, 120.0]
        duration = visual.motion_duration(start, goal, gripper_change=100)
        steps = visual.motion_steps(duration)
        previous, previous_grip = start, 0.0
        for k in range(1, steps + 1):
            fraction = visual.eased(k / steps)
            waypoint = visual.interpolation(start, goal, fraction)
            self.assertLess(visual.lag(waypoint, previous), demo.MAX_JOINT_STEP_DEG)
            self.assertLess(abs(100 * fraction - previous_grip), demo.MAX_JOINT_STEP_DEG)
            previous, previous_grip = waypoint, 100 * fraction
        self.assertEqual(visual.motion_steps(0), 1)

    def test_target_limits_intersect_model_and_recorded_travel(self):
        model = {name: (-100, 100) for name in demo.JOINTS}
        recorded = {name: (-105, 105) for name in demo.JOINTS}
        recorded["elbow_flex"] = (-40, 40)
        limits = visual.intersect_limits(model, recorded)
        self.assertEqual(limits["shoulder_lift"], (-100, 100))
        self.assertEqual(limits["elbow_flex"], (-40, 40))

    def test_slider_bounds_round_inwards_and_values_stay_inside(self):
        # A URDF limit of -1.74533 rad is -100.00004°; a slider cannot start outside [min, max].
        limits = {name: (-100.00004285756798, 96.7995) for name in demo.JOINTS}
        bounds = visual.slider_bounds(limits)["shoulder_lift"]
        self.assertEqual(bounds, (-100.0, 96.5))
        self.assertGreaterEqual(bounds[0], limits["shoulder_lift"][0])
        self.assertLessEqual(bounds[1], limits["shoulder_lift"][1])
        self.assertEqual(visual.slider_value(-103.5, bounds), -100.0)
        self.assertEqual(visual.slider_value(-100.00004285756798, bounds), -100.0)
        self.assertEqual(visual.slider_value(12.26, bounds), 12.5)
        self.assertEqual(visual.slider_value(200, bounds), 96.5)

    def test_ik_seed_clamps_a_resting_arm_into_the_model(self):
        limits = {name: (-100.0, 100.0) for name in demo.JOINTS}
        resting = [9.1, -103.5, 96.9, -97.3, 6.5]
        self.assertEqual(list(visual.ik_seed(resting, None, limits)), [9.1, -100.0, 96.9, -97.3, 6.5])
        self.assertEqual(list(visual.ik_seed(resting, (1, 2, 3, 4, 5), limits)), [1, 2, 3, 4, 5])
        self.assertEqual(list(visual.ik_seed([0] * 5, None, limits)), [0] * 5)

    def test_encoder_counts_match_the_calibration_table(self):
        # Follower shoulder_lift recorded 929..3306: the middle reads 0 deg, the rest pose 940 reads -103.5 deg.
        lift = SimpleNamespace(range_min=929, range_max=3306)
        self.assertEqual(visual.counts_from_degrees(0, lift), 2118)
        self.assertEqual(visual.counts_from_degrees(-103.5, lift), 940)
        gripper = SimpleNamespace(range_min=2000, range_max=3539)
        self.assertEqual(visual.counts_from_percent(0, gripper), 2000)
        self.assertEqual(visual.counts_from_percent(100, gripper), 3539)
        self.assertEqual(visual.counts_from_percent(3.6387, gripper), 2056)

    def test_preview_is_faster_than_execution_but_never_instant(self):
        start, goal = [0] * 5, [60, 0, 0, 0, 0]
        self.assertLess(visual.preview_duration(start, goal), visual.motion_duration(start, goal, 0, visual.ANIMATION_MIN_SECONDS))
        self.assertAlmostEqual(visual.preview_duration(start, goal), 60 / visual.PREVIEW_RATE_DEG_S)
        self.assertEqual(visual.preview_duration(start, [1, 0, 0, 0, 0]), visual.PREVIEW_MIN_SECONDS)
        # Hardware speed is set by the servo, not by the preview: the eased peak (pi/2 times the rate)
        # has to stay under the speed max_relative_target still lets the servo reach, or the arm runs
        # against its own cap on every move. Lag fitted to the 2026-09-08 log: 2.04 deg behind at 15.7 deg/s.
        SERVO_LAG_SECONDS_PER_DEG = 0.13
        self.assertLess(visual.MOTION_RATE_DEG_S * math.pi / 2, demo.MAX_JOINT_STEP_DEG / SERVO_LAG_SECONDS_PER_DEG)

    def test_contact_stop_needs_both_position_error_and_load(self):
        self.assertEqual(visual.load_percent(-437), 43.7)  # sign-decoded tenths of a percent
        current, commanded = [0, -60, 30, 0, 0], [0, -55, 30, 0, 0]  # F2 is 5 deg behind
        heavy = [5, 72, 10, 3, 2]
        self.assertEqual(visual.blocked_joints(current, commanded, heavy, 4.0, 60.0), [("shoulder_lift", 5.0, 72.0)])
        self.assertEqual(visual.blocked_joints(current, commanded, [5, 30, 10, 3, 2], 4.0, 60.0), [])  # behind but lightly loaded: gravity, not contact
        self.assertEqual(visual.blocked_joints(commanded, commanded, heavy, 4.0, 60.0), [])  # loaded but following: lifting, not contact
        from unittest.mock import patch
        with patch.object(visual, "run"), patch("sys.stderr"):
            for bad in (["--contact-error-deg", "0"], ["--contact-error-deg", "9"], ["--contact-load-pct", "101"]):
                with self.assertRaises(SystemExit):
                    visual.main(["--model-dir", "not-a-device", "--web-port", "4602", *bad])

    def test_emergency_stop_never_releases_torque(self):
        # The only functions allowed to switch torque off are the explicit release and the exit path.
        tree = ast.parse(Path(visual.__file__).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name in ("emergency_stop", "abort_and_hold", "clear_emergency_stop", "hold_here"):
                calls = {n.func.attr for n in ast.walk(node) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
                self.assertNotIn("disable_torque", calls, node.name)
                self.assertNotIn("release_torque", calls, node.name)
        source = Path(visual.__file__).read_text()
        for phrase in ("紧急停止", "EMERGENCY STOP", "解除紧急停止", "Clear emergency stop", "1 · 上电", "3 · 断电", "Power on", "Power off"):
            self.assertIn(phrase, source)

    def test_settling_ends_by_tolerance_or_by_timeout_never_hangs(self):
        self.assertEqual(visual.settle_state(0.5, 0.0, 0), "done")
        self.assertEqual(visual.settle_state(1.2, 0.0, 0), "wait")
        # The gripper has to arrive too: max_relative_target caps it on its own 0-100 scale, and the
        # loop stops sending the moment settling ends, so a gripper still short would stay short.
        self.assertEqual(visual.settle_state(0.5, 20.0, 0), "wait")
        self.assertEqual(visual.settle_state(0.5, visual.SETTLE_TOLERANCE_PCT, 0), "done")
        ticks_to_timeout = int(visual.SETTLE_TIMEOUT_SECONDS / visual.UPDATE_SECONDS)
        self.assertEqual(visual.settle_state(1.2, 0.0, ticks_to_timeout - 1), "wait")
        self.assertEqual(visual.settle_state(1.2, 0.0, ticks_to_timeout), "timeout")
        self.assertEqual(visual.settle_state(0.5, 20.0, ticks_to_timeout), "timeout")
        self.assertGreaterEqual(visual.REPLAN_TOLERANCE_DEG, 1.0)  # holding sag must not force endless re-planning

    def test_pose_reads_are_retried(self):
        # The follower is read through get_observation, the call lerobot-teleoperate uses; the retry
        # budget goes in through the config, so a single bad packet still cannot end the program.
        source = Path(visual.__file__).read_text()
        self.assertIn("num_read_retries=READ_RETRIES", source)
        self.assertIn("arm.get_observation()", source)
        self.assertNotIn('sync_read("Present_Position"', source)
        self.assertGreaterEqual(visual.READ_RETRIES, 2)

    def test_reading_margin_covers_a_motor_holding_against_a_stop(self):
        # Measured: torque on against the shoulder_lift stop read 1.7 deg past the hand-recorded minimum.
        self.assertGreaterEqual(demo.LIMIT_MARGIN_DEG, 3.0)
        self.assertLess(demo.LIMIT_MARGIN_DEG, 15.0)  # still small next to a wrong-file mismatch
        source = Path(visual.__file__).read_text()
        self.assertIn("except ValueError as exc:", source)  # an out-of-range reading stops and holds, it never exits

    def test_debug_log_is_written_and_old_logs_are_pruned(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            for _ in range(4):
                path = visual.start_log(SimpleNamespace(model_dir="m", web_port=4602, locale="cn", hardware=False), directory=tmp, keep=3)
                visual.LOG.info("probe line")
                time.sleep(1.05)  # distinct timestamps in the file names
            logs = sorted(Path(tmp).glob("so101_visual_control_*.log"))
            self.assertEqual(len(logs), 3)
            self.assertEqual(logs[-1], path)
            text = path.read_text(encoding="utf-8")
            self.assertIn("args {", text)
            self.assertIn("lerobot", text)
            self.assertIn("probe line", text)
            for handler in list(visual.LOG.handlers):
                visual.LOG.removeHandler(handler); handler.close()

    def test_joints_on_a_stop_are_named_and_holds_clamp_inside_the_travel(self):
        limits = {name: (-105.48, 105.48) for name in demo.JOINTS}  # recorded +-100.48 plus the 5 deg reading margin
        rest = [13.98, -103.69, 97.01, -102.29, 6.37]
        blockers = demo.joints_on_a_stop(rest, limits)
        self.assertEqual([b[0] for b in blockers], ["shoulder_lift", "elbow_flex", "wrist_flex"])
        self.assertEqual(blockers[0][2], (-100.48, 100.48))
        self.assertEqual(demo.joints_on_a_stop([0, -60, 60, -30, 0], limits), [])
        self.assertEqual(demo.joints_on_a_stop([0, -95.0, 0, 0, 0], limits), [])  # exactly 5.48 deg inside: allowed
        self.assertEqual(demo.clamp_for_hold(rest, limits), [13.98, -98.48, 97.01, -98.48, 6.37])

    def test_release_torque_keeps_going_past_a_motor_in_protection(self):
        events = []
        def write(register, motor, value, **kwargs):
            events.append(("write", motor, value))
            if motor == "shoulder_lift":
                raise RuntimeError("[RxPacketError] Overload error!")
        def read(register, motor, **kwargs):
            if motor == "shoulder_lift":
                raise RuntimeError("[RxPacketError] Overload error!")
            return 0
        bus = SimpleNamespace(motors={name: object() for name in scan.JOINT_NAMES}, write=write, read=read,
                              sync_write=lambda register, values, **kwargs: events.append(("broadcast", tuple(values))))
        unconfirmed = demo.release_torque(bus)
        self.assertEqual([e[1] for e in events if e[0] == "write"], list(scan.JOINT_NAMES))  # every motor still written
        self.assertEqual(events[-1][0], "broadcast")
        self.assertEqual(list(unconfirmed), ["shoulder_lift"])

    def test_arming_needs_the_exact_word(self):
        self.assertTrue(visual.arming_requested(" ENABLE "))
        for text in ("enable", "", "ENABLE now", "yes"):
            self.assertFalse(visual.arming_requested(text))
        self.assertEqual(visual.lag([0, 0, 0, 0, 0], [1, -3, 2, 0, 0]), 3)


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
        tree = ast.parse(Path(scan.__file__).read_text())
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


class HoldTests(unittest.TestCase):
    def fixture(self, fail_at=None, bad_raw=False):
        events = []
        def record(event):
            events.append(event)
            if event == fail_at:
                raise RuntimeError("injected test failure")
        def read_raw(register, *args, **kwargs):
            if register == "Present_Position":
                record("read-final-position")
                return {name: (4095 if bad_raw and name == "elbow_flex" else 2048) for name in scan.JOINT_NAMES}
            # A still, unloaded arm: no load, no current, each motor at its full torque ceiling.
            return {name: {"Max_Torque_Limit": 1000}.get(register, 0) for name in scan.JOINT_NAMES}
        def read_goal(*args, **kwargs):
            record("confirm-hold")
            return 2048
        def sync_write(register, *args, **kwargs):
            record("hold-target" if register == "Goal_Position" else "sync-" + register)
        bus = SimpleNamespace(
            motors={name: SimpleNamespace(model="sts3215") for name in scan.JOINT_NAMES},
            model_resolution_table={"sts3215": 4096},
            disable_torque=lambda *args, **kwargs: record("disable"), configure_motors=lambda: record("configure"),
            write=lambda register, *args, **kwargs: record("write-" + register), sync_read=read_raw, read=read_goal,
            sync_write=sync_write, enable_torque=lambda *args, **kwargs: record("enable"))
        arm = SimpleNamespace(bus=bus,
            config=SimpleNamespace(position_p_coefficient=16, position_i_coefficient=0, position_d_coefficient=32),
            calibration={name: SimpleNamespace(range_min=1024, range_max=3072) for name in scan.JOINT_NAMES})
        return arm, {name: (-80, 80) for name in demo.JOINTS}, events

    def test_enable_only_after_final_read_and_hold(self):
        from unittest.mock import patch
        arm, limits, events = self.fixture()
        with patch.object(demo.time, "sleep"):
            demo.configure_and_hold(arm, position_mode=0)
        self.assertEqual(events[0], "disable")
        self.assertEqual(events.count("confirm-hold"), 6)
        self.assertLess(events.index("configure"), events.index("read-final-position"))
        self.assertLess(events.index("read-final-position"), events.index("hold-target"))
        self.assertLess(events.index("hold-target"), events.index("confirm-hold"))
        self.assertLess(events.index("confirm-hold"), events.index("enable"))
        # Soft start: the torque limit is lowered before torque comes on, the goal is written once
        # more with torque on, and each motor's own limit is restored last.
        enable = events.index("enable")
        limit_writes = [i for i, event in enumerate(events) if event == "sync-Torque_Limit"]
        self.assertEqual(len(limit_writes), 2)
        self.assertLess(limit_writes[0], enable)
        self.assertIn("hold-target", events[enable + 1:])
        self.assertEqual(events[-1], "sync-Torque_Limit")

    def test_each_configuration_failure_never_enables(self):
        for stage in ("disable", "configure", "write-Operating_Mode", "write-P_Coefficient", "write-Protection_Current", "read-final-position", "hold-target", "confirm-hold"):
            with self.subTest(stage=stage):
                arm, limits, events = self.fixture(fail_at=stage)
                with self.assertRaises(RuntimeError):
                    demo.configure_and_hold(arm, position_mode=0)
                self.assertNotIn("enable", events)

    def test_undelivered_hold_target_never_enables(self):
        arm, limits, events = self.fixture()
        arm.bus.read = Mock(return_value=0)
        with self.assertRaises(RuntimeError):
            demo.configure_and_hold(arm, position_mode=0)
        self.assertNotIn("enable", events)

    def test_post_configuration_position_rechecked(self):
        arm, limits, events = self.fixture(bad_raw=True)
        with self.assertRaises(ValueError):
            demo.configure_and_hold(arm, position_mode=0)
        self.assertNotIn("hold-target", events)
        self.assertNotIn("enable", events)

    def test_calibrated_limits_cover_the_recorded_travel(self):
        # Readings are checked against the arm's own recorded range plus a small margin,
        # so an arm resting on a real mechanical stop is not refused by the URDF limit.
        arm, _, _ = self.fixture()
        arm.calibration["shoulder_pan"] = SimpleNamespace(range_min=2000, range_max=2100)
        low, high = demo.calibrated_limits(arm)["shoulder_pan"]
        self.assertAlmostEqual(high, 100 * 180 / 4095 + demo.LIMIT_MARGIN_DEG)
        self.assertAlmostEqual(low, -high)
        arm.calibration["shoulder_lift"] = SimpleNamespace(range_min=803, range_max=3185)  # real Follower, ±104.7°
        low, high = demo.calibrated_limits(arm)["shoulder_lift"]
        self.assertGreater(high, 100)  # the pinned URDF stops at 100°; the reading at the stop must pass
        demo.validate_joints([0, -(3185 - 803) * 180 / 4095, 0, 0, 0], demo.calibrated_limits(arm))


class BoundaryTests(unittest.TestCase):
    def setUp(self):
        self.limits = {name: (-90, 90) for name in demo.JOINTS}
        self.args = SimpleNamespace(max_joint_step_deg=2, session_joint_envelope_deg=8, session_xyz_envelope_mm=20)

    def test_gripper_must_not_enter_five_joint_solver(self):
        with self.assertRaises(ValueError):
            demo.validate_joints([0] * 6, self.limits)

    def test_nonfinite_and_outside_joint_values(self):
        for value in (math.nan, math.inf, 91):
            with self.assertRaises(ValueError):
                demo.validate_joints([value, 0, 0, 0, 0], self.limits)

    def test_small_step_allowed(self):
        demo.check_step([0]*5, [1]*5, [0]*5, [0, 0, 0.002], [0]*3, self.args)

    def test_large_joint_step_rejected(self):
        with self.assertRaises(ValueError):
            demo.check_step([0]*5, [3]*5, [0]*5, [0]*3, [0]*3, self.args)

    def test_joint_session_envelope_rejected(self):
        with self.assertRaises(ValueError):
            demo.check_step([8]*5, [9]*5, [0]*5, [0]*3, [0]*3, self.args)

    def test_xyz_session_envelope_rejected(self):
        with self.assertRaises(ValueError):
            demo.check_step([0]*5, [1]*5, [0]*5, [0, 0, 0.021], [0]*3, self.args)


class SoftStartTests(unittest.TestCase):
    """configure_and_hold enables torque through a watched soft start, against a fake six-motor bus."""

    NAMES = (*demo.JOINTS, "gripper")

    def make_arm(self, push=(), cured_by_rewrite=True, fault_after_reads=None):
        from types import SimpleNamespace
        names = self.NAMES
        regs = {n: {"Present_Position": 2048, "Goal_Position": 0, "Torque_Enable": 0, "Lock": 0, "Torque_Limit": 1000,
                    "Max_Torque_Limit": 500 if n == "gripper" else 1000, "Present_Load": 0, "Present_Current": 0,
                    "Goal_Position_2": 0} for n in names}
        state = SimpleNamespace(pushing=set(push), log=[], position_reads=0)

        class Bus:
            motors = {n: SimpleNamespace(model="sts3215") for n in names}
            model_resolution_table = {"sts3215": 4096}

            def disable_torque(self, motors=None, num_retry=0):
                for n in names:
                    regs[n]["Torque_Enable"] = 0
                state.log.append(("disable_torque",))

            def enable_torque(self, motors=None, num_retry=0):
                for n in names:
                    regs[n]["Torque_Enable"] = 1
                state.log.append(("enable_torque",))

            def configure_motors(self):
                pass

            def write(self, reg, name, value, normalize=True, num_retry=0):
                regs[name][reg] = value
                state.log.append(("write", reg, name, value))

            def read(self, reg, name, normalize=True, num_retry=0):
                return regs[name][reg]

            def sync_write(self, reg, values, normalize=True, num_retry=0):
                for n, v in values.items():
                    regs[n][reg] = v
                state.log.append(("sync_write", reg, dict(values)))
                if reg == "Goal_Position" and cured_by_rewrite and any(regs[n]["Torque_Enable"] for n in values):
                    state.pushing -= set(values)  # a goal written with torque on is the one the servo follows

            def sync_read(self, reg, normalize=True, num_retry=0):
                if reg == "Present_Position":
                    state.position_reads += 1
                    if fault_after_reads is not None and state.position_reads > fault_after_reads and any(regs[n]["Torque_Enable"] for n in names):
                        raise ConnectionError("bad packet")
                    # A pushing motor with torque on drives 3 counts per read at the capped load.
                    for n in names:
                        if n in state.pushing and regs[n]["Torque_Enable"]:
                            regs[n]["Present_Position"] -= 3
                            regs[n]["Present_Load"] = -regs[n]["Torque_Limit"]  # tenths of a percent, signed
                        else:
                            regs[n]["Present_Load"] = 0
                return {n: regs[n][reg] for n in names}

        calibration = {n: SimpleNamespace(range_min=1024, range_max=3072) for n in names}
        config = SimpleNamespace(position_p_coefficient=16, position_i_coefficient=0, position_d_coefficient=32)
        return SimpleNamespace(bus=Bus(), calibration=calibration, config=config), regs, state

    def run_hold(self, arm):
        from unittest.mock import patch
        with patch.object(demo.time, "sleep"):
            demo.configure_and_hold(arm, 0)

    def test_calm_arm_powers_on_at_low_torque_and_gets_its_own_limit_back(self):
        arm, regs, state = self.make_arm()
        self.run_hold(arm)
        kinds = [(e[0], e[1]) if e[0] == "sync_write" else e for e in state.log]
        lowered = kinds.index(("sync_write", "Torque_Limit"))
        enabled = kinds.index(("enable_torque",))
        goals = [i for i, k in enumerate(kinds) if k == ("sync_write", "Goal_Position")]
        self.assertLess(lowered, enabled)  # the limit is lowered before torque comes on
        self.assertTrue(any(i < enabled for i in goals) and any(i > enabled for i in goals))  # goal written before and again after
        self.assertEqual(state.log[lowered][2]["shoulder_lift"], demo.SOFT_START_TORQUE_LIMIT)
        self.assertTrue(all(regs[n]["Torque_Enable"] == 1 for n in self.NAMES))
        self.assertEqual(regs["shoulder_lift"]["Torque_Limit"], 1000)  # restored to each motor's own ceiling
        self.assertEqual(regs["gripper"]["Torque_Limit"], 500)

    def test_a_joint_that_drives_after_torque_on_is_given_its_goal_again_and_settles(self):
        arm, regs, state = self.make_arm(push={"shoulder_lift"})
        self.run_hold(arm)
        after_enable = state.log[state.log.index(("enable_torque",)) + 1:]
        rewrites = [e for e in after_enable if e[0] == "sync_write" and e[1] == "Goal_Position"]
        self.assertGreaterEqual(len(rewrites), 1)
        self.assertTrue(all(regs[n]["Torque_Enable"] == 1 for n in self.NAMES))
        self.assertEqual(regs["shoulder_lift"]["Goal_Position"], regs["shoulder_lift"]["Present_Position"])
        self.assertEqual(regs["shoulder_lift"]["Torque_Limit"], 1000)

    def test_a_joint_that_keeps_driving_has_torque_released_and_is_named(self):
        arm, regs, state = self.make_arm(push={"shoulder_lift"}, cured_by_rewrite=False)
        with self.assertRaises(demo.SoftStartFailed) as raised:
            self.run_hold(arm)
        self.assertIn("shoulder_lift", str(raised.exception))
        self.assertEqual(raised.exception.unconfirmed, {})
        self.assertTrue(all(regs[n]["Torque_Enable"] == 0 for n in self.NAMES))
        # The limit stays lowered: restoring it while a motor is still driving would be a full-force push.
        self.assertEqual(regs["shoulder_lift"]["Torque_Limit"], demo.SOFT_START_TORQUE_LIMIT)

    def test_a_bus_fault_during_the_watch_releases_torque(self):
        arm, regs, state = self.make_arm(fault_after_reads=2)
        with self.assertRaises(demo.SoftStartFailed) as raised:
            self.run_hold(arm)
        self.assertIn("bus fault", str(raised.exception))
        self.assertTrue(all(regs[n]["Torque_Enable"] == 0 for n in self.NAMES))


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

    def test_agreement_separates_following_reversal_and_stalling(self):
        rising = [float(i) for i in range(20)]
        self.assertEqual(teleop.agreement(rising, rising), 1.0)
        self.assertEqual(teleop.agreement(rising, [-v for v in rising]), 0.0)
        self.assertEqual(teleop.agreement(rising, [3.0] * 20), 0.0)  # stalled: never moves with it
        self.assertIsNone(teleop.agreement([1.0] * 20, rising))  # the leader never moved: nothing to judge
        self.assertEqual(teleop.longest_still_run([3.0] * 7 + [4.0, 5.0]), 7)

    def test_diagnostics_never_write_a_motor_register(self):
        # The whole point of registers/compare is that they cannot move an arm. Teleoperation and
        # recording stay with the official tools, so no action is ever sent either.
        tree = ast.parse(Path(teleop.__file__).read_text())
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



parser = argparse.ArgumentParser()
parser.add_argument("--model-dir", type=Path)
args = parser.parse_args()


@unittest.skipUnless(args.model_dir, "optional numerical check requires a pinned model directory and kinematics dependencies")
class NumericalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kinematics, cls.limits = demo.load_kinematics(args.model_dir)

    def test_six_small_axis_targets(self):
        import numpy as np
        seed = np.array(demo.PREVIEW_JOINTS_DEG)
        for axis in range(3):
            for sign in (-1, 1):
                target = self.kinematics.forward_kinematics(seed)[:3, 3].copy()
                target[axis] += sign * demo.STEP_MM / 1000
                solved, actual, residual = demo.solve_position(self.kinematics, seed, target, self.limits)
                self.assertEqual(len(solved), 5)
                self.assertLessEqual(residual, demo.IK_TOLERANCE_MM)
                self.assertLessEqual(np.linalg.norm(actual - target) * 1000, demo.IK_TOLERANCE_MM)

    def test_unreachable_target_rejected(self):
        with self.assertRaises((ValueError, RuntimeError)):
            demo.solve_position(self.kinematics, demo.PREVIEW_JOINTS_DEG, [1, 1, 1], self.limits)

    def test_ctrl_c_exits_quickly_and_the_port_can_be_reused_at_once(self):
        # The one test that opens a listener: a real viewer on a free loopback port, a client
        # left connected, then SIGINT. It must exit with 130 within seconds, and the port must be
        # bindable immediately even though the closed connection lingers in TIME_WAIT.
        import socket, subprocess, signal, time
        with socket.socket() as probe:
            probe.bind((visual.LOOPBACK, 0))
            port = probe.getsockname()[1]
        process = subprocess.Popen([sys.executable, "-u", str(ROOT / "IK/so101_visual_control.py"), "--model-dir", str(args.model_dir),
                                    "--web-port", str(port)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        try:
            started = time.monotonic()
            while "Open http" not in process.stdout.readline():
                self.assertLess(time.monotonic() - started, 90, "viewer did not start")
            from websockets.sync.client import connect
            client = connect(f"ws://{visual.LOOPBACK}:{port}", open_timeout=10)
            interrupted = time.monotonic()
            process.send_signal(signal.SIGINT)
            self.assertEqual(process.wait(timeout=10), 130)
            self.assertLess(time.monotonic() - interrupted, 5)
            client.close()
            visual.validate_web_port(port)  # would raise "Address already in use" without SO_REUSEADDR
        finally:
            if process.poll() is None:
                process.kill()
            process.stdout.close()

    def test_wrist_roll_stays_put_while_dragging(self):
        import numpy as np
        rest = np.array([13.98, -103.69, 97.01, -102.29, 6.37])
        seed = visual.ik_seed(rest, None, self.limits)
        demo.lock_wrist_roll(self.kinematics, rest[4])
        goal = seed.copy()
        for dx, dy, dz in ((0.02, 0, 0.05), (0.03, 0.01, 0.05), (0.03, 0.02, 0.06), (0.05, 0.02, 0.08)):
            target = self.kinematics.forward_kinematics(seed)[:3, 3] + np.array([dx, dy, dz])
            goal, _, _ = demo.solve_position(self.kinematics, goal, target, self.limits)
            self.assertLess(abs(goal[4] - rest[4]), 2.0)  # 2026-09-08: without the lock a drag planned a 155 deg roll

    def test_viser_urdf_meshes_and_fk_match_the_solver(self):
        # Real Viser/yourdfpy loading; only scene transport is a test fixture.
        # No listener, browser, serial port, or hardware is opened by this test.
        import numpy as np
        import yourdfpy
        from functools import partial
        from viser.extras import ViserUrdf
        model_path = args.model_dir / demo.URDF_NAME
        model = yourdfpy.URDF.load(model_path, filename_handler=partial(yourdfpy.filename_handler_magic, dir=model_path.parent))
        scene = SimpleNamespace(add_frame=Mock(side_effect=lambda *a, **kw: SimpleNamespace(**kw)), add_mesh_simple=Mock())
        viewer = ViserUrdf(SimpleNamespace(scene=scene), model, root_node_name="/test", mesh_color_override=visual.CURRENT_COLOR)
        self.assertGreater(scene.add_mesh_simple.call_count, 0)
        for call in scene.add_mesh_simple.call_args_list:
            self.assertGreater(len(call.args[1]), 0)
            self.assertGreater(len(call.args[2]), 0)
            self.assertTrue(np.isfinite(call.args[1]).all())
        for seed in (demo.PREVIEW_JOINTS_DEG, [10, -25, 55, -25, 10]):
            viewer.update_cfg(np.array(visual.viewer_configuration(viewer.get_actuated_joint_names(), seed)))
            np.testing.assert_allclose(model.get_transform("gripper_frame_link"), self.kinematics.forward_kinematics(seed), atol=1e-8)


if __name__ == "__main__":
    unittest.main(argv=[__file__], verbosity=2)
