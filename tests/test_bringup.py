#!/usr/bin/env python3
"""Software-only tests for the Bringup and IK programs. Fakes are test fixtures, never hardware evidence.

    python tests/test_bringup.py
    python tests/test_bringup.py --model-dir models/so101   # adds the numerical IK checks
"""
import argparse
import ast
import importlib.util
import math
import sys
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
        self.assertIn("disable_torque", called)
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
        self.assertEqual(visual.MOTION_RATE_DEG_S, 10.0)  # Hardware speed is untouched by the preview speed.

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
            if isinstance(node, ast.FunctionDef) and node.name in ("emergency_stop", "abort_and_hold", "clear_emergency_stop"):
                calls = {n.func.attr for n in ast.walk(node) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
                self.assertNotIn("disable_torque", calls, node.name)
        for phrase in ("紧急停止", "EMERGENCY STOP", "解除紧急停止", "Clear emergency stop"):
            self.assertIn(phrase, Path(visual.__file__).read_text())

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
        def read_raw(*args, **kwargs):
            record("read-final-position")
            return {name: (4095 if bad_raw and name == "elbow_flex" else 2048) for name in scan.JOINT_NAMES}
        def read_goal(*args, **kwargs):
            record("confirm-hold")
            return 2048
        bus = SimpleNamespace(
            motors={name: SimpleNamespace(model="sts3215") for name in scan.JOINT_NAMES},
            model_resolution_table={"sts3215": 4096},
            disable_torque=lambda: record("disable"), configure_motors=lambda: record("configure"),
            write=lambda register, *args: record("write-" + register), sync_read=read_raw, read=read_goal,
            sync_write=lambda *args, **kwargs: record("hold-target"), enable_torque=lambda: record("enable"))
        arm = SimpleNamespace(bus=bus,
            config=SimpleNamespace(position_p_coefficient=16, position_i_coefficient=0, position_d_coefficient=32),
            calibration={name: SimpleNamespace(range_min=1024, range_max=3072) for name in scan.JOINT_NAMES})
        return arm, {name: (-80, 80) for name in demo.JOINTS}, events

    def test_enable_only_after_final_read_and_hold(self):
        arm, limits, events = self.fixture()
        demo.configure_and_hold(arm, position_mode=0)
        self.assertEqual(events[0], "disable")
        self.assertEqual(events[-1], "enable")
        self.assertEqual(events.count("confirm-hold"), 6)
        self.assertLess(events.index("read-final-position"), events.index("hold-target"))
        self.assertLess(events.index("hold-target"), events.index("confirm-hold"))
        self.assertLess(events.index("configure"), events.index("read-final-position"))

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
