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


class EndEffectorPipelineTests(unittest.TestCase):
    """The pipeline these programs assemble is LeRobot's; these tests pin how it is assembled."""

    def test_the_five_steps_are_lerobots_own_and_in_the_documented_order(self):
        source = (ROOT / "IK/so101_cartesian_demo.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        built = next(node for node in ast.walk(tree)
                     if isinstance(node, ast.FunctionDef) and node.name == "build_robot_action_processor")
        imported = {alias.name for node in ast.walk(built) if isinstance(node, ast.ImportFrom)
                    for alias in node.names}
        steps = [node.func.id for node in ast.walk(built)
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                 and node.func.id in imported and node.func.id != "RobotProcessorPipeline"]
        self.assertEqual(steps, ["MapDeltaActionToRobotActionStep", "EEReferenceAndDelta",
                                 "EEBoundsAndSafety", "GripperVelocityToJoint",
                                 "InverseKinematicsEEToJoints"])
        modules = {node.module for node in ast.walk(built) if isinstance(node, ast.ImportFrom)}
        self.assertTrue(all(module.startswith("lerobot.") for module in modules), modules)

    def test_position_only_ik_because_the_arm_has_five_joints(self):
        # InverseKinematicsEEToJoints' own docstring: 0.0 is position-only IK, for under-actuated
        # arms like this one. Anything else asks a 5-DOF arm to match an orientation it cannot.
        source = (ROOT / "IK/so101_cartesian_demo.py").read_text(encoding="utf-8")
        self.assertIn("orientation_weight=0.0", source)
        self.assertNotIn("orientation_weight=0.01", source)

    def test_no_solver_iteration_of_our_own(self):
        # 2026-09-09: the previous version iterated the official step until its own residual
        # criterion was met. One call per frame is the official design; the loop rate is the
        # convergence. A hand-rolled loop is how a program stops being the official one.
        source = (ROOT / "IK/so101_cartesian_demo.py").read_text(encoding="utf-8")
        self.assertNotIn("IK_MAX_ITERATIONS", source)
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, (ast.For, ast.While)):
                calls = [child.func.attr for child in ast.walk(node)
                         if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute)]
                self.assertNotIn("inverse_kinematics", calls, "IK is called inside a loop")

    def test_neither_program_adds_a_target_clamp(self):
        # 2026-09-08: --robot.max_relative_target=2 was read as a speed limit. On a proportional
        # servo it clamps the goal-to-present gap, which is what makes the force, so it capped
        # output at about a tenth and the shoulder could not lift its own weight.
        for name in ("IK/so101_ee_teleop.py", "IK/so101_visual_control.py"):
            source = (ROOT / name).read_text(encoding="utf-8")
            self.assertNotIn("max_relative_target=", source, name)

    def test_bounds_must_be_three_finite_ordered_metres(self):
        good = demo.bounds_dict(demo.DEFAULT_BOUNDS_M["min"], demo.DEFAULT_BOUNDS_M["max"])
        self.assertEqual(sorted(good), ["max", "min"])
        for low, high in (((0, 0, 0), (0, 1, 1)), ((0, 0), (1, 1)),
                          ((0, 0, float("nan")), (1, 1, 1)), ((1, 1, 1), (0, 0, 0))):
            with self.assertRaises(ValueError):
                demo.bounds_dict(low, high)

    def test_the_default_workspace_sits_inside_the_pinned_models_reach(self):
        # Sampled from the URDF joint limits, 2026-09-09: x [-0.34, 0.48], y [-0.44, 0.44],
        # z [-0.22, 0.53] metres. The teaching box must stay inside that and above the base plane.
        reach_min, reach_max = (-0.34, -0.44, -0.22), (0.48, 0.44, 0.53)
        for axis in range(3):
            self.assertGreater(demo.DEFAULT_BOUNDS_M["min"][axis], reach_min[axis])
            self.assertLess(demo.DEFAULT_BOUNDS_M["max"][axis], reach_max[axis])
        self.assertGreater(demo.DEFAULT_BOUNDS_M["min"][2], 0.0, "the box must not reach below the base plane")

    def test_one_frame_asks_for_one_step_at_most(self):
        self.assertGreaterEqual(demo.MAX_EE_STEP_M, demo.EE_STEP_M)
        self.assertEqual(demo.EE_STEP_M, demo.STEP_MM / 1000)


class ViserPageTests(unittest.TestCase):
    """The page is an input device: it reports deltas and never commands a joint."""

    def test_the_page_reports_what_the_official_keyboard_device_reports(self):
        page = visual.ViserEndEffectorTeleop(visual.ViserTeleopConfig(id="t", model_dir="."), None)
        from lerobot.teleoperators.keyboard import KeyboardEndEffectorTeleop
        official = KeyboardEndEffectorTeleop.action_features.fget(
            SimpleNamespace(config=SimpleNamespace(use_gripper=True)))
        self.assertEqual(page.action_features["names"], official["names"])

    def test_the_page_is_a_lerobot_teleoperator(self):
        from lerobot.teleoperators import Teleoperator
        self.assertTrue(issubclass(
            type(visual.build(visual.ViserTeleopConfig(id="t", model_dir="."), None)), Teleoperator))

    def test_a_thrown_handle_still_asks_for_one_step(self):
        step = 0.002
        far = visual.drag_to_units([10.0, -10.0, 10.0], [0, 0, 0], step)
        self.assertTrue((abs(far) <= visual.MAX_UNITS_PER_FRAME).all())
        half = visual.drag_to_units([step / 2, 0, 0], [0, 0, 0], step)
        self.assertAlmostEqual(float(half[0]), 0.5)
        for noisy in ([1e-12, 0, 0], [float("nan"), 0, 0], [float("inf"), 0, 0]):
            self.assertEqual(list(visual.drag_to_units(noisy, [0, 0, 0], step)), [0.0, 0.0, 0.0])
        with self.assertRaises(ValueError):
            visual.drag_to_units([0, 0, 0], [0, 0, 0], 0.0)

    def test_an_undragged_handle_commands_nothing(self):
        page = visual.ViserEndEffectorTeleop(visual.ViserTeleopConfig(id="t", model_dir="."), None)
        page.handle = SimpleNamespace(position=(9.0, 9.0, 9.0))
        self.assertEqual(page.get_action(), {"delta_x": 0.0, "delta_y": 0.0, "delta_z": 0.0,
                                             "gripper": visual.GRIPPER_HOLD})

    def test_a_gripper_click_is_one_frame_long(self):
        page = visual.ViserEndEffectorTeleop(visual.ViserTeleopConfig(id="t", model_dir="."), None)
        page.handle = SimpleNamespace(position=(0.0, 0.0, 0.0))
        page._set_gripper(visual.GRIPPER_CLOSE)
        self.assertEqual(page.get_action()["gripper"], visual.GRIPPER_CLOSE)
        self.assertEqual(page.get_action()["gripper"], visual.GRIPPER_HOLD)

    def test_the_visual_joint_mapping_uses_names_radians_and_the_reported_opening(self):
        names = ("gripper", "wrist_roll", "wrist_flex", "elbow_flex", "shoulder_lift", "shoulder_pan")
        values = visual.viewer_configuration(names, (10, -20, 30, -40, 50), 0)
        self.assertAlmostEqual(values[names.index("shoulder_pan")], math.radians(10))
        self.assertAlmostEqual(values[names.index("wrist_roll")], math.radians(50))
        self.assertAlmostEqual(values[names.index("gripper")], visual.GRIPPER_URDF_RANGE_RAD[0])
        self.assertAlmostEqual(
            visual.viewer_configuration(names, (0, 0, 0, 0, 0), 100)[names.index("gripper")],
            visual.GRIPPER_URDF_RANGE_RAD[1])
        with self.assertRaises(ValueError):
            visual.viewer_configuration(("shoulder_pan",), (10, -20, 30, -40, 50), 0)

    def test_the_display_tap_returns_the_action_untouched(self):
        seen = []
        page = SimpleNamespace(observe=seen.append)
        action = {"delta_x": 0.25, "delta_y": 0.0, "delta_z": 0.0, "gripper": 1}
        observation = {f"{name}.pos": 0.0 for name in demo.MOTORS}
        self.assertEqual(visual.display_tap(page)((dict(action), observation)), action)
        self.assertEqual(seen, [observation])

    def test_the_page_binds_loopback_only_and_needs_an_explicit_port(self):
        self.assertEqual(visual.LOOPBACK, "127.0.0.1")
        for port in (80, 1023, 65536, 0):
            with self.assertRaises(ValueError):
                visual.validate_web_port(port)

    def test_a_port_left_in_time_wait_does_not_block_the_next_run(self):
        # The port table records this: closing the page and restarting within a minute used to
        # fail with Errno 98 while the last connection sat in TIME_WAIT. A live listener must
        # still be reported; only the lingering socket is forgiven.
        import socket
        with socket.socket() as listener:
            listener.bind((visual.LOOPBACK, 0))
            port = listener.getsockname()[1]
            listener.listen(1)
            with self.assertRaises(OSError):
                visual.validate_web_port(port)
        visual.validate_web_port(port)

    def test_the_keyboard_program_only_assembles_official_parts(self):
        # It exists because no lerobot command wires the end-effector pipeline; it must add
        # nothing else. Everything it calls comes from lerobot or from the model helpers.
        tree = ast.parse((ROOT / "IK/so101_ee_teleop.py").read_text(encoding="utf-8"))
        run = next(node for node in ast.walk(tree)
                   if isinstance(node, ast.FunctionDef) and node.name == "run")
        modules = {node.module for node in ast.walk(run) if isinstance(node, ast.ImportFrom)}
        self.assertTrue(all(module.startswith("lerobot.") for module in modules), modules)
        called = {node.func.id for node in ast.walk(run)
                  if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
        self.assertEqual(called - {"SO101Follower", "SO101FollowerConfig", "teleop_loop", "Path",
                                   "KeyboardEndEffectorTeleop", "KeyboardEndEffectorTeleopConfig",
                                   "make_default_processors"}, set())
        self.assertNotIn("class ", (ROOT / "IK/so101_ee_teleop.py").read_text(encoding="utf-8"))


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



parser = argparse.ArgumentParser()
parser.add_argument("--model-dir", type=Path)
args = parser.parse_args()


@unittest.skipUnless(args.model_dir, "optional numerical check requires a pinned model directory and kinematics dependencies")

@unittest.skipUnless(args.model_dir, "optional numerical check requires a pinned model directory and kinematics dependencies")
class NumericalTests(unittest.TestCase):
    """Against the real pinned model and LeRobot's real solver. No hardware, no listener."""

    @classmethod
    def setUpClass(cls):
        import numpy as np
        cls.np = np
        cls.kinematics, cls.limits = demo.load_kinematics(args.model_dir)
        cls.bounds = demo.bounds_dict(demo.DEFAULT_BOUNDS_M["min"], demo.DEFAULT_BOUNDS_M["max"])

    def solve_one_step(self, seed, delta):
        pose = self.kinematics.forward_kinematics(seed).copy()
        target = pose[:3, 3] + self.np.asarray(delta, dtype=float)
        pose[:3, 3] = target
        solved = self.kinematics.inverse_kinematics(seed, pose, position_weight=1.0, orientation_weight=0.0)
        actual = self.kinematics.forward_kinematics(solved)[:3, 3]
        return solved, float(self.np.linalg.norm(actual - target) * 1000)

    def test_one_official_call_reaches_a_one_step_target(self):
        # Measured 2026-09-09: a single call leaves 0.008 mm at the 2 mm step this course uses.
        # That is why nothing iterates: the control loop rate is the convergence.
        seed = self.np.array(demo.PREVIEW_JOINTS_DEG)
        for axis in range(3):
            for sign in (-1, 1):
                delta = self.np.zeros(3)
                delta[axis] = sign * demo.STEP_MM / 1000
                solved, residual = self.solve_one_step(seed, delta)
                self.assertEqual(len(solved), 5)
                self.assertLess(residual, 0.1)

    def test_wrist_roll_is_masked_out_of_the_problem(self):
        # 2026-09-08: position-only IK left the roll axis free, and a drag once planned 155 deg
        # of it. placo's mask_dof removes the joint from the solve; LeRobot exposes the solver.
        seed = self.np.array([13.98, -103.69, 97.01, -102.29, 6.37])
        goal = seed.copy()
        for delta in ((0.002, 0, 0.002), (0.002, 0.001, 0.002), (0.002, 0.002, 0.002)):
            pose = self.kinematics.forward_kinematics(goal).copy()
            pose[:3, 3] = pose[:3, 3] + self.np.asarray(delta)
            goal = self.kinematics.inverse_kinematics(goal, pose, position_weight=1.0, orientation_weight=0.0)
            self.assertAlmostEqual(float(goal[4]), float(seed[4]), places=6)
        unmasked, _ = demo.load_kinematics(args.model_dir, lock_wrist_roll=False)
        self.assertNotEqual(id(unmasked), id(self.kinematics))

    def test_the_assembled_pipeline_returns_one_target_per_motor(self):
        pipeline = demo.build_robot_action_processor(self.kinematics, self.bounds)
        observation = dict(zip((f"{name}.pos" for name in demo.MOTORS),
                               (0.0, -30.0, 60.0, -30.0, 0.0, 50.0)))
        action = pipeline(({"delta_x": 0.0, "delta_y": 0.0, "delta_z": 1.0, "gripper": 1}, observation))
        self.assertEqual(sorted(action), sorted(f"{name}.pos" for name in demo.MOTORS))
        moved = self.kinematics.forward_kinematics(
            self.np.array([action[f"{name}.pos"] for name in demo.JOINTS]))[:3, 3]
        start = self.kinematics.forward_kinematics(
            self.np.array([observation[f"{name}.pos"] for name in demo.JOINTS]))[:3, 3]
        self.assertAlmostEqual(float((moved - start)[2]) * 1000, demo.STEP_MM, places=1)

    def test_a_held_command_stops_at_the_workspace_edge(self):
        pipeline = demo.build_robot_action_processor(self.kinematics, self.bounds)
        observation = dict(zip((f"{name}.pos" for name in demo.MOTORS),
                               (0.0, -30.0, 60.0, -30.0, 0.0, 50.0)))
        for _ in range(300):  # far more frames than it takes to reach the edge
            observation = {key: float(value) for key, value in
                           pipeline(({"delta_x": 1.0, "delta_y": 0.0, "delta_z": 1.0, "gripper": 1}, observation)).items()}
        reached = self.kinematics.forward_kinematics(
            self.np.array([observation[f"{name}.pos"] for name in demo.JOINTS]))[:3, 3]
        self.assertTrue((reached <= self.np.asarray(self.bounds["max"]) + 1e-6).all(), reached)
        self.assertTrue((reached >= self.np.asarray(self.bounds["min"]) - 1e-6).all(), reached)
        self.assertAlmostEqual(float(observation["wrist_roll.pos"]), 0.0, places=6)

    def test_the_gripper_opens_and_closes_within_its_own_scale(self):
        pipeline = demo.build_robot_action_processor(self.kinematics, self.bounds)
        observation = dict(zip((f"{name}.pos" for name in demo.MOTORS),
                               (0.0, -30.0, 60.0, -30.0, 0.0, 50.0)))
        still = {"delta_x": 0.0, "delta_y": 0.0, "delta_z": 0.0}
        closed = pipeline(({**still, "gripper": visual.GRIPPER_CLOSE}, observation))["gripper.pos"]
        opened = pipeline(({**still, "gripper": visual.GRIPPER_OPEN}, observation))["gripper.pos"]
        self.assertGreater(closed, observation["gripper.pos"])
        self.assertLess(opened, observation["gripper.pos"])
        for _ in range(500):
            observation = {key: float(value) for key, value in
                           pipeline(({**still, "gripper": visual.GRIPPER_OPEN}, observation)).items()}
        self.assertGreaterEqual(observation["gripper.pos"], demo.GRIPPER_MIN_PCT)

    def test_preview_prints_the_solver_it_used_and_its_own_residual(self):
        import contextlib, io as _io
        buffer = _io.StringIO()
        with contextlib.redirect_stdout(buffer):
            demo.preview(SimpleNamespace(model_dir=args.model_dir,
                                         joints_deg=list(demo.PREVIEW_JOINTS_DEG),
                                         delta_mm=[0.0, 0.0, demo.STEP_MM]))
        report = json.loads(buffer.getvalue())
        self.assertIn("orientation_weight=0.0", report["solver"])
        self.assertIn("no hardware", report["mode"])
        self.assertLess(report["position_error_mm"], 0.1)
        self.assertEqual(report["solved_degrees"][4], 0.0)

    def test_viser_urdf_meshes_and_fk_match_the_solver(self):
        # Real Viser/yourdfpy loading; only scene transport is a test fixture.
        # No listener, browser, serial port, or hardware is opened by this test.
        import numpy as np
        import yourdfpy
        from functools import partial
        from viser.extras import ViserUrdf
        model_path = args.model_dir / demo.URDF_NAME
        loaded = yourdfpy.URDF.load(model_path, filename_handler=partial(yourdfpy.filename_handler_magic, dir=model_path.parent))
        scene = SimpleNamespace(add_frame=Mock(side_effect=lambda *a, **kw: SimpleNamespace(**kw)), add_mesh_simple=Mock())
        viewer = ViserUrdf(SimpleNamespace(scene=scene), loaded, root_node_name="/test", mesh_color_override=visual.ARM_COLOR)
        self.assertGreater(scene.add_mesh_simple.call_count, 0)
        for call in scene.add_mesh_simple.call_args_list:
            self.assertGreater(len(call.args[1]), 0)
            self.assertGreater(len(call.args[2]), 0)
            self.assertTrue(np.isfinite(call.args[1]).all())
        for seed in (demo.PREVIEW_JOINTS_DEG, [10, -25, 55, -25, 10]):
            viewer.update_cfg(np.array(visual.viewer_configuration(viewer.get_actuated_joint_names(), seed, 0)))
            np.testing.assert_allclose(loaded.get_transform("gripper_frame_link"),
                                       self.kinematics.forward_kinematics(seed), atol=1e-8)


if __name__ == "__main__":
    unittest.main(argv=[__file__], verbosity=2)
