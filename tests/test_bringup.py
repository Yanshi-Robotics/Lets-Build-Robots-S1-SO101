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
demo = load("cartesian_demo", ROOT / "IK/so101_cartesian_demo.py")
sys.modules["so101_cartesian_demo"] = demo
visual = load("visual_control", ROOT / "IK/so101_visual_control.py")
teleop = load("teleop_log", ROOT / "Teleop/so101_teleop_log.py")
cameras = load("camera_check", ROOT / "Cameras/so101_camera_check.py")
sys.modules["so101_camera_check"] = cameras
policy_view = load("policy_view", ROOT / "Cameras/so101_policy_view.py")
load("so101_ee_teleop", ROOT / "IK/so101_ee_teleop.py")


class SolverTests(unittest.TestCase):
    """The solver is this course's own placo servo. These pin why, and what it may do."""

    def test_the_solver_is_placos_and_not_lerobots(self):
        # 2026-09-10. LeRobot 0.6.1's RobotKinematics.inverse_kinematics takes a single Newton
        # step and never iterates -- a target 10 cm away came back as joints still 149.7 mm short
        # -- and it omits update_kinematics() before solving, so the same inputs gave four
        # different answers in four calls. Upstream fixed only the first, after 0.6.1 shipped,
        # and 0.6.1 is still the newest release. ⛔ Do not route the control path back through it.
        source = (ROOT / "IK/so101_cartesian_demo.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported = {alias.name for node in ast.walk(tree)
                    if isinstance(node, (ast.Import, ast.ImportFrom)) for alias in node.names}
        self.assertIn("placo", imported)
        self.assertNotIn("RobotKinematics", imported, "the control path is not routed back to it")
        self.assertFalse([node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
                          and node.name == "build_robot_action_processor"])

    def test_position_only_ik_because_the_arm_has_five_joints(self):
        # ⚠️ Measured 2026-09-10, asking for (0.20, 0.10, 0.15) m while holding the start
        # orientation: 119 mm short at weight 0.1, 36 mm short at 0.01, exactly on it at 0.0.
        # The handle carries no rotation, so the task is switched off rather than weighted down.
        self.assertEqual(demo.ORIENTATION_WEIGHT, 0.0)
        source = (ROOT / "IK/so101_cartesian_demo.py").read_text(encoding="utf-8")
        self.assertIn('mask_dof("wrist_roll")', source,
                      "position-only IK leaves the roll axis free to walk into a limit")

    def test_the_control_path_takes_one_solver_step_a_tick(self):
        # ⭐ servo_step is a servo, not a solve: one step, then the next tick looks again. The
        # iterating version exists (solve_pose) and is model-only, because a converged answer is
        # a whole journey collapsed into one command -- the shape of the 2026-09-08 incident.
        source = (ROOT / "IK/so101_cartesian_demo.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        step = next(node for node in ast.walk(tree)
                    if isinstance(node, ast.FunctionDef) and node.name == "servo_step")
        self.assertFalse([node for node in ast.walk(step) if isinstance(node, (ast.For, ast.While))],
                         "servo_step must not loop")
        loop = next(node for node in ast.walk(tree)
                    if isinstance(node, ast.FunctionDef) and node.name == "control_loop")
        called = {node.func.attr for node in ast.walk(loop)
                  if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
        self.assertIn("servo_step", called)
        self.assertNotIn("solve_pose", called, "the control loop never converges a whole journey")

    def test_the_command_can_never_lead_the_arm_by_more_than_its_budget(self):
        # ⭐ This bound is the force budget. Because the command is re-seeded from the
        # measurement every tick and the solver is velocity limited, the gap between what a
        # motor is told and where it is cannot exceed one tick of travel, whatever the target.
        if not args.model_dir:
            self.skipTest("needs the pinned model")
        import numpy
        servo, _limits = demo.load_servo(str(args.model_dir))
        budget = servo.max_joint_step_deg
        start = {**dict(zip(demo.JOINTS, demo.PREVIEW_JOINTS_DEG)), "gripper": 0.0}
        far = servo.fk(start).copy()
        far[:3, 3] = numpy.array(demo.DEFAULT_BOUNDS_M["max"])
        commanded = servo.servo_step(start, far)
        worst = max(abs(commanded[name] - start[name]) for name in demo.JOINTS)
        self.assertLessEqual(worst, budget + 1e-6, "one tick asked for more than its budget")

    def test_the_speed_and_the_force_are_two_different_numbers(self):
        # ⛔ The invariant this whole rewrite rests on. MAX_JOINT_SPEED_RAD_S decides how far a
        # command may lead the arm, and on an STS3215 that lead is the only thing that makes
        # force: upstream issue #3400 measures 2.3 deg of steady-state error at Feetech's P=32
        # and 5.5 deg at LeRobot's default P=16. A budget below that cannot hold the arm up, let
        # alone move it -- which is exactly the fault this replaced.
        budget = math.degrees(demo.MAX_JOINT_SPEED_RAD_S * demo.CONTROL_DT)
        self.assertGreater(budget, 2.3, "below the error measured at P=32 the arm cannot move")
        self.assertEqual(demo.SERVO_P_COEFFICIENT, 32, "LeRobot's default of 16 doubles that error")
        self.assertGreater(demo.REF_LINEAR_SPEED_MPS, 0)
        self.assertLess(demo.REF_LINEAR_SPEED_MPS, 0.5, "a teaching arm travels slowly")

    def test_the_pose_the_arm_parks_in_commands_nothing(self):
        # ⛔ 2026-09-10 on the bench: the handle mode refused to enable because shoulder_lift
        # read -102.7 while the pinned model stops at -100. That pose is not a fault -- it is
        # where an unpowered arm falls -- and refusing it asked the operator to hold the arm up
        # with one hand and click with the other.
        # ⭐ The cause was the solver being given the model's limits instead of this arm's own
        # recorded travel. On the model's limits the seed has to be pulled into range, and
        # closing that gap is a real command: 2.7 deg, with nobody asking. On the arm's travel
        # there is no gap. This pins the difference.
        if not args.model_dir:
            self.skipTest("needs the pinned model")
        parked = {"shoulder_pan": -5.4, "shoulder_lift": -102.7, "elbow_flex": 95.0,
                  "wrist_flex": -92.0, "wrist_roll": 8.8, "gripper": 1.8}
        travel = {"shoulder_pan": 110.0, "shoulder_lift": 104.5, "elbow_flex": 98.4,
                  "wrist_flex": 104.8, "wrist_roll": 180.0}

        on_the_model, _limits = demo.load_servo(str(args.model_dir))
        held = on_the_model.servo_step(parked, on_the_model.fk(parked))
        self.assertGreater(max(abs(held[name] - parked[name]) for name in demo.JOINTS), 2.0,
                           "the model's limits are what used to move the arm")

        on_the_arm, _limits = demo.load_servo(str(args.model_dir))
        on_the_arm.use_recorded_travel(travel)
        self.assertEqual(on_the_arm.inside_the_model(parked), parked, "nothing left to clamp")
        still = on_the_arm.servo_step(parked, on_the_arm.fk(parked))
        self.assertLess(max(abs(still[name] - parked[name]) for name in demo.JOINTS), 0.05,
                        "enabling from the parked pose must command nothing at all")

    def test_a_joint_the_model_cannot_express_is_named_with_the_way_back(self):
        # ⛔ Reproduced 2026-09-10 from the 2026-09-08 hardware log: this follower parks at
        # shoulder_lift -103.8 and wrist_flex -100.2, outside the pinned model. placo clamps its
        # answer into that range, so replaying the parked pose through a solve moves four joints
        # by up to 5.8 deg before anyone has asked for anything.
        limits = {"shoulder_pan": (-110.0, 110.0), "shoulder_lift": (-100.0, 100.0),
                  "elbow_flex": (-96.83, 96.83), "wrist_flex": (-95.0, 95.0),
                  "wrist_roll": (-157.0, 163.0)}
        parked = dict(zip(demo.JOINTS, (-5.36, -103.78, 97.10, -100.18, 8.84)))
        complaints = demo.joints_outside_the_model(parked, limits)
        self.assertEqual(len(complaints), 3)
        named = " ".join(complaints)
        for joint in ("shoulder_lift", "elbow_flex", "wrist_flex"):
            self.assertIn(joint, named)
        self.assertIn("Move it at least", named)
        self.assertEqual(demo.joints_outside_the_model(
            dict(zip(demo.JOINTS, demo.PREVIEW_JOINTS_DEG)), limits), [])

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
        self.assertGreaterEqual(demo.DEFAULT_BOUNDS_M["min"][2], 0.0,
                                "the box must not reach below the base plane")


class PageAndKeyboardTests(unittest.TestCase):
    """The browser page holds what the operator asked for. ⛔ It never talks to the arm."""

    def test_the_page_does_not_touch_the_arm(self):
        # ⛔ Viser callbacks run on the web server's own threads. Two threads sharing one serial
        # bus corrupt each other's packets in ways that look like a hardware fault, so every
        # callback writes down a wish and returns; the control loop is the only caller.
        tree = ast.parse((ROOT / "IK/so101_visual_control.py").read_text(encoding="utf-8"))
        page = next(node for node in ast.walk(tree)
                    if isinstance(node, ast.ClassDef) and node.name == "ViserPage")
        called = {node.func.attr for node in ast.walk(page)
                  if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
        for forbidden in ("send_action", "sync_read", "sync_write", "disable_torque", "read"):
            self.assertNotIn(forbidden, called, "the page reached for the arm")

    def test_a_released_key_no_longer_cancels_one_still_held(self):
        # ⚠️ Upstream, open as PR #3947: _drain_pressed_keys records a released key as False
        # instead of dropping it, and get_action then walks that dictionary assigning one axis at
        # a time -- so a key released a moment ago writes its zero over the key still held.
        from pynput import keyboard as keys
        from lerobot.teleoperators.keyboard import KeyboardEndEffectorTeleop
        device = demo.keyboard_device("t")
        device._on_press(keys.Key.up)
        device._on_press(keys.Key.down)
        device._on_release(keys.Key.down)
        upstream = KeyboardEndEffectorTeleop(device.config)
        upstream.event_queue, upstream.current_pressed = device.event_queue, {}
        self.assertEqual(demo.keyboard_device("t").__class__.__mro__[1],
                         KeyboardEndEffectorTeleop, "it is still LeRobot's device underneath")
        held = KeyboardEndEffectorTeleop.get_action.__wrapped__(device)
        self.assertEqual(float(held["delta_y"]), -1.0,
                         "the still-held arrow key must survive the released one")

    def test_a_released_left_ctrl_is_not_a_gripper_command(self):
        # ⚠️ Upstream, reproduced 2026-09-09: ctrl_l computes `int(val) - 1`, so releasing it
        # reports -1 rather than 1 -- a fourth value where the protocol has three.
        from pynput import keyboard as keys
        from lerobot.teleoperators.keyboard import KeyboardEndEffectorTeleop, KeyboardEndEffectorTeleopConfig
        device = KeyboardEndEffectorTeleop(KeyboardEndEffectorTeleopConfig(id="t", use_gripper=True))
        device._on_press(keys.Key.ctrl_l)
        device._drain_pressed_keys()
        held = KeyboardEndEffectorTeleop.get_action.__wrapped__(device)
        self.assertEqual(demo.gripper_from_keyboard(held), demo.GRIPPER_CLOSE)
        device._on_release(keys.Key.ctrl_l)
        device._drain_pressed_keys()
        released = KeyboardEndEffectorTeleop.get_action.__wrapped__(device)
        self.assertEqual(released["gripper"], -1, "the upstream behaviour this guards against")
        self.assertEqual(demo.gripper_from_keyboard(released), demo.GRIPPER_HOLD)
        self.assertEqual(demo.gripper_from_keyboard({}), demo.GRIPPER_HOLD)

    def test_the_page_says_which_keys_it_believes_are_down(self):
        # A keyboard that reaches nothing looks exactly like a program that does nothing.
        self.assertEqual(demo.keys_held({}), "")
        self.assertEqual(demo.keys_held({"delta_y": -1.0}), "y")
        self.assertIn("close", demo.keys_held({"gripper": demo.GRIPPER_CLOSE}))

    def test_the_arrow_keys_move_the_target_inside_the_workspace(self):
        # ⭐ The keys move the target, not the arm: an absolute pose this program owns, which is
        # why nothing here has to integrate anything from a measurement.
        bounds = demo.bounds_dict((0.0, -0.2, 0.0), (0.4, 0.2, 0.4))
        moved = demo.handle_after_keys((0.2, 0.0, 0.2), {"delta_x": 1.0}, bounds, 0.001)
        self.assertAlmostEqual(float(moved[0]), 0.201)
        clamped = demo.handle_after_keys((0.4, 0.0, 0.2), {"delta_x": 1.0}, bounds, 0.001)
        self.assertAlmostEqual(float(clamped[0]), 0.4, msg="the box is a hard edge for the target")

    def test_the_visual_joint_mapping_uses_names_and_radians(self):
        # ⛔ get_actuated_joint_names() is ordered by the URDF's topology, which is the reverse
        # of the bus order. A vector assembled by position draws a plausible arm in a wrong pose.
        names = ("gripper", "wrist_roll", "wrist_flex", "elbow_flex", "shoulder_lift", "shoulder_pan")
        travel = (-0.2, 1.7)
        degrees = dict(zip(demo.JOINTS, (10, -20, 30, -40, 50)))
        values = visual.viewer_configuration(names, {**degrees, "gripper": 0}, travel)
        self.assertAlmostEqual(values[names.index("shoulder_pan")], math.radians(10))
        self.assertAlmostEqual(values[names.index("wrist_roll")], math.radians(50))
        self.assertAlmostEqual(values[names.index("gripper")], travel[0])
        self.assertAlmostEqual(
            visual.viewer_configuration(names, {**degrees, "gripper": 100}, travel)[names.index("gripper")],
            travel[1])
        with self.assertRaises(ValueError):
            visual.viewer_configuration(("shoulder_pan",), {**degrees, "gripper": 0}, travel)

    def test_the_gripper_travel_is_read_from_the_model_not_written_down(self):
        # ⚠️ Which end is open has not been watched on hardware, so nothing claims it. This only
        # maps LeRobot's 0-100 onto whatever the pinned model says the joint can do.
        if not args.model_dir:
            self.skipTest("needs the pinned model")
        low, high = visual.gripper_urdf_range(str(args.model_dir))
        self.assertLess(low, high)
        self.assertNotIn("GRIPPER_URDF_RANGE_RAD",
                         (ROOT / "IK/so101_visual_control.py").read_text(encoding="utf-8"))

    def test_the_page_binds_loopback_only_and_needs_an_explicit_port(self):
        self.assertEqual(visual.LOOPBACK, "127.0.0.1")
        for port in (80, 1023, 65536, 0):
            with self.assertRaises(ValueError):
                visual.validate_web_port(port)

    def test_a_port_left_in_time_wait_does_not_block_the_next_run(self):
        # The port table records this: restarting within a minute used to fail with Errno 98
        # while the last connection sat in TIME_WAIT. A live listener must still be reported.
        import socket
        with socket.socket() as listener:
            listener.bind((visual.LOOPBACK, 0))
            port = listener.getsockname()[1]
            listener.listen(1)
            with self.assertRaises(OSError):
                visual.validate_web_port(port)
        visual.validate_web_port(port)


class ProgramShapeTests(unittest.TestCase):
    """What the two programs are allowed to be, read off their source."""

    def test_neither_program_adds_a_target_clamp(self):
        # 2026-09-08: --robot.max_relative_target=2 was read as a speed limit. On a proportional
        # servo it clamps the goal-to-present gap, which is what makes the force, so it capped
        # output at about a tenth and the shoulder could not lift its own weight.
        for name in ("IK/so101_ee_teleop.py", "IK/so101_visual_control.py"):
            self.assertNotIn("max_relative_target=", (ROOT / name).read_text(encoding="utf-8"), name)

    def test_the_page_keeps_holding_when_the_program_stops(self):
        # 2026-09-09, Jeff: an arm under IK has to stay held, or it drops at the end of a move.
        # disable_torque_on_disconnect defaults to True, which releases every motor on exit.
        # ⭐ 2026-09-10: the arm itself moved into the shared module, so that is where the
        # setting lives now. The notice stayed with the program that prints it.
        self.assertIn("disable_torque_on_disconnect=False",
                      (ROOT / "IK/so101_cartesian_demo.py").read_text(encoding="utf-8"))
        self.assertIn("Cutting DC power", visual.SAFETY_NOTICE)
        self.assertIn("--release-torque", visual.SAFETY_NOTICE)

    def test_there_is_exactly_one_control_loop_and_both_programs_share_it(self):
        # ⚠️ Reversed on 2026-09-10, deliberately. This used to require that every frame came
        # from LeRobot's teleop_loop. It cannot any more: teleop_loop exists to carry a
        # teleoperator's action through identity processors, and the thing that made the arm
        # unable to move was the target those processors were handed. Owning the loop is what
        # lets the target be an absolute pose walked towards at a set speed.
        # ⛔ What replaces the old rule: the loop lives in the shared module and is imported by
        # both programs. A loop copied into either of them is a loop that is right in one place.
        shared = (ROOT / "IK/so101_cartesian_demo.py").read_text(encoding="utf-8")
        self.assertIn("def control_loop(", shared)
        for name in ("IK/so101_ee_teleop.py", "IK/so101_visual_control.py"):
            source = (ROOT / name).read_text(encoding="utf-8")
            self.assertNotIn("def control_loop(", source, f"{name} has a loop of its own")
            self.assertIn("control_loop(", source, f"{name} does not use the shared loop")
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(node, (ast.For, ast.While)):
                    calls = {child.func.attr for child in ast.walk(node)
                             if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute)}
                    self.assertNotIn("send_action", calls, f"{name} drives the robot itself")

    def test_the_command_line_program_proves_its_keyboard_before_powering(self):
        # 2026-09-09: it used to connect the follower -- which enables torque -- before the
        # keyboard existed. A silent input device then looked exactly like a dead program, with
        # the arm live the whole time.
        source = (ROOT / "IK/so101_ee_teleop.py").read_text(encoding="utf-8")
        # ⭐ Torque now arrives inside the shared loop, the first time a mode is enabled, so the
        # thing to be before is the loop rather than a connect() this file no longer calls.
        self.assertLess(source.index("teleop.connect()"), source.index("control_loop("))
        self.assertLess(source.index("wait_for_a_key("), source.index("control_loop("))

    def test_the_goal_is_parked_before_a_single_motor_is_powered(self):
        # 2026-09-09, third attempt on hardware. An unpowered SO-101 falls onto its shoulder stop
        # and stays there, so refusing to start from a stop refused the only pose the arm has.
        # The hazard was never the stop: it is torque arriving while Goal_Position is somewhere
        # else. connect() never writes a goal, and a DC power cycle leaves every motor holding 0.
        shared = (ROOT / "IK/so101_cartesian_demo.py").read_text(encoding="utf-8")
        hold = shared[shared.index("    def hold(self, degrees):", shared.index("class LiveArm")):]
        self.assertLess(hold.index("park_the_goal"), hold.index("robot.connect()"))
        self.assertLess(hold.index("robot.connect()"), hold.index("goal_diverged"))
        for name in ("IK/so101_ee_teleop.py", "IK/so101_visual_control.py"):
            self.assertIn("read_before_power", (ROOT / name).read_text(encoding="utf-8"), name)
        # ⚠️ Reversed 2026-09-10. This used to assert the opposite, on the belief that placo does
        # not clamp an out-of-limit joint. It does: replaying the 2026-09-08 parked pose through
        # a solve moves four joints by up to 5.8 deg with nothing commanded, because the solver
        # clamps its answer into the model's range. The solving modes refuse that pose instead.
        self.assertIn("joints_outside_the_model", shared)

    def test_the_encoder_resolution_is_lerobots_not_ours(self):
        # Every stop distance depends on this. A copy of the number here is a second source of
        # truth that can stop matching the library without anyone noticing.
        from lerobot.motors.feetech.tables import MODEL_RESOLUTION
        self.assertAlmostEqual(demo.degrees_per_step(),
                               360 / (MODEL_RESOLUTION[demo.MOTOR_MODEL] - 1))
        source = (ROOT / "IK/so101_cartesian_demo.py").read_text(encoding="utf-8")
        for literal in ("4095", "4096"):
            self.assertNotIn(literal, source, "the encoder resolution must not be written down")

    def test_parking_the_goal_asks_every_motor_to_hold_where_it_is(self):
        written = {}
        robot = SimpleNamespace(bus=SimpleNamespace(
            sync_write=lambda name, values, **kw: written.update({name: dict(values)}),
            sync_read=lambda name, motors=None, **kw: (
                dict(written.get("Goal_Position", {})) if name == "Goal_Position"
                else {n: 1.0 for n in demo.MOTORS})))
        observation = {f"{name}.pos": 1.0 for name in demo.MOTORS}
        self.assertEqual(demo.park_the_goal(robot, observation), {name: 1.0 for name in demo.MOTORS})
        self.assertEqual(set(written), {"Goal_Position"}, "nothing else is written before torque")
        self.assertEqual(demo.goal_diverged(robot), [], "a parked goal is not a divergence")
        # A DC power cycle leaves every goal at 0, which is one end of the travel: that is the
        # shape this has to catch, not a degree of sag.
        written["Goal_Position"]["shoulder_lift"] = 1.0 - demo.GOAL_MARGIN_DEG - 1.0
        diverged = demo.goal_diverged(robot)
        self.assertEqual(len(diverged), 1, diverged)
        self.assertIn("shoulder_lift", diverged[0])
        written["Goal_Position"]["shoulder_lift"] = 1.0 - demo.GOAL_MARGIN_DEG + 0.5
        self.assertEqual(demo.goal_diverged(robot), [], "sag inside the margin is not a fault")

    def test_a_stop_is_reported_but_never_refused(self):
        source = (ROOT / "IK/so101_visual_control.py").read_text(encoding="utf-8")
        body = source[source.index("def self_check("):source.index("def run(args):")]
        after_stop = body[body.index("joints_on_a_stop"):]
        self.assertNotIn("raise CheckFailed", after_stop[:after_stop.index('report("joint travel"')])

    def test_the_stop_note_measures_this_arm_not_the_model(self):
        # 2026-09-09 on hardware: the refusal quoted the URDF's -100..+100 while this arm's
        # recorded travel is +-104.48, so "1.2 deg short of the stop" was printed as "3.3 deg over
        # a limit", and it never said how far to move.
        # ⛔ Not read from calibration/: that directory is git-ignored and belongs to one arm, so
        # a test reading it cannot run anywhere else. These are the raw ranges recorded on the
        # course follower on 2026-09-09, with the expectation derived from them rather than typed.
        recorded = {"shoulder_lift": (929, 3306), "elbow_flex": (926, 3163),
                    "shoulder_pan": (635, 3375), "wrist_flex": (835, 3218),
                    "wrist_roll": (0, 4095), "gripper": (2000, 3539)}
        calibration = {name: {"homing_offset": 0, "range_min": low, "range_max": high}
                       for name, (low, high) in recorded.items()}
        travel = demo.travel_degrees(calibration)
        per_step = demo.degrees_per_step()
        for name, (low, high) in recorded.items():
            self.assertAlmostEqual(travel[name], (high - low) / 2 * per_step)
        self.assertAlmostEqual(travel["shoulder_lift"], 104.48, places=1)
        self.assertAlmostEqual(travel["elbow_flex"], 98.33, places=1)
        folded = dict(zip((f"{name}.pos" for name in demo.MOTORS),
                          (-30.95, -103.6, 97.0, 85.93, 6.20, 33.0)))
        notes = demo.joints_on_a_stop(folded, calibration)
        self.assertEqual(len(notes), 2, notes)
        self.assertIn("shoulder_lift is 0.9 deg from the end of its travel", notes[0])
        self.assertIn("Move it at least", notes[0], "it must say how far, not just that it is close")
        clear = dict(zip((f"{name}.pos" for name in demo.MOTORS),
                         (*demo.PREVIEW_JOINTS_DEG, 33.0)))
        self.assertEqual(demo.joints_on_a_stop(clear, calibration), [])
        if not args.model_dir:
            return
        # ⭐ Reaching past the model is safe to compute with: placo does not clamp, so forward
        # kinematics for an out-of-limit joint is the pose the arm is really in.
        servo, limits = demo.load_servo(args.model_dir)
        for name, (low, high) in limits.items():
            self.assertGreater(travel[name], high, f"{name}: this arm reaches past the model")
            self.assertLess(-travel[name], low, name)
        # ⭐ Forward kinematics never clamps, because saying where the arm really is is its job.
        # ⚠️ The solver is the opposite: its joint limits are hard constraints, so `servo_step`
        # pulls its seed inside them rather than raise QPError on the pose the arm parks in.
        past = {**dict(zip(demo.JOINTS, (0.0, -103.6, 97.0, 0.0, 0.0))), "gripper": 0.0}
        servo.fk(past)
        self.assertAlmostEqual(math.degrees(servo.robot.get_joint("shoulder_lift")), -103.6,
                               places=3, msg="clamping in fk would falsify every pose")
        self.assertAlmostEqual(servo.inside_the_model(past)["shoulder_lift"], -100.0, places=3)
        servo.servo_step(past, servo.fk(past))      # ⛔ must not raise

    def test_the_workspace_opens_far_enough_to_contain_the_parked_pose(self):
        # An arm rests where gravity leaves it, which on this follower is 20 mm below the base
        # plane. EEBoundsAndSafety clips rather than refuses, so a box that excluded the parked
        # pose would walk the arm to its edge on the first frame.
        asked = demo.bounds_dict((0.0, -0.22, 0.0), (0.38, 0.22, 0.42))
        widened = demo.bounds_including(asked, (0.099, 0.035, -0.020))
        self.assertAlmostEqual(float(widened["min"][2]), -0.020 - demo.WORKSPACE_MARGIN_M)
        self.assertEqual(demo.outside_the_workspace((0.099, 0.035, -0.020), widened), [])
        unchanged = demo.bounds_including(asked, (0.2, 0.0, 0.2))
        self.assertEqual(list(unchanged["min"]), list(asked["min"]), "a contained pose widens nothing")

    def test_the_self_check_runs_before_any_server_or_any_motor(self):
        # Jeff, 2026-09-09: a page that opens looking healthy while none of its modes can work is
        # worse than no page. The check has to precede both the server and the power.
        source = (ROOT / "IK/so101_visual_control.py").read_text(encoding="utf-8")
        body = source[source.index("def run(args):"):]
        self.assertLess(body.index("self_check("), body.index("page.start("))
        self.assertLess(body.index("self_check("), body.index("arm.open_bus()"))
        self.assertLess(body.index("page.start("), body.index("arm.open_bus()"))

    def test_only_ending_a_mode_releases_the_arm(self):
        # Jeff, 2026-09-09: a crash must not let go, because releasing a raised arm drops it.
        # Only the End path releases, and the program says so on any other way out.
        shared = (ROOT / "IK/so101_cartesian_demo.py").read_text(encoding="utf-8")
        loop = shared[shared.index("def control_loop("):shared.index("def step_towards(")]
        self.assertEqual(loop.count("arm.release()"), 1, "exactly one release, on the End path")
        self.assertLess(loop.index("take_end_request()"), loop.index("arm.release()"))
        self.assertNotIn("disable_torque", loop, "the loop never reaches past the arm to let go")
        # ⛔ And a stall stops the arm pushing without letting go of it.
        self.assertIn("stop_pushing", loop)
        source = (ROOT / "IK/so101_visual_control.py").read_text(encoding="utf-8")
        self.assertIn("except BaseException", source, "Ctrl+C counts as an unsafe stop too")
        self.assertIn("SAFETY_NOTICE", source)
        self.assertIn("--release-torque", source)

    def test_the_pre_power_read_opens_and_hands_back_the_port_untouched(self):
        # It must read over bus.connect(), which only opens the port, and never over
        # Robot.connect(), which ends in configure() with torque on. It must also hand the port
        # back closed, so the official connect() can open it straight afterwards.
        events, reading = [], {name: 1.0 for name in demo.MOTORS}
        torque = dict.fromkeys(demo.MOTORS, 0) | {"shoulder_pan": 1}

        def sync_read(data_name, motors=None, *, normalize=True, num_retry=0):
            events.append(f"read {data_name} normalize={normalize}")
            if data_name == "Present_Position":
                return dict(reading)
            if data_name == "Torque_Enable":
                return dict(torque)
            return dict.fromkeys(demo.MOTORS, 0)

        robot = SimpleNamespace(
            config=SimpleNamespace(num_read_retries=3),
            bus=SimpleNamespace(
                connect=lambda: events.append("bus.connect"),
                disconnect=lambda disable_torque=True: events.append(f"bus.disconnect({disable_torque})"),
                sync_read=sync_read))
        observation, powered, stored = demo.read_pose_before_power(robot)
        self.assertEqual(observation, {f"{name}.pos": 1.0 for name in demo.MOTORS})
        self.assertEqual(powered, ["shoulder_pan"])
        self.assertEqual(set(stored), set(demo.MOTORS), "the stored calibration is read too")
        self.assertEqual(events[0], "bus.connect")
        self.assertEqual(events[-1], "bus.disconnect(False)", "the port must be handed back closed")
        self.assertIn("read Torque_Enable normalize=False", events)

    def test_the_release_command_reports_what_it_let_go_of(self):
        # Jeff, 2026-09-09: after an unsafe stop the operator releases the arm themselves. The
        # command reads first, so it can say nothing was held rather than pretend it did something.
        events, deg = [], dict(zip(demo.MOTORS, (0.0, -30.0, 60.0, -30.0, 0.0, 50.0)))
        torque = dict.fromkeys(demo.MOTORS, 1)

        def sync_read(name, motors=None, *, normalize=True, num_retry=0):
            events.append(f"read {name}")
            return dict(deg) if name == "Present_Position" else dict(torque)

        def disable():
            events.append("disable_torque")
            torque.update(dict.fromkeys(demo.MOTORS, 0))

        robot = SimpleNamespace(bus=SimpleNamespace(
            connect=lambda: events.append("connect"), sync_read=sync_read,
            disable_torque=disable, is_connected=True,
            disconnect=lambda disable_torque=True: events.append(f"disconnect({disable_torque})")))
        fake_arm = SimpleNamespace(
            robot=robot, holding=False,
            open_bus=lambda: robot.bus.connect(),
            close_bus=lambda: robot.bus.disconnect(False),
            read=lambda: {name: float(value) for name, value in deg.items()})
        import contextlib, io as _io
        with patch.object(demo, "LiveArm", lambda *a, **k: fake_arm), \
                contextlib.redirect_stdout(_io.StringIO()):
            self.assertEqual(visual.release_only(
                SimpleNamespace(port="p", robot_id="r", calibration_dir="c")), 0)
        self.assertIn("disable_torque", events)
        self.assertEqual(events[-1], "disconnect(False)")
        self.assertLess(events.index("read Torque_Enable"), events.index("disable_torque"),
                        "it must look before it lets go")
        events.clear()
        with patch.object(demo, "LiveArm", lambda *a, **k: fake_arm), \
                contextlib.redirect_stdout(_io.StringIO()):
            self.assertEqual(visual.release_only(
                SimpleNamespace(port="p", robot_id="r", calibration_dir="c")), 0)
        self.assertNotIn("disable_torque", events, "already released means nothing to do")

    def test_the_keyboard_gate_gives_up_instead_of_powering_the_arm(self):
        import so101_ee_teleop as keyboard_program
        silent = SimpleNamespace(get_action=lambda: {"delta_x": 0.0, "delta_y": 0.0, "delta_z": 0.0})
        original = keyboard_program.KEY_CHECK_SECONDS
        keyboard_program.KEY_CHECK_SECONDS = 0.2
        try:
            with self.assertRaises(RuntimeError) as gave_up:
                keyboard_program.wait_for_a_key(silent)
        finally:
            keyboard_program.KEY_CHECK_SECONDS = original
        self.assertIn("the arm was never powered", str(gave_up.exception))
        keyboard_program.wait_for_a_key(
            SimpleNamespace(get_action=lambda: {"delta_x": 0.0, "delta_y": -1.0, "delta_z": 0.0}))


class ControlLoopTests(unittest.TestCase):
    """What the one loop actually does, driven with a page that is not a browser.

    ⭐ These are the tests the 2026-09-10 rewrite exists for. Every one of them fails against
    the version that could not move the arm, because that version's fault was not a crash: it
    asked, politely and for ever, for about one degree.
    """

    class Page:
        """The page interface, with nothing behind it but the wishes the test wants recorded."""

        def __init__(self, target, ticks, mode=None):
            import numpy
            self.target = numpy.asarray(target, dtype=float)
            self.handle = self.target.copy()
            self.ticks, self.seen = ticks, []
            self.mode = demo.HANDLE if mode is None else mode
            self.armed_as, self.asked, self.gripper = None, False, 0.0

        def take_arm_request(self):
            if self.asked:
                return None
            self.asked = True
            return self.mode

        def take_end_request(self):
            return False

        def handle_xyz(self):
            return self.handle

        def gripper_target(self):
            return self.gripper

        def set_gripper_target(self, percent):
            self.gripper = float(percent)

        def nudge_gripper(self, percent):
            self.gripper += float(percent)

        def move_handle(self, xyz):
            import numpy
            # ⛔ Only while nothing is live: once armed, the handle is the operator's, and the
            # loop snapping it back to the gripper is what the real page must never do either.
            if self.armed_as is None:
                self.handle = numpy.asarray(xyz, dtype=float)

        def armed(self, mode):
            self.armed_as = mode
            self.handle = self.target.copy()

        def disarmed(self, message=""):
            self.armed_as = None

        def show(self, measured, commanded, status, note="", keys=""):
            self.seen.append((dict(measured), dict(commanded)))
            if len(self.seen) >= self.ticks:
                raise StopIteration

    def drive(self, target, ticks, arm=None, mode=None, leader=None):
        if not args.model_dir:
            self.skipTest("needs the pinned model")
        servo, limits = demo.load_servo(str(args.model_dir))
        start = {**dict(zip(demo.JOINTS, demo.PREVIEW_JOINTS_DEG)), "gripper": 0.0}
        arm = demo.ModelArm(start) if arm is None else arm
        bounds = demo.bounds_dict(demo.DEFAULT_BOUNDS_M["min"], demo.DEFAULT_BOUNDS_M["max"])
        page = self.Page(target, ticks, mode)
        stalled = None
        # ⚠️ The loop paces itself to real time; the tests do not have that long.
        with patch.object(demo.time, "sleep", lambda _seconds: None):
            try:
                demo.control_loop(page, arm, servo, bounds, limits, leader=leader)
            except StopIteration:
                pass
            except demo.FollowingLost as lost:
                stalled = str(lost)
        return servo, arm, page, stalled

    def test_a_stalled_arm_is_commanded_up_to_its_budget_and_no_further(self):
        # ⭐ The judge. Against the version this replaced, the command stayed about 1.09 deg
        # ahead of the arm for ever, whatever the target -- far below the 2.3 deg of position
        # error a Feetech STS3215 needs at P=32 before it makes any useful force, so the arm
        # never moved. The command must be free to lead by the whole budget, and by no more.
        class Stuck(demo.ModelArm):
            def send(self, degrees):
                pass                      # the commands go out; the arm does not follow

        servo, _arm, page, _stalled = self.drive((0.30, 0.0, 0.25), 120, arm=Stuck(
            {**dict(zip(demo.JOINTS, demo.PREVIEW_JOINTS_DEG)), "gripper": 0.0}))
        leads = [max(abs(commanded[name] - measured[name]) for name in demo.JOINTS)
                 for measured, commanded in page.seen if page.armed_as]
        budget = servo.max_joint_step_deg
        self.assertGreater(max(leads), budget * 0.9,
                           "a held-back arm must be commanded with the whole budget")
        self.assertLessEqual(max(leads), budget + 1e-6, "and never with more than the budget")

    def test_the_arm_travels_at_the_set_speed_and_stops_when_it_arrives(self):
        servo, arm, page, stalled = self.drive((0.30, 0.0, 0.25), 400)
        import numpy
        reached = servo.gripper_xyz(arm.read())
        self.assertIsNone(stalled, "a reachable target is not a stall")
        self.assertLess(float(numpy.linalg.norm(reached - page.target)) * 1000, demo.ARRIVED_MM)
        moved = [numpy.linalg.norm(servo.gripper_xyz(b) - servo.gripper_xyz(a))
                 for (_m, a), (_n, b) in zip(page.seen, page.seen[1:])]
        a_tick = demo.REF_LINEAR_SPEED_MPS * demo.CONTROL_DT
        self.assertLessEqual(max(moved), a_tick * 1.05, "the gripper outran the set speed")
        # ⭐ And once it is there, it stops pushing: the command settles onto the measurement.
        settled = page.seen[-1]
        self.assertLess(max(abs(settled[1][name] - settled[0][name]) for name in demo.JOINTS), 0.05)

    def test_a_target_it_cannot_reach_is_stopped_without_being_let_go(self):
        # ⛔ Stopping is not releasing. The force comes out of the motors -- the goal is parked
        # where they stand -- but letting go of a raised arm drops it, so nothing lets go.
        released, parked = [], []

        class Watched(demo.ModelArm):
            def release(self):
                released.append(True)

            def stop_pushing(self, degrees):
                parked.append(dict(degrees))

        _servo, _arm, _page, stalled = self.drive(
            (0.38, 0.22, 0.42), 900, arm=Watched(
                {**dict(zip(demo.JOINTS, demo.PREVIEW_JOINTS_DEG)), "gripper": 0.0}))
        self.assertIsNotNone(stalled, "an unreachable target must be reported, not chased for ever")
        self.assertIn("stopped closing", stalled)
        self.assertEqual(released, [], "a stall must never release the arm")
        self.assertEqual(len(parked), 1, "it must stop pushing, exactly once")

    def test_the_jaw_does_not_jump_when_a_mode_starts(self):
        # 2026-09-09 on hardware: the slider sat at its default while the jaw read 1.8, so
        # enabling any mode drove the gripper across its travel before anyone touched a control.
        start = {**dict(zip(demo.JOINTS, demo.PREVIEW_JOINTS_DEG)), "gripper": 1.8}
        _servo, arm, page, _stalled = self.drive((0.30, 0.0, 0.25), 6, arm=demo.ModelArm(start))
        self.assertAlmostEqual(page.gripper, 1.8, places=6,
                               msg="the slider moves to the jaw, not the jaw to the slider")
        self.assertLess(abs(arm.read()["gripper"] - 1.8), demo.GRIPPER_STEP_PCT + 1e-6)

    def test_the_leader_mode_never_reaches_the_solver(self):
        # ⛔ Lesson 7's chain, and the only mode that worked before this rewrite. It passes joint
        # angles straight through: no solver, no speed limit, no stall check -- the hand on the
        # leader is all three.
        reading = {f"{name}.pos": 12.5 for name in demo.MOTORS}
        leader = SimpleNamespace(get_action=lambda: dict(reading))
        servo, arm, _page, stalled = self.drive((0.38, 0.22, 0.42), 40, mode=demo.LEADER,
                                                leader=leader)
        self.assertIsNone(stalled, "the leader mode is not watched for stalling")
        self.assertEqual({name: round(value, 3) for name, value in arm.read().items()},
                         {name: 12.5 for name in demo.MOTORS})


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



parser = argparse.ArgumentParser()
parser.add_argument("--model-dir", type=Path)
args = parser.parse_args()


@unittest.skipUnless(args.model_dir, "optional numerical check requires a pinned model directory and kinematics dependencies")

@unittest.skipUnless(args.model_dir, "optional numerical check requires a pinned model directory and kinematics dependencies")
class NumericalTests(unittest.TestCase):
    """Against the real pinned model and the real solver. No hardware, no listener."""

    @classmethod
    def setUpClass(cls):
        import numpy
        cls.np = numpy
        cls.servo, cls.limits = demo.load_servo(args.model_dir)
        cls.bounds = demo.bounds_dict(demo.DEFAULT_BOUNDS_M["min"], demo.DEFAULT_BOUNDS_M["max"])
        cls.start = {**dict(zip(demo.JOINTS, demo.PREVIEW_JOINTS_DEG)), "gripper": 0.0}

    def test_a_small_move_is_walked_off_in_a_few_ticks(self):
        # ⭐ A tick is a step, not a solve. The velocity limit is nowhere near reached for a
        # millimetre, but the regularisation that keeps the spare joint from drifting also damps
        # the step, so it takes a handful of ticks -- 20 ms of them -- rather than one.
        for axis in range(3):
            for sign in (-1, 1):
                pose = self.servo.fk(self.start).copy()
                delta = self.np.zeros(3)
                delta[axis] = sign * 0.001
                pose[:3, 3] = pose[:3, 3] + delta
                walking = dict(self.start)
                for _tick in range(10):
                    walking = self.servo.servo_step(walking, pose)
                reached = self.servo.gripper_xyz(walking)
                self.assertLess(float(self.np.linalg.norm(reached - pose[:3, 3])) * 1000, 0.1)

    def test_a_converged_solve_is_exact_where_lerobots_single_step_was_not(self):
        # ⛔ Why the solver was replaced. LeRobot 0.6.1 takes one Newton step and stops: measured
        # 2026-09-10, a target 10 cm away came back 149.7 mm short. Iterating reaches it.
        pose = self.servo.fk(self.start).copy()
        pose[:3, 3] = pose[:3, 3] + self.np.array([0.0, 0.0, 0.10])
        solved = self.servo.solve_pose(self.start, pose)
        reached = self.servo.gripper_xyz(solved)
        self.assertLess(float(self.np.linalg.norm(reached - pose[:3, 3])) * 1000, 0.5)

    def test_a_target_one_step_ahead_of_the_measurement_cannot_move_this_arm(self):
        # ⛔ The fault this rewrite removed, kept as a test so it cannot come back. LeRobot's
        # end-effector steps take their reference from the observation they are handed, so
        # handing them the measurement makes the target `FK(measured) + one step` -- and the
        # solver then travels one step and no further, whatever is held down.
        # ⚠️ The number is the whole story: a Feetech STS3215 needs about 2.3 deg of position
        # error at P=32 before it makes useful force (upstream issue #3400), and this asks for
        # less than half of that. On hardware the arm simply did not move.
        pose = self.servo.fk(self.start).copy()
        pose[:3, 3] = pose[:3, 3] + self.np.array([0.0, 0.0, 0.002])
        commanded = self.servo.servo_step(self.start, pose)
        lead = max(abs(commanded[name] - self.start[name]) for name in demo.JOINTS)
        self.assertLess(lead, 2.3, "if this ever exceeds the servo's error band, say why")
        # ⭐ And the fix, measured the same way: a target that is where the operator actually
        # wants the gripper gets the whole budget instead.
        far = self.servo.fk(self.start).copy()
        far[:3, 3] = self.np.array(demo.DEFAULT_BOUNDS_M["max"])
        wanted = self.servo.servo_step(self.start, far)
        self.assertGreater(max(abs(wanted[name] - self.start[name]) for name in demo.JOINTS), 2.3)

    def test_wrist_roll_is_masked_out_of_the_problem(self):
        # 2026-09-08: position-only IK left the roll axis free, and a drag once planned 155 deg
        # of it. placo's mask_dof removes the joint from the solve.
        walking = {**dict(zip(demo.JOINTS, (13.98, -103.69, 97.01, -102.29, 6.37))), "gripper": 0.0}
        for delta in ((0.002, 0, 0.002), (0.002, 0.001, 0.002), (0.002, 0.002, 0.002)):
            pose = self.servo.fk(walking).copy()
            pose[:3, 3] = pose[:3, 3] + self.np.asarray(delta)
            walking = self.servo.servo_step(walking, pose)
            self.assertAlmostEqual(walking["wrist_roll"], 6.37, places=4)

    def test_a_commanded_pose_carries_one_target_per_motor(self):
        pose = self.servo.fk(self.start)
        commanded = self.servo.servo_step(self.start, pose)
        self.assertEqual(set(commanded), set(demo.MOTORS))
        self.assertTrue(all(math.isfinite(value) for value in commanded.values()))

    def test_the_chased_pose_never_leaves_the_workspace(self):
        # The box is a hard edge for the target, so a handle outside it cannot walk the arm out.
        outside = self.servo.fk(self.start).copy()
        outside[:3, 3] = self.np.array([2.0, 2.0, 2.0])
        clamped = demo.clamp_into_bounds(outside, self.bounds)
        self.assertTrue((clamped[:3, 3] <= demo.bounds_high(self.bounds) + 1e-9).all())
        self.assertTrue((clamped[:3, 3] >= demo.bounds_low(self.bounds) - 1e-9).all())

    def test_the_chased_pose_stops_ahead_of_the_gripper(self):
        # ⭐ The second limit on force, and the one that binds first. Measured 2026-09-10: at
        # 20 mm a reaching-out arm is only commanded 1.72 deg, below what this servo needs.
        here = self.servo.gripper_xyz(self.start)
        far = self.servo.fk(self.start).copy()
        far[:3, 3] = self.np.array(demo.DEFAULT_BOUNDS_M["max"])
        reference = self.servo.fk(self.start)
        for _tick in range(500):
            reference = demo.advance_reference(reference, far, here)
        lead = float(self.np.linalg.norm(reference[:3, 3] - here)) * 1000
        self.assertLessEqual(lead, demo.MAX_REF_LEAD_MM + 1e-6)
        self.assertGreater(lead, demo.MAX_REF_LEAD_MM - 1.0, "it must actually reach its limit")

    def test_the_gripper_moves_one_step_at_a_time_and_stops_on_its_target(self):
        self.assertAlmostEqual(demo.step_towards(50.0, 80.0, 1.0), 51.0)
        self.assertAlmostEqual(demo.step_towards(50.0, 20.0, 1.0), 49.0)
        self.assertAlmostEqual(demo.step_towards(50.0, 50.4, 1.0), 50.4, msg="never overshoot")
        self.assertAlmostEqual(demo.step_towards(50.0, 50.0, 1.0), 50.0)

    def test_preview_reports_the_solver_it_used_and_its_own_residual(self):
        import contextlib, io as _io
        printed = _io.StringIO()
        with contextlib.redirect_stdout(printed):
            demo.preview(SimpleNamespace(model_dir=args.model_dir,
                                         joints_deg=demo.PREVIEW_JOINTS_DEG,
                                         delta_mm=(0.0, 0.0, 20.0), max_ticks=500))
        report = json.loads(printed.getvalue())
        self.assertIn("placo", report["solver"])
        self.assertEqual(report["mode"], "model-only; no hardware")
        self.assertLess(report["position_error_mm"], 0.1)
        # ⭐ Both halves: the answer, and the number of ticks the arm would really take to walk
        # there. 20 mm at the set speed is 0.4 s, and saying so is the point of the lesson.
        self.assertAlmostEqual(report["seconds_to_arrive"],
                               0.020 / demo.REF_LINEAR_SPEED_MPS, places=1)

    def test_viser_meshes_load_and_agree_with_the_solver(self):
        import yourdfpy
        from viser.extras import ViserUrdf
        urdf = yourdfpy.URDF.load(str(Path(args.model_dir) / demo.URDF_NAME))
        names = [joint.name for joint in urdf.actuated_joints]
        self.assertEqual(set(names), set(demo.MOTORS))
        # The viewer is fed by name, because this order is not the bus order.
        values = visual.viewer_configuration(names, self.start, (-0.2, 1.7))
        self.assertEqual(len(values), len(names))
        self.assertAlmostEqual(values[names.index("shoulder_lift")],
                               math.radians(self.start["shoulder_lift"]))
        self.assertTrue(hasattr(ViserUrdf, "update_cfg"))

    def test_the_page_builds_on_a_real_server_and_never_reaches_for_an_arm(self):
        import socket
        with socket.socket() as probe:
            probe.bind((visual.LOOPBACK, 0))
            port = probe.getsockname()[1]
        page = visual.ViserPage(args.model_dir, port, self.servo, self.bounds,
                                (demo.HANDLE, demo.KEYBOARD), live=False)
        try:
            page.start(self.start)
            self.assertIsNone(page.take_arm_request(), "nothing is asked for until a click")
            page._request_arm()
            self.assertEqual(page.take_arm_request(), demo.HANDLE)
            self.assertIsNone(page.take_arm_request(), "a wish is taken once")
            page.armed(demo.HANDLE)
            page._request_end()
            self.assertTrue(page.take_end_request())
            page.set_gripper_target(12.0)
            self.assertAlmostEqual(page.gripper_target(), 12.0)
            page.move_handle((0.25, 0.0, 0.2))
            self.assertAlmostEqual(float(page.handle_xyz()[0]), 0.25, places=3)
        finally:
            page.stop()


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


class CameraCheckTests(unittest.TestCase):
    """The camera program's judgements, exercised without a camera attached.

    Everything tested here decides something an operator would otherwise have to notice by
    eye: which path names a camera, and whether two configured streams are fit to record.
    """

    def configured(self, **overrides):
        entry = {"path": "/dev/video0", "width": 1280, "height": 720,
                 "fps": 30, "fourcc": "MJPG", "rotation": 0}
        return {name: {**entry, **overrides.get(name, {})} for name in ("top", "wrist")}

    def measured(self, fps=30.0, identical=0, frames=90):
        return {name: {"fps": fps, "identical": identical, "frames": frames,
                       "width": 1280, "height": 720} for name in ("top", "wrist")}

    def test_by_id_is_preferred_only_while_it_names_one_camera(self):
        links = {"/dev/video4": [("by-id", Path("/dev/v4l/by-id/usb-Model-video-index0")),
                                 ("by-path", Path("/dev/v4l/by-path/pci-0-usb-0:4.3:1.0-video-index0"))]}
        alone, _ = cameras.preferred_path(Path("/dev/video4"), links, model_is_duplicated=False)
        self.assertEqual(alone.parent.name, "by-id")
        # Two cameras of one model: udev keeps a single by-id link and it points at whichever
        # enumerated last, so the link that exists is the wrong thing to write down.
        duplicated, reason = cameras.preferred_path(Path("/dev/video4"), links, model_is_duplicated=True)
        self.assertEqual(duplicated.parent.name, "by-path")
        self.assertIn("serial", reason)

    def test_a_camera_with_no_link_at_all_reports_the_bare_number_as_unstable(self):
        path, reason = cameras.preferred_path(Path("/dev/video9"), {}, model_is_duplicated=False)
        self.assertEqual(str(path), "/dev/video9")
        self.assertIn("changes when the camera is replugged", reason)

    def test_two_entries_on_one_camera_are_named_before_anything_is_opened(self):
        # One camera entered twice also fails to open the second time. That failure must not
        # be reported as "could not be opened", which sends the operator to the wrong problem.
        verdict, detail = cameras.verdict(self.configured(), {}, {"top": "usb-1-4.4", "wrist": "usb-1-4.4"})
        self.assertEqual(verdict, "SAME CAMERA")
        self.assertIn("usb-1-4.4", detail)

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
