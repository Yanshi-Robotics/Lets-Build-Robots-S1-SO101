"""Model-level checks for the Cartesian pipeline. No hardware, no browser needed.

    python Cartesian/test_cartesian.py --model-dir models/so101
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

import numpy as np
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))

import planner  # noqa: E402
from runlog import KEEP_RUNS, RunLog  # noqa: E402
from so101_arm import FakeArm, deg_from_q, q_from_deg  # noqa: E402
from so101_model import ARM_JOINTS, Model, calibration_half_travel_deg, rot_z  # noqa: E402
from solver import Solver  # noqa: E402
from target import HOLD, MODE_DRAG, MODE_GAMEPAD, MODE_LEADER, STOP, CommandBox, JointCommand, Target, TargetBox  # noqa: E402
from viewer import _rotation_about_z  # noqa: E402

MODEL_DIR: Path | None = None
PREVIEW_Q = np.array([0.0, math.radians(-30), math.radians(60), math.radians(-30), 0.0, 30.0])


def model() -> Model:
    return Model(MODEL_DIR)


class ModelTests(unittest.TestCase):
    def test_zero_pose_matches_urdf(self):
        T = model().fk(np.zeros(6))
        np.testing.assert_allclose(T[:3, 3], [0.3914, 0.0, 0.2265], atol=1e-3)
        np.testing.assert_allclose(T[:3, 2], [1.0, 0.0, 0.0], atol=1e-4)

    def test_three_pitch_axes_are_parallel(self):
        m = model()
        R0 = m.fk(np.zeros(6))[:3, :3]
        R1 = m.fk(np.array([0.0, -0.5, 1.0, -0.5, 0.0, 0.0]))[:3, :3]
        np.testing.assert_allclose(R0, R1, atol=1e-6)

    def test_tool_yaw_follows_pan(self):
        m = model()
        q = np.zeros(6)
        q[0] = 0.7
        yaw_from_axis = m.decompose(m.fk(q)[:3, :3])[0]
        self.assertAlmostEqual(yaw_from_axis, m.tool_yaw(q), places=4)
        self.assertAlmostEqual(abs(m.tool_yaw(q)), 0.7, places=6)

    def test_compose_decompose_roundtrip(self):
        m = model()
        rng = np.random.default_rng(0)
        for _ in range(50):
            yaw, pitch, roll = rng.uniform(-3.0, 3.0), rng.uniform(-1.4, 1.4), rng.uniform(-2.5, 2.5)
            got = m.decompose(m.compose(yaw, pitch, roll))
            np.testing.assert_allclose(got, (yaw, pitch, roll), atol=1e-4)

    def test_decompose_with_known_yaw_handles_straight_down(self):
        m = model()
        yaw, pitch, roll = 0.4, math.pi / 2, 0.3
        got = m.decompose(m.compose(yaw, pitch, roll), yaw=yaw)
        np.testing.assert_allclose(got, (yaw, pitch, roll), atol=1e-4)

    def test_positive_pitch_points_down(self):
        d = model().compose(0.0, 0.5, 0.0)[:, 2]
        self.assertLess(d[2], 0.0)

    def test_positive_roll_is_positive_wrist_roll(self):
        m = model()
        q = PREVIEW_Q.copy()
        q[4] = 0.8
        self.assertAlmostEqual(m.target_from_q(q).roll, 0.8, places=4)

    def test_calibration_half_travel(self):
        cal = {"shoulder_pan": {"range_min": 635, "range_max": 3375}}
        self.assertAlmostEqual(calibration_half_travel_deg(cal)["shoulder_pan"], 1370 * 360 / 4095, places=6)

    def test_calibration_narrows_limits(self):
        cal = {"shoulder_pan": {"id": 1, "drive_mode": 0, "homing_offset": 0, "range_min": 1500, "range_max": 2600}}
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(cal, f)
        m = Model(MODEL_DIR, Path(f.name))
        half = math.radians(550 * 360 / 4095)
        self.assertAlmostEqual(m.limits["shoulder_pan"][1], half, places=6)
        self.assertAlmostEqual(m.limits["shoulder_lift"][1], Model(MODEL_DIR).limits["shoulder_lift"][1])


class SolverTests(unittest.TestCase):
    def setUp(self):
        self.m = model()
        self.s = Solver(self.m)

    def check(self, target: Target):
        sol = self.s.solve(PREVIEW_Q, target)
        self.assertTrue(sol.converged, sol)
        self.assertLess(sol.error.position_m, 0.001)
        self.assertLess(abs(sol.error.pitch_rad), math.radians(0.5))
        self.assertLess(abs(sol.error.roll_rad), math.radians(0.5))
        for i, name in enumerate(ARM_JOINTS):
            lo, hi = self.m.limits[name]
            self.assertTrue(lo - 1e-6 <= sol.q_goal[i] <= hi + 1e-6, (name, sol.q_goal[i]))
        self.assertEqual(sol.q_goal[5], target.gripper_pct)
        return sol

    def test_reachable_targets(self):
        for xyz, pitch, roll in [
            ((0.25, 0.05, 0.15), 0, 0), ((0.25, 0.05, 0.15), 60, 0), ((0.25, 0.05, 0.15), 60, 45),
            ((0.15, -0.2, 0.05), 90, 0), ((0.15, -0.2, 0.05), 90, 30), ((0.1, 0.25, 0.02), 90, 0),
            ((0.2, 0.0, 0.3), -60, 0), ((0.3, 0.1, 0.1), 30, -90),
        ]:
            with self.subTest(xyz=xyz, pitch=pitch, roll=roll):
                self.check(Target(xyz, math.radians(pitch), math.radians(roll), 30.0))

    def test_solution_reproduces_target_through_fk(self):
        target = Target((0.2, -0.1, 0.08), math.radians(75), math.radians(20), 30.0)
        sol = self.check(target)
        back = self.m.target_from_q(sol.q_goal)
        np.testing.assert_allclose(back.xyz, target.xyz, atol=1e-3)
        self.assertAlmostEqual(back.pitch, target.pitch, places=2)
        self.assertAlmostEqual(back.roll, target.roll, places=2)

    def test_roll_only_change_keeps_position(self):
        base = self.m.target_from_q(PREVIEW_Q)
        sol = self.check(Target(base.xyz, base.pitch, base.roll + 1.0, 30.0))
        self.assertLess(sol.error.position_m, 0.001)

    def test_unreachable_target_reports_residual(self):
        sol = self.s.solve(PREVIEW_Q, Target((0.8, 0.0, 0.1), 0.0, 0.0, 30.0))
        self.assertFalse(sol.converged)
        self.assertGreater(sol.error.position_m, 0.1)
        self.assertTrue(np.all(np.isfinite(sol.q_goal)))
        self.assertFalse(sol.error.reachable())


class PlannerTests(unittest.TestCase):
    limits = planner.Limits(max_speed=np.array([2.0] * 5 + [150.0]), max_lead=np.array([math.radians(15)] * 5 + [100.0]))

    def test_speed_limit(self):
        q = np.zeros(6)
        goal = np.array([1.0, -1.0, 0.001, 0.0, 0.0, 100.0])
        st = planner.step(q, goal, q, 0.02, self.limits)
        np.testing.assert_allclose(st.q_next[:2], [0.04, -0.04])
        self.assertAlmostEqual(st.q_next[2], 0.001)
        self.assertAlmostEqual(st.q_next[5], 3.0)
        self.assertTrue(st.speed_limited)
        self.assertFalse(st.lead_limited)

    def test_lead_clamp(self):
        q_cmd = np.array([math.radians(14.5), 0, 0, 0, 0, 0.0])
        goal = np.array([1.0, 0, 0, 0, 0, 0.0])
        meas = np.zeros(6)
        st = planner.step(q_cmd, goal, meas, 0.02, self.limits)
        self.assertAlmostEqual(st.q_next[0], math.radians(15))
        self.assertTrue(st.lead_limited)

    def test_at_goal_is_a_no_op(self):
        q = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 40.0])
        st = planner.step(q, q, q, 0.02, self.limits)
        np.testing.assert_allclose(st.q_next, q)
        self.assertFalse(st.speed_limited or st.lead_limited)


class ViewerMathTests(unittest.TestCase):
    def test_rotation_about_z(self):
        import viser.transforms as tf
        base = np.array([0.7, 0.1, -0.3, 0.2])
        base = base / np.linalg.norm(base)
        R = tf.SO3(base).as_matrix()
        for angle in (0.3, -1.2, 3.0, -3.1):
            now = tf.SO3.from_matrix(R @ rot_z(angle)).wxyz
            self.assertAlmostEqual(_rotation_about_z(base, now), angle, places=6)


class ArmTests(unittest.TestCase):
    def test_deg_roundtrip(self):
        deg = {"shoulder_pan": 10.0, "shoulder_lift": -20.0, "elbow_flex": 30.0, "wrist_flex": -40.0, "wrist_roll": 50.0, "gripper": 60.0}
        for k, v in deg_from_q(q_from_deg(deg)).items():
            self.assertAlmostEqual(v, deg[k])

    def test_fake_arm_echoes(self):
        arm = FakeArm()
        arm.send_deg({"shoulder_pan": 1.0, "shoulder_lift": 2.0, "elbow_flex": 3.0, "wrist_flex": 4.0, "wrist_roll": 5.0, "gripper": 6.0})
        self.assertEqual(arm.read_deg()["wrist_roll"], 5.0)


class RunLogTests(unittest.TestCase):
    def test_writes_events_and_ticks_and_prunes(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for i in range(KEEP_RUNS + 3):
                (root / f"2000-01-01_{i:06d}_model").mkdir()
            log = RunLog(root, "model")
            log.event("hello", value=1.23456789)
            log.tick({"tick": 0, "q": np.array([0.1, 0.2]), "nested": {"a": 1.0}})
            log.close()
            self.assertTrue((root / "latest").is_symlink())
            lines = (log.dir / "ticks.jsonl").read_text().splitlines()
            rec = json.loads(lines[0])
            self.assertEqual(rec["q"], [0.1, 0.2])
            self.assertIn("t", rec)
            self.assertIn("hello", (log.dir / "run.log").read_text())
            runs = [p for p in root.iterdir() if p.is_dir() and not p.is_symlink()]
            self.assertEqual(len(runs), KEEP_RUNS + 1)


class NullViewer:
    """Same surface as viewer.Viewer, draws nothing."""

    def __init__(self):
        self.synced: list[Target] = []
        self.pushes = 0
        self.mode = MODE_DRAG
        self.statuses: list[tuple] = []

        self.page = MODE_DRAG
        self.follower = None
        self.pairing_views = []
        self.ghosts = []
        self.pair_button = SimpleNamespace(visible=False)

    def set_source_status(self, mode, ok, text):
        self.statuses.append((mode, ok, text))

    def show_pad(self, show):
        pass

    def show_pairing(self, view):
        self.pairing_views.append(view)

    def show_pad_state(self, values, held, note, active):
        pass

    def show_ghost(self, q):
        self.ghosts.append(q)

    def sync_target(self, target: Target) -> None:
        self.synced.append(target)

    def push(self, *args, **kwargs) -> None:
        self.pushes += 1


class FakeLiveArm(FakeArm):
    mode = "live"


class ControlLoopTests(unittest.TestCase):
    class TestClock:
        """A monotonic clock the control loop and the test advance in lockstep.

        ⛔ Why this exists rather than sleeping on the wall clock: the loop runs a real
        100 Hz thread, so how far it gets in 1.8 s of wall time depends on how much CPU that
        thread was given. On the macOS CI runner it got about a quarter of the rate and four
        assertions about distance travelled failed — the program was fine, the measurement
        was not. Here the loop may only run up to the instant the test has granted, so the
        same run produces the same numbers on any machine, as fast as the CPU allows.

        The loop reads the clock many times per tick and sleeps once at the end of one, so
        one sleep is one tick of virtual time.
        """

        # ⚠️ Virtual time accumulates one dt at a time, so the two comparisons below need a
        # tolerance: without it the loop's next target lands a hair above the instant the test
        # granted and both sides wait for each other. A microsecond is far under any interval
        # this loop deals in (a tick is 10 ms) and far over the accumulated float error.
        EPS = 1e-6

        # Starts away from zero, as a real monotonic clock does: the loop compares `now`
        # against fields initialised to 0.0 (the gamepad poll stamp), and starting at zero
        # would quietly change which branch runs on the first tick. ⛔ Not 1e6 either —
        # adding 0.01 to a number that large throws away the precision this needs.
        def __init__(self, start=1000.0):
            self.start = start
            self._t = start
            self._granted = start
            self._released = False
            self._cv = threading.Condition()

        def now(self):
            with self._cv:
                return self._t

        def sleep(self, seconds):
            """The loop's own sleep. It blocks until the test grants that much virtual time."""
            with self._cv:
                target = self._t + max(seconds, 0.0)
                while not self._released and target > self._granted + self.EPS:
                    if not self._cv.wait(timeout=30.0):
                        raise AssertionError("the test never granted more virtual time")
                self._t = target
                self._cv.notify_all()

        def run_to(self, instant, timeout=60.0):
            """Let the loop run up to `instant` (seconds from start) and wait until it is there."""
            goal = self.start + instant
            deadline = time.monotonic() + timeout
            with self._cv:
                self._granted = max(self._granted, goal)
                self._cv.notify_all()
                while self._t < goal - self.EPS:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0 or not self._cv.wait(timeout=remaining):
                        raise AssertionError(
                            f"the control loop stalled at t={self._t - self.start:.3f}s "
                            f"while the test waited for t={instant:.3f}s")

        def release(self):
            """Stop holding the loop back, so stop() can join the thread."""
            with self._cv:
                self._released = True
                self._cv.notify_all()

    def test_the_real_clock_is_what_the_program_runs_on(self):
        """Every other test here hands the loop a virtual clock, so this one guards the default.

        ⛔ Without it, someone could make the injected clock mandatory, or point the default at
        the virtual one, and every test would still pass while the shipped program no longer
        slept between ticks.
        """
        from cartesian_control import ControlLoop
        m = model()
        loop = ControlLoop(m, Solver(m), None, planner.Limits(
            max_speed=np.array([2.0] * 5 + [150.0]), max_lead=np.array([math.radians(15)] * 5 + [100.0])),
            100.0, 100.0, TargetBox(m.target_from_q(q_from_deg(FakeArm().read_deg()))), CommandBox(),
            NullViewer(), RunLog(Path(tempfile.mkdtemp()), "defaults"))
        self.assertIs(loop.clock, time.monotonic)
        self.assertIs(loop.sleep, time.sleep)
        # The log stamps its ticks from the same clock, and by default that is the real one too.
        self.assertIs(RunLog(Path(tempfile.mkdtemp()), "defaults")._clock, time.monotonic)

    def run_loop(self, arm, seconds, actions, gamepad=None, pad=None, leader=None, mode=MODE_DRAG, map_path=None):
        """arm: a FakeLiveArm standing in for the real one, or None for simulation only.
        actions: list of (time_offset_s, callable(targets, commands))."""
        from cartesian_control import ControlLoop
        m = model()
        s = Solver(m)
        limits = planner.Limits(max_speed=np.array([2.0] * 5 + [150.0]), max_lead=np.array([math.radians(15)] * 5 + [100.0]))
        if arm is not None:
            arm.connect()
        q0 = q_from_deg(arm.read_deg() if arm is not None else FakeArm().read_deg())
        targets, commands = TargetBox(m.target_from_q(q0)), CommandBox()
        viewer = NullViewer()
        viewer.mode = mode
        tmp = tempfile.mkdtemp()
        clock = self.TestClock()
        # ⭐ Same clock for the log: the tests filter ticks by the `t` they recorded.
        log = RunLog(Path(tmp), "test", clock=clock.now)
        loop = ControlLoop(m, s, arm, limits, 100.0, 100.0, targets, commands, viewer, log,
                           gamepad_map_path=map_path, leader_factory=(lambda: leader) if leader else None,
                           clock=clock.now, sleep=clock.sleep)
        loop.pad, loop.gamepad = pad, gamepad     # an already-open, paired pad, as hot-plug would leave it
        loop.start()
        # Each action lands at a tick boundary, with the loop held at exactly that instant.
        for offset, action in actions:
            clock.run_to(offset)
            action(targets, commands)
        clock.run_to(seconds)
        clock.release()          # ⛔ before stop(), or the loop is still waiting and join times out
        loop.stop()
        log.close()
        self.assertTrue(loop.thread.is_alive() is False)
        ticks = [json.loads(l) for l in (log.dir / "ticks.jsonl").read_text().splitlines()]
        return loop, viewer, ticks, m

    def test_model_arm_walks_to_target_at_bounded_speed(self):
        target = Target((0.25, 0.10, 0.12), math.radians(45), math.radians(20), 60.0)
        loop, viewer, ticks, m = self.run_loop(
            None, 1.5, [(0.2, lambda t, c: t.set(target, "test"))])
        self.assertGreater(len(ticks), 100)
        solved = [t for t in ticks if "solve" in t]
        self.assertEqual(len(solved), 1)
        self.assertTrue(solved[0]["solve"]["converged"])
        # every step within the speed limit (2 rad/s at 100 Hz = 0.02 rad, in degrees 1.146)
        for a, b in zip(ticks, ticks[1:]):
            for name in ARM_JOINTS:
                self.assertLessEqual(abs(b["q_cmd_deg"][name] - a["q_cmd_deg"][name]), math.degrees(0.02) + 1e-3)  # 1e-3: log rounding
        last = ticks[-1]
        self.assertLess(last["err_mm"]["cmd"], 1.0)
        self.assertLess(last["err_mm"]["meas"], 1.0)
        self.assertTrue(last["reachable"])
        self.assertAlmostEqual(last["q_cmd_deg"]["gripper"], 60.0, places=3)

    def test_ring_moves_one_joint_and_syncs_the_ball(self):
        loop, viewer, ticks, m = self.run_loop(
            None, 1.2, [(0.2, lambda t, c: c.push_joint(JointCommand("shoulder_pan", 0.5)))])
        ringed = [t for t in ticks if "ring" in t]
        self.assertEqual(len(ringed), 1)
        self.assertEqual(ringed[0]["target"]["source"], "ring")
        self.assertEqual(len([t for t in ticks if "solve" in t]), 0)  # a ring sets the goal directly
        last = ticks[-1]
        self.assertAlmostEqual(last["q_goal_deg"]["shoulder_pan"], math.degrees(0.5), places=3)
        self.assertAlmostEqual(last["q_cmd_deg"]["shoulder_pan"], math.degrees(0.5), places=2)
        self.assertAlmostEqual(last["q_goal_deg"]["elbow_flex"], 60.0, places=3)
        self.assertGreaterEqual(len(viewer.synced), 2)  # initial + ring

    def test_gamepad_mode_pushes_the_target(self):
        from gamepad_control import GamepadSource, LINEAR_SPEED_MPS
        pad = FakePad()
        mapping = {"axes": {"reach": {"axis": 0, "sign": 1, "kind": "stick", "rest": 0.0},
                            "waist_left": {"axis": 4, "sign": 1, "kind": "trigger", "rest": -1.0}},
                   "buttons": {"gripper_open": 1, "hold": 11, "stop": 10}}
        source = GamepadSource(pad, mapping)

        def push(t, c):
            pad.axes[0] = 1.0           # left stick right: reach out, i.e. +x while the arm points along +x

        def release(t, c):
            pad.axes[0] = 0.0
            pad.press(1, True)          # hold B ...

        def let_go(t, c):
            pad.press(1, False)         # ... for 0.5 s, then let go: the gripper must stop where it is

        loop, viewer, ticks, m = self.run_loop(None, 1.8, [(0.2, push), (0.7, release), (1.2, let_go)],
                                               gamepad=source, pad=pad, mode=MODE_GAMEPAD)
        from gamepad_control import GRIPPER_SPEED_PCT_S
        start = ticks[0]["target"]["xyz"][0]
        end = ticks[-1]["target"]["xyz"][0]
        self.assertAlmostEqual(end - start, LINEAR_SPEED_MPS * 0.5, delta=0.006)   # 0.5 s at full stick
        self.assertEqual(ticks[-1]["target"]["source"], "gamepad")
        self.assertAlmostEqual(ticks[-1]["target"]["gripper_pct"], 30.0 + GRIPPER_SPEED_PCT_S * 0.5, delta=4.0)
        self.assertLess(ticks[-1]["target"]["gripper_pct"], 100.0)                  # did not run to the end
        self.assertGreater(len([t for t in ticks if "solve" in t]), 5)              # solved as the stick moved it
        self.assertLess(ticks[-1]["err_mm"]["cmd"], 1.0)

    def test_leader_mode_copies_joints_without_solving(self):
        class FakeLeader:
            deg = dict(FakeArm().read_deg())

            def connect(self):
                pass

            def read_deg(self):
                return dict(self.deg)

        leader = FakeLeader()

        def move(t, c):
            leader.deg["shoulder_pan"] = 25.0
            leader.deg["gripper"] = 80.0

        loop, viewer, ticks, m = self.run_loop(None, 1.5, [(0.2, move)], leader=leader, mode=MODE_LEADER)
        self.assertEqual(len([t for t in ticks if "solve" in t]), 0)
        self.assertEqual(ticks[-1]["target"]["source"], "leader")
        self.assertAlmostEqual(ticks[-1]["q_goal_deg"]["shoulder_pan"], 25.0, places=3)
        self.assertAlmostEqual(ticks[-1]["q_cmd_deg"]["shoulder_pan"], 25.0, places=2)
        self.assertAlmostEqual(ticks[-1]["q_cmd_deg"]["gripper"], 80.0, places=2)

    def test_switching_to_the_simulated_follower_leaves_the_real_arm_alone(self):
        from target import FOLLOWER_REAL, FOLLOWER_SIM
        target = Target((0.25, 0.10, 0.12), math.radians(45), 0.0, 30.0)
        arm = FakeLiveArm()
        sent = []
        original = arm.send_deg
        arm.send_deg = lambda goal: (sent.append(dict(goal)), original(goal))
        loop, viewer, ticks, m = self.run_loop(arm, 2.6, [
            (0.2, lambda t, c: c.push_button(f"follower:{FOLLOWER_SIM}")),
            (0.4, lambda t, c: t.set(target, "test")),
            (1.6, lambda t, c: c.push_button(f"follower:{FOLLOWER_REAL}")),
        ])
        self.assertEqual(sent, [])                                    # nothing ever written to the real arm
        self.assertFalse(arm.torque_on)
        sim = [t for t in ticks if t["follower"] == "sim"]
        self.assertTrue(sim and all(t["following"] for t in sim))
        self.assertLess(sim[-1]["err_mm"]["cmd"], 1.0)                # the simulation reached the target
        back = [t for t in ticks if t["follower"] == "real" and t["t"] > ticks[0]["t"] + 1.7]
        self.assertTrue(back and not any(t["following"] for t in back))   # real again: waits for Hold
        self.assertEqual(back[-1]["target"]["source"], "arm")            # and the ball went back to the real pose

    def test_unpaired_pad_pairs_inside_the_gamepad_page(self):
        from gamepad_pairing import STEPS
        pad = FakePad()
        pad.axes[4] = pad.axes[5] = -1.0
        tmp = Path(tempfile.mkdtemp()) / "map.json"

        def on_gamepad_page(t, c):
            self.current_viewer.page = MODE_GAMEPAD
            c.push_button("page:Gamepad")

        actions = [(0.2, on_gamepad_page)]
        # walk the ten steps: each axis push lasts 0.25 s, then 0.25 s rest
        moves = [(1, -1.0), (0, 1.0), (3, -1.0), (2, 1.0), (4, 1.0), (5, 1.0)]
        t = 0.6
        for idx, value in moves:
            actions.append((t, lambda tt, c, i=idx, v=value: pad.axes.__setitem__(i, v)))
            actions.append((t + 0.25, lambda tt, c, i=idx: pad.axes.__setitem__(i, -1.0 if i in (4, 5) else 0.0)))
            t += 0.5
        for idx in (1, 2, 7, 6):
            actions.append((t, lambda tt, c, i=idx: pad.press(i, True)))
            actions.append((t + 0.15, lambda tt, c, i=idx: pad.press(i, False)))
            t += 0.4
        self.current_viewer = None
        from cartesian_control import ControlLoop  # noqa: F401
        orig = self.run_loop

        def run(*a, **k):
            return orig(*a, **k)
        # the viewer object is created inside run_loop; grab it through a small hook
        NullViewer_init = NullViewer.__init__

        def hooked(v):
            NullViewer_init(v)
            self.current_viewer = v
        NullViewer.__init__ = hooked
        try:
            loop, viewer, ticks, m = run(None, t + 0.5, actions, pad=pad, map_path=tmp)
        finally:
            NullViewer.__init__ = NullViewer_init
        self.assertTrue(tmp.exists(), "pairing wrote the map")
        written = json.loads(tmp.read_text())["pads"]["fake"]
        self.assertEqual(set(written["axes"]), {s.key for s in STEPS if s.kind == "axis"})
        self.assertEqual(written["buttons"], {"gripper_open": 1, "gripper_close": 2, "hold": 7, "stop": 6})
        self.assertIsNotNone(loop.gamepad)                            # usable right away, no restart
        self.assertTrue(any(v is not None and not v.done for v in viewer.pairing_views))
        self.assertTrue(any(q is not None for q in viewer.ghosts))   # the demo played on the ghost arm
        self.assertEqual(viewer.statuses[-1][:2], (MODE_GAMEPAD, True))

    def test_hold_refused_without_a_mode(self):
        arm = FakeLiveArm()
        loop, viewer, ticks, m = self.run_loop(arm, 0.8, [(0.2, lambda t, c: c.push_button(HOLD))], mode=None)
        self.assertFalse(any(t["following"] for t in ticks))
        self.assertFalse(arm.torque_on)

    def test_live_arm_ignores_ball_until_hold(self):
        target = Target((0.25, 0.10, 0.12), math.radians(45), 0.0, 30.0)
        arm = FakeLiveArm()
        loop, viewer, ticks, m = self.run_loop(arm, 2.4, [
            (0.2, lambda t, c: t.set(target, "test")),
            (0.5, lambda t, c: c.push_button(HOLD)),
            (0.7, lambda t, c: t.set(target, "test")),
            (2.1, lambda t, c: c.push_button(STOP)),
        ])
        before_hold = [t for t in ticks if not t["following"] and t["t"] < ticks[0]["t"] + 0.45]
        self.assertTrue(before_hold)
        self.assertTrue(all(t["q_cmd_deg"] == t["q_meas_deg"] for t in before_hold))
        self.assertTrue(all(not t["torque"] for t in before_hold))
        after_hold = [t for t in ticks if t["following"]]
        self.assertTrue(after_hold)
        self.assertTrue(all(t["torque"] for t in after_hold))
        # "hold" re-synced the target to the arm, so the first set was dropped and the second solved
        self.assertGreaterEqual(len([t for t in ticks if "solve" in t]), 1)
        self.assertLess(after_hold[-1]["err_mm"]["cmd"], 2.0)
        stopped = [t for t in ticks if "button" in t and t["button"] == STOP]
        self.assertEqual(len(stopped), 1)
        self.assertFalse(ticks[-1]["following"])
        self.assertTrue(ticks[-1]["torque"])


class FakePad:
    """Stands in for gamepad.Gamepad: the test sets axes/buttons and queues changes."""

    def __init__(self, n_axes=8, n_buttons=16):
        self.name = "fake"
        self.connected = True
        self.error = None
        self.axes = [0.0] * n_axes
        self.buttons = [False] * n_buttons
        self.axes[4] = self.axes[5] = -1.0   # triggers rest at -1
        self._changes = []

    def snapshot(self):
        return list(self.axes), list(self.buttons)

    def drain(self):
        out, self._changes = self._changes, []
        return out

    def press(self, idx, down=True):
        from gamepad import Change
        self.buttons[idx] = down
        self._changes.append(Change("button", idx, 1.0 if down else 0.0))

    def stop(self):
        pass


class GamepadMapFileTests(unittest.TestCase):
    def test_per_device_read_write_and_old_format(self):
        from gamepad_control import load_map, read_maps, write_map
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "map.json"
            self.assertIsNone(load_map(path, "any"))
            axes = {k: {"axis": i, "sign": 1, "kind": "stick", "rest": 0.0} for i, k in enumerate(["move_z", "reach", "pitch", "roll", "waist_left", "waist_right"])}
            buttons = {"gripper_open": 1, "gripper_close": 2, "hold": 7, "stop": 6}
            write_map(path, "Old", {"axes": axes, "buttons": buttons})
            write_map(path, "New", {"axes": {"reach": axes["reach"]}, "buttons": {"stop": 6}})   # an older layout
            self.assertEqual(set(read_maps(path)), {"Old", "New"})               # both kept on file ...
            self.assertEqual(load_map(path, "Old")["buttons"], buttons)
            self.assertIsNone(load_map(path, "New"))                             # ... but the incomplete one is "not paired"
            self.assertIsNone(load_map(path, "Unknown"))


class DofTrajectoryTests(unittest.TestCase):
    """The clips behind Lesson 8's six-degree-of-freedom animations (Cartesian/6dof-demo)."""

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(Path(__file__).resolve().parent / "6dof-demo"))
        import trajectories
        cls.t = trajectories
        cls.data = trajectories.build(MODEL_DIR, None)      # SO-101 only; the Panda needs a download

    def test_translation_clips_move_one_axis_only(self):
        for axis, key in enumerate(("x", "y", "z")):
            P = np.array(self.data["clips"][key]["tool_position"])
            span = P.max(axis=0) - P.min(axis=0)
            self.assertAlmostEqual(span[axis], 2 * self.t.TRAVEL_M, delta=0.002, msg=key)
            for other in range(3):
                if other != axis:
                    self.assertLess(span[other], 0.001, f"{key} leaked into axis {other}")

    def test_orientation_clips_hold_the_tool_point(self):
        m = Model(MODEL_DIR)
        for key, amplitude in (("pitch", self.t.PITCH_RAD), ("roll", self.t.ROLL_RAD)):
            clip = self.data["clips"][key]
            P = np.array(clip["tool_position"])
            self.assertLess(np.max(P.max(axis=0) - P.min(axis=0)), 0.002, key)
            angles = [m.decompose(np.array(R), yaw=m.tool_yaw(np.array(q)))[1 if key == "pitch" else 2]
                      for R, q in zip(clip["tool_rotation"], clip["frames"])]
            self.assertAlmostEqual(max(angles) - min(angles), 2 * amplitude, delta=math.radians(1.0), msg=key)

    def test_clips_are_ping_pong_loops(self):
        for key, clip in self.data["clips"].items():
            frames = np.array(clip["frames"])
            self.assertEqual(len(frames), round(self.t.FPS * self.t.CLIP_SECONDS))
            # the sine wave returns to zero: the last frame is one step short of the first, within one step of travel
            self.assertLess(np.linalg.norm(np.array(clip["tool_position"][-1]) - np.array(clip["tool_position"][0])), 0.012, key)


class GamepadIntegrateTests(unittest.TestCase):
    def test_waist_turns_about_the_base_and_reach_slides_out(self):
        from gamepad_control import GamepadState, WAIST_SPEED_RAD_S, LINEAR_SPEED_MPS, integrate
        t = Target((0.3, 0.0, 0.15), 0.0, 0.0, 30.0)
        turned = integrate(t, GamepadState(waist_rate=1.0), 0.5, (-1, -1, -1), (1, 1, 1), (-3, 3))
        self.assertAlmostEqual(math.hypot(*turned.xyz[:2]), 0.3, places=9)              # same reach
        self.assertAlmostEqual(math.atan2(turned.xyz[1], turned.xyz[0]), WAIST_SPEED_RAD_S * 0.5, places=9)
        out = integrate(t, GamepadState(reach_rate=1.0), 0.5, (-1, -1, -1), (1, 1, 1), (-3, 3))
        self.assertAlmostEqual(out.xyz[0], 0.3 + LINEAR_SPEED_MPS * 0.5, places=9)
        self.assertEqual(out.xyz[1], 0.0)
        up = integrate(t, GamepadState(vz=-1.0), 0.5, (-1, -1, -1), (1, 1, 1), (-3, 3))
        self.assertAlmostEqual(up.xyz[2], 0.15 - LINEAR_SPEED_MPS * 0.5, places=9)
        nose_up = integrate(t, GamepadState(pitch_rate=1.0), 0.1, (-1, -1, -1), (1, 1, 1), (-3, 3))
        self.assertLess(nose_up.pitch, 0.0)                                             # stick forward = nose up = pitch decreases


class PairingTests(unittest.TestCase):
    def setUp(self):
        from gamepad_pairing import Pairing
        self.pad = FakePad()
        self.tmp = Path(tempfile.mkdtemp())
        self.log = RunLog(self.tmp, "test")
        self.p = Pairing(self.pad, self.tmp / "map.json", self.log, 0.0)
        self.now = 0.0

    def tearDown(self):
        self.log.close()

    def ticks(self, n):
        for _ in range(n):
            self.now += 0.02
            self.view = self.p.tick(self.now)

    def push_axis(self, idx, value, hold=8):
        self.pad.axes[idx] = value
        self.ticks(hold)
        self.pad.axes[idx] = -1.0 if idx in (4, 5) else 0.0
        self.ticks(2)

    def press(self, idx):
        self.pad.press(idx, True)
        self.ticks(2)
        self.pad.press(idx, False)
        self.ticks(2)

    def test_full_pairing_writes_map(self):
        p = self.p
        self.ticks(1)
        self.assertEqual(self.view.highlight, "left_stick")
        self.assertIn("Step 1 / 10", self.view.step_text)
        self.push_axis(1, -1.0)   # left stick forward: Linux reports up as negative
        self.assertEqual(p.mapping["axes"]["move_z"], {"axis": 1, "sign": -1, "kind": "stick", "rest": 0.0})
        self.assertEqual(p.index, 1)
        self.push_axis(0, 1.0)    # reach
        self.push_axis(3, -1.0)   # pitch
        self.push_axis(2, 1.0)    # roll
        self.push_axis(4, 1.0)    # LT
        self.assertEqual(p.mapping["axes"]["waist_left"]["kind"], "trigger")
        self.push_axis(5, 1.0)    # RT
        for idx in (1, 2, 7, 6):  # B X Start Back
            self.press(idx)
        self.assertTrue(p.done)
        self.assertTrue(self.view.done)
        written = json.loads((self.tmp / "map.json").read_text())["pads"]["fake"]
        self.assertEqual(written["buttons"], {"gripper_open": 1, "gripper_close": 2, "hold": 7, "stop": 6})
        self.assertEqual(set(written["axes"]), {"move_z", "reach", "pitch", "roll", "waist_left", "waist_right"})

    def test_small_or_brief_axis_motion_is_ignored(self):
        self.push_axis(1, -0.4)          # not far enough
        self.assertEqual(self.p.index, 0)
        self.pad.axes[1] = -1.0
        self.ticks(2)                    # not long enough
        self.pad.axes[1] = 0.0
        self.ticks(2)
        self.assertEqual(self.p.index, 0)

    def test_reused_control_is_refused(self):
        p = self.p
        self.push_axis(1, -1.0)          # move_z <- axis 1
        self.push_axis(1, 1.0)           # asked for reach, gave the same axis
        self.assertEqual(p.index, 1)
        self.assertIn("already", p.message)
        self.push_axis(0, 1.0)
        self.assertEqual(p.index, 2)
        p.index = 6
        p._begin_step(self.now)
        self.press(1)                    # gripper_open <- button 1
        self.press(1)                    # gripper_close: same button, refused
        self.assertEqual(p.index, 7)
        self.press(2)
        self.assertEqual(p.index, 8)

    def test_restart_clears_mapping(self):
        self.push_axis(1, -1.0)
        self.p.restart(self.now)
        self.assertEqual(self.p.index, 0)
        self.assertEqual(self.p.mapping["axes"], {})

    def test_demo_targets(self):
        from gamepad_pairing import demo_target, STEPS
        base = Target((0.3, 0.0, 0.15), 0.0, 0.0, 30.0)
        for step in STEPS:
            if step.demo is None:
                continue
            t = demo_target(base, step.demo, 1.0)
            self.assertNotEqual(t, base, step.key)
        turned = demo_target(base, ("waist", 1), 1.0)
        self.assertAlmostEqual(math.hypot(*turned.xyz[:2]), 0.3, places=9)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, required=True)
    args, rest = ap.parse_known_args()
    MODEL_DIR = args.model_dir
    unittest.main(argv=[sys.argv[0]] + rest, verbosity=2)
