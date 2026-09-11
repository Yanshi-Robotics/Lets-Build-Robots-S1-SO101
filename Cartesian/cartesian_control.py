"""Cartesian control of the SO-101: drag a ball in the browser, the arm follows.

    # model only, no hardware
    python Cartesian/cartesian_control.py --model-dir models/so101 --model-only

    # real arm (torque stays off until you press "Hold and follow" in the page)
    python Cartesian/cartesian_control.py --model-dir models/so101 \\
        --port /dev/ttyACM0 --robot-id so101-follower --calibration-dir calibration/follower

Then open http://127.0.0.1:4602 .

This file is the control loop. Every tick it runs the six stages in order:
sense (so101_arm) -> target (target / viewer) -> compare (compare) -> solve (solver)
-> plan (planner) -> execute (so101_arm), and writes what each stage saw to the run log.
It is the only thread that talks to the robot model and to the serial bus.
"""

from __future__ import annotations

import argparse
import math
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import planner  # noqa: E402
from compare import REACH_TOLERANCE_M, error  # noqa: E402
from runlog import RunLog  # noqa: E402
from so101_arm import SERVO_P_COEFFICIENT, Arm, FakeArm, deg_from_q, q_from_deg  # noqa: E402
from so101_model import ARM_JOINTS, GRIPPER_INDEX, Model  # noqa: E402
from solver import Solver  # noqa: E402
from target import HOLD, RELEASE, STOP, CommandBox, TargetBox  # noqa: E402
from viewer import LOOPBACK, WEB_PORT, Viewer  # noqa: E402

HERE = Path(__file__).resolve().parent

CONTROL_HZ_LIVE = 30       # one sync_read + one sync_write take 6-10 ms on the 1 Mbaud bus
CONTROL_HZ_MODEL = 50      # no bus; just smooth animation
MAX_JOINT_SPEED_RAD_S = 2.0   # keeps up with a dragged ball without whipping; STS3215 no-load is ~6 rad/s
MAX_GRIPPER_SPEED_PCT_S = 150.0
MAX_LEAD_DEG = 15.0        # position error is torque on an STS3215: never push harder than this
MAX_GRIPPER_LEAD_PCT = 100.0  # the gripper is allowed to squeeze; its torque is capped in configure()
# Workspace the ball may be dragged in, metres in the base frame. Reach is ~0.40 m; nothing below the table.
DEFAULT_BOUNDS_MIN_M = (0.0, -0.25, 0.0)
DEFAULT_BOUNDS_MAX_M = (0.40, 0.25, 0.45)
# In live mode with torque off, the ball mirrors the real arm; only re-sync when it moved this much.
ARM_MOVED_RAD = math.radians(0.5)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--model-dir", type=Path, required=True, help="directory holding so101_new_calib.urdf and assets/")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--model-only", action="store_true", help="no hardware; a simulated arm follows the ball")
    mode.add_argument("--port", help="serial port of the follower arm, e.g. /dev/ttyACM0")
    p.add_argument("--robot-id", default="so101-follower", help="calibration file name without .json")
    p.add_argument("--calibration-dir", type=Path, default=Path("calibration/follower"))
    p.add_argument("--p-coefficient", type=int, default=SERVO_P_COEFFICIENT, help="STS3215 position P gain")
    p.add_argument("--hz", type=float, default=None, help=f"control rate; default {CONTROL_HZ_LIVE} live, {CONTROL_HZ_MODEL} model")
    p.add_argument("--max-joint-speed", type=float, default=MAX_JOINT_SPEED_RAD_S, help="rad/s per arm joint")
    p.add_argument("--max-lead-deg", type=float, default=MAX_LEAD_DEG, help="how far the command may run ahead of the measured joint")
    p.add_argument("--bounds-min-m", type=float, nargs=3, default=DEFAULT_BOUNDS_MIN_M, metavar=("X", "Y", "Z"))
    p.add_argument("--bounds-max-m", type=float, nargs=3, default=DEFAULT_BOUNDS_MAX_M, metavar=("X", "Y", "Z"))
    p.add_argument("--web-port", type=int, default=WEB_PORT)
    p.add_argument("--release-torque", action="store_true", help="turn torque off on exit (default: keep holding)")
    p.add_argument("--logs-dir", type=Path, default=HERE / "logs")
    return p.parse_args(argv)


class ControlLoop:
    def __init__(self, model: Model, solver: Solver, arm, limits: planner.Limits, hz: float,
                 targets: TargetBox, commands: CommandBox, viewer: Viewer, log: RunLog):
        self.model, self.solver, self.arm, self.limits = model, solver, arm, limits
        self.dt = 1.0 / hz
        self.targets, self.commands, self.viewer, self.log = targets, commands, viewer, log
        self.live = arm.mode == "live"
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self._run_guarded, name="control", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self._stop.set()
        self.thread.join(timeout=2.0)

    def _run_guarded(self) -> None:
        try:
            self._run()
        except Exception:
            self.log.exception("control loop died")
            self._stop.set()

    def _sync_from(self, q: np.ndarray, source: str) -> tuple[int, object]:
        target = self.model.target_from_q(q)
        version = self.targets.set(target, source)
        self.viewer.sync_target(target)
        return version, target

    def _run(self) -> None:
        model, solver, arm, log = self.model, self.solver, self.arm, self.log
        q_meas = q_from_deg(arm.read_deg())
        q_cmd = q_meas.copy()
        q_goal = q_meas.copy()
        solution = None
        following = not self.live       # the simulated arm follows from the start
        last_version, _ = self._sync_from(q_meas, "arm")
        last_synced_q = q_meas.copy()
        log.event("loop start", live=self.live, hz=1.0 / self.dt, q_deg=deg_from_q(q_meas))

        tick = 0
        t_prev = time.monotonic()
        while not self._stop.is_set():
            t0 = time.monotonic()
            record: dict = {"tick": tick, "period_ms": (t0 - t_prev) * 1000.0}
            t_prev = t0

            # 1. sense
            q_meas = q_from_deg(arm.read_deg())
            t1 = time.monotonic()

            # 2. target: buttons, a dragged ring, or the box
            joint_cmd, buttons = self.commands.drain()
            for button in buttons:
                log.event("button", name=button)
                if button == HOLD:
                    arm.hold()
                    q_cmd, q_goal = q_meas.copy(), q_meas.copy()
                    last_version, _ = self._sync_from(q_meas, "arm")
                    following = True
                elif button == STOP:
                    following = False
                elif button == RELEASE:
                    arm.release()
                    following = False
                record["button"] = button

            if joint_cmd is not None and following:
                q_goal = q_goal.copy()
                low, high = model.limits[joint_cmd.joint]
                q_goal[ARM_JOINTS.index(joint_cmd.joint)] = min(max(joint_cmd.angle, low), high)
                q_goal = model.clamp(q_goal)
                last_version, target = self._sync_from(q_goal, "ring")   # ring moves the goal; the ball follows
                solution = None
                record["ring"] = {"joint": joint_cmd.joint, "angle_deg": math.degrees(joint_cmd.angle)}

            target, version, source = self.targets.snapshot()

            # 3+4. compare and solve, only when the target changed
            t2 = time.monotonic()
            if version != last_version:
                solution = solver.solve(q_cmd, target)
                q_goal = solution.q_goal
                last_version = version
                record["solve"] = {"iterations": solution.iterations, "converged": solution.converged,
                                   "error_mm": solution.error.position_m * 1000.0,
                                   "pitch_err_deg": math.degrees(solution.error.pitch_rad),
                                   "roll_err_deg": math.degrees(solution.error.roll_rad)}
                if not solution.converged:
                    log.warning("target not reached by solver", source=source, target=target.__dict__, **record["solve"])
            t3 = time.monotonic()

            # 5. plan
            if following:
                step = planner.step(q_cmd, q_goal, q_meas if self.live else q_cmd, self.dt, self.limits)
                q_cmd = step.q_next
                record["speed_limited"] = step.speed_limited
                record["lead_limited"] = step.lead_limited
            else:
                # torque off or stopped: the command is wherever the arm is, and the ball mirrors the arm
                q_cmd, q_goal = q_meas.copy(), q_meas.copy()
                if np.any(np.abs(q_meas[:5] - last_synced_q[:5]) > ARM_MOVED_RAD) or tick == 0:
                    last_version, target = self._sync_from(q_meas, "arm")
                    last_synced_q = q_meas.copy()
            t4 = time.monotonic()

            # 6. execute
            if following:
                arm.send_deg(deg_from_q(q_cmd))
            t5 = time.monotonic()

            # what the eyes and the log get
            goal_err = solution.error if solution is not None else error(model, q_goal, target)
            cmd_err = error(model, q_cmd, target)
            meas_err = error(model, q_meas, target) if self.live else cmd_err
            reachable = goal_err.reachable(REACH_TOLERANCE_M)
            frames = model.joint_frames(q_goal)
            tool_R = model.compose(model.tool_yaw(q_goal), target.pitch, target.roll)
            self.viewer.push(q_meas, q_cmd, q_goal, frames, tool_R, reachable,
                             self._status(following, target, goal_err, cmd_err, meas_err, q_meas, record))

            record.update({
                "ms": {"read": (t1 - t0) * 1e3, "solve": (t3 - t2) * 1e3, "plan": (t4 - t3) * 1e3, "write": (t5 - t4) * 1e3},
                "following": following, "torque": arm.torque_on,
                "target": {"xyz": target.xyz, "pitch_deg": math.degrees(target.pitch),
                           "roll_deg": math.degrees(target.roll), "gripper_pct": target.gripper_pct,
                           "version": version, "source": source},
                "q_meas_deg": deg_from_q(q_meas), "q_goal_deg": deg_from_q(q_goal), "q_cmd_deg": deg_from_q(q_cmd),
                "err_mm": {"goal": goal_err.position_m * 1e3, "cmd": cmd_err.position_m * 1e3, "meas": meas_err.position_m * 1e3},
                "reachable": reachable,
            })
            log.tick(record)
            tick += 1

            remaining = self.dt - (time.monotonic() - t0)
            if remaining > 0:
                time.sleep(remaining)

    def _status(self, following, target, goal_err, cmd_err, meas_err, q_meas, record) -> str:
        mode = "LIVE" if self.live else "MODEL"
        if self.live:
            state = "following the ball" if following else ("torque on, stopped" if self.arm.torque_on else "torque off, ball mirrors the arm")
        else:
            state = "simulated arm follows the ball"
        lines = [
            f"**{mode}** · {state}",
            f"target xyz = ({target.xyz[0]:.3f}, {target.xyz[1]:.3f}, {target.xyz[2]:.3f}) m · "
            f"pitch {math.degrees(target.pitch):.0f}° · roll {math.degrees(target.roll):.0f}° · gripper {target.gripper_pct:.0f}%",
            f"error: solver {goal_err.position_m * 1e3:.1f} mm · command {cmd_err.position_m * 1e3:.1f} mm · "
            f"arm {meas_err.position_m * 1e3:.1f} mm" + ("" if goal_err.reachable() else " · **out of reach**"),
            "measured deg: " + ", ".join(f"{math.degrees(q_meas[i]):.1f}" for i in range(5)) + f" · gripper {q_meas[GRIPPER_INDEX]:.0f}%",
            f"loop {1000.0 / max(record['period_ms'], 1e-3):.0f} Hz",
        ]
        return "  \n".join(lines)


def main(argv=None) -> int:
    args = parse_args(argv)
    live = args.port is not None
    hz = args.hz or (CONTROL_HZ_LIVE if live else CONTROL_HZ_MODEL)
    log = RunLog(args.logs_dir, "live" if live else "model")
    log.event("arguments", **{k: str(v) for k, v in vars(args).items()})

    if live:
        arm = Arm(args.port, args.robot_id, args.calibration_dir, args.p_coefficient)
        calibration_path = arm.calibration_path
    else:
        arm = FakeArm()
        calibration_path = None
    model = Model(args.model_dir, calibration_path)
    log.event("model", urdf=str(model.urdf_path), calibration=str(calibration_path),
              limits_deg={k: [math.degrees(a), math.degrees(b)] for k, (a, b) in model.limits.items()},
              pan_to_yaw=model.pan_to_yaw)
    solver = Solver(model)
    limits = planner.Limits(
        max_speed=np.array([args.max_joint_speed] * 5 + [MAX_GRIPPER_SPEED_PCT_S]),
        max_lead=np.array([math.radians(args.max_lead_deg)] * 5 + [MAX_GRIPPER_LEAD_PCT]),
    )

    arm.connect()
    q0 = q_from_deg(arm.read_deg())
    log.event("arm connected", mode=arm.mode, q_deg=deg_from_q(q0), torque=arm.torque_on)
    initial = model.target_from_q(q0)
    targets, commands = TargetBox(initial), CommandBox()
    viewer = Viewer(model, args.bounds_min_m, args.bounds_max_m, LOOPBACK, args.web_port, live,
                    targets, commands, initial)
    log.event("page up", url=f"http://{LOOPBACK}:{args.web_port}")
    print(f"open http://{LOOPBACK}:{args.web_port}  (logs: {log.dir})", flush=True)

    loop = ControlLoop(model, solver, arm, limits, hz, targets, commands, viewer, log)
    loop.start()
    try:
        while loop.thread.is_alive():
            time.sleep(0.2)
        return 1  # the loop only ends on its own if it crashed (see run.log)
    except KeyboardInterrupt:
        log.event("Ctrl-C")
        return 0
    finally:
        loop.stop()
        arm.close(release_torque=args.release_torque)
        log.event("closed", torque_released=args.release_torque and live)
        log.close()


if __name__ == "__main__":
    sys.exit(main())
