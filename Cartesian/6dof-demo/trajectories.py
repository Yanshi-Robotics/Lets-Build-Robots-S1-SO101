"""Joint-angle trajectories for the six-degree-of-freedom animations in Lesson 8.

    python Cartesian/6dof-demo/trajectories.py --model-dir models/so101 --output Cartesian/6dof-demo/trajectories.json

Five clips on the SO-101 (x, y, z, pitch, roll) come from the course's own solver: the
tool target moves back and forth along one degree of freedom and `Solver.solve` gives the
joint angles for every frame. The sixth, yaw, is the one the SO-101 cannot do without
moving the tool, so it is shown on a Franka Panda (seven joints): placo holds the hand's
position fixed and turns its orientation about the vertical axis. That clip is only
produced when `panda/panda.urdf` has been fetched (`panda/fetch_panda.py`).

The JSON holds, per clip, the joint names, one row of angles per frame (radians; the
SO-101 gripper in percent), and the tool position / rotation per frame so the renderer
can place its arrows. Every clip is a ping-pong: it ends where it starts.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from so101_arm import MODEL_START_DEG, q_from_deg  # noqa: E402
from so101_model import ALL_JOINTS, Model  # noqa: E402
from solver import Solver  # noqa: E402
from target import Target  # noqa: E402

FPS = 15
CLIP_SECONDS = 3.0            # one full back-and-forth
TRAVEL_M = 0.04               # ± for x, y, z
PITCH_RAD = math.radians(25)  # ±
ROLL_RAD = math.radians(40)   # ±
YAW_RAD = math.radians(60)    # ± on the Panda: wide, so the elbow visibly swings while the hand stays
PANDA_HOME_DEG = (0, -20, 0, -120, 0, 180, 45)   # elbow bent, hand level and pointing forward, wrist joints well inside their limits
PANDA_TOOL = "panda_hand"
PANDA_POSTURE_WEIGHT = 5e-3   # far below the hand task: the base follows the yaw only as far as the hand allows
PANDA_BASE_FOLLOW = 0.9       # how much of the hand's yaw the base is asked to take on
PANDA_ITERATIONS = 200        # QP steps per frame; the posture task makes convergence slower
PANDA_POSITION_TOL_M = 0.002  # the hand may not drift more than this while yawing

SO101_CLIPS = {
    "x": ("xyz", (1, 0, 0)), "y": ("xyz", (0, 1, 0)), "z": ("xyz", (0, 0, 1)),
    "pitch": ("pitch", None), "roll": ("roll", None),
}


def ping_pong(n: int) -> list[float]:
    """n values of a smooth -1 .. +1 .. -1 wave, starting and ending at 0: sin over one period."""
    return [math.sin(2 * math.pi * i / n) for i in range(n)]


def so101_clip(model: Model, solver: Solver, kind: str, arg, base: Target, q0: np.ndarray) -> dict:
    n = round(FPS * CLIP_SECONDS)
    q = q0.copy()
    rows, tool_p, tool_R = [], [], []
    for w in ping_pong(n):
        if kind == "xyz":
            target = base.with_xyz(np.array(base.xyz) + TRAVEL_M * w * np.array(arg, dtype=float))
        elif kind == "pitch":
            target = Target(base.xyz, base.pitch + PITCH_RAD * w, base.roll, base.gripper_pct)
        else:
            target = Target(base.xyz, base.pitch, base.roll + ROLL_RAD * w, base.gripper_pct)
        sol = solver.solve(q, target)
        if not sol.converged:
            raise SystemExit(f"STOP: {kind} clip target not reached at w={w:.2f}: {sol.error}")
        q = sol.q_goal
        T = model.fk(q)
        rows.append([float(v) for v in q])
        tool_p.append([float(v) for v in T[:3, 3]])
        tool_R.append([[float(v) for v in row] for row in T[:3, :3]])
    return {"robot": "so101", "joints": list(ALL_JOINTS), "fps": FPS, "frames": rows,
            "tool_position": tool_p, "tool_rotation": tool_R, "tool_frame": "gripper_frame_link"}


def panda_yaw_clip(urdf: Path) -> dict:
    import placo

    robot = placo.RobotWrapper(str(urdf))
    joints = [f"panda_joint{i}" for i in range(1, 8)]
    for name, deg in zip(joints, PANDA_HOME_DEG):
        robot.set_joint(name, math.radians(deg))
    robot.update_kinematics()
    T0 = np.array(robot.get_T_world_frame(PANDA_TOOL))

    solver = robot.make_solver()
    solver.mask_fbase(True)
    task = solver.add_frame_task(PANDA_TOOL, T0)
    task.configure("hand", "soft", 1.0, 1.0)
    # The Panda has one joint more than a pose needs, so the elbow can swing without the hand
    # moving. A weak posture task asks the base to follow the yaw: the whole arm visibly turns
    # while the hand pose task (much heavier) keeps the hand's position exactly where it is.
    posture = solver.add_joints_task()
    posture.configure("posture", "soft", PANDA_POSTURE_WEIGHT)
    solver.add_regularization_task(1e-4)
    solver.mask_dof("panda_finger_joint1")
    solver.mask_dof("panda_finger_joint2")
    solver.enable_joint_limits(True)
    solver.dt = 1.0

    def rot_z(a):
        c, s = math.cos(a), math.sin(a)
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    n = round(FPS * CLIP_SECONDS)
    rows, tool_p, tool_R = [], [], []
    for w in ping_pong(n):
        T = T0.copy()
        T[:3, :3] = rot_z(YAW_RAD * w) @ T0[:3, :3]      # same position, turned about the vertical axis
        task.T_world_frame = T
        posture.set_joint("panda_joint1", YAW_RAD * w * PANDA_BASE_FOLLOW)
        for _ in range(PANDA_ITERATIONS):
            solver.solve(True)
            robot.update_kinematics()
        Tn = np.array(robot.get_T_world_frame(PANDA_TOOL))
        if np.linalg.norm(Tn[:3, 3] - T0[:3, 3]) > PANDA_POSITION_TOL_M:
            raise SystemExit("STOP: Panda hand position drifted during the yaw clip")
        rows.append([float(robot.get_joint(j)) for j in joints])
        tool_p.append([float(v) for v in Tn[:3, 3]])
        tool_R.append([[float(v) for v in row] for row in Tn[:3, :3]])
    return {"robot": "panda", "joints": joints, "fps": FPS, "frames": rows,
            "tool_position": tool_p, "tool_rotation": tool_R, "tool_frame": PANDA_TOOL}


def build(model_dir: Path, panda_urdf: Path | None) -> dict:
    model = Model(model_dir)
    solver = Solver(model)
    q0 = q_from_deg(MODEL_START_DEG)
    base = model.target_from_q(q0)
    clips = {name: so101_clip(model, solver, kind, arg, base, q0) for name, (kind, arg) in SO101_CLIPS.items()}
    if panda_urdf is not None and panda_urdf.exists():
        clips["yaw"] = panda_yaw_clip(panda_urdf)
    return {"fps": FPS, "seconds": CLIP_SECONDS, "clips": clips}


def main(argv=None) -> int:
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--panda-urdf", type=Path, default=here / "panda" / "panda.urdf")
    p.add_argument("--output", type=Path, default=here / "trajectories.json")
    args = p.parse_args(argv)
    data = build(args.model_dir, args.panda_urdf)
    args.output.write_text(json.dumps(data), encoding="utf-8")
    print(f"{args.output}: clips {sorted(data['clips'])}, {len(next(iter(data['clips'].values()))['frames'])} frames each")
    return 0


if __name__ == "__main__":
    sys.exit(main())
