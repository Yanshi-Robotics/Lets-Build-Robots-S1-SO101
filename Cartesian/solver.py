"""Stage 4 of the pipeline: a target pose in, a joint goal out.

The solver is placo's `KinematicsSolver` (a QP over joint displacements with joint limits
as constraints). One frame task on the tool frame, position weighted above orientation so
that an unreachable target still ends up as close as possible in position. The gripper
is masked out; it never takes part in reaching a pose.

`solve` iterates the QP from a seed (the current command) until the tool is within
tolerance of the target or the iteration budget is spent. Every iteration re-reads the
arm's current yaw and builds the target orientation from it: yaw is not something the
SO-101 can choose independently of position (see so101_model.py), so it is filled in
from the arm rather than asked of the user.

Seeding from the current command keeps the answer on the same branch of the arm
(elbow up stays elbow up) instead of jumping to another valid but far-away solution.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from compare import Error, error
from so101_model import GRIPPER, GRIPPER_INDEX, TOOL_FRAME, Model
from target import Target

POSITION_WEIGHT = 1.0
ORIENTATION_WEIGHT = 0.3     # below position: when both cannot be met, keep the point
REGULARIZATION = 1e-3        # tiny pull toward "do not move", keeps the QP well posed near singularities
SOLVE_TOLERANCE_M = 0.0005   # stop iterating when the tool point is within half a millimetre
SOLVE_TOLERANCE_RAD = math.radians(0.1)
SOLVE_MAX_ITERATIONS = 200   # only unreachable targets ever get near this


@dataclass(frozen=True)
class Solution:
    q_goal: np.ndarray
    error: Error          # of q_goal against the target
    iterations: int
    converged: bool


class Solver:
    def __init__(self, model: Model, position_weight: float = POSITION_WEIGHT,
                 orientation_weight: float = ORIENTATION_WEIGHT, regularization: float = REGULARIZATION,
                 tolerance_m: float = SOLVE_TOLERANCE_M, tolerance_rad: float = SOLVE_TOLERANCE_RAD,
                 max_iterations: int = SOLVE_MAX_ITERATIONS):
        self.model = model
        self.tolerance_m = tolerance_m
        self.tolerance_rad = tolerance_rad
        self.max_iterations = max_iterations

        self.solver = model.robot.make_solver()
        self.solver.mask_fbase(True)              # the base is bolted to the table
        self.task = self.solver.add_frame_task(TOOL_FRAME, np.eye(4))
        self.task.configure("tool", "soft", position_weight, orientation_weight)
        self.solver.add_regularization_task(regularization)
        self.solver.mask_dof(GRIPPER)
        self.solver.enable_joint_limits(True)
        self.solver.enable_velocity_limits(False)  # speed is the planner's job, not the solver's
        self.solver.dt = 1.0

    def solve(self, q_seed: np.ndarray, target: Target) -> Solution:
        model = self.model
        model.fk(q_seed)  # loads the seed into placo's state
        iterations = 0
        converged = False
        while iterations < self.max_iterations:
            self.task.T_world_frame = model.pose_from_target(target, model.current_yaw())
            self.solver.solve(True)
            model.robot.update_kinematics()
            iterations += 1

            q_now = model.state_q()
            err = error(model, q_now, target)  # note: error() runs FK, which re-loads q_now into placo
            if err.position_m <= self.tolerance_m and max(abs(err.pitch_rad), abs(err.roll_rad)) <= self.tolerance_rad:
                converged = True
                break

        q_goal = model.state_q()
        q_goal[GRIPPER_INDEX] = target.gripper_pct
        q_goal = model.clamp(q_goal)
        return Solution(q_goal=q_goal, error=error(model, q_goal, target), iterations=iterations, converged=converged)
