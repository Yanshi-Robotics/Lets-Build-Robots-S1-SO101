"""Stage 3 of the pipeline: how far is the tool from the target?

Pure functions over a `Model`, a joint vector and a `Target`. The control loop uses the
result for three things: deciding whether the target is reachable (ball colour), showing
the numbers in the status panel, and writing them to the tick log so that "the arm is not
where I asked" can be split into "the solver did not get there" (error of q_goal) versus
"the arm did not follow the command" (error of q_meas).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from so101_model import Model, wrap_angle
from target import Target

# The ball turns red when the solved pose is farther than this from the target.
REACH_TOLERANCE_M = 0.005


@dataclass(frozen=True)
class Error:
    position_m: float       # |target xyz - tool xyz|
    pitch_rad: float        # signed, target - tool
    roll_rad: float         # signed, target - tool
    vector_m: tuple[float, float, float]  # target xyz - tool xyz, per axis

    def reachable(self, tolerance_m: float = REACH_TOLERANCE_M) -> bool:
        return self.position_m <= tolerance_m


def error(model: Model, q: np.ndarray, target: Target) -> Error:
    T = model.fk(q)
    diff = np.asarray(target.xyz, dtype=float) - T[:3, 3]
    _, pitch, roll = model.decompose(T[:3, :3], yaw=model.tool_yaw(q))
    return Error(
        position_m=float(np.linalg.norm(diff)),
        pitch_rad=wrap_angle(target.pitch - pitch),
        roll_rad=wrap_angle(target.roll - roll),
        vector_m=tuple(float(v) for v in diff),
    )
