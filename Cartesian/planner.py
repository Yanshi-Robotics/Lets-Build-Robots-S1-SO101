"""Stage 5 of the pipeline: walk the command toward the goal at a bounded speed.

`step` is a pure function so it can be tested without a robot. Two limits:

* speed: no joint may move more than `max_speed * dt` per tick, so a target that jumps
  across the workspace still produces a smooth motion;
* lead: on the real arm the command may not run ahead of the measured position by more
  than `max_lead`. The STS3215 turns position error into torque, so an unbounded lead is
  an unbounded push against whatever is blocking the arm.

Units follow the joint vector: q[0:5] radians, q[5] gripper percent.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Limits:
    max_speed: np.ndarray   # per joint, rad/s (gripper: %/s)
    max_lead: np.ndarray    # per joint, rad (gripper: %)


@dataclass(frozen=True)
class Step:
    q_next: np.ndarray
    speed_limited: bool     # at least one joint wanted to move faster than allowed
    lead_limited: bool      # at least one joint was held back by the lead clamp


def step(q_cmd: np.ndarray, q_goal: np.ndarray, q_meas: np.ndarray, dt: float, limits: Limits) -> Step:
    max_delta = limits.max_speed * dt
    wanted = q_goal - q_cmd
    delta = np.clip(wanted, -max_delta, max_delta)
    q_next = q_cmd + delta

    lead = q_next - q_meas
    clamped = np.clip(lead, -limits.max_lead, limits.max_lead)
    lead_limited = bool(np.any(np.abs(lead - clamped) > 1e-12))
    q_next = q_meas + clamped

    return Step(
        q_next=q_next,
        speed_limited=bool(np.any(np.abs(wanted - delta) > 1e-12)),
        lead_limited=lead_limited,
    )
