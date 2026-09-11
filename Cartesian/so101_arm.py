"""Stages 1 and 6 of the pipeline: read the servos, write the servos.

`Arm` wraps LeRobot 0.6.1's `SO101Follower`. It is the only code that touches the serial
bus, and it must only ever be called from the control thread: two threads sharing one
Feetech bus corrupt each other's packets.

`FakeArm` has the same interface and no hardware behind it; `read_deg` returns whatever
was last sent. The control loop does not know which one it holds.

Units at this boundary are LeRobot's: degrees for the five arm joints (zero = middle of
the calibrated range, which is also the URDF's zero), 0..100 percent for the gripper.
`q_from_deg` / `deg_from_q` convert to the radian joint vector used everywhere else.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from so101_model import ALL_JOINTS, ARM_JOINTS, GRIPPER, GRIPPER_INDEX

# STS3215 factory position P gain. LeRobot writes 16, which makes the arm so soft that a
# command leading the measured position by less than about 2.3 degrees does not move it.
SERVO_P_COEFFICIENT = 32
TORQUE_WRITE_RETRIES = 3   # an overloaded servo may answer a Torque_Enable write with an error bit once

# Where the simulated arm starts: elbow bent, tool level, gripper a third open.
MODEL_START_DEG = {"shoulder_pan": 0.0, "shoulder_lift": -30.0, "elbow_flex": 60.0,
                   "wrist_flex": -30.0, "wrist_roll": 0.0, GRIPPER: 30.0}


def q_from_deg(angles: dict[str, float]) -> np.ndarray:
    q = np.zeros(6)
    for i, name in enumerate(ARM_JOINTS):
        q[i] = math.radians(angles[name])
    q[GRIPPER_INDEX] = float(angles[GRIPPER])
    return q


def deg_from_q(q: np.ndarray) -> dict[str, float]:
    out = {name: math.degrees(float(q[i])) for i, name in enumerate(ARM_JOINTS)}
    out[GRIPPER] = float(q[GRIPPER_INDEX])
    return out


class FakeArm:
    mode = "model"

    def __init__(self, start_deg: dict[str, float] = MODEL_START_DEG):
        self._deg = dict(start_deg)
        self.torque_on = False

    def connect(self) -> None:
        pass

    def read_deg(self) -> dict[str, float]:
        return dict(self._deg)

    def hold(self) -> None:
        self.torque_on = True

    def send_deg(self, goal: dict[str, float]) -> None:
        self._deg = {name: float(goal[name]) for name in ALL_JOINTS}

    def release(self) -> None:
        self.torque_on = False

    def close(self, release_torque: bool) -> None:
        pass


class Arm:
    mode = "live"

    def __init__(self, port: str, robot_id: str, calibration_dir: Path, p_coefficient: int = SERVO_P_COEFFICIENT):
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

        self.robot = SO101Follower(SO101FollowerConfig(
            port=port, id=robot_id, calibration_dir=Path(calibration_dir),
            use_degrees=True, cameras={},
            disable_torque_on_disconnect=False,   # `close` decides, see below
            position_p_coefficient=p_coefficient,
        ))
        self.torque_on = False

    @property
    def calibration_path(self) -> Path:
        return self.robot.calibration_fpath

    def connect(self) -> None:
        """Open the bus and make sure the motors carry this arm's calibration. No torque."""
        if not self.robot.calibration:
            raise RuntimeError(
                f"no calibration for id {self.robot.id} in {self.robot.calibration_dir}; "
                "run Bringup/so101_calibrate.py first"
            )
        bus = self.robot.bus
        bus.connect()
        if not bus.is_calibrated:
            # The motors' stored homing offsets and limits do not match the follower's
            # calibration file: either this port is the leader arm, or the arm was
            # recalibrated. Refuse rather than write the follower's numbers into it.
            bus.disconnect(disable_torque=False)
            raise RuntimeError(
                f"motors on {self.robot.config.port} do not carry the follower calibration "
                f"{self.calibration_path}: wrong port (leader and follower swapped?) or re-run calibration"
            )

    def read_deg(self) -> dict[str, float]:
        return self.robot.bus.sync_read("Present_Position", num_retry=self.robot.config.num_read_retries)

    def hold(self) -> None:
        """Turn torque on without moving: park the goal at the present position first.

        On power-up the Goal_Position register may hold anything; enabling torque with
        a stale goal makes the arm jump there. `configure()` then writes P/I/D and the
        gripper's current limits, and leaves torque enabled (that is how LeRobot wrote it).
        """
        present = self.read_deg()
        self.robot.bus.sync_write("Goal_Position", present)
        self.robot.configure()
        self.torque_on = True

    def send_deg(self, goal: dict[str, float]) -> None:
        self.robot.bus.sync_write("Goal_Position", {name: float(goal[name]) for name in ALL_JOINTS})

    def release(self) -> None:
        """Torque off on every motor, one by one: a servo in overload answers the write with
        an error bit and LeRobot raises on it, which must not stop the other five."""
        failed = {}
        for name in ALL_JOINTS:
            try:
                self.robot.bus.disable_torque(name, num_retry=TORQUE_WRITE_RETRIES)
            except RuntimeError as e:
                failed[name] = str(e)
        self.torque_on = bool(failed)
        if failed:
            raise RuntimeError(f"torque still on: {failed}")

    def close(self, release_torque: bool) -> None:
        """Default keeps torque on so the arm does not fall when the program exits."""
        if release_torque:
            self.release()
        self.robot.bus.disconnect(disable_torque=False)
