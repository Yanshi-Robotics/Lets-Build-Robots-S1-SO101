"""Stage 2, leader-arm flavour: a second SO-101 you move by hand gives joint goals directly.

There is no Cartesian target to solve here; the follower's joint goal is the leader's
joint reading, and the ball is put wherever that lands. Torque on the leader is off
(that is what makes it movable by hand). Read on the control thread only, like the
follower: it is a serial bus.
"""

from __future__ import annotations

from pathlib import Path


class Leader:
    def __init__(self, port: str, leader_id: str, calibration_dir: Path):
        from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig

        self.device = SO101Leader(SO101LeaderConfig(
            port=port, id=leader_id, calibration_dir=Path(calibration_dir), use_degrees=True))

    def connect(self) -> None:
        if not self.device.calibration:
            raise RuntimeError(
                f"no calibration for id {self.device.id} in {self.device.calibration_dir}; "
                "run Bringup/so101_calibrate.py --role leader first"
            )
        bus = self.device.bus
        bus.connect()
        if not bus.is_calibrated:
            # same check as the follower: the motors must carry the leader's own calibration
            bus.disconnect(disable_torque=False)
            raise RuntimeError(
                f"motors on {self.device.config.port} do not carry the leader calibration "
                f"{self.device.calibration_fpath}: wrong port (leader and follower swapped?) or re-run calibration"
            )
        self.device.configure()                              # torque off, position mode

    def read_deg(self) -> dict[str, float]:
        return self.device.bus.sync_read("Present_Position", num_retry=self.device.config.num_read_retries)

    def close(self) -> None:
        self.device.bus.disconnect(disable_torque=False)
