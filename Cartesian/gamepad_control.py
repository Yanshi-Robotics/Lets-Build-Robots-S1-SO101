"""Stage 2, gamepad flavour: sticks and triggers become a moving `Target`.

Nothing here knows the pad's axis numbers; those come from `gamepad_map.json` written by
gamepad_setup.py. Each tick the control loop asks `GamepadSource.read()` for velocities
(-1..1 per direction) and button edges, then `integrate()` moves the current target by
`speed x dt`. Sticks at rest change nothing, so the target only gets a new version when
the user is actually pushing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from gamepad import Gamepad
from target import Target

DEADZONE = 0.12               # sticks never rest at exactly zero
LINEAR_SPEED_MPS = 0.08       # tool speed at full stick
ANGULAR_SPEED_RAD_S = 1.2     # pitch / roll rate at full trigger / stick
PITCH_LIMIT_RAD = np.radians(120)   # keep the sliders' range; the arm cannot do more anyway


def read_maps(path: Path) -> dict[str, dict]:
    """All pairings on file, keyed by the pad's device name. Missing file: none."""
    path = Path(path)
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    if "pads" in data:
        return data["pads"]
    if "device" in data:            # first version of the file: one pad, unnamed section
        return {data["device"]: {"axes": data["axes"], "buttons": data["buttons"]}}
    return {}


def write_map(path: Path, device: str, mapping: dict) -> None:
    """Add or replace one pad's pairing, keeping the others."""
    pads = read_maps(path)
    pads[device] = {"axes": mapping["axes"], "buttons": mapping["buttons"]}
    Path(path).write_text(json.dumps({"pads": pads}, indent=2, ensure_ascii=False) + "\n")


def load_map(path: Path, device: str) -> dict | None:
    """The pairing for this pad, or None if it has never been paired."""
    return read_maps(path).get(device)


@dataclass(frozen=True)
class GamepadState:
    vx: float = 0.0            # forward (+x), -1..1
    vy: float = 0.0            # right (-y), -1..1
    vz: float = 0.0            # up, -1..1
    roll_rate: float = 0.0     # -1..1
    pitch_rate: float = 0.0    # +1 = pitching down at full trigger
    pressed: tuple[str, ...] = field(default_factory=tuple)   # button keys that went down this tick

    @property
    def moving(self) -> bool:
        return any(abs(v) > 0 for v in (self.vx, self.vy, self.vz, self.roll_rate, self.pitch_rate))


class GamepadSource:
    def __init__(self, pad: Gamepad, mapping: dict):
        self.pad = pad
        self.axes = mapping["axes"]
        self.buttons = mapping["buttons"]
        self._was_down: dict[str, bool] = {k: False for k in self.buttons}

    def _axis(self, axes: list[float], key: str) -> float:
        e = self.axes.get(key)
        if e is None or e["axis"] >= len(axes):
            return 0.0
        v = axes[e["axis"]]
        if e["kind"] == "trigger":
            v = (v - e["rest"]) / 2.0            # -1..1 -> 0..1
        else:
            v = (v - e["rest"]) * e["sign"]
        if abs(v) < DEADZONE:
            return 0.0
        # rescale so the output starts from 0 right outside the deadzone
        return float(np.sign(v) * (abs(v) - DEADZONE) / (1.0 - DEADZONE))

    def read(self) -> GamepadState:
        axes, buttons = self.pad.snapshot()
        pressed = []
        for key, idx in self.buttons.items():
            down = idx < len(buttons) and buttons[idx]
            if down and not self._was_down[key]:
                pressed.append(key)
            self._was_down[key] = down
        return GamepadState(
            vx=self._axis(axes, "move_x"),
            vy=self._axis(axes, "move_y"),
            vz=self._axis(axes, "move_z"),
            roll_rate=self._axis(axes, "roll"),
            pitch_rate=self._axis(axes, "pitch_down") - self._axis(axes, "pitch_up"),
            pressed=tuple(pressed),
        )


def integrate(target: Target, state: GamepadState, dt: float, bounds_min, bounds_max,
              roll_limits: tuple[float, float]) -> Target:
    xyz = np.array(target.xyz) + LINEAR_SPEED_MPS * dt * np.array([state.vx, -state.vy, state.vz])
    xyz = np.clip(xyz, bounds_min, bounds_max)
    pitch = float(np.clip(target.pitch + ANGULAR_SPEED_RAD_S * dt * state.pitch_rate, -PITCH_LIMIT_RAD, PITCH_LIMIT_RAD))
    roll = float(np.clip(target.roll + ANGULAR_SPEED_RAD_S * dt * state.roll_rate, roll_limits[0], roll_limits[1]))
    gripper = target.gripper_pct
    if "gripper_open" in state.pressed:
        gripper = 100.0
    if "gripper_close" in state.pressed:
        gripper = 0.0
    return Target(xyz=tuple(float(v) for v in xyz), pitch=pitch, roll=roll, gripper_pct=gripper)
