"""Stage 2, gamepad flavour: sticks and triggers become a moving `Target`.

The layout follows Interbotix's X-Series arm joystick control (the same class of
five-degree-of-freedom arm as the SO-101), with Xbox names for the PS buttons:

    LT / RT               turn the whole arm about its base (waist), left / right
    left stick  up/down   tool up / down (z)
    left stick  left/right tool out / in along the arm (reach)
    right stick up/down   pitch up / down
    right stick left/right roll
    B / X                 gripper open / close, only while held
    Start / Back          Hold and follow / Stop  (Interbotix: home / sleep poses)

The tool target is kept in base-frame x, y, z, but the sticks move it in cylindrical
terms: the waist angle, the reach (distance from the base axis) and the height. That
is what a five-joint arm can actually do; pushing "left" in base-frame y would only
make the solver turn the base anyway, so the pad turns it directly.

Nothing here knows the pad's axis numbers; those come from `gamepad_map.json` written
by the pairing on the Gamepad page (gamepad_pairing.py). Each tick the control loop asks `GamepadSource.read()` for rates
(-1..1) and button edges, then `integrate()` moves the current target by `speed x dt`.
Sticks at rest change nothing, so the target only gets a new version when the user is
actually pushing.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from gamepad import Gamepad
from target import Target

DEADZONE = 0.12               # sticks never rest at exactly zero
LINEAR_SPEED_MPS = 0.08       # reach / height speed at full stick
WAIST_SPEED_RAD_S = 0.8       # base turn at full trigger
ANGULAR_SPEED_RAD_S = 1.2     # pitch / roll rate at full stick
GRIPPER_SPEED_PCT_S = 60.0    # B / X move the gripper only while held: full travel in under two seconds
PITCH_LIMIT_RAD = np.radians(120)   # keep the sliders' range; the arm cannot do more anyway
MIN_REACH_M = 0.03            # the tool cannot be pulled onto the base axis; the solver would have no yaw to hold

# What a pairing must contain to be usable (the steps in gamepad_pairing.py write exactly these).
AXIS_KEYS = ("move_z", "reach", "pitch", "roll", "waist_left", "waist_right")
BUTTON_KEYS = ("gripper_open", "gripper_close", "hold", "stop")


def read_maps(path: Path) -> dict[str, dict]:
    """All pairings on file, keyed by the pad's device name. Missing file: none."""
    path = Path(path)
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    return data.get("pads", {})


def write_map(path: Path, device: str, mapping: dict) -> None:
    """Add or replace one pad's pairing, keeping the others."""
    pads = read_maps(path)
    pads[device] = {"axes": mapping["axes"], "buttons": mapping["buttons"]}
    Path(path).write_text(json.dumps({"pads": pads}, indent=2, ensure_ascii=False) + "\n")


def load_map(path: Path, device: str) -> dict | None:
    """The pairing for this pad, or None if it has never been paired with the current layout."""
    mapping = read_maps(path).get(device)
    if mapping is None:
        return None
    if not all(k in mapping.get("axes", {}) for k in AXIS_KEYS) or not all(k in mapping.get("buttons", {}) for k in BUTTON_KEYS):
        return None   # paired under an older layout: pair again
    return mapping


@dataclass(frozen=True)
class GamepadState:
    waist_rate: float = 0.0    # +1 = turning left (counter-clockwise seen from above) at full trigger
    reach_rate: float = 0.0    # +1 = tool moving out, away from the base axis
    vz: float = 0.0            # +1 = up
    pitch_rate: float = 0.0    # +1 = nose up (stick forward), as Interbotix defines "increase pitch"
    roll_rate: float = 0.0     # -1..1
    gripper_rate: float = 0.0  # +1 while B (open) is held, -1 while X (close) is held, 0 when neither
    pressed: tuple[str, ...] = field(default_factory=tuple)   # hold / stop: edges, they change state once

    @property
    def moving(self) -> bool:
        return any(abs(v) > 0 for v in (self.waist_rate, self.reach_rate, self.vz, self.pitch_rate, self.roll_rate, self.gripper_rate))


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

    def mirror(self) -> tuple[dict[str, float], dict[str, bool]]:
        """Raw per-key values (sticks -1..1, triggers 0..1) and held buttons, for drawing the pad."""
        axes, buttons = self.pad.snapshot()
        values = {}
        for key, e in self.axes.items():
            if e["axis"] >= len(axes):
                continue
            v = axes[e["axis"]]
            values[key] = (v - e["rest"]) / 2.0 if e["kind"] == "trigger" else (v - e["rest"]) * e["sign"]
        held = {key: idx < len(buttons) and buttons[idx] for key, idx in self.buttons.items()}
        return values, held

    def read(self) -> GamepadState:
        axes, buttons = self.pad.snapshot()
        held = {key: idx < len(buttons) and buttons[idx] for key, idx in self.buttons.items()}
        pressed = []
        for key in ("hold", "stop"):
            if held.get(key) and not self._was_down[key]:
                pressed.append(key)
        self._was_down.update(held)
        return GamepadState(
            waist_rate=self._axis(axes, "waist_left") - self._axis(axes, "waist_right"),
            reach_rate=self._axis(axes, "reach"),
            vz=self._axis(axes, "move_z"),
            pitch_rate=self._axis(axes, "pitch"),
            roll_rate=self._axis(axes, "roll"),
            gripper_rate=float(bool(held.get("gripper_open"))) - float(bool(held.get("gripper_close"))),
            pressed=tuple(pressed),
        )


def integrate(target: Target, state: GamepadState, dt: float, bounds_min, bounds_max,
              roll_limits: tuple[float, float]) -> Target:
    x, y, z = target.xyz
    # cylindrical: the waist turns the target about the base axis, reach slides it in and out
    waist = math.atan2(y, x) + WAIST_SPEED_RAD_S * dt * state.waist_rate
    reach = max(MIN_REACH_M, math.hypot(x, y) + LINEAR_SPEED_MPS * dt * state.reach_rate)
    z = z + LINEAR_SPEED_MPS * dt * state.vz
    xyz = np.clip([reach * math.cos(waist), reach * math.sin(waist), z], bounds_min, bounds_max)
    pitch = float(np.clip(target.pitch - ANGULAR_SPEED_RAD_S * dt * state.pitch_rate, -PITCH_LIMIT_RAD, PITCH_LIMIT_RAD))
    roll = float(np.clip(target.roll + ANGULAR_SPEED_RAD_S * dt * state.roll_rate, roll_limits[0], roll_limits[1]))
    gripper = float(np.clip(target.gripper_pct + GRIPPER_SPEED_PCT_S * dt * state.gripper_rate, 0.0, 100.0))
    return Target(xyz=tuple(float(v) for v in xyz), pitch=pitch, roll=roll, gripper_pct=gripper)
