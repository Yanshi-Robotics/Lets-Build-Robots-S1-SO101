"""A gamepad drawn from viser primitives, so a page can point at one of its controls.

The pad lives under one scene node (`root`), in its own frame: x to the right, y away
from the player, z up. Every control is a named part that can be highlighted, and the
two stick knobs can be tilted to mirror the real sticks. Sizes are metres, roughly a
real controller, so the pad can sit next to the arm in the same scene.
"""

from __future__ import annotations

import math

import numpy as np
import viser
import viser.transforms as tf

BODY_COLOR = (70, 74, 82)
KNOB_COLOR = (30, 30, 34)
TRIM_COLOR = (120, 124, 132)
HIGHLIGHT_COLOR = (255, 214, 0)
FACE_COLORS = {"a": (60, 200, 90), "b": (225, 60, 60), "x": (60, 120, 230), "y": (240, 200, 40)}
STICK_TRAVEL_M = 0.010     # how far a knob is drawn off centre at full deflection
TRIGGER_TRAVEL_M = 0.006   # how far a trigger sinks when fully pulled

# Part name -> (kind, centre (x, y, z), size). kind: box or ball. Sizes: box (w, d, h), ball radius.
PARTS: dict[str, tuple[str, tuple[float, float, float], tuple[float, float, float] | float]] = {
    "body": ("box", (0.0, 0.0, 0.0), (0.160, 0.100, 0.028)),
    "left_grip": ("box", (-0.058, -0.045, -0.006), (0.034, 0.050, 0.026)),
    "right_grip": ("box", (0.058, -0.045, -0.006), (0.034, 0.050, 0.026)),
    "left_stick_base": ("box", (-0.045, 0.006, 0.016), (0.026, 0.026, 0.004)),
    "left_stick": ("ball", (-0.045, 0.006, 0.030), 0.011),
    "right_stick_base": ("box", (0.028, -0.022, 0.016), (0.026, 0.026, 0.004)),
    "right_stick": ("ball", (0.028, -0.022, 0.030), 0.011),
    "dpad_h": ("box", (-0.028, -0.022, 0.017), (0.024, 0.008, 0.006)),
    "dpad_v": ("box", (-0.028, -0.022, 0.017), (0.008, 0.024, 0.006)),
    "a": ("ball", (0.056, -0.007, 0.017), 0.0055),
    "b": ("ball", (0.068, 0.006, 0.017), 0.0055),
    "x": ("ball", (0.044, 0.006, 0.017), 0.0055),
    "y": ("ball", (0.056, 0.019, 0.017), 0.0055),
    "back": ("box", (-0.012, 0.010, 0.016), (0.008, 0.005, 0.004)),
    "start": ("box", (0.012, 0.010, 0.016), (0.008, 0.005, 0.004)),
    "lb": ("box", (-0.045, 0.052, 0.006), (0.036, 0.010, 0.009)),
    "rb": ("box", (0.045, 0.052, 0.006), (0.036, 0.010, 0.009)),
    "lt": ("box", (-0.045, 0.056, -0.008), (0.028, 0.012, 0.016)),
    "rt": ("box", (0.045, 0.056, -0.008), (0.028, 0.012, 0.016)),
}
LABELS = {"left_stick": "L", "right_stick": "R", "a": "A", "b": "B", "x": "X", "y": "Y",
          "back": "Back", "start": "Start", "lb": "LB", "rb": "RB", "lt": "LT", "rt": "RT"}


def _base_color(name: str) -> tuple[int, int, int]:
    if name in FACE_COLORS:
        return FACE_COLORS[name]
    if name.endswith("stick") or name.startswith("dpad") or name in ("back", "start", "lt", "rt"):
        return KNOB_COLOR
    if name in ("lb", "rb") or name.endswith("base"):
        return TRIM_COLOR
    return BODY_COLOR


class PadView:
    def __init__(self, server: viser.ViserServer, root: str, position, wxyz, scale: float = 1.0):
        self.server = server
        self.root = server.scene.add_frame(root, show_axes=False, position=position, wxyz=wxyz, scale=scale)
        self.parts: dict[str, viser.BoxHandle | viser.IcosphereHandle] = {}
        self.centres = {name: np.array(c) for name, (_, c, _) in PARTS.items()}
        for name, (kind, centre, size) in PARTS.items():
            path = f"{root}/{name}"
            if kind == "box":
                self.parts[name] = server.scene.add_box(path, color=_base_color(name), dimensions=size, position=centre)
            else:
                self.parts[name] = server.scene.add_icosphere(path, radius=size, color=_base_color(name), position=centre)
        for name, text in LABELS.items():
            c = self.centres[name]
            server.scene.add_label(f"{root}/label_{name}", text, position=(c[0], c[1], c[2] + 0.016),
                                   anchor="bottom-center", font_screen_scale=0.8)
        self._highlighted: str | None = None

    def highlight(self, name: str | None, phase: float = 1.0) -> None:
        """Paint one part yellow (phase 0..1 fades toward the base colour for a pulse)."""
        if self._highlighted and self._highlighted != name:
            self.parts[self._highlighted].color = _base_color(self._highlighted)
        self._highlighted = name
        if name is None:
            return
        base = np.array(_base_color(name), dtype=float)
        color = base + (np.array(HIGHLIGHT_COLOR, dtype=float) - base) * phase
        self.parts[name].color = tuple(int(v) for v in color)

    def set_stick(self, name: str, x: float, y: float) -> None:
        """Move a knob: x right, y forward, both -1..1."""
        c = self.centres[name]
        self.parts[name].position = (c[0] + STICK_TRAVEL_M * x, c[1] + STICK_TRAVEL_M * y, c[2])

    def set_trigger(self, name: str, amount: float) -> None:
        """Sink a trigger, amount 0..1."""
        c = self.centres[name]
        self.parts[name].position = (c[0], c[1], c[2] - TRIGGER_TRAVEL_M * amount)

    def set_button(self, name: str, pressed: bool) -> None:
        c = self.centres[name]
        self.parts[name].position = (c[0], c[1], c[2] - (0.003 if pressed else 0.0))


def pad_wxyz(yaw_deg: float, tilt_deg: float) -> np.ndarray:
    """Turn the pad about z so its near edge faces `yaw_deg` (0 = -y), then lean its face up by `tilt_deg`."""
    return (tf.SO3.from_z_radians(math.radians(yaw_deg)) @ tf.SO3.from_x_radians(math.radians(tilt_deg))).wxyz
