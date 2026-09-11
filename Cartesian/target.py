"""Stage 2 of the pipeline: what the user wants the tool to do.

A `Target` is the only thing an input device has to produce. The ball in the browser
produces one, the sliders produce one, a joint ring produces one (after FK), and so will
a keyboard, a gamepad or a hand tracker later. Everything downstream (compare, solve,
plan, execute) only ever sees a `Target`.

`TargetBox` is the hand-off point between the UI thread (viser callbacks) and the
control thread. It carries a version number so the control thread can tell "the target
changed, solve again" from "same target, keep walking".

`CommandBox` carries the few things that are not a target: a joint ring being dragged
(joint name + angle) and the three hardware buttons. They are drained once per tick.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Target:
    """Tool target in the robot base frame.

    xyz         : tool point, metres.
    pitch       : radians. 0 = tool pointing horizontally forward, positive = pointing down.
    roll        : radians. 0 = the orientation the tool has when wrist_roll is at 0.
    gripper_pct : 0 (closed) .. 100 (open), LeRobot's convention.
    """

    xyz: tuple[float, float, float]
    pitch: float
    roll: float
    gripper_pct: float

    def with_xyz(self, xyz) -> "Target":
        return replace(self, xyz=(float(xyz[0]), float(xyz[1]), float(xyz[2])))


class TargetBox:
    """Latest target + a version counter. `set` from any thread, `snapshot` from the control thread."""

    def __init__(self, initial: Target):
        self._lock = threading.Lock()
        self._target = initial
        self._version = 0
        self._source = "init"

    def set(self, target: Target, source: str) -> int:
        with self._lock:
            self._target = target
            self._version += 1
            self._source = source
            return self._version

    def snapshot(self) -> tuple[Target, int, str]:
        with self._lock:
            return self._target, self._version, self._source


@dataclass(frozen=True)
class JointCommand:
    """A joint ring is being dragged: put this joint at this angle (radians)."""

    joint: str
    angle: float


HOLD, STOP, RELEASE = "hold", "stop", "release"

# Where the target comes from. One mode is enabled at a time on the page (or none, the
# default: the page only shows the arm); the control loop reads it every tick.
MODE_DRAG, MODE_GAMEPAD, MODE_LEADER = "Drag to move", "Gamepad", "Leader arm"
MODES = (MODE_DRAG, MODE_GAMEPAD, MODE_LEADER)


class CommandBox:
    """Ring drags and hardware buttons, drained once per control tick."""

    def __init__(self):
        self._lock = threading.Lock()
        self._joint: JointCommand | None = None
        self._buttons: list[str] = []

    def push_joint(self, command: JointCommand) -> None:
        with self._lock:
            self._joint = command  # only the latest position of the ring matters

    def push_button(self, name: str) -> None:
        with self._lock:
            self._buttons.append(name)

    def drain(self) -> tuple[JointCommand | None, list[str]]:
        with self._lock:
            joint, buttons = self._joint, self._buttons
            self._joint, self._buttons = None, []
            return joint, buttons
