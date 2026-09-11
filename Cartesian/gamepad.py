"""Read a gamepad through the Linux joystick interface (/dev/input/js*). No extra packages.

Every event on the device is 8 bytes: time (u32 ms), value (s16), type (u8), number (u8).
Type 1 = button, 2 = axis; bit 0x80 marks the synthetic events the kernel sends on open
to report the current state of every axis and button.

`Gamepad` keeps the latest state (axes in -1..1, buttons as bools) and a queue of
changes for anything that wants to react to presses rather than poll. Reading happens
in its own thread; nothing here knows what the axes mean, that is `gamepad_map.json`'s
job (written by the pairing on the Gamepad page, gamepad_pairing.py).
"""

from __future__ import annotations

import os
import select
import struct
import threading
from collections import deque
from dataclasses import dataclass
from pathlib import Path

EVENT_FORMAT = "IhBB"
EVENT_SIZE = struct.calcsize(EVENT_FORMAT)
EVENT_BUTTON, EVENT_AXIS, EVENT_INIT = 0x01, 0x02, 0x80
AXIS_FULL_SCALE = 32767.0
DEVICE_GLOB = "js*"
DEVICE_DIR = Path("/dev/input")
EVENT_QUEUE_LENGTH = 256   # a wizard consumes these; keep enough to not lose a press
READ_TIMEOUT_S = 0.2       # how often the reader thread checks whether it should stop


@dataclass(frozen=True)
class Change:
    kind: str      # "axis" or "button"
    index: int
    value: float   # axis: -1..1; button: 1.0 pressed / 0.0 released


def device_name(path: Path) -> str:
    sys_name = Path("/sys/class/input") / path.name / "device" / "name"
    try:
        return sys_name.read_text().strip()
    except OSError:
        return path.name


def discover() -> list[tuple[Path, str]]:
    """Every joystick device the kernel exposes, with its human-readable name."""
    return [(p, device_name(p)) for p in sorted(DEVICE_DIR.glob(DEVICE_GLOB))]


class Gamepad:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.name = device_name(self.path)
        self.axes: list[float] = []
        self.buttons: list[bool] = []
        self.changes: deque[Change] = deque(maxlen=EVENT_QUEUE_LENGTH)
        self.connected = False
        self.error: str | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._fd: int | None = None
        self._thread = threading.Thread(target=self._run, name="gamepad", daemon=True)

    def start(self) -> None:
        self._fd = os.open(self.path, os.O_RDONLY | os.O_NONBLOCK)
        self.connected = True
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None

    def snapshot(self) -> tuple[list[float], list[bool]]:
        with self._lock:
            return list(self.axes), list(self.buttons)

    def drain(self) -> list[Change]:
        with self._lock:
            out = list(self.changes)
            self.changes.clear()
            return out

    def _run(self) -> None:
        assert self._fd is not None
        while not self._stop.is_set():
            ready, _, _ = select.select([self._fd], [], [], READ_TIMEOUT_S)
            if not ready:
                continue
            try:
                data = os.read(self._fd, EVENT_SIZE * 64)
            except BlockingIOError:
                continue
            except OSError as e:  # unplugged
                self.error = str(e)
                self.connected = False
                return
            with self._lock:
                for i in range(0, len(data) - EVENT_SIZE + 1, EVENT_SIZE):
                    _, value, kind, number = struct.unpack(EVENT_FORMAT, data[i:i + EVENT_SIZE])
                    init = bool(kind & EVENT_INIT)
                    kind &= ~EVENT_INIT
                    if kind == EVENT_AXIS:
                        while len(self.axes) <= number:
                            self.axes.append(0.0)
                        v = value / AXIS_FULL_SCALE
                        self.axes[number] = v
                        if not init:
                            self.changes.append(Change("axis", number, v))
                    elif kind == EVENT_BUTTON:
                        while len(self.buttons) <= number:
                            self.buttons.append(False)
                        self.buttons[number] = bool(value)
                        if not init:
                            self.changes.append(Change("button", number, float(bool(value))))
