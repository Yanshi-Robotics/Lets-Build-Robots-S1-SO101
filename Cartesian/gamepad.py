"""Read a gamepad, through whichever interface this operating system offers.

Linux reads /dev/input/js* directly and needs no extra package. Every event on the device
is 8 bytes: time (u32 ms), value (s16), type (u8), number (u8). Type 1 = button, 2 = axis;
bit 0x80 marks the synthetic events the kernel sends on open to report the current state
of every axis and button. Reading happens in its own thread.

macOS and Windows have no such device file, so there the reading goes through pygame's
joystick module, which is LeRobot's own choice as well: `pip install "lerobot[gamepad]"`
pulls in pygame, and lerobot's GamepadController uses it. It is imported lazily, so the
course environment on Linux never needs it installed.

`Gamepad` keeps the latest state (axes in -1..1, buttons as bools) and a queue of changes
for anything that wants to react to presses rather than poll. Nothing here knows what the
axes mean, that is `gamepad_map.json`'s job (written by the pairing on the Gamepad page,
gamepad_pairing.py).

⛔ The public face — start / stop / snapshot / drain / axes / buttons / name / connected /
error — is what cartesian_control.py, gamepad_control.py and gamepad_pairing.py use. Keep
it as it is: gamepad_pairing.py's text is on screen in a recorded episode and is frozen.
"""

from __future__ import annotations

import os
import select
import struct
import sys
import threading
from collections import deque
from dataclasses import dataclass
from pathlib import Path

IS_LINUX = sys.platform.startswith("linux")
IS_MACOS = sys.platform == "darwin"

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


def current_backend() -> str:
    """Which reader this platform uses.

    Stored in gamepad_map.json so that a pairing made on one operating system is not
    silently reused on another: the axis and button numbers a pad reports through the Linux
    joystick interface and through SDL are different, so the same map would drive the wrong
    joint. A map written before this field existed counts as "jsdev", which is what it was.
    """
    return "jsdev" if IS_LINUX else "pygame"


def discover() -> list[tuple[object, str]]:
    """Every gamepad this computer offers, with its human-readable name.

    Linux returns (Path, name) and ⛔ must keep doing so: cartesian_control.py hands the
    first element straight to Gamepad(), logs it with str(), and gamepad_map.json is keyed
    on the name read from /sys. The pygame path returns (int index, name) instead.
    """
    if IS_LINUX:
        return [(p, device_name(p)) for p in sorted(DEVICE_DIR.glob(DEVICE_GLOB))]
    return _pygame_discover()


class _JsdevPad:
    """The Linux reader. ⛔ Unchanged from when this file had only one backend."""

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


# --- the pygame backend, for systems without /dev/input/js* -------------------------
# Installed by the course extra: python -m pip install "lerobot[gamepad,feetech]==0.6.1"
# (lerobot's own GamepadController reads a pad through pygame too).

_PYGAME_HINT = ('reading a gamepad on this system needs pygame: '
                'python -m pip install "lerobot[gamepad,feetech]==0.6.1"')

# How much an axis has to move before it counts as a change. The Linux backend reports
# every kernel event; polling has to filter, or resting-stick noise fills the queue and
# pushes real presses out of it. Far below the 0.7 the pairing wizard needs to see.
_AXIS_CHANGE_EPS = 0.05


def _load_pygame():
    """pygame, with the two environment variables that matter set before it is imported."""
    # Without this the pad stops feeding this program as soon as the terminal window takes
    # focus back — which is how the course runs it, so it is not optional.
    os.environ.setdefault("SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS", "1")
    # A dummy video driver only where there is no desktop at all (CI, a headless Linux box).
    # ⛔ Never on macOS with a desktop: SDL discovers pads there through a Core Foundation
    # run loop, and the dummy driver weakens exactly that (libsdl-org/SDL#11742).
    if not IS_MACOS and sys.platform != "win32" and not os.environ.get("DISPLAY"):
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    try:
        import pygame
    except ImportError as exc:
        raise RuntimeError(_PYGAME_HINT) from exc
    return pygame


def _pygame_boot(pygame) -> None:
    """init pygame and its joystick module. Safe to call again; pygame.init() is idempotent."""
    pygame.init()
    pygame.joystick.init()
    if os.environ.get("SDL_VIDEODRIVER") == "dummy" and pygame.display.get_surface() is None:
        # The event queue is documented to misbehave when no video mode was ever set, even
        # with the dummy driver. Nothing is ever drawn on it.
        pygame.display.set_mode((1, 1))


def _pygame_discover() -> list[tuple[int, str]]:
    """Every pad pygame can see, as (index, name). Empty when pygame is not installed."""
    try:
        pygame = _load_pygame()
    except RuntimeError:
        return []   # the page then says "No gamepad detected", same as having no pad
    _pygame_boot(pygame)
    return [(index, pygame.joystick.Joystick(index).get_name())
            for index in range(pygame.joystick.get_count())]


class _PygamePad:
    """A pad read through pygame, by polling rather than in a reader thread.

    ⚠️ SDL requires that init, the event pump and quit all happen on one thread, and on
    macOS that thread has to be the main one, because SDL drives AppKit's event loop there
    and `nextEventMatchingMask:` is main-thread only. This program's control loop runs in a
    daemon thread named "control" (cartesian_control.py builds it), so on macOS this
    backend is used off the main thread. That is why it polls inside snapshot()/drain()
    instead of starting a thread of its own — everything then happens on whichever single
    thread the control loop is — and why macOS prints a warning once. Windows SDL has no
    main-thread rule, so it is unaffected.

    ⏳ Making the main thread pump events would need cartesian_control.py's `main()` to
    change shape; not worth doing until someone can verify it on a real Mac.
    """

    def __init__(self, index: int):
        self.path = int(index)
        self._pygame = _load_pygame()
        _pygame_boot(self._pygame)
        if self.path >= self._pygame.joystick.get_count():
            raise RuntimeError(f"no gamepad at index {self.path}")
        self._joystick = self._pygame.joystick.Joystick(self.path)
        self.name = self._joystick.get_name()
        self.axes: list[float] = []
        self.buttons: list[bool] = []
        self.changes: deque[Change] = deque(maxlen=EVENT_QUEUE_LENGTH)
        self.connected = False
        self.error: str | None = None
        self._owner: int | None = None

    def _claim_thread(self) -> None:
        """Remember which thread drives SDL, and refuse a second one."""
        current = threading.get_ident()
        if self._owner is None:
            self._owner = current
            if IS_MACOS and threading.current_thread() is not threading.main_thread():
                print("WARNING: reading the gamepad from a thread other than the main one. On macOS "
                      "SDL may not report the pad at all. If it stays undetected, that is this known "
                      "limit, not your pad.", file=sys.stderr)
        elif current != self._owner:
            raise RuntimeError("pygame must be driven from one thread only; this is a second one")

    def start(self) -> None:
        self._claim_thread()
        self._joystick.init()
        self.connected = True
        self._poll()

    def stop(self) -> None:
        self.connected = False
        try:
            self._joystick.quit()
            # quit() has to run on the thread that init()ed, or macOS tears down IOKit
            # objects with SDL still holding pointers into them (libsdl-org/SDL#12255).
            self._pygame.joystick.quit()
            self._pygame.quit()
        except Exception:   # noqa: BLE001 - shutting down; a failure here must not mask the real one
            pass

    def _poll(self) -> None:
        """Pump SDL, then read every axis and button and record what moved.

        Even a program that only calls get_axis/get_button has to pump the queue every
        frame; without it the values never change. event.get() pumps as a side effect.
        """
        self._claim_thread()
        try:
            self._pygame.event.get()
            axes = [self._joystick.get_axis(i) for i in range(self._joystick.get_numaxes())]
            buttons = [bool(self._joystick.get_button(i)) for i in range(self._joystick.get_numbuttons())]
        except Exception as exc:  # noqa: BLE001 - unplugged, or SDL gave up on the device
            self.error = str(exc)
            self.connected = False
            return
        for index, value in enumerate(axes):
            previous = self.axes[index] if index < len(self.axes) else None
            if previous is None or abs(value - previous) > _AXIS_CHANGE_EPS:
                if previous is not None:
                    self.changes.append(Change("axis", index, value))
        for index, pressed in enumerate(buttons):
            previous = self.buttons[index] if index < len(self.buttons) else None
            if previous is not None and pressed != previous:
                self.changes.append(Change("button", index, float(pressed)))
        self.axes, self.buttons = axes, buttons

    def snapshot(self) -> tuple[list[float], list[bool]]:
        self._poll()
        return list(self.axes), list(self.buttons)

    def drain(self) -> list[Change]:
        self._poll()
        out = list(self.changes)
        self.changes.clear()
        return out


class Gamepad:
    """A gamepad, read through whatever this operating system offers.

    ⛔ The attribute and method names below are the whole public face; gamepad_pairing.py
    and gamepad_control.py type-annotate against this class and read `.axes` directly.
    """

    def __init__(self, handle):
        self._impl = _JsdevPad(Path(handle)) if IS_LINUX else _PygamePad(int(handle))

    # -- delegated state ------------------------------------------------------------
    @property
    def path(self):
        return self._impl.path

    @property
    def name(self) -> str:
        return self._impl.name

    @property
    def axes(self) -> list[float]:
        return self._impl.axes

    @property
    def buttons(self) -> list[bool]:
        return self._impl.buttons

    @property
    def connected(self) -> bool:
        return self._impl.connected

    @property
    def error(self) -> str | None:
        return self._impl.error

    # -- delegated behaviour --------------------------------------------------------
    def start(self) -> None:
        self._impl.start()

    def stop(self) -> None:
        self._impl.stop()

    def snapshot(self) -> tuple[list[float], list[bool]]:
        return self._impl.snapshot()

    def drain(self) -> list[Change]:
        return self._impl.drain()
