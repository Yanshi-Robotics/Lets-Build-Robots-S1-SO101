"""Pairing a gamepad: which axis or button is which, learned one control at a time.

Pure state machine, no viser in here. The control loop calls `tick()` once per tick with
the pad's current state and gets back a `PairingView` describing what the page should
show: which drawn control to light up, the text for the step, the arm demo to play on
the ghost arm, and whether pairing is done. The page (viewer.py) only draws that.

The layout being paired is Interbotix's X-Series arm layout with Xbox names, see
gamepad_control.py. Ten steps: two per stick, the two triggers, four buttons.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from gamepad import Gamepad
from gamepad_control import GamepadState, write_map

AXIS_DETECT = 0.7          # an axis counts as "moved" past this much of its full travel (0..2 scale)
AXIS_RELEASED = 0.2        # ... and as released below this
AXIS_STABLE_TICKS = 5      # 100 ms of the same axis before we believe it
TRIGGER_REST = -0.9        # an axis resting here is a trigger (they idle at -1, sticks at 0)
PULSE_HZ = 1.5             # how fast the asked-for control blinks
DEMO_PERIOD_S = 2.0        # one back-and-forth of the ghost-arm demo
DEMO_TRAVEL_M = 0.04
DEMO_ANGLE_RAD = math.radians(30)


@dataclass(frozen=True)
class Step:
    key: str            # name in gamepad_map.json
    kind: str           # "axis" or "button"
    part: str           # part on the drawn pad to highlight
    ask: str
    does: str
    demo: tuple | None  # ("xyz", direction) | ("pitch", sign) | ("roll", sign) | ("waist", sign) | ("gripper", sign) | None
    hint_tilt: tuple[float, float] | None = None   # for sticks: which way to lean the knob in the hint


STEPS = (
    Step("move_z", "axis", "left_stick", "Push the LEFT stick fully FORWARD",
         "tool moves up (+z)", ("xyz", (0, 0, 1)), (0, 1)),
    Step("reach", "axis", "left_stick", "Push the LEFT stick fully RIGHT",
         "tool reaches out, away from the base", ("xyz", (1, 0, 0)), (1, 0)),
    Step("pitch", "axis", "right_stick", "Push the RIGHT stick fully FORWARD",
         "tool pitches up (nose rises)", ("pitch", -1), (0, 1)),
    Step("roll", "axis", "right_stick", "Push the RIGHT stick fully RIGHT",
         "wrist rolls (roll +)", ("roll", 1), (1, 0)),
    Step("waist_left", "axis", "lt", "Pull the LEFT trigger (LT) all the way",
         "whole arm turns left about the base", ("waist", 1)),
    Step("waist_right", "axis", "rt", "Pull the RIGHT trigger (RT) all the way",
         "whole arm turns right about the base", ("waist", -1)),
    Step("gripper_open", "button", "b", "Press B", "gripper opens while held", ("gripper", 1)),
    Step("gripper_close", "button", "x", "Press X", "gripper closes while held", ("gripper", -1)),
    Step("hold", "button", "start", "Press Start", "real arm: torque on, start following", None),
    Step("stop", "button", "back", "Press Back (Select)", "real arm: stop following, keep torque", None),
)

# What each control does, shown on the Gamepad page while it is the one being used.
EXPLAIN = {
    "left_stick": "**Left stick** — forward/back moves the tool up and down; left/right slides it out and in "
                  "along the arm. Push further to move faster; let go and the arm stops where it is.",
    "right_stick": "**Right stick** — forward/back pitches the tool (nose up / down); left/right rolls the wrist "
                   "about the tool axis. The tool point stays put while rolling: the other joints compensate.",
    "lt": "**LT** — turns the whole arm left about its base. The pull depth sets the speed.",
    "rt": "**RT** — turns the whole arm right about its base. Turning the base is the only way this arm "
          "changes where the tool faces in the horizontal plane.",
    "b": "**B** — opens the gripper while held. Let go and it stops.",
    "x": "**X** — closes the gripper while held. Let go and it stops; on the real arm keep it short "
         "on an object, the servo turns position error into force.",
    "start": "**Start** — on the real arm: torque on, start following.",
    "back": "**Back** — on the real arm: stop following, keep torque.",
}
IDLE_TEXT = "Move a stick or press a button; the control you use lights up on the drawn pad."


def active_control(state: GamepadState) -> str | None:
    """The drawn-pad part doing the most right now, or None when the pad is at rest."""
    candidates = {
        "left_stick": math.hypot(state.reach_rate, state.vz),
        "right_stick": math.hypot(state.pitch_rate, state.roll_rate),
        "lt": max(0.0, state.waist_rate), "rt": max(0.0, -state.waist_rate),
        "b": max(0.0, state.gripper_rate), "x": max(0.0, -state.gripper_rate),
        "start": 1.0 if "hold" in state.pressed else 0.0, "back": 1.0 if "stop" in state.pressed else 0.0,
    }
    part, strength = max(candidates.items(), key=lambda kv: kv[1])
    return part if strength > 0 else None


@dataclass(frozen=True)
class PairingView:
    """What the page should show this tick."""
    done: bool
    step_text: str
    progress_text: str
    highlight: str | None          # drawn-pad part to light up
    pulse: float                   # 0..1 brightness of the highlight
    hint: tuple[str, float, float] | None   # (stick part, x, y) to lean the knob as a hint
    demo: tuple | None             # the step's arm demo spec, played on the ghost arm
    demo_phase: float              # 0..1 position in the demo's back-and-forth


class Pairing:
    def __init__(self, pad: Gamepad, map_path: Path, log, now: float):
        self.pad, self.map_path, self.log = pad, map_path, log
        self.mapping: dict = {"device": pad.name, "axes": {}, "buttons": {}}
        self.index = 0
        self.done = False
        self.phase = "wait"        # wait -> release -> (next step)
        self.rest: list[float] = []
        self.candidate: tuple[int, int] | None = None   # (axis index, stable ticks)
        self.message = ""
        self.step_started = now
        self._begin_step(now)

    @property
    def step(self) -> Step:
        return STEPS[self.index]

    def restart(self, now: float) -> None:
        self.mapping = {"device": self.pad.name, "axes": {}, "buttons": {}}
        self.index, self.done = 0, False
        self.log.event("pairing restarted", device=self.pad.name)
        self._begin_step(now)

    def _begin_step(self, now: float) -> None:
        self.phase = "wait"
        self.candidate = None
        self.message = ""
        self.step_started = now
        self.rest, _ = self.pad.snapshot()
        self.pad.drain()

    def _finish(self) -> None:
        self.done = True
        write_map(self.map_path, self.mapping["device"], self.mapping)
        self.log.event("pairing complete", path=str(self.map_path), mapping=self.mapping)

    def _advance(self, now: float) -> None:
        self.index += 1
        if self.index >= len(STEPS):
            self._finish()
        else:
            self._begin_step(now)

    def _record_axis(self, idx: int, delta: float) -> None:
        kind = "trigger" if self.rest[idx] < TRIGGER_REST else "stick"
        for key, entry in self.mapping["axes"].items():
            if entry["axis"] == idx:
                self.message = f"⚠️ axis {idx} is already **{key}** — try a different control"
                self.phase = "release"
                self.log.warning("axis already used", axis=idx, used_for=key, asked=self.step.key)
                return
        self.mapping["axes"][self.step.key] = {"axis": idx, "sign": 1 if delta > 0 else -1, "kind": kind,
                                               "rest": round(self.rest[idx], 3)}
        self.message = f"✅ axis {idx} ({kind}, {'+' if delta > 0 else '−'}) — now let go"
        self.phase = "release"
        self.log.event("axis mapped", step=self.step.key, axis=idx, sign=1 if delta > 0 else -1, kind=kind, delta=delta)

    def _record_button(self, idx: int) -> None:
        for key, used in self.mapping["buttons"].items():
            if used == idx:
                self.message = f"⚠️ button {idx} is already **{key}** — press a different one"
                self.phase = "release"
                self.log.warning("button already used", button=idx, used_for=key, asked=self.step.key)
                return
        self.mapping["buttons"][self.step.key] = idx
        self.message = f"✅ button {idx} — now let go"
        self.phase = "release"
        self.log.event("button mapped", step=self.step.key, button=idx)

    def _detect(self, now: float) -> None:
        axes, buttons = self.pad.snapshot()
        changes = self.pad.drain()
        step = self.step
        if len(self.rest) < len(axes):
            self.rest = self.rest + axes[len(self.rest):]
        deltas = [a - r for a, r in zip(axes, self.rest)]

        if self.phase == "wait":
            if step.kind == "axis":
                idx = int(np.argmax(np.abs(deltas))) if deltas else -1
                if idx >= 0 and abs(deltas[idx]) >= AXIS_DETECT:
                    if self.candidate and self.candidate[0] == idx:
                        self.candidate = (idx, self.candidate[1] + 1)
                    else:
                        self.candidate = (idx, 1)
                    if self.candidate[1] >= AXIS_STABLE_TICKS:
                        self._record_axis(idx, deltas[idx])
                else:
                    self.candidate = None
            else:
                for change in changes:
                    if change.kind == "button" and change.value > 0.5:
                        self._record_button(change.index)
                        break
        elif self.phase == "release":
            axes_quiet = all(abs(d) < AXIS_RELEASED for d in deltas)
            buttons_quiet = not any(buttons)
            if axes_quiet and buttons_quiet:
                if step.key in self.mapping["axes"] or step.key in self.mapping["buttons"]:
                    self._advance(now)
                else:
                    self.phase = "wait"   # conflict case: back to waiting for the right control
                    self.candidate = None

    def _progress_text(self) -> str:
        rows = []
        for i, s in enumerate(STEPS):
            if s.key in self.mapping["axes"]:
                e = self.mapping["axes"][s.key]
                got = f"axis {e['axis']} {'+' if e['sign'] > 0 else '−'} ({e['kind']})"
            elif s.key in self.mapping["buttons"]:
                got = f"button {self.mapping['buttons'][s.key]}"
            elif i == self.index and not self.done:
                got = "⬅ now"
            else:
                got = ""
            rows.append(f"| {s.key} | {s.does} | {got} |")
        return ((self.message + "\n\n") if self.message else "") + \
            "| step | does | mapped to |\n|---|---|---|\n" + "\n".join(rows)

    def tick(self, now: float) -> PairingView:
        if not self.done:
            self._detect(now)
        if self.done:
            return PairingView(done=True, step_text=f"## Paired\n\n**{self.pad.name}** is paired; enable Gamepad below.",
                               progress_text=self._progress_text(), highlight=None, pulse=0.0, hint=None,
                               demo=None, demo_phase=0.0)
        s = self.step
        t = now - self.step_started
        pulse = 0.5 + 0.5 * math.sin(2 * math.pi * PULSE_HZ * t)
        hint = (s.part, s.hint_tilt[0] * pulse, s.hint_tilt[1] * pulse) if s.hint_tilt and self.phase == "wait" else None
        return PairingView(
            done=False,
            step_text=f"### Step {self.index + 1} / {len(STEPS)}\n\n## {s.ask}\n\n**It will:** {s.does}",
            progress_text=self._progress_text(),
            highlight=s.part, pulse=pulse, hint=hint,
            demo=s.demo, demo_phase=0.5 + 0.5 * math.sin(2 * math.pi * t / DEMO_PERIOD_S),
        )


def demo_target(base, spec, phase: float):
    """The ghost arm's target for a demo spec at phase 0..1 (a back-and-forth)."""
    from target import Target
    what, arg = spec
    if what == "xyz":
        return base.with_xyz(np.array(base.xyz) + DEMO_TRAVEL_M * phase * np.array(arg, dtype=float))
    if what == "pitch":
        return Target(base.xyz, base.pitch + DEMO_ANGLE_RAD * phase * arg, base.roll, base.gripper_pct)
    if what == "roll":
        return Target(base.xyz, base.pitch, base.roll + DEMO_ANGLE_RAD * phase * arg, base.gripper_pct)
    if what == "waist":
        x, y, z = base.xyz
        a = math.atan2(y, x) + DEMO_ANGLE_RAD * phase * arg
        r = math.hypot(x, y)
        return base.with_xyz((r * math.cos(a), r * math.sin(a), z))
    if what == "gripper":
        pct = 30 + 70 * phase if arg > 0 else 60 * (1 - phase)
        return Target(base.xyz, base.pitch, base.roll, pct)
    return base
