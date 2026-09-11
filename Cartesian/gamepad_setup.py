"""Pair a gamepad with the SO-101: one control at a time, watched in 3D.

    python Cartesian/gamepad_setup.py --model-dir models/so101

Open http://127.0.0.1:4602 . The page draws the gamepad and, next to it, the arm. The
control being asked for glows; the arm shows what it will do. Move or press it on the
real pad and the wizard records which axis or button that was, then moves on. When all
steps are done the mapping is written to `Cartesian/gamepad_map.json`, which is what
unlocks gamepad control of the arm.

No gamepad plugged in? The page says so and keeps looking.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import viser
from viser.extras import ViserUrdf

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gamepad import Gamepad, discover  # noqa: E402
from gamepad_view import PadView, pad_wxyz  # noqa: E402
from runlog import RunLog  # noqa: E402
from so101_arm import MODEL_START_DEG, q_from_deg  # noqa: E402
from so101_model import ARM_JOINTS, GRIPPER_INDEX, Model  # noqa: E402
from solver import Solver  # noqa: E402
from target import Target  # noqa: E402
from viewer import LOOPBACK, WEB_PORT, MEASURED_COLOR  # noqa: E402

HERE = Path(__file__).resolve().parent
MAP_PATH = HERE / "gamepad_map.json"

LOOP_HZ = 50
AXIS_DETECT = 0.7          # an axis counts as "moved" past this much of its full travel (0..2 scale)
AXIS_RELEASED = 0.2        # ... and as released below this
AXIS_STABLE_TICKS = 5      # 100 ms of the same axis before we believe it
TRIGGER_REST = -0.9        # an axis resting here is a trigger (they idle at -1, sticks at 0)
PULSE_HZ = 1.5             # how fast the asked-for control blinks
DEMO_PERIOD_S = 2.0        # one back-and-forth of the arm demo
DEMO_TRAVEL_M = 0.04
DEMO_ANGLE_RAD = math.radians(30)
PAD_POSITION_M = (0.05, -0.38, 0.12)  # in front of the arm, on the viewer's side
PAD_SCALE = 1.8                       # drawn larger than life so its labels read across the scene
PAD_YAW_DEG = 55                      # turn the pad so its near edge faces the camera
PAD_TILT_DEG = 55                     # and lean its face up toward the camera
CAMERA_POSITION_M = (0.52, -0.62, 0.36)
CAMERA_LOOK_AT_M = (0.12, -0.12, 0.10)
POLL_DEVICE_S = 1.0        # how often to look for a pad when there is none


@dataclass(frozen=True)
class Step:
    key: str            # name in gamepad_map.json
    kind: str           # "axis" or "button"
    part: str           # part on the drawn pad to highlight
    ask_zh: str
    ask_en: str
    does_zh: str
    does_en: str
    demo: tuple | None  # ("xyz", direction) | ("pitch", sign) | ("roll", sign) | ("gripper", sign) | None
    hint_tilt: tuple[float, float] | None = None   # for sticks: which way to lean the knob in the hint


STEPS = (
    Step("move_x", "axis", "left_stick", "把左摇杆向前推到底", "Push the LEFT stick fully FORWARD",
         "末端向前（+x）", "tool moves forward (+x)", ("xyz", (1, 0, 0)), (0, 1)),
    Step("move_y", "axis", "left_stick", "把左摇杆向右推到底", "Push the LEFT stick fully RIGHT",
         "末端向右（−y）", "tool moves right (−y)", ("xyz", (0, -1, 0)), (1, 0)),
    Step("move_z", "axis", "right_stick", "把右摇杆向前推到底", "Push the RIGHT stick fully FORWARD",
         "末端向上（+z）", "tool moves up (+z)", ("xyz", (0, 0, 1)), (0, 1)),
    Step("roll", "axis", "right_stick", "把右摇杆向右推到底", "Push the RIGHT stick fully RIGHT",
         "手腕滚转（roll +）", "wrist rolls (roll +)", ("roll", 1), (1, 0)),
    Step("pitch_down", "axis", "lt", "把左扳机 LT 按到底", "Pull the LEFT trigger (LT) all the way",
         "工具低头（pitch +）", "tool pitches down (pitch +)", ("pitch", 1)),
    Step("pitch_up", "axis", "rt", "把右扳机 RT 按到底", "Pull the RIGHT trigger (RT) all the way",
         "工具抬头（pitch −）", "tool pitches up (pitch −)", ("pitch", -1)),
    Step("gripper_open", "button", "a", "按 A", "Press A", "夹爪打开", "gripper opens", ("gripper", 1)),
    Step("gripper_close", "button", "b", "按 B", "Press B", "夹爪关闭", "gripper closes", ("gripper", -1)),
    Step("hold", "button", "start", "按 Start", "Press Start",
         "真机：上力矩并开始跟随", "real arm: torque on, start following", None),
    Step("stop", "button", "back", "按 Back（Select）", "Press Back (Select)",
         "真机：停止跟随，保持力矩", "real arm: stop following, keep torque", None),
)


class Wizard:
    def __init__(self, model: Model, server: viser.ViserServer, log: RunLog, map_path: Path = MAP_PATH):
        self.model, self.server, self.log = model, server, log
        self.map_path = map_path
        self.solver = Solver(model)
        self.pad: Gamepad | None = None
        self.mapping: dict = {"device": None, "axes": {}, "buttons": {}}
        self.index = 0
        self.done = False
        self.phase = "wait"        # wait -> release -> (next step)
        self.rest: list[float] = []
        self.candidate: tuple[int, int] | None = None   # (axis index, stable ticks)
        self.message = ""
        self.step_started = time.monotonic()
        self._last_poll = 0.0

        scene, gui = server.scene, server.gui
        scene.set_up_direction("+z")

        @server.on_client_connect
        def _(client: viser.ClientHandle) -> None:
            client.camera.position = CAMERA_POSITION_M
            client.camera.look_at = CAMERA_LOOK_AT_M

        scene.add_grid("/ground", width=1.2, height=1.2, plane="xy", cell_size=0.05)
        self.arm = ViserUrdf(server, model.urdf_path, root_node_name="/arm", mesh_color_override=MEASURED_COLOR)
        self._urdf_joints = self.arm.get_actuated_joint_names()
        self.view = PadView(server, "/pad", PAD_POSITION_M, pad_wxyz(PAD_YAW_DEG, PAD_TILT_DEG), scale=PAD_SCALE)
        self.arm_label = scene.add_label("/arm_label", "", position=(0.15, 0.0, 0.42), anchor="bottom-center")

        self.q_demo = q_from_deg(MODEL_START_DEG)
        self.base_target = model.target_from_q(self.q_demo)

        with gui.add_folder("Gamepad pairing"):
            self.device_text = gui.add_markdown("looking for a gamepad ...")
            self.step_text = gui.add_markdown("")
            self.progress = gui.add_markdown("")
            restart = gui.add_button("Restart pairing", icon=viser.Icon.REFRESH)
        restart.on_click(lambda _: self.restart())
        self._restart_requested = False

    # ---- device -----------------------------------------------------------------------

    def _ensure_pad(self) -> bool:
        if self.pad is not None and self.pad.connected:
            return True
        if self.pad is not None:  # was connected, now gone
            self.log.warning("gamepad lost", error=self.pad.error)
            self.pad.stop()
            self.pad = None
            self.device_text.content = "**Gamepad unplugged** — plug it back in / 手柄断开了，插回去"
        now = time.monotonic()
        if now - self._last_poll < POLL_DEVICE_S:
            return False
        self._last_poll = now
        found = discover()
        if not found:
            self.device_text.content = "**No gamepad found** — plug one in / 没检测到手柄，插上后自动开始"
            return False
        path, name = found[0]
        self.pad = Gamepad(path)
        self.pad.start()
        time.sleep(0.1)  # let the kernel's initial state events arrive
        self.pad.drain()
        axes, buttons = self.pad.snapshot()
        self.mapping["device"] = name
        self.device_text.content = f"**{name}** at `{path}` — {len(axes)} axes, {len(buttons)} buttons"
        self.log.event("gamepad found", path=str(path), name=name, axes=len(axes), buttons=len(buttons))
        self._begin_step()
        return True

    # ---- steps ------------------------------------------------------------------------

    def restart(self) -> None:
        self._restart_requested = True

    def _do_restart(self) -> None:
        self._restart_requested = False
        self.mapping = {"device": self.mapping["device"], "axes": {}, "buttons": {}}
        self.index, self.done = 0, False
        self.log.event("pairing restarted")
        self._begin_step()

    @property
    def step(self) -> Step:
        return STEPS[self.index]

    def _begin_step(self) -> None:
        self.phase = "wait"
        self.candidate = None
        self.message = ""
        self.step_started = time.monotonic()
        if self.pad is not None:
            self.rest, _ = self.pad.snapshot()
            self.pad.drain()
        self.view.highlight(None)
        s = self.step
        self.step_text.content = (f"### Step {self.index + 1} / {len(STEPS)}\n\n"
                                  f"## {s.ask_zh}\n{s.ask_en}\n\n**作用 / does:** {s.does_zh} · {s.does_en}")
        self.arm_label.text = f"{s.does_zh}  ·  {s.does_en}"

    def _finish(self) -> None:
        self.done = True
        self.view.highlight(None)
        self.map_path.write_text(json.dumps(self.mapping, indent=2, ensure_ascii=False) + "\n")
        self.log.event("pairing complete", path=str(self.map_path), mapping=self.mapping)
        self.step_text.content = ("## 配对完成 / Pairing complete\n\n"
                                  f"Mapping written to `{self.map_path.name}`. Gamepad control is now unlocked "
                                  "for the next stage. Move the sticks: the drawn pad follows.")
        self.arm_label.text = ""

    def _advance(self) -> None:
        self.index += 1
        if self.index >= len(STEPS):
            self._finish()
        else:
            self._begin_step()

    def _record_axis(self, idx: int, delta: float) -> None:
        kind = "trigger" if self.rest[idx] < TRIGGER_REST else "stick"
        for key, entry in self.mapping["axes"].items():
            if entry["axis"] == idx:
                self.message = (f"⚠️ axis {idx} is already **{key}** — try a different control / "
                                f"这个轴已经分配给 {key} 了，换一个")
                self.phase = "release"
                self.log.warning("axis already used", axis=idx, used_for=key, asked=self.step.key)
                return
        self.mapping["axes"][self.step.key] = {"axis": idx, "sign": 1 if delta > 0 else -1, "kind": kind,
                                               "rest": round(self.rest[idx], 3)}
        self.message = f"✅ axis {idx} ({kind}, {'+' if delta > 0 else '−'}) — let go / 松开"
        self.phase = "release"
        self.log.event("axis mapped", step=self.step.key, axis=idx, sign=1 if delta > 0 else -1, kind=kind, delta=delta)

    def _record_button(self, idx: int) -> None:
        for key, used in self.mapping["buttons"].items():
            if used == idx:
                self.message = (f"⚠️ button {idx} is already **{key}** — press a different one / "
                                f"这个键已经分配给 {key} 了，换一个")
                self.phase = "release"
                self.log.warning("button already used", button=idx, used_for=key, asked=self.step.key)
                return
        self.mapping["buttons"][self.step.key] = idx
        self.message = f"✅ button {idx} — let go / 松开"
        self.phase = "release"
        self.log.event("button mapped", step=self.step.key, button=idx)

    def _detect(self) -> None:
        assert self.pad is not None
        if self.done:
            return
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
                    self._advance()
                else:
                    self.phase = "wait"   # conflict case: back to waiting for the right control
                    self.candidate = None

    # ---- drawing ----------------------------------------------------------------------

    def _mirror_pad(self) -> None:
        """Move the drawn sticks/triggers/buttons the way the real ones are, for what is mapped."""
        if self.pad is None:
            return
        axes, buttons = self.pad.snapshot()
        m = self.mapping["axes"]

        def value(key: str) -> float:
            e = m.get(key)
            if e is None or e["axis"] >= len(axes):
                return 0.0
            v = axes[e["axis"]]
            if e["kind"] == "trigger":
                return (v - e["rest"]) / 2.0
            return (v - e["rest"]) * e["sign"]

        hinting = None if self.done or self.phase != "wait" else self.step.part   # the hint owns that knob
        if hinting != "left_stick":
            self.view.set_stick("left_stick", value("move_y"), value("move_x"))
        if hinting != "right_stick":
            self.view.set_stick("right_stick", value("roll"), value("move_z"))
        self.view.set_trigger("lt", max(0.0, value("pitch_down")))
        self.view.set_trigger("rt", max(0.0, value("pitch_up")))
        for key, part in (("gripper_open", "a"), ("gripper_close", "b"), ("hold", "start"), ("stop", "back")):
            idx = self.mapping["buttons"].get(key)
            if idx is not None and idx < len(buttons):
                self.view.set_button(part, buttons[idx])

    def _demo_arm(self, t: float) -> None:
        s = self.step if not self.done else None
        wave = math.sin(2 * math.pi * t / DEMO_PERIOD_S)
        target = self.base_target
        if s is not None and s.demo is not None:
            what, arg = s.demo
            if what == "xyz":
                d = np.array(arg, dtype=float)
                target = target.with_xyz(np.array(target.xyz) + DEMO_TRAVEL_M * (0.5 + 0.5 * wave) * d)
            elif what == "pitch":
                target = Target(target.xyz, target.pitch + DEMO_ANGLE_RAD * (0.5 + 0.5 * wave) * arg, target.roll, target.gripper_pct)
            elif what == "roll":
                target = Target(target.xyz, target.pitch, target.roll + DEMO_ANGLE_RAD * (0.5 + 0.5 * wave) * arg, target.gripper_pct)
            elif what == "gripper":
                pct = 30 + 70 * (0.5 + 0.5 * wave) if arg > 0 else 60 * (0.5 - 0.5 * wave)
                target = Target(target.xyz, target.pitch, target.roll, pct)
        sol = self.solver.solve(self.q_demo, target)
        self.q_demo = sol.q_goal
        by_name = {name: float(self.q_demo[i]) for i, name in enumerate(ARM_JOINTS)}
        by_name["gripper"] = self.model.gripper_angle(float(self.q_demo[GRIPPER_INDEX]))
        self.arm.update_cfg(np.array([by_name[n] for n in self._urdf_joints]))

    def _draw_progress(self) -> None:
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
            rows.append(f"| {s.key} | {s.does_en} | {got} |")
        self.progress.content = ((self.message + "\n\n") if self.message else "") + \
            "| step | does | mapped to |\n|---|---|---|\n" + "\n".join(rows)

    # ---- loop -------------------------------------------------------------------------

    def run(self, stop: threading.Event) -> None:
        dt = 1.0 / LOOP_HZ
        last_progress = 0.0
        while not stop.is_set():
            t0 = time.monotonic()
            if self._restart_requested:
                self._do_restart()
            if self._ensure_pad() and not self.done:
                self._detect()
            if self.pad is not None and not self.done:   # re-checked: _detect may have just finished the last step
                t = t0 - self.step_started
                pulse = 0.5 + 0.5 * math.sin(2 * math.pi * PULSE_HZ * t)
                self.view.highlight(self.step.part, pulse)
                if self.step.hint_tilt and self.phase == "wait":
                    self.view.set_stick(self.step.part, *(np.array(self.step.hint_tilt) * pulse))
            self._mirror_pad()
            self._demo_arm(t0 - self.step_started)
            if t0 - last_progress > 0.2:
                last_progress = t0
                self._draw_progress()
            remaining = dt - (time.monotonic() - t0)
            if remaining > 0:
                time.sleep(remaining)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--web-port", type=int, default=WEB_PORT)
    p.add_argument("--logs-dir", type=Path, default=HERE / "logs")
    args = p.parse_args(argv)

    log = RunLog(args.logs_dir, "gamepad-setup")
    model = Model(args.model_dir)
    server = viser.ViserServer(host=LOOPBACK, port=args.web_port, label="SO-101 gamepad pairing", verbose=False)
    wizard = Wizard(model, server, log)
    print(f"open http://{LOOPBACK}:{args.web_port}  (logs: {log.dir})", flush=True)
    stop = threading.Event()
    thread = threading.Thread(target=wizard.run, args=(stop,), name="wizard", daemon=True)
    thread.start()
    try:
        while thread.is_alive():
            time.sleep(0.2)
        return 1
    except KeyboardInterrupt:
        return 0
    finally:
        stop.set()
        if wizard.pad is not None:
            wizard.pad.stop()
        log.close()


if __name__ == "__main__":
    sys.exit(main())
