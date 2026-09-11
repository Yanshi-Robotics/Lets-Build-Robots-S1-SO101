"""The browser page: what RViz + MoveIt show, done with viser.

Scene
    /measured        the arm as the servos report it (solid). Model mode: the command.
    /commanded       the arm as commanded this tick (translucent orange ghost). Live only.
    /target          the draggable ball = the position part of the Target (translation-only
                     gizmo; the SO-101 cannot take a free orientation, see so101_model.py)
    /target/tool     small axes showing the target tool orientation (pitch/roll + arm yaw)
    /rings/<joint>   one rotation ring per arm joint, on the joint axis; drag = turn the joint

GUI
    pitch / roll / gripper sliders, the three hardware buttons (live only), a status panel.

Threading rule: viser runs callbacks on its own thread. Callbacks here only write into
`TargetBox` / `CommandBox`. The control thread calls `push` / `sync_target` to update
what is drawn. Nothing in this file touches the robot model or the serial bus.
"""

from __future__ import annotations

import math
import time

import numpy as np
import viser
import viser.transforms as tf
from viser.extras import ViserUrdf

from so101_model import ARM_JOINTS, GRIPPER_INDEX, Model
from target import HOLD, RELEASE, STOP, CommandBox, JointCommand, Target, TargetBox

LOOPBACK = "127.0.0.1"   # the page has no login; local access only
WEB_PORT = 4602          # registered in the machine-wide port table for the Season-1 viser page

GRID_SIZE_M = 1.0
GRID_CELL_M = 0.05
BALL_RADIUS_M = 0.012
BALL_GIZMO_SCALE = 0.12
RING_SCALE = 0.09
TOOL_AXES_LENGTH_M = 0.04
MEASURED_COLOR = (0.55, 0.60, 0.68)
COMMANDED_COLOR = (1.0, 0.55, 0.1, 0.35)   # RGBA: translucent orange ghost
BALL_OK_COLOR = (40, 200, 90)
BALL_UNREACHABLE_COLOR = (230, 50, 50)
STATUS_HZ = 5            # status text is for eyes, not for control; 5 Hz is enough
CAMERA_POSITION_M = (0.45, -0.55, 0.35)   # where a new browser tab starts looking from
CAMERA_LOOK_AT_M = (0.2, 0.0, 0.15)       # roughly the middle of the arm's workspace
PITCH_SLIDER_DEG = 180   # +/- range. The three pitch joints can sum well past 90.


def _wxyz(R: np.ndarray) -> np.ndarray:
    return tf.SO3.from_matrix(R).wxyz


def _rotation_about_z(wxyz_base, wxyz_now) -> float:
    """Angle the gizmo was turned about its own z axis, from the two quaternions."""
    M = tf.SO3(np.asarray(wxyz_base)).as_matrix().T @ tf.SO3(np.asarray(wxyz_now)).as_matrix()
    return math.atan2(M[1, 0], M[0, 0])


class Viewer:
    def __init__(self, model: Model, bounds_min, bounds_max, host: str, port: int, live: bool,
                 targets: TargetBox, commands: CommandBox, initial: Target):
        self.model = model
        self.live = live
        self.targets = targets
        self.commands = commands
        self.server = viser.ViserServer(host=host, port=port, label="SO-101 Cartesian control", verbose=False)
        scene, gui = self.server.scene, self.server.gui
        scene.set_up_direction("+z")

        @self.server.on_client_connect
        def _(client: viser.ClientHandle) -> None:
            client.camera.position = CAMERA_POSITION_M
            client.camera.look_at = CAMERA_LOOK_AT_M
        scene.add_grid("/ground", width=GRID_SIZE_M, height=GRID_SIZE_M, plane="xy", cell_size=GRID_CELL_M)

        self.measured = ViserUrdf(self.server, model.urdf_path, root_node_name="/measured",
                                  mesh_color_override=MEASURED_COLOR)
        self.commanded = ViserUrdf(self.server, model.urdf_path, root_node_name="/commanded",
                                   mesh_color_override=COMMANDED_COLOR)
        self._urdf_joint_names = self.measured.get_actuated_joint_names()
        if not live:
            self.commanded.show_visual = False

        # --- the ball ---------------------------------------------------------------
        limits = tuple((float(lo), float(hi)) for lo, hi in zip(bounds_min, bounds_max))
        self.ball = scene.add_transform_controls("/target", scale=BALL_GIZMO_SCALE, disable_rotations=True,
                                                 translation_limits=limits, position=initial.xyz)
        self.ball_ok = scene.add_icosphere("/target/ball", radius=BALL_RADIUS_M, color=BALL_OK_COLOR)
        self.ball_bad = scene.add_icosphere("/target/ball_unreachable", radius=BALL_RADIUS_M,
                                            color=BALL_UNREACHABLE_COLOR, visible=False)
        self.tool_axes = scene.add_frame("/target/tool", axes_length=TOOL_AXES_LENGTH_M, axes_radius=0.002,
                                         origin_radius=0.0)
        self._ball_dragging = False
        self.ball.on_drag_start(lambda _: self._set_ball_dragging(True))
        self.ball.on_drag_end(lambda _: self._set_ball_dragging(False))
        self.ball.on_update(lambda _: self._publish_target("ball"))

        # --- sliders ----------------------------------------------------------------
        roll_lo, roll_hi = (math.degrees(v) for v in model.limits["wrist_roll"])
        with gui.add_folder("Target"):
            self.pitch = gui.add_slider("pitch (deg)", -PITCH_SLIDER_DEG, PITCH_SLIDER_DEG, 1.0,
                                        round(math.degrees(initial.pitch)),
                                        hint="0 = tool level, pointing forward; positive = pointing down")
            self.roll = gui.add_slider("roll (deg)", round(roll_lo), round(roll_hi), 1.0,
                                       round(math.degrees(initial.roll)),
                                       hint="0 = wrist_roll at zero; positive = positive wrist_roll")
            self.gripper = gui.add_slider("gripper (%)", 0, 100, 1, round(initial.gripper_pct),
                                          hint="LeRobot's 0 = closed, 100 = open")
        for slider in (self.pitch, self.roll, self.gripper):
            slider.on_update(lambda _: self._publish_target("slider"))

        # --- joint rings ------------------------------------------------------------
        self.rings: dict[str, viser.TransformControlsHandle] = {}
        self._ring_base_wxyz: dict[str, np.ndarray] = {}
        self._ring_base_angle: dict[str, float] = {}
        self._ring_dragging: set[str] = set()
        self._last_q_goal = np.zeros(6)
        for name in ARM_JOINTS:
            # drei's PivotControls draws the ring about z only when the x and y axes are
            # both active (the ring lies in the xy plane); arrows and planes are switched
            # off separately, so what remains is exactly one ring about the joint axis.
            ring = scene.add_transform_controls(f"/rings/{name}", scale=RING_SCALE, active_axes=(True, True, False),
                                                disable_axes=True, disable_sliders=True, line_width=3.0)
            self.rings[name] = ring
            self._ring_base_wxyz[name] = np.array([1.0, 0.0, 0.0, 0.0])
            self._ring_base_angle[name] = 0.0
            ring.on_drag_start(lambda _, n=name: self._ring_drag_start(n))
            ring.on_drag_end(lambda _, n=name: self._ring_dragging.discard(n))
            ring.on_update(lambda _, n=name: self._ring_update(n))

        # --- buttons (live only) ----------------------------------------------------
        if live:
            with gui.add_folder("Arm"):
                hold = gui.add_button("Hold and follow", hint="park the goal at the present position, torque on, then follow the ball")
                stop = gui.add_button("Stop", color="orange", hint="stop sending goals; torque stays on, the arm holds")
                release = gui.add_button("Release torque", color="red", hint="torque off; catch the arm")
            hold.on_click(lambda _: self.commands.push_button(HOLD))
            stop.on_click(lambda _: self.commands.push_button(STOP))
            release.on_click(lambda _: self.commands.push_button(RELEASE))

        self.status = gui.add_markdown("starting")
        self._status_at = 0.0
        self._syncing = False   # set while sync_target writes the widgets, so their callbacks stay quiet

    # ---- callbacks (viser thread) -----------------------------------------------------

    def _set_ball_dragging(self, on: bool) -> None:
        self._ball_dragging = on

    def _publish_target(self, source: str) -> None:
        if self._syncing:
            return
        p = self.ball.position
        self.targets.set(Target(xyz=(float(p[0]), float(p[1]), float(p[2])),
                                pitch=math.radians(self.pitch.value), roll=math.radians(self.roll.value),
                                gripper_pct=float(self.gripper.value)), source)

    def _ring_drag_start(self, name: str) -> None:
        self._ring_dragging.add(name)
        self._ring_base_wxyz[name] = np.array(self.rings[name].wxyz)
        self._ring_base_angle[name] = float(self._last_q_goal[ARM_JOINTS.index(name)])

    def _ring_update(self, name: str) -> None:
        if name not in self._ring_dragging:
            return
        delta = _rotation_about_z(self._ring_base_wxyz[name], self.rings[name].wxyz)
        self.commands.push_joint(JointCommand(joint=name, angle=self._ring_base_angle[name] + delta))

    # ---- updates (control thread) -----------------------------------------------------

    def _cfg(self, q: np.ndarray) -> np.ndarray:
        by_name = {name: float(q[i]) for i, name in enumerate(ARM_JOINTS)}
        by_name["gripper"] = self.model.gripper_angle(float(q[GRIPPER_INDEX]))
        return np.array([by_name[name] for name in self._urdf_joint_names])

    def sync_target(self, target: Target) -> None:
        """The target came from somewhere other than this page (a ring, the real arm): show it."""
        self._syncing = True
        try:
            if not self._ball_dragging:
                self.ball.position = np.asarray(target.xyz)
            self.pitch.value = round(math.degrees(target.pitch))
            self.roll.value = round(math.degrees(target.roll))
            self.gripper.value = round(target.gripper_pct)
        finally:
            self._syncing = False

    def push(self, q_meas: np.ndarray, q_cmd: np.ndarray, q_goal: np.ndarray,
             joint_frames: dict[str, np.ndarray], tool_R: np.ndarray, reachable: bool, status: str) -> None:
        self.measured.update_cfg(self._cfg(q_meas))
        if self.live:
            self.commanded.update_cfg(self._cfg(q_cmd))
        self._last_q_goal = np.array(q_goal)

        for name, ring in self.rings.items():
            if name in self._ring_dragging:
                continue  # the user owns it right now; overwriting it would fight the drag
            T = joint_frames[name]
            ring.position = T[:3, 3]
            ring.wxyz = _wxyz(T[:3, :3])

        self.tool_axes.wxyz = _wxyz(tool_R)
        self.ball_ok.visible = reachable
        self.ball_bad.visible = not reachable

        now = time.monotonic()
        if now - self._status_at >= 1.0 / STATUS_HZ:
            self._status_at = now
            self.status.content = status
