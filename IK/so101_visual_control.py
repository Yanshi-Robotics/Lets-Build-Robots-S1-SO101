#!/usr/bin/env python3
"""One browser page for the SO-101: drag the gripper where you want it and the arm follows.

    handle     drag the handle in the three-dimensional view; the arm tracks it live
    keyboard   the arrow keys move that same handle, so the arm tracks it the same way
    leader     the leader arm, joint for joint, with no solver in the path -- Lesson 7's chain

    --model-only        the page with no arm attached at all: the same solver, the same handle,
                        nothing to plug in and nothing that can be damaged
    --measure-following what this particular arm needs before it will move (see below)
    --release-torque    let go of every motor and exit, for after a stop left the arm held

⭐ There is exactly one control loop, it runs at 50 Hz, and it is the only thing in this program
that touches the serial port. Every button and handle in the browser does nothing except write
down what the operator wants; the loop reads that on its next tick. ⛔ Do not call the arm from
a Viser callback: those run on the web server's own threads, and two threads sharing one serial
bus corrupt each other's packets in ways that look like a hardware fault.

How a session goes:

    1. a self-check runs first. If anything fails, nothing is started and nothing is powered.
    2. the page opens showing the arm exactly where it is. Nothing is powered, so the arm can
       still be moved by hand, and the page follows it.
    3. pick a mode and press Enable. Only then are the motors powered, and only one mode is live
       at a time.
    4. press End to finish: the mode stops, torque is released, and the page returns to
       read-only, ready for another mode.

⛔ If the program stops any other way -- a crash, Ctrl+C, or the arm being held back until the
loop gives up -- torque is deliberately NOT released, because letting go of a raised arm drops
it. The program says so on its way out and prints the `--release-torque` command.

⚠️ Why this program does not use LeRobot's end-effector pipeline, which it used until
2026-09-10: on hardware, neither the keyboard nor the handle could move the arm at all. The
reasons are measured and written up in `so101_cartesian_demo.py`; the short version is that
those steps are handed `FK(measured) + a small step` as their target, which asks the solver to
travel that small step and no further, and a step that small is far below the position error
this servo needs before it makes any force at all. ⛔ Do not reintroduce a target built by
adding a delta to the measured pose.
"""
from __future__ import annotations

import argparse
import math
import socket
import sys
import threading
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import so101_cartesian_demo as model  # noqa: E402  (same folder, as the lesson downloads it)

VISER_VERSION = "1.1.0"
LOOPBACK = "127.0.0.1"  # Hardware control must never bind a LAN or public interface.
DEFAULT_WEB_PORT = 4602

ARM_COLOR = (0.18, 0.48, 0.72, 1.0)
COMMAND_COLOR = (1.0, 0.61, 0.12, 0.35)
GRID_SIZE_M = 1.2
GRID_CELL_M = 0.05
HANDLE_SCALE = 0.12

SAFETY_NOTICE = (
    "SAFETY: the program stopped without a mode being ended, so torque was NOT released and the\n"
    "        arm is still held. Releasing a raised arm drops it, which is why nothing let go.\n"
    "        When the arm is supported, release it with:\n"
    "            python IK/so101_visual_control.py --release-torque --model-dir {model_dir} \\\n"
    "                --port {port} --robot-id {robot_id} --calibration-dir {calibration_dir}\n"
    "        Cutting DC power does the same thing instantly."
)


class CheckFailed(Exception):
    """One self-check item that has to be fixed before anything is started."""


def release_command(args):
    """The whole command, ready to paste.

    ⚠️ An operator who has just been refused is holding an arm they cannot put down. Telling
    them which flag to look up is not help; the four arguments they would have to retype are
    already in `args`.
    """
    return (f"            python IK/so101_visual_control.py --release-torque \\\n"
            f"                --model-dir {args.model_dir} --port {args.port} \\\n"
            f"                --robot-id {args.robot_id} "
            f"--calibration-dir {args.calibration_dir}")


def validate_web_port(port):
    if not 1024 <= port <= 65535:
        raise ValueError("Choose an explicitly assigned unprivileged web port")
    with socket.socket() as probe:
        # A port left in TIME_WAIT by the previous run must not block a restart; a live
        # listener on it still fails the bind and is reported.
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind((LOOPBACK, port))




# ══ The page ═════════════════════════════════════════════════════════════════════════════════

def gripper_urdf_range(model_dir):
    """How far the model's jaw opens, in radians, read from the model rather than written down."""
    joint = model.load_description(Path(model_dir) / model.URDF_NAME).find("./joint[@name='gripper']/limit")
    if joint is None:
        raise ValueError("The pinned model has no gripper joint limit")
    return float(joint.attrib["lower"]), float(joint.attrib["upper"])


def viewer_configuration(names, degrees, gripper_range):
    """The actuated-joint vector ViserUrdf wants, built by name and never by position.

    ⛔ `get_actuated_joint_names()` is ordered by the URDF's topology, which is not the order the
    motors are on the bus. Measured: it hands back the gripper first and shoulder_pan last, the
    exact reverse. A vector assembled by position would draw a plausible-looking arm in the
    wrong pose, and nothing would report it.
    """
    by_name = {name: math.radians(float(degrees[name])) for name in model.JOINTS}
    low, high = gripper_range
    opening = float(np.clip(float(degrees.get("gripper", model.GRIPPER_MIN_PCT)),
                            model.GRIPPER_MIN_PCT, model.GRIPPER_MAX_PCT))
    by_name["gripper"] = low + (high - low) * opening / 100
    if set(names) != set(by_name):
        raise ValueError("Unexpected actuated joints in the pinned visual model")
    return [by_name[name] for name in names]


class ViserPage:
    """The browser page. ⛔ Every method here runs on a web server thread except `show`.

    Callbacks write down what the operator asked for and return. The control loop picks it up on
    its next tick and is the only thing that talks to the arm.
    """

    def __init__(self, model_dir, web_port, servo, bounds, modes, live):
        self.model_dir = Path(model_dir)
        self.web_port = int(web_port)
        self.servo = servo
        self.bounds = bounds
        self.modes = tuple(modes)
        self.live = bool(live)
        self._lock = threading.Lock()
        self._arm_request = None
        self._end_request = False
        self._armed = None
        self._note = ""
        self.server = None

    # -- setting up --------------------------------------------------------------------------
    def start(self, degrees):
        import viser
        from viser.extras import ViserUrdf

        if version("viser") != VISER_VERSION:
            raise RuntimeError(f"Use viser[urdf]=={VISER_VERSION}")
        self.gripper_range = gripper_urdf_range(self.model_dir)
        urdf_path = self.model_dir / model.URDF_NAME
        self.server = viser.ViserServer(host=LOOPBACK, port=self.web_port,
                                        label="SO-101 . Drag to move")
        self.server.scene.add_grid("/ground", width=GRID_SIZE_M, height=GRID_SIZE_M,
                                   plane="xy", cell_size=GRID_CELL_M)
        self.arm = ViserUrdf(self.server, urdf_path, root_node_name="/arm",
                             mesh_color_override=ARM_COLOR)
        self.commanded = ViserUrdf(self.server, urdf_path, root_node_name="/commanded",
                                   mesh_color_override=COMMAND_COLOR)
        self.joint_names = self.arm.get_actuated_joint_names()
        self.handle = self.server.scene.add_transform_controls(
            "/target", scale=HANDLE_SCALE,
            # ⭐ Position only. The arm places its gripper with five joints, so it cannot be
            # asked for an orientation as well; the solver is told to ignore rotation entirely
            # and the handle says the same thing by not offering a rotation ring.
            disable_rotations=True,
            translation_limits=tuple((float(lo), float(hi)) for lo, hi
                                     in zip(model.bounds_low(self.bounds), model.bounds_high(self.bounds))),
        )
        self._build_panel()
        self.move_handle(self.servo.gripper_xyz(degrees))
        self.show(degrees, degrees, "Nothing is powered." if self.live else "No arm attached.")
        print(f"Open http://{LOOPBACK}:{self.web_port} to drive the arm.")

    def stop(self):
        if self.server is not None:
            self.server.stop()
            self.server = None

    def _build_panel(self):
        gui = self.server.gui
        self.status = gui.add_markdown("Reading the arm.")
        self.notice = gui.add_markdown("")
        self.mode_dropdown = gui.add_dropdown(
            "Mode", [model.MODE_LABELS[mode] for mode in self.modes],
            initial_value=model.MODE_LABELS[self.modes[0]])
        self.enable_button = gui.add_button("Enable this mode")
        self.end_button = gui.add_button("End this mode")
        self.gripper_slider = gui.add_slider(
            "Gripper (LeRobot 0-100)", min=model.GRIPPER_MIN_PCT, max=model.GRIPPER_MAX_PCT,
            step=1, initial_value=model.GRIPPER_MIN_PCT)
        with gui.add_folder("Arrow keys"):
            gui.add_markdown(
                "Arrows move the handle in x and y, shift and right shift move it down and up, "
                "left and right ctrl close and open the gripper.\n\n"
                "Keys are read for the whole desktop, so this page does not need focus.")
            self.keyboard_state = gui.add_markdown("no key held")
        with gui.add_folder("Joints"):
            self.joint_state = gui.add_markdown("")
        self.enable_button.on_click(lambda _event: self._request_arm())
        self.end_button.on_click(lambda _event: self._request_end())
        self._refresh_controls()

    # -- what the loop reads -----------------------------------------------------------------
    def take_arm_request(self):
        with self._lock:
            wanted, self._arm_request = self._arm_request, None
            return wanted

    def take_end_request(self):
        with self._lock:
            asked, self._end_request = self._end_request, False
            return asked

    def handle_xyz(self):
        return np.asarray(self.handle.position, dtype=float)

    def gripper_target(self):
        return float(self.gripper_slider.value)

    def nudge_gripper(self, percent):
        self.set_gripper_target(self.gripper_slider.value + percent)

    def set_gripper_target(self, percent):
        self.gripper_slider.value = float(np.clip(percent, model.GRIPPER_MIN_PCT,
                                                  model.GRIPPER_MAX_PCT))

    # -- what the loop writes ----------------------------------------------------------------
    def move_handle(self, xyz):
        """Put the handle somewhere. ⛔ Only while no mode is live, or the operator is fighting it."""
        self.handle.position = tuple(float(v) for v in np.asarray(xyz, dtype=float))

    def show(self, measured, commanded, status, note="", keys=""):
        """Paint the page. ⭐ Called from the control loop only, once a tick."""
        self.arm.update_cfg(viewer_configuration(self.joint_names, measured, self.gripper_range))
        self.commanded.update_cfg(viewer_configuration(self.joint_names, commanded, self.gripper_range))
        self.status.content = status
        if note != self._note:
            self._note = note
            self.notice.content = note
        self.keyboard_state.content = keys or "no key held"
        self.joint_state.content = "\n\n".join(
            f"`{name:<14}` {float(measured[name]):+8.2f} deg"
            + (f"  (told {float(commanded[name]):+8.2f})"
               if abs(float(commanded[name]) - float(measured[name])) >= 0.05 else "")
            for name in model.JOINTS if name in measured)

    def armed(self, mode):
        with self._lock:
            self._armed = mode
        self._refresh_controls()

    def disarmed(self, message=""):
        with self._lock:
            self._armed = None
        self._refresh_controls()
        if message:
            self.notice.content = message
            self._note = message

    # -- callbacks: these write down a wish and return ---------------------------------------
    def _request_arm(self):
        with self._lock:
            if self._armed is None:
                self._arm_request = self._chosen_mode()

    def _request_end(self):
        with self._lock:
            if self._armed is not None:
                self._end_request = True

    def _chosen_mode(self):
        return next(mode for mode, label in model.MODE_LABELS.items()
                    if label == self.mode_dropdown.value)

    def _refresh_controls(self):
        with self._lock:
            armed = self._armed
        self.enable_button.disabled = armed is not None
        self.end_button.disabled = armed is None
        self.mode_dropdown.disabled = armed is not None
        self.handle.visible = armed != model.LEADER





# ══ Before anything starts ═══════════════════════════════════════════════════════════════════

LESSON_7_CHECK = ("Run Lesson 7's register check to see which arm is on which port:\n"
                  "        python Teleop/so101_teleop_log.py registers "
                  "--leader-port <LEADER_PORT> --follower-port <FOLLOWER_PORT>")


def report(name, detail=""):
    print(f"  ok      {name}" + (f"  ({detail})" if detail else ""), flush=True)


def self_check(args, arm, leader, servo, limits, bounds, wants_keyboard, keyboard_available):
    """Everything that must be true before a server is started or a motor is powered.

    ⛔ Nothing here energises anything and nothing here opens the page: a page that comes up
    looking healthy while none of its modes can work is worse than no page at all.
    """
    print("Self-check:", flush=True)
    if version("lerobot") != model.LEROBOT_VERSION or version("viser") != VISER_VERSION:
        raise CheckFailed(f"Install lerobot=={model.LEROBOT_VERSION} and viser=={VISER_VERSION}; "
                          f"found {version('lerobot')} and {version('viser')}.")
    report("versions", f"lerobot {model.LEROBOT_VERSION}, viser {VISER_VERSION}")
    report("pinned model", f"{Path(args.model_dir) / model.URDF_NAME} matches its recorded hash")

    try:
        validate_web_port(args.web_port)
    except OSError as exc:
        raise CheckFailed(f"Web port {args.web_port} cannot be bound ({exc}).\n"
                          "        Close whatever is listening, or pass another --web-port.") from exc
    report("web port", str(args.web_port))

    if not arm.is_live:
        report("arm", "none attached; this is the model on its own")
        return arm.read(), bounds

    try:
        observation, powered, in_the_motors = arm.read_before_power()
    except Exception as exc:
        raise CheckFailed(f"The follower on {args.port} did not answer ({exc}).\n"
                          "        Check the USB cable, the DC supply, and the port name.") from exc
    report("follower bus", f"{len(observation)} motors answered on {args.port}")

    if powered:
        raise CheckFailed(
            f"Torque is already enabled on {', '.join(powered)}.\n"
            "        A previous run stopped without ending its mode, so it deliberately did not\n"
            "        let go -- releasing a raised arm drops it. Support the arm, then run:\n\n"
            + release_command(args) + "\n\n"
            "        Cutting DC power does the same thing instantly.")
    report("follower torque", "off on all six motors")

    disagreeing = model.calibration_disagrees(in_the_motors, arm.robot.calibration)
    if disagreeing:
        raise CheckFailed(
            f"The follower's motors hold a different calibration from {args.calibration_dir}: "
            f"{', '.join(disagreeing)}.\n        " + LESSON_7_CHECK)
    report("follower calibration", "every motor matches its file")

    degrees = model.joint_degrees(observation)

    # ⚠️ Reported, never refused. An unpowered SO-101 falls onto its shoulder stop and rests
    # there, so refusing this would refuse the pose the arm is always in when you walk up to it.
    on_a_stop = model.joints_on_a_stop(observation, arm.robot.calibration)
    report("joint travel", "no joint is against a stop" if not on_a_stop
           else f"{len(on_a_stop)} joint(s) resting on a stop; they will be held where they are")

    # ⚠️ Also reported rather than refused here, because the leader mode works from a pose the
    # model cannot express. The solving modes refuse it at the moment Enable is pressed.
    past_the_model = model.joints_outside_the_model(degrees, limits)
    report("model range", "every joint is inside the pinned model" if not past_the_model
           else f"{len(past_the_model)} joint(s) past the model; the handle and arrow-key modes "
                "will not enable until they are moved back")

    here = servo.gripper_xyz(degrees)
    widened = model.bounds_including(bounds, here)
    changed = not (np.allclose(widened["min"], model.bounds_low(bounds))
                   and np.allclose(widened["max"], model.bounds_high(bounds)))
    report("workspace", f"gripper at x={here[0]:.3f} y={here[1]:.3f} z={here[2]:.3f} m"
           + (f"; box widened to {tuple(round(float(v), 3) for v in widened['min'])}.."
              f"{tuple(round(float(v), 3) for v in widened['max'])} to contain it" if changed else ""))

    if leader is None:
        report("leader arm", "not requested; that mode is left out")
    else:
        try:
            _reading, leader_powered, leader_motors = model.read_pose_before_power(leader)
        except Exception as exc:
            raise CheckFailed(f"The leader on {args.leader_port} did not answer ({exc}).\n"
                              "        Check its cable and port, or drop --leader-port.") from exc
        leader_disagrees = model.calibration_disagrees(leader_motors, leader.calibration)
        if leader_disagrees:
            raise CheckFailed(
                f"The leader's motors hold a different calibration from "
                f"{args.leader_calibration_dir}: {', '.join(leader_disagrees)}.\n        "
                + LESSON_7_CHECK)
        report("leader arm", f"answered on {args.leader_port}, calibration matches"
               + (f", torque on {', '.join(leader_powered)}" if leader_powered else ""))

    if wants_keyboard and not keyboard_available:
        raise CheckFailed(
            "LeRobot's keyboard device cannot capture keys here.\n"
            "        pynput needs an X11 session: not Wayland, and not a bare ssh connection.\n"
            "        Check `echo $XDG_SESSION_TYPE` prints x11, or pass --no-keyboard.")
    report("keyboard capture", "pynput can read keys in this session" if wants_keyboard
           else "not requested; that mode is left out")

    print("Self-check passed.\n", flush=True)
    return degrees, widened


# ══ Measuring what this particular arm needs ═════════════════════════════════════════════════
# ⭐ MAX_JOINT_SPEED_RAD_S decides how far a command may lead the arm, and on this servo that
# lead is the force. Too little and the arm cannot move at all -- which is the fault this whole
# program was rewritten to fix. The number that matters is how far a motor settles from where it
# was told to go, and that is a property of this arm, this gain and this joint's own load, so it
# is measured rather than assumed.
#
# The measurement is the one from upstream issue #3400: approach the same angle from below and
# from above, and see whether the motor stops in the same place both times. It does not, and the
# distance between the two landings is the width of the band inside which the servo makes no
# useful force.
MEASURE_EXCURSION_DEG = 8.0
MEASURE_SPEED_DEG_S = 20.0
MEASURE_SETTLE_S = 0.7
# What the recommendation leaves on top of the widest band measured.
MEASURE_HEADROOM = 1.5


def ramp_joint(arm, held, name, target, hz=model.CONTROL_HZ, speed=MEASURE_SPEED_DEG_S):
    """Walk one joint to an angle at a gentle fixed rate, holding every other joint still."""
    step = speed / hz
    while abs(held[name] - target) > 1e-9:
        gap = target - held[name]
        held[name] += math.copysign(min(step, abs(gap)), gap)
        arm.send(held)
        time.sleep(1 / hz)
    time.sleep(MEASURE_SETTLE_S)


def measure_following(args):
    """Approach one angle from both sides, joint by joint, and report what this arm needs.

    ⚠️ This moves the arm. It moves one joint at a time, by MEASURE_EXCURSION_DEG and back, at a
    fixed gentle rate, and it refuses to start from a pose where that is a bad idea.
    ⛔ It does not release torque when it finishes, for the same reason nothing else here does.
    """
    servo, limits = model.load_servo(args.model_dir)
    arm = model.LiveArm(args.port, args.robot_id, args.calibration_dir, p_coefficient=args.p_coefficient)
    observation, powered, in_the_motors = arm.read_before_power()
    if powered:
        raise CheckFailed(
            f"Torque is already enabled on {', '.join(powered)}.\n"
            "        Support the arm, then run:\n\n" + release_command(args))
    disagreeing = model.calibration_disagrees(in_the_motors, arm.robot.calibration)
    if disagreeing:
        raise CheckFailed(f"Calibration disagrees on {', '.join(disagreeing)}.\n        "
                          + LESSON_7_CHECK)
    start = model.joint_degrees(observation)
    blocked = (model.joints_on_a_stop(observation, arm.robot.calibration)
               + model.joints_outside_the_model(start, limits))
    if blocked:
        raise CheckFailed("This measurement moves each joint by "
                          f"{MEASURE_EXCURSION_DEG:.0f} deg and back, so it will not start from "
                          "here:\n        " + "\n        ".join(blocked))

    print(f"Measuring with P={args.p_coefficient}. Each joint moves "
          f"{MEASURE_EXCURSION_DEG:.0f} deg and back, twice. Keep clear.\n", flush=True)
    arm.hold(start)
    widest, rows = 0.0, []
    try:
        for name in model.JOINTS:
            held = dict(start)
            landings = []
            for direction in (-1, +1):
                ramp_joint(arm, held, name, start[name] + direction * MEASURE_EXCURSION_DEG)
                ramp_joint(arm, held, name, start[name])
                landings.append(float(arm.read()[name]))
            band = abs(landings[0] - landings[1])
            held_error = max(abs(value - start[name]) for value in landings)
            widest = max(widest, band, held_error)
            rows.append((name, start[name], landings[0], landings[1], band, held_error))
            print(f"  {name:<14} told {start[name]:+7.2f}   from below {landings[0]:+7.2f}   "
                  f"from above {landings[1]:+7.2f}   band {band:5.2f} deg", flush=True)
    finally:
        print(flush=True)

    needed = math.radians(widest * MEASURE_HEADROOM) / model.CONTROL_DT
    allowed = math.degrees(model.MAX_JOINT_SPEED_RAD_S * model.CONTROL_DT)
    print(f"Widest band or holding error: {widest:.2f} deg.\n"
          f"A command may currently lead the arm by {allowed:.2f} deg "
          f"(MAX_JOINT_SPEED_RAD_S = {model.MAX_JOINT_SPEED_RAD_S} at {model.CONTROL_HZ} Hz).")
    if widest * MEASURE_HEADROOM > allowed:
        print(f"⛔ That is not enough for this arm. Set MAX_JOINT_SPEED_RAD_S to at least "
              f"{needed:.1f} in so101_cartesian_demo.py, or raise --p-coefficient.")
    else:
        print("✅ Enough, with headroom. No change needed.")
    print("\n" + SAFETY_NOTICE.format(model_dir=args.model_dir, port=args.port,
                                      robot_id=args.robot_id,
                                      calibration_dir=args.calibration_dir))
    return 0


def release_only(args):
    """Let go of every motor and exit. ⚠️ The arm drops if it is not supported."""
    arm = model.LiveArm(args.port, args.robot_id, args.calibration_dir)
    arm.open_bus()
    try:
        before = arm.read()
        powered = [name for name, value in
                   arm.robot.bus.sync_read("Torque_Enable", normalize=False).items() if value]
        if not powered:
            print("Every motor is already released; nothing to do.")
            return 0
        print(f"Releasing {', '.join(powered)}. Support the arm.", flush=True)
        arm.robot.bus.disable_torque()
        after = arm.read()
        for name in model.MOTORS:
            print(f"    {name:<14} {after[name]:+8.2f}   moved {after[name] - before[name]:+6.2f}")
        return 0
    finally:
        arm.close_bus()


# ══ Putting it together ══════════════════════════════════════════════════════════════════════

def run(args):
    servo, limits = model.load_servo(args.model_dir, max_joint_speed=args.max_joint_speed)
    bounds = model.bounds_dict(args.bounds_min_m, args.bounds_max_m)
    wants_keyboard = not args.no_keyboard
    modes = [model.HANDLE] + ([model.KEYBOARD] if wants_keyboard else [])
    leader = None

    if args.model_only:
        arm = model.ModelArm({**dict(zip(model.JOINTS, model.PREVIEW_JOINTS_DEG)),
                        "gripper": model.GRIPPER_MIN_PCT})
    else:
        arm = model.LiveArm(args.port, args.robot_id, args.calibration_dir,
                      p_coefficient=args.p_coefficient)
        if args.leader_port:
            from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig
            leader = SO101Leader(SO101LeaderConfig(
                port=args.leader_port, id=args.leader_id,
                calibration_dir=Path(args.leader_calibration_dir), use_degrees=True))
            modes.append(model.LEADER)

    keyboard_available = False
    if wants_keyboard:
        from lerobot.utils.keyboard_input import pynput_can_capture
        keyboard_available = pynput_can_capture()

    # ⛔ Before any server and before any motor.
    degrees, bounds = self_check(args, arm, leader, servo, limits, bounds,
                                 wants_keyboard, keyboard_available)

    keyboard = model.keyboard_device(args.robot_id) if wants_keyboard else None
    if keyboard is not None:
        keyboard.connect()
    if leader is not None:
        leader.connect()  # SOLeader.configure() disables torque: it stays free to move by hand

    page = ViserPage(args.model_dir, args.web_port, servo, bounds, modes, arm.is_live)
    page.start(degrees)
    if arm.is_live:
        arm.open_bus()
        print("Nothing is powered. Move the arm by hand if you like; press Enable in a mode to "
              "start.\n", flush=True)
    try:
        model.control_loop(page, arm, servo, bounds, limits, leader=leader, keyboard=keyboard)
    except model.FollowingLost as lost:
        # ⛔ Stopped and still held. The force is already out of the motors -- the loop parked
        # the goal where they stand before raising -- but letting go is a decision made with a
        # hand on the arm, so the program says how and stops.
        print(f"\nSTOP-HOLD: {lost}\n"
              "           The mode is stopped and the arm is still held where it stands.",
              file=sys.stderr, flush=True)
        raise
    except BaseException:
        raise
    finally:
        # ⛔ Only a live arm can be left holding, and only a live arm can drop.
        if arm.is_live and arm.holding:
            print("\n" + SAFETY_NOTICE.format(
                model_dir=args.model_dir, port=args.port, robot_id=args.robot_id,
                calibration_dir=args.calibration_dir), file=sys.stderr, flush=True)
        if keyboard is not None:
            keyboard.disconnect()
        if leader is not None:
            leader.disconnect()
        page.stop()
        if arm.is_live:
            arm.close_bus()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--model-only", action="store_true",
                        help="open the page with no arm attached: no serial port is opened")
    parser.add_argument("--measure-following", action="store_true",
                        help="measure how far this arm settles from where it is told, joint by "
                             "joint, and say whether the command lead allowed is enough")
    parser.add_argument("--release-torque", action="store_true",
                        help="let go of every motor and exit; for after the program stopped "
                             "without a mode being ended. The arm drops if it is not supported.")
    parser.add_argument("--port", help="follower serial port")
    parser.add_argument("--robot-id", default="so101-follower")
    parser.add_argument("--calibration-dir", default="calibration/follower")
    parser.add_argument("--leader-port", help="add the leader-arm mode; omit to leave it out")
    parser.add_argument("--leader-id")
    parser.add_argument("--leader-calibration-dir", default="calibration/leader")
    parser.add_argument("--web-port", type=int, default=DEFAULT_WEB_PORT)
    parser.add_argument("--no-keyboard", action="store_true",
                        help="leave out the arrow-key mode, and its need for an X11 session")
    parser.add_argument("--max-joint-speed", type=float, default=model.MAX_JOINT_SPEED_RAD_S,
                        help="radians a second, per joint: how far a command may lead the arm, "
                             "which on this servo is how hard it pushes")
    parser.add_argument("--p-coefficient", type=int, default=model.SERVO_P_COEFFICIENT,
                        help="proportional gain written to every motor (Feetech ships 32; "
                             "LeRobot's own default is 16)")
    parser.add_argument("--bounds-min-m", nargs=3, type=float, default=model.DEFAULT_BOUNDS_M["min"])
    parser.add_argument("--bounds-max-m", nargs=3, type=float, default=model.DEFAULT_BOUNDS_M["max"])
    args = parser.parse_args(argv)

    if not args.model_only and not args.port:
        parser.error("--port is required unless --model-only is given")
    if args.leader_port and not args.leader_id:
        parser.error("--leader-port needs --leader-id, the name its calibration file is under")
    if args.max_joint_speed <= 0:
        parser.error("--max-joint-speed must be positive")

    try:
        if args.release_torque:
            return release_only(args)
        if args.measure_following:
            return measure_following(args)
        return run(args)
    except CheckFailed as failed:
        print(f"\nSelf-check FAILED, so nothing was started and nothing was powered:\n"
              f"    {failed}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"STOP: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
