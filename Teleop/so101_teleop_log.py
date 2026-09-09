#!/usr/bin/env python3
"""SO-101 teleoperation diagnostics for LeRobot 0.6.1: three checks, none of which drive a motor.

Teleoperation itself is always the official `lerobot-teleoperate`, and recording is always the
official `lerobot-record`. This program never writes a motor register and never sends an action;
it fills the three gaps those tools leave:

  registers  the protection, calibration and goal registers of both arms, next to the
             calibration files, so a latched motor or a stale goal is visible before power-on.
  compare    both arms read side by side with torque off, joint by joint. This is the check
             that separates "the two arms disagree about zero" from "a joint runs backwards".
  report     a dataset recorded by `lerobot-record`, graded per joint: how far the Follower
             stayed behind the Leader, and whether the two moved the same way at all.

Read-only software does not guarantee an unpowered or motionless arm. Both arms must be
supported, the DC cutoff within reach, and `registers` and `compare` refuse to run while any
motor has torque enabled.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

JOINT_NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
ARM_JOINTS = JOINT_NAMES[:5]  # the gripper is a 0-100 opening, not an angle
LABELS = {"leader": "L", "follower": "F"}
LEROBOT_VERSION = "0.6.1"
ENCODER_STEPS = 4095  # one turn, matching LeRobot's model_resolution_table minus one

# Registers read by `registers`, in the order they are printed. Values are raw: sign decoding is
# LeRobot's business (Present_Load and Homing_Offset carry a sign bit), normalisation is not applied.
REGISTERS = (
    "Torque_Enable", "Status", "Operating_Mode", "Phase",
    "Torque_Limit", "Max_Torque_Limit", "Protective_Torque", "Overload_Torque",
    "Protection_Current", "Protection_Time", "Unloading_Condition",
    "Present_Voltage", "Present_Temperature", "Present_Load", "Present_Current",
    "Goal_Position", "Present_Position",
    "Homing_Offset", "Min_Position_Limit", "Max_Position_Limit",
    "P_Coefficient", "I_Coefficient", "D_Coefficient",
)
# Status (0x41) is a bit field: a set bit is an error the servo latched.
STATUS_BITS = ((0, "voltage"), (1, "sensor"), (2, "temperature"), (3, "current"), (4, "angle"), (5, "overload"))
# Registers the calibration file also carries, compared motor by motor.
CALIBRATED_REGISTERS = {"Homing_Offset": "homing_offset", "Min_Position_Limit": "range_min", "Max_Position_Limit": "range_max"}

COMPARE_HZ = 5.0
COMPARE_PRINT_SECONDS = 1.0  # Sample fast, print slowly: both hands are on the arms.
STOP_MARGIN_STEPS = 40  # About 3.5 deg. Closer than this to a recorded end counts as against the stop.
LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_KEEP = 20
LOG = logging.getLogger("so101_teleop_log")


def wrapped_difference(degrees):
    """Fold an angle difference into [-180, 180); half a turn folds to -180.

    Only wrist_roll needs this: LeRobot records its range as a full turn on both arms, so its zero
    sits opposite a seam. Two arms 10 degrees apart across that seam read 350 degrees apart without
    the fold. Every other joint travels well under a turn, and its seam is outside the travel.
    """
    if not math.isfinite(float(degrees)):
        raise ValueError("Angle difference must be finite")
    return (float(degrees) + 180.0) % 360.0 - 180.0


def joint_difference(name, leader_value, follower_value):
    """Leader minus Follower for one joint, folded only where the seam can be crossed."""
    difference = float(leader_value) - float(follower_value)
    return wrapped_difference(difference) if name == "wrist_roll" else difference


def decode_status(value):
    """Names of the error bits the servo has latched; empty when the motor is healthy."""
    return [name for bit, name in STATUS_BITS if int(value) >> bit & 1]


MOVED_DEGREES = 0.2  # A couple of encoder steps: enough to have a direction rather than noise.
PAIRED_TICKS_NEEDED = 5  # Fewer than this and the two arms were not moved together enough to judge.


def agreement(leader_series, follower_series):
    """How often the two arms moved the same way, over the ticks where BOTH of them moved.

    1.0 is same direction every time, 0.0 is opposite every time. Returns None when the two were
    not moved together often enough to say anything, which is the common case when a person moves
    them by hand one at a time: an arm that simply stayed still must not be read as a reversed one.
    """
    if len(leader_series) != len(follower_series):
        raise ValueError("Series must be the same length")
    same = paired = 0
    for index in range(1, len(leader_series)):
        leader_step = leader_series[index] - leader_series[index - 1]
        follower_step = follower_series[index] - follower_series[index - 1]
        if abs(leader_step) < MOVED_DEGREES or abs(follower_step) < MOVED_DEGREES:
            continue
        paired += 1
        same += (leader_step > 0) == (follower_step > 0)
    return same / paired if paired >= PAIRED_TICKS_NEEDED else None


def longest_still_run(series, tolerance=0.05):
    """Longest run of consecutive samples in which the value did not change."""
    longest = run = 1 if series else 0
    for index in range(1, len(series)):
        run = run + 1 if abs(series[index] - series[index - 1]) < tolerance else 1
        longest = max(longest, run)
    return longest


def start_log(name, args):
    """One log per run next to this program; the folder is git-ignored and pruned."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = LOG_DIR / f"so101_teleop_log_{name}_{datetime.now():%Y%m%d_%H%M%S}.log"
    for handler in list(LOG.handlers):
        LOG.removeHandler(handler)
        handler.close()
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s.%(msecs)03d %(message)s", "%H:%M:%S"))
    LOG.addHandler(handler)
    LOG.setLevel(logging.DEBUG)
    LOG.propagate = False
    for old in sorted(LOG_DIR.glob("so101_teleop_log_*.log"))[:-LOG_KEEP]:
        old.unlink()
    LOG.info("start %s", " ".join(sys.argv))
    LOG.info("args %s", vars(args))
    LOG.info("lerobot %s", version("lerobot"))
    return path


def say(text):
    """Print for the operator and keep the same line in the log."""
    print(text)
    LOG.info(text)


def open_bus(role, port, motor_ids=None):
    """A read-only bus for one arm. No calibration: `registers` wants raw register values."""
    from lerobot.motors import Motor, MotorNormMode
    from lerobot.motors.feetech import FeetechMotorsBus
    return FeetechMotorsBus(port=port, motors={
        name: Motor(index, "sts3215", MotorNormMode.DEGREES)
        for index, name in enumerate(JOINT_NAMES, 1)
    })


def open_arm(role, port, arm_id, calibration_dir):
    """A calibrated arm object for `compare`, connected the way the course connects: bus only.

    The official Robot.connect() also runs configure(), which re-enables torque on exit, so it is
    never used here. Only the bus handshake runs, and that reads.
    """
    if role == "follower":
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
        return SO101Follower(SO101FollowerConfig(port=port, id=arm_id, calibration_dir=Path(calibration_dir), use_degrees=True, cameras={}))
    from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig
    return SO101Leader(SO101LeaderConfig(port=port, id=arm_id, calibration_dir=Path(calibration_dir), use_degrees=True))


def refuse_if_powered(bus, role):
    """Both read-only modes require every motor released: a held arm can move when a hand pushes it."""
    powered = [name for name in JOINT_NAMES if bus.read("Torque_Enable", name, normalize=False) != 0]
    if powered:
        raise RuntimeError(
            f"{role}: torque is enabled on {', '.join(powered)}. This program only reads; power the arm "
            "off (or cut DC power) before running it."
        )


def dump_registers(bus, role, calibration):
    """Every register in REGISTERS for six motors, plus the calibration comparison."""
    prefix = LABELS[role]
    rows = {}
    for index, name in enumerate(JOINT_NAMES, 1):
        rows[name] = {register: bus.read(register, name, normalize=False) for register in REGISTERS}
        values = rows[name]
        errors = decode_status(values["Status"])
        say(f"{prefix}{index} {name}")
        say(f"    status {values['Status']}"
            + (f"  ERRORS: {', '.join(errors)}" if errors else "  (no error bits)")
            + f"   torque_enable {values['Torque_Enable']}")
        say(f"    torque limit {values['Torque_Limit']} of max {values['Max_Torque_Limit']}"
            f"   protective {values['Protective_Torque']}   overload threshold {values['Overload_Torque']}"
            f"   protection current {values['Protection_Current']}")
        say(f"    voltage {values['Present_Voltage'] / 10:.1f} V   temperature {values['Present_Temperature']} C"
            f"   load {values['Present_Load']}   current {values['Present_Current']}")
        say(f"    goal {values['Goal_Position']}   present {values['Present_Position']}"
            f"   difference {values['Goal_Position'] - values['Present_Position']} steps")
        say(f"    P {values['P_Coefficient']}  I {values['I_Coefficient']}  D {values['D_Coefficient']}"
            f"   mode {values['Operating_Mode']}   phase {values['Phase']}"
            f"   unloading condition {values['Unloading_Condition']}")
        for register, field in CALIBRATED_REGISTERS.items():
            expected = calibration.get(name, {}).get(field)
            verdict = "MATCH" if expected is not None and int(expected) == int(values[register]) else "MISMATCH"
            say(f"    {register} {values[register]}   file {expected}   {verdict}")
    return rows


def summarise_registers(rows, role):
    """One line per problem, so the operator knows whether to continue."""
    problems = []
    for name, values in rows.items():
        errors = decode_status(values["Status"])
        if errors:
            problems.append(f"{name}: status bits {', '.join(errors)}")
        if values["Torque_Limit"] != values["Max_Torque_Limit"]:
            problems.append(f"{name}: torque limit {values['Torque_Limit']} below its maximum {values['Max_Torque_Limit']}"
                            " (it lives in RAM; cutting DC power reloads it)")
    # A goal far from the present position is worth one line for the arm, not six. Every motor
    # reading a goal of 0 is what a DC power cycle leaves behind: enabling torque before writing a
    # goal would send each motor toward step 0, which is a hard stop.
    stale = [name for name, values in rows.items() if abs(values["Goal_Position"] - values["Present_Position"]) > 100]
    if len(stale) == len(rows) and all(values["Goal_Position"] == 0 for values in rows.values()):
        problems.append("every motor holds a goal of 0, which is what a DC power cycle leaves. Enabling torque "
                        "before writing a goal would drive all six toward step 0. Write the present position first.")
    elif stale:
        problems.append(f"{', '.join(stale)}: the stored goal is far from the present position; enabling torque "
                        "would move there")
    for problem in problems:
        say(f"  {role}: {problem}")
    return problems


def calibration_triples(source, motors):
    """(homing_offset, range_min, range_max) per motor, from a file or from a register dump."""
    if all(isinstance(value, dict) and "homing_offset" in value for value in source.values()):
        return {name: tuple(int(source[name][field]) for field in ("homing_offset", "range_min", "range_max"))
                for name in motors if name in source}
    return {name: tuple(int(source[name][register]) for register in CALIBRATED_REGISTERS)
            for name in motors if name in source}


def swapped_ports(rows, calibrations):
    """True when each arm's registers are the other role's calibration file, motor for motor.

    Linux does not keep serial device names stable, so the arm on a given port changes between
    sessions. Calibration lives in the motors, so a swap shows up here as every motor mismatching
    its own file while matching the other one exactly. Twelve MISMATCH lines mean this far more
    often than they mean a broken calibration.
    """
    roles = ("leader", "follower")
    if not all(rows.get(role) and calibrations.get(role) for role in roles):
        return False
    crossed = all(
        calibration_triples(rows[role], JOINT_NAMES) == calibration_triples(calibrations[other], JOINT_NAMES)
        for role, other in (roles, roles[::-1])
    )
    matched = any(
        calibration_triples(rows[role], JOINT_NAMES) == calibration_triples(calibrations[role], JOINT_NAMES)
        for role in roles
    )
    return crossed and not matched


def run_registers(args):
    calibrations = {}
    for role, path in (("leader", args.leader_calibration), ("follower", args.follower_calibration)):
        file = Path(path)
        calibrations[role] = json.loads(file.read_text(encoding="utf-8")) if file.is_file() else {}
        if not calibrations[role]:
            say(f"{role}: no calibration file at {file}; register comparison is skipped")
    problems, dumps = [], {}
    ports = {"leader": args.leader_port, "follower": args.follower_port}
    for role, port in ports.items():
        bus = open_bus(role, port)
        try:
            bus.connect()
            refuse_if_powered(bus, role)
            say(f"=== {role} on {port} ===")
            dumps[role] = dump_registers(bus, role, calibrations[role])
            problems += summarise_registers(dumps[role], role)
        finally:
            if bus.is_connected:
                bus.disconnect(disable_torque=False)
    say("")
    if swapped_ports(dumps, calibrations):
        say("PORTS ARE SWAPPED. Each arm holds the other one's calibration, motor for motor, so the two")
        say(f"ports are the other way round: the Leader is on {ports['follower']} and the Follower is on {ports['leader']}.")
        say("Serial device names are assigned in the order the boards enumerate, so they change between")
        say("sessions. Swap the two --*-port arguments and run this again; every MISMATCH above should clear.")
        say("Until they agree, any teleoperation command would read one arm through the other's calibration.")
        return 1
    say("SUMMARY: nothing to flag." if not problems else f"SUMMARY: {len(problems)} thing(s) to look at, listed above.")
    return 0


def compare_line(index, name, leader_degrees, leader_raw, follower_degrees, follower_raw):
    unit = "%" if name == "gripper" else "d"
    difference = joint_difference(name, leader_degrees, follower_degrees)
    flag = "  <-- wrapped" if name == "wrist_roll" and abs(leader_degrees - follower_degrees) > 180 else ""
    return (f"  {index} {name:<14} L {leader_degrees:8.2f}{unit} ({leader_raw:4d})"
            f"   F {follower_degrees:8.2f}{unit} ({follower_raw:4d})   diff {difference:+8.2f}{flag}")


def stop_reference(samples, ranges):
    """The sample where both arms sat closest to the same mechanical end, and the difference there.

    A joint's zero is the midpoint of its recorded travel, so the two arms only become comparable
    when both are against the same physical end. Finding that instant by eye while both hands are on
    the arms does not work; this finds it in the recording instead.

    What it checks is that the recorded travel still matches the real stops: an arm reads minus half
    its own recorded range at its own low stop, so a difference here means the two recordings differ,
    not that a joint is reversed. Direction cannot be judged this way at all, and is judged during
    teleoperation instead. wrist_roll has no ends to find, LeRobot recording its range as a full turn.
    """
    best = None
    for sample in samples:
        distances, ends = {}, {}
        for role in ("leader", "follower"):
            low, high = ranges[role]
            raw = sample[f"{role}_raw"]
            distances[role] = min(raw - low, high - raw)
            ends[role] = "low" if raw - low < high - raw else "high"
        if ends["leader"] != ends["follower"]:
            continue
        worst = max(distances.values())
        if worst > STOP_MARGIN_STEPS:
            continue
        if best is None or worst < best[0]:
            best = (worst, ends["leader"], sample["difference"])
    return best


def summarise_compare(history, ranges):
    """One line per joint: what the two arms did, and whether they did it together."""
    say("")
    say("=== summary ===")
    say(f"  {'joint':<15}{'diff seen':>18}{'direction':>11}   reference")
    verdicts = []
    for name in JOINT_NAMES:
        samples = history[name]
        differences = [sample["difference"] for sample in samples]
        score = agreement([sample["leader"] for sample in samples], [sample["follower"] for sample in samples])
        best = None if name == "wrist_roll" else stop_reference(samples, ranges[name])
        if name == "wrist_roll":
            reference = "no stop: align both wrists by sight, then read the difference"
        elif best is None:
            reference = "never both against the same stop; push both there and run again"
        else:
            reference = f"both on the {best[1]} stop: difference {best[2]:+.2f}"
            if abs(best[2]) > 3:
                verdicts.append(f"{name}: {best[2]:+.1f} apart with both arms on the {best[1]} stop; that is a"
                                " calibration difference, not a pose difference")
        say(f"  {name:<15}{min(differences):+8.2f} .. {max(differences):+6.2f}"
            + (f"{score:>11.2f}" if score is not None else f"{'-':>11}")
            + f"   {reference}")
        if score is not None and score < 0.5:
            verdicts.append(f"{name}: the two arms read opposite ways while both were moving (agreement"
                            f" {score:.2f}). Move this one joint on both arms together and run again to confirm")
        elif score is not None and score < 0.9:
            verdicts.append(f"{name}: only {score:.0%} of the paired movements agreed; move it further and more"
                            " slowly to be sure")
    unmoved = [name for name in JOINT_NAMES
               if agreement([sample["leader"] for sample in history[name]],
                            [sample["follower"] for sample in history[name]]) is None]
    if unmoved:
        say("")
        say("  Direction not tested on these: the two arms were never moved together enough.")
        say(f"  {', '.join(unmoved)}")
        say("  Moving one arm at a time cannot show direction; that is judged during teleoperation.")
    say("")
    for verdict in verdicts:
        say(f"  {verdict}")
    if not verdicts and not unmoved:
        say("  Every joint moved the same way on both arms, and the stop references agree within 3 degrees.")


def run_compare(args):
    # Defined before connecting: Ctrl+C during the handshake must not fall over the summary.
    arms, buses, history, ranges = {}, {}, {}, {}
    try:
        for role, port, arm_id, calibration_dir in (
            ("leader", args.leader_port, args.leader_id, args.leader_calibration_dir),
            ("follower", args.follower_port, args.follower_id, args.follower_calibration_dir),
        ):
            arm = open_arm(role, port, arm_id, calibration_dir)
            arms[role] = arm
            buses[role] = arm.bus
            arm.bus.connect()
            refuse_if_powered(arm.bus, role)
            if not arm.calibration:
                raise RuntimeError(f"{role}: no calibration loaded from {calibration_dir}; run Lesson 6 first")
            if not arm.bus.is_calibrated:
                raise RuntimeError(f"{role}: the calibration file does not match what the motors hold; recalibrate")
        say("Both arms are released. Use both hands, watch the arms rather than the screen, and do this:")
        say("  1. push the same joint of both arms against the same end of its travel, one joint at a time;")
        say("  2. then turn each joint of both arms the same way, slowly, through a wide angle;")
        say("  3. turn both wrist rolls to the same orientation by sight.")
        say("Press Ctrl+C when you are done. Everything is judged from the recording, not from the screen.")
        ranges.update({name: {role: (arms[role].calibration[name].range_min, arms[role].calibration[name].range_max)
                              for role in arms} for name in JOINT_NAMES})
        history.update({name: [] for name in JOINT_NAMES})
        last_print = 0.0
        while True:
            tick = time.monotonic()
            readings = {role: (arms[role].bus.sync_read("Present_Position"),
                               arms[role].bus.sync_read("Present_Position", normalize=False)) for role in arms}
            for name in JOINT_NAMES:
                history[name].append({
                    "leader": readings["leader"][0][name], "follower": readings["follower"][0][name],
                    # Raw steps as well: the recorded travel is in steps, so the stop test needs them.
                    "leader_raw": readings["leader"][1][name], "follower_raw": readings["follower"][1][name],
                    "difference": joint_difference(name, readings["leader"][0][name], readings["follower"][0][name]),
                })
            if tick - last_print >= COMPARE_PRINT_SECONDS:
                last_print = tick
                say("")
                for index, name in enumerate(JOINT_NAMES, 1):
                    say(compare_line(index, name,
                                     readings["leader"][0][name], readings["leader"][1][name],
                                     readings["follower"][0][name], readings["follower"][1][name]))
            time.sleep(max(0.0, 1.0 / COMPARE_HZ - (time.monotonic() - tick)))
    except KeyboardInterrupt:
        if history:
            summarise_compare(history, ranges)
        say("")
        say("Stopped. No motor was written; the arms were never powered by this program.")
        return 0
    finally:
        for bus in buses.values():
            if bus.is_connected:
                bus.disconnect(disable_torque=False)


def load_recording(root):
    """Leader actions and Follower states from a dataset written by `lerobot-record`."""
    import pandas as pd
    root = Path(root)
    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    features = info["features"]
    if "action" not in features or "observation.state" not in features:
        raise ValueError(f"{root} has no action/observation.state features; is it a teleoperation recording?")
    parquets = sorted(root.glob("data/**/*.parquet"))
    if not parquets:
        raise ValueError(f"No parquet files under {root / 'data'}")
    frame = pd.concat([pd.read_parquet(path) for path in parquets], ignore_index=True)
    return features["action"]["names"], features["observation.state"]["names"], frame, info.get("fps", 30)


def run_report(args):
    action_names, state_names, frame, fps = load_recording(args.dataset)
    if action_names != state_names:
        say(f"WARNING: the two arms report different joint names: {action_names} against {state_names}")
    say(f"{len(frame)} frames at {fps} Hz ({len(frame) / fps:.1f} s) from {args.dataset}")
    say("")
    say("  joint            start diff   max diff   agreement   Follower still")
    verdicts = []
    for index, name in enumerate(action_names):
        joint = name.removesuffix(".pos")
        leader = [float(row[index]) for row in frame["action"]]
        follower = [float(row[index]) for row in frame["observation.state"]]
        differences = [joint_difference(joint, a, b) for a, b in zip(leader, follower)]
        worst = max(differences, key=abs)
        score = agreement(leader, follower)
        still = longest_still_run(follower)
        say(f"  {joint:<14} {differences[0]:+9.2f} {worst:+10.2f}   "
            + (f"{score:8.2f}" if score is not None else "       -")
            + f"   {still / fps:6.1f} s")
        if score is not None and score < 0.5:
            verdicts.append(f"{joint}: moved opposite to the Leader more often than with it (agreement {score:.2f})")
        elif score is not None and score < 0.9:
            verdicts.append(f"{joint}: followed only {score:.0%} of the time; it is lagging or stalling, not reversed")
        if still / fps > 1.0:
            verdicts.append(f"{joint}: did not move at all for {still / fps:.1f} s while the Leader did")
        if abs(differences[0]) > 5:
            verdicts.append(f"{joint}: started {differences[0]:+.1f} degrees away from the Leader")
    say("")
    if verdicts:
        for verdict in verdicts:
            say(f"  {verdict}")
    else:
        say("  Every joint followed the Leader in the same direction, with no stalls.")
    return 0


def add_arm_arguments(parser, calibration=False):
    parser.add_argument("--leader-port", required=True)
    parser.add_argument("--follower-port", required=True)
    if calibration:
        parser.add_argument("--leader-id", required=True)
        parser.add_argument("--follower-id", required=True)
        parser.add_argument("--leader-calibration-dir", default="calibration/leader")
        parser.add_argument("--follower-calibration-dir", default="calibration/follower")
    else:
        parser.add_argument("--leader-calibration", default="calibration/leader/so101-leader.json")
        parser.add_argument("--follower-calibration", default="calibration/follower/so101-follower.json")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="mode", required=True)
    add_arm_arguments(commands.add_parser("registers", help="dump protection and calibration registers"))
    add_arm_arguments(commands.add_parser("compare", help="read both arms side by side, torque off"), calibration=True)
    report = commands.add_parser("report", help="grade a dataset recorded by lerobot-record")
    report.add_argument("--dataset", required=True, help="the --dataset.root given to lerobot-record")
    args = parser.parse_args(argv)
    try:
        if args.mode != "report" and version("lerobot") != LEROBOT_VERSION:
            raise RuntimeError(f"Use lerobot=={LEROBOT_VERSION} in the course environment.")
        path = start_log(args.mode, args)
        print(f"Log: {path}")
        return {"registers": run_registers, "compare": run_compare, "report": run_report}[args.mode](args)
    except KeyboardInterrupt:
        print("Interrupted. This program never switches motor power on or off.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"STOP: {exc}\nCut motor power before touching wiring. This program does not switch it off.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
