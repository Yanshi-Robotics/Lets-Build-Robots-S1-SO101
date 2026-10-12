#!/usr/bin/env python3
"""Record SO-101 teleop shutdown and standalone manual-release evidence (Linux, LeRobot 0.6.1).

Support both arms before starting and before Ctrl+C. Teleoperation can move real hardware.
Only Ctrl+C or normal loop completion requests automatic release. Fault exits preserve the
current torque state. Register OFF never establishes physical release.

    python Teleop/so101_shutdown_debug.py teleop -- <official teleoperate options>
    python Teleop/so101_shutdown_debug.py release --from-run <teleop log directory>
    python Teleop/so101_shutdown_debug.py note --run-dir <log directory> --text "physical observation"
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import faulthandler
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import selectors
import signal
import subprocess
import sys
import time
import traceback
import uuid

from _shutdown_trace import Trace, exception_info

REPO = Path(__file__).resolve().parents[1]
VERSION = (REPO / "VERSION").read_text(encoding="utf-8").strip()
LOG_ROOT = REPO / "Teleop/logs/shutdown-debug"
EXPECTED = {"lerobot": "0.6.1", "feetech-servo-sdk": "1.0.0", "pyserial": "3.5"}
POLL_SECONDS = 0.1  # Supervision responsiveness; this adds no bus reads during teleop.
DEFAULT_CLEANUP_SECONDS = 30.0
DEFAULT_TERMINATE_SECONDS = 5.0


def save_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_release():
    spec = importlib.util.spec_from_file_location("debug_release", REPO / "Bringup/so101_release_torque.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ownership(port):
    device = os.path.realpath(port)
    try:
        result = subprocess.run(["fuser", device], capture_output=True, text=True, timeout=3, check=False)
        return {"port": port, "device": device, "code": result.returncode,
                "stdout": result.stdout, "stderr": result.stderr,
                "idle": result.returncode == 1 and not result.stdout.strip() and not result.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"port": port, "device": device, "idle": False, "exception": exception_info(exc)}


def options_dict(arguments):
    """Accept the course's --name=value and --name value forms; reject ambiguous repeats."""
    values = {}
    index = 0
    while index < len(arguments):
        token = arguments[index]
        if not token.startswith("--"):
            raise ValueError(f"Expected an official --option, got {token!r}")
        if "=" in token:
            name, value = token.split("=", 1)
        else:
            name = token
            index += 1
            if index >= len(arguments) or arguments[index].startswith("--"):
                raise ValueError(f"Missing value for {name}")
            value = arguments[index]
        if name in values:
            raise ValueError(f"Repeated option: {name}")
        values[name] = value
        index += 1
    return values


def teleop_ports(arguments):
    values = options_dict(arguments)
    for key, expected in (("--robot.type", "so101_follower"), ("--teleop.type", "so101_leader")):
        if values.get(key) != expected:
            raise ValueError(f"This diagnostic requires {key}={expected}")
    # No visualizer/camera servers are started by this hardware diagnostic.
    if values.get("--display_data", "false").lower() != "false":
        raise ValueError("Use --display_data=false for this diagnostic")
    if json.loads(values.get("--robot.cameras", "{}")) != {}:
        raise ValueError("Use --robot.cameras={} for this diagnostic")
    ports = {role: values[f"--{prefix}.port"] for role, prefix in (("follower", "robot"), ("leader", "teleop"))}
    if len({os.path.realpath(port) for port in ports.values()}) != 2:
        raise ValueError("Leader and follower must have different serial devices")
    for prefix in ("robot", "teleop"):
        if not values.get(f"--{prefix}.id") or not values.get(f"--{prefix}.calibration_dir"):
            raise ValueError(f"Supply --{prefix}.id and --{prefix}.calibration_dir explicitly")
    return ports


def child_main(args, official_args):
    faulthandler.enable()
    faulthandler.register(signal.SIGUSR1, all_threads=True)
    trace = Trace(args.run_dir, "child", tail_packets=args.tail_packets, snapshot_budget=args.snapshot_budget)
    trace.install(gate_fault_cleanup=True)
    received = []
    in_cleanup = False

    def stop(signum, _frame):
        received.append({"signal": signum, "monotonic_ns": time.monotonic_ns()})
        if in_cleanup:
            return  # The supervisor forwards only once; do not interrupt per-joint cleanup.
        if signum == signal.SIGINT:
            trace.reason = "ctrl_c"
            raise KeyboardInterrupt
        trace.reason = "external_signal"
        raise SystemExit(128 + signum)

    previous = {sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    code = 0
    try:
        import lerobot.scripts.lerobot_teleoperate as upstream
        from lerobot.robots.so_follower.so_follower import SOFollower
        from lerobot.teleoperators.so_leader.so_leader import SOLeader
        from lerobot.motors.feetech import FeetechMotorsBus
        from lerobot.motors.motors_bus import SerialMotorsBus
        from scservo_sdk.protocol_packet_handler import protocol_packet_handler
        paths = {Path(inspect.getfile(obj)) for obj in
                 (upstream, SOFollower, SOLeader, FeetechMotorsBus, SerialMotorsBus, protocol_packet_handler)}
        trace.event("installed_source_hashes", sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                                                      for path in paths})

        def no_calibration(_arm):
            raise RuntimeError("Calibration mismatch or missing file: debug run refuses to recalibrate/write calibration")
        trace.patch(SOFollower, "calibrate", no_calibration)
        trace.patch(SOLeader, "calibrate", no_calibration)
        original_loop = upstream.teleop_loop
        def loop(*positional, **keyword):
            nonlocal in_cleanup
            for bus in trace.buses:
                trace.snapshot(bus, "configured_before_loop")
            trace.event("loop_begin")
            trace.running = True
            try:
                result = original_loop(*positional, **keyword)
                trace.reason = "normal"
                return result
            except KeyboardInterrupt:
                trace.reason = "ctrl_c"
                raise
            except BaseException as exc:
                if trace.reason != "external_signal":
                    trace.reason = "runtime_fault"
                trace.event("loop_error", exception=exception_info(exc))
                raise
            finally:
                in_cleanup = True
                trace.running = False
                trace.flush_tail("loop_end")
                trace.event("loop_end", reason=trace.reason, signals=received)
        trace.patch(upstream, "teleop_loop", loop)
        sys.argv = ["lerobot-teleoperate", *official_args]
        upstream.main()
    except KeyboardInterrupt:
        code = 130
    except BaseException as exc:
        code = exc.code if isinstance(exc, SystemExit) and isinstance(exc.code, int) else 1
        trace.event("child_error", exception=exception_info(exc))
        traceback.print_exc()
    finally:
        in_cleanup = True
        trace.running = False
        trace.flush_tail("child_finally")
        # Upstream's first disconnect exception can leave later arms untouched. Observe them
        # before closing descriptors, without any extra torque or configuration writes.
        for bus in trace.buses:
            if bus.is_connected:
                try:
                    trace.snapshot(bus, "residual_owned_bus_before_close")
                    bus.disconnect(disable_torque=False)
                except BaseException as exc:
                    code = code or 1
                    trace.event("residual_close_error", port=bus.port, exception=exception_info(exc))
        save_json(Path(args.run_dir) / "child-result.json",
                  {"reason": trace.reason, "exit_code": code, "signals": received})
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        # Keep the fault gate installed through interpreter teardown; a late destructor must
        # not regain the original default torque-disabling disconnect after a close failure.
        trace.close(restore=False)
    return code


def supervise(command, directory, trace, *, cleanup_seconds, terminate_seconds):
    """Own one process group, tee its output, and bound shutdown from the first stop request."""
    requests = []
    def stop(signum, _frame):
        requests.append({"signal": signum, "monotonic_ns": time.monotonic_ns()})
    previous = {sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    child = None
    forced = False
    deadline = None
    terminating = False
    forwarded = 0
    dumped = False
    selector = selectors.DefaultSelector()
    try:
        with (directory / "terminal.log").open("xb", buffering=0) as terminal:
            child = subprocess.Popen(command, stdin=None, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     start_new_session=True)
            trace.event("child_started", pid=child.pid, pgid=child.pid)
            os.set_blocking(child.stdout.fileno(), False)
            selector.register(child.stdout, selectors.EVENT_READ)
            while child.poll() is None or selector.get_map():
                for key, _ in selector.select(POLL_SECONDS):
                    data = os.read(key.fileobj.fileno(), 65536)
                    if data:
                        terminal.write(data)
                        sys.stdout.write(data.decode("utf-8", "replace"))
                        sys.stdout.flush()
                    else:
                        selector.unregister(key.fileobj)
                now = time.monotonic()
                if requests and not forwarded:
                    trace.event("stop_received", **requests[0])
                    if child.poll() is None:
                        os.killpg(child.pid, requests[0]["signal"])
                        trace.event("stop_forwarded", signal=requests[0]["signal"], pgid=child.pid)
                    forwarded = 1
                    deadline = now + cleanup_seconds
                if (deadline is not None and not dumped and child.poll() is None
                        and now >= deadline - min(1.0, cleanup_seconds / 2)):
                    # The dedicated child registers this diagnostic signal before importing LeRobot.
                    os.killpg(child.pid, signal.SIGUSR1)
                    dumped = True
                    trace.event("stack_dump_requested", pid=child.pid)
                if deadline is not None and now >= deadline and (child.poll() is None or selector.get_map()):
                    forced = True
                    sig = signal.SIGKILL if terminating else signal.SIGTERM
                    trace.event("cleanup_timeout", escalation=signal.Signals(sig).name)
                    try:
                        os.killpg(child.pid, sig)
                    except ProcessLookupError:
                        pass
                    if terminating:
                        break
                    terminating = True
                    deadline = now + terminate_seconds
                # A descendant keeping stdout open must not make the wrapper wait indefinitely.
                if child.poll() is not None and deadline is None:
                    deadline = now + cleanup_seconds
            code = child.wait(timeout=terminate_seconds)
            try:
                os.killpg(child.pid, 0)
                group_gone = False
            except ProcessLookupError:
                group_gone = True
            import psutil
            members = []
            for process in psutil.process_iter(["pid", "ppid", "name", "status"]):
                try:
                    if os.getpgid(process.pid) == child.pid:
                        members.append(process.info)
                except (ProcessLookupError, PermissionError):
                    continue
            trace.event("child_exited", code=code, group_gone=group_gone, forced=forced,
                       requests=requests, remaining_group=members)
            return {"code": code, "group_gone": group_gone, "forced": forced,
                    "requests": requests, "remaining_group": members}
    finally:
        selector.close()
        if child is not None:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=terminate_seconds)
            child.stdout.close()
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def collect_ports(ports, trace, *, release_requested):
    """Once the child group is gone, read each idle port and optionally call the existing release."""
    release = load_release()
    from lerobot.motors.motors_bus import get_address
    results = {}
    for role, port in ports.items():
        owner = ownership(port)
        trace.event("port_ownership", role=role, stage="after_child", **owner)
        if not owner["idle"]:
            results[role] = {"unresolved": ["ownership_unknown"], "release_attempted": False}
            continue
        bus = None
        try:
            bus = release.open_bus(port, attempts=1, output=lambda line: trace.event("release_output", text=line))
            before = trace.snapshot(bus, "before_additional_release" if release_requested else "fault_read_only")
            unresolved = None
            if release_requested:
                trace.event("additional_release_begin", role=role, port=port)
                unresolved = release.release_bus(bus, get_address,
                    output=lambda line: trace.event("release_output", text=line))
                after = trace.snapshot(bus, "after_additional_release")
                trace.event("additional_release_end", role=role, unresolved=unresolved)
            results[role] = {"unresolved": unresolved, "release_attempted": release_requested,
                             "physical_release": "UNVERIFIED", "before": before,
                             "after": after if release_requested else None}
        except Exception as exc:
            trace.event("post_capture_error", role=role, exception=exception_info(exc))
            results[role] = {"unresolved": ["unknown"], "exception": exception_info(exc)}
        finally:
            if bus is not None and bus.is_connected:
                try:
                    bus.disconnect(disable_torque=False)
                except Exception as exc:
                    trace.event("post_close_error", role=role, exception=exception_info(exc))
                    results[role]["close_error"] = str(exc)
    return results


def run_main(args, official_args):
    if sys.platform != "linux":
        raise RuntimeError("This diagnostic currently supports Linux only")
    versions = {package: importlib.metadata.version(package) for package in EXPECTED}
    if versions != EXPECTED:
        raise RuntimeError(f"Version mismatch: expected {EXPECTED}, found {versions}")
    parent_run = None
    if args.mode == "teleop":
        ports = teleop_ports(official_args)
    else:
        parent_run = str(Path(args.from_run).resolve())
        ports = json.loads((Path(parent_run) / "metadata.json").read_text())["ports"]
        if set(ports) != {"leader", "follower"} or len({os.path.realpath(p) for p in ports.values()}) != 2:
            raise ValueError("Source run must identify two distinct SO-101 arm ports")
    args.output_root.mkdir(parents=True, exist_ok=True)
    directory = args.output_root / f"{datetime.now():%Y%m%d-%H%M%S}-{args.mode}-{uuid.uuid4().hex[:8]}"
    directory.mkdir(mode=0o700)
    trace = Trace(directory, "supervisor", tail_packets=args.tail_packets, snapshot_budget=args.snapshot_budget)
    trace.install()
    metadata = {"debug_version": VERSION, "mode": args.mode, "parent_run": parent_run,
                "started": datetime.now(timezone.utc).isoformat(), "versions": versions,
                "ports": ports, "official_args": official_args, "python": sys.executable,
                "source_sha256": {str(path.relative_to(REPO)): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in (Path(__file__), Path(__file__).with_name("_shutdown_trace.py"),
                                 REPO / "Bringup/so101_release_torque.py")},
                "settings": {name: getattr(args, name) for name in
                    ("snapshot_budget", "tail_packets", "cleanup_seconds", "terminate_seconds")}}
    if args.mode == "teleop":
        values = options_dict(official_args)
        metadata["calibration_files"] = {}
        for role, prefix in (("leader", "teleop"), ("follower", "robot")):
            path = Path(values[f"--{prefix}.calibration_dir"]) / (values[f"--{prefix}.id"] + ".json")
            metadata["calibration_files"][role] = {"path": str(path.resolve()), "exists": path.is_file()}
            if path.is_file():
                metadata["calibration_files"][role].update(
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest(), content=json.loads(path.read_text()))
    save_json(directory / "metadata.json", metadata)
    print(f"SO-101 debug {VERSION}: {directory}", flush=True)
    print("Support both arms before Ctrl+C. Register OFF is not proof of physical release.", flush=True)
    summary = {"physical_release": "UNVERIFIED", "auto_release": False, "reason": "preflight_failure"}
    code = 2
    try:
        for role, port in ports.items():
            owner = ownership(port)
            trace.event("port_ownership", role=role, stage="preflight", **owner)
            if not owner["idle"]:
                raise RuntimeError(f"Serial ownership not clear: {port}; no bus opened")
        if args.mode == "teleop":
            command = [sys.executable, "-u", str(Path(__file__).resolve()), "_child", "--run-dir", str(directory),
                       "--snapshot-budget", str(args.snapshot_budget), "--tail-packets", str(args.tail_packets),
                       "--", *official_args]
            outcome = supervise(command, directory, trace, cleanup_seconds=args.cleanup_seconds,
                                terminate_seconds=args.terminate_seconds)
            result_file = directory / "child-result.json"
            child = json.loads(result_file.read_text()) if result_file.exists() else {"reason": "unknown"}
            release_requested = (child["reason"] in ("normal", "ctrl_c")
                                 and outcome["group_gone"] and not outcome["forced"]
                                 and all(request["signal"] == signal.SIGINT for request in outcome["requests"]))
            summary.update(reason=child["reason"], child=outcome, auto_release=release_requested)
            code = outcome["code"] if outcome["code"] >= 0 else 128 - outcome["code"]
            summary["ownership_after_child"] = {role: ownership(port) for role, port in ports.items()}
            trace.event("ownership_after_child", ports=summary["ownership_after_child"])
            if not outcome["group_gone"] or outcome["forced"]:
                raise RuntimeError("Clean child-group exit not confirmed; no diagnostic serial open or release")
        else:
            release_requested = True  # Explicit manual release requested by the operator.
            summary.update(reason="manual_release", auto_release=False)
            code = 0
        summary["arms"] = collect_ports(ports, trace, release_requested=release_requested)
        if any(value.get("unresolved") or value.get("close_error") for value in summary["arms"].values()):
            code = code or 1
    except KeyboardInterrupt:
        code = 130
        summary["interrupted"] = True
        trace.event("capture_interrupted")
    except Exception as exc:
        trace.event("supervisor_error", exception=exception_info(exc))
        summary["exception"] = exception_info(exc)
        print(f"STOP: {exc}", file=sys.stderr, flush=True)
        code = code or 2
    finally:
        summary["exit_code"] = code
        summary["finished"] = datetime.now(timezone.utc).isoformat()
        save_json(directory / "summary.json", summary)
        trace.event("finished", summary=summary)
        trace.close()
    print(f"\nEvidence saved: {directory}\nStop reason: {summary['reason']}; exit code: {code}", flush=True)
    for role, result in summary.get("arms", {}).items():
        if result.get("release_attempted") and result.get("unresolved") == []:
            message = "Torque_Enable=0 in all release verification samples"
        elif result.get("unresolved"):
            message = f"release/communication unresolved: {result['unresolved']}"
        else:
            message = "no release requested; register snapshot saved"
        print(f"{role}: {message}", flush=True)
    print("Physical release UNVERIFIED. Record stiffness, affected joints, and any DC power change.", flush=True)
    return code


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    split = argv.index("--") if "--" in argv else len(argv)
    own, official_args = argv[:split], argv[split + 1:]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", action="version", version=VERSION)
    sub = parser.add_subparsers(dest="mode", required=True)
    for mode in ("teleop", "release", "_child"):
        cmd = sub.add_parser(mode)
        cmd.add_argument("--snapshot-budget", type=float, default=4.0,
                         help="seconds per register snapshot, plus at most one SDK transaction")
        cmd.add_argument("--tail-packets", type=int, default=2000,
                         help="successful teleop packets retained in RAM; errors and cleanup are saved immediately")
        if mode == "_child":
            cmd.add_argument("--run-dir", type=Path, required=True)
        else:
            cmd.add_argument("--output-root", type=Path, default=LOG_ROOT)
            cmd.add_argument("--cleanup-seconds", type=float, default=DEFAULT_CLEANUP_SECONDS)
            cmd.add_argument("--terminate-seconds", type=float, default=DEFAULT_TERMINATE_SECONDS)
        if mode == "release":
            cmd.add_argument("--from-run", type=Path, required=True)
    note = sub.add_parser("note")
    note.add_argument("--run-dir", type=Path, required=True)
    note.add_argument("--text", required=True)
    args = parser.parse_args(own)
    if args.mode == "note":
        if not (args.run_dir / "metadata.json").is_file():
            parser.error("Not a diagnostic run directory")
        with (args.run_dir / "observations.jsonl").open("a", encoding="utf-8") as file:
            file.write(json.dumps({"time": datetime.now(timezone.utc).isoformat(), "text": args.text,
                                   "source": "operator_observation"}, ensure_ascii=False) + "\n")
        print(f"Observation saved: {args.run_dir}")
        return 0
    for name in ("snapshot_budget", "cleanup_seconds", "terminate_seconds"):
        value = getattr(args, name, 1)
        if not math.isfinite(value) or value <= 0:
            parser.error(f"{name} must be finite and positive")
    if args.tail_packets < 1:
        parser.error("tail_packets must be positive")
    if args.mode == "_child":
        return child_main(args, official_args)
    if args.mode == "release" and official_args:
        parser.error("release uses ports from --from-run; do not append teleop options")
    try:
        return run_main(args, official_args)
    except Exception as exc:
        print(f"STOP: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
