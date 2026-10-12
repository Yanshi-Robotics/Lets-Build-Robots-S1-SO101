"""Process-local tracing for the SO-101 diagnostic entry point; never a second bus reader."""
from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
import functools
import json
from pathlib import Path
import time
import traceback
import types

from so101_teleop_log import REGISTERS

REGISTERS = ("Model_Number", "Firmware_Major_Version", "Firmware_Minor_Version", *REGISTERS, "Lock")


def exception_info(exc):
    return {"type": type(exc).__name__, "message": str(exc),
            "traceback": "".join(traceback.format_exception(exc))}


class Trace:
    def __init__(self, directory, name, *, tail_packets=2000, snapshot_budget=4.0):
        self.file = (Path(directory) / f"{name}.jsonl").open("x", encoding="utf-8", buffering=1)
        self.tail = deque(maxlen=tail_packets)
        self.overwritten = 0
        self.snapshot_budget = snapshot_budget
        self.running = False
        self.reason = "startup_fault"
        self.buses = []
        self.patches = []

    def event(self, kind, **fields):
        item = {"time": datetime.now(timezone.utc).isoformat(), "monotonic_ns": time.monotonic_ns(),
                "event": kind, **fields}
        self.file.write(json.dumps(item, ensure_ascii=False, default=str) + "\n")
        return item

    def wire(self, direction, port, packet, comm):
        raw = list(packet or [])
        item = {"time": datetime.now(timezone.utc).isoformat(), "monotonic_ns": time.monotonic_ns(),
                "event": "packet", "direction": direction, "port": port.port_name,
                "hex": bytes(raw).hex(), "comm": comm,
                "motor_id": raw[2] if len(raw) > 2 else None,
                "instruction" if direction == "tx" else "packet_error": raw[4] if len(raw) > 4 else None}
        error = comm != 0 or (direction == "rx" and item["packet_error"] not in (None, 0))
        if self.running and not error:
            if len(self.tail) == self.tail.maxlen:
                self.overwritten += 1
            self.tail.append(item)
        else:
            if error:
                self.flush_tail("communication_or_device_error")
            self.file.write(json.dumps(item) + "\n")

    def flush_tail(self, cause):
        self.event("packet_tail", cause=cause, count=len(self.tail), overwritten=self.overwritten)
        while self.tail:
            self.file.write(json.dumps(self.tail.popleft()) + "\n")
        self.overwritten = 0

    def snapshot(self, bus, stage):
        """Bounded raw reads on the bus already owned by this process, with no writes/retries."""
        from lerobot.motors.motors_bus import get_address
        deadline = time.monotonic() + self.snapshot_budget
        rows = {}
        self.event("snapshot_begin", port=bus.port, stage=stage)
        if getattr(getattr(bus, "port_handler", None), "is_using", False):
            # Ctrl+C can interrupt an SDK transaction before it clears this flag. Preserve
            # that evidence; resetting the flag here would alter the behavior being diagnosed.
            return self.event("snapshot", port=bus.port, stage=stage, complete=False, rows=rows,
                              sdk_is_using=True, omitted="SDK transaction flag is busy; values are UNKNOWN")
        # Critical registers for ALL joints come first, even if the budget is exhausted later.
        critical = ("Torque_Enable", "Status", "Present_Position", "Goal_Position")
        registers = (*critical, *(name for name in REGISTERS if name not in critical))
        for register in registers:
            for name, motor in bus.motors.items():
                if time.monotonic() >= deadline:
                    return self.event("snapshot", port=bus.port, stage=stage, complete=False, rows=rows,
                                      omitted="snapshot time budget exhausted; omitted values are UNKNOWN")
                address, length = get_address(bus.model_ctrl_table, motor.model, register)
                try:
                    value, comm, error = bus._read(address, length, motor.id, raise_on_error=False)
                    reading = {"raw": value, "comm": comm, "packet_error": error,
                               "valid": bus._is_comm_success(comm)}
                except Exception as exc:
                    reading = {"valid": False, "exception": exception_info(exc)}
                rows.setdefault(name, {})[register] = reading
        return self.event("snapshot", port=bus.port, stage=stage, complete=True, rows=rows)

    def patch(self, obj, name, replacement):
        original = getattr(obj, name)
        self.patches.append((obj, name, original))
        setattr(obj, name, replacement)
        return original

    def install(self, *, gate_fault_cleanup=False):
        """Instrument installed libraries in memory; site-packages remain unchanged."""
        import serial
        import scservo_sdk.port_handler as sdk_port
        from scservo_sdk.protocol_packet_handler import protocol_packet_handler as PacketHandler
        from lerobot.motors.feetech import FeetechMotorsBus

        # SDK-local module proxy: request pySerial's advisory exclusive lock before opening.
        # fuser is checked separately; exclusivity does not block non-cooperating processes.
        proxy = types.SimpleNamespace(**vars(sdk_port.serial))
        def exclusive_serial(*args, **kwargs):
            kwargs["exclusive"] = True
            return serial.Serial(*args, **kwargs)
        proxy.Serial = exclusive_serial
        self.patch(sdk_port, "serial", proxy)

        original_tx = PacketHandler.txPacket
        @functools.wraps(original_tx)
        def tx(handler, port, packet):
            try:
                result = original_tx(handler, port, packet)
            except BaseException as exc:
                self.event("packet_tx_exception", port=port.port_name, intended_hex=bytes(packet).hex(),
                           exception=exception_info(exc))
                raise
            self.wire("tx", port, packet, result)
            return result
        self.patch(PacketHandler, "txPacket", tx)
        original_rx = PacketHandler.rxPacket
        @functools.wraps(original_rx)
        def rx(handler, port):
            try:
                packet, result = original_rx(handler, port)
            except BaseException as exc:
                self.event("packet_rx_exception", port=port.port_name, exception=exception_info(exc))
                raise
            self.wire("rx", port, packet, result)
            return packet, result
        self.patch(PacketHandler, "rxPacket", rx)

        original_connect = FeetechMotorsBus.connect
        @functools.wraps(original_connect)
        def connect(bus, *args, **kwargs):
            self.buses.append(bus)
            self.event("bus_connect_begin", port=bus.port, handshake=kwargs.get("handshake", True))
            try:
                result = original_connect(bus, *args, **kwargs)
            except BaseException as exc:
                self.event("bus_connect_error", port=bus.port, exception=exception_info(exc))
                raise
            if gate_fault_cleanup:
                self.snapshot(bus, "connected_before_configure")
            self.event("bus_connect_end", port=bus.port)
            return result
        self.patch(FeetechMotorsBus, "connect", connect)

        original_disconnect = FeetechMotorsBus.disconnect
        @functools.wraps(original_disconnect)
        def disconnect(bus, disable_torque=True):
            allowed = not gate_fault_cleanup or self.reason in ("ctrl_c", "normal")
            requested = disable_torque
            disable_torque = requested and allowed
            self.running = False
            self.flush_tail("disconnect")
            self.event("bus_disconnect_begin", port=bus.port, requested_disable=requested,
                       actual_disable=disable_torque, stop_reason=self.reason,
                       caller=[f"{frame.filename}:{frame.lineno}:{frame.name}"
                               for frame in traceback.extract_stack(limit=7)[:-1]])
            if gate_fault_cleanup:
                self.snapshot(bus, "before_official_disconnect")
            try:
                result = original_disconnect(bus, disable_torque=disable_torque)
            except BaseException as exc:
                self.event("bus_disconnect_error", port=bus.port, exception=exception_info(exc))
                raise
            self.event("bus_disconnect_end", port=bus.port, connected=bus.is_connected)
            return result
        self.patch(FeetechMotorsBus, "disconnect", disconnect)

        original_disable = FeetechMotorsBus.disable_torque
        @functools.wraps(original_disable)
        def disable(bus, *args, **kwargs):
            self.event("disable_torque_begin", port=bus.port)
            try:
                result = original_disable(bus, *args, **kwargs)
            except BaseException as exc:
                self.event("disable_torque_error", port=bus.port, exception=exception_info(exc))
                raise
            finally:
                if bus.is_connected and self.reason in ("normal", "ctrl_c"):
                    self.snapshot(bus, "after_official_disable_attempt_before_close")
            self.event("disable_torque_end", port=bus.port)
            return result
        self.patch(FeetechMotorsBus, "disable_torque", disable)

        original_enable = FeetechMotorsBus.enable_torque
        @functools.wraps(original_enable)
        def enable(bus, *args, **kwargs):
            import sys
            if gate_fault_cleanup and sys.exc_info()[0] is not None:
                # LeRobot's torque_disabled() re-enables in finally, even when configure fails.
                # Do not add an enable command while propagating a startup/configuration fault.
                self.event("enable_suppressed_during_exception", port=bus.port)
                return None
            return original_enable(bus, *args, **kwargs)
        self.patch(FeetechMotorsBus, "enable_torque", enable)

    def close(self, *, restore=True):
        self.running = False
        self.flush_tail("trace_close")
        if restore:
            for obj, name, original in reversed(self.patches):
                setattr(obj, name, original)
        self.file.close()
