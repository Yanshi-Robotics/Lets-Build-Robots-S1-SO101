# Teleoperation shutdown

The supervised entry point in v0.1.1 runs the installed LeRobot teleoperation loop and records its shutdown. Support both arms before starting and before requesting release: disabling torque can let an arm fall. Keep motor power accessible, and do not force a stiff joint.

## Environment

Use Linux, Python 3.12 and the repository's virtual environment. Install the pinned packages and the Linux `fuser` utility before connecting hardware:

```bash
python -m pip install "lerobot[core_scripts,feetech]==0.6.1" "feetech-servo-sdk==1.0.0" "pyserial==3.5" "psutil==7.2.2"
python Teleop/so101_shutdown_debug.py --version
```

The version command prints `0.1.1`. Identify each arm's stable USB path and use its existing calibration file. USB enumeration alone does not identify which physical arm is attached. Check the power adapter against the motor variant; follow the [LeRobot assembly instructions](https://huggingface.co/docs/lerobot/main/en/assemble_so101).

## Start and stop

Set `FOLLOWER_PORT` and `LEADER_PORT` to the two verified device paths. The IDs below require `calibration/follower/so101-follower.json` and `calibration/leader/so101-leader.json`; substitute your own IDs and directories together when they differ.

```bash
python -u Teleop/so101_shutdown_debug.py teleop -- \
  --robot.type=so101_follower \
  --robot.port="$FOLLOWER_PORT" \
  --robot.id=so101-follower \
  --robot.calibration_dir=calibration/follower \
  --teleop.type=so101_leader \
  --teleop.port="$LEADER_PORT" \
  --teleop.id=so101-leader \
  --teleop.calibration_dir=calibration/leader \
  --robot.cameras='{}' \
  --display_data=false \
  --fps=30
```

This command connects to real hardware, configures the motors and sends position targets. The diagnostic currently requires cameras and visualization to be disabled. It refuses recalibration rather than silently replacing an existing calibration. Run it directly, without wrapping it in `Bringup/so101_run.py`.

After supporting both arms, press Ctrl+C once. Wait for `Evidence saved:` and retain the printed directory. Normal loop completion also requests release. Startup failures, runtime failures, external termination and forced shutdown do not request additional release. A fault can leave some joints enabled and others disabled; it does not guarantee that the arm is supported.

## Recover manually

After the original control process has exited, support both arms and request release using that run's directory:

```bash
python -u Teleop/so101_shutdown_debug.py release --from-run "$RUN_DIR"
```

The supervisor checks port ownership before connecting. It writes only `Torque_Enable=0`, attempts each joint independently and verifies the reported state repeatedly. If a port is busy or ownership is unknown, resolve the confirmed owner before retrying; do not open a competing controller.

`Torque_Enable=0` describes the register. Check physical release gently while motor power remains connected, and record whether power changed. If the arm remains stiff, support it and cut motor power. Do not send speculative protection, calibration or position writes to unlock it.

```bash
python Teleop/so101_shutdown_debug.py note --run-dir "$RUN_DIR" --text "Operator observation, affected joints and any motor power change"
```

The note command only appends an observation; it does not connect to a motor bus. Notes remain operator reports rather than sensor measurements.

## Read the evidence

Runs are stored under `Teleop/logs/shutdown-debug/`, which Git ignores. `metadata.json` records versions, arguments, source hashes and calibration data. `child.jsonl` records the upstream loop and cleanup; `supervisor.jsonl` records process ownership and subsequent recovery. `terminal.log` retains output and exceptions; `summary.json` retains the stop reason and exit code. Operator notes are in `observations.jsonl`.

Successful loop packets are retained in a bounded tail; errors and cleanup packets are saved immediately. Phase snapshots use the bus already owned by the process, with a time budget and explicit unknown values. No additional periodic register reader runs during teleoperation. An interrupted SDK transaction can leave a snapshot incomplete; missing values are not OFF.

Logs contain local paths, USB identities and calibration data. Review them before sharing. The program prints physical release as unverified because it cannot measure holding force; an operator observation supplies that separate result.

## Compatibility

The entry point was physically checked on Linux with the pinned LeRobot environment on 2026-10-12. Other platforms, recording and Cartesian control require separate validation. `Bringup/so101_run.py` remains a legacy wrapper with different fault-release behavior. Use the supervised entry point for the stop policy described here.

The [changelog](../CHANGELOG.md) records the release and its known limits. Source references are the installed LeRobot 0.6.1 `lerobot_teleoperate.py`, `so_follower.py`, `motors_bus.py` and `feetech.py`, together with the repository's supervisor, trace and per-motor release modules.

## Changes behind the fix

The previous wrapper installed `SIG_IGN` before launching the controller. An ignored SIGINT can survive into a Python child, preventing the intended interrupt handler from running. The new supervisor installs a callable parent handler and launches the controller in a separate session. It forwards the first stop request once to that owned process group. Its cleanup timeout begins at the stop request, rather than limiting the entire teleoperation session.

Bus recovery waits for the complete child group to disappear and for both serial ports to be idle. A forced shutdown or an unknown owner blocks automatic recovery. This prevents the supervisor from deliberately taking over a bus while its controller is still alive; the ownership check is not a system-wide lock against another program starting later.

The release routine attempts every motor independently, records communication results and device error bytes, and performs three delayed verification rounds. One joint's exception cannot skip the remaining joints. Unknown replies stay unknown. Physical release still requires the operator's observation.

The supervisor also passes the stop reason into upstream cleanup. Normal completion and Ctrl+C allow release. Startup and runtime failures suppress additional release, including implicit destructor cleanup, and retain the original error code. The process-local adapter suppresses the configuration context manager's automatic re-enable when configuration raises. These adapters leave the installed package files unchanged. A failure during sequential enabling can still leave a mixture of enabled and disabled joints, requiring explicit recovery.

## Recovery evidence from 2026-10-12

The operator ran the diagnostic predecessor, `0.2.0-debug.1`, on Linux with the dependencies listed above. The release uses the same control, shutdown and tracing logic; its version string now comes from the repository's `VERSION` file.

1. The first startup could not discover any follower motor. The operator confirmed that follower motor power had not been connected.
2. On the next startup, all six follower IDs and models were discovered and calibration matched. Enabling motors 1 and 2 succeeded. Enabling motor 3 timed out after about 50 milliseconds with SDK communication result -6. A subsequent read of motor 3 succeeded and reported OFF. This localized the failure to the enabling stage; it did not identify why that reply was missing.
3. Standalone manual recovery began with follower torque states `[1, 1, 0, 0, 0, 0]`. It wrote only `Torque_Enable=0` to all twelve motors across the two arms. All three verification rounds reported OFF, and the operator confirmed physical release. A later idle bus check received all six follower positions in 30 consecutive rounds.
4. The next teleoperation loop ran for approximately 36.22 seconds. One Ctrl+C produced one forwarded SIGINT. The child group exited without forced termination, both ports were idle, and recovery verification finished with all joints OFF and exit code 0. The operator confirmed physical release after Ctrl+C.

In the successful Ctrl+C run, upstream cleanup had already disabled torque before the supervisor's additional release. That evidence confirms the complete path and does not attribute success to the additional write alone. The earlier motor-3 enable timeout and historical incidents in which valid OFF replies coincided with a stiff arm remain unexplained.

Release checks also passed 75 Bringup tests, 11 stop-safety tests, 13 diagnostic tests and 39 Cartesian regression tests. Three integrations used the installed LeRobot against emulated servo buses: normal completion, Ctrl+C with an upstream cleanup failure, and a runtime communication failure followed by explicit manual recovery. Those integrations verify software behavior; the two operator observations provide the separate physical evidence.
