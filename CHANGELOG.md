# Changelog

## [0.1.2] - 2026-10-12

### Fixed

- Limit the legacy wrapper's two real process-group tests to POSIX systems. The v0.1.1 Windows CI run incorrectly expected that unsupported wrapper to launch a controller.
- Add a regression that verifies an unsupported platform starts no child, attempts no release and reports the POSIX requirement with a failure code.

The control, signal, release and tracing implementation is unchanged from v0.1.1. This patch corrects the test contract; it does not add Windows shutdown support. The operator-confirmed Linux recovery results and known limits below still apply. The published v0.1.1 tag is preserved.

## [0.1.1] - 2026-10-12

### Fixed

- Add a Linux teleoperation supervisor that forwards the first Ctrl+C once to its own child process group. The cleanup deadline starts at the stop request, allowing sessions to run longer than the cleanup timeout.
- Confirm that the child process group has exited and both serial ports are idle before opening a bus for additional torque release. Refuse release when ownership is unknown or shutdown required forced termination.
- Release each motor independently and check all six torque registers in three verification rounds. A failed motor does not prevent an attempt on the remaining motors.
- Distinguish intentional Ctrl+C and normal completion from startup, runtime and external-signal failures. Fault cleanup preserves the existing torque state, including through upstream destructors, and keeps the original failure code.
- Suppress upstream configuration's automatic torque re-enable while an exception is propagating. A partially enabled arm remains a fault requiring explicit recovery.

### Added

- `Teleop/so101_shutdown_debug.py` provides supervised teleoperation, standalone manual release using saved port identities, and operator observations.
- `Teleop/_shutdown_trace.py` records serial packets, communication errors, register snapshots, cleanup callers, process ownership and installed source hashes without changing the installed LeRobot files.
- Offline regression tests and real LeRobot integration tests against emulated servo buses cover normal completion, terminal Ctrl+C, cleanup failures, runtime communication failures and manual recovery.
- [Shutdown instructions](docs/teleop-shutdown.md) describe the supported entry point and recovery procedure.

### Validation and scope

On 2026-10-12, the operator confirmed physical release after standalone manual recovery and after a successful teleoperation session stopped with Ctrl+C. The manual recovery changed the follower's remaining enabled motors 1 and 2 to OFF. A separate read-only check received all six follower positions in 30 rounds. The Ctrl+C run forwarded one SIGINT, exited without forced termination and finished with all torque verification samples OFF.

In that successful Ctrl+C run, upstream cleanup had already disabled torque before the supervisor's additional release. The result verifies the complete path; it does not establish which change alone explains earlier powered-but-stiff incidents. An earlier startup timeout while enabling follower motor 3 remains unexplained.

The supervised entry point supports Linux with the pinned dependencies in the instructions. Windows, macOS, recording and Cartesian shutdown are outside this release's physical acceptance. The legacy `Bringup/so101_run.py` is not the validated entry point: it can still perform additional release after a child fault. Full safety work planned for v0.2.0 remains separate.

[0.1.2]: https://github.com/Yanshi-Robotics/Lets-Build-Robots-S1-SO101/releases/tag/v0.1.2
[0.1.1]: https://github.com/Yanshi-Robotics/Lets-Build-Robots-S1-SO101/releases/tag/v0.1.1
