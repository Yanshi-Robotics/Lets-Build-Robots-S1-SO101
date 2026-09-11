# Cartesian control for the SO-101

Open a page, drag a ball, the arm follows. Each joint also carries a ring you can turn
directly, the way MoveIt's interactive markers work in RViz. The same program drives the
simulated arm or the real one.

```sh
# simulated arm, no hardware
.venv/bin/python Cartesian/cartesian_control.py --model-dir models/so101 --model-only

# real arm (torque stays off until you press "Hold and follow" in the page)
.venv/bin/python Cartesian/cartesian_control.py --model-dir models/so101 \
    --port /dev/ttyACM0 --robot-id so101-follower --calibration-dir calibration/follower
```

Then open <http://127.0.0.1:4602>. The page only listens on localhost.

`models/so101/` must hold `so101_new_calib.urdf` and its `assets/` from
[TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100). The Python
environment is the course `.venv` (placo 0.9.15, viser 1.1.0, lerobot 0.6.1).

## How it works

Every control tick runs six stages in a fixed order. Each stage is one file, and each
stage's input and output go to the run log, so a misbehaving session can be traced to the
stage that went wrong.

```
 1 sense          2 target          3 compare        4 solve            5 plan              6 execute
 read the six  -> what the user  -> target minus  -> IK: target pose -> walk the command  -> write the goal
 servo angles     asked for         where we are     -> joint goal       toward the goal     to the servos
 (so101_arm)      (target, viewer)  (compare)        (solver)            (planner)           (so101_arm)
```

1. **Sense.** Read the present position of all six servos over the bus. LeRobot reports
   degrees with zero at the middle of the calibrated range, which is also the URDF's zero.
   In model mode the "measurement" is simply the last command.
2. **Target.** The ball's position, the pitch and roll sliders and the gripper slider
   together make one `Target`: x, y, z, pitch, roll, gripper. A joint ring is another way
   to produce a target: turning it sets that joint's goal directly and the ball jumps to
   where the tool then is. Any future input device (keyboard, gamepad, hand tracking)
   only has to produce a `Target`.
3. **Compare.** Forward kinematics turns the current joint angles into a tool pose; the
   difference to the target is the error in millimetres and degrees. The ball turns red
   when the solved pose cannot get within 5 mm of it.
4. **Solve.** Only when the target changed. The target is turned into a 4x4 pose and
   placo's QP solver iterates from the current command until the tool is within 0.5 mm
   and 0.1 degrees, or gives up after 200 iterations (an unreachable target). The result
   is a joint goal.
5. **Plan.** Each tick the command moves toward the goal by at most `max speed x dt`
   per joint. On the real arm the command may also not lead the measured position by
   more than 15 degrees: an STS3215 turns position error into torque, and a command that
   runs away from a blocked arm is a command to push harder.
6. **Execute.** Write the command as `Goal_Position`.

### What the SO-101 itself dictates

Measured on the URDF with placo, not assumed:

* It has **five** arm joints plus a gripper: `shoulder_pan` about the vertical axis, then
  `shoulder_lift`, `elbow_flex`, `wrist_flex` about three **parallel** horizontal axes, then
  `wrist_roll` about the tool's own axis. Setting the three pitch joints to (-0.5, +1.0,
  -0.5) rad leaves the tool orientation exactly as at zero, so the tool's pitch is the sum
  of the three.
* The tool's **yaw is not free**: it equals the pan angle (pan = 0.7 rad gives tool yaw
  = -0.7 rad, to four decimals). A six-degree-of-freedom gizmo would let you ask for
  something the arm cannot do, so the ball is translation-only and the sliders give
  pitch and roll. The program fills in yaw from the arm's own pan angle before every
  solve, so the solver is always handed a pose this arm can reach exactly.
* The tool point sits **7.9 mm off the roll axis**: turning only `wrist_roll` moves it on a
  small circle. Roll therefore goes through the solver together with position so the other
  joints compensate, instead of being written straight to the joint.

Definitions of the two angles: **pitch** = angle of the tool's approach axis below
horizontal (0 = pointing forward, +90 = pointing straight down). **roll** = `wrist_roll`
joint angle (0 = the orientation the URDF gives at `wrist_roll = 0`; the tool frame has a
built-in 2.8 degree tilt which is included in that zero).

## Files

| File | Stage | What it owns |
|---|---|---|
| `target.py` | 2 | `Target`, the thread-safe `TargetBox` (with a version counter) and `CommandBox` (rings, buttons) |
| `so101_model.py` | — | URDF loading, FK, `compose`/`decompose` between (yaw, pitch, roll) and a rotation matrix, joint limits |
| `compare.py` | 3 | error between a joint vector and a target |
| `solver.py` | 4 | placo QP, iterated to convergence |
| `planner.py` | 5 | speed limit and lead clamp, pure functions |
| `so101_arm.py` | 1, 6 | `Arm` (LeRobot `SO101Follower`), `FakeArm`, degree/radian conversion |
| `viewer.py` | 2 | the viser page: two arms, ball, sliders, rings, buttons, status |
| `runlog.py` | — | one directory per run under `logs/` |
| `cartesian_control.py` | loop | arguments, wiring, the control thread |
| `test_cartesian.py` | — | model-level tests, no hardware |

Only `so101_model.py` and `so101_arm.py` know which robot this is. The control thread is
the only thread that touches the model or the serial bus; viser callbacks write into the
boxes and nothing else.

## The page

* **Solid arm**: measured position (model mode: the command).
* **Orange ghost** (live mode): the command being sent this tick. The gap between ghost
  and solid arm is the lead, i.e. the force being applied.
* **Ball with arrows**: target position. Green = reachable, red = the solver could not get
  within 5 mm. Small axes on the ball show the target tool orientation.
* **Rings**: one per arm joint, on the joint axis. Drag to turn that joint; the ball follows.
* **Sliders**: pitch, roll (degrees), gripper (percent).
* **Buttons** (live only): *Hold and follow* parks the goal at the present position, turns
  torque on and starts following the ball. *Stop* stops sending; torque stays on, the
  arm holds. *Release torque* switches torque off.

## Real arm: the order of things

1. Start with `--port ...`. The bus opens, calibration is checked, **torque stays off**.
   The page shows the arm as the servos report it; the ball mirrors the arm. Move the arm
   by hand and the solid arm on screen must move the same way. If a joint moves the wrong
   way on screen, the degree-to-radian convention for that joint is wrong: fix it in
   `so101_arm.py`, nowhere else.
2. Press **Hold and follow**. Nothing moves: the goal was parked at the present position.
3. Drag the ball a few centimetres. The ghost leads, the solid arm follows, they meet.
4. Ctrl-C keeps torque on so the arm does not drop. Pass `--release-torque` to switch it
   off on exit.

Limits that apply on the real arm, all overridable on the command line: speed
(`--max-joint-speed`, 2 rad/s), lead (`--max-lead-deg`, 15), the ball's workspace
(`--bounds-min-m`, `--bounds-max-m`), and joint limits narrowed to the calibrated range.
The servo position gain is written at hold time (`--p-coefficient`, default 32 = factory;
LeRobot's 16 leaves the arm too soft to move on small commands).

## Reading the logs

Each run writes `logs/<date>_<time>_<model|live>/`; `logs/latest` points at the newest.

* `run.log`: arguments, URDF and calibration used, joint limits, every target change that
  the solver could not reach, every button, exceptions.
* `ticks.jsonl`: one line per tick with the numbers every stage saw:

```
t, tick, period_ms            timing of the loop
ms.read / solve / plan / write  time spent in stages 1, 4, 5, 6
target.{xyz, pitch_deg, roll_deg, gripper_pct, version, source}   stage 2 (source: ball / slider / ring / arm)
solve.{iterations, converged, error_mm, ...}   stage 4, only on ticks that solved
q_goal_deg, q_cmd_deg, q_meas_deg              stages 4, 5, 1
err_mm.{goal, cmd, meas}      tool distance to target for the goal, the command, the measurement
speed_limited, lead_limited   stage 5 clamps
following, torque, reachable
```

Where to look when the arm does not do what you asked:

| Symptom in `ticks.jsonl` | Stage to suspect |
|---|---|
| `target.version` never changes while you drag | 2: the page is not delivering (browser, websocket) |
| `solve.converged` false, `err_mm.goal` large | 4: target out of reach or outside joint limits |
| `q_cmd_deg` not moving toward `q_goal_deg` | 5: `lead_limited` true means the arm is blocked or torque is off |
| `q_cmd_deg` fine but `q_meas_deg` lags or stalls | 6 / hardware: bus, torque, servo gain, collision |
| `period_ms` far above 33 (live) | the loop is starved; look at `ms.*` |

Quick extracts:

```sh
jq -c '[.t, .target.source, .err_mm.goal, .err_mm.meas, .lead_limited]' Cartesian/logs/latest/ticks.jsonl | tail
jq -c 'select(.solve) | .solve' Cartesian/logs/latest/ticks.jsonl
```

## Tests

```sh
.venv/bin/python Cartesian/test_cartesian.py --model-dir models/so101
```

Covers FK against known poses, the parallel-axis and yaw facts above, `compose`/`decompose`
round trips (including the straight-down pose), solver convergence on reachable targets
and its residual on unreachable ones, the planner's two clamps, the ring angle math, the
calibration-to-limits conversion, the run log, and the full control loop with a fake arm
(target walk, ring, hold/stop).
