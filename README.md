# Let's Build Robots · Season 1 · SO-101

![Python 3.12](https://img.shields.io/badge/Python-3.12-3776ab?style=flat-square)
![LeRobot 0.6.1](https://img.shields.io/badge/LeRobot-0.6.1-ffcc4d?style=flat-square)
![License: all rights reserved](https://img.shields.io/badge/License-All_rights_reserved-555?style=flat-square)

<a href="README.md"><img src="https://img.shields.io/badge/Language-English-2f81f7?style=flat-square" alt="English"></a>
<a href="docs/i18n/zh/README.md"><img src="https://img.shields.io/badge/%E8%AF%AD%E8%A8%80-%E7%AE%80%E4%BD%93%E4%B8%AD%E6%96%87-e67e22?style=flat-square" alt="简体中文"></a>

Course programs for Season 1 of *Let's Build Robots*, a video course that builds and trains the LeRobot SO-101 arm.

---

## Overview

Season 1 starts with printing and assembling two SO-101 arms and ends with a trained policy that sorts objects by color and hands them to a person. Each lesson at the [Yanshi Robotics Learning Center](https://www.yanshirobotics.com/learn/lets-build-robots/season-1) that uses a program shows a download button, and next to the button it prints the file's path in this repository. Clone the repository once and every file is where the lesson says it is.

---

## What's in the repository

- **`Bringup/`**: `so101_bus_check.py` reads all six motors on one bus; `so101_calibrate.py` records each joint's mid position and range. Used in Lesson 6. `so101_release_torque.py` switches torque off on all six even when one answers with an error bit, which is the case LeRobot's own shutdown cannot finish.
- **`Cameras/`**: `so101_camera_check.py` lists the cameras that deliver frames, names the stable path for each, and serves them live on a local page where a view is matched to a mount, turned the right way up and saved to `cameras.json`; `check` measures first what the two configured cameras deliver alone and together. `so101_policy_view.py` then shows the same live frames as ACT, π₀, π₀.₅ and SmolVLA each reshape them. Used in Lesson 10. Linux reads the device list and reports a stable path per camera; macOS and Windows have no such list, so `list` tries camera numbers instead (`--max-index` sets how many) and the pictures are what tells one camera from another.
- **`Cartesian/`**: Cartesian control of the arm, one page at <http://127.0.0.1:4602>. `cartesian_control.py` shows the arm and lets one source at a time move it: drag a ball (and turn a ring on each joint), a paired gamepad, or a leader arm; `--model-only` runs the same page with a simulated arm. A gamepad is paired on the same page the first time it is plugged in, and a **Follower** switch drives the simulated arm instead of the real one. `fetch_model.py` downloads the arm model. The other files are the stages of the control loop (sense, target, compare, solve, plan, execute) and are explained in `Cartesian/README.md`. Used in Lesson 8. Run them from the repository root; they import each other by folder.
- **`Teleop/`**: `so101_teleop_log.py` dumps both arms' protection and calibration registers, checks the two arms against each other, and grades a recording made by `lerobot-record`. Teleoperation itself stays with the official `lerobot-teleoperate`. Used in Lesson 7.
- **`tests/`**: 50 tests that run the Bringup, Teleop and Cameras programs against fake hardware; `Cartesian/test_cartesian.py` holds the 38 for the Cartesian programs.
- **`Teleop/logs/`** (git-ignored): one debug log per diagnostics run, the last 20 kept, plus anything `lerobot-record` writes there. Send the newest one when something goes wrong.

The lessons print these paths, so files here are not moved or renamed. Later tasks get their own folders, such as `ACT-1-Pick`.

---

## Installation

Python 3.12 and LeRobot 0.6.1, in a virtual environment at the repository root. This is the same sequence Lessons 3, 7, and 8 use. Creating and activating that environment is the one step that differs per operating system:

| system | create | activate |
|---|---|---|
| Linux, macOS | `python3.12 -m venv .venv` | `source .venv/bin/activate` |
| Windows 11 | `py -3.12 -m venv .venv` | `.\.venv\Scripts\Activate.ps1` |

Everything after that is the same on all three. Each line is one line on purpose: PowerShell continues a command with a backtick rather than a backslash, and a backtick followed by a space silently stops continuing, so this repository writes long commands unbroken.

```bash
python -m pip install --upgrade pip
python -m pip install "lerobot[feetech]==0.6.1"
python -m pip install "viser[urdf]==1.1.0" "lerobot[kinematics,feetech]==0.6.1" "cmeel-urdfdom==4.0.1" "cmeel-tinyxml2==10.0.0"
python -m pip install "lerobot[core_scripts,feetech]==0.6.1"
python -m pip check
```

Lesson 9 reads a gamepad. Linux uses the kernel's joystick interface and needs nothing extra; macOS and Windows read it through pygame, which LeRobot ships as an extra, so add it there:

```bash
python -m pip install "lerobot[gamepad,feetech]==0.6.1"
```

The lessons write the activation step as `<ACTIVATE_ENV>` so either environment manager fits. With conda, run `conda create -n lerobot-s1 python=3.12` and `conda activate lerobot-s1`, then the same `pip install` lines. Conda is not hardware-validated for this course; the pinned versions are what matters.

GitHub Actions runs this install sequence and the tests on Ubuntu 24.04, macOS and Windows 11 after every push, each in that system's own shell, so the three paths are checked rather than argued; a separate job asserts that the wheels behind Lesson 8 are still published where this section says they are. `pip check` prints `No broken requirements found.` when the environment is complete. The course is hardware-validated on Ubuntu 24.04; the macOS and Windows paths follow the upstream documentation and the published wheels, and are not hardware-validated. Two platform facts are worth knowing before starting:

- Lesson 8 cannot be installed on Windows. Its solver `placo` and the five `cmeel-*` packages behind it publish wheels for Linux and macOS only — no Windows wheel exists at any version. Use WSL 2 or a cloud machine for that lesson, or skip it; nothing later in the season depends on it. macOS is fine, Intel and Apple Silicon both.
- On Linux the default torch wheel includes CUDA and the environment takes about 6.6 GB. On Windows `pip` installs a CPU-only torch even on a machine with an NVIDIA card ([lerobot#4093](https://github.com/huggingface/lerobot/issues/4093)), so run `python -c "import torch; print(torch.cuda.is_available())"` and reinstall torch from the PyTorch index if it prints `False`.

Download the arm model once. It comes from TheRobotStudio's pinned revision:

```bash
python Cartesian/fetch_model.py --model-dir models/so101
```

The command ends with `models/so101: 15 files present`. If it prints `download failed`, check the network and run it again; files already downloaded are kept.

---

## Running the programs

Run everything from the repository root. The lessons use relative paths such as `models/so101` and `calibration/follower`, and those resolve only from here. Activate the environment first — `source .venv/bin/activate` on Linux and macOS, `.\.venv\Scripts\Activate.ps1` on Windows 11 — then:

```bash
python Cartesian/cartesian_control.py --model-dir models/so101 --model-only
```

This opens the control page with a simulated arm; drag the green ball and the arm follows. It touches no hardware. The programs in `Bringup/`, `Teleop/` and, with `--port`, `Cartesian/` do open a serial port; Lessons 6, 7 and 8 cover the power and safety steps for those, and this file does not repeat them.

Before committing a change to a program, run the tests:

```bash
python tests/test_bringup.py                                 # Bringup, Teleop, Cameras; fake hardware only, ends with OK
python Cartesian/test_cartesian.py --model-dir models/so101  # the Cartesian programs against the real model, ends with OK
```

Neither command opens a serial port.

Several paths are ignored by git. `.venv/` is large and specific to one machine. `calibration/` holds the mid positions and ranges of one particular arm and is wrong for any other. `models/` is downloaded by the `fetch_model.py` command above. `cameras.json` and `Cartesian/gamepad_map.json` describe one workbench's cameras and one pad.

---

## License

The course programs are all rights reserved. LeRobot is Apache-2.0 ([huggingface/lerobot](https://github.com/huggingface/lerobot)); the SO-101 model is also Apache-2.0 ([TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100)).
