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

- **`Bringup/`**: `so101_bus_check.py` reads all six motors on one bus; `so101_calibrate.py` records each joint's mid position and range. Used in Lesson 6.
- **`IK/`**: `so101_cartesian_demo.py` solves inverse kinematics for a gripper target; `so101_visual_control.py` shows the arm in a browser, plans a path to a dragged target, and in hardware mode reads the arm and executes the path after you arm it with ENABLE. Used in Lesson 7. The two files import each other and stay in the same folder.
- **`tests/`**: 43 tests that run the programs against fake hardware.
- **`tools/`**: `build_control_poses.py` computes the poses drawn in the lesson illustrations.

The lessons print these paths, so files here are not moved or renamed. Later tasks get their own folders, such as `ACT-1-Pick`.

---

## Installation

Python 3.12 and LeRobot 0.6.1, in a virtual environment at the repository root. This is the same sequence Lessons 3, 7, and 8 use.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install "lerobot[feetech]==0.6.1"
python -m pip install "viser[urdf]==1.1.0" "lerobot[kinematics,feetech]==0.6.1" \
  "cmeel-urdfdom==4.0.1" "cmeel-tinyxml2==10.0.0"
python -m pip install "lerobot[core_scripts,feetech]==0.6.1"
python -m pip check
```

`pip check` prints `No broken requirements found.` when the environment is complete. On Linux the default torch wheel includes CUDA and the environment takes about 6.6 GB; the training lessons later in the season use it.

Download the arm model once. It comes from TheRobotStudio's pinned revision:

```bash
python IK/so101_cartesian_demo.py prepare --model-dir models/so101
```

The command prints `Model ready: models/so101/so101_new_calib.urdf`. If it prints `STOP`, fix the reported problem and run it again.

---

## Running the programs

Run everything from the repository root. The lessons use relative paths such as `models/so101` and `calibration/follower`, and those resolve only from here.

```bash
source .venv/bin/activate
python IK/so101_cartesian_demo.py preview --model-dir models/so101 --delta-mm 0 0 2
```

This computes the joint angles that move the gripper 2 mm up and prints them as JSON; `position_error_mm` should be below 0.1. It touches no hardware. The programs in `Bringup/`, and the `jog` and `--readback` modes in `IK/`, do open a serial port; Lessons 6 and 7 cover the power and safety steps for those, and this file does not repeat them.

Before committing a change to a program, run the tests:

```bash
python tests/test_bringup.py                            # fake hardware only, ends with OK (skipped=3)
python tests/test_bringup.py --model-dir models/so101   # adds the numerical IK checks, ends with OK
```

Neither command opens a serial port.

Three folders are ignored by git. `.venv/` is large and specific to one machine. `calibration/` holds the mid positions and ranges of one particular arm and is wrong for any other. `models/` is downloaded by the `prepare` command above.

---

## License

The course programs are all rights reserved. LeRobot is Apache-2.0 ([huggingface/lerobot](https://github.com/huggingface/lerobot)); the SO-101 model is also Apache-2.0 ([TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100)).
