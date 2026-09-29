# 一起手搓机器人 · 第一季 · SO-101

![Python 3.12](https://img.shields.io/badge/Python-3.12-3776ab?style=flat-square)
![LeRobot 0.6.1](https://img.shields.io/badge/LeRobot-0.6.1-ffcc4d?style=flat-square)
![License: all rights reserved](https://img.shields.io/badge/License-All_rights_reserved-555?style=flat-square)

<a href="../../../README.md"><img src="https://img.shields.io/badge/Language-English-2f81f7?style=flat-square" alt="English"></a>
<a href="README.md"><img src="https://img.shields.io/badge/%E8%AF%AD%E8%A8%80-%E7%AE%80%E4%BD%93%E4%B8%AD%E6%96%87-e67e22?style=flat-square" alt="简体中文"></a>

《一起手搓机器人》第一季的课程程序。这一季是一套视频课，从零搭起并训练 LeRobot SO-101 机械臂。

---

## 概览

第一季从打印、组装两只 SO-101 机械臂开始，到训练出一个按颜色分拣物品、再把东西递给人的策略为止。[偃师造物学习中心](https://www.yanshirobotics.com/learn/lets-build-robots/season-1)里每一课的下载按钮旁边，都印着那个文件在这个仓库里的路径。克隆一次，课程点到的每个文件都在它说的位置上。

---

## 仓库里有什么

- **`Bringup/`**：`so101_bus_check.py` 读一条总线上的六台电机；`so101_calibrate.py` 记录每个关节的中位和活动范围。第 6 课用。`so101_release_torque.py` 在某台电机带着错误状态位回包时也把六台的力矩全部关掉——那正是 LeRobot 自己的退出流程做不完的情况。
- **`Cameras/`**：`so101_camera_check.py` 列出能出图的摄像头、给出各自该用的稳定路径，再把它们实时显示在一个本地网页上，认出哪台装在哪、把画面转正、存进 `cameras.json`；`check` 会先测出两台相机单开与同开的速率。`so101_policy_view.py` 接着把同样的实时画面按 ACT、π₀、π₀.₅、SmolVLA 各自的预处理显示出来。第 9 课用。Linux 能读到设备列表、给出每台相机的稳定路径；macOS 与 Windows 没有这个列表，`list` 改成逐个试相机序号（`--max-index` 决定试到几号），认哪台是哪台靠看画面。
- **`Cartesian/`**：机械臂的笛卡尔控制，一个页面 <http://127.0.0.1:4602>。`cartesian_control.py` 显示机械臂，同一时间只让一种来源控制它：拖一个球（每个关节上还有一个可转的环）、配好对的手柄、或示教臂；加 `--model-only` 用模拟臂打开同一个页面。手柄第一次插上时在同一个页面里配对；页面上的 **Follower** 开关可以改为控制模拟臂而不是真臂。`fetch_model.py` 下载机械臂模型。其余文件是控制循环的各段（感知、目标、比较、求解、规划、执行），`Cartesian/README.md` 里有说明。第 8 课用。要在仓库根目录运行，它们按文件夹互相导入。
- **`Teleop/`**：`so101_teleop_log.py` 打印两只臂的保护与校准寄存器、把两只臂对着读，并给 `lerobot-record` 录下的数据打分。遥操作本身仍然用官方的 `lerobot-teleoperate`。第 7 课用。
- **`tests/`**：50 条测试，用假硬件跑 Bringup、Teleop、Cameras 的程序；Cartesian 的 38 条在 `Cartesian/test_cartesian.py`。
- **`Teleop/logs/`**（不进 git）：诊断程序每次运行一份调试日志，只留最近 20 份，`lerobot-record` 录下的数据也放这里。出问题时把最新那份发出来。

课程正文印着这些路径，所以这里的文件不会移动或改名。以后的任务各占一个新文件夹，例如 `ACT-1-Pick`。

---

## 安装

Python 3.12 和 LeRobot 0.6.1，装在仓库根目录的虚拟环境里。命令顺序和第 3、7、8 课一样。建环境和激活环境是唯一按系统不同的一步：

| 系统 | 建环境 | 激活 |
|---|---|---|
| Linux、macOS | `python3.12 -m venv .venv` | `source .venv/bin/activate` |
| Windows 11 | `py -3.12 -m venv .venv` | `.\.venv\Scripts\Activate.ps1` |

之后三个系统完全一样。每条命令都写成一行是有意的：PowerShell 续行用的是反引号而不是反斜杠，而反引号后面跟一个空格就会静默失效，所以这个仓库的长命令一律不折行。

```bash
python -m pip install --upgrade pip
python -m pip install "lerobot[feetech]==0.6.1"
python -m pip install "viser[urdf]==1.1.0" "lerobot[kinematics,feetech]==0.6.1" "cmeel-urdfdom==4.0.1" "cmeel-tinyxml2==10.0.0"
python -m pip install "lerobot[core_scripts,feetech]==0.6.1"
python -m pip check
```

第 9 课要读手柄。Linux 用内核自带的 joystick 接口，不用另外装东西；macOS 与 Windows 通过 pygame 读，LeRobot 把它作为一个可选项发布，所以在那两个系统上补一条：

```bash
python -m pip install "lerobot[gamepad,feetech]==0.6.1"
```

课程正文把激活这一步写成 `<ACTIVATE_ENV>`，两种环境管理器都能用。conda 用 `conda create -n lerobot-s1 python=3.12` 建环境、`conda activate lerobot-s1` 激活，再跑同样几条 `pip install`。conda 没有在本课程实机验证过，真正要紧的是上面那些锁定版本。

GitHub Actions 在每次推送后都把上面这套安装步骤和测试在 Ubuntu 24.04、macOS 与 Windows 11 上各跑一遍，每个系统用它自己的终端，所以三条路是验过的、不是推出来的；另有一个作业专门核对第 8 课那几个包是否仍然只发布本节说的那几种安装文件。`pip check` 输出 `No broken requirements found.` 就装好了。本课程在 Ubuntu 24.04 上实机验证过；macOS 与 Windows 的命令按上游文档和这些包实际发布的安装文件写，没有实机验证过。开始之前有两件与系统有关的事要知道：

- 第 8 课在 Windows 上装不了。它的求解器 `placo` 和它依赖的五个 `cmeel-*` 包只发布 Linux 与 macOS 的安装文件，任何版本都没有 Windows 的。那一课用 WSL 2 或云主机，或者跳过——这一季后面没有任何内容依赖它。macOS 可以，Intel 和 Apple Silicon 都行。
- Linux 上默认的 torch 带 CUDA，整个环境约 6.6 GB。Windows 上 `pip` 即使在装了 NVIDIA 显卡的机器上也会装成只有 CPU 的 torch（[lerobot#4093](https://github.com/huggingface/lerobot/issues/4093)），所以跑一下 `python -c "import torch; print(torch.cuda.is_available())"`，打印 `False` 就从 PyTorch 官方索引重装一次 torch。

机械臂模型下载一次即可，来自 TheRobotStudio 的锁定版本：

```bash
python Cartesian/fetch_model.py --model-dir models/so101
```

末行显示 `models/so101: 15 files present` 即完成。显示 `download failed` 就检查网络再跑一次，已下载的文件会保留。

---

## 运行程序

所有命令都在仓库根目录执行。课程里 `models/so101`、`calibration/follower` 这类相对路径，只有在这里才成立。先激活环境——Linux 与 macOS 是 `source .venv/bin/activate`，Windows 11 是 `.\.venv\Scripts\Activate.ps1`——然后：

```bash
python Cartesian/cartesian_control.py --model-dir models/so101 --model-only
```

这条命令用模拟臂打开控制页面，拖绿球，臂就跟着走；它不碰硬件。`Bringup/`、`Teleop/` 里的程序，以及带 `--port` 的 `Cartesian/` 程序会打开串口，上电与安全步骤写在第 6、7、8 课里，这里不重复。

改过程序，提交前先跑测试：

```bash
python tests/test_bringup.py                                 # Bringup、Teleop、Cameras；只用假硬件，末行是 OK
python Cartesian/test_cartesian.py --model-dir models/so101  # Cartesian 程序对着真实模型跑，末行是 OK
```

两条都不打开串口。

有几个路径被 git 忽略。`.venv/` 太大，也只对这一台机器有效；`calibration/` 记的是某一只机械臂的中位和活动范围，换一只就不对；`models/` 由上面的 `fetch_model.py` 命令下载；`cameras.json` 与 `Cartesian/gamepad_map.json` 记的是这一张工作台上的相机和这一只手柄。

---

## 许可

课程程序保留所有权利。LeRobot 采用 Apache-2.0（[huggingface/lerobot](https://github.com/huggingface/lerobot)）；SO-101 模型同样采用 Apache-2.0（[TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100)）。
