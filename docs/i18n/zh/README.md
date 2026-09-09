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

- **`Bringup/`**：`so101_bus_check.py` 读一条总线上的六台电机；`so101_calibrate.py` 记录每个关节的中位和活动范围。第 6 课用。
- **`IK/`**：`so101_cartesian_demo.py` 下载锁定模型，并用 LeRobot 自带的 IK 解一步笛卡尔位移；`so101_ee_teleop.py` 用方向键、`so101_visual_control.py` 用浏览器里的手柄驱动机械臂，两者都走 LeRobot 的末端流水线。第 8 课用。三个文件必须放在同一个文件夹：两个遥操作程序从第一个里取模型相关的函数。
- **`Teleop/`**：`so101_teleop_log.py` 打印两只臂的保护与校准寄存器、把两只臂对着读，并给 `lerobot-record` 录下的数据打分。遥操作本身仍然用官方的 `lerobot-teleoperate`。第 7 课用。
- **`tests/`**：47 条测试，用假硬件跑这些程序。
- **`Teleop/logs/`**（不进 git）：诊断程序每次运行一份调试日志，只留最近 20 份，`lerobot-record` 录下的数据也放这里。出问题时把最新那份发出来。
- **`tools/`**：`build_control_poses.py` 计算课程原理图里画的姿态。

课程正文印着这些路径，所以这里的文件不会移动或改名。以后的任务各占一个新文件夹，例如 `ACT-1-Pick`。

---

## 安装

Python 3.12 和 LeRobot 0.6.1，装在仓库根目录的虚拟环境里。命令顺序和第 3、7、8 课一样：

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

课程正文把激活这一步写成 `<ACTIVATE_ENV>`，两种环境管理器都能用。venv 代入 `source .venv/bin/activate`，Windows 11 代入 `.\.venv\Scripts\Activate.ps1`。conda 用 `conda create -n lerobot-s1 python=3.12` 建环境、`conda activate lerobot-s1` 激活，再跑同样几条 `pip install`。conda 没有在本课程实机验证过，真正要紧的是上面那些锁定版本。

`pip check` 输出 `No broken requirements found.` 就装好了。Linux 上默认的 torch 带 CUDA，整个环境约 6.6 GB；这一季后面训练的课会用到它。

机械臂模型下载一次即可，来自 TheRobotStudio 的锁定版本：

```bash
python IK/so101_cartesian_demo.py prepare --model-dir models/so101
```

终端显示 `Model ready: models/so101/so101_new_calib.urdf` 即完成。显示 `STOP` 就先解决它报的问题，再跑一次。

---

## 运行程序

所有命令都在仓库根目录执行。课程里 `models/so101`、`calibration/follower` 这类相对路径，只有在这里才成立。

```bash
source .venv/bin/activate
python IK/so101_cartesian_demo.py preview --model-dir models/so101 --delta-mm 0 0 2
```

这条命令算出让夹爪上移 2 mm 所需的关节角度，以 JSON 打印出来，`position_error_mm` 应小于 0.1；它不碰硬件。`Bringup/` 里的程序，以及 `IK/` 的两个遥操作程序会打开串口，上电与安全步骤写在第 6、7、8 课里，这里不重复。

改过程序，提交前先跑测试：

```bash
python tests/test_bringup.py                            # 只用假硬件，末行是 OK (skipped=7)
python tests/test_bringup.py --model-dir models/so101   # 加上数值 IK 检查，末行是 OK
```

两条都不打开串口。

有三个文件夹被 git 忽略。`.venv/` 太大，也只对这一台机器有效；`calibration/` 记的是某一只机械臂的中位和活动范围，换一只就不对；`models/` 由上面的 `prepare` 命令下载。

---

## 许可

课程程序保留所有权利。LeRobot 采用 Apache-2.0（[huggingface/lerobot](https://github.com/huggingface/lerobot)）；SO-101 模型同样采用 Apache-2.0（[TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100)）。
