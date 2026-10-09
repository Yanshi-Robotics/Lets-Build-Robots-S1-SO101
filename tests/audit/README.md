# tests/audit

这里放的是**给审计用的可复跑证据**：不碰硬件、可反复运行、结论从假总线的寄存器读回来。

## 为什么单独一个目录

`tests/test_bringup.py` 是每次改动都跑的离线单测（秒级、不开子进程）。这里的脚本会 fork 子进程、开伪终端、跑真的 `lerobot-teleoperate`，一次要几分钟——放进那个套件只会让套件变得又慢又容易挂。

判据：**这里的东西是给人看结论的，不是给 CI 当闸门的。**

## 有哪两个

| 脚本 | 回答什么问题 | 大约多久 |
|---|---|---|
| [`pty_bus_experiment.py`](pty_bus_experiment.py) | Ctrl-C 之后还剩几台电机带电；`so101_release_torque.py` 与 `so101_run.py` 各能不能收干净 | 1–2 分钟 |
| [`pty_ctrl_c_terminal.py`](pty_ctrl_c_terminal.py) | 在**真终端**里按 Ctrl-C（内核向整个前台进程组发 SIGINT）时，包装器还能不能把力矩卸掉 | 约 1 分钟 |

## 怎么跑

```bash
.venv/bin/python tests/audit/pty_bus_experiment.py
.venv/bin/python tests/audit/pty_ctrl_c_terminal.py
```

两个都以末行若干 `ok` 收尾；有 `FAIL` 就是这条结论在这台机器上不再成立。详细日志与临时校准文件写在 `/tmp/so101-*-*/`，脚本会把路径打出来。

## 它们的证据怎么被引用

审计报告用「脚本路径 + 它这次打出来的结论」当证据，不复制输出正文。例：

> 证据: `tests/audit/pty_bus_experiment.py` 实测：裸跑留 `[2,3,4,5,6]` 带电、卸力程序后 0 台、包装器后 0 台。

## 改动这里的规矩

- ⛔ 不许写真实串口、不许打开 `/dev/ttyACM*`、不许碰任何硬件。要碰硬件的实验属于现场验收，由人执行。
- ⛔ 不许把结论写进脚本的输出里当装饰；输出只放实测数字，结论由读的人（或审计报告）下。
- 真机行为变了（例如上游修了 `disconnect`），**先改这两个脚本让它们重新说实话**，再改报告。
