# AutoDL ACT 云端训练脚手架

[English](../../README.md)

这个文件夹用于提交一次 ACT 云端实验、取回完整结果，并确认资源关停。程序由课程维护，通过 AutoDL 官方 HTTP API 管理容器实例 Pro；实际训练使用 LeRobot。管理程序只用 Python 3.12 标准库，传输和连接依靠 OpenSSH 客户端。

在代码仓根目录运行命令，保留完整的 autoDL 文件夹。真实付费训练、关机和释放需要云端验收，离线测试通过不能证明这些操作已经成功。

## 配置本地访问

注册、实名认证后，从“控制台 → 账号 → 设置 → 开发者 Token”取得凭据，用隐藏输入保存：

```bash
python autoDL/autodl_act.py auth
python autoDL/autodl_act.py preflight
```

默认文件为用户主目录下的 `.config/autodl/token`，三个系统都按各自用户主目录解析。Linux 与 macOS 的权限只允许本人访问；Windows 使用私人用户目录，并检查继承的访问权限。已有文件不会覆盖。读取顺序为显式指定的 `--token-file`、AUTODL_TOKEN 环境变量、默认文件、终端隐藏输入。Token 不作为命令参数，不上传云端，也不写入实验记录。

准备 OpenSSH 密钥对，默认使用 `~/.ssh/id_ed25519` 和旁边的公钥文件。加密私钥需要通过 SSH Agent 提供。新实例只收到公钥，私钥保留本地。每次实验单独记录首次连接的主机密钥，后续变化会拒绝连接。

## 准备配置并提交

把 [配置示例](../../configs/act.example.json) 复制到 outputs/cloud-configs，填写数据路径和新的实验名称。核对 Pro 实例的小时报价，填写 hourly_price_yuan 与带时区的 quote_checked_at。示例故意不填价格：可以预览，真实提交会拒绝未经确认的报价。也可以用 `configure --help` 查看生成配置的参数。

```bash
python autoDL/autodl_act.py submit --config autoDL/configs/act.example.json --dry-run
python autoDL/autodl_act.py submit --config outputs/cloud-configs/act-cloud-001.json
```

官方附录列出的 base-image-mbr2n4urrc 是 Miniconda 基础镜像，编号指定初始环境。脚手架会创建 Python 3.12.3、FFmpeg 6.1.1 环境，安装锁定的训练依赖；PyTorch 2.11.0 与 Torchvision 0.26.0 从官方 CUDA 13.0 索引安装。已有私有镜像可以复用匹配的环境。训练前必须通过 CUDA 运算和视频解码检查。

只上传冻结数据的 meta、data、videos 三个目录，逐个核对 SHA-256；venv、凭据、机械臂标定和内部报告不上传。dataset_repo 可以填写 local/数据集名称，这不会向 Hugging Face 发布数据。

默认批量大小 8、4 个数据加载进程、随机种子 1000，关闭混合精度、W&B 和 Hub 发布。先从头更新 1000 步、保存恢复点并测量速度；预算允许时恢复优化器与随机状态，继续到总计 100000 步，每 20000 步保存。两段真实命令都会记录。预计时间不足时保留恢复点，结果标为未完成。

## 预算与资源清理

创建前确认现金余额足以覆盖配置的费用上限。创建后再次读取实际单价；无法确认单价，或实际单价高于已核对报价时，程序停止配置环境，将新实例关机并释放。

从创建请求之前开始计时，安装环境、上传、训练和下载全部计入。系统盘费用默认预留 0.20 元，覆盖官方未扩容系统盘最多两个自然日的费用；另留下载时间和两分钟关机时间。配置环境前必须确认云端独立计时关机程序在运行。本地控制终端仍需保持在线，才能取回结果。

创建接口不会自动重试。返回结果未知时按唯一实例名称查询，仍无法确认就先检查控制台，不能直接重新提交。

下载校验成功后才关机并释放实例；训练失败也可以取回可恢复的完整结果。训练已经开始、下载却失败时，程序先关机并保留云端文件，系统盘仍继续收费，直到实例释放。程序不会删除尚未校验的结果。controller.json 中的费用估算不能替代实际账单。

```bash
python autoDL/autodl_act.py status --instance pro-YOUR_INSTANCE
python autoDL/autodl_act.py stop --instance pro-YOUR_INSTANCE
```

## 结果与恢复

每次实验写入新的 outputs/cloud/实验名称目录。controller.json 记录实例与关停结果，controller.log 记录安装和传输；result 目录保存训练日志、实际命令、环境、数据哈希、文件校验、完成状态，以及完整模型和优化器检查点。result/health 保留首次恢复点。推理时使用整个 pretrained_model 目录。

恢复时使用新的配置与实验名称，提供包含 training_state 的完整本地检查点：

```bash
python autoDL/autodl_act.py submit --config outputs/cloud-configs/act-cloud-002.json --resume-checkpoint outputs/cloud/act-cloud-001/result/train/checkpoints/020000
```

训练完成不能证明机器人任务成功，需要独立评估数据和用户操作的真机测试。全部示范都参加训练时，其预测误差只属于训练拟合检查。

## 离线检查与官方资料

```bash
python -m unittest discover -s tests -p test_autodl_act.py -v
```

测试使用明确的夹具，不联网、不连接 SSH、不使用 GPU，也不租用资源。Linux 云端验收与 macOS、Windows 控制端验证分别记录。

[官方 Pro API](https://www.autodl.com/docs/instance_pro_api/) · [计费规则](https://www.autodl.com/docs/price/) · [关机命令](https://www.autodl.com/docs/save_money/)
