# AutoDL ACT training scaffold

[中文说明](docs/zh/README.md)

Run one ACT experiment on AutoDL Container Instance Pro, retrieve the complete result and verify resource cleanup. This is maintained by the course, using the official HTTP API; it is not an official AutoDL CLI. Local management uses Python 3.12's standard library and an OpenSSH client. Cloud training uses LeRobot.

Run commands from the repository root. Keep the whole `autoDL/` folder: the entry point needs its example configuration, dependency pins and independent shutdown watchdog. Real paid training, power-off and release still require provider acceptance; offline tests do not establish these results.

## Configure local access

Register, complete verification and obtain Console → Account → Settings → Developer Token. Save it through a hidden prompt:

```bash
python autoDL/autodl_act.py auth
python autoDL/autodl_act.py preflight
```

The default file is `~/.config/autodl/token` on all systems, with `~` resolved to your user home. POSIX permissions are owner-only; on Windows keep it inside your private user profile and check its inherited access permissions. An existing file is never overwritten. Explicit `--token-file` takes precedence over `AUTODL_TOKEN`, the default file and a hidden terminal prompt. No command accepts a token argument, and tokens are never sent to workers or written to experiment records.

Configure an OpenSSH key pair; the default is `~/.ssh/id_ed25519` and its `.pub` file. Encrypted private keys must be available through your SSH agent. The public key is installed on the new instance; the private key stays local. Each experiment records first-use SSH host keys and rejects subsequent changes.

## Prepare and submit an experiment

Copy [configs/act.example.json](configs/act.example.json) into `outputs/cloud-configs/`, edit dataset paths and choose a new run name. Confirm the **Pro** hourly quote and fill `hourly_price_yuan` and `quote_checked_at` with a timezone-aware ISO timestamp. The example deliberately has no price; dry-run works, paid submission refuses an unconfirmed quote. Alternatively, `configure --help` shows how to generate this file from arguments.

```bash
python autoDL/autodl_act.py submit --config autoDL/configs/act.example.json --dry-run
python autoDL/autodl_act.py submit --config outputs/cloud-configs/act-cloud-001.json
```

The official API appendix lists `base-image-mbr2n4urrc` as a Miniconda base image. It identifies the initial environment, not the final training stack. The scaffold creates Python 3.12.3 with FFmpeg 6.1.1 and installs [requirements-cloud.txt](requirements-cloud.txt); PyTorch 2.11.0 and Torchvision 0.26.0 come from the official CUDA 13.0 wheel index. A private image can reuse an existing matching environment. CUDA operations and video decoding must pass before training.

Only `meta/`, `data/` and `videos/` are uploaded, with file-by-file SHA-256 comparison. Local venvs, credentials, calibration and reports are excluded. Dataset `repo_id` can be `local/<name>`; this does not publish anything to Hugging Face.

The default experiment uses batch size 8, four data workers, seed 1000, no AMP, no W&B and no Hub publishing. It trains 1000 updates from scratch, saves a health checkpoint and measures speed. If the remaining allowance can cover the target, it resumes optimizer/random state to 100000 total updates, saving every 20000 steps. Both real commands are recorded. A slow health phase is retained as an incomplete experiment rather than exceeding the planned time allowance.

## Budget and cleanup

Before creation, verified cash balance must cover the configured budget. After creation, the actual instance rate must be readable and no higher than the confirmed quote; otherwise setup is cancelled and the new instance is stopped and released.

The budget clock starts before the sole creation request. Environment setup, upload and retrieval count toward it. `storage_reserve_yuan` defaults to 0.20 CNY for up to two calendar days at the documented unexpanded-disk rate; `retrieve_minutes` reserves transfer time and two more minutes reserve power-off. An independent cloud watchdog must be verified before environment setup. Keep the local controller online for result retrieval.

The creation endpoint is never automatically retried. An unknown response is checked against the unique instance name; if it cannot be resolved, inspect that name in the console before taking another action.

Downloaded files are verified before power-off and release. A training failure can still yield a valid, recoverable result bundle. If retrieval fails after training started, the scaffold powers off and preserves cloud files: **system-disk fees continue until release**. It never deletes unverified results. Final charges come from the provider bill, not the estimate in `controller.json`.

```bash
python autoDL/autodl_act.py status --instance pro-YOUR_INSTANCE
python autoDL/autodl_act.py stop --instance pro-YOUR_INSTANCE
```

## Results and recovery

Each new experiment has its own `outputs/cloud/<run>/` directory. `controller.json` records the owned instance and observed cleanup; `controller.log` covers setup/transfer. `result/` contains `run.log`, actual commands, environment, dataset hashes, checksums, completion status, and complete `train/checkpoints/` policy/optimizer directories. `result/health/` retains the initial recovery point. Keep the entire `pretrained_model/` directory for inference.

To resume, choose a new configuration and run name, then pass a complete local checkpoint including `training_state/`:

```bash
python autoDL/autodl_act.py submit --config outputs/cloud-configs/act-cloud-002.json --resume-checkpoint outputs/cloud/act-cloud-001/result/train/checkpoints/020000
```

A completed training run does not prove robot task success. Use independent evaluation data and user-operated robot tests. If all demonstrations were used for training, their prediction error is a training-fit check.

## Offline checks and official references

```bash
python -m unittest discover -s tests -p test_autodl_act.py -v
```

These tests use explicit fixtures without network, SSH, GPU or paid resources. Linux cloud acceptance and the macOS/Windows controller paths remain separate checks.

[Official Pro API](https://www.autodl.com/docs/instance_pro_api/) · [Billing](https://www.autodl.com/docs/price/) · [Shutdown command](https://www.autodl.com/docs/save_money/)
