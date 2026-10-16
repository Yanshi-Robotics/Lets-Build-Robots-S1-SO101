#!/usr/bin/env python3
"""ACT course runner: AutoDL Pro API locally, standard LeRobot training remotely.

Python 3.12 standard library only. No paid request is made by --help or --dry-run.
API reference: https://www.autodl.com/docs/instance_pro_api/
Course-maintained scaffold, not an official AutoDL CLI.
"""
from __future__ import annotations

import argparse
import base64
import getpass
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from urllib.parse import urlencode
from urllib.request import Request, urlopen

API_BASE = "https://api.autodl.com"
HTTP_TIMEOUT_SECONDS = 20
STOP_WAIT_SECONDS = 60
POLL_SECONDS = 3
STOP_ATTEMPTS = 3  # Power-off is idempotent; retry a lost response, never retry creation.
HASH_CHUNK_BYTES = 1024 * 1024  # Stream files in 1 MiB blocks, including optimizer state.
STOPPED_STATES = {"shutdown", "stopped"}
ENV_PACKAGES = ("lerobot", "torch", "torchvision", "torchcodec", "safetensors")
SHUTDOWN_RESERVE_SECONDS = 120  # Leave two minutes for observed API power-off.
QUOTE_MAX_AGE_SECONDS = 3600  # A paid submission needs a quote checked within one hour.
PRICE_UNITS_PER_YUAN = 1000  # Official Pro frontend index.454ad1a2.js divides payg_price by 1e3 (2026-10-14).
READ_ONLY_RECOVERY_ATTEMPTS = 3  # Resolve a lost creation response, never repeat the paid request.
SSH_CONNECT_TIMEOUT_SECONDS = 10  # Fail individual connections while allowing bounded boot retries.
STARTUP_READY_SECONDS = 300  # Initial image startup, public key and watchdog readiness allowance.
RECOVERY_FILES = ("pretrained_model/config.json", "pretrained_model/model.safetensors",
                  "pretrained_model/train_config.json", "pretrained_model/policy_preprocessor.json",
                  "pretrained_model/policy_postprocessor.json", "training_state/training_step.json",
                  "training_state/optimizer_param_groups.json", "training_state/optimizer_state.safetensors",
                  "training_state/rng_state.safetensors")
ETA_MARGIN = 1.15  # Allow 15% beyond the measured first-1000-step time.
EXPECTED_PACKAGES = {"lerobot": "0.6.1", "torch": "2.11.0", "torchvision": "0.26.0",
                     "torchcodec": "0.11.1", "safetensors": "0.8.0"}
DEFAULT_TOKEN_FILE = Path.home() / ".config" / "autodl" / "token"


def accepted_versions(name: str) -> tuple[str, ...]:
    version = EXPECTED_PACKAGES[name]
    # Official CUDA-index wheels include +cu130; PyPI baseline wheels omit it.
    return (version, version + "+cu130") if name in {"torch", "torchvision"} else (version,)


def read_token(token_file: Path | None = None) -> str:
    """Explicit file > environment > default file > hidden prompt. Never print it."""
    if token_file is None and os.environ.get("AUTODL_TOKEN"):
        return os.environ["AUTODL_TOKEN"].strip()
    path = (token_file or DEFAULT_TOKEN_FILE).expanduser()
    if path.is_file():
        if os.name != "nt" and path.stat().st_mode & 0o077:
            raise PermissionError(f"Restrict token file permissions: chmod 600 {path}")
        token = path.read_text(encoding="utf-8").strip()
        if not token or "\n" in token or "\r" in token:
            raise ValueError("Token file must contain one nonempty line")
        return token
    if token_file is not None:
        raise FileNotFoundError("Specified token file is missing")
    if not sys.stdin.isatty():
        raise ValueError("Configure the local token file with the auth command first")
    return getpass.getpass("AutoDL developer token (hidden): ").strip()


def configure_token(path: Path) -> None:
    path = path.expanduser()
    if path.exists():
        raise FileExistsError("Token file already exists; remove or replace it explicitly")
    token = getpass.getpass("AutoDL developer token (hidden): ").strip()
    if not token or "\n" in token or "\r" in token:
        raise ValueError("A single nonempty developer token is required")
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w", encoding="utf-8") as stream:
        stream.write(token + "\n")
    print(f"Saved local credential: {path}")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(HASH_CHUNK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def dataset_manifest(root: Path) -> dict[str, str]:
    for folder in ("meta", "data", "videos"):
        directory = root / folder
        if not directory.is_dir() or directory.is_symlink() or any(p.is_symlink() for p in directory.rglob("*")):
            raise ValueError("Frozen meta/data/videos must be real directories without symlinks")
    paths = [p for folder in ("meta", "data", "videos") for p in (root / folder).rglob("*") if p.is_file()]
    if not (root / "meta" / "info.json").is_file() or not paths:
        raise ValueError("Missing frozen dataset files")
    return {p.relative_to(root).as_posix(): sha256(p) for p in sorted(paths)}


def page_items(data: object, operation: str) -> list[dict]:
    """AutoDL returns list:null for some empty pages; require an explicit zero count."""
    if not isinstance(data, dict) or "list" not in data:
        raise ValueError(f"AutoDL {operation} returned an invalid list response")
    rows = data["list"]
    total = data.get("result_total")
    if total is not None and (type(total) is not int or total < 0):
        raise ValueError(f"AutoDL {operation} returned an invalid result count")
    if rows is None and type(total) is int and total == 0:
        return []
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"AutoDL {operation} returned invalid list entries")
    if total == 0 and rows or total and not rows:
        raise ValueError(f"AutoDL {operation} returned inconsistent list/count metadata")
    return rows


class Api:
    def __init__(self, token: str):
        if not token.strip():
            raise ValueError("An AutoDL developer token is required")
        self.token = token.strip()

    def call(self, operation: str, payload: dict) -> object:
        route = "/api/v1/dev/wallet/balance" if operation == "wallet" else f"/api/v1/dev/instance/pro/{operation}"
        url = API_BASE + route
        headers = {"Authorization": self.token, "Content-Type": "application/json"}
        if operation in {"status", "snapshot"}:
            request = Request(url + "?" + urlencode(payload), headers=headers, method="GET")
        else:
            request = Request(url, data=json.dumps(payload).encode(), headers=headers, method="POST")
        # Creation is deliberately not retried: an unknown response may have created an instance.
        with urlopen(request, timeout=HTTP_TIMEOUT_SECONDS) as response:
            result = json.load(response)
        if result.get("code") != "Success":
            raise RuntimeError(f"AutoDL {operation} failed: {result.get('code', 'unknown code')}")
        return result.get("data")

    def instances(self) -> list[dict]:
        found = []
        page = 1
        while True:
            result = self.call("list", {"page_index": page, "page_size": 100})
            found.extend(page_items(result, "list"))
            if page >= int(result.get("max_page", 1)):
                return found
            page += 1

    def release(self, instance: str) -> None:
        if self.call("status", {"instance_uuid": instance}) not in STOPPED_STATES:
            raise ValueError("Release requires an observed stopped instance")
        try:
            self.call("release", {"instance_uuid": instance})
        except Exception:
            pass  # A lost response is resolved by listing; do not claim success from POST alone.
        deadline = time.monotonic() + STOP_WAIT_SECONDS
        while time.monotonic() < deadline:
            if not any(row.get("uuid") == instance for row in self.instances()):
                return
            time.sleep(POLL_SECONDS)
        raise TimeoutError("Instance release was not confirmed")

    def stop(self, instance: str) -> str:
        payload = {"instance_uuid": instance}
        try:
            state = self.call("status", payload)
            if state in STOPPED_STATES:
                return str(state)
        except Exception:
            pass  # A failed status read must never prevent the power-off request.
        deadline = time.monotonic() + STOP_WAIT_SECONDS
        for _ in range(STOP_ATTEMPTS):
            if time.monotonic() >= deadline:
                break
            try:
                self.call("power_off", payload)
            except Exception:
                pass  # A lost response can still mean that power-off succeeded.
            for _ in range(STOP_WAIT_SECONDS // STOP_ATTEMPTS // POLL_SECONDS):
                if time.monotonic() >= deadline:
                    break
                try:
                    state = self.call("status", payload)
                    if state in STOPPED_STATES:
                        return str(state)
                except Exception:
                    pass
                time.sleep(POLL_SECONDS)
        raise TimeoutError("Power-off requested, but stopped state was not confirmed")


def training_command(args: argparse.Namespace, bundle: Path) -> list[str]:
    executable = str(Path(sys.executable).with_name("lerobot-train"))
    if args.resume_checkpoint:
        source = Path(args.resume_checkpoint).resolve()
        for needed in RECOVERY_FILES:
            if not (source / needed).is_file():
                raise ValueError(f"Incomplete recovery checkpoint: {needed}")
        recovered_step = json.loads((source / "training_state/training_step.json").read_text(encoding="utf-8"))["step"]
        if type(recovered_step) is not int or recovered_step < 0:
            raise ValueError("Invalid recovery step")
        target = bundle / "train" / "checkpoints" / f"{recovered_step:06d}"
        shutil.copytree(source, target)
        config_path = target / "pretrained_model" / "train_config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config["dataset"]["repo_id"] != args.dataset_repo:
            raise ValueError("Resume dataset identity differs; verify the frozen version first")
        config.update(output_dir=str(bundle / "train"), steps=args.steps, save_freq=args.save_freq,
                      batch_size=args.batch_size, num_workers=args.num_workers, seed=args.seed)
        config["dataset"]["root"] = str(Path(args.dataset_root).resolve())
        config["policy"].update(device="cuda", push_to_hub=False)
        config["wandb"]["enable"] = False
        write_json(config_path, config)
        return [executable, f"--config_path={config_path}", "--resume=true"]
    return [executable, "--policy.type=act", f"--dataset.repo_id={args.dataset_repo}",
            f"--dataset.root={Path(args.dataset_root).resolve()}", f"--output_dir={bundle / 'train'}",
            f"--job_name={args.run_name}", "--policy.device=cuda", "--policy.use_amp=false",
            "--policy.push_to_hub=false", f"--steps={args.steps}", f"--batch_size={args.batch_size}",
            f"--num_workers={args.num_workers}", f"--save_freq={args.save_freq}", "--log_freq=200",
            f"--seed={args.seed}",
            "--wandb.enable=false", "--wandb.disable_artifact=true"]


def worker(args: argparse.Namespace) -> int:
    """Runs on the Linux cloud instance; never reads an AutoDL API token."""
    bundle = Path(args.run_name).resolve()
    bundle.mkdir(parents=False, exist_ok=False)
    result = {"started_at": now(), "exit_code": None, "completed": False}
    training_deadline = time.monotonic() + args.train_seconds
    try:
        environment = {name: importlib.metadata.version(name) for name in ENV_PACKAGES}
        environment["python"] = sys.version
        write_json(bundle / "environment.json", environment)
        write_json(bundle / "environment-packages.json", dict(sorted(
            (dist.metadata["Name"], dist.version) for dist in importlib.metadata.distributions()
            if dist.metadata["Name"])))
        if any(environment[name] not in accepted_versions(name) for name in EXPECTED_PACKAGES):
            raise ValueError("Cloud training package versions differ from the pinned baseline")
        info = Path(args.dataset_root) / "meta" / "info.json"
        if not info.is_file():
            raise ValueError("Missing frozen dataset meta/info.json")
        data = dataset_manifest(Path(args.dataset_root))
        write_json(bundle / "dataset-sha256.json", data)
        expected = json.loads(Path(args.expected_data).read_text(encoding="utf-8"))
        if data != expected:
            raise ValueError("Uploaded dataset differs from the local frozen dataset")
        # This subprocess uses the cloud GPU, never the local controller's GPU.
        check_code = (
            "import torch,json;from torchcodec.decoders import VideoDecoder;import sys;"
            "assert torch.cuda.is_available();assert torch.version.cuda=='13.0';"
            "torch.ones(1,device='cuda').sum().item();"
            "v=VideoDecoder(sys.argv[1]);v[0];"
            "print(json.dumps({'gpu':torch.cuda.get_device_name(0),'cuda':torch.version.cuda}))"
        )
        video = next(Path(args.dataset_root).joinpath("videos").rglob("*.mp4"))
        gpu_check = subprocess.run([sys.executable, "-c", check_code, str(video)], check=True,
                                   capture_output=True, text=True, timeout=HTTP_TIMEOUT_SECONDS * 3)
        environment.update(json.loads(gpu_check.stdout))
        write_json(bundle / "environment.json", environment)
        commands = []
        starting_step = 0
        if args.resume_checkpoint:
            starting_step = json.loads((Path(args.resume_checkpoint) / "training_state/training_step.json").read_text(encoding="utf-8"))["step"]
            if starting_step >= args.steps:
                raise ValueError("Recovery checkpoint already reached the requested total steps")
        with (bundle / "run.log").open("w", encoding="utf-8") as log:
            if not args.resume_checkpoint and args.steps > args.health_steps:
                health = argparse.Namespace(**vars(args))
                health.steps = args.health_steps
                health.save_freq = args.health_steps
                health_bundle = bundle / "health"
                health_bundle.mkdir()
                command = training_command(health, health_bundle)
                commands.append(command)
                write_json(bundle / "run.json", {"commands": commands, "protocol": "health-then-resume",
                                                "dataset_info_sha256": sha256(info)})
                health_started = time.monotonic()
                health_done = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                             timeout=max(1, training_deadline - time.monotonic()), check=False)
                if health_done.returncode:
                    raise RuntimeError("Initial training failed; see run.log")
                log.flush()
                losses = re.findall(r"(?:^|\s)(?:loss|l1_loss|kld_loss):\s*(\S+)", (bundle / "run.log").read_text(encoding="utf-8"))
                if not losses or not all(math.isfinite(float(value)) for value in losses):
                    raise ValueError("Health phase has missing or nonfinite loss metrics")
                health_checkpoint = (health_bundle / "train/checkpoints/last").resolve()
                if not (health_checkpoint / "training_state/training_step.json").is_file():
                    raise ValueError("Health phase did not save a recoverable checkpoint")
                starting_step = json.loads((health_checkpoint / "training_state/training_step.json").read_text(encoding="utf-8"))["step"]
                if starting_step != args.health_steps:
                    raise ValueError("Health checkpoint step differs from the requested health updates")
                estimate = (time.monotonic() - health_started) / args.health_steps * (args.steps - args.health_steps) * ETA_MARGIN
                result["estimated_remaining_seconds_after_health"] = estimate
                args.resume_checkpoint = str(health_checkpoint)
                if estimate > training_deadline - time.monotonic():
                    raise TimeoutError("Measured speed cannot finish within the remaining budget; health checkpoint retained")
            command = training_command(args, bundle)
            commands.append(command)
            write_json(bundle / "run.json", {"commands": commands, "protocol": "health-then-resume" if len(commands) > 1 else "resume" if starting_step else "single-phase", "dataset_info_sha256": sha256(info),
                                            "resume_checkpoint": args.resume_checkpoint})
            remaining = training_deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(command, args.train_seconds)
            completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                       timeout=remaining, check=False)
        result["exit_code"] = completed.returncode
        checkpoint = bundle / "train" / "checkpoints" / "last" / "pretrained_model"
        needed = ("config.json", "model.safetensors", "train_config.json",
                  "policy_preprocessor.json", "policy_postprocessor.json")
        result["completed"] = completed.returncode == 0 and all((checkpoint / f).is_file() for f in needed)
        if result["completed"]:
            state = checkpoint.parent / "training_state/training_step.json"
            result["completed"] = state.is_file() and json.loads(state.read_text(encoding="utf-8"))["step"] == args.steps
            first_save = (starting_step // args.save_freq + 1) * args.save_freq
            required_steps = set(range(first_save, args.steps + 1, args.save_freq)) | {args.steps}
            for step in required_steps:
                saved = bundle / "train/checkpoints" / f"{step:06d}"
                if not all((saved / name).is_file() for name in RECOVERY_FILES):
                    result["completed"] = False
                    break
                if json.loads((saved / "training_state/training_step.json").read_text(encoding="utf-8"))["step"] != step:
                    result["completed"] = False
                    break
                for name in RECOVERY_FILES:
                    if name.endswith(".json"):
                        json.loads((saved / name).read_text(encoding="utf-8"))
            if result["completed"]:
                finite_code = (
                    "import sys,torch\nfrom pathlib import Path\nfrom safetensors import safe_open\n"
                    "from safetensors.torch import load_file\n"
                    "for folder in map(Path,sys.argv[1:]):\n"
                    " for name in ('training_state/optimizer_state.safetensors','training_state/rng_state.safetensors'):\n"
                    "  with safe_open(str(folder/name),framework='pt',device='cpu') as saved:\n"
                    "   assert list(saved.keys())\n"
                    " assert all(torch.isfinite(t).all().item() for t in load_file(str(folder/'pretrained_model/model.safetensors'),device='cpu').values())\n"
                )
                checkpoints = [str(bundle / "train/checkpoints" / f"{step:06d}") for step in sorted(required_steps)]
                subprocess.run([sys.executable, "-c", finite_code, *checkpoints], check=True,
                               capture_output=True, text=True, timeout=max(1, training_deadline - time.monotonic()))
                result["finite_weights_verified"] = True
    except (subprocess.TimeoutExpired, TimeoutError) as exc:
        result.update(exit_code=124, completed=False, error=str(exc))
    except Exception as exc:
        result.update(exit_code=1, completed=False, error=str(exc))
    finally:
        result["ended_at"] = now()
        write_json(bundle / "result.json", result)
        manifest = {p.relative_to(bundle).as_posix(): sha256(p) for p in sorted(bundle.rglob("*"))
                    if p.is_file() and not p.is_symlink() and p.name != "sha256.json"}
        write_json(bundle / "sha256.json", manifest)
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["completed"] else int(result["exit_code"] or 1)


def verify_bundle(bundle: Path) -> dict:
    manifest = json.loads((bundle / "sha256.json").read_text(encoding="utf-8"))
    if not manifest:
        raise ValueError("Empty download manifest")
    for name, expected in manifest.items():
        path = (bundle / name).resolve()
        if not path.is_relative_to(bundle.resolve()) or not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Missing or changed downloaded file: {name}")
    return json.loads((bundle / "result.json").read_text(encoding="utf-8"))


def load_config(path: Path, *, paid: bool = False) -> dict:
    defaults = json.loads((Path(__file__).parent / "configs/act.example.json").read_text(encoding="utf-8"))
    supplied = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(supplied, dict) or set(supplied) - set(defaults):
        raise ValueError("Use only the documented non-secret configuration fields")
    config = {**defaults, **supplied}
    for key in ("steps", "batch_size", "num_workers", "save_freq", "health_steps", "max_minutes", "retrieve_minutes", "cuda_min"):
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if type(config["seed"]) is not int or config["seed"] < 0:
        raise ValueError("seed must be a nonnegative integer")
    for key in ("max_cost_yuan", "storage_reserve_yuan"):
        if type(config[key]) not in (float, int) or not 0 < config[key] < float("inf"):
            raise ValueError(f"{key} must be a positive finite number")
    if config["max_cost_yuan"] <= config["storage_reserve_yuan"]:
        raise ValueError("Storage reserve consumes the budget")
    if config["max_minutes"] > 24 * 60:
        raise ValueError("This scaffold supports runs of at most 24 hours")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", config["run_name"]):
        raise ValueError("Choose a shell-safe unique run_name")
    for key in ("image_uuid", "gpu_spec"):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", config[key]):
            raise ValueError(f"Invalid {key}")
    for key in ("remote_dir", "remote_env"):
        if not re.fullmatch(r"/[A-Za-z0-9_./-]+", config[key]) or ".." in config[key].split("/"):
            raise ValueError(f"{key} must be a shell-safe absolute Linux path")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", config["dataset_repo"]):
        raise ValueError("dataset_repo must be a valid namespace/name; local/name needs no Hub publication")
    if paid:
        price = config["hourly_price_yuan"]
        if type(price) not in (float, int) or not 0 < price < float("inf"):
            raise ValueError("Fill in the actual Pro hourly price before submitting")
        checked = datetime.fromisoformat(config["quote_checked_at"] or "")
        if checked.tzinfo is None or not 0 <= time.time() - checked.timestamp() <= QUOTE_MAX_AGE_SECONDS:
            raise ValueError("Recheck the official Pro price; the quote must be within one hour")
        budget_seconds(config)
    return config


def budget_seconds(config: dict) -> int:
    seconds = min(config["max_minutes"] * 60,
                  int((config["max_cost_yuan"] - config["storage_reserve_yuan"]) / config["hourly_price_yuan"] * 3600))
    if seconds <= config["retrieve_minutes"] * 60 + SHUTDOWN_RESERVE_SECONDS:
        raise ValueError("Budget cannot cover retrieval and shutdown reserves")
    return seconds


def startup_command(config: dict, remote: str, public_key: str, deadline: float) -> str:
    """Install the public key and detached timer at startup; neither contains a secret."""
    watchdog = (Path(__file__).with_name("watchdog.py")).read_text(encoding="utf-8")
    boot = (
        "import base64,os,pathlib,subprocess,sys;"
        f"root=pathlib.Path({remote!r});root.mkdir(parents=True,exist_ok=True);"
        "keys=pathlib.Path('/root/.ssh');keys.mkdir(mode=0o700,exist_ok=True);"
        "auth=keys/'authorized_keys';"
        f"pub={public_key!r};old=auth.read_text() if auth.exists() else '';"
        "auth.write_text(old+('' if pub in old.splitlines() else ('\\n' if old and not old.endswith('\\n') else '')+pub+'\\n'));auth.chmod(0o600);"
        f"script=root/'watchdog.py';script.write_bytes(base64.b64decode({base64.b64encode(watchdog.encode()).decode()!r}));"
        "log=(root/'watchdog.log').open('a');"
        f"subprocess.Popen([sys.executable,str(script),'--deadline',{str(deadline)!r},'--receipt',str(root/'watchdog-ready.json')],"
        "stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)"
    )
    return "python3 -c " + shlex.quote(boot)


def bootstrap_commands(config: dict, remote: str) -> str:
    python = config["remote_env"] + "/bin/python"
    check = "import sys,importlib.metadata as m;assert sys.version_info[:2]==(3,12);" + ";".join(
        f"assert m.version({name!r}) in {accepted_versions(name)!r}" for name in EXPECTED_PACKAGES)
    # Reuse a verified private environment; base images may have an older Python/CUDA.
    install = [
        'conda_bin="$(command -v conda || true)"',
        'if [ -z "$conda_bin" ]; then conda_bin=/root/miniconda3/bin/conda; fi',
        f'if [ ! -x {shlex.quote(python)} ]; then "$conda_bin" create -y --override-channels -c conda-forge -p {shlex.quote(config["remote_env"])} python=3.12.3 pip ffmpeg=6.1.1; fi',
        shlex.join([python, "-m", "pip", "install", "--no-cache-dir", "torch==2.11.0", "torchvision==0.26.0", "--index-url", "https://download.pytorch.org/whl/cu130"]),
        shlex.join([python, "-m", "pip", "install", "--no-cache-dir", "-r", remote + "/requirements-cloud.txt"]),
    ]
    return "set -e; if ! " + shlex.join([python, "-c", check]) + "; then " + "; ".join(install) + "; fi; " + shlex.join([python, "-m", "pip", "check"])


def submit(config: dict, api: Api) -> int:
    """One owned instance, a deadline covering all paid work, and observed cleanup."""
    data_root = Path(config["dataset_root"]).expanduser().resolve()
    data = dataset_manifest(data_root)  # Check/hash everything before spending money.
    for name in ("ssh", "scp"):
        if not shutil.which(name):
            raise FileNotFoundError(f"Install an OpenSSH client providing {name}")
    key = Path(config["ssh_key"]).expanduser().resolve()
    public_key = Path(str(key) + ".pub").read_text(encoding="utf-8").strip()
    if not key.is_file() or "\n" in public_key or not public_key.startswith(("ssh-ed25519 ", "ssh-rsa ", "ecdsa-sha2-")):
        raise ValueError("A matching local SSH private/public key pair is required")
    resume = Path(config["resume_checkpoint"]).expanduser().resolve() if config["resume_checkpoint"] else None
    if resume and not all((resume / name).is_file() for name in RECOVERY_FILES):
        raise ValueError("Incomplete local recovery checkpoint")
    if resume and any(p.is_symlink() for p in resume.rglob("*")):
        raise ValueError("Recovery checkpoint contents must not contain symlinks")
    wallet = api.call("wallet", {})
    assets = wallet.get("assets") if isinstance(wallet, dict) else None
    if type(assets) not in (int, float) or not math.isfinite(assets) or assets / PRICE_UNITS_PER_YUAN < config["max_cost_yuan"]:
        raise ValueError("Verified cash balance must cover the experiment budget; voucher eligibility is not assumed")
    if config["image_uuid"].startswith("image-"):
        images = api.call("image/private/list", {"page_index": 1, "page_size": 100})
        if not any(row.get("image_uuid") == config["image_uuid"] and row.get("status") == "finished" for row in page_items(images, "image/private/list")):
            raise ValueError("Selected private image is not available with finished status")
    destination = Path(config["output_root"]) / config["run_name"]
    destination.mkdir(parents=True, exist_ok=False)
    write_json(destination / "dataset-upload.json", data)
    write_json(destination / "submission-config.json", config)
    suffix = uuid.uuid4().hex[:12]
    name = config["run_name"] + "-" + suffix
    remote = config["remote_dir"].rstrip("/") + "/" + name
    seconds = budget_seconds(config)
    started = time.time()
    deadline = time.monotonic() + seconds
    work_deadline = deadline - SHUTDOWN_RESERVE_SECONDS
    receipt = {"instance_name": name, "instance_uuid": None, "started_at": now(),
               "deadline_epoch": started + seconds, "budget_yuan": config["max_cost_yuan"],
               "hourly_quote_yuan": config["hourly_price_yuan"], "create_outcome": "unknown",
               "shutdown_verified": False, "release_verified": False, "download_verified": False,
               "training_started": False, "scaffold_sha256": sha256(Path(__file__))}
    receipt_path = destination / "controller.json"

    def save():
        write_json(receipt_path, receipt)

    def execute(command, *, check=True):
        remaining = work_deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Paid-work deadline reached")
        with (destination / "controller.log").open("a", encoding="utf-8") as log:
            environment = {k: v for k, v in os.environ.items() if k not in {"AUTODL_TOKEN", "AUTODL_PRO_TOKEN"}}
            return subprocess.run(command, check=check, timeout=remaining, stdout=log, stderr=subprocess.STDOUT, env=environment)

    instance = None
    downloaded = False
    result = None
    status_code = 1
    save()  # Reserve the run identity before the sole creation request.
    try:
        payload = {"req_gpu_amount": 1, "expand_system_disk_by_gb": 0, "gpu_spec_uuid": config["gpu_spec"],
                   "image_uuid": config["image_uuid"], "cuda_v_from": config["cuda_min"],
                   "instance_name": name, "start_command": startup_command(config, remote, public_key, started + seconds)}
        instance = api.call("create", payload)
        if not isinstance(instance, str) or not instance.startswith("pro-"):
            raise ValueError("AutoDL returned an invalid instance ID; inspect the reserved run name")
        receipt.update(instance_uuid=instance, create_outcome="created")
        save()
        print(f"Created {instance}; all paid work has a {seconds // 60}-minute ceiling.", flush=True)
        snapshot = None
        while time.monotonic() < work_deadline:
            snapshot = api.call("snapshot", {"instance_uuid": instance})
            if snapshot.get("proxy_host") and snapshot.get("ssh_port"):
                break
            time.sleep(POLL_SECONDS)
        host = snapshot.get("proxy_host", "")
        port = snapshot.get("ssh_port", 0)
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.-]*", host) or type(port) is not int or not 0 < port <= 65535:
            raise ValueError("Invalid SSH connection metadata from AutoDL")
        # Snapshot can contain root_password and jupyter_token. Never persist/print it.
        receipt["connection"] = {"host": host, "port": port}
        receipt["provider_price_raw"] = {k: snapshot[k] for k in ("payg_price", "origin_pay_price") if k in snapshot}
        price_raw = snapshot.get("payg_price")
        if type(price_raw) not in (int, float) or not 0 < price_raw < float("inf"):
            raise ValueError("Actual instance rate could not be verified; stopping before setup")
        actual_price = price_raw / PRICE_UNITS_PER_YUAN
        receipt["actual_hourly_price_yuan"] = actual_price
        if actual_price > config["hourly_price_yuan"]:
            raise ValueError("Actual Pro rate exceeds the confirmed quote; stopping before setup")
        save()
        target = "root@" + host
        known_hosts = (destination.resolve() / "known_hosts").as_posix()
        options = ["-i", str(key), "-o", "BatchMode=yes", "-o", f"ConnectTimeout={SSH_CONNECT_TIMEOUT_SECONDS}",
                   "-o", "StrictHostKeyChecking=accept-new", "-o", f'UserKnownHostsFile="{known_hosts}"']
        ssh = ["ssh", "-p", str(port), *options, target]
        scp = ["scp", "-P", str(port), *options]
        ready_check = (
            "import json,os,time;"
            f"r=json.load(open({(remote + '/watchdog-ready.json')!r}));"
            f"assert r['ready'] and abs(r['deadline']-{started + seconds!r})<1;"
            f"assert abs(r['checked_at']-time.time())<{STARTUP_READY_SECONDS};os.kill(r['pid'],0)"
        )
        while execute(ssh + [shlex.join(["python3", "-c", ready_check])], check=False).returncode:
            if time.monotonic() >= min(work_deadline, deadline - seconds + STARTUP_READY_SECONDS):
                raise TimeoutError("SSH/public key/independent shutdown timer could not be verified")
            time.sleep(POLL_SECONDS)
        receipt["watchdog_verified"] = True
        save()
        for filename in ("autodl_act.py", "requirements-cloud.txt"):
            execute(scp + [str(Path(__file__).with_name(filename)), f"{target}:{remote}/{filename}"])
        print("Preparing the pinned training environment and uploading frozen data.", flush=True)
        execute(ssh + ["bash -lc " + shlex.quote(bootstrap_commands(config, remote))])
        remote_data = remote + "/dataset"
        execute(ssh + ["mkdir -p " + shlex.quote(remote_data)])
        execute(scp + ["-r", *[str(data_root / folder) for folder in ("meta", "data", "videos")], f"{target}:{remote_data}/"])
        execute(scp + [str(destination / "dataset-upload.json"), f"{target}:{remote}/expected-data.json"])
        remote_resume = None
        if resume:
            execute(ssh + ["mkdir -p " + shlex.quote(remote + "/resume")])
            execute(scp + ["-r", str(resume), f"{target}:{remote}/resume/"])
            remote_resume = remote + "/resume/" + resume.name
        train_seconds = int(work_deadline - time.monotonic() - config["retrieve_minutes"] * 60)
        if train_seconds <= 0:
            raise TimeoutError("Environment/upload consumed the training allowance")
        command = [config["remote_env"] + "/bin/python", "autodl_act.py", "worker", "--dataset-root", remote_data,
                   "--dataset-repo", config["dataset_repo"], "--run-name", "result", "--expected-data", remote + "/expected-data.json",
                   "--train-seconds", str(train_seconds), "--max-minutes", str(config["max_minutes"])]
        for field in ("steps", "batch_size", "num_workers", "seed", "save_freq", "health_steps"):
            command.extend(["--" + field.replace("_", "-"), str(config[field])])
        if remote_resume:
            command.extend(["--resume-checkpoint", remote_resume])
        receipt["training_started"] = True
        save()
        print("Training: first save a health checkpoint, then resume to the total target within budget.", flush=True)
        library_path = shlex.quote(config["remote_env"] + "/lib")
        trained = execute(ssh + [f'cd {shlex.quote(remote)} && export LD_LIBRARY_PATH={library_path}${{LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}} && {shlex.join(command)}'], check=False)
        receipt["worker_exit_code"] = trained.returncode
        print("Retrieving the experiment bundle before stopping the instance.", flush=True)
        bundle = destination / "result"
        bundle.mkdir()
        execute(scp + ["-r", f"{target}:{remote}/result/.", str(bundle)])
        result = verify_bundle(bundle)
        downloaded = True
        receipt.update(download_verified=True, training=result)
        status_code = 0 if trained.returncode == 0 and result["completed"] else 1
    except (Exception, KeyboardInterrupt) as exc:
        receipt["error"] = f"{type(exc).__name__}: {exc}"
        if instance is None:
            try:
                for attempt in range(READ_ONLY_RECOVERY_ATTEMPTS):
                    matches = [row for row in api.instances() if row.get("name") == name]
                    if len(matches) == 1:
                        instance = matches[0]["uuid"]
                        receipt.update(instance_uuid=instance, create_outcome="recovered-after-unknown-response")
                        break
                    if len(matches) > 1 or time.monotonic() >= work_deadline:
                        break
                    if attempt + 1 < READ_ONLY_RECOVERY_ATTEMPTS:
                        time.sleep(POLL_SECONDS)
            except Exception:
                pass
        print(receipt["error"], file=sys.stderr)
    finally:
        if instance:
            try:
                receipt["final_state"] = api.stop(instance)
                receipt["shutdown_verified"] = True
                if downloaded or not receipt["training_started"]:
                    api.release(instance)
                    receipt["release_verified"] = True
                else:
                    receipt["storage_retained_reason"] = "Training may contain unique results; verified download required before release"
                    print("Compute stopped; cloud results retained. System-disk billing continues until release.", file=sys.stderr)
            except Exception as exc:
                status_code = 1
                receipt["cleanup_error"] = f"{type(exc).__name__}: {exc}"
                print(f"CLEANUP NOT VERIFIED: inspect {instance} in AutoDL now.", file=sys.stderr)
        else:
            receipt["create_outcome"] = "unknown-inspect-instance-name"
            print(f"Creation outcome unknown: check AutoDL for {name}. Never resubmit this run.", file=sys.stderr)
        receipt["ended_at"] = now()
        receipt["elapsed_seconds"] = time.time() - started
        receipt["estimated_compute_cost_yuan"] = round(receipt["elapsed_seconds"] / 3600 * receipt.get("actual_hourly_price_yuan", config["hourly_price_yuan"]), 4)
        receipt["billing_note"] = "Quote-based estimate, not a settled provider bill; disk fees are separate"
        save()
    return status_code


def positive(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("Must be a positive integer")
    return result


def parser() -> argparse.ArgumentParser:
    main = argparse.ArgumentParser(description=__doc__)
    sub = main.add_subparsers(dest="action", required=True)
    auth = sub.add_parser("auth", help="Save a local developer token with hidden input; no API call")
    auth.add_argument("--token-file", type=Path, default=DEFAULT_TOKEN_FILE)
    configure = sub.add_parser("configure", help="Write a non-secret experiment configuration; no API call")
    configure.add_argument("--config", type=Path, required=True)
    configure.add_argument("--gpu-spec", default="5090-p")
    configure.add_argument("--image-uuid", default="base-image-mbr2n4urrc")
    configure.add_argument("--dataset-root", required=True)
    configure.add_argument("--dataset-repo", required=True)
    configure.add_argument("--run-name", required=True)
    configure.add_argument("--hourly-price-yuan", type=float, required=True)
    configure.add_argument("--max-cost-yuan", type=float, default=10)
    configure.add_argument("--max-minutes", type=positive, default=240)
    configure.add_argument("--retrieve-minutes", type=positive, default=15)
    configure.add_argument("--steps", type=positive, default=100000)
    configure.add_argument("--save-freq", type=positive, default=20000)
    configure.add_argument("--ssh-key", default="~/.ssh/id_ed25519")
    submission = sub.add_parser("submit", help="Create one paid instance, train, retrieve and clean up")
    submission.add_argument("--config", type=Path, required=True)
    submission.add_argument("--resume-checkpoint")
    submission.add_argument("--dry-run", action="store_true")
    submission.add_argument("--token-file", type=Path)
    preflight = sub.add_parser("preflight", help="Read wallet, instances and private images without renting compute")
    preflight.add_argument("--token-file", type=Path)
    for name in ("status", "stop"):
        action = sub.add_parser(name)
        action.add_argument("--instance", required=True)
        action.add_argument("--token-file", type=Path)
    for name in ("worker",):
        action = sub.add_parser(name)
        action.add_argument("--dataset-root", required=True)
        action.add_argument("--dataset-repo", required=True)
        action.add_argument("--run-name", required=True)
        action.add_argument("--max-minutes", type=positive, required=True)
        action.add_argument("--steps", type=positive, default=100000)
        action.add_argument("--batch-size", type=positive, default=8)
        action.add_argument("--num-workers", type=positive, default=4)
        action.add_argument("--save-freq", type=positive, default=20000)
        action.add_argument("--seed", type=int, default=1000)
        action.add_argument("--health-steps", type=positive, default=1000)
        action.add_argument("--resume-checkpoint")
        if name == "worker":
            action.add_argument("--expected-data", required=True)
            action.add_argument("--train-seconds", type=positive, required=True)
    return main


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.action == "auth":
        configure_token(args.token_file)
        return 0
    if args.action == "configure":
        if args.config.exists():
            raise FileExistsError("Configuration already exists; edit it explicitly or choose a new run")
        config = json.loads((Path(__file__).parent / "configs/act.example.json").read_text(encoding="utf-8"))
        config.update({k: v for k, v in vars(args).items() if k in config})
        config["quote_checked_at"] = now()
        if not config["run_name"] or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", config["run_name"]):
            raise ValueError("Choose a shell-safe unique run_name")
        args.config.parent.mkdir(parents=True, exist_ok=True)
        with args.config.open("x", encoding="utf-8") as stream:
            json.dump(config, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        print(f"Saved non-secret configuration: {args.config}")
        return 0
    if args.action == "submit":
        config = load_config(args.config, paid=not args.dry_run)
        if args.resume_checkpoint:
            config["resume_checkpoint"] = args.resume_checkpoint
        if args.dry_run:
            print(json.dumps({"config": config, "paid_work_seconds": budget_seconds(config) if config["hourly_price_yuan"] else None,
                              "quote_required": not bool(config["hourly_price_yuan"] and config["quote_checked_at"])}, ensure_ascii=False, indent=2))
            print("DRY RUN: no token, API, SSH, upload or training was used.")
            return 0
        return submit(config, Api(read_token(args.token_file)))
    if hasattr(args, "run_name") and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.run_name):
        raise ValueError("Use letters, digits, underscores, dots and hyphens for the run name")
    if args.action == "worker":
        return worker(args)
    api = Api(read_token(args.token_file))
    if args.action == "preflight":
        wallet = api.call("wallet", {})
        instances = api.instances()
        images = api.call("image/private/list", {"page_index": 1, "page_size": 100})
        safe_wallet = {k + "_yuan": wallet[k] / PRICE_UNITS_PER_YUAN for k in ("assets", "accumulate", "voucher_balance") if type(wallet.get(k)) in (int, float)} if isinstance(wallet, dict) else {"available": True}
        print(json.dumps({"wallet": safe_wallet, "instances": [
            {k: r.get(k) for k in ("uuid", "name", "status", "gpu_spec_uuid")} for r in instances],
            "private_images": [{k: r.get(k) for k in ("image_uuid", "name", "status")} for r in page_items(images, "image/private/list")]}, ensure_ascii=False, indent=2))
        return 0
    if args.action == "status":
        print(api.call("status", {"instance_uuid": args.instance}))
    elif args.action == "stop":
        print(api.stop(args.instance))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (Exception, KeyboardInterrupt) as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(1)
