"""Verify exact local telemetry and checkpoints for one completed LeRobot run.

The config and metrics must belong to the supplied run. This verifies saved
training evidence, not held-out accuracy or physical robot performance.
"""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify(config_path: Path, metrics_directory: Path):
    config = json.loads(config_path.read_text())
    steps, frequency = config["steps"], config["log_freq"]
    with (metrics_directory / "metrics.jsonl").open() as stream:
        records = [json.loads(line) for line in stream]
    with (metrics_directory / "metrics.csv").open(newline="") as stream:
        csv_records = list(csv.DictReader(stream))
    expected = list(range(frequency, steps + 1, frequency))
    if [row["step"] for row in records] != expected:
        raise ValueError("Metric steps are incomplete, duplicated or out of order")
    if len(csv_records) != len(records):
        raise ValueError("CSV and JSONL record counts differ")
    required = {"loss", "l1_loss", "kld_loss", "grad_norm", "lr", "update_s", "dataloading_s"}
    for raw, row in zip(records, csv_records, strict=True):
        if not required.issubset(raw):
            raise ValueError("Required metrics are missing")
        if raw.keys() != row.keys():
            raise ValueError("CSV and JSONL columns differ")
        for key, value in raw.items():
            if key == "recorded_at_utc":
                if row[key] != value:
                    raise ValueError("CSV and JSONL timestamps differ")
            elif not isinstance(value, (int, float)) or not math.isfinite(value) or float(row[key]) != value:
                raise ValueError(f"Nonfinite or inconsistent metric {key} at {raw['step']}")
        # The loss is computed in float32 before its interval average is logged.
        if not math.isclose(raw["loss"], raw["l1_loss"] + config["policy"]["kl_weight"] * raw["kld_loss"], rel_tol=1e-5, abs_tol=1e-7):
            raise ValueError("ACT total loss does not match its recorded L1 and weighted KL")
    checkpoint_steps = list(range(config["save_freq"], steps + 1, config["save_freq"]))
    if steps not in checkpoint_steps:
        checkpoint_steps.append(steps)
    checkpoint_root = Path(config["output_dir"]) / "checkpoints"
    checkpoints = []
    for step in checkpoint_steps:
        from lerobot.common.train_utils import get_step_identifier
        directory = checkpoint_root / get_step_identifier(step, steps)
        state = directory / "training_state/training_step.json"
        if json.loads(state.read_text())["step"] != step:
            raise ValueError(f"Invalid checkpoint state at {step}")
        model = directory / "pretrained_model/model.safetensors"
        saved_config = directory / "pretrained_model/train_config.json"
        saved = json.loads(saved_config.read_text())
        differences = [key for key in saved.keys() | config.keys() if saved.get(key) != config.get(key)]
        if differences:
            raise ValueError(f"Checkpoint config differs: {differences}")
        if not model.is_file() or model.stat().st_size == 0:
            raise ValueError(f"Missing model at {step}")
        checkpoints.append({"step": step, "path": str(directory), "model_sha256": sha256(model)})
    return {
        "complete": True,
        "training_steps": steps,
        "metric_points": len(records),
        "first_step": records[0]["step"],
        "last_step": records[-1]["step"],
        "log_frequency": frequency,
        "config_sha256": sha256(config_path),
        "csv_sha256": sha256(metrics_directory / "metrics.csv"),
        "jsonl_sha256": sha256(metrics_directory / "metrics.jsonl"),
        "checkpoints": checkpoints,
        "validation_split": config["dataset"]["eval_split"],
        "physical_acceptance": "not performed",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--metrics-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = verify(args.config, args.metrics_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Verified {result['metric_points']} exact records and {len(result['checkpoints'])} checkpoints")


if __name__ == "__main__":
    main()
