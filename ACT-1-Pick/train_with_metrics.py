"""Run the installed LeRobot trainer and preserve exact, interval-mean metrics.

No optimizer, sampler, loss, or shared environment source is changed. The handler
reads MetricsTracker before terminal formatting and before its meters are reset.
Usage: python ACT-1-Pick/train_with_metrics.py --metrics-dir RUN/metrics --config_path=CONFIG
"""

import argparse
import csv
from datetime import datetime, timezone
import json
import logging
import math
from pathlib import Path
import sys


REQUIRED_METRICS = (
    "loss", "l1_loss", "kld_loss", "grad_norm", "lr", "update_s", "dataloading_s"
)


class ExactMetricsHandler(logging.Handler):
    """Fail loudly on missing/nonfinite records or disk errors; never round steps."""

    def __init__(self, directory: Path, tracker_type: type):
        super().__init__()
        directory.mkdir(parents=True, exist_ok=True)
        self.tracker_type = tracker_type
        self.jsonl = (directory / "metrics.jsonl").open("x", encoding="utf-8")
        self.csv_file = (directory / "metrics.csv").open("x", newline="", encoding="utf-8")
        self.writer = None
        self.last_step = 0

    def emit(self, record):
        if not isinstance(record.msg, self.tracker_type):
            return
        raw = record.msg.to_dict()
        if "l1_loss" not in raw:  # Evaluation trackers are not training records.
            return
        if any(k not in raw for k in REQUIRED_METRICS):
            raise RuntimeError(f"Missing training metrics: {set(REQUIRED_METRICS) - raw.keys()}")
        step = raw.pop("steps")
        if type(step) is not int or step <= self.last_step:
            raise RuntimeError(f"Invalid or duplicate training step: {step}")
        if any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in raw.values()):
            raise RuntimeError("Nonfinite or nonnumeric training metric")
        row = {"step": step, "recorded_at_utc": datetime.now(timezone.utc).isoformat(), **raw}
        if self.writer is None:
            self.writer = csv.DictWriter(self.csv_file, fieldnames=list(row))
            self.writer.writeheader()
        self.jsonl.write(json.dumps(row, allow_nan=False) + "\n")
        self.writer.writerow(row)
        self.jsonl.flush()
        self.csv_file.flush()
        self.last_step = step

    def close(self):
        self.jsonl.close()
        self.csv_file.close()
        super().close()


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--metrics-dir", required=True, type=Path)
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    from lerobot.scripts import lerobot_train as trainer
    from lerobot.utils.logging_utils import MetricsTracker

    original_init = trainer.init_logging
    handlers = []

    def init_with_metrics(*a, **kw):
        original_init(*a, **kw)
        accelerator = kw.get("accelerator")
        if accelerator is not None and not accelerator.is_main_process:
            return
        handler = ExactMetricsHandler(args.metrics_dir, MetricsTracker)
        logging.getLogger().addHandler(handler)
        handlers.append(handler)

    trainer.init_logging = init_with_metrics
    try:
        trainer.main()
    finally:
        for handler in handlers:
            logging.getLogger().removeHandler(handler)
            handler.close()


if __name__ == "__main__":
    main()
