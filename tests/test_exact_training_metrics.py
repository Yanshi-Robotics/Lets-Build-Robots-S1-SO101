"""CPU checks for exact LeRobot telemetry, including fail-closed disk writes."""

import csv
import importlib.util
import json
import logging
from pathlib import Path
import tempfile
import unittest

from lerobot.utils.logging_utils import AverageMeter, MetricsTracker

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("train_with_metrics", ROOT / "ACT-1-Pick/train_with_metrics.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ExactTrainingMetricsTests(unittest.TestCase):
    def test_source_precision_survives_terminal_rounding_and_reset(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "temp") as directory:
            handler = module.ExactMetricsHandler(Path(directory), MetricsTracker)
            tracker = MetricsTracker(8, 31968, 90, {
                k: AverageMeter(k, ":.3f") for k in module.REQUIRED_METRICS
            }, initial_step=199)
            tracker.step()
            for key in module.REQUIRED_METRICS:
                setattr(tracker, key, 0.123456789012345)
                setattr(tracker, key, 0.987654321012345)
            source = tracker.to_dict()
            record = logging.LogRecord("test", logging.INFO, "", 0, tracker, (), None)
            handler.emit(record)
            tracker.reset_averages()
            handler.close()
            raw = json.loads((Path(directory) / "metrics.jsonl").read_text())
            with (Path(directory) / "metrics.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(raw["step"], 200)
            for key in module.REQUIRED_METRICS:
                self.assertEqual(raw[key], source[key])
                self.assertEqual(float(rows[0][key]), source[key])

    def test_disk_failure_propagates(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "temp") as directory:
            handler = module.ExactMetricsHandler(Path(directory), MetricsTracker)
            tracker = MetricsTracker(8, 31968, 90, {
                k: AverageMeter(k) for k in module.REQUIRED_METRICS
            }, initial_step=200)
            handler.jsonl.close()
            with self.assertRaises(ValueError):
                handler.emit(logging.LogRecord("test", logging.INFO, "", 0, tracker, (), None))
            handler.close()


if __name__ == "__main__":
    unittest.main()
