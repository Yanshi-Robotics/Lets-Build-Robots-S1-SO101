"""CPU-only checks for real metric provenance, fragments and log parsing."""
import csv
import importlib.util
import json
import os
from pathlib import Path
import struct
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
TASK_TEMP = ROOT / 'temp/ep13-curve-20261014'
TASK_TEMP.mkdir(parents=True, exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR', str(TASK_TEMP / 'matplotlib'))
spec = importlib.util.spec_from_file_location('training_curve', ROOT / 'ACT-1-Pick/plot_training_curve.py')
curve = importlib.util.module_from_spec(spec)
spec.loader.exec_module(curve)


class TrainingCurveTest(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory(dir=TASK_TEMP)
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name)

    def test_real_example_has_five_records_and_two_fragments(self):
        rows, approximate = curve.read_metrics(ROOT / 'ACT-1-Pick/examples/exp1-v1/training_metrics.csv')
        provenance = json.loads((ROOT / 'ACT-1-Pick/examples/exp1-v1/provenance.json').read_text())
        self.assertFalse(approximate)
        self.assertFalse(provenance['continuous_history'])
        self.assertEqual([row['step'] for row in rows], [1400, 3000, 5000, 99800, 100000])
        self.assertEqual(rows[0]['loss'], 1.657)
        self.assertEqual(rows[-1]['loss'], 0.052)
        self.assertEqual([len(section) for section in curve.metric_sections(rows, 10000)], [3, 2])

    def test_console_abbreviations_stay_approximate(self):
        log = self.path / 'train.log'
        log.write_text('\x1b[32mINFO step:1K smpl:11K loss:1.657 l1_loss:0.346 kld_loss:0.131\x1b[0m\n')
        rows, approximate = curve.read_metrics(log)
        self.assertTrue(approximate)
        self.assertEqual(rows[0]['step'], 1000)  # Never invent the precise 1400 from a rounded label.

    def test_progress_lines_and_resumed_records(self):
        log = self.path / 'train.log'
        log.write_text('Training: 98%| 980/1000\rINFO step:1000 smpl:8K loss:0.3 l1_loss:0.2 kld_loss:0.01\nINFO step:800 smpl:6K loss:0.4 l1_loss:0.3 kld_loss:0.01\n')
        rows, approximate = curve.read_metrics(log)
        self.assertFalse(approximate)
        self.assertEqual(len(curve.metric_sections(rows, 10000)), 2)

    def test_utf16_and_utf8_bom_logs(self):
        for encoding in ['utf-16', 'utf-8-sig']:
            log = self.path / (encoding + '.log')
            log.write_text('INFO step:200 smpl:1600 loss:0.3 l1_loss:0.2 kld_loss:0.01\n', encoding=encoding)
            rows, approximate = curve.read_metrics(log)
            self.assertEqual(rows[0]['step'], 200)
            self.assertEqual(rows[0]['loss'], 0.3)
            self.assertFalse(approximate)

    def test_nonfinite_metrics_are_not_hidden(self):
        log = self.path / 'train.log'
        log.write_text('INFO step:200 smpl:1600 loss:nan l1_loss:0.2 kld_loss:0.01\n')
        with self.assertRaisesRegex(ValueError, 'Nonfinite'):
            curve.read_metrics(log)

    def test_missing_metrics_cannot_be_reconstructed(self):
        log = self.path / 'train.log'
        log.write_text('INFO Checkpoint policy after step 100000\nINFO End of training\n')
        with self.assertRaisesRegex(ValueError, 'No ACT loss records'):
            curve.read_metrics(log)

    def test_csv_requires_named_metrics(self):
        table = self.path / 'metrics.csv'
        table.write_text('step,loss\n1000,0.2\n')
        with self.assertRaisesRegex(ValueError, 'CSV requires'):
            curve.read_metrics(table)

    def test_plot_is_a_real_png(self):
        rows, approximate = curve.read_metrics(ROOT / 'ACT-1-Pick/examples/exp1-v1/training_metrics.csv')
        output = self.path / 'curve.png'
        curve.plot_metrics(rows, approximate, output, 10000)
        payload = output.read_bytes()
        self.assertEqual(payload[:8], b'\x89PNG\r\n\x1a\n')
        width, height = struct.unpack('>II', payload[16:24])
        self.assertGreater(width, 1000)
        self.assertGreater(height, 500)


if __name__ == '__main__':
    unittest.main()
