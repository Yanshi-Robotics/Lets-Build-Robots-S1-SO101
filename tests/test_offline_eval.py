"""CPU-only regression for the evaluator's observation path. Fixtures are not robot evidence.

Run: .venv/bin/python tests/test_offline_eval.py
"""
import importlib.util
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('offline_eval', ROOT / 'ACT-1-Pick/offline_eval.py')
evaluator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluator)


class ObservationPathTests(unittest.TestCase):
    def test_recorded_state_is_used_and_action_labels_never_initialize_feedback(self, prediction=10.0, wrong_shape=False, feedback=10.0):
        import torch
        seen_open, seen_feedback = [], []
        # Deliberately different state and action labels expose substitution or label leakage.
        items = [dict(action=torch.tensor([80.0]), **{'observation.state': torch.tensor([float(t + 2)])}) for t in range(3)]
        meta = SimpleNamespace(camera_keys=[], robot_type='fixture', stats={}, info=SimpleNamespace(features={'action': {'names': ['gripper.pos']}}))
        dataset = SimpleNamespace(meta=meta, hf_dataset={'episode_index': [0, 0, 0]})

        class Dataset:
            def __init__(self, *args, **kwargs):
                self.meta, self.hf_dataset = dataset.meta, dataset.hf_dataset
            def __getitem__(self, index):
                return items[index]

        class Policy:
            config = SimpleNamespace(chunk_size=1, n_action_steps=1)
            def eval(self): pass
            def reset(self): pass
            def predict_action_chunk(self, obs):
                seen_open.append(obs['observation.state'].item())
                return torch.tensor([[[prediction], [prediction]]]) if wrong_shape else torch.tensor([[[prediction]]])
            def select_action(self, obs):
                seen_feedback.append(obs['observation.state'].item())
                return torch.tensor([[feedback]])

        modules = {name: ModuleType(name) for name in ['lerobot.configs.policies', 'lerobot.datasets.lerobot_dataset', 'lerobot.policies.factory']}
        modules['lerobot.configs.policies'].PreTrainedConfig = SimpleNamespace(from_pretrained=lambda _: SimpleNamespace())
        modules['lerobot.datasets.lerobot_dataset'].LeRobotDataset = Dataset
        modules['lerobot.policies.factory'].make_policy = lambda *a, **kw: Policy()
        modules['lerobot.policies.factory'].make_pre_post_processors = lambda **kw: (lambda x: x, lambda x: x)
        args = SimpleNamespace(episodes=[0], dataset_repo_id='fixture', dataset_root=Path('fixture'), action_units='degrees', model_dir=None, max_frames_per_episode=0)
        with patch.dict(sys.modules, modules):
            result = evaluator.evaluate_one(Path('fixture'), args, torch.device('cpu'), {})
        self.assertEqual(seen_open, [2.0, 3.0, 4.0])
        expected_second = 2.0 + evaluator.FOLLOW_ALPHA * (10.0 - 2.0)
        self.assertAlmostEqual(seen_feedback[0], 2.0)
        self.assertAlmostEqual(seen_feedback[1], expected_second, places=5)
        self.assertNotIn(80.0, seen_open + seen_feedback)
        self.assertEqual(result['open_loop']['per_joint_mean_abs'], {'gripper': 70.0})
        self.assertEqual(result['frames_scored'], 3)

    def test_nonfinite_feedback_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Feedback prediction'):
            self.test_recorded_state_is_used_and_action_labels_never_initialize_feedback(feedback=float('nan'))

    def test_wrong_prediction_shape_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Prediction shape'):
            self.test_recorded_state_is_used_and_action_labels_never_initialize_feedback(wrong_shape=True)

    def test_nonfinite_prediction_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'non-finite'):
            self.test_recorded_state_is_used_and_action_labels_never_initialize_feedback(float('nan'))


if __name__ == '__main__':
    unittest.main()
