"""Regression tests for the trajectory split and frozen blind dataset loader."""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.io import savemat
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"tools"))
from train_generalization import check_disjoint, read_dataset


class BlindProtocolTests(unittest.TestCase):
    protocol = {
        "regular_trajectories": 1, "combined_trajectories": 1,
        "regular_nonzero_steps": 2, "combined_nonzero_steps": 2,
        "reference_residual_limit_N": 2.1e-8,
    }
    device = torch.device("cpu")

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)

    def save_sample(self, name, token="frozen", force=(0, 0.1, 0.3), sag=0.2):
        spec = dict(id=name, axes=[4, 3, 4], power=2, yaw_deg=12, tilt_deg=8,
                    center_mm=[1, 0], depths_mm=[0.12, 0.32])
        heights = np.full((201, 201, 3), 5, dtype=np.float32)
        heights[50:60, 50:60, 1:] -= [sag, 2*sag]
        diagnostics = np.zeros((3, 8))
        diagnostics[:, 4] = force
        diagnostics[:, 6] = 1
        contents = dict(spec=spec, heights=heights, travel=[0, 0.12, 0.32],
                        force=force, diagnostics=diagnostics, seconds=1)
        if token is not None:
            contents["freezeToken"] = token
        savemat(self.folder/f"{name}.mat", contents)

    def blind_batch(self):
        self.save_sample("blind_test_01", sag=0.1)
        self.save_sample("blind_combined_01", sag=0.3)

    def test_training_ignores_legacy_test_files_and_rejects_blind_files(self):
        self.save_sample("train_01", token=None)
        (self.folder/"combined_01.mat").touch()
        groups, _ = read_dataset([self.folder], self.device, self.protocol)
        self.assertEqual(set(groups), {"train"})
        self.blind_batch()
        with self.assertRaisesRegex(ValueError, "cannot be supplied"):
            read_dataset([self.folder], self.device, self.protocol)

    def test_blind_scoring_checks_complete_batch_and_token(self):
        self.save_sample("blind_test_01")
        with self.assertRaisesRegex(ValueError, "trajectory list"):
            read_dataset([self.folder], self.device, self.protocol, "frozen")
        self.blind_batch()
        with self.assertRaisesRegex(ValueError, "not generated after"):
            read_dataset([self.folder], self.device, self.protocol, "different")
        groups, records = read_dataset([self.folder], self.device, self.protocol, "frozen")
        np.testing.assert_array_equal(groups["test"][2], [0.12, 0.32])
        self.assertEqual(sum(row["frames"] for row in records), 4)
        self.save_sample("blind_test_01", force=(0, 0, 0.3))
        with self.assertRaisesRegex(ValueError, "failed the complete-trajectory"):
            read_dataset([self.folder], self.device, self.protocol, "frozen")

    def test_overlap_checks_names_pose_and_measured_frames(self):
        old = dict(trajectory="train_01", tool_pose=[2, 4], frame_sha256=["same"])
        new = dict(trajectory="blind_test_01", tool_pose=[3, 5], frame_sha256=["different"])
        check_disjoint([old], [new])
        for field in old:
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "splits share"):
                check_disjoint([old], [{**new, field: old[field]}])


if __name__ == "__main__":
    unittest.main()
