"""CPU checks of committed-byte integrity and actual reader validation boundaries."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from so101.setup_checks import (
    ROOT,
    SETUP_SCRATCH_MARKER,
    check_subset,
    check_versions,
    cleanup_setup_checkpoints,
    verify_prepared_manifest,
)


class ScratchCleanupTest(unittest.TestCase):
    def fixture(self, temp):
        scratch = Path(temp) / "so101-setup-dry-run-abcdefgh"
        stage = scratch / "dry_run/stage_1_align"
        checkpoint = stage / "checkpoint-2"
        checkpoint.mkdir(parents=True)
        (checkpoint / "weights.safetensors").write_bytes(b"generated setup weights")
        (stage / "train.log").write_text("verification log")
        (scratch / ".so101-setup-verification").write_text(SETUP_SCRATCH_MARKER)
        return scratch, checkpoint

    def test_only_owned_setup_checkpoint_removed_and_logs_retained(self):
        with tempfile.TemporaryDirectory() as temp:
            scratch, checkpoint = self.fixture(temp)
            foundation = Path(temp) / "hf-cache/foundation.safetensors"
            foundation.parent.mkdir()
            foundation.write_bytes(b"never delete foundation")
            training = Path(temp) / "user-training/checkpoint-25000"
            training.mkdir(parents=True)
            self.assertEqual(cleanup_setup_checkpoints(scratch), 1)
            self.assertFalse(checkpoint.exists())
            self.assertEqual((checkpoint.parent / "train.log").read_text(), "verification log")
            self.assertEqual(foundation.read_bytes(), b"never delete foundation")
            self.assertTrue(training.is_dir())
            self.assertEqual(cleanup_setup_checkpoints(scratch), 0)

    def test_regular_outputs_or_missing_marker_are_never_removed(self):
        with tempfile.TemporaryDirectory() as temp:
            scratch, checkpoint = self.fixture(temp)
            (scratch / ".so101-setup-verification").unlink()
            with self.assertRaisesRegex(RuntimeError, "marker"):
                cleanup_setup_checkpoints(scratch)
            self.assertTrue(checkpoint.is_dir())
            ordinary = Path(temp) / "user-training"
            scratch.rename(ordinary)
            with self.assertRaisesRegex(RuntimeError, "mktemp"):
                cleanup_setup_checkpoints(ordinary)
            self.assertTrue((ordinary / "dry_run/stage_1_align/checkpoint-2").is_dir())

    def test_checkpoint_symlink_rejects_all_cleanup_before_mutating_any_path(self):
        with tempfile.TemporaryDirectory() as temp:
            scratch, checkpoint = self.fixture(temp)
            cache = Path(temp) / "hf-cache"
            cache.mkdir()
            (cache / "foundation.safetensors").write_bytes(b"protected")
            (checkpoint.parent / "checkpoint-3").symlink_to(cache, target_is_directory=True)
            with self.assertRaisesRegex(RuntimeError, "Unsafe"):
                cleanup_setup_checkpoints(scratch)
            self.assertTrue(checkpoint.is_dir())
            self.assertEqual((cache / "foundation.safetensors").read_bytes(), b"protected")


class ManifestTest(unittest.TestCase):
    def fixture(self, root):
        root = Path(root)
        (root / "main").mkdir()
        (root / "main/data.txt").write_bytes(b"committed dataset bytes")
        manifest = {
            "main/data.txt": hashlib.sha256((root / "main/data.txt").read_bytes()).hexdigest()
        }
        (root / "prepared_sha256.json").write_text(json.dumps(manifest))
        return root

    def test_committed_dataset_bytes_verify_and_tampering_aborts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = self.fixture(temp)
            self.assertEqual(verify_prepared_manifest(root), 1)
            (root / "main/data.txt").write_bytes(b"changed")
            with self.assertRaisesRegex(RuntimeError, "file changed"):
                verify_prepared_manifest(root)

    def test_added_missing_and_symlink_files_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = self.fixture(temp)
            extra = root / "added.txt"
            extra.write_text("extra")
            with self.assertRaisesRegex(RuntimeError, "files differ"):
                verify_prepared_manifest(root)
            extra.unlink()
            original = root / "main/data.txt"
            original.unlink()
            with self.assertRaisesRegex(RuntimeError, "files differ"):
                verify_prepared_manifest(root)
            original.symlink_to(root / "prepared_sha256.json")
            with self.assertRaisesRegex(RuntimeError, "symlinks"):
                verify_prepared_manifest(root)

    def test_unbuilt_dataset_requires_phase3(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(RuntimeError, "Phase 3"):
                verify_prepared_manifest(temp)


class ReaderTest(unittest.TestCase):
    def fixture(self, root):
        root = Path(root)
        (root / "meta").mkdir()
        (root / "meta/info.json").write_text(
            json.dumps({"total_episodes": 2, "total_frames": 32, "fps": 30})
        )
        modality = (ROOT / "so101/provenance/modality.json").read_text()
        (root / "meta/modality.json").write_text(modality)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        values = {
            "video.room": [frame] * 16,
            "video.wrist": [frame] * 16,
            "language.annotation.human.task_description": ["pick and place"] * 16,
        }
        for modality in ("state", "action"):
            values[f"{modality}.single_arm"] = [np.zeros(5)] * 16
            values[f"{modality}.gripper"] = [np.zeros(1)] * 16
        episodes = [pd.DataFrame(values), pd.DataFrame(values)]

        class Reader:
            episode_lengths = [16, 16]

            def __len__(self):
                return 2

            def __getitem__(self, episode):
                return episodes[episode]

        return root, Reader(), episodes

    def test_both_cameras_language_and_six_joint_reader_outputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root, loader, _ = self.fixture(temp)
            self.assertIn("2 episodes / 32 frames", check_subset(root, lambda _: loader))

    def test_invalid_last_camera_gripper_and_missing_language_abort(self):
        with tempfile.TemporaryDirectory() as temp:
            root, loader, episodes = self.fixture(temp)
            original = episodes[1].copy()
            episodes[1].at[0, "video.wrist"] = np.zeros((10, 10, 3))
            with self.assertRaisesRegex(RuntimeError, "Camera"):
                check_subset(root, lambda _: loader)
            episodes[1] = original.copy()
            episodes[1]["action.gripper"] = [np.zeros(2)] * 16
            with self.assertRaisesRegex(RuntimeError, "joint"):
                check_subset(root, lambda _: loader)
            episodes[1] = original.copy()
            episodes[1].at[0, "language.annotation.human.task_description"] = ""
            with self.assertRaisesRegex(RuntimeError, "instructions"):
                check_subset(root, lambda _: loader)

    def test_modality_or_reader_total_drift_aborts(self):
        with tempfile.TemporaryDirectory() as temp:
            root, loader, _ = self.fixture(temp)
            loader.episode_lengths = [16, 15]
            with self.assertRaisesRegex(RuntimeError, "totals"):
                check_subset(root, lambda _: loader)
            (root / "meta/modality.json").write_text("{}")
            with self.assertRaisesRegex(RuntimeError, "mapping"):
                check_subset(root, lambda _: loader)

    def test_version_mismatch_aborts_before_importing_gpu_runtime(self):
        with patch("so101.setup_checks.platform.python_version", return_value="0.0.0"):
            with self.assertRaisesRegex(RuntimeError, "python: expected"):
                check_versions()


if __name__ == "__main__":
    unittest.main()
