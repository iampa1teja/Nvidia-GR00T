"""CPU regressions for measurement boundaries, ambiguous thresholds and raw immutability."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from so101.audit_grasps import (
    CALIBRATED_LIMITS_DEG,
    FORCED_BAD,
    JOINTS,
    METRICS,
    calibrated_state_degrees,
    classify,
    cube_face_yaw,
    episode_metrics,
    read_dataset,
    run_audit,
    snapshot,
    square_angle,
)


def fixture():
    n = 180
    actions = np.zeros((n, 6))
    actions[:, 5] = 100
    actions[60:67, 5] = np.linspace(100, 0, 7)
    actions[67:, 5] = 0
    joints = np.zeros((n, 6))
    joints[:, 1] = np.arange(n) * 0.1
    positions = np.zeros((n, 3))
    positions[90:, 2] = 0.02
    sidecar = {
        "position": positions,
        "orientation_wxyz": np.tile([1, 0, 0, 0], (n, 1)),
    }
    return actions, joints, sidecar


def write_raw(root, count=5):
    """Minimal real parquet/sidecar fixture with complete timestamps and stats."""
    root = Path(root)
    (root / "meta").mkdir(parents=True)
    (root / "data/chunk-000").mkdir(parents=True)
    (root / "pick_place_meta").mkdir()
    features = {
        key: {"names": [j + ".pos" for j in JOINTS]} for key in ("action", "observation.state")
    }
    info = {"fps": 30, "total_episodes": count, "total_frames": count * 180, "features": features}
    (root / "meta/info.json").write_text(json.dumps(info))
    (root / "meta/stats.json").write_text(
        json.dumps({"action": {"min": [0] * 6, "max": [100] * 6}})
    )
    for episode in range(count):
        actions, states, sidecar = fixture()
        n = len(actions)
        frames = [
            {
                "episode_index": episode,
                "frame_index": i,
                "timestamp": i / 30,
                "action": actions[i].tolist(),
                "observation.state": states[i].tolist(),
            }
            for i in range(n)
        ]
        pq.write_table(
            pa.Table.from_pylist(frames), root / f"data/chunk-000/file-{episode:03d}.parquet"
        )
        sidecar = {key: val.tolist() for key, val in sidecar.items()}
        sidecar.update(
            episode_index=episode,
            num_frames=n,
            fps=30,
            reference_frame="robot base link (base)",
            units="m",
        )
        (root / f"pick_place_meta/episode_{episode:06d}.json").write_text(json.dumps(sidecar))
    (root / "videos").mkdir()
    (root / "videos/placeholder.mp4").write_bytes(b"raw-video-unchanged\x00")
    return root


class MeasurementsTest(unittest.TestCase):
    def measure(self, data, closed=0, opened=100):
        return episode_metrics(*data, closed, opened, 1)

    def test_clean_sustained_close_and_all_requested_metrics(self):
        row = self.measure(fixture())
        self.assertEqual(row["close_frame"], 64)
        self.assertEqual(row["close_duration_frames"], 6)
        self.assertEqual(row["lift_frame"], 90)
        self.assertTrue(row["lift_ok"])
        self.assertEqual(
            {k: row[k] for k in METRICS},
            {
                "hover_s": 0.0,
                "close_attempts": 0,
                "dither": 0,
                "close_duration_frames": 6,
                "roll_alignment_deg": 0.0,
            },
        )

    def test_hover_and_joint_direction_reversals(self):
        data = fixture()
        data[1][10:40, 1] = 1
        data[1][45:60, 2] = np.arange(15) % 2
        row = self.measure(data)
        self.assertAlmostEqual(row["hover_s"], 29 / 30)
        self.assertEqual(row["dither"], 13)

    def test_short_attempts_initial_close_and_exact_ten_frame_boundary(self):
        data = fixture()
        data[0][:5, 5] = 0  # Initial opening must not count as an attempted close.
        data[0][20:29, 5] = 0
        row = self.measure(data)
        self.assertEqual(row["close_frame"], 64)
        self.assertEqual(row["close_attempts"], 1)
        data[0][20:30, 5] = 0
        self.assertEqual(self.measure(data)["close_frame"], 20)

    def test_midpoint_is_strict_and_min_lift_is_inclusive(self):
        data = fixture()
        data[0][60:70, 5] = 50
        data[0][70:, 5] = 0
        data[2]["position"][90:, 2] = 0.01
        row = self.measure(data)
        self.assertEqual(row["close_frame"], 70)
        self.assertEqual(row["lift_frame"], 90)

    def test_lift_must_be_new_and_later_than_close(self):
        data = fixture()
        data[2]["position"][10:, 2] = 0.02
        row = self.measure(data)
        self.assertFalse(row["lift_ok"])
        self.assertAlmostEqual(row["cube_rise_anywhere_m"], 0.02)
        data[2]["position"][90:, 2] = 0.04
        self.assertEqual(self.measure(data)["lift_frame"], 90)

    def test_duration_missing_is_not_zero_and_reopen_invalidates_transition(self):
        data = fixture()
        data[0][64:100, 5] = 40
        data[0][100:110, 5] = 100
        row = self.measure(data)
        self.assertIsNone(row["close_duration_frames"])
        data[0][:, 5] *= 0.4
        row = self.measure(data)
        self.assertIsNone(row["close_frame"])
        self.assertIsNone(row["hover_s"])
        self.assertFalse(row["reaches_midpoint"])

    def test_geometry_square_symmetry_and_vertical_axis(self):
        data = fixture()
        data[1][:, 4] = 95
        self.assertAlmostEqual(self.measure(data)["roll_alignment_deg"], 5)
        self.assertEqual(square_angle(45), -45)
        self.assertEqual(square_angle(90), 0)
        # 90 degrees about Y: local X is vertical; choose local Y rather than unstable Euler yaw.
        self.assertAlmostEqual(abs(square_angle(cube_face_yaw([1, 0, 1, 0]))), 0)
        with self.assertRaises(ValueError):
            cube_face_yaw([0, 0, 0, 0])

    def test_calibration_matches_recorded_mapping_and_clamps(self):
        states = np.array([[-100] * 5 + [0], [0] * 5 + [50], [100] * 6], dtype=float)
        result = calibrated_state_degrees(states)
        self.assertEqual(result[0, 0], -110)
        self.assertEqual(result[2, 0], 110)
        self.assertEqual(result[1, 0], np.mean(CALIBRATED_LIMITS_DEG[0]))
        self.assertEqual(result[1, 5], 45)

    def test_invalid_timelines_and_thresholds_abort(self):
        data = fixture()
        data[2]["position"] = data[2]["position"][:-1]
        with self.assertRaises(ValueError):
            self.measure(data)
        with self.assertRaises(ValueError):
            self.measure(fixture(), closed=100, opened=0)
        with self.assertRaises(ValueError):
            episode_metrics(*fixture(), 0, 100, 1, fps=20)


class ClassificationTest(unittest.TestCase):
    def row(self, index):
        return {"episode": index, **episode_metrics(*fixture(), 0, 100, 1)}

    def test_reference_excludes_forced_failures_and_p75_ties_are_clean(self):
        rows = [self.row(i) for i in (1, 2, 4, 5)]
        forced = self.row(72)
        forced["hover_s"] = 1000
        rows.append(forced)
        rows[-2]["hover_s"] = 4
        thresholds = classify(rows)
        self.assertEqual(thresholds["hover_s"], 1.0)
        self.assertEqual(rows[-2]["classification"], "hesitant")
        self.assertEqual(forced["classification"], "bad")
        self.assertFalse(forced["provisional_bad"])
        tied = self.row(7)
        tied["hover_s"] = 1
        thresholds = classify([tied] * 4)
        self.assertEqual(tied["classification"], "clean")
        self.assertEqual(thresholds["hover_s"], 1)

    def test_all_mandated_bad_and_global_threshold_ambiguity(self):
        rows = [self.row(i) for i in sorted(FORCED_BAD)]
        data = fixture()
        data[0][:, 5] *= 0.4
        ambiguous = {"episode": 8, **episode_metrics(*data, 0, 100, 1)}
        rows.extend([ambiguous, self.row(1)])
        classify(rows)
        self.assertTrue(all(r["classification"] == "bad" for r in rows[:-1]))
        self.assertTrue(ambiguous["provisional_bad"])
        self.assertEqual(ambiguous["measurement_status"], "threshold_unresolved")
        self.assertIsNone(ambiguous["hover_s"])

    def test_incomplete_duration_only_flags_when_full_open_was_observed(self):
        absent = self.row(7)
        absent.update(reaches_90pct_open=False, close_duration_frames=None)
        observed = self.row(8)
        observed["close_duration_frames"] = None
        classify([self.row(1), absent, observed])
        self.assertEqual(absent["classification"], "clean")
        self.assertEqual(observed["classification"], "hesitant")

    def test_empty_reference_aborts(self):
        with self.assertRaises(ValueError):
            classify([self.row(0), self.row(3)])


class DatasetTest(unittest.TestCase):
    def test_report_machine_metadata_and_every_raw_hash_are_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            root = write_raw(Path(temp) / "raw")
            before = snapshot(root)
            out = Path(temp) / "audit"
            metadata = run_audit(root, out)
            self.assertEqual(before, snapshot(root))
            self.assertTrue(metadata["source_unchanged"])
            self.assertFalse(metadata["dataset_built"])
            self.assertTrue(metadata["approval_required"])
            self.assertEqual(metadata["categories"]["bad"], [0, 3])
            self.assertEqual(sorted(metadata["proposed_exclusions"]), ["0", "3"])
            self.assertEqual(len(metadata["episodes"]), 5)
            self.assertEqual(metadata["percentile_method"], "linear")
            for name in (
                "audit.csv",
                "summary.md",
                "metadata.json",
                "source_sha256.json",
                "distributions.png",
                "gripper_thresholds.png",
            ):
                self.assertGreater((out / name).stat().st_size, 0)
            manifest = json.loads((out / "source_sha256.json").read_text())
            self.assertIn("videos/placeholder.mp4", manifest["dataset"])
            self.assertEqual(manifest["dataset"], before)
            self.assertFalse((Path(temp) / "datasets").exists())
            # All source and threshold-derived report values reproduce exactly.
            again = run_audit(root, Path(temp) / "audit2")
            self.assertEqual(metadata, again)
            with self.assertRaises(ValueError):
                run_audit(root, out)
            with self.assertRaises(ValueError):
                run_audit(root, root / "audit")

    def test_bad_timestamp_joint_names_duplicate_and_missing_frames_abort(self):
        with tempfile.TemporaryDirectory() as temp:
            root = write_raw(Path(temp) / "raw")
            path = root / "data/chunk-000/file-000.parquet"
            frames = pq.read_table(path).to_pylist()
            frames[1]["timestamp"] = 1
            pq.write_table(pa.Table.from_pylist(frames), path)
            with self.assertRaisesRegex(ValueError, "timestamps"):
                read_dataset(root)
            frames[1]["timestamp"] = 1 / 30
            pq.write_table(pa.Table.from_pylist(frames + [frames[0]]), path)
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                read_dataset(root)
            pq.write_table(pa.Table.from_pylist(frames[:-1]), path)
            with self.assertRaisesRegex(ValueError, "Missing"):
                read_dataset(root)
            pq.write_table(pa.Table.from_pylist(frames), path)
            info_path = root / "meta/info.json"
            info = json.loads(info_path.read_text())
            info["features"]["action"]["names"].reverse()
            info_path.write_text(json.dumps(info))
            with self.assertRaisesRegex(ValueError, "ordering"):
                read_dataset(root)

    def test_wrong_stats_and_extra_sidecar_abort_without_report(self):
        with tempfile.TemporaryDirectory() as temp:
            root = write_raw(Path(temp) / "raw")
            stats_path = root / "meta/stats.json"
            stats = json.loads(stats_path.read_text())
            stats["action"]["max"][5] = 200
            stats_path.write_text(json.dumps(stats))
            out = Path(temp) / "audit"
            with self.assertRaisesRegex(ValueError, "extrema disagree"):
                run_audit(root, out)
            self.assertFalse(out.exists())
            (root / "pick_place_meta/episode_000099.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "Extra sidecars"):
                read_dataset(root)


if __name__ == "__main__":
    unittest.main()
