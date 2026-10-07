import copy
import json
import shutil
import subprocess

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from so101.build_dataset import (
    check_sizes,
    clip_bounds,
    main,
    manifest_hash,
    rewrite_indices,
    sha256,
    snapshot,
    source_manifest,
    source_path,
    validate_decisions,
)


def approval(tmp_path):
    manifest = {"dataset": {"data/example": "a" * 64}, "external_sidecars": {}}
    audit = {
        "schema_version": 1,
        "fps": 30,
        "source_manifest_sha256": manifest_hash(manifest),
        "episodes": [
            {"episode_index": 0, "classification": "bad", "close_frame": None, "lift_frame": None},
            {"episode_index": 1, "classification": "clean", "close_frame": 50, "lift_frame": 70},
        ],
    }
    path = tmp_path / "metadata.json"
    path.write_text(json.dumps(audit))
    path.with_name("source_sha256.json").write_text(json.dumps(manifest))
    decisions = {
        "approved": True,
        "main_episodes": [1],
        "grasp_window_episodes": [1],
        "exclusions": {"0": "mandated"},
        "audit_sha256": sha256(path),
        "source_manifest_sha256": manifest_hash(manifest),
    }
    return decisions, path, audit, manifest


def test_approval_bound_to_report_and_forced_exclusions(tmp_path):
    decisions, path, audit, manifest = approval(tmp_path)
    assert validate_decisions(decisions, path, audit, {0, 1}, manifest)[0]["main_episodes"] == [1]
    decisions["main_episodes"] = [0, 1]
    with pytest.raises(ValueError, match="Mandated"):
        validate_decisions(decisions, path, audit, {0, 1}, manifest)
    decisions["main_episodes"] = [1]
    path.write_text("changed")
    with pytest.raises(ValueError, match="does not match"):
        validate_decisions(decisions, path, audit, {0, 1}, manifest)


def test_approval_rejects_changed_source_and_incomplete_audit(tmp_path):
    decisions, path, audit, manifest = approval(tmp_path)
    changed = copy.deepcopy(manifest)
    changed["dataset"]["data/example"] = "b" * 64
    with pytest.raises(ValueError, match="raw source manifest"):
        validate_decisions(decisions, path, audit, {0, 1}, changed)
    audit["episodes"].pop(0)
    path.write_text(json.dumps(audit))
    decisions["audit_sha256"] = sha256(path)
    with pytest.raises(ValueError, match="every source episode"):
        validate_decisions(decisions, path, audit, {0, 1}, manifest)


def test_cannot_build_unapproved_or_unmeasured_window(tmp_path):
    decisions, path, audit, manifest = approval(tmp_path)
    decisions["approved"] = False
    with pytest.raises(ValueError, match="approved"):
        validate_decisions(decisions, path, audit, {0, 1}, manifest)
    decisions["approved"] = True
    audit["episodes"][1]["close_frame"] = None
    path.write_text(json.dumps(audit))
    decisions["audit_sha256"] = sha256(path)
    with pytest.raises(ValueError, match="measured clean"):
        validate_decisions(decisions, path, audit, {0, 1}, manifest)


def test_clip_includes_lift_plus_one_second_and_bounds():
    assert clip_bounds({"close_frame": 100, "lift_frame": 140}, 250) == (55, 171)
    assert clip_bounds({"close_frame": 10, "lift_frame": 90}, 100) == (0, 100)
    with pytest.raises(ValueError):
        clip_bounds({"close_frame": 100, "lift_frame": 90}, 250)


def test_renumbering_preserves_joint_values_and_dtypes():
    source = pa.table(
        {
            "episode_index": [9, 9],
            "index": [50, 51],
            "frame_index": [12, 13],
            "timestamp": pa.array([0.4, 0.4333], type=pa.float32()),
            "action": [[1.0, 2.0], [3.0, 4.0]],
            "observation.state": [[5.0, 6.0], [7.0, 8.0]],
        }
    )
    result = rewrite_indices(source, 0, 20, 30)
    assert result["action"].equals(source["action"])
    assert result["observation.state"].equals(source["observation.state"])
    assert result["index"].to_pylist() == [20, 21]
    assert result["frame_index"].to_pylist() == [0, 1]
    assert result.schema == source.schema
    assert source["episode_index"].to_pylist() == [9, 9]


def test_github_file_limit_is_strict(tmp_path):
    file = tmp_path / "large.bin"
    with file.open("wb") as stream:
        stream.truncate(95_000_000)
    with pytest.raises(RuntimeError, match="STOP"):
        check_sizes(tmp_path)


def test_plain_git_total_limit_includes_every_file(tmp_path):
    for index in range(11):
        with (tmp_path / f"part{index}.bin").open("wb") as stream:
            stream.truncate(94_000_000)
    with pytest.raises(RuntimeError, match="STOP"):
        check_sizes(tmp_path)


def test_source_paths_and_symlinks_cannot_escape(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    target = tmp_path / "outside"
    target.write_text("read only")
    with pytest.raises(ValueError, match="leaves dataset"):
        source_path(raw, "../outside")
    (raw / "link").symlink_to(target)
    with pytest.raises(ValueError, match="symlinks"):
        snapshot(raw)


def synthetic_source(tmp_path):
    """Small recordings with distinct YUV frames; no real raw data is accessed."""
    source = tmp_path / "raw"
    sidecars = source / "pick_place_meta"
    (source / "meta/episodes").mkdir(parents=True)
    sidecars.mkdir()
    frames = 80
    video = tmp_path / "synthetic.mp4"
    pixels = b"".join(
        bytes([16 + index * 2]) * (640 * 480)
        + bytes([80 + index]) * (320 * 240)
        + bytes([170 - index]) * (320 * 240)
        for index in range(frames)
    )
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "yuv420p",
            "-s",
            "640x480",
            "-r",
            "30",
            "-i",
            "pipe:0",
            "-frames:v",
            str(frames),
            "-c:v",
            "libaom-av1",
            "-crf",
            "20",
            "-b:v",
            "0",
            "-cpu-used",
            "8",
            "-threads",
            "1",
            str(video),
        ],
        input=pixels,
        check=True,
    )
    from so101.build_dataset import JOINT_NAMES, VIDEO_KEYS

    features = {
        key: {"dtype": "float32", "shape": [6], "names": JOINT_NAMES}
        for key in ("observation.state", "action")
    }
    features["timestamp"] = {"dtype": "float32", "shape": [1], "names": None}
    for key in ("index", "frame_index", "episode_index", "task_index"):
        features[key] = {"dtype": "int64", "shape": [1], "names": None}
    for key in VIDEO_KEYS:
        features[key] = {
            "dtype": "video",
            "shape": [480, 640, 3],
            "names": ["height", "width", "channels"],
            "info": {
                "video.codec": "av1",
                "video.pix_fmt": "yuv420p",
                "video.fps": 30,
                "video.height": 480,
                "video.width": 640,
                "video.channels": 3,
                "has_audio": False,
            },
        }
    info = {
        "codebase_version": "v3.0",
        "robot_type": "so101_follower",
        "fps": 30,
        "total_episodes": 2,
        "total_frames": frames * 2,
        "chunks_size": 1000,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": features,
    }
    (source / "meta/info.json").write_text(json.dumps(info))
    task = "Synthetic pick and place"
    pq.write_table(
        pa.Table.from_pandas(pd.DataFrame({"task_index": [0]}, index=[task])),
        source / "meta/tasks.parquet",
    )
    records = []
    for ep in range(2):
        values = (np.arange(frames * 6).reshape(frames, 6) / 10 + ep).astype(np.float32)
        table = pa.table(
            {
                "episode_index": np.full(frames, ep),
                "index": np.arange(ep * frames, (ep + 1) * frames),
                "frame_index": np.arange(frames),
                "timestamp": np.arange(frames, dtype=np.float32) / 30,
                "observation.state": pa.array(values.tolist(), type=pa.list_(pa.float32(), 6)),
                "action": pa.array((values + 0.125).tolist(), type=pa.list_(pa.float32(), 6)),
                "task_index": np.zeros(frames, dtype=np.int64),
            }
        )
        dest = source / info["data_path"].format(chunk_index=0, file_index=ep)
        dest.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, dest)
        record = {
            "episode_index": ep,
            "tasks": [task],
            "length": frames,
            "data/chunk_index": 0,
            "data/file_index": ep,
            "dataset_from_index": ep * frames,
            "dataset_to_index": (ep + 1) * frames,
        }
        for key in VIDEO_KEYS:
            for field, value in {
                "chunk_index": 0,
                "file_index": ep,
                "from_timestamp": 0.0,
                "to_timestamp": frames / 30,
            }.items():
                record[f"videos/{key}/{field}"] = value
            dest = source / info["video_path"].format(video_key=key, chunk_index=0, file_index=ep)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(video, dest)
        records.append(record)
        (sidecars / f"episode_{ep:06d}.json").write_text(
            json.dumps(
                {
                    "episode_index": ep,
                    "num_frames": frames,
                    "fps": 30,
                    "units": "m",
                    "reference_frame": "robot base link (base)",
                    "position": [[0.0, 0.0, 0.04]] * frames,
                    "orientation_wxyz": [[1.0, 0.0, 0.0, 0.0]] * frames,
                }
            )
        )
    pq.write_table(pa.Table.from_pylist(records), source / "meta/episodes/episodes.parquet")
    manifest = source_manifest(source, sidecars)
    audit = {
        "schema_version": 1,
        "source": str(source),
        "sidecars": str(sidecars),
        "fps": 30,
        "source_manifest_sha256": manifest_hash(manifest),
        "arm_speed_p25_deg_s": 1.25,
        "thresholds": {"hover_s": 0.5, "roll_alignment_deg": 20.0},
        "percentile_method": "linear",
        "threshold_reference_episodes": [1],
        "min_lift_m": 0.01,
        "lift_reference": "cube at close",
        "global_gripper": {"closed": 0.0, "opened": 100.0, "midpoint": 50.0},
        "episodes": [
            {
                "episode_index": 0,
                "classification": "bad",
                "frames": frames,
                "close_frame": None,
                "lift_frame": None,
            },
            {
                "episode_index": 1,
                "classification": "clean",
                "frames": frames,
                "close_frame": 46,
                "lift_frame": 48,
            },
        ],
    }
    audit_dir = tmp_path / "audit"
    audit_dir.mkdir()
    audit_path = audit_dir / "metadata.json"
    audit_path.write_text(json.dumps(audit))
    (audit_dir / "source_sha256.json").write_text(json.dumps(manifest))
    decisions = {
        "approved": True,
        "main_episodes": [1],
        "grasp_window_episodes": [1],
        "exclusions": {"0": "mandated test exclusion"},
        "audit_sha256": sha256(audit_path),
        "source_manifest_sha256": manifest_hash(manifest),
    }
    decisions_path = audit_dir / "decisions.json"
    decisions_path.write_text(json.dumps(decisions))
    return source, audit_path, decisions_path


def test_synthetic_build_clips_exact_frames_and_reads_both_sets_deterministically(tmp_path):
    source, audit, decisions = synthetic_source(tmp_path)
    before = snapshot(source)
    outputs = [tmp_path / "first", tmp_path / "second"]
    for output in outputs:
        main(
            [
                "--source",
                str(source),
                "--audit",
                str(audit),
                "--decisions",
                str(decisions),
                "--output",
                str(output),
            ]
        )
        report = json.loads((output / "build_report.json").read_text())
        assert report["counts"]["main"]["frames"] == 80
        assert report["counts"]["grasp_window"]["frames"] == 78
        assert all(row["reader_check"]["passed"] for row in report["counts"].values())
        for subset in ("main", "grasp_window"):
            assert (
                json.loads((output / subset / "meta/info.json").read_text())["codebase_version"]
                == "v2.1"
            )
            assert json.loads((output / subset / "meta/relative_stats.json").read_text()) == {
                "__fingerprints__": {}
            }
        window = output / "grasp_window"
        mapping = json.loads((window / "episode_mapping.json").read_text())[0]
        assert mapping == {
            "episode_index": 0,
            "source_episode_index": 1,
            "start_frame": 1,
            "stop_frame_exclusive": 79,
        }
        raw = pq.read_table(source / "data/chunk-000/file-001.parquet").slice(1, 78)
        clipped = pq.read_table(window / "data/chunk-000/episode_000000.parquet")
        assert clipped["action"].equals(raw["action"])
        assert clipped["observation.state"].equals(raw["observation.state"])
        assert clipped["frame_index"].to_pylist() == list(range(78))
        readme = (output / "README.md").read_text()
        assert "mandated test exclusion" in readme and "1.25" in readme and "78" in readme
        assert snapshot(source) == before
    assert snapshot(outputs[0]) == snapshot(outputs[1])


def test_approval_changes_during_build_cannot_publish_dataset(tmp_path, monkeypatch):
    source, audit, decisions = synthetic_source(tmp_path)
    before = snapshot(source)

    def changed_approval(source, dest, *args):
        dest.mkdir(parents=True)
        audit.write_text("audit changed during the build")
        return {"episodes": 1, "frames": 80, "mapping": []}

    monkeypatch.setattr("so101.build_dataset.build_subset", changed_approval)
    output = tmp_path / "unpublished"
    with pytest.raises(ValueError, match="does not match this audit"):
        main(
            [
                "--source",
                str(source),
                "--audit",
                str(audit),
                "--decisions",
                str(decisions),
                "--output",
                str(output),
            ]
        )
    assert not output.exists()
    assert snapshot(source) == before
