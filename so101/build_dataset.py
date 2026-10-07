"""Deterministically build approved SO-101 GR00T datasets without modifying raw data.

Requires an explicit decisions JSON with approved=true, main_episodes,
grasp_window_episodes, exclusions, audit_sha256 (metadata.json SHA256), and
source_manifest_sha256 (the audited raw dataset and sidecars).
The approval file is created only after the user reviews the audit.
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
FORCED_BAD = {0, 3, 6, 51, 72}
DATA_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
VIDEO_PATH = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
VIDEO_KEYS = {"observation.images.realsense_rgb", "observation.images.wrist_cam"}
JOINT_NAMES = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def snapshot(root):
    root = Path(root)
    if not root.is_dir():
        raise ValueError(f"Source directory is missing: {root}")
    files = sorted(Path(root).rglob("*"))
    if any(p.is_symlink() for p in files):
        raise ValueError("Dataset must contain ordinary files, not symlinks")
    return {str(p.relative_to(root)): sha256(p) for p in files if p.is_file()}


def source_manifest(source, sidecars):
    """Match the audit's content identity; paths do not enter the aggregate."""
    if not sidecars.is_dir():
        raise ValueError(f"Source sidecars directory is missing: {sidecars}")
    return {
        "dataset": snapshot(source),
        "external_sidecars": snapshot(sidecars) if not sidecars.is_relative_to(source) else {},
    }


def manifest_hash(manifest):
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def source_path(source, relative):
    """Metadata may only select ordinary files inside the raw source."""
    path = source / relative
    if Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ValueError(f"Source metadata path leaves dataset: {relative}")
    if not path.resolve().is_relative_to(source.resolve()) or not path.is_file():
        raise ValueError(f"Source metadata path is missing or leaves dataset: {relative}")
    return path


def write_json(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def load_records(source):
    records = []
    for path in sorted((source / "meta/episodes").rglob("*.parquet")):
        records.extend(pq.read_table(path).to_pylist())
    records.sort(key=lambda row: row["episode_index"])
    return records


def validate_decisions(decisions, audit_path, audit, episode_ids, manifest):
    if decisions.get("approved") is not True:
        raise ValueError("Explicit user-approved episode decisions are required")
    audit_bytes = Path(audit_path).read_bytes()
    if decisions.get("audit_sha256") != hashlib.sha256(audit_bytes).hexdigest():
        raise ValueError("Approval does not match this audit metadata")
    if audit != json.loads(audit_bytes):
        raise ValueError("Audit metadata changed while loading")
    if audit.get("schema_version") != 1 or audit.get("fps") != 30:
        raise ValueError("Unsupported audit schema or frame rate")
    actual_hash = manifest_hash(manifest)
    if (
        decisions.get("source_manifest_sha256") != actual_hash
        or audit.get("source_manifest_sha256") != actual_hash
    ):
        raise ValueError("Approval or audit does not match the raw source manifest")
    audit_manifest = Path(audit_path).with_name("source_sha256.json")
    if not audit_manifest.is_file() or json.loads(audit_manifest.read_text()) != manifest:
        raise ValueError("Audit source manifest is missing or differs from the raw source")
    selections = {}
    for name in ("main_episodes", "grasp_window_episodes"):
        ids = decisions.get(name)
        if not isinstance(ids, list) or not ids or any(type(i) is not int for i in ids):
            raise ValueError(f"{name} must be a nonempty integer list")
        if len(set(ids)) != len(ids) or not set(ids) <= episode_ids:
            raise ValueError(f"Duplicate or unknown episodes in {name}")
        if set(ids) & FORCED_BAD:
            raise ValueError("Mandated bad episodes cannot enter training data")
        selections[name] = sorted(ids)
    if not set(selections["grasp_window_episodes"]) <= set(selections["main_episodes"]):
        raise ValueError("Grasp-window episodes must also belong to main")
    rows = {}
    for row in audit.get("episodes", []):
        ep = row.get("episode_index", row.get("episode"))
        if type(ep) is not int or ep in rows:
            raise ValueError("Audit must contain unique integer episode indices")
        if "episode" in row and row["episode"] != ep:
            raise ValueError("Conflicting audit episode aliases")
        rows[ep] = row
    if set(rows) != episode_ids:
        raise ValueError("Audit must cover every source episode exactly once")
    for ep in selections["grasp_window_episodes"]:
        row = rows[ep]
        if (
            row["classification"] != "clean"
            or row["close_frame"] is None
            or row["lift_frame"] is None
        ):
            raise ValueError(f"Grasp window requires measured clean close and lift: {ep}")
    exclusions = decisions.get("exclusions", {})
    excluded = episode_ids - set(selections["main_episodes"])
    if (
        not isinstance(exclusions, dict)
        or set(exclusions) != {str(ep) for ep in excluded}
        or not all(isinstance(reason, str) and reason.strip() for reason in exclusions.values())
    ):
        raise ValueError("Every excluded episode needs an explicit reason")
    return selections, rows


def clip_bounds(row, length, fps=30):
    close, lift = row["close_frame"], row["lift_frame"]
    if (
        type(close) is not int
        or type(lift) is not int
        or not 0 <= close <= lift < length
        or fps != 30
    ):
        raise ValueError("Invalid grasp-window boundaries")
    start = max(0, close - round(1.5 * fps))
    stop = min(length, lift + round(1.0 * fps) + 1)
    if stop - start < 16:
        raise ValueError("Grasp window is shorter than the 16-action horizon")
    return start, stop


def rewrite_indices(table, episode, offset, fps):
    n = table.num_rows
    arrays = {
        "episode_index": np.full(n, episode),
        "index": np.arange(offset, offset + n),
        "frame_index": np.arange(n),
        "timestamp": np.arange(n) / fps,
    }
    for name, values in arrays.items():
        field = table.schema.field(name)
        table = table.set_column(
            table.schema.get_field_index(name), field, pa.array(values, type=field.type)
        )
    return table


def numeric_stats(table):
    result = {}
    for name in table.column_names:
        values = np.asarray(table[name].to_pylist())
        if values.dtype.kind not in "fiu":
            continue
        if values.ndim == 1:
            values = values[:, None]
        if not np.isfinite(values).all():
            raise ValueError(f"Nonfinite data in {name}")
        result[name] = {
            key: fn(values, axis=0).tolist()
            for key, fn in (("min", np.min), ("max", np.max), ("mean", np.mean), ("std", np.std))
        }
        result[name].update(
            count=[len(values)],
            q01=np.quantile(values, 0.01, axis=0).tolist(),
            q99=np.quantile(values, 0.99, axis=0).tolist(),
        )
    return result


def encoder_versions():
    """Read the actual encoder library loaded by FFmpeg, using a tiny one-frame probe."""
    build = subprocess.run(
        ["ffmpeg", "-version"], check=True, capture_output=True, text=True, timeout=20
    ).stdout
    probe = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-nostdin",
            "-loglevel",
            "verbose",
            "-f",
            "lavfi",
            "-i",
            "color=size=16x16:rate=30",
            "-frames:v",
            "1",
            "-c:v",
            "libaom-av1",
            "-threads",
            "1",
            "-f",
            "null",
            "-",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
    )
    match = re.search(r"\[libaom-av1 @ [^\]]+\]\s+(v[0-9][^\s]*)", probe.stderr)
    if match is None:
        raise RuntimeError("FFmpeg did not identify its libaom encoder version")
    return {
        "ffmpeg": build.splitlines()[0],
        "ffmpeg_libraries": [
            line.strip() for line in build.splitlines() if line.lstrip().startswith("lib")
        ],
        "libaom": match.group(1),
        "ffmpeg_binary_sha256": sha256(shutil.which("ffmpeg")),
        "determinism_scope": "Byte-identical builds within these encoder versions; other encoder versions may produce different AV1 bytes with the same verified YUV/RGB pixels.",
    }


def extract_video(src, dst, start, stop, fps, full_length):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if start == 0 and stop == full_length:
        shutil.copyfile(src, dst)
        if sha256(src) != sha256(dst):
            raise RuntimeError("Full-episode video copy mismatch")
        return "byte_copy"
    # Decode by frame number, not keyframe seeking. Single-thread lossless AV1
    # retains the source codec, pixel format, dimensions and exact selected pixels.
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-i",
            str(src),
            "-map_metadata",
            "-1",
            "-an",
            "-vf",
            f"trim=start_frame={start}:end_frame={stop},setpts=N/({fps}*TB)",
            "-c:v",
            "libaom-av1",
            "-crf",
            "0",
            "-b:v",
            "0",
            "-cpu-used",
            "8",
            "-threads",
            "1",
            "-row-mt",
            "0",
            "-pix_fmt",
            "yuv420p",
            "-r",
            str(fps),
            "-fflags",
            "+bitexact",
            "-flags:v",
            "+bitexact",
            str(dst),
        ],
        check=True,
    )
    return "frame_exact_lossless_av1"


def validate_video(path, frames, fps, source=None, start=0):
    import av

    def pixel_hashes(frame):
        # RGB also checks color-range/colorspace metadata consumed by training.
        return tuple(
            hashlib.sha256(frame.to_ndarray(format=format).tobytes()).hexdigest()
            for format in ("yuv420p", "rgb24")
        )

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "NONE"
        stream.codec_context.thread_count = 1
        if (
            len(container.streams.video) != 1
            or container.streams.audio
            or stream.codec_context.format.name != "yuv420p"
            or stream.codec_context.name not in {"av1", "libdav1d"}
            or float(stream.average_rate) != fps
        ):
            raise ValueError(f"Unexpected video format: {path}")
        decoded = []
        for index, frame in enumerate(container.decode(stream)):
            if (frame.width, frame.height) != (640, 480):
                raise ValueError(f"Unexpected video dimensions: {path}")
            if abs(float(frame.pts * frame.time_base) - index / fps) > 1e-5:
                raise ValueError(f"Video timestamp mismatch: {path}, frame {index}")
            decoded.append(pixel_hashes(frame))
        if len(decoded) != frames:
            raise ValueError(f"Video frame count mismatch: {path}")
    if source is not None:
        with av.open(str(source)) as container:
            source_hashes = []
            for index, frame in enumerate(container.decode(video=0)):
                if start <= index < start + frames:
                    source_hashes.append(pixel_hashes(frame))
                if index >= start + frames:
                    break
        if source_hashes != decoded:
            raise ValueError(f"Clipped video pixels differ from source: {path}")


def validate_raw_episode(table, record, fps):
    """Reject timeline and unit-shape drift before slicing the unchanged values."""
    length = int(record["length"])
    if length < 16 or table.num_rows != length:
        raise ValueError("Source episode is incomplete or shorter than the action horizon")
    for key in ("observation.state", "action"):
        values = np.asarray(table[key].to_pylist())
        if values.shape != (length, 6) or not np.isfinite(values).all():
            raise ValueError(f"Expected six finite recorded values in {key}")
    for key, expected in (
        ("episode_index", np.full(length, record["episode_index"])),
        ("frame_index", np.arange(length)),
        ("index", np.arange(record["dataset_from_index"], record["dataset_to_index"])),
    ):
        if not np.array_equal(table[key].to_numpy(), expected):
            raise ValueError(f"Source episode timeline mismatch: {key}")
    if not np.allclose(table["timestamp"].to_numpy(), np.arange(length) / fps, atol=1e-6):
        raise ValueError("Source timestamps do not match the recorded frame rate")


def validate_subset(dest, expected):
    """Exercise the unchanged GR00T reader, including both cameras and language."""
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader

    from so101.so100_config import so100_config

    loader = LeRobotEpisodeLoader(dest, so100_config, decoder_kwargs={"num_ffmpeg_threads": 1})
    if len(loader) != expected["episodes"] or sum(loader.episode_lengths) != expected["frames"]:
        raise RuntimeError("GR00T reader episode/frame counts differ from the build")
    for ep, length in enumerate(loader.episode_lengths):
        loaded = loader[ep]
        table = pq.read_table(
            dest / DATA_PATH.format(episode_chunk=ep // loader.chunk_size, episode_index=ep)
        )
        if len(loaded) != length:
            raise RuntimeError("GR00T reader truncates an episode")
        for modality, column in (("state", "observation.state"), ("action", "action")):
            actual = np.concatenate(
                [np.stack(loaded[f"{modality}.{key}"]) for key in ("single_arm", "gripper")],
                axis=1,
            )
            if not np.array_equal(actual, np.asarray(table[column].to_pylist())):
                raise RuntimeError(f"GR00T reader changed recorded units: {modality}")
        for camera in ("room", "wrist"):
            if any(np.asarray(frame).shape != (480, 640, 3) for frame in loaded[f"video.{camera}"]):
                raise RuntimeError(f"GR00T reader camera shape mismatch: {camera}")
        if not all(loaded["language.annotation.human.task_description"]):
            raise RuntimeError("GR00T reader lost task descriptions")
    statistics = loader.get_dataset_statistics()
    for modality in ("state", "action"):
        for joint, dimension in (("single_arm", 5), ("gripper", 1)):
            for key in ("mean", "std", "min", "max", "q01", "q99"):
                values = statistics[modality][joint][key]
                if len(values) != dimension or not np.isfinite(values).all():
                    raise RuntimeError("GR00T reader statistics are incomplete")
    return {"episodes": len(loader), "frames": sum(loader.episode_lengths), "passed": True}


def build_subset(source, dest, info, records, ids, audit_rows, window, sidecars=None):
    dest.mkdir(parents=True)
    (dest / "meta").mkdir()
    (dest / "pick_place_meta").mkdir()
    metadata, episode_stats, mapping, used_tasks = [], [], [], set()
    sidecars = source / "pick_place_meta" if sidecars is None else sidecars
    offset = 0
    for new_ep, old_ep in enumerate(ids):
        record = records[old_ep]
        length = int(record["length"])
        start, stop = (
            clip_bounds(audit_rows[old_ep], length, info["fps"]) if window else (0, length)
        )
        data_src = source_path(
            source,
            info["data_path"].format(
                chunk_index=record["data/chunk_index"], file_index=record["data/file_index"]
            ),
        )
        raw = pq.read_table(data_src)
        raw = raw.filter(pc.equal(raw["episode_index"], old_ep))
        validate_raw_episode(raw, record, info["fps"])
        selected = raw.slice(start, stop - start)
        table = rewrite_indices(selected, new_ep, offset, info["fps"])
        for key in ("action", "observation.state"):
            if not selected[key].equals(table[key]):
                raise RuntimeError("Joint values changed")
        kwargs = {"episode_chunk": new_ep // info["chunks_size"], "episode_index": new_ep}
        data_dest = dest / DATA_PATH.format(**kwargs)
        data_dest.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, data_dest)
        if not pq.read_table(data_dest).equals(table):
            raise RuntimeError("Parquet roundtrip mismatch")
        for key in sorted(VIDEO_KEYS):
            if (
                record[f"videos/{key}/from_timestamp"] != 0
                or abs(record[f"videos/{key}/to_timestamp"] * info["fps"] - length) > 1e-5
            ):
                raise ValueError("Expected one source episode per video")
            src = source_path(
                source,
                info["video_path"].format(
                    video_key=key,
                    chunk_index=record[f"videos/{key}/chunk_index"],
                    file_index=record[f"videos/{key}/file_index"],
                ),
            )
            dst = dest / VIDEO_PATH.format(video_key=key, **kwargs)
            mode = extract_video(src, dst, start, stop, info["fps"], length)
            validate_video(
                dst, stop - start, info["fps"], source=src if window else None, start=start
            )
            print(f"PASS video {dest.name}/{new_ep}/{key}: {mode}", flush=True)
        sidecar = json.loads(source_path(sidecars, f"episode_{old_ep:06d}.json").read_text())
        if (
            sidecar["episode_index"] != old_ep
            or sidecar["num_frames"] != length
            or sidecar["fps"] != info["fps"]
            or sidecar["units"] != "m"
            or sidecar["reference_frame"] != "robot base link (base)"
        ):
            raise ValueError("Source sidecar identity/units differ from the audited dataset")
        for key in ("position", "orientation_wxyz"):
            if len(sidecar[key]) != length:
                raise ValueError("Sidecar length mismatch")
            sidecar[key] = sidecar[key][start:stop]
        sidecar.update(episode_index=new_ep, num_frames=stop - start)
        write_json(dest / f"pick_place_meta/episode_{new_ep:06d}.json", sidecar)
        metadata.append({"episode_index": new_ep, "tasks": record["tasks"], "length": stop - start})
        episode_stats.append({"episode_index": new_ep, "stats": numeric_stats(table)})
        used_tasks.update(table["task_index"].to_pylist())
        mapping.append(
            {
                "episode_index": new_ep,
                "source_episode_index": old_ep,
                "start_frame": start,
                "stop_frame_exclusive": stop,
            }
        )
        offset += stop - start
    tasks_table = pq.read_table(source / "meta/tasks.parquet").to_pandas()
    tasks = [
        {"task_index": int(row["task_index"]), "task": str(task)}
        for task, row in tasks_table.iterrows()
        if row["task_index"] in used_tasks
    ]
    if {task["task_index"] for task in tasks} != used_tasks:
        raise ValueError("Source task descriptions do not cover the selected episode task IDs")
    for name, rows in (("episodes", metadata), ("episodes_stats", episode_stats), ("tasks", tasks)):
        (dest / f"meta/{name}.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows)
        )
    new_info = copy.deepcopy(info)
    new_info.update(
        codebase_version="v2.1",
        total_episodes=len(ids),
        total_frames=offset,
        total_chunks=(len(ids) + info["chunks_size"] - 1) // info["chunks_size"],
        total_videos=len(ids) * 2,
        total_tasks=len(tasks),
        splits={"train": f"0:{len(ids)}"},
        data_path=DATA_PATH,
        video_path=VIDEO_PATH,
    )
    write_json(dest / "meta/info.json", new_info)
    modality = json.loads((ROOT / "so101/provenance/modality.json").read_text())
    write_json(dest / "meta/modality.json", modality)
    shutil.copyfile(ROOT / "so101/so100_config.py", dest / "so100_config.py")
    write_json(dest / "episode_mapping.json", mapping)
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.data.stats import generate_rel_stats, generate_stats

    from so101.so100_config import so100_config  # registers NEW_EMBODIMENT

    if so100_config["action"].delta_indices != list(range(16)):
        raise RuntimeError("SO-101 data configuration changed from the approved action horizon")
    generate_stats(dest)
    generate_rel_stats(dest, EmbodimentTag.NEW_EMBODIMENT)
    result = {"episodes": len(ids), "frames": offset, "mapping": mapping}
    result["reader_check"] = validate_subset(dest, result)
    return result


def check_sizes(root):
    files = sorted(
        ((p.stat().st_size, str(p.relative_to(root))) for p in root.rglob("*") if p.is_file()),
        reverse=True,
    )
    total = sum(size for size, _ in files)
    report = {
        "total_bytes": total,
        "largest_files": files[:20],
        "passed": total < 1_000_000_000 and all(size < 95_000_000 for size, _ in files),
    }
    if not report["passed"]:
        raise RuntimeError("STOP: plain-Git size limits exceeded: " + json.dumps(report))
    return report


def make_readme(audit, decisions, counts, encoder=None):
    """Embed the reviewable decisions and numerical thresholds in the dataset."""
    source_episodes = len(audit["episodes"])
    source_frames = sum(row["frames"] for row in audit["episodes"])
    lines = [
        "# SO-101 pick-place data",
        "",
        f"Source: original pick_place_v1 ({source_episodes} episodes, {source_frames} frames, 30 fps), including pick_place_meta sidecars. The raw source is unmodified. No new demonstrations or domain randomisation.",
        "",
        "Build command (from this repository after audit approval):",
        "",
        "```bash",
        ".venv/bin/python -m so101.build_dataset --source <raw-pick_place_v1> --audit audit/metadata.json --decisions audit/approved_decisions.json",
        "# For sidecars outside the raw dataset, add --sidecars <pick_place_meta>.",
        "```",
        "",
        f"Approved audit SHA-256: `{decisions['audit_sha256']}`.",
        f"Audited source-manifest SHA-256: `{decisions['source_manifest_sha256']}`.",
        "",
        "| Subset | Episodes | Frames | Mixture weight |",
        "|---|---:|---:|---:|",
        f"| main | {counts['main']['episodes']} | {counts['main']['frames']} | 1.0 |",
        f"| grasp_window | {counts['grasp_window']['episodes']} | {counts['grasp_window']['frames']} | 2.0 |",
        "",
        "Main source episode IDs: " + ", ".join(map(str, sorted(decisions["main_episodes"]))) + ".",
        "Grasp-window source episode IDs: "
        + ", ".join(map(str, sorted(decisions["grasp_window_episodes"])))
        + ".",
        "",
        "Approved exclusions:",
        "",
        "| Source episode | Reason |",
        "|---:|---|",
    ]
    lines.extend(
        f"| {ep} | {reason.replace('|', '/').replace(chr(10), ' ')} |"
        for ep, reason in sorted(decisions["exclusions"].items(), key=lambda row: int(row[0]))
    )
    lines += [
        "",
        "Audit thresholds (strictly greater than the measured reference P75 is hesitant; percentile method: "
        + str(audit.get("percentile_method", "linear"))
        + "):",
        "",
        "| Metric | Threshold |",
        "|---|---:|",
        f"| Arm speed P25, degrees/s | {audit['arm_speed_p25_deg_s']:.9g} |",
    ]
    lines.extend(f"| {key} | {value:.9g} |" for key, value in sorted(audit["thresholds"].items()))
    lines += [
        "",
        "Global gripper close/open and derived endpoints, in original recorded action units: `"
        + json.dumps(audit["global_gripper"], sort_keys=True)
        + "`.",
        f"Minimum post-close cube lift: {audit['min_lift_m']:.9g} m. Reference: {audit['lift_reference']}.",
        "Reference episodes for the percentile thresholds: "
        + ", ".join(map(str, audit["threshold_reference_episodes"]))
        + ".",
        "",
        "An episode is bad if mandated or if the primary close/lift measurement is absent; an observed incomplete 90%-10% closure is hesitant. Missing global-midpoint metrics can be threshold-unresolved; the approved episode decisions above govern inclusion. Roll alignment is the recorded calibrated wrist-roll/arm-direction proxy folded to 45 degrees, not a contact or task-success measurement.",
        "",
        "Grasp windows contain approved clean episodes from close minus 1.5 s through the frame at lift plus 1.0 s, clipped to episode boundaries. Cameras: room/wrist; five arm joints plus gripper; absolute actions with horizon 16; original LeRobot arm units (-100 to 100) and gripper units (0 to 100), without conversion. Videos retain 640x480, 30 fps, AV1, yuv420p and no audio. Full episodes are byte-copied; clips use deterministic single-thread lossless AV1 and every decoded YUV and RGB frame is checked against the source.",
        "",
        "Both subsets use GR00T v2.1 episode metadata, tasks, numeric statistics and the unchanged GR00T reader. These are ordinary Git files. The build enforces total size below 1 GB and every file below 95 MB; no Git LFS is needed. Counts, renumbering and clip bounds are also in build_report.json; full measurements in audit_metadata.json; source hashes in source_sha256.json; prepared-file hashes in prepared_sha256.json.",
    ]
    if encoder is not None:
        lines += [
            "",
            f"Encoder: {encoder['ffmpeg']}; libaom {encoder['libaom']}. {encoder['determinism_scope']}",
        ]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--sidecars", type=Path, help="Defaults to source/pick_place_meta")
    parser.add_argument("--audit", default=ROOT / "audit/metadata.json", type=Path)
    parser.add_argument("--decisions", required=True, type=Path)
    parser.add_argument("--output", default=ROOT / "datasets/pick_place_groot", type=Path)
    args = parser.parse_args(argv)
    source, output = args.source.resolve(), args.output.resolve()
    sidecars = args.sidecars.resolve() if args.sidecars is not None else source / "pick_place_meta"
    staging = output.with_name(output.name + ".building")
    for path in (output, staging):
        if path == source or path in source.parents or source in path.parents or path.exists():
            raise ValueError(f"Output must be new and separate from raw data: {path}")
        if path == sidecars or path in sidecars.parents or sidecars in path.parents:
            raise ValueError(f"Output must be separate from raw sidecars: {path}")
    before = source_manifest(source, sidecars)
    audit = json.loads(args.audit.read_text())
    decisions = json.loads(args.decisions.read_text())
    audited_source, audited_sidecars = Path(audit["source"]), Path(audit["sidecars"])
    if audited_sidecars.is_relative_to(audited_source):
        if not sidecars.is_relative_to(source) or sidecars.relative_to(
            source
        ) != audited_sidecars.relative_to(audited_source):
            raise ValueError("Selected embedded sidecars differ from the audit")
    elif sidecars.is_relative_to(source):
        raise ValueError("Audit used external sidecars; pass their directory with --sidecars")
    info = json.loads((source / "meta/info.json").read_text())
    if info["fps"] != 30 or info["codebase_version"] != "v3.0":
        raise ValueError("Expected the recorded 30 fps v3.0 dataset")
    if {k for k, v in info["features"].items() if v["dtype"] == "video"} != VIDEO_KEYS:
        raise ValueError("Source camera keys differ from the saved pick-place config")
    for key in ("observation.state", "action"):
        if (
            info["features"][key]["names"] != JOINT_NAMES
            or info["features"][key]["shape"] != [6]
            or info["features"][key]["dtype"] != "float32"
        ):
            raise ValueError("Source joint order, dimensions or recorded dtype changed")
    for key in VIDEO_KEYS:
        feature = info["features"][key]
        video = feature["info"]
        if feature["shape"] != [480, 640, 3] or any(
            video.get(name) != expected
            for name, expected in {
                "video.codec": "av1",
                "video.pix_fmt": "yuv420p",
                "video.fps": 30,
                "has_audio": False,
            }.items()
        ):
            raise ValueError("Source video settings differ from the saved pick-place format")
    record_list = load_records(source)
    records = {int(r["episode_index"]): r for r in record_list}
    if (
        len(records) != len(record_list)
        or set(records) != set(range(info["total_episodes"]))
        or sum(int(r["length"]) for r in record_list) != info["total_frames"]
    ):
        raise ValueError("Source episode metadata is incomplete")
    selections, rows = validate_decisions(decisions, args.audit, audit, set(records), before)
    encoder = encoder_versions()
    staging.mkdir(parents=True)
    counts = {}
    for name, selection in (("main", "main_episodes"), ("grasp_window", "grasp_window_episodes")):
        counts[name] = build_subset(
            source,
            staging / name,
            info,
            records,
            selections[selection],
            rows,
            name == "grasp_window",
            sidecars,
        )
    if source_manifest(source, sidecars) != before:
        raise RuntimeError("Raw source changed during build")
    validate_decisions(decisions, args.audit, audit, set(records), before)
    if json.loads(args.decisions.read_text()) != decisions:
        raise RuntimeError("Approved decisions changed during build")
    write_json(staging / "source_sha256.json", before)
    write_json(staging / "audit_decisions.json", decisions)
    shutil.copyfile(args.audit, staging / "audit_metadata.json")
    write_json(
        staging / "build_report.json",
        {
            "counts": counts,
            "raw_source_unchanged": True,
            "video_policy": "Full episodes byte-copied; clips frame-exact lossless AV1, original resolution/fps/pixel format",
            "encoder_versions": encoder,
        },
    )
    (staging / "README.md").write_text(make_readme(audit, decisions, counts, encoder))
    size_report = check_sizes(staging)
    write_json(staging / "prepared_sha256.json", snapshot(staging))
    size_report = check_sizes(staging)
    staging.rename(output)
    print(
        json.dumps(
            {
                "output": str(output),
                "counts": {
                    k: {x: y for x, y in v.items() if x != "mapping"} for k, v in counts.items()
                },
                "size_check": size_report,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
