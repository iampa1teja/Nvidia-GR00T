"""Setup checks for pinned versions, imports, GPU and committed dataset readers."""

import argparse
import hashlib
import importlib
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import shutil
import sys


ROOT = Path(__file__).resolve().parents[1]
IMPORTS = (
    "torch",
    "torchvision",
    "transformers",
    "torchcodec",
    "flash_attn",
    "bitsandbytes",
    "numpy",
    "pandas",
    "pyarrow",
    "av",
    "deepspeed",
    "gr00t",
    "so101.so100_config",
)
SETUP_SCRATCH_MARKER = "SO-101 setup verification scratch v1\n"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def torch_fingerprint():
    """Version and installed RECORD fingerprint detect installer replacement."""
    dist = importlib.metadata.distribution("torch")
    record = dist.read_text("RECORD")
    if record is None:
        raise RuntimeError("Torch installation has no RECORD fingerprint")
    return {
        "version": dist.version,
        "record_sha256": hashlib.sha256(record.encode()).hexdigest(),
        "location": str(dist.locate_file("")),
    }


def cleanup_setup_checkpoints(scratch):
    """Remove setup's generated checkpoints, retaining its logs and metrics.

    A mktemp name and ownership marker are mandatory. Validate every path before
    removing anything. Caches and ordinary training outputs cannot satisfy this
    boundary; symlinks are rejected throughout.
    """
    scratch = Path(scratch)
    if scratch.is_symlink() or not scratch.is_dir():
        raise RuntimeError("Setup scratch must be an ordinary existing directory")
    if not re.fullmatch(r"so101-setup-dry-run-[A-Za-z0-9]{8}", scratch.name):
        raise RuntimeError("Refusing cleanup outside setup's mktemp directory")
    scratch = scratch.resolve()
    marker = scratch / ".so101-setup-verification"
    if marker.is_symlink() or not marker.is_file() or marker.read_text() != SETUP_SCRATCH_MARKER:
        raise RuntimeError("Setup scratch ownership marker missing or invalid")
    stage = scratch / "dry_run/stage_1_align"
    if any(path.is_symlink() for path in (scratch / "dry_run", stage)):
        raise RuntimeError("Setup stage directories must not be symlinks")
    if not stage.resolve().is_relative_to(scratch):
        raise RuntimeError("Setup stage leaves the owned scratch directory")
    candidates = sorted(stage.glob("checkpoint-*"))
    for candidate in candidates:
        if (
            not re.fullmatch(r"checkpoint-[0-9]+", candidate.name)
            or candidate.is_symlink()
            or not candidate.is_dir()
            or candidate.resolve().parent != stage.resolve()
            or not candidate.resolve().is_relative_to(scratch)
            or any(path.is_symlink() for path in candidate.rglob("*"))
        ):
            raise RuntimeError("Unsafe checkpoint path in setup scratch; no cleanup performed")
    for candidate in candidates:
        shutil.rmtree(candidate)
    return len(candidates)


def check_versions():
    expected = json.loads((ROOT / "so101/provenance/environment_versions.json").read_text())[
        "versions"
    ]
    actual = {"python": platform.python_version()}
    package_names = {
        "torch": "torch",
        "transformers": "transformers",
        "torchcodec": "torchcodec",
        "bitsandbytes": "bitsandbytes",
        "numpy": "numpy",
        "av": "av",
        "flash_attn": "flash-attn",
    }
    for key, package in package_names.items():
        value = importlib.metadata.version(package)
        # Official FlashAttention wheel includes its build compatibility suffix.
        actual[key] = value.split("+", 1)[0] if key == "flash_attn" else value
    for key, value in actual.items():
        if value != expected[key]:
            raise RuntimeError(f"{key}: expected {expected[key]}, observed {value}")
    import torch

    if torch.version.cuda != expected["cuda"]:
        raise RuntimeError(
            f"Torch CUDA: expected {expected['cuda']}, observed {torch.version.cuda}"
        )
    return ", ".join(f"{key}={value}" for key, value in actual.items())


def check_gpu():
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("No CUDA GPU is visible; start the container with --gpus all")
    gpus = []
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        gpus.append(f"{index}: {props.name} ({props.total_memory / 2**30:.2f} GiB)")
    return "; ".join(gpus)


def verify_prepared_manifest(root):
    """Verify every committed dataset byte, rejecting added files and symlinks."""
    root = Path(root)
    path = root / "prepared_sha256.json"
    if not path.is_file():
        raise RuntimeError(f"Prepared dataset is missing: {path}; Phase 3 approval/build required")
    expected = json.loads(path.read_text())
    if not isinstance(expected, dict) or not expected:
        raise RuntimeError("Prepared dataset hash manifest is empty or invalid")
    paths = list(root.rglob("*"))
    if any(p.is_symlink() for p in paths):
        raise RuntimeError("Prepared dataset must contain ordinary files, not symlinks")
    actual_files = {str(p.relative_to(root)) for p in paths if p.is_file() and p != path}
    if actual_files != set(expected):
        raise RuntimeError("Prepared dataset files differ from the committed manifest")
    for name, digest in expected.items():
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise RuntimeError("Prepared dataset manifest contains an unsafe path")
        if sha256(root / name) != digest:
            raise RuntimeError(f"Prepared dataset file changed: {name}")
    return len(expected)


def check_subset(path, reader_factory=None):
    """Read first/last episodes through GR00T, checking both cameras and joints."""
    import numpy as np

    path = Path(path)
    info = json.loads((path / "meta/info.json").read_text())
    count, frames = info["total_episodes"], info["total_frames"]
    if (
        type(count) is not int
        or count < 1
        or type(frames) is not int
        or frames < 1
        or info["fps"] != 30
    ):
        raise RuntimeError(f"Invalid dataset counts/fps: {path}")
    actual_modality = json.loads((path / "meta/modality.json").read_text())
    expected_modality = json.loads((ROOT / "so101/provenance/modality.json").read_text())
    if actual_modality != expected_modality:
        raise RuntimeError(f"Saved camera/joint mapping differs: {path}")
    if reader_factory is None:
        from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader

        from so101.so100_config import so100_config

        loader = LeRobotEpisodeLoader(path, so100_config, decoder_kwargs={"num_ffmpeg_threads": 1})
    else:
        loader = reader_factory(path)
    if len(loader) != count or sum(loader.episode_lengths) != frames:
        raise RuntimeError(f"GR00T reader totals differ: {path}")
    for ep in sorted({0, count - 1}):
        loaded = loader[ep]
        length = loader.episode_lengths[ep]
        if len(loaded) != length:
            raise RuntimeError(f"GR00T reader episode length differs: {path}/{ep}")
        for camera in ("room", "wrist"):
            if any(np.asarray(frame).shape != (480, 640, 3) for frame in loaded[f"video.{camera}"]):
                raise RuntimeError(f"Camera cannot be read at recorded shape: {path}/{ep}/{camera}")
        for modality in ("state", "action"):
            for joint, dimension in (("single_arm", 5), ("gripper", 1)):
                values = np.stack(loaded[f"{modality}.{joint}"])
                if values.shape != (length, dimension) or not np.isfinite(values).all():
                    raise RuntimeError(f"Invalid joint data: {path}/{ep}/{modality}.{joint}")
        if not all(loaded["language.annotation.human.task_description"]):
            raise RuntimeError(f"Missing task instructions: {path}/{ep}")
    return f"{count} episodes / {frames} frames; first/last room+wrist decoded"


def run_checks(dataset_root=None):
    failures = []

    def report(label, action):
        try:
            result = action()
            print(f"PASS {label}" + (f": {result}" if result else ""), flush=True)
        except Exception as exc:
            failures.append(label)
            print(f"FAIL {label}: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)

    for name in IMPORTS:
        report(f"import {name}", lambda name=name: importlib.import_module(name) and None)
    report("pinned versions", check_versions)
    report("CUDA GPU", check_gpu)
    dataset_root = Path(dataset_root or ROOT / "datasets/pick_place_groot")
    report(
        "committed dataset hashes",
        lambda: f"{verify_prepared_manifest(dataset_root)} files verified",
    )
    for subset in ("main", "grasp_window"):
        report(f"dataset {subset}", lambda subset=subset: check_subset(dataset_root / subset))
    if failures:
        raise RuntimeError("Setup checks failed: " + ", ".join(failures))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--torch-fingerprint", action="store_true")
    parser.add_argument("--cleanup-setup-checkpoints", type=Path)
    parser.add_argument("--dataset-root", type=Path)
    args = parser.parse_args()
    if args.torch_fingerprint:
        print(json.dumps(torch_fingerprint(), sort_keys=True))
    elif args.cleanup_setup_checkpoints:
        removed = cleanup_setup_checkpoints(args.cleanup_setup_checkpoints)
        print(
            f"PASS removed {removed} ephemeral setup checkpoint directories; verification logs retained"
        )
    else:
        run_checks(args.dataset_root)


if __name__ == "__main__":
    main()
