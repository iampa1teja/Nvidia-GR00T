"""Resolve the two pinned official snapshots, verifying revision file checksums.

A complete pre-warmed cache is used without network calls or downloads. Setup
may fill missing files; training only accepts an already verified cache.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path


PROVENANCE = Path(__file__).resolve().parent / "provenance"
EXPECTED = {
    "foundation": ("nvidia/GR00T-N1.7-3B", "2fc962b973bccdd5d8ce4f67cc63b264d6886495"),
    "backbone": ("nvidia/Cosmos-Reason2-2B", "9ce19a195e423419c349abfc86fd07178b230561"),
}


def pinned_models():
    models = json.loads((PROVENANCE / "training_contract.json").read_text())["models"]
    if set(models) != set(EXPECTED):
        raise ValueError("Exactly foundation and backbone must be configured")
    for role, (repo_id, revision) in EXPECTED.items():
        if (models[role]["repo_id"], models[role]["revision"]) != (repo_id, revision):
            raise ValueError(f"Unapproved model or revision: {role}")
    return {
        role: {"repo_id": value["repo_id"], "revision": value["revision"]}
        for role, value in models.items()
    }


def hub_cache():
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"]).expanduser()
    if os.environ.get("HF_HOME"):
        return Path(os.environ["HF_HOME"]).expanduser() / "hub"
    cache_root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return cache_root / "huggingface/hub"


def file_matches(path, entry):
    if not path.is_file() or path.stat().st_size != entry["size"]:
        return False
    if entry.get("sha256"):
        digest = hashlib.sha256()
        expected = entry["sha256"]
    elif entry.get("git_blob_sha1"):
        digest = hashlib.sha1()
        digest.update(f"blob {entry['size']}\0".encode())
        expected = entry["git_blob_sha1"]
    else:
        raise ValueError(f"Missing revision checksum for {path.name}")
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest() == expected


def resolve_snapshots(download=False):
    pins = pinned_models()
    manifest = json.loads((PROVENANCE / "hf_files.json").read_text())
    result = {}
    for role, spec in pins.items():
        entry = manifest[role]
        if any(entry[key] != spec[key] for key in ("repo_id", "revision")):
            raise ValueError(f"Manifest revision mismatch: {role}")
        folder = (
            hub_cache()
            / ("models--" + spec["repo_id"].replace("/", "--"))
            / "snapshots"
            / spec["revision"]
        )
        invalid = []
        for name, metadata in entry["files"].items():
            if Path(name).is_absolute() or ".." in Path(name).parts:
                raise ValueError("Invalid snapshot manifest path")
            if not file_matches(folder / name, metadata):
                invalid.append(name)
        if invalid:
            if not download:
                raise RuntimeError(
                    f"Run setup_env.sh: {spec['repo_id']}@{spec['revision']} missing or invalid files: {invalid}"
                )
            from huggingface_hub import snapshot_download

            # Never force-redownload a matching blob. Corrupt existing blobs
            # require explicit repair rather than silently trusting hub metadata.
            corrupt = [name for name in invalid if (folder / name).exists()]
            if corrupt:
                raise RuntimeError(
                    f"Corrupt cached files require repair: {spec['repo_id']}: {corrupt}"
                )
            snapshot_download(
                repo_id=spec["repo_id"],
                revision=spec["revision"],
                cache_dir=str(hub_cache()),
                allow_patterns=invalid,
                token=os.environ.get("HF_TOKEN") or None,
                force_download=False,
            )
            failures = [
                name
                for name, meta in entry["files"].items()
                if not file_matches(folder / name, meta)
            ]
            if failures:
                raise RuntimeError(f"FAIL revision verification: {spec['repo_id']}: {failures}")
            print(
                f"PASS downloaded missing files and verified {spec['repo_id']}@{spec['revision']}",
                flush=True,
            )
        else:
            print(
                f"PASS reused verified snapshot {spec['repo_id']}@{spec['revision']} (no download)",
                flush=True,
            )
        result[role] = folder
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    paths = resolve_snapshots(download=args.download)
    print(json.dumps({key: str(value) for key, value in paths.items()}, indent=2))


if __name__ == "__main__":
    main()
