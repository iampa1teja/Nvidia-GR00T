import hashlib
import json

import pytest
from so101 import weights


def warm_cache(tmp_path, monkeypatch):
    provenance = tmp_path / "provenance"
    provenance.mkdir()
    pins = {
        role: {"repo_id": repo, "revision": rev} for role, (repo, rev) in weights.EXPECTED.items()
    }
    (provenance / "training_contract.json").write_text(json.dumps({"models": pins}))
    manifest = {}
    for role, spec in pins.items():
        blob = b"verified bytes"
        folder = (
            tmp_path
            / "hub"
            / ("models--" + spec["repo_id"].replace("/", "--"))
            / "snapshots"
            / spec["revision"]
        )
        folder.mkdir(parents=True)
        (folder / "config.json").write_bytes(blob)
        manifest[role] = {
            **spec,
            "files": {
                "config.json": {
                    "size": len(blob),
                    "git_blob_sha1": hashlib.sha1(
                        f"blob {len(blob)}\0".encode() + blob
                    ).hexdigest(),
                }
            },
        }
    (provenance / "hf_files.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(weights, "PROVENANCE", provenance)
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    return provenance


def test_matching_snapshots_reused_without_network(tmp_path, monkeypatch):
    warm_cache(tmp_path, monkeypatch)
    import huggingface_hub

    monkeypatch.setattr(
        huggingface_hub, "snapshot_download", lambda **kw: pytest.fail("download on warm cache")
    )
    assert set(weights.resolve_snapshots(download=True)) == {"foundation", "backbone"}


def test_corrupt_blob_cannot_pass_revision_only_check(tmp_path, monkeypatch):
    warm_cache(tmp_path, monkeypatch)
    paths = weights.resolve_snapshots()
    (paths["foundation"] / "config.json").write_bytes(b"corrupt! bytes")
    with pytest.raises(RuntimeError, match="missing or invalid"):
        weights.resolve_snapshots()
    with pytest.raises(RuntimeError, match="Corrupt cached"):
        weights.resolve_snapshots(download=True)


def test_other_model_cannot_replace_foundation(tmp_path, monkeypatch):
    provenance = warm_cache(tmp_path, monkeypatch)
    config = json.loads((provenance / "training_contract.json").read_text())
    config["models"]["foundation"]["repo_id"] = "nvidia/SO_ARM_Starter_Gr00tN17"
    (provenance / "training_contract.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="Unapproved"):
        weights.resolve_snapshots()
