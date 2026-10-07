"""CPU checks for exact trainable sets, complete loads and honest stage provenance."""

import copy
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest
from so101 import launcher
from so101.hardware import CapacityExhausted, GPUBusyError, GPUInfo, GPUProcess, memory_ladder
from so101.weights import pinned_models
import torch
from torch import nn


class CategoryLinear(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.W = nn.Parameter(torch.randn(12, input_dim, output_dim))
        self.b = nn.Parameter(torch.ones(12, output_dim))

    def forward(self, values, categories):
        return torch.bmm(values, self.W[categories]) + self.b[categories].unsqueeze(1)


class SmallDiT(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer_blocks = nn.ModuleList([nn.Linear(4, 4), nn.Linear(4, 4)])
        self.output = nn.Linear(4, 4)

    def forward(self, values):
        for block in self.transformer_blocks:
            values = block(values).tanh()
        return self.output(values)


class TinyGr00t(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(add_pos_embed=True, use_vlln=True)
        self.backbone = nn.Linear(4, 4)
        self.action_head = nn.Module()
        self.action_head.state_encoder = CategoryLinear(6, 4)
        self.action_head.action_encoder = CategoryLinear(6, 4)
        self.action_head.action_decoder = CategoryLinear(4, 6)
        self.action_head.position_embedding = nn.Embedding(3, 4)
        self.action_head.vlln = nn.LayerNorm(4)
        self.action_head.vl_self_attention = nn.Linear(4, 4)
        self.action_head.model = SmallDiT()
        self.action_head.mask_token = nn.Parameter(torch.ones(4))

    def forward(self, values):
        categories = torch.tensor([10] * values.shape[0])
        hidden = self.action_head.state_encoder(values, categories)
        hidden += self.action_head.action_encoder(values, categories)
        hidden = self.action_head.vlln(hidden)
        hidden = self.action_head.vl_self_attention(hidden)
        hidden = self.action_head.model(hidden)
        return self.action_head.action_decoder(hidden, categories)


def test_upstream_integrity_ignores_generated_files_but_rejects_source_changes(
    tmp_path, monkeypatch
):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    source = tmp_path / "gr00t/model.py"
    source.parent.mkdir()
    source.write_text("original source\n")
    (tmp_path / ".gitignore").write_text("outputs/\nuv.lock\n__pycache__/\nignored_model.py\n")
    subprocess.run(["git", "add", ".gitignore", "gr00t"], cwd=tmp_path, check=True)
    monkeypatch.setenv("GIT_TEST_ASSUME_DIFFERENT_OWNER", "1")
    evidence = tmp_path / "provenance"
    evidence.mkdir()
    launcher.write_json(
        evidence / "upstream_gr00t_sha256.json", {"gr00t/model.py": launcher.sha256(source)}
    )
    monkeypatch.setattr(launcher, "PROVENANCE", evidence)
    ignored = tmp_path / "gr00t/outputs/local.png"
    ignored.parent.mkdir()
    ignored.write_bytes(b"local capture")
    (tmp_path / "gr00t/uv.lock").write_text("local environment\n")
    launcher.verify_upstream(tmp_path)
    addition = tmp_path / "gr00t/new_model.py"
    addition.write_text("unexpected source\n")
    with pytest.raises(RuntimeError, match="new_model.py"):
        launcher.verify_upstream(tmp_path)
    addition.unlink()
    addition = tmp_path / "gr00t/ignored_model.py"
    addition.write_text("unexpected ignored source\n")
    with pytest.raises(RuntimeError, match="ignored_model.py"):
        launcher.verify_upstream(tmp_path)
    addition.unlink()
    source.write_text("changed source\n")
    with pytest.raises(RuntimeError, match="model.py"):
        launcher.verify_upstream(tmp_path)
    source.unlink()
    with pytest.raises(RuntimeError, match="model.py"):
        launcher.verify_upstream(tmp_path)


def test_git_output_trusts_only_the_resolved_repository(tmp_path, monkeypatch):
    calls = []

    def check_output(command, **kwargs):
        calls.append((command, kwargs))
        return "commit\n"

    monkeypatch.setattr(launcher.subprocess, "check_output", check_output)
    assert launcher._git_output(tmp_path / "subdir/..", "rev-parse", "HEAD") == "commit\n"
    assert calls == [
        (
            ["git", "-c", f"safe.directory={tmp_path}", "-C", str(tmp_path), "rev-parse", "HEAD"],
            {"text": True},
        )
    ]


def test_loading_requires_all_loading_info_and_all_weights():
    valid = {"missing_keys": [], "unexpected_keys": [], "mismatched_keys": []}
    launcher.assert_complete_loading(valid)
    for key in valid:
        bad = dict(valid)
        bad[key] = ["action_head.mask_token"]
        with pytest.raises(RuntimeError, match="Incomplete checkpoint weights"):
            launcher.assert_complete_loading(bad)
        bad = dict(valid)
        del bad[key]
        with pytest.raises(RuntimeError, match="loading information"):
            launcher.assert_complete_loading(bad)
    with pytest.raises(RuntimeError, match="Incomplete checkpoint weights"):
        launcher.assert_complete_loading(dict(valid, error_msgs=["bad shard"]))


@pytest.mark.parametrize("stage", [1, 2])
def test_frozen_parameter_selection_is_exact(stage):
    model = TinyGr00t()
    expected = launcher.freeze_for_stage(model, stage)
    assert all(not param.requires_grad for param in model.backbone.parameters())
    assert not model.action_head.mask_token.requires_grad
    assert all(param.requires_grad for param in model.action_head.state_encoder.parameters())
    assert all(
        param.requires_grad == (stage == 2) for param in model.action_head.model.parameters()
    )
    assert launcher.assert_trainable(model, expected) > 0
    model.action_head.mask_token.requires_grad_(True)
    with pytest.raises(RuntimeError, match="unexpected=.*mask_token"):
        launcher.assert_trainable(model, expected)


def test_missing_configured_adapter_fails():
    model = TinyGr00t()
    del model.action_head.state_encoder
    with pytest.raises(RuntimeError, match="missing required module"):
        launcher.freeze_for_stage(model, 1)
    model = TinyGr00t()
    del model.action_head.vlln
    with pytest.raises(RuntimeError, match="missing its configured vision-language adapter"):
        launcher.freeze_for_stage(model, 1)


def test_fresh_adapter_reset_only_category10_and_repeatable():
    original = TinyGr00t()
    first, second = copy.deepcopy(original), copy.deepcopy(original)
    changed = launcher.fresh_embodiment_adapters(first, 10, 42)
    launcher.fresh_embodiment_adapters(second, 10, 42)
    assert len(changed) == 6
    original_state = original.state_dict()
    for name, param in first.named_parameters():
        assert torch.equal(param, second.state_dict()[name])
        if name in changed:
            assert torch.equal(param[:10], original_state[name][:10])
            assert torch.equal(param[11:], original_state[name][11:])
            assert not torch.equal(param[10], original_state[name][10])
            if name.endswith(".b"):
                assert torch.count_nonzero(param[10]) == 0
        else:
            assert torch.equal(param, original_state[name])


def test_gradients_flow_through_frozen_eval_dit_and_checkpointing_matches():
    plain = TinyGr00t()
    launcher.freeze_for_stage(plain, 1)
    checked = copy.deepcopy(plain)
    plain.action_head.model.eval()
    checked.action_head.model.eval()
    launcher.checkpoint_blocks(checked.action_head.model)
    values = torch.randn(2, 1, 6)
    plain(values).square().mean().backward()
    checked(values).square().mean().backward()
    for (name, param), (_, other) in zip(plain.named_parameters(), checked.named_parameters()):
        if param.grad is None:
            assert other.grad is None
        else:
            assert torch.allclose(param.grad, other.grad, atol=1e-6), name
    assert plain.action_head.state_encoder.W.grad[10].abs().sum() > 0
    assert all(param.grad is None for param in plain.action_head.model.parameters())
    assert all(param.grad is None for param in plain.backbone.parameters())


def test_optimizer_has_per_group_rates_and_protects_non_selected_bank_rows():
    model = TinyGr00t()
    expected = launcher.freeze_for_stage(model, 2)
    launcher.guard_embodiment_gradients(model)
    original = model.action_head.state_encoder.W.detach().clone()
    groups = launcher.optimizer_groups(model, 2, weight_decay=0.9)
    assert {group["lr"] for group in groups} == {3e-5, 1e-4}
    bank = model.action_head.state_encoder.W
    group = next(group for group in groups if any(param is bank for param in group["params"]))
    assert group["weight_decay"] == 0
    optimizer = torch.optim.AdamW(groups)
    launcher.assert_optimizer(model, optimizer, expected)
    # An intentionally all-row loss still cannot update other embodiment rows.
    sum(param.square().sum() for param in model.parameters() if param.requires_grad).backward()
    optimizer.step()
    assert torch.equal(bank[:10], original[:10])
    assert torch.equal(bank[11:], original[11:])
    assert not torch.equal(bank[10], original[10])
    optimizer.param_groups[0]["params"].append(model.backbone.weight)
    with pytest.raises(RuntimeError, match="Optimizer parameters differ"):
        launcher.assert_optimizer(model, optimizer, expected)


def test_duplicate_optimizer_parameters_are_rejected():
    model = TinyGr00t()
    expected = launcher.freeze_for_stage(model, 1)
    optimizer = torch.optim.AdamW(launcher.optimizer_groups(model, 1))
    optimizer.param_groups[0]["params"].append(optimizer.param_groups[0]["params"][0])
    with pytest.raises(RuntimeError, match="Optimizer parameters differ"):
        launcher.assert_optimizer(model, optimizer, expected)


def test_stage1_loss_reports_still_falling_without_extension():
    trend = launcher.loss_trend([{"loss": 10 - index * 0.08} for index in range(100)])
    assert trend["status"] == "still_falling"
    assert trend["automatic_extension"] is False
    assert launcher.loss_trend([{"loss": 2} for _ in range(30)])["status"] == "flattening"
    assert launcher.loss_trend([{"loss": 2}])["status"] == "insufficient_history"


@pytest.mark.parametrize(
    "field",
    [
        "stage",
        "mode",
        "max_steps",
        "init",
        "models",
        "data_sha256",
        "repo_commit",
        "upstream_commit",
        "memory",
        "learning_rates",
        "seed",
    ],
)
def test_resume_must_preserve_full_contract(field):
    previous = {
        key: f"same-{key}"
        for key in (
            "stage",
            "mode",
            "max_steps",
            "init",
            "models",
            "data_sha256",
            "repo_commit",
            "upstream_commit",
            "memory",
            "learning_rates",
            "seed",
        )
    }
    current = dict(previous)
    launcher.assert_resume_compatible(previous, current)
    current[field] = "changed"
    with pytest.raises(RuntimeError, match="Resume changes"):
        launcher.assert_resume_compatible(previous, current)


def checkpoint(root, steps, *, complete=True):
    directory = root / f"checkpoint-{steps}"
    directory.mkdir(parents=True)
    (directory / "config.json").write_text("{}")
    (directory / "model.safetensors").write_bytes(b"cpu-test-weight-placeholder")
    (directory / "trainer_state.json").write_text(json.dumps({"global_step": steps}))
    if complete:
        for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth"):
            (directory / name).write_bytes(b"cpu-test-state-placeholder")
    return directory


def stage1_record(root, steps, mode="full"):
    stage = root / "stage_1_align"
    directory = checkpoint(stage, steps)
    record = {
        "stage": 1,
        "mode": mode,
        "max_steps": steps,
        "models": pinned_models(),
        "init": pinned_models()["foundation"],
        "data_sha256": "known-data",
        "status": "completed",
    }
    launcher.write_json(stage / "run_config.json", record)
    return directory, record


def test_full_stage2_requires25k_official_stage1_not_a_dry_or_smoke(tmp_path):
    directory, record = stage1_record(tmp_path, 25000)
    assert launcher._stage1_init(tmp_path, "full", 60000, "known-data") == directory
    for mode, steps in (("dry_run", 2), ("smoke", 50)):
        other_root = tmp_path / mode
        short, _ = stage1_record(other_root, steps, mode)
        assert launcher._stage1_init(other_root, mode, steps, "known-data") == short
        with pytest.raises(RuntimeError, match="not an audited Stage 1"):
            launcher._stage1_init(other_root, "full", 60000, "known-data")
    record["init"] = {"repo_id": "nvidia/SO_ARM_Starter_Gr00tN17", "revision": "unapproved"}
    launcher.write_json(tmp_path / "stage_1_align/run_config.json", record)
    with pytest.raises(RuntimeError, match="not an audited Stage 1"):
        launcher._stage1_init(tmp_path, "full", 60000, "known-data")


def test_stage2_rejects_incomplete_stage1(tmp_path):
    _, record = stage1_record(tmp_path, 25000)
    record["status"] = "started"
    launcher.write_json(tmp_path / "stage_1_align/run_config.json", record)
    with pytest.raises(RuntimeError, match="completed Stage 1 final"):
        launcher._stage1_init(tmp_path, "full", 60000, "known-data")


def test_resume_requires_full_state_and_every_weight_shard(tmp_path):
    directory = checkpoint(tmp_path, 2, complete=False)
    launcher.verify_checkpoint(directory)
    with pytest.raises(RuntimeError, match="optimizer, scheduler"):
        launcher.verify_checkpoint(directory, resume=True)
    (directory / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"tensor": "missing.safetensors"}})
    )
    with pytest.raises(RuntimeError, match="weight shards missing"):
        launcher.verify_checkpoint(directory)


def test_stage_config_uses_foundation_architecture_and_exact_dataset_mixture(tmp_path):
    foundation = tmp_path / "foundation"
    foundation.mkdir()
    (foundation / "config.json").write_text(
        json.dumps(
            {
                "action_horizon": 40,
                "max_state_dim": 132,
                "max_action_dim": 132,
                "hidden_size": 1024,
                "input_embedding_dim": 1536,
                "max_num_embodiments": 32,
                "model_name": "nvidia/Cosmos-Reason2-2B",
            }
        )
    )
    gpu = GPUInfo(0, "GPU-test", "RTX 4090", 24564, 0, 0)
    memory = memory_ladder(2, [gpu])[3]
    spec = {
        "stage": 2,
        "dataset_root": str(tmp_path / "dataset"),
        "max_steps": 60000,
        "load_checkpoint": "/approved/stage1/checkpoint-25000",
        "output_dir": "/output/stage2",
        "resume_checkpoint": None,
        "mode": "full",
        "memory": memory.as_dict(),
    }
    config = launcher.build_config(spec, foundation)
    assert config.model.action_horizon == 40
    assert config.model.max_state_dim == config.model.max_action_dim == 132
    assert config.training.max_steps == 60000
    assert config.training.warmup_steps == 1000
    assert config.training.save_steps == 5000
    assert config.training.save_total_limit is None
    assert config.training.gradient_checkpointing is False
    assert config.training.global_batch_size == 16
    assert config.training.gradient_accumulation_steps == 2
    assert [item.mix_ratio for item in config.data.datasets] == [1.0, 2.0]
    assert [Path(item.dataset_paths[0]).name for item in config.data.datasets] == [
        "main",
        "grasp_window",
    ]
    assert len(config.data.modality_configs["new_embodiment"]["action"].delta_indices) == 16
    assert config.model.tune_diffusion_model
    assert not config.model.tune_llm and not config.model.tune_visual
    assert config.model.state_dropout_prob == 0.2


def test_evaluation_handoff_has_correct_checkpoints_and_gripper_flags():
    commands = launcher.evaluation_commands(Path("/external_storage/models/our_training"))
    assert len(commands) == 9
    assert "checkpoint-25000" in commands[1]
    assert "checkpoint-10000" in commands[2]
    assert "checkpoint-60000" in commands[-2]
    assert "milestones/checkpoint-200000" in commands[-1]
    for command in commands[1:]:
        assert "--action_horizon 8" in command
        assert "--gripper_latch" in command and "--log_gripper" in command
        assert "--models_dir" in command and "--dataset /workspace/" in command


def test_unapproved_dataset_cli_cannot_load_a_model(monkeypatch, tmp_path):
    monkeypatch.setattr(launcher, "verify_upstream", lambda: None)
    monkeypatch.setattr(
        launcher,
        "verify_dataset",
        lambda root: (_ for _ in ()).throw(RuntimeError("Prepared dataset missing")),
    )
    monkeypatch.setattr(
        launcher,
        "resolve_snapshots",
        lambda **kwargs: pytest.fail("No model lookup before data approval"),
    )
    with pytest.raises(RuntimeError, match="Prepared dataset missing"):
        launcher.main(["--dry_run", "--output_root", str(tmp_path)])


def cpu_orchestration(monkeypatch):
    gpu = GPUInfo(0, "GPU-cpu-test", "RTX 4090", 24564, 0, 0)
    monkeypatch.setattr(launcher, "detect_gpus", lambda: (gpu,))
    args = SimpleNamespace(stage="all", resume=False, smoke_steps=None, dry_run=False)
    return gpu, args, {"foundation": Path("/cpu-test/foundation")}


def test_resume_all_starts_an_unstarted_stage2(monkeypatch, tmp_path):
    gpu, args, snapshots = cpu_orchestration(monkeypatch)
    args.resume = True
    initial, _ = stage1_record(tmp_path, 25000)
    calls = []

    def worker(spec, gpus):
        calls.append(copy.deepcopy(spec))
        assert gpus == (gpu,)
        assert spec["init"] == {"path": str(initial)}
        assert spec["resume_checkpoint"] is None
        if spec["mode"] == "full":
            checkpoint(Path(spec["output_dir"]), spec["max_steps"])
        return 0, False

    monkeypatch.setattr(launcher, "_run_subprocess", worker)
    result = launcher._run_stage(
        2, args, tmp_path, snapshots, "known-data", (gpu,), "commit", "full"
    )
    assert result.name == "checkpoint-60000"
    assert [call["mode"] for call in calls] == ["probe", "full"]
    assert calls[0]["max_steps"] == 2


def test_late_oom_resumes_last_valid_checkpoint_without_optimizer_conversion(monkeypatch, tmp_path):
    gpu, args, snapshots = cpu_orchestration(monkeypatch)
    calls = []
    directory = tmp_path / "stage_1_align"

    def worker(spec, gpus):
        calls.append(copy.deepcopy(spec))
        if spec["mode"] == "probe":
            return 0, False
        if len(calls) == 2:
            saved = checkpoint(directory, 5000)
            launcher.write_json(saved / "run_config.json", spec)
            checkpoint(directory, 10000, complete=False)  # Interrupted, not resumable.
            return 1, True
        assert spec["resume_checkpoint"] == str(directory / "checkpoint-5000")
        assert spec["load_checkpoint"] == spec["resume_checkpoint"]
        assert spec["memory"]["optim"] == "adamw_torch"
        assert spec["memory"]["effective_batch_size"] == 32
        checkpoint(directory, spec["max_steps"])
        return 0, False

    monkeypatch.setattr(launcher, "_run_subprocess", worker)
    result = launcher._run_stage(
        1, args, tmp_path, snapshots, "known-data", (gpu,), "commit", "full"
    )
    assert result.name == "checkpoint-25000"
    assert len(calls) == 4
    assert calls[2]["mode"] == "probe"
    assert calls[2]["max_steps"] == 5002
    assert calls[-1]["max_steps"] == 25000
    assert calls[-1]["memory"]["per_gpu_batch_size"] == 16


def test_exhaustion_requires_actual_cuda_failures_on_full_ladder(monkeypatch, tmp_path):
    gpu, args, snapshots = cpu_orchestration(monkeypatch)
    stage1_record(tmp_path, 25000)
    attempts = []

    def worker(spec, gpus):
        assert spec["mode"] == "probe"
        attempts.append(spec["memory"])
        return 1, True

    monkeypatch.setattr(launcher, "_run_subprocess", worker)
    with pytest.raises(CapacityExhausted) as raised:
        launcher._run_stage(2, args, tmp_path, snapshots, "known-data", (gpu,), "commit", "full")
    assert raised.value.clean_exit
    assert len(attempts) == 8
    assert attempts[-1]["per_gpu_batch_size"] == 1
    assert attempts[-1]["optim"] == "adamw_bnb_8bit"
    assert attempts[-1]["gradient_checkpointing"]


def test_non_oom_failure_never_retries_or_calls_stage_complete(monkeypatch, tmp_path):
    gpu, args, snapshots = cpu_orchestration(monkeypatch)
    attempts = []

    def worker(spec, gpus):
        attempts.append(spec)
        return 1, False

    monkeypatch.setattr(launcher, "_run_subprocess", worker)
    with pytest.raises(RuntimeError, match="other than CUDA OOM"):
        launcher._run_stage(1, args, tmp_path, snapshots, "known-data", (gpu,), "commit", "full")
    assert len(attempts) == 1
    assert not list(tmp_path.rglob("checkpoint-*"))


@pytest.mark.parametrize("processes", [(), (GPUProcess(2977, "unknown", "G", 18),)])
def test_stale_utilization_settles_after_worker(monkeypatch, processes):
    busy = GPUInfo(0, "GPU-test", "RTX 4090", 24564, 30, 90, processes)
    idle = GPUInfo(0, "GPU-test", "RTX 4090", 24564, 30, 0, processes)
    samples = iter([(busy,), (idle,)])
    monkeypatch.setattr(launcher, "detect_gpus", lambda: next(samples))
    sleeps = []
    monkeypatch.setattr(launcher.time, "sleep", sleeps.append)
    assert launcher._preflight_after_worker((idle,)) == (idle,)
    assert sleeps == [0.25]


def test_gpu_owner_is_never_waited_out_or_stopped(monkeypatch):
    busy = GPUInfo(0, "GPU-test", "RTX 4090", 24564, 0, 0, (GPUProcess(42, "teleop", "C", 0),))
    monkeypatch.setattr(launcher, "detect_gpus", lambda: (busy,))
    monkeypatch.setattr(
        launcher.time, "sleep", lambda value: pytest.fail("An actual owner must fail preflight")
    )
    with pytest.raises(GPUBusyError, match="teleop"):
        launcher._preflight_after_worker((busy,))


def test_host_output_mount_mapping_preserves_baseline(monkeypatch):
    monkeypatch.setenv("SO101_HOST_OUTPUT_ROOT", "/host/custom-models")
    commands = launcher.evaluation_commands(Path("/external_storage/models/our_training"))
    assert "--models_dir /host/custom-models/our_training" in commands[1]
    assert "--models_dir /external_storage/models" in commands[-1]
    commands = launcher.evaluation_commands(Path("/other/mounted/output"))
    assert "--models_dir /other/mounted/output" in commands[1]


def test_explicit_resume_oom_preserves_updates_effective_batch_and_optimizer(monkeypatch, tmp_path):
    gpu, args, snapshots = cpu_orchestration(monkeypatch)
    args.resume = True
    directory = tmp_path / "stage_1_align"
    selected = memory_ladder(1, [gpu])[3]
    previous = launcher._make_spec(
        1,
        "full",
        25000,
        directory,
        snapshots,
        "known-data",
        selected,
        pinned_models()["foundation"],
        "commit",
        args,
    )
    initial = checkpoint(directory, 5000)
    launcher.write_json(directory / "run_config.json", previous)
    launcher.write_json(initial / "run_config.json", previous)
    calls = []

    def worker(spec, gpus):
        calls.append(copy.deepcopy(spec))
        launcher.write_json(directory / "run_config.json", spec)
        if len(calls) == 1:
            saved = checkpoint(directory, 10000)
            launcher.write_json(saved / "run_config.json", spec)
            return 1, True
        assert spec["load_checkpoint"] == str(directory / "checkpoint-10000")
        assert spec["resume_checkpoint"] == spec["load_checkpoint"]
        assert spec["memory"]["optim"] == "adamw_bnb_8bit"
        assert spec["memory"]["effective_batch_size"] == 32
        assert spec["memory"]["per_gpu_batch_size"] == 8
        assert spec["memory_recovery"]["resumed_from_step"] == 10000
        assert spec["memory_recovery"]["original_memory"] == selected.as_dict()
        final = checkpoint(directory, 25000)
        launcher.write_json(final / "run_config.json", spec)
        return 0, False

    monkeypatch.setattr(launcher, "_run_subprocess", worker)
    result = launcher._run_stage(
        1, args, tmp_path, snapshots, "known-data", (gpu,), "commit", "full"
    )
    assert result.name == "checkpoint-25000"
    assert len(calls) == 2
    final_record = launcher.read_json(directory / "run_config.json")
    assert len(final_record["memory_recovery"]["attempts"]) == 2
    assert final_record["memory"]["effective_batch_size"] == 32


def test_stage2_capacity_exit_prints_only_completed_stage1_and_baseline(
    monkeypatch, tmp_path, capsys
):
    gpu, _, _ = cpu_orchestration(monkeypatch)
    stage1_record(tmp_path, 25000)
    foundation = tmp_path / "foundation"
    foundation.mkdir()
    (foundation / "embodiment_id.json").write_text(
        json.dumps({"oxe_droid_relative_eef_relative_joint": 13})
    )
    monkeypatch.setattr(launcher, "verify_upstream", lambda: None)
    monkeypatch.setattr(launcher, "verify_dataset", lambda root: "known-data")
    monkeypatch.setattr(launcher, "resolve_snapshots", lambda **kwargs: {"foundation": foundation})
    monkeypatch.setattr(launcher, "preflight_gpus", lambda *args: (gpu,))
    monkeypatch.setattr(launcher, "_disk_preflight", lambda *args: None)
    monkeypatch.setattr(launcher.subprocess, "check_output", lambda *args, **kwargs: "commit\n")

    def exhausted(*args):
        raise CapacityExhausted(2, memory_ladder(2, [gpu]))

    monkeypatch.setattr(launcher, "_run_stage", exhausted)
    assert launcher.main(["--stage", "2", "--output_root", str(tmp_path)]) == 0
    output = capsys.readouterr().out
    assert "stage_1_align/checkpoint-25000" in output
    assert "milestones/checkpoint-200000" in output
    assert "--gripper_latch --log_gripper" in output
    assert "stage_2_main/checkpoint" not in output
