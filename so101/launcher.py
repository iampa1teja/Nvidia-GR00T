"""Audited two-stage SO-101 training from the pinned official GR00T foundation.

The orchestration path uses the standard library and read-only hardware checks.
Torch is imported only in the explicitly launched worker or CPU audit helpers.
All changes to GR00T behavior are temporary runtime patches.
"""

import argparse
from collections import defaultdict
from contextlib import ExitStack
from dataclasses import replace
import functools
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time
from unittest.mock import patch

from so101.hardware import (
    CapacityExhausted,
    GPUBusyError,
    MemoryConfig,
    detect_gpus,
    is_cuda_oom,
    memory_ladder,
    preflight_gpus,
)
from so101.weights import pinned_models, resolve_snapshots


ROOT = Path(__file__).resolve().parents[1]
PROVENANCE = ROOT / "so101/provenance"
DIT = "action_head.model"
ADAPTERS = (
    "action_head.state_encoder",
    "action_head.action_encoder",
    "action_head.action_decoder",
    "action_head.position_embedding",
    "action_head.vlln",
    "action_head.vl_self_attention",
)
CATEGORY_ADAPTERS = ADAPTERS[:3]
GIB = 1024**3


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(payload):
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _git_output(root, *arguments):
    root = Path(root).resolve()
    # Container root reads a host-owned bind mount. Trust only this repo for
    # this command, without modifying global Git settings or trusting others.
    return subprocess.check_output(
        ["git", "-c", f"safe.directory={root}", "-C", str(root), *arguments], text=True
    )


def verify_upstream(root=ROOT):
    root = Path(root)
    expected = read_json(PROVENANCE / "upstream_gr00t_sha256.json")
    names = _git_output(root, "ls-files", "-co", "--exclude-standard", "-z", "--", "gr00t").split(
        "\0"
    )
    names.extend(
        str(path.relative_to(root))
        for path in (root / "gr00t").rglob("*.py")
        if "__pycache__" not in path.parts
    )
    current = {name: sha256(root / name) for name in set(names) if name and (root / name).is_file()}
    if current != expected:
        changed = sorted(set(current) ^ set(expected))
        changed.extend(
            name for name in current.keys() & expected.keys() if current[name] != expected[name]
        )
        raise RuntimeError(f"Pinned upstream gr00t/ files changed: {sorted(changed)}")


def verify_dataset(root):
    """Verify the shipped plain-Git data, both subsets and every file hash."""
    root = Path(root)
    manifest_path = root / "prepared_sha256.json"
    if not manifest_path.is_file():
        raise RuntimeError(
            "Approved prepared dataset missing; build datasets/pick_place_groot first"
        )
    expected = read_json(manifest_path)
    for name in expected:
        path = Path(name)
        if path.is_absolute() or ".." in path.parts:
            raise RuntimeError("Dataset manifest has an unsafe path")
    current = {}
    for path in root.rglob("*"):
        if path.is_symlink():
            raise RuntimeError(f"Dataset must be ordinary files: {path}")
        if path.is_file() and path != manifest_path:
            current[str(path.relative_to(root))] = sha256(path)
    if current != expected:
        raise RuntimeError("Prepared dataset differs from its committed SHA256 manifest")
    contract = read_json(PROVENANCE / "data_contract.json")
    for name in ("main", "grasp_window"):
        dataset = root / name
        info = read_json(dataset / "meta/info.json")
        if info["fps"] != contract["video"]["fps"] or info["total_episodes"] <= 0:
            raise RuntimeError(f"Invalid dataset metadata: {name}")
        if read_json(dataset / "meta/modality.json") != read_json(PROVENANCE / "modality.json"):
            raise RuntimeError(
                f"Dataset modalities differ from the saved pick-place settings: {name}"
            )
        if (dataset / "so100_config.py").read_bytes() != (
            ROOT / "so101/so100_config.py"
        ).read_bytes():
            raise RuntimeError(f"Dataset modality registration differs: {name}")
        if not list((dataset / "data").rglob("*.parquet")) or not list(
            (dataset / "videos").rglob("*.mp4")
        ):
            raise RuntimeError(f"Dataset has no readable frames/videos: {name}")
    return canonical_hash(current)


def assert_complete_loading(info):
    required = ("missing_keys", "unexpected_keys", "mismatched_keys")
    if not isinstance(info, dict) or any(name not in info for name in required):
        raise RuntimeError("Loader did not report complete checkpoint loading information")
    errors = {name: info[name] for name in (*required, "error_msgs") if info.get(name)}
    if errors:
        raise RuntimeError(f"Incomplete checkpoint weights: {errors}")


def _belongs(name, prefix):
    return name == prefix or name.startswith(prefix + ".")


def expected_trainable_names(model, stage):
    if stage not in (1, 2):
        raise ValueError("Stage must be 1 or 2")
    names = {name for name, _ in model.named_parameters()}
    required = (*CATEGORY_ADAPTERS, DIT)
    for prefix in required:
        if not any(_belongs(name, prefix) for name in names):
            raise RuntimeError(f"Official model is missing required module: {prefix}")
    if getattr(model.config, "add_pos_embed", False):
        if not any(_belongs(name, ADAPTERS[3]) for name in names):
            raise RuntimeError("Official model is missing its configured position projector")
    if getattr(model.config, "use_vlln", False):
        if not any(_belongs(name, ADAPTERS[4]) for name in names):
            raise RuntimeError("Official model is missing its configured vision-language adapter")
    prefixes = ADAPTERS + ((DIT,) if stage == 2 else ())
    return {name for name in names if any(_belongs(name, prefix) for prefix in prefixes)}


def assert_trainable(model, expected):
    actual = {name for name, param in model.named_parameters() if param.requires_grad}
    if actual != set(expected) or not actual:
        raise RuntimeError(
            f"Trainable parameter audit failed: unexpected={sorted(actual - set(expected))}, "
            f"missing={sorted(set(expected) - actual)}"
        )
    return sum(param.numel() for name, param in model.named_parameters() if name in actual)


def freeze_for_stage(model, stage):
    allowed = expected_trainable_names(model, stage)
    for name, param in model.named_parameters():
        param.requires_grad_(name in allowed)
    head = model.get_submodule("action_head")
    head.tune_projector = head.tune_vlln = True
    head.tune_diffusion_model = stage == 2
    for flag in ("tune_llm", "tune_visual"):
        setattr(model.config, flag, False)
        setattr(model.get_submodule("backbone"), flag, False)
    model.config.tune_projector = model.config.tune_vlln = True
    model.config.tune_diffusion_model = stage == 2
    assert_trainable(model, allowed)
    return allowed


def parameter_table(model, expected):
    assert_trainable(model, expected)
    groups = defaultdict(lambda: defaultdict(int))
    for name, param in model.named_parameters():
        prefix = next(
            (prefix for prefix in (*ADAPTERS, DIT) if _belongs(name, prefix)), name.split(".")[0]
        )
        groups[prefix][param.requires_grad] += param.numel()
    rows = []
    print("Module path | Parameters | Trainable", flush=True)
    for prefix, totals in sorted(groups.items()):
        for trainable, count in sorted(totals.items()):
            rows.append({"module": prefix, "parameters": count, "trainable": trainable})
            print(f"{prefix} | {count:,} | {'yes' if trainable else 'no'}", flush=True)
    print(f"Trainable total: {assert_trainable(model, expected):,}", flush=True)
    return rows


def fresh_embodiment_adapters(model, index=10, seed=42):
    """Reset only the NEW_EMBODIMENT rows using upstream's 0.02-normal init."""
    import torch

    generator = torch.Generator(device="cpu").manual_seed(seed)
    names = []
    with torch.no_grad():
        for name, param in model.named_parameters():
            if not any(_belongs(name, prefix) for prefix in CATEGORY_ADAPTERS):
                continue
            if (
                name.rsplit(".", 1)[-1] not in {"W", "b"}
                or param.ndim < 2
                or index >= param.shape[0]
            ):
                raise RuntimeError(f"Unexpected embodiment adapter layout: {name}")
            if param.is_meta:
                raise RuntimeError(f"Adapter weights were not materialized: {name}")
            if name.endswith(".W"):
                replacement = 0.02 * torch.randn(param[index].shape, generator=generator)
                param[index].copy_(replacement.to(device=param.device, dtype=param.dtype))
            else:
                param[index].zero_()
            names.append(name)
    if not names:
        raise RuntimeError("No fresh SO-101 adapter rows were initialized")
    return names


def guard_embodiment_gradients(model, index=10):
    """Train one bank row; the optimizer excludes bank tensors from weight decay."""
    handles = []
    for name, param in model.named_parameters():
        if param.requires_grad and any(_belongs(name, prefix) for prefix in CATEGORY_ADAPTERS):

            def mask(gradient, selected=index):
                row = gradient[selected].clone()
                gradient.zero_()
                gradient[selected].copy_(row)
                return gradient

            handles.append(param.register_hook(mask))
    return handles


def checkpoint_blocks(dit):
    """Non-reentrant recomputation also retains gradients through a frozen DiT."""
    import torch
    from torch.utils.checkpoint import checkpoint

    if not getattr(dit, "transformer_blocks", None):
        raise RuntimeError("Cannot checkpoint the official DiT: transformer blocks missing")
    for block in dit.transformer_blocks:
        original = block.forward

        @functools.wraps(original)
        def forward(*args, _forward=original, **kwargs):
            # Stage 1's frozen DiT is put in eval mode by upstream. Autograd
            # remains required for its state/action/vision-language inputs.
            if torch.is_grad_enabled():
                return checkpoint(_forward, *args, use_reentrant=False, **kwargs)
            return _forward(*args, **kwargs)

        block.forward = forward


def optimizer_groups(model, stage, weight_decay=1e-5):
    expected = expected_trainable_names(model, stage)
    assert_trainable(model, expected)
    grouped = defaultdict(list)
    for name, param in model.named_parameters():
        if name not in expected:
            continue
        category = any(_belongs(name, prefix) for prefix in CATEGORY_ADAPTERS)
        lr = 3e-5 if stage == 2 and _belongs(name, DIT) else 1e-4
        decay = 0.0 if category or param.ndim < 2 or name.endswith(".bias") else weight_decay
        grouped[(lr, decay)].append(param)
    return [
        {"params": params, "lr": lr, "weight_decay": decay}
        for (lr, decay), params in sorted(grouped.items())
    ]


def assert_optimizer(model, optimizer, expected):
    assert_trainable(model, expected)
    actual = [id(param) for group in optimizer.param_groups for param in group["params"]]
    wanted = {id(param) for name, param in model.named_parameters() if name in expected}
    if len(actual) != len(set(actual)) or set(actual) != wanted:
        raise RuntimeError("Optimizer parameters differ from the audited trainable parameters")


def loss_trend(history):
    """Compare medians of the last two ten-percent windows; do not extend a stage."""
    values = [
        float(row["loss"]) for row in history if "loss" in row and math.isfinite(float(row["loss"]))
    ]
    if len(values) < 20:
        return {"status": "insufficient_history", "loss_records": len(values)}
    import statistics

    width = max(5, len(values) // 10)
    earlier = statistics.median(values[-2 * width : -width])
    recent = statistics.median(values[-width:])
    decrease = (earlier - recent) / max(abs(earlier), 1e-12)
    return {
        "status": "still_falling"
        if decrease > 0.05
        else "rising"
        if decrease < -0.05
        else "flattening",
        "earlier_median": earlier,
        "recent_median": recent,
        "fractional_decrease": decrease,
        "window_records": width,
        "criterion": "Median loss falls >5% between the last two 10%-of-history windows",
        "automatic_extension": False,
    }


def _latest_checkpoint(directory, *, require_complete=False):
    checkpoints = [
        (int(path.name.split("-")[-1]), path)
        for path in Path(directory).glob("checkpoint-*")
        if path.is_dir() and re.fullmatch(r"checkpoint-\d+", path.name)
    ]
    for _, path in sorted(checkpoints, reverse=True):
        if require_complete:
            try:
                verify_checkpoint(path, resume=True)
            except RuntimeError:
                continue
        return path
    return None


def verify_checkpoint(path, *, resume=False):
    path = Path(path)
    if path.is_symlink():
        raise RuntimeError(f"Training checkpoint must be an ordinary directory: {path}")
    if not (path / "config.json").is_file():
        raise RuntimeError(f"Checkpoint config missing: {path}")
    index = path / "model.safetensors.index.json"
    if index.is_file():
        files = set(read_json(index)["weight_map"].values())
    else:
        files = {file.name for file in path.glob("*.safetensors")}
    if not files or any(
        Path(name).name != name or not (path / name).is_file() or (path / name).is_symlink()
        for name in files
    ):
        raise RuntimeError(f"Checkpoint weight shards missing: {path}")
    if resume and any(
        not (path / name).is_file()
        for name in ("trainer_state.json", "optimizer.pt", "scheduler.pt")
    ):
        raise RuntimeError(
            "Resume requires optimizer, scheduler and trainer state, not model-only weights"
        )
    if resume and not list(path.glob("rng_state*.pth")):
        raise RuntimeError("Resume checkpoint has no RNG state")


def assert_resume_compatible(previous, current):
    fields = (
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
    differences = [field for field in fields if previous.get(field) != current.get(field)]
    if differences:
        raise RuntimeError(f"Resume changes the recorded training contract: {differences}")


def _stage_directory(output_root, stage):
    return Path(output_root) / ("stage_1_align" if stage == 1 else "stage_2_main")


def _stage1_init(output_root, mode, max_steps, data_hash):
    stage1 = _stage_directory(output_root, 1)
    record = read_json(stage1 / "run_config.json")
    if (
        record.get("stage") != 1
        or record.get("mode") != mode
        or record.get("models") != pinned_models()
        or record.get("data_sha256") != data_hash
        or record.get("init", {}).get("repo_id") != pinned_models()["foundation"]["repo_id"]
        or record.get("init", {}).get("revision") != pinned_models()["foundation"]["revision"]
    ):
        raise RuntimeError(
            "Stage 2 init is not an audited Stage 1 from the pinned official foundation"
        )
    expected_steps = 25000 if mode == "full" else max_steps
    if record.get("max_steps") != expected_steps:
        raise RuntimeError("Stage 1 run length differs from the required final checkpoint")
    path = stage1 / f"checkpoint-{expected_steps}"
    verify_checkpoint(path)
    state = read_json(path / "trainer_state.json")
    if state["global_step"] != expected_steps or record.get("status") != "completed":
        raise RuntimeError("Stage 2 requires the completed Stage 1 final checkpoint")
    return path


def _disk_preflight(output_root, snapshots, stages, mode):
    probe = Path(output_root)
    while not probe.exists():
        probe = probe.parent
    weight_bytes = sum(
        path.stat().st_size for path in snapshots["foundation"].glob("*.safetensors")
    )
    # FP32 saved model + optimizer states can exceed the BF16 foundation size.
    count = sum((5 if stage == 1 else 12) for stage in stages) if mode == "full" else len(stages)
    required = count * max(16 * GIB, 3 * weight_bytes) + 20 * GIB
    free = shutil.disk_usage(probe).free
    print(
        f"Disk preflight: {free / GIB:.1f} GiB free, {required / GIB:.1f} GiB reserved for retained checkpoints",
        flush=True,
    )
    if free < required:
        raise RuntimeError(
            "Insufficient disk space for all requested checkpoints and optimizer states"
        )


def build_config(spec, foundation):
    """Build architecture exclusively from the official snapshot, then apply data/training controls."""
    from gr00t.configs.base_config import Config
    from gr00t.configs.data.data_config import SingleDatasetConfig
    from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config

    from so101.so100_config import so100_config

    contract = read_json(PROVENANCE / "training_contract.json")
    config = Config()
    config.model = Gr00tN1d7Config(**read_json(Path(foundation) / "config.json"))
    settings = read_json(PROVENANCE / "processor_settings.json")["settings"]
    for key, value in settings.items():
        target = "use_albumentations_transforms" if key == "use_albumentations" else key
        if hasattr(config.model, target):
            setattr(config.model, target, value)
    if settings.get("shortest_image_edge") is not None:
        config.model.image_crop_size = config.model.image_target_size = None
    config.model.tune_llm = config.model.tune_visual = False
    config.model.tune_projector = config.model.tune_vlln = True
    config.model.tune_diffusion_model = spec["stage"] == 2
    config.model.model_revision = None  # Local sources; backbone has its own pinned revision.
    config.data.modality_configs = {"new_embodiment": so100_config}
    for key, value in read_json(PROVENANCE / "data_settings.json")["settings"].items():
        setattr(config.data, key, value)
    config.data.datasets = [
        SingleDatasetConfig(
            dataset_paths=[str(Path(spec["dataset_root"]) / item["subdirectory"])],
            embodiment_tag="new_embodiment",
            mix_ratio=item["mix_ratio"],
        )
        for item in contract["dataset"]["mixture"]
    ]
    plan = contract["stages"][str(spec["stage"])]
    for key in ("warmup_steps", "lr_scheduler_type", "save_total_limit"):
        setattr(config.training, key, plan[key])
    config.training.max_steps = spec["max_steps"]
    config.training.warmup_ratio = 0.0
    config.training.save_steps = min(plan["save_steps"], spec["max_steps"])
    config.training.start_from_checkpoint = spec["load_checkpoint"]
    config.training.output_dir = spec["output_dir"]
    config.training.experiment_name = None
    config.training.resume_from_checkpoint = bool(spec["resume_checkpoint"])
    config.training.skip_weight_loading = False
    config.training.save_only_model = False
    config.training.global_batch_size = spec["memory"]["global_batch_size"]
    config.training.per_gpu_batch_size = None
    config.training.gradient_accumulation_steps = spec["memory"]["gradient_accumulation_steps"]
    config.training.num_gpus = spec["memory"]["num_gpus"]
    config.training.gradient_checkpointing = (
        False  # Real DiT wrapper instead of upstream's unused flag.
    )
    config.training.optim = spec["memory"]["optim"]
    config.training.learning_rate = 1e-4
    config.training.logging_steps = 1 if spec["mode"] != "full" else 10
    config.training.use_ddp = True
    config.training.dataloader_num_workers = read_json(PROVENANCE / "data_settings.json")[
        "loader_workers"
    ]
    config.training.use_wandb = config.training.upload_checkpoints = False
    config.training.transformers_local_files_only = True
    return config


def run_worker(spec_path):
    """One fresh process/rank; called only after the parent's GPU preflight."""
    worker_started = time.perf_counter()
    from gr00t.experiment import experiment
    from gr00t.model.gr00t_n1d7 import processing_gr00t_n1d7 as processing, setup
    from gr00t.utils.dist_utils import run_on_rank0
    import torch
    import transformers
    from transformers import Qwen3VLForConditionalGeneration

    spec = read_json(spec_path)
    if spec["models"] != pinned_models():
        raise RuntimeError("Worker model pins differ from the permitted official sources")
    snapshots = resolve_snapshots(download=False)
    config = build_config(spec, snapshots["foundation"])
    expected = set()
    record = dict(spec)
    original_load = setup.AutoModel.from_pretrained
    original_create = setup.Gr00tN1d7Pipeline._create_model
    original_backbone = Qwen3VLForConditionalGeneration.from_pretrained
    original_processor = processing.build_processor
    original_trainer = experiment.Gr00tTrainer
    rank = int(os.environ.get("RANK", "0"))
    output_dir = Path(spec["output_dir"])

    def save_record():
        if rank == 0:
            write_json(output_dir / "run_config.json", record)

    def load(*args, **kwargs):
        kwargs["output_loading_info"] = True
        kwargs["config"] = config.model
        # With an explicit config HF forwards remaining kwargs to __init__.
        # These upstream overrides have already been applied to config.model.
        for key in (
            "tune_llm",
            "tune_visual",
            "tune_projector",
            "tune_diffusion_model",
            "tune_vlln",
            "state_dropout_prob",
            "backbone_trainable_params_fp32",
            "load_bf16",
        ):
            kwargs.pop(key, None)
        model, info = original_load(*args, **kwargs)
        assert_complete_loading(info)
        if any(param.is_meta for param in model.parameters()):
            raise RuntimeError("Checkpoint loading left unmaterialized model parameters")
        record["checkpoint_loading"] = {
            key: 0 for key in ("missing_keys", "unexpected_keys", "mismatched_keys")
        }
        return model, info

    def backbone(model_name, **kwargs):
        if model_name != pinned_models()["backbone"]["repo_id"]:
            raise RuntimeError(f"Unapproved backbone requested: {model_name}")
        kwargs.update(
            local_files_only=True,
            revision=pinned_models()["backbone"]["revision"],
            output_loading_info=True,
        )
        model, info = original_backbone(str(snapshots["backbone"]), **kwargs)
        assert_complete_loading(info)
        return model

    def processor(model_name, kwargs):
        if model_name != pinned_models()["backbone"]["repo_id"]:
            raise RuntimeError(f"Unapproved processor requested: {model_name}")
        return original_processor(
            str(snapshots["backbone"]),
            dict(kwargs, local_files_only=True, revision=pinned_models()["backbone"]["revision"]),
        )

    def data_processor(*args, **kwargs):
        # Upstream from_pretrained drops some preprocessing overrides. Construct
        # with the full saved data-only contract so every recorded setting is used.
        settings = read_json(PROVENANCE / "processor_settings.json")["settings"]
        return processing.Gr00tN1d7Processor(
            **settings,
            modality_configs=config.data.modality_configs,
            statistics=None,
            embodiment_id_mapping={"new_embodiment": spec["embodiment_index"]},
            max_state_dim=config.model.max_state_dim,
            max_action_dim=config.model.max_action_dim,
            max_action_horizon=config.model.action_horizon,
            model_name=pinned_models()["backbone"]["repo_id"],
            model_type=config.model.backbone_model_type,
            transformers_loading_kwargs={"local_files_only": True},
        )

    def create(pipeline):
        model = original_create(pipeline)
        if spec["stage"] == 1 and spec["resume_checkpoint"] is None:
            record["fresh_adapter_parameters"] = fresh_embodiment_adapters(
                model, spec["embodiment_index"], spec["seed"]
            )
        expected.update(freeze_for_stage(model, spec["stage"]))
        record["trainable_parameter_count"] = assert_trainable(model, expected)
        record["allocated_requires_grad_parameter_count"] = record["trainable_parameter_count"]
        record["active_trainable_parameter_count"] = sum(
            param.numel() // param.shape[0]
            if any(_belongs(name, prefix) for prefix in CATEGORY_ADAPTERS)
            else param.numel()
            for name, param in model.named_parameters()
            if name in expected
        )
        record["parameter_groups"] = parameter_table(model, expected)
        record["gradient_bank_guards"] = len(
            guard_embodiment_gradients(model, spec["embodiment_index"])
        )
        record["torch_version"] = torch.__version__
        record["transformers_version"] = transformers.__version__
        record["ema"] = "unsupported by the pinned upstream trainer; not enabled"
        if spec["memory"]["gradient_checkpointing"]:
            checkpoint_blocks(model.get_submodule(DIT))
            print(
                "Gradient checkpointing: non-reentrant DiT blocks (including frozen Stage 1 DiT)",
                flush=True,
            )
        save_record()
        return model

    class Metrics(transformers.TrainerCallback):
        def __init__(self):
            self.started = None
            self.durations = []
            self.checkpoint_durations = []
            self.setup_seconds = 0.0

        def on_train_begin(self, *args, **kwargs):
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            self.started = time.perf_counter()
            self.setup_seconds = self.started - worker_started

        def on_step_begin(self, *args, **kwargs):
            if self.started is None:
                torch.cuda.synchronize()
                self.started = time.perf_counter()

        def on_step_end(self, arguments, state, control, **kwargs):
            torch.cuda.synchronize()
            now = time.perf_counter()
            if self.started is not None:
                self.durations.append(now - self.started)
            self.started = now
            if spec["mode"] == "probe":
                control.should_save = False
            return control

        def on_save(self, arguments, state, control, **kwargs):
            if state.is_world_process_zero:
                write_json(
                    output_dir / f"checkpoint-{state.global_step}" / "run_config.json", record
                )

        def on_train_end(self, arguments, state, control, **kwargs):
            # Full and smoke timing exclude five warmup updates when available.
            measured = self.durations[5:] or self.durations
            seconds = sum(measured) / len(measured) if measured else None
            plan = read_json(PROVENANCE / "training_contract.json")["stages"][str(spec["stage"])][
                "max_steps"
            ]
            save_count = plan // 5000
            checkpoint_seconds = (
                sum(self.checkpoint_durations) / len(self.checkpoint_durations)
                if self.checkpoint_durations
                else None
            )
            training_seconds = plan * seconds if seconds is not None else None
            metrics = {
                "completed_steps": state.global_step,
                "seconds_per_step": seconds,
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / GIB,
                "peak_reserved_gib": torch.cuda.max_memory_reserved() / GIB,
                "seconds_per_step_includes_data_wait": True,
                "setup_seconds": self.setup_seconds,
                "checkpoint_seconds": self.checkpoint_durations,
                "mean_checkpoint_seconds": checkpoint_seconds,
                "full_stage_checkpoint_count": save_count,
                "full_stage_training_hours": training_seconds / 3600
                if training_seconds is not None
                else None,
                "full_stage_eta_hours": (
                    self.setup_seconds + training_seconds + save_count * checkpoint_seconds
                )
                / 3600
                if training_seconds is not None and checkpoint_seconds is not None
                else None,
                "loss_trend": loss_trend(state.log_history),
            }
            write_json(output_dir / f"metrics_rank{rank}.json", metrics)
            if state.is_world_process_zero:
                record["status"] = (
                    "completed" if state.global_step == spec["max_steps"] else "incomplete"
                )
                record["metrics"] = metrics
                save_record()
                if spec["mode"] != "probe":
                    write_json(
                        output_dir / f"checkpoint-{state.global_step}" / "run_config.json", record
                    )
                print("TRAINING_METRICS " + json.dumps(metrics), flush=True)
                if spec["stage"] == 1:
                    trend = metrics["loss_trend"]
                    print(
                        "Stage 1 loss: "
                        + trend["status"]
                        + "; stage length will not be extended automatically",
                        flush=True,
                    )

    metrics = Metrics()

    class AuditedTrainer(original_trainer):
        def create_optimizer(self):
            if self.optimizer is None:
                groups = optimizer_groups(self.model, spec["stage"], self.args.weight_decay)
                kwargs = {
                    "lr": self.args.learning_rate,
                    "betas": (self.args.adam_beta1, self.args.adam_beta2),
                    "eps": self.args.adam_epsilon,
                }
                if spec["memory"]["optim"] == "adamw_bnb_8bit":
                    from bitsandbytes.optim import AdamW8bit

                    self.optimizer = AdamW8bit(groups, **kwargs)
                else:
                    self.optimizer = torch.optim.AdamW(groups, **kwargs)
            assert_optimizer(self.model, self.optimizer, expected)
            record["optimizer_groups"] = [
                {key: value for key, value in group.items() if key in {"lr", "weight_decay"}}
                | {"parameters": sum(param.numel() for param in group["params"])}
                for group in self.optimizer.param_groups
            ]
            print(
                "Per-group LR: DiT=3e-5, adapters=1e-4"
                if spec["stage"] == 2
                else "Adapter LR: 1e-4",
                flush=True,
            )
            save_record()
            return self.optimizer

        def compute_loss(self, model, inputs, **kwargs):
            ids = inputs["inputs"]["embodiment_id"]
            if not bool((ids == spec["embodiment_index"]).all().item()):
                raise RuntimeError("Training batch contains a different embodiment adapter index")
            assert_trainable(self.model, expected)
            return super().compute_loss(model, inputs, **kwargs)

        def train(self, *args, **kwargs):
            self.add_callback(metrics)
            kwargs["resume_from_checkpoint"] = spec["resume_checkpoint"]
            return super().train(*args, **kwargs)

        def _save_checkpoint(self, *args, **kwargs):
            if spec["mode"] != "probe":
                started = time.perf_counter()
                result = super()._save_checkpoint(*args, **kwargs)
                torch.cuda.synchronize()
                finished = time.perf_counter()
                metrics.checkpoint_durations.append(finished - started)
                # Do not include saving the previous checkpoint in s/step.
                metrics.started = finished
                return result

        def save_model(self, output_dir=None, *args, **kwargs):
            # Periodic/final checkpoint directories already contain full models.
            # Upstream's extra root model copy is unnecessary and not resumable.
            if output_dir is not None and spec["mode"] != "probe":
                return super().save_model(output_dir, *args, **kwargs)

    try:
        with ExitStack() as stack:
            stack.enter_context(patch.object(setup.AutoModel, "from_pretrained", load))
            stack.enter_context(
                patch.object(Qwen3VLForConditionalGeneration, "from_pretrained", backbone)
            )
            stack.enter_context(patch.object(processing, "build_processor", processor))
            stack.enter_context(
                patch.object(setup.AutoProcessor, "from_pretrained", data_processor)
            )
            stack.enter_context(patch.object(setup.Gr00tN1d7Pipeline, "_create_model", create))
            stack.enter_context(patch.object(experiment, "Gr00tTrainer", AuditedTrainer))
            experiment.run(config)
        run_on_rank0(
            write_json,
            output_dir / "worker_result.json",
            {"status": "completed"},
            label="worker result",
        )
    except Exception as error:
        write_json(
            output_dir / f"failure_rank{rank}.json",
            {
                "error": str(error),
                "cuda_oom": is_cuda_oom(error),
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / GIB
                if torch.cuda.is_initialized()
                else 0.0,
            },
        )
        raise


def _run_subprocess(spec, gpus):
    directory = Path(spec["output_dir"])
    directory.mkdir(parents=True, exist_ok=True)
    spec_path = directory / "worker_spec.json"
    write_json(spec_path, spec)
    command = [sys.executable]
    if len(gpus) > 1:
        command += [
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={len(gpus)}",
            "--module",
            "so101.launcher",
        ]
    else:
        command += ["-m", "so101.launcher"]
    command += ["--_worker", str(spec_path)]
    env = dict(os.environ)
    env.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "NO_ALBUMENTATIONS_UPDATE": "1",
            "PYTHONFAULTHANDLER": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "CUDA_VISIBLE_DEVICES": ",".join(gpu.uuid for gpu in gpus),
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
    for name in (
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "GROOT_SKIP_HF_MODEL_WEIGHTS",
        "GROOT_HF_LOCAL_FIRST",
        "PYTEST_CURRENT_TEST",
    ):
        env.pop(name, None)
    print("+ " + shlex.join(command), flush=True)
    cuda_oom = False
    with (directory / "train.log").open("a", buffering=1) as logfile:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            logfile.write(line)
            print(line, end="", flush=True)
            cuda_oom |= is_cuda_oom(line)
        returncode = process.wait()
    return returncode, cuda_oom


def evaluation_commands(output_root, *, include_stage2=True):
    output_root = Path(output_root)
    host_root = os.environ.get("SO101_HOST_OUTPUT_ROOT")
    if host_root and output_root.is_relative_to("/external_storage/models"):
        output_root = Path(host_root) / output_root.relative_to("/external_storage/models")
    commands = ["cd ~/Sim-to-Real-SO-101-Workshop"]
    common = [
        "--dataset",
        "/workspace/Sim-to-Real-SO-101-Workshop/datasets/pick_place_v1",
        "--eval_set",
        "/workspace/Sim-to-Real-SO-101-Workshop/eval_sets/pick_place_v1_75ep_seed1984.json",
        "--splits",
        "id,ood,yaw,train",
        "--repeats",
        "1",
        "--robot_start",
        "recorded",
        "--episode_length_s",
        "30",
        "--action_horizon",
        "8",
        "--gripper_latch",
        "--log_gripper",
        "--port",
        "5560",
    ]
    checkpoints = ["stage_1_align/checkpoint-25000"]
    if include_stage2:
        checkpoints += [f"stage_2_main/checkpoint-{step}" for step in range(10000, 60001, 10000)]
    for checkpoint in checkpoints:
        commands.append(
            shlex.join(
                [
                    "./docker/eval_pick_place.sh",
                    "--model",
                    checkpoint,
                    "--models_dir",
                    str(output_root),
                    *common,
                ]
            )
        )
    commands.append(
        shlex.join(
            [
                "./docker/eval_pick_place.sh",
                "--model",
                "so101_pick_place_sim_n17_200k/milestones/checkpoint-200000",
                "--models_dir",
                "/external_storage/models",
                *common,
            ]
        )
    )
    return commands


def _preflight_after_worker(expected_gpus):
    """Allow the departed worker's utilization sampling window to settle."""
    deadline = time.monotonic() + 10
    while True:
        detected = detect_gpus()
        if [gpu.uuid for gpu in detected] != [gpu.uuid for gpu in expected_gpus]:
            raise RuntimeError("Visible GPU inventory changed during the memory ladder")
        try:
            return preflight_gpus(detected)
        except GPUBusyError:
            # An actual owner remains a hard preflight failure. Only ownerless
            # telemetry can be a stale NVML sample after our worker has exited.
            if any(gpu.processes for gpu in detected) or time.monotonic() >= deadline:
                raise
            time.sleep(0.25)


def _make_spec(
    stage, mode, steps, output_dir, snapshots, data_hash, memory, init, repo_commit, args
):
    contract = read_json(PROVENANCE / "training_contract.json")
    return {
        "stage": stage,
        "mode": mode,
        "max_steps": steps,
        "flags": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
            if not key.startswith("_")
        },
        "output_dir": str(output_dir),
        "dataset_root": str(ROOT / contract["dataset"]["root"]),
        "data_sha256": data_hash,
        "models": pinned_models(),
        "init": init,
        "load_checkpoint": str(snapshots["foundation"]) if stage == 1 else init["path"],
        "resume_checkpoint": None,
        "repo_commit": repo_commit,
        "upstream_commit": contract["upstream"]["commit"],
        "memory": memory.as_dict(),
        "learning_rates": {"adapters": 1e-4} if stage == 1 else {"dit": 3e-5, "adapters": 1e-4},
        "embodiment": "new_embodiment",
        "embodiment_index": contract["embodiment"]["projector_index"],
        "seed": read_json(PROVENANCE / "data_settings.json")["settings"]["seed"],
        "mixture": contract["dataset"]["mixture"],
        "status": "started",
    }


def _run_stage(stage, args, output_root, snapshots, data_hash, gpus, repo_commit, mode):
    contract = read_json(PROVENANCE / "training_contract.json")
    steps = (
        2
        if mode == "dry_run"
        else args.smoke_steps
        if mode == "smoke"
        else contract["stages"][str(stage)]["max_steps"]
    )
    directory = _stage_directory(output_root, stage)
    if stage == 1:
        init = pinned_models()["foundation"]
    else:
        init = {"path": str(_stage1_init(output_root, mode, steps, data_hash))}
    rungs = memory_ladder(stage, gpus)
    existing = directory / "run_config.json"
    resuming = args.resume and not (
        args.stage == "all"
        and stage == 2
        and not existing.exists()
        and _latest_checkpoint(directory) is None
    )
    resume_checkpoint = _latest_checkpoint(directory, require_complete=True) if resuming else None
    if not resuming and (existing.exists() or _latest_checkpoint(directory)):
        raise RuntimeError(
            f"Stage output already exists: {directory}; use --resume or a new output root"
        )
    if resuming:
        if not existing.is_file() or resume_checkpoint is None:
            raise RuntimeError(f"--resume requires an audited checkpoint in {directory}")
        previous = read_json(existing)
        selected = MemoryConfig(
            **{key: previous["memory"][key] for key in MemoryConfig.__dataclass_fields__}
        )
        spec = _make_spec(
            stage, mode, steps, directory, snapshots, data_hash, selected, init, repo_commit, args
        )
        assert_resume_compatible(previous, spec)
        if selected.num_gpus != len(gpus):
            raise RuntimeError("Resume GPU count differs from the saved batch configuration")
        verify_checkpoint(resume_checkpoint, resume=True)
        if (
            read_json(resume_checkpoint / "trainer_state.json")["global_step"] == steps
            and previous.get("status") == "completed"
        ):
            print(f"Stage {stage} already completed: {resume_checkpoint}", flush=True)
            return resume_checkpoint
        candidates = [selected]
        signatures = {
            (selected.per_gpu_batch_size, selected.gradient_checkpointing, selected.optim)
        }
        for candidate in rungs:
            if candidate.per_gpu_batch_size > selected.per_gpu_batch_size:
                continue
            if selected.gradient_checkpointing and not candidate.gradient_checkpointing:
                continue
            candidate = replace(
                candidate,
                optim=selected.optim,
                gradient_accumulation_steps=selected.effective_batch_size
                // candidate.global_batch_size,
            )
            signature = (
                candidate.per_gpu_batch_size,
                candidate.gradient_checkpointing,
                candidate.optim,
            )
            if signature not in signatures:
                signatures.add(signature)
                candidates.append(candidate)
        history = []
        attempted = []
        for memory in candidates:
            _preflight_after_worker(gpus)
            resumed_step = read_json(resume_checkpoint / "trainer_state.json")["global_step"]
            spec = _make_spec(
                stage, mode, steps, directory, snapshots, data_hash, memory, init, repo_commit, args
            )
            spec["resume_checkpoint"] = spec["load_checkpoint"] = str(resume_checkpoint)
            spec["gpu"] = [gpu.as_dict() for gpu in gpus]
            spec["memory_recovery"] = {
                "original_memory": selected.as_dict(),
                "resumed_from_step": resumed_step,
                "previous_attempts": history.copy(),
                "optimizer_state_preserved": True,
            }
            status, oom = _run_subprocess(spec, gpus)
            attempted.append(memory)
            history.append(
                {
                    "memory": memory.as_dict(),
                    "resumed_from_step": resumed_step,
                    "returncode": status,
                    "cuda_oom": oom,
                }
            )
            write_json(directory / "resume_memory_ladder.json", history)
            if status == 0:
                result = directory / f"checkpoint-{steps}"
                verify_checkpoint(result, resume=True)
                if (directory / "run_config.json").is_file():
                    completed = read_json(directory / "run_config.json")
                    completed["memory_recovery"]["attempts"] = history
                    write_json(directory / "run_config.json", completed)
                    write_json(result / "run_config.json", completed)
                return result
            if not oom:
                raise RuntimeError(
                    f"Resumed Stage {stage} failed for a reason other than CUDA OOM; inspect {directory}/train.log"
                )
            resume_checkpoint = _latest_checkpoint(directory, require_complete=True)
            if resume_checkpoint is None:
                raise RuntimeError("No complete checkpoint remains for memory recovery")
            print(
                f"Resume CUDA OOM: retain step {read_json(resume_checkpoint / 'trainer_state.json')['global_step']}; try a smaller microbatch with the same optimizer",
                flush=True,
            )
        raise CapacityExhausted(stage, attempted)
    attempts = []
    attempted_configs = []
    recovery_checkpoint = None
    recovery_optim = None
    previous_recovery_memory = None
    for planned_memory in rungs:
        memory = planned_memory
        if recovery_optim is not None:
            memory = replace(memory, optim=recovery_optim)
            signature = (memory.per_gpu_batch_size, memory.gradient_checkpointing, memory.optim)
            if signature == previous_recovery_memory:
                continue
            previous_recovery_memory = signature
        _preflight_after_worker(gpus)
        spec = _make_spec(
            stage, mode, steps, directory, snapshots, data_hash, memory, init, repo_commit, args
        )
        spec["gpu"] = [gpu.as_dict() for gpu in gpus]
        if recovery_checkpoint is not None:
            spec["resume_checkpoint"] = spec["load_checkpoint"] = str(recovery_checkpoint)
            spec["optimizer_preserved_from_checkpoint"] = recovery_optim
        print(f"Stage {stage} rung {memory.rung}: {json.dumps(memory.as_dict())}", flush=True)
        if mode == "full":
            probe_steps = 2
            if recovery_checkpoint is not None:
                probe_steps += read_json(recovery_checkpoint / "trainer_state.json")["global_step"]
            probe = dict(
                spec,
                mode="probe",
                max_steps=probe_steps,
                output_dir=str(directory / "memory_probes" / f"rung-{memory.rung}"),
            )
            status, oom = _run_subprocess(probe, gpus)
            attempted_configs.append(memory)
            attempts.append(
                {"memory": memory.as_dict(), "probe_returncode": status, "cuda_oom": oom}
            )
            write_json(directory / "memory_ladder.json", attempts)
            if status:
                if oom:
                    continue
                raise RuntimeError(
                    f"Stage {stage} memory probe failed for a reason other than CUDA OOM"
                )
            _preflight_after_worker(gpus)
        status, oom = _run_subprocess(spec, gpus)
        if mode != "full":
            attempted_configs.append(memory)
            attempts.append({"memory": memory.as_dict(), "returncode": status, "cuda_oom": oom})
        else:
            attempts[-1].update(training_returncode=status, training_cuda_oom=oom)
        write_json(directory / "memory_ladder.json", attempts)
        if status == 0:
            result = directory / f"checkpoint-{steps}"
            verify_checkpoint(result, resume=True)
            print(f"Stage {stage} final checkpoint: {result}", flush=True)
            return result
        if not oom:
            raise RuntimeError(
                f"Stage {stage} failed for a reason other than CUDA OOM; inspect {directory}/train.log"
            )
        recovery_checkpoint = _latest_checkpoint(directory, require_complete=True)
        if recovery_checkpoint is not None:
            saved = read_json(recovery_checkpoint / "run_config.json")
            recovery_optim = saved["memory"]["optim"]
            previous_recovery_memory = (
                memory.per_gpu_batch_size,
                memory.gradient_checkpointing,
                recovery_optim,
            )
            print(
                f"Resuming {recovery_checkpoint} on the next memory rung; preserving {recovery_optim} optimizer states",
                flush=True,
            )
    raise CapacityExhausted(stage, attempted_configs)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("1", "2", "all"), default="all")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--output_root",
        type=Path,
        default=Path(os.environ.get("OUTPUT_ROOT", "/external_storage/models/so101_official_n17")),
    )
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--dry_run",
        action="store_true",
        help="Actually run 2 optimizer updates per stage in output_root/dry_run",
    )
    modes.add_argument(
        "--smoke_steps",
        type=int,
        help="Run a short verification in output_root/smoke_<steps>, normally 50",
    )
    parser.add_argument("--_worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args._worker is not None:
        return run_worker(args._worker)
    if args.smoke_steps is not None and args.smoke_steps < 2:
        parser.error("--smoke_steps must be at least 2")
    verify_upstream()
    dataset_root = ROOT / read_json(PROVENANCE / "training_contract.json")["dataset"]["root"]
    data_hash = verify_dataset(dataset_root)
    snapshots = resolve_snapshots(download=False)
    tags = read_json(snapshots["foundation"] / "embodiment_id.json")
    if any(re.search(r"so[_-]?(?:100|101|arm)", tag, re.IGNORECASE) for tag in tags):
        raise RuntimeError(
            "STOP: official base has an SO-100/SO-101-style pretrained head; ask the user before choosing an embodiment"
        )
    gpus = preflight_gpus()
    mode = "dry_run" if args.dry_run else "smoke" if args.smoke_steps is not None else "full"
    output_root = args.output_root.expanduser().resolve()
    if mode != "full":
        output_root /= "dry_run" if mode == "dry_run" else f"smoke_{args.smoke_steps}"
    stages = (1, 2) if args.stage == "all" else (int(args.stage),)
    _disk_preflight(output_root, snapshots, stages, mode)
    repo_commit = _git_output(ROOT, "rev-parse", "HEAD").strip()
    print(f"Official pretrained tags: {', '.join(sorted(tags))}", flush=True)
    print(
        "Using NEW_EMBODIMENT with fresh SO-101 state/action adapters at projector index 10",
        flush=True,
    )
    finals = []
    try:
        for stage in stages:
            finals.append(
                _run_stage(stage, args, output_root, snapshots, data_hash, gpus, repo_commit, mode)
            )
    except CapacityExhausted as error:
        print(str(error), flush=True)
        for checkpoint in finals:
            print(f"Completed checkpoint: {checkpoint}", flush=True)
        if error.stage == 2 and mode == "full":
            stage1 = _stage_directory(output_root, 1) / "checkpoint-25000"
            if stage1 not in finals:
                print(f"Completed Stage 1 checkpoint: {stage1}", flush=True)
            print(
                "Evaluate completed Stage 1 and the baseline in the Sim-to-Real repo after training ends:",
                flush=True,
            )
            print("\n".join(evaluation_commands(output_root, include_stage2=False)), flush=True)
        return error.exit_code
    if mode == "full":
        print(
            "Evaluate in the Sim-to-Real repo with docker/eval_pick_place.sh after training ends:",
            flush=True,
        )
        print("\n".join(evaluation_commands(output_root)), flush=True)
    else:
        print(
            f"PASS: {mode} completed; checkpoint paths: {', '.join(map(str, finals))}", flush=True
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
