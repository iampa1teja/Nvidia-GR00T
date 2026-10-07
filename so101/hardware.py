"""Read-only GPU preflight and deterministic training memory configurations.

This module uses the standard library only. Importing it, detecting GPUs, and
planning a stage never import Torch, create a CUDA context, or stop a process.
The launcher must probe each rung in a fresh worker and only retry CUDA OOMs.
"""

from collections.abc import Mapping, Sequence
import csv
from dataclasses import asdict, dataclass
import io
import math
import os
import subprocess
import xml.etree.ElementTree as ET


class HardwareError(RuntimeError):
    """A GPU inventory or training capacity check cannot be satisfied."""


class HardwareDetectionError(HardwareError):
    """nvidia-smi did not provide a complete, trustworthy GPU inventory."""


@dataclass(frozen=True)
class GPUProcess:
    pid: int
    name: str
    kind: str
    used_memory_mib: int | None


@dataclass(frozen=True)
class GPUInfo:
    index: int
    uuid: str
    name: str
    total_memory_mib: int
    used_memory_mib: int
    utilization_percent: int
    processes: tuple[GPUProcess, ...] = ()

    @property
    def free_memory_mib(self) -> int:
        return self.total_memory_mib - self.used_memory_mib

    def as_dict(self) -> dict:
        return asdict(self)


class GPUBusyError(HardwareError):
    """At least one selected GPU is owned or active; no processes are stopped."""

    def __init__(self, gpus: Sequence[GPUInfo], reasons: Sequence[str]):
        self.gpus = tuple(gpus)
        self.reasons = tuple(reasons)
        super().__init__(
            "GPU preflight failed: "
            + "; ".join(reasons)
            + ". Wait for the GPU owner to finish before training."
        )


@dataclass(frozen=True)
class MemoryConfig:
    """One memory rung; global batch has upstream's pre-accumulation meaning."""

    stage: int
    rung: int
    num_gpus: int
    per_gpu_batch_size: int
    gradient_accumulation_steps: int
    gradient_checkpointing: bool
    optim: str

    @property
    def global_batch_size(self) -> int:
        return self.per_gpu_batch_size * self.num_gpus

    @property
    def effective_batch_size(self) -> int:
        return self.global_batch_size * self.gradient_accumulation_steps

    def as_dict(self) -> dict:
        return {
            **asdict(self),
            "global_batch_size": self.global_batch_size,
            "effective_batch_size": self.effective_batch_size,
        }


class CapacityExhausted(HardwareError):
    """The launcher has observed CUDA OOM on every full-stage memory rung.

    ``clean_exit`` is true for Stage 2: the all-stage launcher can keep its
    completed Stage 1 and report the capacity limit without a failing exit.
    Merely constructing the ladder is insufficient to raise this exception.
    """

    def __init__(self, stage: int, attempted: Sequence[MemoryConfig]):
        _validate_stage(stage)
        if not attempted or any(config.stage != stage for config in attempted):
            raise ValueError("Capacity exhaustion needs attempted rungs from the same stage")
        self.stage = stage
        self.attempted = tuple(attempted)
        self.clean_exit = stage == 2
        self.exit_code = 0 if self.clean_exit else 1
        final = attempted[-1]
        optimizer = (
            "8-bit AdamW" if final.optim == "adamw_bnb_8bit" else "the saved AdamW optimizer"
        )
        message = (
            f"Stage {stage} cannot fit after {len(attempted)} memory rungs, ending at "
            f"microbatch {final.per_gpu_batch_size}, "
            f"{'gradient checkpointing' if final.gradient_checkpointing else 'no checkpointing'} "
            f"and {optimizer}. "
        )
        if self.clean_exit:
            message += (
                "Retain the completed Stage 1 checkpoint and exit cleanly; run Stage 2 "
                "on GPUs with more memory. The Stage 2 trainable set has not been reduced."
            )
        else:
            message += "Run Stage 1 on GPUs with more memory."
        super().__init__(message)


def _query(arguments: list[str], timeout: float) -> str:
    try:
        return subprocess.run(
            ["nvidia-smi", *arguments],
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        ).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        detail = getattr(exc, "stderr", None) or str(exc)
        raise HardwareDetectionError(f"Unable to query nvidia-smi: {detail.strip()}") from exc


def _integer(value: str, field: str) -> int:
    try:
        result = int(value.strip())
    except (TypeError, ValueError) as exc:
        raise HardwareDetectionError(f"nvidia-smi returned invalid {field}: {value!r}") from exc
    if result < 0:
        raise HardwareDetectionError(f"nvidia-smi returned negative {field}: {value!r}")
    return result


def _process_inventory(document: str) -> dict[str, tuple[GPUProcess, ...]]:
    try:
        root = ET.fromstring(document)
    except ET.ParseError as exc:
        raise HardwareDetectionError("nvidia-smi returned malformed process XML") from exc
    result = {}
    for gpu in root.findall("gpu"):
        uuid = (gpu.findtext("uuid") or "").strip()
        if not uuid or uuid in result:
            raise HardwareDetectionError("GPU process inventory has missing or duplicate UUIDs")
        processes = []
        # Unlike --query-compute-apps, this also includes graphics owners.
        for process in gpu.findall("./processes/process_info"):
            memory = (process.findtext("used_memory") or "N/A").strip()
            processes.append(
                GPUProcess(
                    pid=_integer(process.findtext("pid") or "", "process PID"),
                    name=(process.findtext("process_name") or "unknown").strip(),
                    kind=(process.findtext("type") or "unknown").strip(),
                    used_memory_mib=None
                    if memory in {"N/A", "[N/A]", "Not Supported", "[Not Supported]"}
                    else _integer(memory.removesuffix("MiB").strip(), "process memory"),
                )
            )
        result[uuid] = tuple(processes)
    return result


def _select_visible(gpus: Sequence[GPUInfo], visible: str | None) -> tuple[GPUInfo, ...]:
    if visible is None:
        return tuple(gpus)
    if not visible.strip() or visible.strip() == "-1":
        return ()
    selected = []
    for token in visible.split(","):
        token = token.strip()
        if token.startswith("MIG-"):
            raise HardwareDetectionError(
                "MIG CUDA_VISIBLE_DEVICES is unsupported; select complete GPUs for this training"
            )
        if token.isdecimal():
            matches = [gpu for gpu in gpus if gpu.index == int(token)]
        elif token.startswith("GPU-"):
            matches = [gpu for gpu in gpus if gpu.uuid.startswith(token)]
        else:
            matches = []
        if len(matches) != 1 or matches[0] in selected:
            raise HardwareDetectionError(
                f"CUDA_VISIBLE_DEVICES entry {token!r} is missing, ambiguous or repeated"
            )
        selected.append(matches[0])
    return tuple(selected)


def detect_gpus(
    *, environ: Mapping[str, str] | None = None, timeout: float = 10.0
) -> tuple[GPUInfo, ...]:
    """Inspect visible GPUs without allocating GPU memory.

    CUDA_VISIBLE_DEVICES ordering is preserved. Both compute and graphics
    owners are retained for preflight. Unknown telemetry fails closed.
    """
    if timeout <= 0:
        raise ValueError("nvidia-smi timeout must be positive")
    environment = os.environ if environ is None else environ
    visible = environment.get("CUDA_VISIBLE_DEVICES")
    if visible is not None and (not visible.strip() or visible.strip() == "-1"):
        return ()
    fields = "index,uuid,name,memory.total,memory.used,utilization.gpu"
    output = _query([f"--query-gpu={fields}", "--format=csv,noheader,nounits"], timeout)
    processes = _process_inventory(_query(["-q", "-x"], timeout))
    gpus = []
    for row in csv.reader(io.StringIO(output), skipinitialspace=True):
        if not row:
            continue
        if len(row) != 6:
            raise HardwareDetectionError(f"Incomplete nvidia-smi GPU record: {row!r}")
        index, uuid, name, total, used, utilization = (entry.strip() for entry in row)
        if uuid not in processes:
            raise HardwareDetectionError(f"Process inventory is missing GPU {uuid}")
        gpu = GPUInfo(
            index=_integer(index, "GPU index"),
            uuid=uuid,
            name=name,
            total_memory_mib=_integer(total, "total memory"),
            used_memory_mib=_integer(used, "used memory"),
            utilization_percent=_integer(utilization, "utilization"),
            processes=processes[uuid],
        )
        if (
            not gpu.name
            or gpu.total_memory_mib == 0
            or gpu.used_memory_mib > gpu.total_memory_mib
            or gpu.utilization_percent > 100
            or any(previous.index == gpu.index or previous.uuid == gpu.uuid for previous in gpus)
        ):
            raise HardwareDetectionError(f"Inconsistent nvidia-smi GPU record: {row!r}")
        gpus.append(gpu)
    return _select_visible(gpus, visible)


def has_workload_processes(gpu: GPUInfo) -> bool:
    """Reject compute/mixed/unknown owners, allowing only small graphics contexts.

    Container PID namespaces may hide an Xorg owner's name, while NVML still
    reports its G type and memory. Total GPU memory and activity are checked
    separately by preflight, so G alone is not enough to permit training.
    """
    return bool(gpu.processes) and not (
        all(
            process.kind == "G" and process.used_memory_mib is not None for process in gpu.processes
        )
        and sum(process.used_memory_mib for process in gpu.processes) <= 128
    )


def preflight_gpus(
    gpus: Sequence[GPUInfo] | None = None,
    *,
    max_used_memory_mib: int = 512,
    max_utilization_percent: int = 10,
) -> tuple[GPUInfo, ...]:
    """Reject busy GPUs, permitting tiny idle display contexts without signals.

    Graphics-only owners need known memory totaling at most 128 MiB, aggregate
    GPU use at most 128 MiB and utilization at most 1%. Compute, mixed and
    unknown owners always fail, even when their reported memory is zero.
    """
    if max_used_memory_mib < 0 or not 0 <= max_utilization_percent <= 100:
        raise ValueError("Invalid GPU preflight thresholds")
    inventory = detect_gpus() if gpus is None else tuple(gpus)
    if not inventory:
        raise HardwareError("No visible NVIDIA GPUs; check --gpus all and CUDA_VISIBLE_DEVICES")
    reasons = []
    for gpu in inventory:
        memory_limit = min(max_used_memory_mib, 128) if gpu.processes else max_used_memory_mib
        utilization_limit = (
            min(max_utilization_percent, 1) if gpu.processes else max_utilization_percent
        )
        if (
            has_workload_processes(gpu)
            or gpu.used_memory_mib > memory_limit
            or gpu.utilization_percent > utilization_limit
        ):
            owners = ", ".join(
                f"PID {process.pid} ({process.kind}: {process.name})" for process in gpu.processes
            )
            reasons.append(
                f"GPU {gpu.index} {gpu.name}: {gpu.used_memory_mib} MiB used, "
                f"{gpu.utilization_percent}% utilization" + (f", {owners}" if owners else "")
            )
    if reasons:
        raise GPUBusyError(inventory, reasons)
    return inventory


def _validate_stage(stage: int) -> None:
    if stage not in (1, 2):
        raise ValueError("Stage must be 1 (adapters) or 2 (DiT + adapters)")


def memory_ladder(
    stage: int,
    gpus: Sequence[GPUInfo],
    *,
    target_effective_batch_size: int | None = None,
) -> tuple[MemoryConfig, ...]:
    """Plan increasingly conservative full-stage settings, without probing.

    The first four 24-GiB rungs follow the workshop's measured DiT ladder:
    32 AdamW; 16 with accumulation; then checkpointing; then 8-bit AdamW.
    Further rungs reduce the microbatch to 1 without reducing trainable sets.
    Both stages retain the target effective batch across every rung; on two
    H200s the default is 64. Actual fit must be established by a training probe.
    """
    _validate_stage(stage)
    if not gpus:
        raise HardwareError("Cannot plan GPU training without a visible GPU")
    if any(gpu.total_memory_mib <= 0 for gpu in gpus):
        raise ValueError("GPU memory must be positive")
    world_size = len(gpus)
    target = (
        max(32, 32 * world_size)
        if target_effective_batch_size is None
        else target_effective_batch_size
    )
    if isinstance(target, bool) or not isinstance(target, int) or target < 1:
        raise ValueError("Target effective batch must be a positive integer")
    smallest_memory = min(gpu.total_memory_mib for gpu in gpus)
    initial = 32 if smallest_memory >= 22 * 1024 else 16 if smallest_memory >= 16 * 1024 else 8
    # A fixed multiple of the largest microbatch is divisible by every rung.
    effective = math.ceil(target / (initial * world_size)) * initial * world_size
    reduced = initial // 2
    settings = [
        (initial, False, "adamw_torch"),
        (reduced, False, "adamw_torch"),
        (reduced, True, "adamw_torch"),
        (reduced, True, "adamw_bnb_8bit"),
    ]
    while reduced > 1:
        reduced //= 2
        settings.append((reduced, True, "adamw_bnb_8bit"))
    return tuple(
        MemoryConfig(
            stage=stage,
            rung=rung,
            num_gpus=world_size,
            per_gpu_batch_size=batch,
            gradient_accumulation_steps=effective // (batch * world_size),
            gradient_checkpointing=checkpointing,
            optim=optim,
        )
        for rung, (batch, checkpointing, optim) in enumerate(settings)
    )


def is_cuda_oom(error: BaseException | str) -> bool:
    """Recognize CUDA OOM without importing Torch or retrying other failures."""
    message = str(error).lower()
    return "cuda out of memory" in message or "cuda error: out of memory" in message
