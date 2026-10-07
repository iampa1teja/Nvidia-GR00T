"""GPU planning tests use recorded-shaped telemetry and make no CUDA allocations."""

import subprocess
from unittest.mock import Mock

import pytest
from so101.hardware import (
    CapacityExhausted,
    GPUBusyError,
    GPUInfo,
    GPUProcess,
    HardwareDetectionError,
    HardwareError,
    detect_gpus,
    is_cuda_oom,
    memory_ladder,
    preflight_gpus,
)


def gpu(index=0, name="NVIDIA GeForce RTX 4090", memory=24564, **overrides):
    return GPUInfo(
        index=index,
        uuid=f"GPU-{index:08d}-aaaa-bbbb-cccc-dddddddddddd",
        name=name,
        total_memory_mib=memory,
        used_memory_mib=overrides.pop("used_memory_mib", 0),
        utilization_percent=overrides.pop("utilization_percent", 0),
        **overrides,
    )


def telemetry(monkeypatch, records, *, process_xml="", xml=None):
    csv = "\n".join(
        f'{item.index}, {item.uuid}, "{item.name}", {item.total_memory_mib}, '
        f"{item.used_memory_mib}, {item.utilization_percent}"
        for item in records
    )
    document = (
        "<nvidia_smi_log>"
        + "".join(
            f"<gpu><uuid>{item.uuid}</uuid><processes>{process_xml}</processes></gpu>"
            for item in records
        )
        + "</nvidia_smi_log>"
        if xml is None
        else xml
    )
    run = Mock(
        side_effect=[
            subprocess.CompletedProcess(["nvidia-smi"], 0, csv, ""),
            subprocess.CompletedProcess(["nvidia-smi"], 0, document, ""),
        ]
    )
    monkeypatch.setattr(subprocess, "run", run)
    return run


def test_gpu_detection_no_gpu_allocation(monkeypatch):
    expected = gpu(name="GPU, with quoted comma")
    run = telemetry(monkeypatch, [expected])
    assert detect_gpus(environ={}) == (expected,)
    assert run.call_count == 2
    assert run.call_args_list[0].args[0] == [
        "nvidia-smi",
        "--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    assert run.call_args_list[1].args[0] == ["nvidia-smi", "-q", "-x"]
    for call in run.call_args_list:
        assert call.kwargs == {
            "check": True,
            "capture_output": True,
            "text": True,
            "timeout": 10.0,
        }


def test_visible_gpu_order_and_uuid_prefix(monkeypatch):
    first, second = gpu(), gpu(1)
    telemetry(monkeypatch, [first, second])
    assert detect_gpus(environ={"CUDA_VISIBLE_DEVICES": f"GPU-00000001, {first.index}"}) == (
        second,
        first,
    )


@pytest.mark.parametrize("visible", ["", " ", "-1"])
def test_disabled_cuda_visibility_never_calls_smi(monkeypatch, visible):
    run = Mock(side_effect=AssertionError("No nvidia-smi expected"))
    monkeypatch.setattr(subprocess, "run", run)
    assert detect_gpus(environ={"CUDA_VISIBLE_DEVICES": visible}) == ()
    run.assert_not_called()


@pytest.mark.parametrize("visible", ["9", "0,0", "GPU-0", "MIG-uuid", "potato", "0,"])
def test_invalid_visible_devices_fail_closed(monkeypatch, visible):
    telemetry(monkeypatch, [gpu(), gpu(1)])
    with pytest.raises(HardwareDetectionError):
        detect_gpus(environ={"CUDA_VISIBLE_DEVICES": visible})


def test_graphics_process_makes_preflight_busy_without_stopping_it(monkeypatch):
    run = telemetry(
        monkeypatch,
        [gpu()],
        process_xml=(
            "<process_info><pid>2345</pid><type>G</type>"
            "<process_name>/usr/bin/Xorg</process_name><used_memory>N/A</used_memory>"
            "</process_info>"
        ),
    )
    detected = detect_gpus(environ={})
    assert detected[0].processes == (GPUProcess(2345, "/usr/bin/Xorg", "G", None),)
    with pytest.raises(GPUBusyError, match="PID 2345") as raised:
        preflight_gpus(detected)
    assert raised.value.gpus == detected
    # Detect/preflight only called read-only nvidia-smi queries; no kill command.
    assert run.call_count == 2


def test_compute_process_memory_parsing(monkeypatch):
    telemetry(
        monkeypatch,
        [gpu()],
        process_xml=(
            "<process_info><pid>1234</pid><type>C+G</type>"
            "<process_name>python</process_name><used_memory>1024 MiB</used_memory>"
            "</process_info>"
        ),
    )
    assert detect_gpus(environ={})[0].processes[0].used_memory_mib == 1024


@pytest.mark.parametrize("name", ["/usr/lib/xorg/Xorg", "unknown"])
def test_tiny_idle_graphics_context_passes_without_touching_owner(name):
    display = gpu(used_memory_mib=30, processes=(GPUProcess(2977, name, "G", 18),))
    assert preflight_gpus([display]) == (display,)


@pytest.mark.parametrize(
    "processes,used,utilization",
    [
        ((GPUProcess(1, "python", "C", 0),), 0, 0),
        ((GPUProcess(1, "python", "C+G", 0),), 0, 0),
        ((GPUProcess(1, "unknown", "unknown", 0),), 0, 0),
        ((GPUProcess(1, "Xorg", "G", None),), 30, 0),
        ((GPUProcess(1, "renderer", "G", 129),), 30, 0),
        ((GPUProcess(1, "Xorg", "G", 18),), 129, 0),
        ((GPUProcess(1, "Xorg", "G", 18),), 30, 2),
        ((GPUProcess(1, "G1", "G", 65), GPUProcess(2, "G2", "G", 64)), 30, 0),
        ((GPUProcess(1, "Xorg", "G", 18), GPUProcess(2, "python", "C", 0)), 30, 0),
    ],
)
def test_graphics_allowance_never_accepts_workloads_or_activity(processes, used, utilization):
    with pytest.raises(GPUBusyError):
        preflight_gpus(
            [gpu(used_memory_mib=used, utilization_percent=utilization, processes=processes)]
        )


def test_smi_missing_or_timeout_is_useful_error(monkeypatch):
    for error in (FileNotFoundError("nvidia-smi"), subprocess.TimeoutExpired("nvidia-smi", 10)):
        monkeypatch.setattr(subprocess, "run", Mock(side_effect=error))
        with pytest.raises(HardwareDetectionError, match="Unable to query nvidia-smi"):
            detect_gpus(environ={})


@pytest.mark.parametrize(
    "document",
    ["not xml", "<nvidia_smi_log/>", "<nvidia_smi_log><gpu><uuid/></gpu></nvidia_smi_log>"],
)
def test_missing_process_telemetry_fails_closed(monkeypatch, document):
    telemetry(monkeypatch, [gpu()], xml=document)
    with pytest.raises(HardwareDetectionError):
        detect_gpus(environ={})


@pytest.mark.parametrize("used,utilization", [(513, 0), (0, 11), (2000, 90)])
def test_busy_without_visible_pid_is_rejected(used, utilization):
    with pytest.raises(GPUBusyError, match="GPU 0"):
        preflight_gpus([gpu(used_memory_mib=used, utilization_percent=utilization)])


def test_selected_idle_gpu_passes_at_thresholds():
    idle = gpu(used_memory_mib=512, utilization_percent=10)
    assert preflight_gpus([idle]) == (idle,)


def test_busy_secondary_gpu_rejects_whole_training():
    with pytest.raises(GPUBusyError, match="GPU 1"):
        preflight_gpus([gpu(), gpu(1, used_memory_mib=2000)])


def test_no_visible_gpu_preflight_and_ladder_reject():
    with pytest.raises(HardwareError, match="No visible"):
        preflight_gpus([])
    with pytest.raises(HardwareError, match="without a visible GPU"):
        memory_ladder(1, [])


@pytest.mark.parametrize("stage", [1, 2])
def test_4090_full_ladder_preserves_batch_and_training_scope(stage):
    rungs = memory_ladder(stage, [gpu()])
    assert [(r.per_gpu_batch_size, r.gradient_checkpointing, r.optim) for r in rungs] == [
        (32, False, "adamw_torch"),
        (16, False, "adamw_torch"),
        (16, True, "adamw_torch"),
        (16, True, "adamw_bnb_8bit"),
        (8, True, "adamw_bnb_8bit"),
        (4, True, "adamw_bnb_8bit"),
        (2, True, "adamw_bnb_8bit"),
        (1, True, "adamw_bnb_8bit"),
    ]
    assert {r.effective_batch_size for r in rungs} == {32}
    assert {r.stage for r in rungs} == {stage}
    assert all(r.global_batch_size == r.per_gpu_batch_size for r in rungs)
    assert rungs[0].as_dict()["effective_batch_size"] == 32
    assert rungs[0].as_dict()["global_batch_size"] == 32


@pytest.mark.parametrize("stage", [1, 2])
def test_two_h200_global_batch_and_accumulation(stage):
    rungs = memory_ladder(stage, [gpu(0, "NVIDIA H200", 143771), gpu(1, "NVIDIA H200", 143771)])
    assert {r.effective_batch_size for r in rungs} == {64}
    assert all(r.num_gpus == 2 for r in rungs)
    assert rungs[0].global_batch_size == 64
    assert rungs[1].global_batch_size == 32
    assert rungs[1].gradient_accumulation_steps == 2
    assert rungs[-1].global_batch_size == 2
    assert rungs[-1].gradient_accumulation_steps == 32


def test_heterogeneous_inventory_uses_smallest_gpu_and_rounds_target_up():
    rungs = memory_ladder(
        2, [gpu(memory=143771), gpu(1, memory=12000)], target_effective_batch_size=65
    )
    assert rungs[0].per_gpu_batch_size == 8
    assert {r.effective_batch_size for r in rungs} == {80}
    assert all(r.effective_batch_size >= 65 for r in rungs)


@pytest.mark.parametrize("target", [0, -1, True, 1.5])
def test_invalid_target_batch(target):
    with pytest.raises(ValueError, match="positive integer"):
        memory_ladder(1, [gpu()], target_effective_batch_size=target)


def test_capacity_exhaustion_reports_clean_stage2_and_failed_stage1():
    stage2 = CapacityExhausted(2, memory_ladder(2, [gpu()]))
    assert stage2.clean_exit
    assert stage2.exit_code == 0
    assert "Retain the completed Stage 1" in str(stage2)
    assert "microbatch 1" in str(stage2)
    stage1 = CapacityExhausted(1, memory_ladder(1, [gpu()]))
    assert not stage1.clean_exit
    assert stage1.exit_code == 1
    assert stage1.attempted[-1].per_gpu_batch_size == 1
    with pytest.raises(ValueError):
        CapacityExhausted(2, [])
    with pytest.raises(ValueError):
        CapacityExhausted(2, memory_ladder(1, [gpu()]))


@pytest.mark.parametrize(
    "error,expected",
    [
        (RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB"), True),
        ("torch.OutOfMemoryError: CUDA out of memory", True),
        ("CUDA error: out of memory", True),
        (MemoryError("CPU allocation failed"), False),
        (RuntimeError("Missing checkpoint weights"), False),
        ("CUDNN_STATUS_NOT_SUPPORTED", False),
    ],
)
def test_only_cuda_oom_is_retryable(error, expected):
    assert is_cuda_oom(error) is expected
