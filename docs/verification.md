# Verification record

Recorded on 2026-10-07. The container image and environment/data/cache checks passed. `setup_env.sh` did not complete: its verification launcher stopped at a gate before model loading. The user then requested no further GPU tests. **Zero model forward/backward passes and zero optimizer updates ran.**

## Results

| Check | Result and evidence |
|---|---|
| Root Dockerfile build from a fresh clone | Passed; CUDA 12.8.0 system image, with no project environment or model weights baked in |
| Pinned environment installation | Passed; Python 3.12.13, Torch 2.9.0+cu128, Transformers 4.57.3 |
| Library imports | Passed: Torch, torchvision, Transformers, TorchCodec, FlashAttention, bitsandbytes, NumPy, pandas, PyArrow, PyAV, DeepSpeed, GR00T and the SO-101 config |
| bitsandbytes/PyAV installation | Passed with `--no-deps`; the Torch fingerprint remained unchanged |
| GPU visibility | Passed; one NVIDIA GeForce RTX 4090, 23.52 GiB visible memory |
| Committed dataset integrity | Passed; all 375 prepared file hashes matched |
| Both dataset readers | Passed; main: 70 episodes / 15,250 frames; grasp_window: 18 episodes / 2,055 frames; first/last room and wrist frames decoded in each subset |
| Exact HF snapshots | Passed; both pinned snapshots verified and reused without downloading |
| Managed Python cache reuse | Passed; a second container reused Python 3.12.13 in 42 ms without downloading |
| CPU tests | Passed; 131 tests, 15 warnings, 11.15 s |
| Pre-commit | Passed; Ruff lint, Ruff format and cross-platform pyproject manifest alignment |
| Shell syntax | Passed for `run_container.sh`, `setup_env.sh` and `train.sh` |
| Locked dependency resolution | Passed with `uv lock --locked`, including a separate fresh-clone container check |
| Training CLI help | Passed without GPU training, including a separate fresh-clone container check |
| Full `setup_env.sh` | Incomplete; its two-update GPU verification did not run |
| `train.sh --dry_run` | Skipped at the user's request; this command performs actual GPU optimizer updates |
| Stage 1 and Stage 2, 50-step smoke tests | Both skipped at the user's request |

The CPU suite tests the audit, deterministic dataset builder, source/data integrity, pinned snapshots, setup checks, hardware choices, stage contracts, parameter audits, resume handling and checkpoint containment. It does not establish that the complete model trains on this machine. The 15 recorded warnings are dependency deprecations.

The verified snapshots were:

- `nvidia/GR00T-N1.7-3B@2fc962b973bccdd5d8ce4f67cc63b264d6886495`
- `nvidia/Cosmos-Reason2-2B@9ce19a195e423419c349abfc86fd07178b230561`

## Fresh-clone setup log

The setup attempt used temporary verification commit `c534ff628fbb3dbaac37e38bbdfa15cc014d3648` in `/tmp/so101-fresh-clone-bd6zp8uz/repo`. This temporary commit included the approved dataset so the code could be checked before the final dataset-only commit. It is a verification candidate, not the published branch tip.

The image was `so101-groot-train:verified`, ID `sha256:bfa2d613d3c08459620e14c3b2022e4f0ecc953e2ae44b1cd381a938fd010034`. Its base was `nvidia/cuda:12.8.0-devel-ubuntu24.04@sha256:9a8fffc32a955361aa66d754e7a0cda2513052eaf5aaf8f3ba69b80578d1c6a9`.

The setup log recorded:

```text
PASS reuse Python 3.12.13 venv
PASS locked GR00T dependencies and editable package
PASS bitsandbytes and PyAV, without dependency resolution
PASS torch unchanged by bitsandbytes/PyAV install
PASS pinned versions: python=3.12.13, torch=2.9.0+cu128, transformers=4.57.3, torchcodec=0.8.0, bitsandbytes=0.50.2, numpy=1.26.4, av=16.0.1, flash_attn=2.8.3
PASS CUDA GPU: 0: NVIDIA GeForce RTX 4090 (23.52 GiB)
PASS committed dataset hashes: 375 files verified
PASS dataset main: 70 episodes / 15250 frames; first/last room+wrist decoded
PASS dataset grasp_window: 18 episodes / 2055 frames; first/last room+wrist decoded
PASS imports, exact versions, visible GPU and both datasets
PASS reused verified snapshot nvidia/GR00T-N1.7-3B@2fc962b973bccdd5d8ce4f67cc63b264d6886495 (no download)
PASS reused verified snapshot nvidia/Cosmos-Reason2-2B@9ce19a195e423419c349abfc86fd07178b230561 (no download)
PASS exactly two pinned model snapshots (reuse matching cache)
so101.hardware.GPUBusyError: GPU preflight failed: GPU 0 NVIDIA GeForce RTX 4090: 30 MiB used, 0% utilization, PID 2977 (G: unknown). Wait for the GPU owner to finish before training.
FAIL two-step Stage 1 dry forward from the official foundation
FAIL setup_env.sh at line 80
```

An earlier attempt stopped at source-integrity verification because its manifest included ignored workstation artifacts absent from a clean clone. That manifest and verifier were corrected. The subsequent attempt above reached GPU preflight and stopped at the idle host graphics context. A narrow exception for small idle graphics contexts was added with CPU tests. That change was not exercised in another GPU run after the user stopped GPU testing. No process was killed.

Temporary candidate `056696b6c828d50e4c762e015c8e08b2fc8539d8`, which includes that change and the dataset, passed the later container `uv lock --locked` and `train.sh --help` checks. These checks do not perform model forward/backward passes. Setup's cleanup of its own successful verification checkpoints has CPU coverage; the real setup never reached that success path.

## Measurements

| Machine | Peak allocated/reserved training memory | Seconds per optimizer update | Stage 1 / Stage 2 / total ETA |
|---|---|---|---|
| One RTX 4090 | Unmeasured | Unmeasured | Unmeasured |
| Two H200s | Unmeasured | Unmeasured | Unmeasured |

GPU visibility and idle preflight memory are not measurements of model training memory. No claim that Stage 2 fits a 4090, or any training-duration estimate, follows from this verification.

GPU execution remains unverified for: complete model loading and its live weight/parameter audits; both stages' forward/backward passes and optimizer updates; the two-update setup check and training dry run; both 50-step smoke tests and finite losses; actual memory-ladder fit and CUDA OOM recovery; checkpoint save/reload/resume with the real model; training data throughput; multi-GPU DDP and the effective batch on two H200s; peak memory, step timings and full-plan ETAs. Full training and policy evaluation were not run.

## H200 hand-off compatibility

`PROMPT_H200.md` preserves the user's supplied hand-off verbatim. Its requested `setup_env.sh --venv ... --base_model_path ...` interface is not implemented in this revision: setup currently accepts no arguments and manages the checkout's `.venv`, and `train.sh` selects that same environment. The model resolver uses the pinned HF cache snapshots rather than accepting an arbitrary local base directory. The H200 agent must implement and CPU-test reuse of the prepared external environment and verified official model path before executing that setup command. Do not substitute a no-argument setup on the prepared H200 machine, since it does not satisfy the hand-off's existing-environment requirements.

## Commands and local evidence

The fresh-clone image/setup attempt used these commands from the temporary checkout, with shared host HF and uv caches and a separate verification output mount:

```bash
docker build -t so101-groot-train:verified .
./run_container.sh --image so101-groot-train:verified --output_root /external_storage/models/so101_repo_verification -- ./setup_env.sh
```

The CPU checks were:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -p pytest_timeout tests/so101/ -m "not gpu" --timeout=300
.venv/bin/pre-commit run --all-files
bash -n run_container.sh setup_env.sh train.sh
uv lock --locked
./train.sh --help
```

The original local logs are outside the repository:

- `/tmp/so101-fresh-docker-build.log`
- `/tmp/so101-fresh-setup.log`
- `/tmp/so101-python-cache-reuse.log`
- `/tmp/so101-final-cpu-tests.log`
- `/tmp/so101-final-precommit.log`
- `/tmp/so101-final-lock.log`
- `/tmp/so101-fresh-cpu-cli.log`

The setup excerpt above preserves the relevant outcomes if those temporary logs are later removed. Models, checkpoint files, Python environments and scratch outputs are not part of this repository.
