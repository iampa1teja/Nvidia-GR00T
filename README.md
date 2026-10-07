# SO-101 pick-place training with GR00T N1.7

Self-contained two-stage training for an SO-101 arm, using the original pick-place demonstrations. The training path uses the dataset committed in this repository and has no dependency on the Sim-to-Real workshop repository.

## Four-step quickstart

1. Clone this repository and enter its directory, replacing `<new-repo-url>` with its Git URL:

   ```bash
   git clone <new-repo-url> so101-groot-train
   cd so101-groot-train
   ```

2. Build and start the container:

   ```bash
   docker build -t so101-groot-train .
   ./run_container.sh
   ```
3. Inside the container, run `./setup_env.sh`.
4. Run `./train.sh`.

The checkout is mounted at `/workspace/so101-groot-train`. Outputs default to the host `/external_storage/models/so101_official_n17`; `run_container.sh --output_root /path` changes the host mount, placing the default run under `/path/so101_official_n17`. Host caches use `${HF_HOME:-$HOME/.cache/huggingface}` and `${UV_CACHE_DIR:-$HOME/.cache/uv}`. Matching, complete pinned HF snapshots are checksum-verified and reused without re-downloading. Set `HF_TOKEN` in the host environment if the gated Cosmos backbone needs authentication; tokens are never written to the repository.

Setup verifies imports, exact versions, GPU visibility, every committed dataset file hash, both dataset readers and both official snapshots. It then performs two Stage 1 forward/backward optimizer updates from the official foundation. These checks exercise the parameter audits and memory ladder. Setup removes its own temporary verification checkpoints after success and retains their logs, run configuration and metrics under the output mount. Full training checkpoints remain retained.

## Initialization and training

Only `nvidia/GR00T-N1.7-3B` initializes Stage 1, with `nvidia/Cosmos-Reason2-2B` as its gated backbone. Both exact revisions are recorded in [UPSTREAM.md](UPSTREAM.md) and in every stage's run configuration. No 200k, course, or other fine-tuned checkpoint initializes Stage 1.

`NEW_EMBODIMENT` uses fresh SO-101 state/action adapters. The data has room and wrist cameras, five arm joints and one gripper, absolute actions in the original recorded units, and a 16-frame action horizon. Foundation model padding capacity is 40 frames.

| Stage | Trainable modules | Steps | Learning rate |
|---|---|---:|---|
| 1: align | State/action adapters and vision-language adapter; DiT, LLM and vision frozen | 25,000 | 1e-4 |
| 2: main | DiT plus adapters; LLM and vision frozen | 60,000 | DiT 3e-5; adapters 1e-4 |

Adapters first learn the new arm's input/output coordinates while retaining the pretrained DiT. Stage 2 starts from the Stage 1 final checkpoint and adapts the full DiT. Both stages use 1,000 warmup steps, a cosine schedule, checkpoints every 5,000 steps, and the main/grasp-window mixture at 1:2. All scheduled checkpoints are retained. The pinned upstream trainer does not support EMA, so EMA is not enabled. Stage 1 reports its loss trend and never extends itself automatically.

```bash
./train.sh --stage 1
./train.sh --stage 2
./train.sh --stage all --resume
./train.sh --output_root /external_storage/models/so101_pick_place
./train.sh --dry_run
```

Resume must use checkpoints from the matching stage and preserve the audited model/data/training settings. Each stage writes `run_config.json`, its parameter audit, training log, and timing/memory measurements. The launcher prints evaluation commands with action horizon 8, `--gripper_latch`, and `--log_gripper`.

## Hardware and verification

Python 3.12.13, Torch 2.9.0+cu128 and Transformers 4.57.3 match the saved run's version evidence. The root Dockerfile supplies CUDA 12.8 system dependencies for RTX 4090 and H200; setup installs the pinned Python environment.

An RTX 4090 uses a memory ladder: smaller microbatches, gradient accumulation, block checkpointing, then 8-bit AdamW. The full DiT remains trainable in Stage 2. If the complete ladder cannot fit Stage 2, the launcher preserves Stage 1 and exits with an explicit capacity report. Two H200s use distributed training with an effective batch of at least 64. GPU preflight does not kill processes or start teleoperation.

The fresh-clone image build, dependency/import checks, GPU visibility, all 375 dataset file hashes, both dataset readers and both pinned HF snapshot cache checks passed. Setup stopped before loading the model; zero model forward/backward passes or optimizer updates ran. Further GPU testing was stopped at the user's request, so `train.sh --dry_run` and both 50-step smoke tests were skipped. Peak training memory, seconds per step and full-plan ETAs remain unmeasured on both the RTX 4090 and two H200s. The [verification report](docs/verification.md) records the checks and their limits.

See [Training on the H200](PROMPT_H200.md). The supplied prompt is preserved verbatim; its external-venv and local-model setup flags still need implementation, as recorded in the [H200 compatibility note](docs/verification.md#h200-hand-off-compatibility).

## Dataset provenance and approval

The source is the original `pick_place_v1`: 75 episodes, 16,448 frames, 30 fps, with synchronized cube-pose sidecars. Raw data is read-only. There are no new demonstrations or domain randomisation. The [grasp audit](audit/summary.md) records the metrics, thresholds, limitations, episode lists and source hashes.

The approved dataset is committed under `datasets/pick_place_groot/`: **70 main episodes / 15,250 frames**, and **18 grasp-window episodes / 2,055 frames**. Only episodes **0, 3, 6, 51, 72** are excluded; hesitant and threshold-unresolved episodes remain in main as explicitly approved. Grasp windows run from close minus 1.5 seconds through lift plus 1 second. Episodes are renumbered with source mappings. All videos were decoded and checked; clipped videos preserve the source YUV and RGB pixels. Dataset size is **339.01 MB**, with a largest file of **12.12 MB**. These are ordinary Git files, below the 1 GB total and 95 MB per-file limits. The [dataset README](datasets/pick_place_groot/README.md) records counts, decisions, thresholds and the reproducible build command.

The preserved 200k checkpoint is evaluation-only and lives outside this repository. Evaluation runs in the Sim-to-Real repository; training does not.

## Code history and the optional dataset removal

Code, setup, documentation, audits and tests are committed before the dataset. The final commit adds only `datasets/pick_place_groot/` as ordinary Git files. The default main checkout includes that commit and supports the four-step training workflow.

When `HEAD` is that dataset-only final commit, dropping it gives a code checkout without the prepared dataset. Save your local work first: the following command discards uncommitted changes and moves the branch back by one commit.

```bash
git reset --hard HEAD~1
```

This removes the committed prepared dataset from the checkout. Restore the dataset commit or rebuild the approved dataset using the recorded build command before running setup or training again. The original raw recordings and external model/cache directories are outside this checkout and are unaffected.

## Tests and attribution

```bash
uv sync --frozen --extra dev --inexact --python 3.12.13
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -p pytest_timeout tests/so101/ -m "not gpu" --timeout=300
```

This is a modified copy of NVIDIA Isaac-GR00T at commit `51d4c89f72fda44cbf77285c6a8114b52676b8a1`. Upstream `gr00t/` code is unchanged; adaptations live under `so101/` and use runtime patches. NVIDIA [LICENSE](LICENSE), [ATTRIBUTIONS.md](ATTRIBUTIONS.md), and the added [NOTICE](NOTICE) apply. [UPSTREAM.md](UPSTREAM.md) records provenance; the original upstream overview is preserved in [getting_started/upstream_readme.md](getting_started/upstream_readme.md).
