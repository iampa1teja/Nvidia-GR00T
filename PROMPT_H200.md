# H200 hand-off: verify and train

You are running on the H200 training machine (2× H200, no Docker, no Isaac Sim). This repo is a self-contained GR00T N1.7 training repo for the SO-101 pick-place policy. Read README.md, UPSTREAM.md, datasets/pick_place_groot/README.md, setup_env.sh, train.sh and so101/ first.

## Machine layout (already prepared; don't recreate)
- `source ~/pavanteja/env.sh` sets BASE=~/pavanteja, HF_HOME=~/pavanteja/hf_cache, and activates the venv ~/pavanteja/venv (Python 3.12; torch 2.9.0+cu128, transformers 4.57.3, torchcodec 0.8.0, av 16.0.1, bitsandbytes already installed).
- Official base weights: ~/pavanteja/models/gr00t_n17_3b_base (nvidia/GR00T-N1.7-3B). The Cosmos-Reason2-2B backbone is in the HF cache. Revisions are in ~/pavanteja/model_revisions.txt.
- This repo is cloned at ~/pavanteja/<repo>. Outputs go to ~/pavanteja/runs (or the --output_root I give).

## Rules
- Never recreate the venv. Never change torch/transformers/torchcodec versions. If something requires that, STOP and ask.
- Initialise ONLY from the official base (nvidia/GR00T-N1.7-3B). Never use any other checkpoint.
- The machine may be shared: check nvidia-smi first. Never kill processes you didn't start. If the GPUs are busy, STOP and report.
- Don't modify the dataset or upstream gr00t/ files. Fix bugs only in so101/, setup_env.sh, train.sh, the tests and the docs.
- Commits: author/committer iampa1teja <ppavanteja@gmail.com>; NO "Co-Authored-By" / "Co-authored-by" / "Generated with" lines. Never commit checkpoints, outputs, caches or secrets. Push to main after `git pull --ff-only`.

## Steps
1. **Environment**
   - `source ~/pavanteja/env.sh`, then `./setup_env.sh --venv ~/pavanteja/venv --base_model_path ~/pavanteja/models/gr00t_n17_3b_base`.
   - It must reuse the venv and the caches: no torch reinstall, no re-download.
   - Report the PASS/FAIL table, plus the base revision vs model_revisions.txt.
2. **Tests:** run the CPU test suite (tests/so101/); all must pass.
3. **Dry run:** `./train.sh --dry_run`. Check that the resolved plan shows:
   - Stage 1: adapters only, 25k steps, lr 1e-4;
   - Stage 2: DiT + adapters, 60k steps, DiT 3e-5 / adapters 1e-4 (or the documented single-LR fallback);
   - mixture main 1.0 / grasp_window 2.0;
   - 2 GPUs, effective batch ≥ 64.
4. **Smoke tests** (50 steps each, separate output dir ~/pavanteja/runs/smoke_<timestamp>):
   - Stage 1 from the official base; Stage 2 from the Stage 1 smoke output (its weights are only for testing the pipeline).
   - For each, report: the audited parameter table (trainable modules and counts match the stage), missing/unexpected keys = 0, peak memory per GPU, s/step, finite losses, a checkpoint save + reload that works, and the data loading rate (no dataloader stalls).
   - Tune the batch/workers for 2× H200 if memory allows (keep effective batch ≥ 64); record the chosen defaults in train.sh/hardware.py and commit them.
5. **Fix and re-run** anything that fails (within the rules), then repeat steps 3–4. Commit fixes with clear messages.
6. **Report and STOP before the full run:**
   - setup results; tests; dry-run plan;
   - the smoke metrics table (stage, trainable params, peak memory/GPU, s/step, loss at steps 1/50);
   - the chosen batch config;
   - ETA for the full plan (Stage 1 + Stage 2, plus checkpoint writes);
   - disk needed vs free at the output root;
   - commits made.
   Wait for my "go".
7. **On "go":**
   - launch `./train.sh --stage all --output_root ~/pavanteja/runs/<run_name>` in tmux or with nohup, logging to <run>/train.log;
   - confirm it passed 100 steps with finite losses;
   - report the PID/tmux session, the log path and the ETA;
   - give the command to follow the log and the command to resume after an interruption (`./train.sh --resume …`).
8. **Hand-off:** print the eval commands from train.sh's final message (they run on the 4090 machine in the Sim-to-Real repo, not here).
