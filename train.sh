#!/usr/bin/env bash
set -euo pipefail
REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"
if [[ ! -x "$REPO_DIR/.venv/bin/python" ]]; then
  echo 'Run ./setup_env.sh inside the container first.' >&2
  exit 1
fi
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export NO_ALBUMENTATIONS_UPDATE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec "$REPO_DIR/.venv/bin/python" -m so101.launcher "$@"
