#!/usr/bin/env bash
# Idempotent setup inside the CUDA container. Never prints HF_TOKEN.
set -Eeuo pipefail
trap 'printf "FAIL setup_env.sh at line %s\n" "$LINENO" >&2' ERR

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
cd -- "$repo_root"
if (($#)); then
    printf 'Usage: ./setup_env.sh\n' >&2
    exit 2
fi
command -v uv >/dev/null || { printf 'FAIL uv: use the root Dockerfile\n' >&2; exit 1; }
command -v python3 >/dev/null || { printf 'FAIL bootstrap Python\n' >&2; exit 1; }

check_step() {
    local label="$1"
    shift
    if "$@"; then
        printf 'PASS %s\n' "$label"
    else
        printf 'FAIL %s\n' "$label" >&2
        return 1
    fi
}

readarray -t versions < <(python3 - <<'PY'
import json
from pathlib import Path
versions = json.loads(Path("so101/provenance/environment_versions.json").read_text())["versions"]
for key in ("python", "bitsandbytes", "av"):
    print(versions[key])
PY
)
if [[ ${#versions[@]} != 3 ]]; then
    printf 'FAIL environment version manifest\n' >&2
    exit 1
fi
python_pin="${versions[0]}"
bnb_pin="${versions[1]}"
av_pin="${versions[2]}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-300}"
export UV_PROJECT_ENVIRONMENT="$repo_root/.venv"

if [[ -x .venv/bin/python ]] && [[ "$(.venv/bin/python -c 'import platform; print(platform.python_version())')" == "$python_pin" ]]; then
    printf 'PASS reuse Python %s venv\n' "$python_pin"
else
    # Existing environments can have broken host-only interpreter symlinks inside
    # the container. Recreate the environment without touching repository data.
    check_step "Python $python_pin venv" uv venv --clear --python "$python_pin" .venv
fi
check_step 'locked GR00T dependencies and editable package' \
    uv sync --frozen --no-dev --inexact --python "$python_pin"

torch_before="$(.venv/bin/python -m so101.setup_checks --torch-fingerprint)"
check_step 'bitsandbytes and PyAV, without dependency resolution' \
    uv pip install --python .venv/bin/python --no-deps \
    "bitsandbytes==$bnb_pin" "av==$av_pin"
torch_after="$(.venv/bin/python -m so101.setup_checks --torch-fingerprint)"
if [[ "$torch_before" != "$torch_after" ]]; then
    printf 'FAIL: torch changed while installing bitsandbytes/PyAV\n' >&2
    exit 1
fi
printf 'PASS torch unchanged by bitsandbytes/PyAV install\n'
check_step 'imports, exact versions, visible GPU and both datasets' \
    .venv/bin/python -m so101.setup_checks
check_step 'exactly two pinned model snapshots (reuse matching cache)' \
    .venv/bin/python -m so101.weights --download

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
output_root="${OUTPUT_ROOT:-/external_storage/models}"
mkdir -p -- "$output_root"
dry_output="$(mktemp -d "$output_root/so101-setup-dry-run-XXXXXXXX")"
printf 'SO-101 setup verification scratch v1\n' > "$dry_output/.so101-setup-verification"
printf 'Setup verification artifacts: %s\n' "$dry_output"
# The launcher loads only the verified official foundation for Stage 1 and
# performs two forward/backward optimizer updates, exercising the memory ladder.
check_step 'two-step Stage 1 dry forward from the official foundation' \
    .venv/bin/python -m so101.launcher --stage 1 --dry_run --output_root "$dry_output"
check_step 'ephemeral setup checkpoints removed; logs and metrics retained' \
    .venv/bin/python -m so101.setup_checks --cleanup-setup-checkpoints "$dry_output"
printf 'PASS setup complete; run ./train.sh\n'
