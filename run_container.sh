#!/usr/bin/env bash
# Mount the checkout, checkpoints and the host's pre-warmed download caches.
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
image="${SO101_IMAGE:-so101-groot-train:latest}"
output_root="${OUTPUT_ROOT:-/external_storage/models}"
shm_size="${SHM_SIZE:-16g}"

usage() {
    cat <<'EOF'
Usage: ./run_container.sh [--image IMAGE] [--output_root HOST_DIRECTORY]
                          [--shm-size SIZE] [-- COMMAND...]

Defaults: image so101-groot-train:latest; output /external_storage/models;
shared memory 16g. Environment overrides: SO101_IMAGE, OUTPUT_ROOT, SHM_SIZE.
HF_HOME and UV_CACHE_DIR select host cache directories. HF_TOKEN passes by name.
The checkout is mounted at /workspace/so101-groot-train and outputs at
/external_storage/models. Training defaults to its so101_official_n17 subdirectory.
With no command, opens an interactive Bash shell.
EOF
}

while (($#)); do
    case "$1" in
        --image|--output_root|--shm-size)
            if (($# < 2)) || [[ "$2" == --* ]]; then
                printf 'FAIL: %s requires a value\n' "$1" >&2
                exit 2
            fi
            case "$1" in
                --image) image="$2" ;;
                --output_root) output_root="$2" ;;
                --shm-size) shm_size="$2" ;;
            esac
            shift 2
            ;;
        --help|-h) usage; exit 0 ;;
        --) shift; break ;;
        *) printf 'FAIL: unknown option %s\n' "$1" >&2; usage >&2; exit 2 ;;
    esac
done

command -v docker >/dev/null || { printf 'FAIL: docker is unavailable\n' >&2; exit 1; }
host_hf_cache="${HF_HOME:-$HOME/.cache/huggingface}"
host_uv_cache="${UV_CACHE_DIR:-$HOME/.cache/uv}"
mkdir -p -- "$output_root" "$host_hf_cache" "$host_uv_cache"
output_root="$(cd -- "$output_root" && pwd -P)"

args=(run --rm --gpus all --shm-size "$shm_size"
    -v "$repo_root:/workspace/so101-groot-train"
    -v "$output_root:/external_storage/models"
    -v "${HF_HOME:-$HOME/.cache/huggingface}:/root/.cache/huggingface"
    -v "${UV_CACHE_DIR:-$HOME/.cache/uv}:/root/.cache/uv"
    --env HF_HOME=/root/.cache/huggingface
    --env HF_HUB_CACHE=/root/.cache/huggingface/hub
    --env UV_CACHE_DIR=/root/.cache/uv
    --env UV_PYTHON_INSTALL_DIR=/root/.cache/uv/python
    --env OUTPUT_ROOT=/external_storage/models/so101_official_n17
    --env "SO101_HOST_OUTPUT_ROOT=$output_root"
    --env HF_TOKEN
    --workdir /workspace/so101-groot-train)
if [[ -t 0 && -t 1 ]]; then
    args+=(-it)
elif [[ $# == 0 ]]; then
    args+=(-i)
fi
args+=("$image")
if (($#)); then
    args+=("$@")
else
    args+=(/bin/bash)
fi
exec docker "${args[@]}"
