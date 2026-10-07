# syntax=docker/dockerfile:1
# Training dependencies and the two model snapshots are installed by setup_env.sh.
FROM nvidia/cuda:12.8.0-devel-ubuntu24.04

ARG UV_VERSION=0.12.19
ENV DEBIAN_FRONTEND=noninteractive \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    HF_HOME=/root/.cache/huggingface \
    HF_HUB_CACHE=/root/.cache/huggingface/hub \
    UV_CACHE_DIR=/root/.cache/uv \
    UV_PYTHON_INSTALL_DIR=/root/.cache/uv/python \
    CUDA_HOME=/usr/local/cuda \
    PYTHONUNBUFFERED=1

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl git build-essential cmake ninja-build pkg-config \
      python3 python3-dev ffmpeg libavcodec-dev libavformat-dev libavutil-dev \
      libavdevice-dev libavfilter-dev libswresample-dev libswscale-dev \
      libaio-dev libnuma-dev libegl1 libgl1 \
    && rm -rf /var/lib/apt/lists/* \
    && curl --retry 5 --fail --silent --show-error --location \
      "https://astral.sh/uv/${UV_VERSION}/install.sh" \
      | env UV_INSTALL_DIR=/usr/local/bin sh

WORKDIR /workspace/so101-groot-train
CMD ["/bin/bash"]
