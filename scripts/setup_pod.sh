#!/usr/bin/env bash
# One-time (and restart-safe) setup for a rented GPU pod with a /workspace volume.
#
# Installs the Rust toolchain + uv ONTO THE VOLUME so they survive pod restarts
# (RunPod resets the container filesystem on stop/start; only /workspace persists).
# Builds the release self-play/gate binaries and syncs the Python training env.
# Writes /workspace/env.sh, which every shell should source before running anything.
#
# Usage:
#   git clone <repo> /workspace/kairnz && cd /workspace/kairnz
#   bash scripts/setup_pod.sh
#   source /workspace/env.sh
#   scripts/run_remote.sh remote-run --filters 128 --blocks 10 \
#       --selfplay-games 256 --selfplay-sims 400 --epochs 10 --threads "$(nproc)"
#
# After a pod RESTART, just re-run this script (fast: toolchains + build are cached
# on the volume) OR simply `source /workspace/env.sh` if the base image kept the
# build tools.
set -euo pipefail

REPO="/workspace/kairnz"
export CARGO_HOME="/workspace/.cargo"
export RUSTUP_HOME="/workspace/.rustup"
UV_DIR="/workspace/.uv-bin"
# Official onnxruntime-gpu wheel, loaded at runtime via ort's load-dynamic feature.
# rc.10 targets ONNX Runtime 1.22, so we pin 1.22.0 for ABI match; unlike the
# pyke-bundled binary it ships native sm_90 (Hopper) kernels, so the H100 runs
# without the multi-minute cuDNN JIT the bundled build triggered.
ORT_GPU_DIR="/workspace/.ortgpu"
ORT_GPU_VERSION="1.22.0"

# System build deps (live in the container, so re-run on each setup; idempotent).
apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates build-essential pkg-config git libssl-dev

# Rust toolchain on the volume (skip if already there).
if [ ! -x "$CARGO_HOME/bin/cargo" ]; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --no-modify-path
fi

# uv on the volume (skip if already there).
if [ ! -x "$UV_DIR/uv" ]; then
    curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$UV_DIR" sh
fi
export PATH="/workspace/.cargo/bin:$UV_DIR:$PATH"

# Hopper-capable onnxruntime-gpu in a dedicated venv on the volume (skip if present).
if [ ! -x "$ORT_GPU_DIR/bin/python" ]; then
    uv venv --python 3.11 "$ORT_GPU_DIR"
fi
if ! ls "$ORT_GPU_DIR"/lib/python*/site-packages/onnxruntime/capi/libonnxruntime.so* >/dev/null 2>&1; then
    VIRTUAL_ENV="$ORT_GPU_DIR" uv pip install "onnxruntime-gpu==${ORT_GPU_VERSION}"
fi

# A sourceable env file. With ort's load-dynamic feature, ORT_DYLIB_PATH selects the
# onnxruntime to dlopen at runtime (the Hopper-capable gpu wheel above). LD_LIBRARY_PATH
# needs: that wheel's capi dir (for libonnxruntime_providers_{cuda,shared}.so, loaded by
# bare name) and the torch venv's cuDNN 9 + CUDA libs (Linux uses LD_LIBRARY_PATH, not PATH).
cat > /workspace/env.sh <<'ENVEOF'
export CARGO_HOME="/workspace/.cargo"
export RUSTUP_HOME="/workspace/.rustup"
export PATH="/workspace/.cargo/bin:/workspace/.uv-bin:$PATH"
export ORT_DYLIB_PATH="$(ls /workspace/.ortgpu/lib/python*/site-packages/onnxruntime/capi/libonnxruntime.so* | head -1)"
export LD_LIBRARY_PATH="$(dirname "$ORT_DYLIB_PATH"):$(echo /workspace/kairnz/train/.venv/lib/python*/site-packages/nvidia/*/lib /workspace/kairnz/train/.venv/lib/python*/site-packages/torch/lib | tr ' ' ':'):${LD_LIBRARY_PATH:-}"
ENVEOF

# shellcheck disable=SC1091
source /workspace/env.sh

# Build the release binaries (incremental; target/ lives on the volume).
( cd "$REPO" && cargo build --release -p kairnz-selfplay )

# Sync the Python training environment (creates train/.venv on the volume).
( cd "$REPO/train" && uv sync )

echo ""
echo "Setup complete. Toolchains, build, and venv are on /workspace (survive restarts)."
echo "In any shell, first run:  source /workspace/env.sh"
echo "Then launch, e.g.:"
echo "  cd /workspace/kairnz && scripts/run_remote.sh remote-run \\"
echo "      --filters 128 --blocks 10 --selfplay-games 256 --selfplay-sims 400 \\"
echo "      --epochs 10 --threads \"\$(nproc)\""
