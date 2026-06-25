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

# A sourceable env file. Three things the ONNX Runtime CUDA provider needs on Linux:
#   1. cuDNN + CUDA libs on LD_LIBRARY_PATH (Linux uses LD_LIBRARY_PATH, not PATH).
#   2. target/release on LD_LIBRARY_PATH so ORT can dlopen libonnxruntime_providers_shared.so
#      (its broker lib, loaded by bare name; the executable's own dir is NOT searched).
#   3. A persistent, large CUDA JIT cache: the bundled provider ships no sm_90 (Hopper)
#      cubin and cuDNN 9 runtime-compiles conv engines, a multi-minute first-run cost.
#      The compiled cache is ~260MB (over the 256MB default), so raise the cap and put it
#      on the volume so the compile happens once, not every run or pod restart.
cat > /workspace/env.sh <<'ENVEOF'
export CARGO_HOME="/workspace/.cargo"
export RUSTUP_HOME="/workspace/.rustup"
export PATH="/workspace/.cargo/bin:/workspace/.uv-bin:$PATH"
export LD_LIBRARY_PATH="/workspace/kairnz/target/release:$(echo /workspace/kairnz/train/.venv/lib/python*/site-packages/nvidia/*/lib /workspace/kairnz/train/.venv/lib/python*/site-packages/torch/lib | tr ' ' ':'):${LD_LIBRARY_PATH:-}"
export CUDA_CACHE_PATH="/workspace/.nv_cache"
export CUDA_CACHE_MAXSIZE="2147483648"
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
