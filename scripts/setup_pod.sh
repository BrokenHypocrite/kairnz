#!/usr/bin/env bash
# One-time setup for a rented GPU pod (Ubuntu with CUDA + cuDNN, e.g. a RunPod or
# Vast.ai PyTorch template). Run from the repo root after cloning. Installs the
# Rust toolchain + uv, builds the release self-play/gate binaries, and syncs the
# Python training env. Takes roughly 15-20 minutes the first time; on a
# persistent volume it only needs to run once.
#
# Usage:
#   git clone <your-repo-url> kairnz && cd kairnz
#   bash scripts/setup_pod.sh
#   scripts/run_remote.sh remote-run --filters 128 --blocks 10 \
#       --selfplay-games 256 --selfplay-sims 400 --max-batch 256 \
#       --leaves-per-step 8 --threads "$(nproc)"
set -euo pipefail

# System build dependencies. libssl-dev is required so the Rust openssl-sys
# dependency builds; pkg-config + build-essential cover the rest.
apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates build-essential pkg-config git libssl-dev

# Rust toolchain (skip if already present).
if ! command -v cargo >/dev/null 2>&1; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
fi
# shellcheck disable=SC1091
[ -f "$HOME/.cargo/env" ] && . "$HOME/.cargo/env"
export PATH="$HOME/.cargo/bin:$PATH"

# uv (Python package/venv manager).
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

# Build the release Rust binaries (self-play + gate).
cargo build --release -p kairnz-selfplay

# Sync the Python training environment (PyTorch + deps).
( cd train && uv sync )

echo ""
echo "Setup complete. GPU inference uses the pod's system cuDNN automatically."
echo "Launch training, e.g.:"
echo "  scripts/run_remote.sh remote-run --filters 128 --blocks 10 \\"
echo "      --selfplay-games 256 --selfplay-sims 400 --max-batch 256 \\"
echo "      --leaves-per-step 8 --threads \"\$(nproc)\""
