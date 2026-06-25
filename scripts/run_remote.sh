#!/usr/bin/env bash
# Launches the training loop headless on a rented box, logging to the work dir,
# and starts the web dashboard. Usage: scripts/run_remote.sh <work-dir> [loop args...]
set -euo pipefail
WORK="${1:-remote-run}"; shift || true
mkdir -p "$WORK"
( cd train && uv run python -m kairnz_train.dashboard --work "../$WORK" --host 127.0.0.1 --port 8080 \
    >"../$WORK/dashboard.log" 2>&1 & )
cd train
uv run python scripts/loop.py --work "../$WORK" --resume "$@" 2>&1 | tee "../$WORK/loop.log"
