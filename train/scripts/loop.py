"""AlphaZero orchestration loop: self-play -> train -> gate -> promote, repeated.

Runs the Rust self-play and gate binaries as subprocesses (with cuDNN on PATH)
and trains in-process between rounds. Maintains best.onnx and a metrics log.
"""

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import torch

from kairnz_train.model import KairnzNet
from kairnz_train.onnx_export import export_onnx
from kairnz_train.orchestrate import (
    PROMOTE_THRESHOLD,
    select_window,
    should_promote,
    subprocess_env,
    train_candidate,
)

# Repository root relative to this script (train/scripts/loop.py -> repo root).
REPO_ROOT = Path(__file__).resolve().parents[2]


def _run_rust(bin_name: str, extra_args: list[str]) -> str:
    """Runs a release Rust binary with cuDNN on PATH; returns its stdout."""
    cmd = [
        "cargo", "run", "--release", "-p", "kairnz-selfplay",
        "--bin", bin_name, "--",
    ] + extra_args
    completed = subprocess.run(
        cmd, cwd=REPO_ROOT, env=subprocess_env(), capture_output=True, text=True, check=True
    )
    return completed.stdout


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Kairnz AlphaZero training loop.")
    parser.add_argument("--work", type=Path, default=Path("loop-run"), help="Working directory.")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--selfplay-games", type=int, default=64)
    parser.add_argument("--selfplay-sims", type=int, default=200)
    parser.add_argument("--gate-games", type=int, default=40)
    parser.add_argument("--gate-sims", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--window", type=int, default=5, help="Recent shards used per training round.")
    parser.add_argument("--filters", type=int, default=64)
    parser.add_argument("--blocks", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    args = parser.parse_args()

    work = args.work
    shards_dir = work / "shards"
    models_dir = work / "models"
    shards_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    best = models_dir / "best.onnx"
    metrics_path = work / "metrics.jsonl"

    # Iteration 0: seed best with a fresh random network at the target size.
    export_onnx(KairnzNet(filters=args.filters, blocks=args.blocks), best)
    print(f"seeded {best}")

    for it in range(args.iterations):
        shard = shards_dir / f"iter{it}.safetensors"
        _run_rust("selfplay", [
            "--model", str(best), "--out", str(shard),
            "--games", str(args.selfplay_games), "--simulations", str(args.selfplay_sims),
            "--seed", str(it),
        ])

        window = select_window(list(shards_dir.glob("*.safetensors")), args.window)
        candidate = models_dir / f"candidate{it}.onnx"
        n = train_candidate(window, candidate, args.filters, args.blocks, args.epochs, args.lr, args.weight_decay)

        gate_out = _run_rust("gate", [
            "--model-a", str(candidate), "--model-b", str(best),
            "--games", str(args.gate_games), "--simulations", str(args.gate_sims),
            "--seed", str(it),
        ])
        score = json.loads(gate_out.strip())["a_score"]
        promoted = should_promote(score, PROMOTE_THRESHOLD)
        if promoted:
            shutil.copyfile(candidate, best)

        row = {"iter": it, "samples": n, "a_score": score, "promoted": promoted}
        with metrics_path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"iter {it}: samples={n} score={score:.4f} promoted={promoted}")


if __name__ == "__main__":
    main()
