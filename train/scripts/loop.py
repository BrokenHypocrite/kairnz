"""AlphaZero orchestration loop: self-play -> train -> gate -> promote, repeated.

Runs the Rust self-play and gate binaries as subprocesses (with cuDNN on PATH)
and trains in-process between rounds. Maintains best.onnx and a metrics log.
"""

import argparse
import json
import math
import shutil
import subprocess
from pathlib import Path

from kairnz_train.model import KairnzNet
from kairnz_train.onnx_export import export_onnx
from kairnz_train.orchestrate import (
    PROMOTE_THRESHOLD,
    resolve_start,
    save_checkpoint,
    select_window,
    should_promote,
    subprocess_env,
    train_candidate,
    write_status,
)

# Repository root relative to this script (train/scripts/loop.py -> repo root).
REPO_ROOT = Path(__file__).resolve().parents[2]


def _decayed_lr(base_lr: float, min_lr: float, it: int, start: int, total: int) -> float:
    """Cosine-decays the learning rate from base_lr to min_lr across the run.

    A fixed LR can stall late-stage convergence; decaying it lets early
    iterations move fast and later ones fine-tune. Returns base_lr for a
    single-iteration run.
    """
    if total <= 1:
        return base_lr
    progress = min(max((it - start) / (total - 1), 0.0), 1.0)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def _run_rust(bin_name: str, extra_args: list[str]) -> str:
    """Runs a release Rust binary with cuDNN on PATH; returns its stdout."""
    cmd = [
        "cargo", "run", "--release", "-p", "kairnz-selfplay",
        "--bin", bin_name, "--",
    ] + extra_args
    completed = subprocess.run(
        cmd, cwd=REPO_ROOT, env=subprocess_env(), capture_output=True, text=True
    )
    if completed.returncode != 0:
        # Surface the subprocess stderr so a failing stage is debuggable rather
        # than hidden behind a bare CalledProcessError.
        raise RuntimeError(
            f"{bin_name} failed (exit {completed.returncode}):\n{completed.stderr}"
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
    parser.add_argument("--min-lr", type=float, default=1e-4,
                        help="Floor for the cosine learning-rate decay across iterations.")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--threads", type=int, default=0, help="Self-play thread/process count (0 = auto).")
    parser.add_argument("--batched", action="store_true",
                        help="Self-play uses one shared batched inference server (GPU-bottleneck mode) "
                             "instead of one ONNX session per worker process.")
    parser.add_argument("--max-batch", type=int, default=256,
                        help="Max positions per GPU batch in batched self-play.")
    parser.add_argument("--leaves-per-step", type=int, default=8,
                        help="Leaves collected per batched MCTS step in batched self-play.")
    parser.add_argument("--resume", action="store_true",
                        help="Continue from existing best model and metrics instead of re-seeding.")
    args = parser.parse_args()

    # Resolve to an absolute path: the Rust self-play and gate subprocesses run
    # from the repo root, so all model/shard paths passed to them must be absolute
    # rather than relative to this script's working directory.
    work = args.work.resolve()
    shards_dir = work / "shards"
    models_dir = work / "models"
    shards_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    best = models_dir / "best.onnx"
    best_pt = models_dir / "best.pt"
    status_path = work / "status.json"
    metrics_path = work / "metrics.jsonl"

    start_iter, seed_fresh = resolve_start(best, best_pt, metrics_path, args.resume)
    if seed_fresh:
        seed_model = KairnzNet(filters=args.filters, blocks=args.blocks)
        save_checkpoint(seed_model, best_pt)
        export_onnx(seed_model, best)
        write_status(status_path, {"iteration": 0, "total_iterations": args.iterations,
                                   "stage": "seeding", "samples": 0, "last_score": None,
                                   "promoted_count": 0})
        print(f"seeded {best}")
    else:
        print(f"resuming from {best} at iteration {start_iter}")

    promoted_count = 0
    for it in range(start_iter, start_iter + args.iterations):
        write_status(status_path, {"iteration": it, "total_iterations": args.iterations,
                                   "stage": "self-play", "samples": 0,
                                   "last_score": None, "promoted_count": promoted_count})
        shard = shards_dir / f"iter{it:04d}.safetensors"
        selfplay_args = [
            "--model", str(best), "--out", str(shard),
            "--games", str(args.selfplay_games), "--simulations", str(args.selfplay_sims),
            "--seed", str(it), "--threads", str(args.threads),
        ]
        if args.batched:
            selfplay_args += [
                "--batched",
                "--max-batch", str(args.max_batch),
                "--leaves-per-step", str(args.leaves_per_step),
            ]
        _run_rust("selfplay", selfplay_args)

        write_status(status_path, {"iteration": it, "total_iterations": args.iterations,
                                   "stage": "training", "samples": 0,
                                   "last_score": None, "promoted_count": promoted_count})
        window = select_window(list(shards_dir.glob("*.safetensors")), args.window)
        candidate = models_dir / f"candidate{it:04d}.onnx"
        iter_lr = _decayed_lr(args.lr, args.min_lr, it, start_iter, args.iterations)
        n = train_candidate(window, candidate, args.filters, args.blocks,
                            args.epochs, iter_lr, args.weight_decay, warm_start=best_pt)

        write_status(status_path, {"iteration": it, "total_iterations": args.iterations,
                                   "stage": "gating", "samples": n,
                                   "last_score": None, "promoted_count": promoted_count})
        gate_out = _run_rust("gate", [
            "--model-a", str(candidate), "--model-b", str(best),
            "--games", str(args.gate_games), "--simulations", str(args.gate_sims),
            "--seed", str(it), "--threads", str(args.threads),
        ])
        score = json.loads(gate_out.strip())["a_score"]
        promoted = should_promote(score, PROMOTE_THRESHOLD)
        if promoted:
            shutil.copyfile(candidate, best)
            shutil.copyfile(candidate.with_suffix(".pt"), best_pt)
            promoted_count += 1

        row = {"iter": it, "samples": n, "a_score": score, "promoted": promoted,
               "lr": round(iter_lr, 6)}
        with metrics_path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        write_status(status_path, {"iteration": it, "total_iterations": args.iterations,
                                   "stage": "done", "samples": n,
                                   "last_score": score, "promoted_count": promoted_count})
        print(f"iter {it}: samples={n} score={score:.4f} promoted={promoted}")


if __name__ == "__main__":
    main()
