# AlphaZero Plan 6: Orchestration Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Tie the pipeline into a self-improving loop: a model-vs-model gate that decides whether a freshly trained candidate is stronger than the current best, and an orchestrator that runs self-play → train → gate → promote on repeat, so the network actually gets better over iterations.

**Architecture:** A Rust gate (in `kairnz-selfplay`) plays a candidate ONNX against the current best over many games with `AzMctsPolicy`, alternating sides, and reports a win score as JSON. A Python orchestrator (`orchestrate.py`) drives the loop: it subprocesses the Rust self-play and gate binaries (injecting torch's cuDNN directory into their PATH programmatically so the ONNX Runtime CUDA provider loads), trains in-process between rounds, and promotes the candidate to best when its gate score clears a threshold. The orchestrator keeps per-iteration artifacts (shards, candidate models, a metrics log) and a single `best.onnx` pointer.

**Tech Stack:** Rust (`kairnz-selfplay`, `kairnz-onnx`, `ort`), Python (the `train/` project), `safetensors`, `serde_json`, `clap`.

## Global Constraints

- The gate uses `AzMctsPolicy` for both sides with a small Dirichlet root noise (`dirichlet_epsilon ~ 0.15`) so games vary and the win score is statistically meaningful (pure argmax would make identical models play one deterministic game). It alternates which model plays P1 across games to cancel first-player bias.
- Gate "A score" = `(a_wins + 0.5 * draws) / games`. The orchestrator promotes the candidate when its score against best is `>= 0.55` (a named threshold).
- Rust subprocesses launched by the orchestrator must run the ONNX Runtime CUDA provider; the orchestrator prepends torch's `lib` directory (which ships `cudnn64_9.dll`) to the subprocess PATH, computed from `torch.__file__`. (This is the same cuDNN dependency the `onnx-check`/`selfplay` Taskfile targets handle via PowerShell; here it is done programmatically in Python.)
- Plane/policy sizes come from `kairnz_train.model` constants and `kairnz_encode`; never re-hardcode 14/6723.
- Rust: named constants, doc comments on public items, `Result` on fallible paths (no unwrap on IO/inference), no em dashes, files under 300 lines.
- Python: docstrings on public functions; no em dashes; config via argparse with named defaults.
- The first real training run is an operational activity (hours on the GPU) driven via the `loop` Taskfile target; this plan delivers the code, not the run.

---

## File Structure

- Create: `crates/kairnz-selfplay/src/gate.rs` — `play_match`, `GateResult`, `run_gate`.
- Modify: `crates/kairnz-selfplay/src/lib.rs` — `pub mod gate;`.
- Create: `crates/kairnz-selfplay/src/bin/gate.rs` — gate CLI emitting JSON.
- Create: `train/kairnz_train/orchestrate.py` — promote/window helpers and the loop.
- Create: `train/scripts/loop.py` — orchestration CLI.
- Create: `train/tests/test_orchestrate.py` — tests for the pure helpers.
- Modify: `Taskfile.yml` — add a `loop` GPU target.

---

### Task 1: The gate (model-vs-model match)

**Files:**
- Modify: `crates/kairnz-selfplay/Cargo.toml` (add the `kairnz-policy` dep)
- Create: `crates/kairnz-selfplay/src/gate.rs`
- Modify: `crates/kairnz-selfplay/src/lib.rs`

**Interfaces:**
- Consumes: `kairnz_onnx::{AzMctsConfig, AzMctsPolicy}` (both re-exported at the `kairnz_onnx` crate root), `kairnz_policy::policy::Policy`, `kairnz_core::{game::Game, config::RuleConfig, outcome::{GameResult, DrawReason}, piece::Player}`.
- Produces:
  - `play_match(config: RuleConfig, p1: &mut dyn Policy, p2: &mut dyn Policy) -> GameResult` — drives two policies from the standard opening to a terminal result.
  - `GateResult { a_wins: u32, b_wins: u32, draws: u32 }` with `a_score(&self) -> f64`.
  - `run_gate(model_a: &Path, model_b: &Path, games: u32, config: AzMctsConfig, rule: RuleConfig, base_seed: u64) -> ort::Result<GateResult>`.

- [ ] **Step 1: Add the `kairnz-policy` dependency**

The gate calls policies through the `kairnz_policy::policy::Policy` trait, so add `kairnz-policy` to `crates/kairnz-selfplay/Cargo.toml` under `[dependencies]` (Rust does not re-export transitive deps, so even though `kairnz-onnx` uses it, `kairnz-selfplay` needs it directly):

```toml
kairnz-policy = { path = "../kairnz-policy" }
```

- [ ] **Step 2: Write the gate module and tests**

Create `crates/kairnz-selfplay/src/gate.rs`:

```rust
//! Model-vs-model gating: play a candidate ONNX against a best ONNX and score it.

use std::path::Path;

use kairnz_core::config::RuleConfig;
use kairnz_core::game::Game;
use kairnz_core::outcome::{DrawReason, GameResult};
use kairnz_core::piece::Player;
use kairnz_onnx::{AzMctsConfig, AzMctsPolicy};
use kairnz_policy::policy::Policy;

/// Plays one game from the standard opening between `p1` (P1) and `p2` (P2),
/// returning the terminal result. Never panics: a `None` choice or an illegal
/// action defensively ends the game.
pub fn play_match(config: RuleConfig, p1: &mut dyn Policy, p2: &mut dyn Policy) -> GameResult {
    let mut game = Game::new_standard(config);
    while game.terminal_result().is_none() {
        let mover = game.pos.to_move;
        let policy: &mut dyn Policy = if mover == Player::P1 { p1 } else { p2 };
        match policy.choose(&game) {
            Some(action) => {
                if game.apply(action).is_err() {
                    break;
                }
            }
            None => break,
        }
    }
    game.terminal_result().unwrap_or(GameResult::Draw(DrawReason::MaxPlies))
}

/// Tally of a gate match from model A's perspective.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GateResult {
    /// Games won by model A.
    pub a_wins: u32,
    /// Games won by model B.
    pub b_wins: u32,
    /// Drawn games.
    pub draws: u32,
}

impl GateResult {
    /// Model A's score: wins plus half-credit for draws, over all games. Returns
    /// 0.0 when no games were played.
    pub fn a_score(&self) -> f64 {
        let total = self.a_wins + self.b_wins + self.draws;
        if total == 0 {
            return 0.0;
        }
        (self.a_wins as f64 + 0.5 * self.draws as f64) / total as f64
    }
}

/// Plays `games` gate games between the models at `model_a` and `model_b`,
/// alternating which model plays P1 to cancel first-player bias, and returns the
/// tally from model A's perspective.
///
/// Both sides use `AzMctsPolicy` with `config` (which should carry a small
/// Dirichlet epsilon so games vary). The models are loaded once and reused.
pub fn run_gate(
    model_a: &Path,
    model_b: &Path,
    games: u32,
    config: AzMctsConfig,
    rule: RuleConfig,
    base_seed: u64,
) -> ort::Result<GateResult> {
    let mut policy_a = AzMctsPolicy::from_path(model_a, config, base_seed)?;
    let mut policy_b = AzMctsPolicy::from_path(model_b, config, base_seed.wrapping_add(1))?;

    let mut result = GateResult { a_wins: 0, b_wins: 0, draws: 0 };
    for g in 0..games {
        let a_is_p1 = g % 2 == 0;
        let outcome = if a_is_p1 {
            play_match(rule.clone(), &mut policy_a, &mut policy_b)
        } else {
            play_match(rule.clone(), &mut policy_b, &mut policy_a)
        };
        match outcome {
            GameResult::Win(winner) => {
                let a_player = if a_is_p1 { Player::P1 } else { Player::P2 };
                if winner == a_player {
                    result.a_wins += 1;
                } else {
                    result.b_wins += 1;
                }
            }
            GameResult::Draw(_) => result.draws += 1,
        }
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn fixture() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../kairnz-onnx/tests/fixtures/random_init.onnx")
    }

    fn gate_config() -> AzMctsConfig {
        // Small sims keep the test fast; epsilon > 0 makes games vary by seed.
        AzMctsConfig { simulations: 16, dirichlet_epsilon: 0.15, ..AzMctsConfig::default() }
    }

    #[test]
    fn a_score_counts_draws_as_half() {
        let r = GateResult { a_wins: 3, b_wins: 1, draws: 2 };
        assert!((r.a_score() - (3.0 + 1.0) / 6.0).abs() < 1e-9);
    }

    #[test]
    fn gate_tally_sums_to_games_and_is_reproducible() {
        let path = fixture();
        let games = 4;
        let r1 = run_gate(&path, &path, games, gate_config(), RuleConfig::default(), 7)
            .expect("gate runs");
        assert_eq!(r1.a_wins + r1.b_wins + r1.draws, games, "tally sums to games");
        assert!((0.0..=1.0).contains(&r1.a_score()));

        let r2 = run_gate(&path, &path, games, gate_config(), RuleConfig::default(), 7)
            .expect("gate runs");
        assert_eq!(r1, r2, "same seed yields the same gate result");
    }
}
```

- [ ] **Step 3: Wire the module**

In `crates/kairnz-selfplay/src/lib.rs`, add `pub mod gate;` alongside the existing module declarations.

- [ ] **Step 4: Run the tests**

Run: `cargo test -p kairnz-selfplay gate`
Expected: PASS (2 tests). The reproducibility test confirms the seeded gate is deterministic.

- [ ] **Step 5: Commit**

```bash
git add crates/kairnz-selfplay/src/gate.rs crates/kairnz-selfplay/src/lib.rs
git commit -m "feat(selfplay): add model-vs-model gate"
```

---

### Task 2: Gate CLI

**Files:**
- Create: `crates/kairnz-selfplay/src/bin/gate.rs`

**Interfaces:**
- Consumes: `run_gate`, `AzMctsConfig`.
- Produces: a `gate` binary that prints the result as a single JSON line `{"a_wins":..,"b_wins":..,"draws":..,"a_score":..}`.

- [ ] **Step 1: Write the CLI**

Create `crates/kairnz-selfplay/src/bin/gate.rs`:

```rust
//! Gate CLI: play model A against model B and print the tally as JSON.

use std::path::PathBuf;
use std::process::ExitCode;

use clap::Parser;
use kairnz_core::config::RuleConfig;
use kairnz_onnx::AzMctsConfig;
use kairnz_selfplay::gate::run_gate;

/// Default Dirichlet root-noise weight for gate variety.
const GATE_DIRICHLET_EPSILON: f64 = 0.15;

#[derive(Parser)]
#[command(about = "Gate a candidate Kairnz model against a best model.")]
struct Args {
    /// Candidate model (scored as A).
    #[arg(long)]
    model_a: PathBuf,
    /// Reference model (B).
    #[arg(long)]
    model_b: PathBuf,
    /// Number of games to play.
    #[arg(long, default_value_t = 40)]
    games: u32,
    /// MCTS simulations per move.
    #[arg(long, default_value_t = 100)]
    simulations: u32,
    /// Base RNG seed.
    #[arg(long, default_value_t = 0)]
    seed: u64,
}

fn main() -> ExitCode {
    let args = Args::parse();
    let config = AzMctsConfig {
        simulations: args.simulations,
        dirichlet_epsilon: GATE_DIRICHLET_EPSILON,
        ..AzMctsConfig::default()
    };

    let result = match run_gate(
        &args.model_a,
        &args.model_b,
        args.games,
        config,
        RuleConfig::default(),
        args.seed,
    ) {
        Ok(r) => r,
        Err(error) => {
            eprintln!("gate failed: {error}");
            return ExitCode::FAILURE;
        }
    };

    println!(
        "{{\"a_wins\":{},\"b_wins\":{},\"draws\":{},\"a_score\":{:.4}}}",
        result.a_wins, result.b_wins, result.draws, result.a_score()
    );
    ExitCode::SUCCESS
}
```

- [ ] **Step 2: Build and smoke-test on CPU**

Run: `cargo build -p kairnz-selfplay`
Expected: builds warning-free.

Run a tiny CPU smoke (candidate vs best = the same fixture):

```bash
cargo run -p kairnz-selfplay --bin gate -- --model-a crates/kairnz-onnx/tests/fixtures/random_init.onnx --model-b crates/kairnz-onnx/tests/fixtures/random_init.onnx --games 2 --simulations 8
```
Expected: prints one JSON line like `{"a_wins":..,"b_wins":..,"draws":..,"a_score":..}` whose counts sum to 2.

- [ ] **Step 3: Commit**

```bash
git add crates/kairnz-selfplay/src/bin/gate.rs
git commit -m "feat(selfplay): add gate CLI emitting JSON"
```

---

### Task 3: Python orchestrator helpers and loop

**Files:**
- Create: `train/kairnz_train/orchestrate.py`
- Create: `train/tests/test_orchestrate.py`

**Interfaces:**
- Produces:
  - `should_promote(a_score: float, threshold: float) -> bool`
  - `select_window(shard_paths: list[Path], window: int) -> list[Path]` — the most recent `window` shards.
  - `torch_lib_dir() -> Path` — torch's bundled library directory (for the Rust subprocess PATH).
  - `subprocess_env() -> dict` — the current environment with `torch_lib_dir()` prepended to PATH.
  - `train_candidate(shard_paths, out_path, filters, blocks, epochs, lr, weight_decay) -> int` — trains a fresh model on the given shards, exports ONNX, returns the sample count.

- [ ] **Step 1: Write the orchestrator module**

Create `train/kairnz_train/orchestrate.py`:

```python
"""Orchestration helpers for the AlphaZero training loop.

The loop itself lives in scripts/loop.py; this module holds the pure, testable
pieces (promotion rule, shard windowing, environment construction) plus a
training entry point.
"""

import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from kairnz_train.dataset import SelfPlayDataset
from kairnz_train.model import KairnzNet
from kairnz_train.onnx_export import export_onnx
from kairnz_train.train import train_epoch

# Promotion threshold: a candidate must score at least this against best.
PROMOTE_THRESHOLD = 0.55


def should_promote(a_score: float, threshold: float = PROMOTE_THRESHOLD) -> bool:
    """Returns True when a candidate's gate score clears the threshold."""
    return a_score >= threshold


def select_window(shard_paths: list[Path], window: int) -> list[Path]:
    """Returns the most recent `window` shard paths (by sorted order)."""
    ordered = sorted(shard_paths)
    if window <= 0:
        return ordered
    return ordered[-window:]


def torch_lib_dir() -> Path:
    """Directory of torch's bundled shared libraries (ships cuDNN on Windows)."""
    return Path(torch.__file__).parent / "lib"


def subprocess_env() -> dict:
    """The current environment with torch's lib dir prepended to PATH.

    The Rust self-play and gate binaries need cuDNN on PATH for the ONNX Runtime
    CUDA provider; torch ships it, so we expose torch's lib dir to the children.
    """
    env = dict(os.environ)
    lib = str(torch_lib_dir())
    env["PATH"] = lib + os.pathsep + env.get("PATH", "")
    return env


def _load_concat(shard_paths: list[Path]) -> dict[str, np.ndarray]:
    """Concatenates the given shard files into the four sample arrays."""
    from safetensors.numpy import load_file

    keys = ("planes", "policy", "value", "legal_mask")
    parts: dict[str, list[np.ndarray]] = {k: [] for k in keys}
    for path in shard_paths:
        loaded = load_file(str(path))
        for key in keys:
            parts[key].append(loaded[key])
    return {key: np.concatenate(parts[key], axis=0) for key in keys}


def train_candidate(
    shard_paths: list[Path],
    out_path: Path,
    filters: int,
    blocks: int,
    epochs: int,
    lr: float,
    weight_decay: float,
) -> int:
    """Trains a fresh KairnzNet on the given shards and exports it to ONNX.

    Returns the number of training samples used.
    """
    data = _load_concat(shard_paths)
    dataset = SelfPlayDataset(data)
    loader = DataLoader(dataset, batch_size=256, shuffle=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = KairnzNet(filters=filters, blocks=blocks).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    for _ in range(epochs):
        train_epoch(model, _on_device(loader, device), optimizer)

    model.to("cpu")
    export_onnx(model, out_path)
    return len(dataset)


def _on_device(loader, device):
    """Yields batches moved to `device`."""
    for batch in loader:
        yield tuple(t.to(device) for t in batch)
```

- [ ] **Step 2: Write the helper tests**

Create `train/tests/test_orchestrate.py`:

```python
"""Tests for the orchestration helpers."""

import os
from pathlib import Path

from kairnz_train.orchestrate import (
    select_window,
    should_promote,
    subprocess_env,
    torch_lib_dir,
)


def test_should_promote_threshold():
    assert should_promote(0.55) is True
    assert should_promote(0.54) is False
    assert should_promote(0.6, threshold=0.7) is False


def test_select_window_keeps_most_recent():
    paths = [Path(f"shard_{i}.safetensors") for i in range(5)]
    window = select_window(paths, 2)
    assert window == [Path("shard_3.safetensors"), Path("shard_4.safetensors")]
    assert select_window(paths, 0) == sorted(paths)


def test_subprocess_env_prepends_torch_lib():
    env = subprocess_env()
    assert env["PATH"].startswith(str(torch_lib_dir()) + os.pathsep)
```

- [ ] **Step 3: Run the tests**

Run: `cd train && uv run pytest tests/test_orchestrate.py -q`
Expected: 3 passed.

- [ ] **Step 4: Commit**

```bash
git add train/kairnz_train/orchestrate.py train/tests/test_orchestrate.py
git commit -m "feat(train): add orchestration helpers"
```

---

### Task 4: Loop CLI and the Taskfile target

**Files:**
- Create: `train/scripts/loop.py`
- Modify: `Taskfile.yml`

**Interfaces:**
- Consumes: `orchestrate` helpers, the `selfplay` and `gate` Rust binaries (via subprocess), `KairnzNet`/`export_onnx` (to seed iteration-0 best).
- Produces: a `loop` CLI that runs the self-play -> train -> gate -> promote loop, and a `task loop` GPU target.

- [ ] **Step 1: Write the loop CLI**

Create `train/scripts/loop.py`:

```python
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
```

- [ ] **Step 2: Add the Taskfile target**

In `Taskfile.yml`, add a `loop` target (it runs the loop CLI; the orchestrator injects the cuDNN PATH for its Rust subprocesses itself, but the trainer also benefits from torch's CUDA, so run it through the same PATH wrapper for consistency):

```yaml
  # Run the full AlphaZero training loop (self-play -> train -> gate -> promote).
  # Long-running; override iterations/games, e.g.
  #   task loop ITERS=20 GAMES=128
  loop:
    vars:
      WORK: '{{.WORK | default "loop-run"}}'
      ITERS: '{{.ITERS | default 10}}'
      GAMES: '{{.GAMES | default 64}}'
    cmds:
      - powershell -NoProfile -Command '$env:PATH = "{{.TORCH_LIB}};$env:PATH"; cd train; uv run python scripts/loop.py --work "../{{.WORK}}" --iterations {{.ITERS}} --selfplay-games {{.GAMES}}'
```

- [ ] **Step 3: Verify the loop CLI parses and the Taskfile is valid**

Run: `cd train && uv run python scripts/loop.py --help`
Expected: prints the argument help (confirms imports resolve and argparse is wired).

Run: `task --list-all`
Expected: the `loop` target is listed without a YAML error. (Do NOT run `task loop`; it is a long GPU job that needs the binaries and produces real artifacts.)

- [ ] **Step 4: Add a `.gitignore` entry for run artifacts**

Append to `.gitignore` so loop outputs are not tracked:

```
# AlphaZero loop run artifacts
loop-run/
*.safetensors
models/
```

Note: the committed test fixture `crates/kairnz-onnx/tests/fixtures/random_init.onnx` is an `.onnx`, not a `.safetensors`, so the `*.safetensors` ignore does not touch it. Confirm `git status` does not newly ignore any tracked file.

- [ ] **Step 5: Run the full Rust and Python suites**

Run: `cargo test -p kairnz-selfplay`
Expected: PASS (gate tests plus the existing self-play tests), warning-free.

Run: `cd train && uv run pytest -q`
Expected: all Python tests pass.

Run: `cargo build --workspace`
Expected: builds.

- [ ] **Step 6: Commit**

```bash
git add train/scripts/loop.py Taskfile.yml .gitignore
git commit -m "feat(train): add orchestration loop CLI and task target"
```

---

## Self-Review Notes

- **Spec coverage:** Implements the spec's Milestone 6 (orchestration: self-play -> train -> gate -> promote, with a strength gate). The first real training run is the operational follow-up driven by `task loop`.
- **Gate soundness:** the gate alternates sides and uses a small Dirichlet noise so candidate-vs-best produces a varied, statistically meaningful score rather than a single deterministic game. The promotion threshold is a named constant.
- **Cross-language coordination:** the Python orchestrator computes torch's lib dir from `torch.__file__` and injects it into the Rust subprocess PATH, so the ONNX Runtime CUDA provider loads without the PowerShell hack. The pure helpers (`should_promote`, `select_window`, `subprocess_env`) are unit-tested; the full loop is integration glue verified by the operational run.
- **Artifacts:** per-iteration shards/candidates and a `metrics.jsonl` live under a gitignored work dir; only `best.onnx` is the rolling pointer. Run artifacts are gitignored.
- **Deferred (flagged):** sequential self-play (from Plan 4) still applies; the loop's throughput is bounded by it. Strength-vs-baselines plotting (AzMctsPolicy vs Random/Greedy/plain-MCTS in `kairnz-bench`) is a reporting nicety that can follow the first run. The gate reloads models per run (once each), not per game.
- **Type consistency:** `play_match`, `GateResult`/`a_score`, `run_gate`, and the Python `should_promote`/`select_window`/`subprocess_env`/`train_candidate` signatures are referenced identically across tasks and the loop CLI.
