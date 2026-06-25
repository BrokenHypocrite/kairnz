# Plan 9 (FULL): Remote Training at Scale + Batched Inference + App AI Upgrade

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax. This plan supersedes the Plan 9 roadmap (`2026-04-...-plan9-remote-training.md`); that file remains the originating design.

**Goal:** Produce a genuinely strong Kairnz model by running long, GPU-saturating training on a rented H100 (or better), monitored from a web dashboard and resumable across interruptions; and make the in-app opponent strong by giving it GPU inference and the same batched search. Finish by measuring absolute strength against fixed baselines.

**Architecture:** Six phases. (A) The training loop resumes from existing artifacts. (B) A Docker image stands the pipeline up headless on a rented box. (C) The throughput core: a `BatchEvaluator` trait with two backends - a shared, thread-safe `InferenceServer` that coalesces leaf evaluations from many self-play games into large GPU batches (sync threads + a dedicated batcher thread, no async runtime), and a direct single-session evaluator; a `BatchedAzMcts` runs virtual-loss MCTS that submits K leaves per search step, so even one game feeds the GPU a batch. (D) A web dashboard serves live progress. (E) The app reuses `BatchedAzMcts` with a direct GPU evaluator for a much stronger interactive opponent. (F) A strength-vs-baselines harness quantifies the result.

**Tech Stack:** Rust (`kairnz-onnx`, `kairnz-selfplay`, `src-tauri`, `ort`, `std::sync::mpsc`/`std::thread`), Python (`train/`, FastAPI or stdlib `http.server`), Docker/CUDA/cuDNN, Tauri/Svelte.

## Key Architecture Decisions (confirmed)

1. **Inference server = synchronous threads + a batcher thread.** No async runtime. Each self-play game runs on its own OS thread (the existing model) and BLOCKS on its leaf-eval request; a dedicated batcher thread drains the request queue into one batched GPU call and scatters results back. Cross-game batching emerges from many threads blocking at once; virtual loss adds intra-game batching on top.
2. **One `BatchEvaluator` trait, two backends.** `InferenceServer` (shared, `Sync`, for self-play/gate across many threads) and `DirectBatchEvaluator` (a `Mutex<OnnxEvaluator>`, for the single-threaded app). `BatchedAzMcts` is generic over `&dyn BatchEvaluator`, so the same search powers self-play and the app.
3. **Determinism stance.** Batched self-play is NOT bit-for-bit reproducible: batch composition depends on timing, and GPU batched matmuls differ from batch-1 in the last bits. This is the standard, accepted AlphaZero tradeoff. Tests assert CORRECTNESS and TOLERANCE (batched ~= sequential within an epsilon; mate-in-one still found; visit counts valid), never bit-exact equality. Each game keeps its own seeded RNG for dirichlet/sampling, so a game is reproducible up to float noise.
4. **App AI is upgraded** (separate user decision): the app gains GPU inference (cuDNN on PATH) and uses `BatchedAzMcts` with `DirectBatchEvaluator`, so Hard mode runs far more simulations per move at the same latency.
5. **Net size stays a CLI arg** (default `128/10`); the runbook (Task F) uses that default but documents overriding it.

## Global Constraints

- `BatchEvaluator::evaluate_batch(&self, batch: &[Vec<f32>], reps: &[u8]) -> ort::Result<Vec<(Vec<f32>, f32)>>` is the single inference seam. `Vec<f32>` planes are length `14*81 = 1134` (already canonical-encoded). Output per row: policy `Vec<f32>` length `6723` and a scalar value.
- Virtual loss: during selection add `VIRTUAL_LOSS` in-flight visits along the path (each counted as a loss for the player to move); remove them at backup. Selection PUCT uses `N_eff = N + vl` and `Q_eff = (W - vl) / N_eff`.
- No new async runtime (no `tokio`). Concurrency uses `std::thread` + `std::sync::mpsc` (or `crossbeam-channel` if already a dependency; otherwise std).
- Rust: named constants (no magic numbers), doc comments on public items, no `unwrap()`/`expect()` on fallible runtime paths (model load, inference, channel recv after shutdown), no em dashes.
- Python: docstrings, argparse defaults, no em dashes. Web dashboard reuses `kairnz_train.orchestrate.load_status`/`load_metrics`.
- The ONNX model is exported with a dynamic batch dimension (Plan 2), so batched inference needs no re-export.
- The existing `AzMcts`/`AzMctsPolicy` (sequential, batch-1) stays for any code path not migrated; do not delete it until its last caller is migrated.

---

## File Structure

- Modify: `train/scripts/loop.py`, `train/kairnz_train/orchestrate.py` (resume helpers).
- Create: `Dockerfile`, `.dockerignore`, `scripts/run_remote.sh`.
- Create: `crates/kairnz-onnx/src/batch.rs` (`BatchEvaluator` trait, `DirectBatchEvaluator`, `InferenceServer`).
- Modify: `crates/kairnz-onnx/src/evaluator.rs` (`evaluate_batch`), `crates/kairnz-onnx/src/mcts.rs` (or new `batched_mcts.rs`: `BatchedAzMcts`), `crates/kairnz-onnx/src/lib.rs` (re-exports).
- Modify: `crates/kairnz-selfplay/src/parallel.rs`, `crates/kairnz-selfplay/src/bin/selfplay.rs`, `crates/kairnz-selfplay/src/gate.rs`.
- Create: `train/kairnz_train/dashboard.py`, `train/kairnz_train/static/dashboard.html`; modify `Taskfile.yml`.
- Modify: `src-tauri/src/ai.rs` (app uses `BatchedAzMcts` + GPU), `ui/src/components/ConfigPanel.svelte` (difficulty), `Taskfile.yml` (`app-gpu` target).
- Create: `crates/kairnz-selfplay/src/bin/strength.rs` (baseline eval), `docs/RUNBOOK-remote-training.md`.

---

# Phase A - Resume

### Task A1: Resume the training loop

**Files:** Modify `train/kairnz_train/orchestrate.py`, `train/scripts/loop.py`; add tests to `train/tests/test_orchestrate_warmstart.py`.

**Interfaces:**
- Produces: `next_iteration(metrics_path) -> int` (one past the last `iter` in metrics.jsonl, or 0); `resolve_start(work, models_dir, resume) -> (start_iter: int, seed_fresh: bool)`.

- [ ] **Step 1: Add resume helpers to `orchestrate.py`**

```python
def next_iteration(metrics_path: Path) -> int:
    """Returns one past the highest `iter` recorded in metrics.jsonl, or 0 if none."""
    rows = load_metrics(metrics_path)
    return max((r["iter"] for r in rows), default=-1) + 1


def resolve_start(best: Path, best_pt: Path, metrics_path: Path, resume: bool) -> tuple[int, bool]:
    """Decides where the loop starts.

    Returns (start_iter, seed_fresh). When `resume` is set and both best artifacts
    exist, continue from the recorded iteration without re-seeding; otherwise start
    at 0 and seed a fresh random network.
    """
    if resume and best.exists() and best_pt.exists():
        return next_iteration(metrics_path), False
    return 0, True
```

- [ ] **Step 2: Wire into `loop.py`**

Add `--resume` (store_true). Replace the unconditional seed block:

```python
    start_iter, seed_fresh = resolve_start(best, best_pt, metrics_path, args.resume)
    if seed_fresh:
        seed_model = KairnzNet(filters=args.filters, blocks=args.blocks)
        save_checkpoint(seed_model, best_pt)
        export_onnx(seed_model, best)
        print(f"seeded {best}")
    else:
        print(f"resuming from {best} at iteration {start_iter}")
    ...
    for it in range(start_iter, start_iter + args.iterations):
        ...
```

(Import `resolve_start`. Iterations now run `start_iter .. start_iter+args.iterations`. Shard/candidate names already use `it:04d`, so they stay unique on resume.)

- [ ] **Step 3: Tests**

Add to `train/tests/test_orchestrate_warmstart.py`:

```python
def test_next_iteration_counts_from_metrics(tmp_path):
    from kairnz_train.orchestrate import next_iteration
    p = tmp_path / "metrics.jsonl"
    assert next_iteration(p) == 0
    p.write_text('{"iter": 0}\n{"iter": 1}\n{"iter": 2}\n')
    assert next_iteration(p) == 3


def test_resolve_start_resumes_only_when_artifacts_exist(tmp_path):
    from kairnz_train.orchestrate import resolve_start
    best, best_pt, metrics = tmp_path / "b.onnx", tmp_path / "b.pt", tmp_path / "m.jsonl"
    assert resolve_start(best, best_pt, metrics, resume=True) == (0, True)  # nothing yet
    best.write_text("x"); best_pt.write_text("x"); metrics.write_text('{"iter": 4}\n')
    assert resolve_start(best, best_pt, metrics, resume=True) == (5, False)
    assert resolve_start(best, best_pt, metrics, resume=False) == (0, True)  # resume off
```

- [ ] **Step 4: Verify + commit**

`cd train && uv run pytest -q` (all pass). `uv run python scripts/loop.py --help` shows `--resume`.
```bash
git add train/kairnz_train/orchestrate.py train/scripts/loop.py train/tests/test_orchestrate_warmstart.py
git commit -m "feat(train): resume the loop from existing best + metrics"
```

---

# Phase B - Headless deployment

### Task B1: Dockerfile + build verification

**Files:** Create `Dockerfile`, `.dockerignore`.

- [ ] **Step 1: `.dockerignore`**

```
target/
train/.venv/
**/node_modules/
ui/dist/
loop-run/
*-run/
models/
*.safetensors
.git/
```

- [ ] **Step 2: `Dockerfile`**

```dockerfile
# CUDA + cuDNN base so the ONNX Runtime CUDA provider loads without the torch-lib hack.
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates build-essential pkg-config git && rm -rf /var/lib/apt/lists/*

# Rust toolchain.
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# uv for the Python side.
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /kairnz
COPY . .

# Build the Rust self-play/gate binaries and sync the Python env.
RUN cargo build --release -p kairnz-selfplay
RUN cd train && uv sync

CMD ["bash"]
```

- [ ] **Step 3: Document the build/run (no agent execution - this is doc + the file)**

The container is GPU-run with `docker run --gpus all`. Note in `docs/RUNBOOK-remote-training.md` (created in Task F2) that an agent without a GPU/Docker cannot `docker build` here; verification of the image is an operational step on the rented box. The deliverable for this task is the correct `Dockerfile`/`.dockerignore`.

- [ ] **Step 4: Commit**
```bash
git add Dockerfile .dockerignore
git commit -m "feat(deploy): Dockerfile for headless GPU training"
```

### Task B2: Remote launch script + Taskfile target

**Files:** Create `scripts/run_remote.sh`; modify `Taskfile.yml`.

- [ ] **Step 1: `scripts/run_remote.sh`**

```bash
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
```

- [ ] **Step 2: Taskfile `remote` target** (mirrors the local `loop` target but `--resume` + dashboard). Add under existing targets, reusing the existing var pattern. Commit:
```bash
git add scripts/run_remote.sh Taskfile.yml
git commit -m "feat(deploy): remote launch script + task target"
```

---

# Phase C - Batched inference + virtual loss (core)

### Task C1: Batched evaluator surface

**Files:** Modify `crates/kairnz-onnx/src/evaluator.rs`; create `crates/kairnz-onnx/src/batch.rs`; modify `lib.rs`.

**Interfaces:**
- Produces: `OnnxEvaluator::evaluate_batch(&mut self, planes: &[Vec<f32>], reps: &[u8]) -> ort::Result<Vec<(Vec<f32>, f32)>>`; `trait BatchEvaluator: Sync { fn evaluate_batch(&self, planes: &[Vec<f32>], reps: &[u8]) -> ort::Result<Vec<(Vec<f32>, f32)>>; }`; `struct DirectBatchEvaluator(Mutex<OnnxEvaluator>)` impl `BatchEvaluator`.

- [ ] **Step 1: `evaluate_batch` on `OnnxEvaluator`**

Read `evaluator.rs`. The existing `evaluate` builds a `[1,14,9,9]` tensor and runs the session. Add a batched version that stacks `B` pre-encoded plane vectors into a `[B,14,9,9]` tensor, runs once, and splits the `[B,6723]` / `[B,1]` outputs into `B` rows. Reuse the existing tensor/encoding helpers; the planes are already canonical (the caller encodes via `encode_planes`, same as `evaluate` does internally - factor the encode step so both share it). Constant: `const PLANE_LEN: usize = 14 * 81;`. Return `ort::Result<Vec<(Vec<f32>, f32)>>`.

- [ ] **Step 2: `batch.rs` - trait + direct backend**

```rust
//! Batched inference: one trait, two backends (direct single-session and a
//! shared cross-thread server).

use std::sync::Mutex;
use crate::evaluator::OnnxEvaluator;

/// A batched policy/value evaluator. `planes[i]` is a canonical 14*81 plane
/// vector; `reps[i]` its repetition count. Returns one (policy, value) per row.
pub trait BatchEvaluator: Sync {
    fn evaluate_batch(&self, planes: &[Vec<f32>], reps: &[u8]) -> ort::Result<Vec<(Vec<f32>, f32)>>;
}

/// Single-session backend (one search at a time; for the app).
pub struct DirectBatchEvaluator {
    inner: Mutex<OnnxEvaluator>,
}

impl DirectBatchEvaluator {
    pub fn new(evaluator: OnnxEvaluator) -> Self {
        Self { inner: Mutex::new(evaluator) }
    }
}

impl BatchEvaluator for DirectBatchEvaluator {
    fn evaluate_batch(&self, planes: &[Vec<f32>], reps: &[u8]) -> ort::Result<Vec<(Vec<f32>, f32)>> {
        let mut guard = self.inner.lock().expect("evaluator mutex poisoned");
        guard.evaluate_batch(planes, reps)
    }
}
```

- [ ] **Step 3: Test (tolerance, not bit-exact)**

`crates/kairnz-onnx/tests/batch.rs`: load the fixture model; for a few positions, compare `evaluate_batch` (as a batch) row outputs to `evaluate` (one at a time) within `1e-4` on the value and on a sampled set of policy logits. Assert lengths (`6723`).

- [ ] **Step 4: Re-export + commit**
`lib.rs`: `mod batch; pub use batch::{BatchEvaluator, DirectBatchEvaluator};`.
```bash
cargo test -p kairnz-onnx batch
git add crates/kairnz-onnx/src/evaluator.rs crates/kairnz-onnx/src/batch.rs crates/kairnz-onnx/src/lib.rs crates/kairnz-onnx/tests/batch.rs
git commit -m "feat(onnx): batched evaluator + BatchEvaluator trait"
```

### Task C2: The shared InferenceServer

**Files:** Modify `crates/kairnz-onnx/src/batch.rs`; `lib.rs`.

**Interfaces:**
- Produces: `InferenceServer::new(evaluator: OnnxEvaluator, max_batch: usize) -> InferenceServer`; impl `BatchEvaluator`; graceful shutdown on drop. Constants `DEFAULT_MAX_BATCH`.

- [ ] **Step 1: Implement the server**

```rust
use std::sync::mpsc::{channel, Receiver, Sender};
use std::thread::JoinHandle;

/// Default maximum positions per GPU batch.
pub const DEFAULT_MAX_BATCH: usize = 256;

struct Request {
    planes: Vec<f32>,
    rep: u8,
    reply: Sender<ort::Result<(Vec<f32>, f32)>>,
}

/// A shared, thread-safe evaluator that coalesces concurrent single-position
/// requests from many self-play threads into batched GPU calls.
pub struct InferenceServer {
    sender: Option<Sender<Request>>,
    batcher: Option<JoinHandle<()>>,
}

impl InferenceServer {
    pub fn new(mut evaluator: OnnxEvaluator, max_batch: usize) -> Self {
        let (tx, rx): (Sender<Request>, Receiver<Request>) = channel();
        let batcher = std::thread::spawn(move || {
            // Block for the first request, then drain everything immediately
            // available up to max_batch, run one batched inference, scatter results.
            while let Ok(first) = rx.recv() {
                let mut batch = vec![first];
                while batch.len() < max_batch {
                    match rx.try_recv() {
                        Ok(r) => batch.push(r),
                        Err(_) => break,
                    }
                }
                let planes: Vec<Vec<f32>> = batch.iter().map(|r| r.planes.clone()).collect();
                let reps: Vec<u8> = batch.iter().map(|r| r.rep).collect();
                match evaluator.evaluate_batch(&planes, &reps) {
                    Ok(results) => {
                        for (req, res) in batch.into_iter().zip(results.into_iter()) {
                            let _ = req.reply.send(Ok(res));
                        }
                    }
                    Err(e) => {
                        // Propagate the error to every waiter in this batch.
                        let msg = e.to_string();
                        for req in batch {
                            let _ = req.reply.send(Err(ort::Error::new(msg.clone())));
                        }
                    }
                }
            }
        });
        Self { sender: Some(tx), batcher: Some(batcher) }
    }
}

impl BatchEvaluator for InferenceServer {
    fn evaluate_batch(&self, planes: &[Vec<f32>], reps: &[u8]) -> ort::Result<Vec<(Vec<f32>, f32)>> {
        // Submit all rows, then collect - lets the batcher merge these with other
        // threads' requests into one GPU call.
        let sender = self.sender.as_ref().expect("server running");
        let mut receivers = Vec::with_capacity(planes.len());
        for (p, r) in planes.iter().zip(reps.iter()) {
            let (tx, rx) = channel();
            sender.send(Request { planes: p.clone(), rep: *r, reply: tx })
                .map_err(|_| ort::Error::new("inference server stopped"))?;
            receivers.push(rx);
        }
        let mut out = Vec::with_capacity(receivers.len());
        for rx in receivers {
            out.push(rx.recv().map_err(|_| ort::Error::new("inference server dropped reply"))?? );
        }
        Ok(out)
    }
}

impl Drop for InferenceServer {
    fn drop(&mut self) {
        // Close the channel so the batcher's recv() returns Err and the thread exits.
        self.sender = None;
        if let Some(h) = self.batcher.take() {
            let _ = h.join();
        }
    }
}
```

(If `ort::Error::new` differs in rc.10, use the crate's actual constructor; the implementer adapts. The exact `ort::Error` construction is the one place to verify against the installed ort version.)

- [ ] **Step 2: Test - concurrency + correctness**

`crates/kairnz-onnx/tests/server.rs`: build an `InferenceServer` from the fixture model; spawn 8 threads, each calling `evaluate_batch` on a couple positions; assert every result matches a direct `evaluate` within `1e-4`. (Proves cross-thread batching returns correct per-row results and the server shuts down cleanly on drop.)

- [ ] **Step 3: Re-export + commit**
```bash
cargo test -p kairnz-onnx server
git add crates/kairnz-onnx/src/batch.rs crates/kairnz-onnx/src/lib.rs crates/kairnz-onnx/tests/server.rs
git commit -m "feat(onnx): shared InferenceServer batches across threads"
```

### Task C3: BatchedAzMcts (virtual loss)

**Files:** Create `crates/kairnz-onnx/src/batched_mcts.rs`; modify `lib.rs`.

**Interfaces:**
- Consumes: `BatchEvaluator`, `AzMctsConfig` (extended), `encode_planes`/`legal_mask`/`action_to_index` from `kairnz-encode`, the existing node/PUCT logic from `mcts.rs` (replicate or share).
- Produces: `BatchedAzMcts::new(eval: &dyn BatchEvaluator, config: AzMctsConfig, seed: u64)`; `search(&mut self, game: &Game) -> Vec<(Action, u32)>`.

- [ ] **Step 1: Extend config**

Add to `AzMctsConfig` (mcts.rs): `pub leaves_per_step: usize` (K; default `8`) and `pub virtual_loss: f32` (default `1.0`). Keep existing fields/defaults.

- [ ] **Step 2: Implement BatchedAzMcts**

Mirror the existing `AzMcts` tree/node representation and PUCT (read `mcts.rs:119-280`), with these changes:
- Each node tracks an extra `vl: u32` (in-flight virtual visits).
- **Selection** (`select_leaf`) walks from root choosing the child maximizing PUCT with `N_eff = N + vl` and `Q_eff = (W - vl) / N_eff.max(1)`; along the chosen path increment each visited node's `vl += 1`; stop at the first unexpanded node (the leaf) or a terminal node; return the path (indices) and leaf.
- **Search loop**:
```
apply root dirichlet noise (seeded RNG) as the existing search does
while root.N < config.simulations:
    let want = min(config.leaves_per_step, config.simulations - root.N as usize)
    collect up to `want` leaves via select_leaf:
        - terminal leaf: compute terminal value, backup immediately (which also
          decrements vl along its path); do not add to the eval batch
        - non-terminal leaf: record (leaf_id, path, planes, rep); it keeps its vl
    if any eval leaves:
        let results = eval.evaluate_batch(&planes_vec, &reps_vec)?  // ONE batched call
        for each (leaf, path, (policy, value)):
            expand(leaf, policy)         // masked + renormalized priors, as AzMcts does
            backup(path, value)          // add real visit/value, decrement vl along path
return root children as (Action, visit_count)
```
- **Backup** adds `+1` visit and the value (with the existing sign convention from `mcts.rs`) to each node on the path AND decrements `vl` by 1 on each (undoing the virtual loss applied during selection).
- Collisions (same leaf selected twice in a batch): allowed; if a leaf in the batch was already expanded by an earlier row this step, skip its expand but still backup. Keep it simple and correct.
- `evaluate_batch` returning `Err` propagates out of `search` -> change `search` to return `ort::Result<Vec<(Action, u32)>>` OR keep `search` infallible by treating an eval error as a hard stop; choose `ort::Result` and update callers. (Decision: `search` returns `ort::Result<Vec<(Action, u32)>>`.)

- [ ] **Step 3: Tests**

`crates/kairnz-onnx/tests/batched_mcts.rs`, using the fixture model and a `DirectBatchEvaluator`:
- `batched_search_returns_valid_visit_distribution`: visit counts sum to ~`simulations`, all returned actions are legal.
- `batched_search_finds_mate_in_one`: set up the same mate-in-one position the existing `AzMcts` test uses (copy it) and assert `BatchedAzMcts` (with `leaves_per_step = 8`) selects the winning action. (Confirms virtual-loss batching does not break tactical correctness.)
- `leaves_per_step_one_matches_sequential_within_tolerance`: with `leaves_per_step = 1` and `virtual_loss = 0`, the visit distribution closely matches the sequential `AzMcts` on a fixed position (top move identical; this is the sanity bridge to the old engine).

- [ ] **Step 4: Re-export + commit**
```bash
cargo test -p kairnz-onnx batched_mcts
git add crates/kairnz-onnx/src/batched_mcts.rs crates/kairnz-onnx/src/mcts.rs crates/kairnz-onnx/src/lib.rs crates/kairnz-onnx/tests/batched_mcts.rs
git commit -m "feat(onnx): BatchedAzMcts with virtual-loss leaf batching"
```

### Task C4: Self-play on the shared server

**Files:** Modify `crates/kairnz-selfplay/src/parallel.rs`, `crates/kairnz-selfplay/src/play.rs` (if it takes `&mut AzMcts`), `crates/kairnz-selfplay/src/bin/selfplay.rs`.

**Interfaces:**
- `parallel_self_play(...)` builds ONE `InferenceServer` (single GPU session) and spawns `threads` game threads, each running a `BatchedAzMcts` that borrows `&server`. Adds `max_batch` and `leaves_per_step` params (CLI: `--max-batch`, `--leaves-per-step`).

- [ ] **Step 1: Rework `parallel_self_play`**

Read `parallel.rs`. Replace the per-thread `OnnxEvaluator` + `AzMcts` with: load ONE `OnnxEvaluator` -> `InferenceServer::new(eval, max_batch)`; `std::thread::scope`; each thread builds a `BatchedAzMcts::new(&server, mcts_config_with(leaves_per_step), base_seed + t)` and plays its games (the play loop calls `mcts.search(&game)?`). Keep per-thread RNG seeding. Collect samples in thread order as before. `play_game` (in `play.rs`) likely takes `&mut AzMcts`; add a sibling `play_game_batched(&mut BatchedAzMcts, ...)` or make the play loop generic over a trait `Searcher { fn search(&mut self, &Game) -> ort::Result<Vec<(Action,u32)>>; }` implemented by both. Prefer a small `Searcher` trait to avoid duplicating `play_game`.

- [ ] **Step 2: CLI flags**

`selfplay.rs`: add `--max-batch` (default `DEFAULT_MAX_BATCH`) and `--leaves-per-step` (default `8`); thread them through.

- [ ] **Step 3: Verify**

`cargo build -p kairnz-selfplay` warning-free. Run a tiny smoke: `cargo run --release -p kairnz-selfplay --bin selfplay -- --model crates/kairnz-onnx/tests/fixtures/random_init.onnx --out /tmp/s.safetensors --games 4 --simulations 32 --threads 2 --max-batch 16 --leaves-per-step 4` completes and writes a shard. (GPU optional; CPU fine for the smoke.) Commit:
```bash
git add crates/kairnz-selfplay/src/parallel.rs crates/kairnz-selfplay/src/play.rs crates/kairnz-selfplay/src/bin/selfplay.rs
git commit -m "feat(selfplay): batched self-play on a shared inference server"
```

### Task C5: Gate on the batched path

**Files:** Modify `crates/kairnz-selfplay/src/gate.rs`, `crates/kairnz-selfplay/src/bin/gate.rs`.

- [ ] **Step 1:** Read `gate.rs`. `run_gate` plays candidate vs best. Give each model its own `DirectBatchEvaluator` (or `InferenceServer`) and play with `BatchedAzMcts` via the `Searcher` trait from C4. Keep `GateResult` and seeds. The gate is two models alternating, so a `DirectBatchEvaluator` per model (single-threaded gate) is sufficient and simplest; use `leaves_per_step` to batch each search.
- [ ] **Step 2:** `cargo test -p kairnz-selfplay` passes (the fast-rule gate test); commit:
```bash
git add crates/kairnz-selfplay/src/gate.rs crates/kairnz-selfplay/src/bin/gate.rs
git commit -m "feat(selfplay): gate uses batched virtual-loss search"
```

---

# Phase D - Web dashboard

### Task D1: Dashboard server + endpoints

**Files:** Create `train/kairnz_train/dashboard.py`; add `train/tests/test_dashboard.py`.

**Interfaces:**
- Produces: a runnable module `python -m kairnz_train.dashboard --work DIR --host H --port P`; testable handler functions `status_payload(work) -> dict` and `metrics_payload(work) -> list[dict]` reading via `load_status`/`load_metrics`.

- [ ] **Step 1:** Implement with stdlib `http.server` (no new dependency): a `ThreadingHTTPServer` whose handler serves `GET /` (the static HTML), `GET /api/status` (`json.dumps(status_payload(work))`), `GET /api/metrics` (`json.dumps(metrics_payload(work))`). `status_payload`/`metrics_payload` are thin wrappers over `load_status`/`load_metrics` so they are unit-testable without a socket. `--host` defaults to `127.0.0.1` (localhost-bind; reach it via SSH tunnel).
- [ ] **Step 2:** `test_dashboard.py`: write a temp `status.json` + `metrics.jsonl`, assert `status_payload`/`metrics_payload` return the parsed structures (and empty defaults when missing).
- [ ] **Step 3:** `cd train && uv run pytest -q`; commit:
```bash
git add train/kairnz_train/dashboard.py train/tests/test_dashboard.py
git commit -m "feat(train): web dashboard server + JSON endpoints"
```

### Task D2: Dashboard page + task target + docs

**Files:** Create `train/kairnz_train/static/dashboard.html`; modify `Taskfile.yml`.

- [ ] **Step 1:** `dashboard.html`: a self-contained page that polls `/api/status` + `/api/metrics` every 1.5s and renders the current iteration/stage, a per-iteration table, and an `a_score` line + cumulative-promotions chart drawn on a `<canvas>` (no CDN dependency - a small hand-rolled line chart keeps it offline-friendly on a headless box). No inline color literals beyond a minimal `<style>` block of named CSS custom properties.
- [ ] **Step 2:** `Taskfile.yml`: a `dashboard` target running `uv run python -m kairnz_train.dashboard --work ../{{.WORK}}`. Document the SSH tunnel (`ssh -L 8080:127.0.0.1:8080 user@box`) in the runbook (Task F2).
- [ ] **Step 3:** `task --list-all` shows `dashboard`; commit:
```bash
git add train/kairnz_train/static/dashboard.html Taskfile.yml
git commit -m "feat(train): live dashboard page + task target"
```

---

# Phase E - App AI upgrade

### Task E1: App uses BatchedAzMcts + GPU

**Files:** Modify `src-tauri/src/ai.rs`.

- [ ] **Step 1:** Read `ai.rs`. Replace the cached `AzMctsPolicy` with a cached `(model, simulations, DirectBatchEvaluator)` driving a `BatchedAzMcts` for the chosen move. `AiEngine::choose` builds/reuses a `DirectBatchEvaluator` from `OnnxEvaluator::from_path` (which already tries the CUDA EP and falls back to CPU - so with cuDNN on PATH the app uses GPU automatically), constructs a `BatchedAzMcts` with `leaves_per_step` (e.g. `8`) and the requested `simulations`, runs `search(&game)?`, and returns the most-visited action. Map `ort::Error` to `String` as today. Keep the `&mut self`-free `Sync` cache shape (Mutex) already in place.
- [ ] **Step 2:** `cargo build -p kairnz-tauri` warning-free; `cargo test -p kairnz-tauri ai` still passes (update the test to the new internal type if needed; it should still assert a legal opening move). Commit:
```bash
git add src-tauri/src/ai.rs
git commit -m "feat(app): app AI uses batched virtual-loss search (GPU-capable)"
```

### Task E2: GPU launch target + difficulty bump

**Files:** Modify `Taskfile.yml`, `ui/src/components/ConfigPanel.svelte`.

- [ ] **Step 1:** `Taskfile.yml`: an `app-gpu` target that prepends `{{.TORCH_LIB}}` to PATH (same pattern as the GPU training targets) then runs `pnpm tauri dev`, so the app's ONNX Runtime finds cuDNN and uses the GPU.
- [ ] **Step 2:** `ConfigPanel.svelte`: raise the `DIFFICULTY_OPTIONS` sim counts now that GPU + batching make them cheap (e.g. Easy `100`, Medium `400`, Hard `1600`). Keep the named-constant pattern and CSS custom properties.
- [ ] **Step 3:** `cd ui && pnpm build` (0 errors); `task --list-all` shows `app-gpu`. Commit:
```bash
git add Taskfile.yml ui/src/components/ConfigPanel.svelte
git commit -m "feat(app): GPU launch target + stronger difficulty levels"
```

---

# Phase F - Validation + runbook

### Task F1: Strength-vs-baselines harness

**Files:** Create `crates/kairnz-selfplay/src/bin/strength.rs`.

**Interfaces:**
- A binary that plays a model (`BatchedAzMcts`) vs each of `RandomPolicy`, `GreedyPolicy`, and a plain (non-neural) MCTS (`MctsPolicy`) over N games each, alternating colors, and prints JSON win/draw/loss + score per baseline.

- [ ] **Step 1:** Reuse `kairnz-policy`'s `RandomPolicy`/`GreedyPolicy`/`MctsPolicy` and the existing match driver (`gate.rs::play_match` or `kairnz-bench::runner::play_game`). For each baseline, play `--games` games (half as P1, half as P2), tally results, print `{"baseline":"random","wins":..,"draws":..,"losses":..,"score":..}` lines. Args: `--model`, `--games` (default 50), `--simulations`, `--seed`.
- [ ] **Step 2:** A small test (fast rule, few games, fixture model) asserting the harness runs and returns a score in `[0,1]` for each baseline. Commit:
```bash
cargo test -p kairnz-selfplay strength
git add crates/kairnz-selfplay/src/bin/strength.rs
git commit -m "feat(eval): strength-vs-baselines harness (random/greedy/mcts)"
```

### Task F2: Runbook

**Files:** Create `docs/RUNBOOK-remote-training.md`.

- [ ] **Step 1:** Write the end-to-end operational guide: rent an H100 box (driver + nvidia-container-toolkit prereqs); `docker build` + `docker run --gpus all -v <data>:/kairnz/remote-run ...`; launch `scripts/run_remote.sh remote-run --iterations 60 --filters 128 --blocks 10 --selfplay-games 256 --selfplay-sims 400 --max-batch 256 --leaves-per-step 8 --threads <cores>`; watch via `task dashboard` over an SSH tunnel; resume after interruption by re-running the same command (`--resume` is in the script); when done, copy `best.onnx` locally and run `strength` to quantify absolute strength; load it in the app via `task app-gpu`. Note the net size is a CLI arg (default 128/10) and the determinism tradeoff. Commit:
```bash
git add docs/RUNBOOK-remote-training.md
git commit -m "docs: remote training runbook"
```

---

## Self-Review Notes

- **Spec coverage:** A (resume), B (headless), C (batched server + virtual loss, both Phase-1 cross-game and Phase-2 intra-game), D (web dashboard), E (app AI upgrade - the confirmed extra), F (absolute-strength validation + runbook). All four roadmap components plus the two user additions are covered.
- **Architecture decisions honored:** sync threads + batcher (C2), one `BatchEvaluator` trait with two backends (C1) reused by self-play (C4), gate (C5), and the app (E1); determinism stance encoded as tolerance tests (C1/C2/C3); net size left as a CLI arg (F2).
- **The unification is the elegant core:** `BatchedAzMcts` is generic over `&dyn BatchEvaluator`, so the exact same virtual-loss search runs in self-play (shared `InferenceServer`, cross-game batches) and in the app (`DirectBatchEvaluator`, single game) - no duplicated search logic.
- **Determinism:** no test asserts bit-exact reproducibility; the bridge test (`leaves_per_step=1, virtual_loss=0` ~ sequential) anchors correctness against the proven engine, and the mate-in-one test guards tactical soundness under batching.
- **Blast radius:** the legacy `AzMcts`/`AzMctsPolicy` stay until their callers migrate (self-play C4, gate C5, app E1); after E1 they may be unused - a later cleanup, not this plan.
- **Risk - the one place to verify against the installed ort:** `ort::Error` construction and the exact batched `Session::run` output extraction (C1/C2). Implementers verify by compiling against the pinned `ort` rc.
- **Operational, not coded here:** the actual `docker build`/GPU run and the multi-day training happen on the rented box (no GPU/Docker in the agent env); the deliverables are the correct Dockerfile, scripts, and runbook.
