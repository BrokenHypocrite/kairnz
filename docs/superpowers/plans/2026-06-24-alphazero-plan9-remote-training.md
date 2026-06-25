# AlphaZero Plan 9: Remote Training Infrastructure (ROADMAP - HOLD)

> **STATUS: WRITTEN BUT NOT FOR IMMEDIATE IMPLEMENTATION.** This plan captures the work needed to run serious training on a rented H100-class box and watch it remotely. Implement it later, after the current local results are accepted. The batched-inference server (Component C) is the largest piece and may warrant its own brainstorming pass before execution.

> **For agentic workers (when activated):** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Make a long, uninterrupted, GPU-saturating training run practical on a rented H100 (or better), monitored from a web dashboard, so we can produce a genuinely strong Kairnz model for AI opponents and analysis engines.

**Why these four:** the current loop (1) cannot resume after an interruption, (2) has no headless/repeatable deployment, (3) underutilizes the GPU (~37%) because self-play does batch-1 inference, and (4) is monitored only by a local desktop window. An H100 is wasted without (3); a multi-day rented run is fragile without (1) and (2); and remote monitoring needs (4).

**Tech Stack:** Rust (`kairnz-onnx`/`kairnz-selfplay`, `ort`, channels), Python (`train/`, a small web server), Docker/CUDA.

---

## Component A: Resume support (small)

**Problem:** `loop.py` unconditionally seeds a fresh random `best.onnx`/`best.pt` every run, so a stopped/reclaimed run loses all progress.

**Change:** make the seed conditional and continue from existing artifacts.
- Add `--resume` (or auto-detect): if `best.onnx` AND `best.pt` exist in the work dir, load them as the starting best instead of seeding random.
- Continue the iteration counter from the last `metrics.jsonl` row (so iteration numbers and the shard window stay consistent); keep appending metrics.
- If `best` exists but metrics do not, start at iteration 0 from the existing best.

**Tasks:** (1) `loop.py` resume logic + a unit-testable `next_iteration(metrics_path) -> int` helper and a `resolve_start(work, resume) -> (start_iter, seeded_bool)` helper, with tests. (2) Document the resume workflow.

**Acceptance:** stopping the loop and re-running with the same `--work --resume` continues from the last best and iteration, not from random.

---

## Component B: Headless deployment + remote box setup (small-medium)

**Problem:** the pipeline is Rust + `ort` + CUDA + cuDNN + torch + a repo build, not a one-liner; it must stand up reproducibly on a rented headless box.

**Change:** a Dockerfile and a launch script.
- `Dockerfile`: base on `nvidia/cuda:12.x-cudnn-runtime-ubuntu22.04` (cuDNN in the image means the ONNX Runtime CUDA provider loads without the torch-lib PATH hack). Install the Rust toolchain, `uv`, copy the repo, `cargo build --release -p kairnz-selfplay`, `uv sync` in `train/`.
- A `scripts/run_remote.sh` (or a Taskfile target) that runs `loop.py` with logging to files under the work dir, and starts the web dashboard (Component D).
- `docker run --gpus all ...` for GPU passthrough; document the rented-box prerequisites (NVIDIA driver, nvidia-container-toolkit).

**Tasks:** (1) Dockerfile + build verification (image builds, `backend_check` reports CUDA inside the container). (2) Launch script + docs (clone-to-running in minutes; how to attach the dashboard).

**Acceptance:** `docker build` then `docker run --gpus all` yields a container where `task onnx-check` (or `backend_check`) prints `backend: CUDA` and `loop.py` runs.

---

## Component C: Batched-inference server (large - the throughput unlock)

**Problem:** self-play does batch-1 GPU inference (one MCTS leaf at a time per thread), so the GPU sits at ~37% utilization. An H100 would be wasted: batch-1 is latency-bound and barely faster than a 4080. Batching leaf evaluations into large GPU batches is the single biggest speedup and the reason to rent a fast GPU at all.

**Design (start with cross-game batching; add virtual loss later):**
- An `InferenceServer` owns the ONNX `Session` and exposes an `mpsc` request channel. A request is `(planes, repetition_count, oneshot_response_sender)`.
- A dedicated batcher thread drains the channel: it collects requests until it has `BATCH_SIZE` of them OR a small timeout (e.g. 1-2 ms) elapses, runs ONE batched inference `[B, 14, 9, 9] -> [B, 6723] + [B, 1]`, and sends each row back through its response sender.
- `AzMcts` is restructured so a leaf evaluation SUBMITS to the server and AWAITS the result instead of calling `evaluator.evaluate` inline. With many concurrent self-play games (the existing thread pool), the batcher naturally fills batches across games.
- **Phase 1 (cross-game batching):** each game still selects one leaf at a time but routes the eval through the central batcher. With N concurrent games, batches of ~N form automatically. Big win, moderate complexity.
- **Phase 2 (intra-game, virtual loss):** to fill batches even with few games, add virtual loss so one search can have multiple leaves in flight: temporarily penalize a selected path so concurrent selections diverge, collect K leaves, submit as a batch, back up when results return. More complex; do only if Phase 1 leaves the GPU underfed.

**Risk/notes:** this restructures the hot path of `AzMcts` and the self-play driver; it interacts with reproducibility (batching order must not change per-game results - keep each game's RNG/selection deterministic and only batch the *evaluation*, which is pure). This component likely deserves its own brainstorming + a dedicated multi-task plan.

**Acceptance:** on the rented GPU, self-play GPU utilization rises from ~37% to ~85%+ and self-play throughput (samples/hour) increases several-fold at the same game/sim settings.

---

## Component D: Web-based training dashboard (small-medium)

**Problem:** monitoring is a local tkinter window; a remote run needs browser-accessible progress.

**Change:** a small web server in `train/` that serves a live dashboard from `status.json` + `metrics.jsonl`.
- A lightweight Python server (FastAPI or stdlib `http.server`) with: `GET /` (an HTML+JS page), `GET /api/status` (returns status.json), `GET /api/metrics` (returns metrics rows). The page polls every ~1.5s and renders the current iteration/stage plus a chart of `a_score` and cumulative promotions over iterations (a small inline canvas or Chart.js via CDN).
- **Remote access + security:** prefer an **SSH tunnel** (`ssh -L 8080:localhost:8080 box`) so nothing is exposed publicly; alternatively bind `0.0.0.0:PORT` behind a simple bearer token. Document both; default to localhost-bind + tunnel.

**Tasks:** (1) the server + endpoints reading the same `load_status`/`load_metrics` helpers (reuse from `kairnz_train.orchestrate`), with tests for the endpoint handlers. (2) the HTML/JS page (chart + table). (3) a `task dashboard` target and SSH-tunnel docs.

**Acceptance:** with the loop running, opening the dashboard URL (via SSH tunnel) shows live iteration/stage, a per-iteration table, and an `a_score`/promotions chart that updates.

---

## Suggested Implementation Order (when activated)

1. **Resume (A)** + **Headless (B)** together - small, and they unblock any remote run.
2. **Web dashboard (D)** - so the remote run is observable.
3. **Batched-inference server (C)** - the big throughput build; brainstorm first, then its own plan. This is what makes an H100 worth renting.
4. Then: the serious run - `128/10` net, many iterations, on the rented box, watched via the dashboard; finish with a strength-vs-baselines (Random/Greedy/plain-MCTS) measurement to quantify absolute strength.

## Out of scope here
- Multi-box horizontal self-play (several rented boxes pooling self-play data) - a further scale-up after the batched server.
- Distributed training (single-GPU training is fine at this net size).
