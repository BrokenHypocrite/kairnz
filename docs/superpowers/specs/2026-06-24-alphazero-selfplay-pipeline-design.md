# Kairnz AlphaZero Self-Play Pipeline (v1) Design

Date: 2026-06-24
Status: Approved for planning

## Goal and Scope

Produce a strong Kairnz model (a benchmark yardstick) via AlphaZero-style
self-play, exported to ONNX, plus a Rust `Policy` that plays using it.

This spec trains the single canonical ruleset (`dragon-default`) first. The
whole pipeline is parameterized by `RuleConfig`, so training a different rule
variant is a config change, not a code change.

In scope:
- State and action encoding for Kairnz with legal-move masking.
- Neural-guided MCTS (PUCT) in Rust using ONNX inference on the GPU.
- A self-play driver that emits training samples.
- A PyTorch trainer that consumes samples and exports ONNX.
- An evaluation gate that promotes stronger candidate models.
- An orchestration loop tying the stages together.
- A Rust inference `Policy` (`OnnxPolicy`) for the app and bench.

Out of scope (each gets its own spec later):
- Online competitive server and matchmaking.
- In-app hints, live scoring.
- Anti-cheat / analysis engines.
- Difficulty-tier tuning for human-vs-AI (the inference policy exists here, but
  tuning strength tiers is deferred).

## Primary Goals (from brainstorming)

- Goal: benchmark strength first ("both, research first"). Build the strongest
  model as a yardstick; derive scaled-down or difficulty-tuned versions later.
- Architecture: hybrid. Self-play stays in Rust; training is PyTorch. ONNX is
  the contract on both ends.
- Hardware target: RTX 4080 Super (16 GB VRAM), AMD 9950X (16 cores / 32
  threads), 96 GB system RAM. Long unattended runs are acceptable.
- Usable budgets (plan to these, not the raw totals): roughly 12 GB of the 16 GB
  VRAM is available for the model plus inference and training batches; system RAM
  idles near 38 percent, so plan for about 40 to 50 GB of usable RAM for replay
  buffers and data loading.

## The Iteration Loop

A classic AlphaZero loop with ONNX as the cross-language contract:

1. Self-play (Rust, GPU inference): play N games with neural-guided MCTS using
   the current best model; emit training samples.
2. Train (Python / PyTorch, GPU): sample a replay buffer, optimize the
   policy-plus-value loss, export a candidate ONNX.
3. Evaluate and gate (Rust, reuse `kairnz-bench`): candidate vs best over M
   games; promote the candidate to best if win rate is at or above a threshold
   (target 55 percent).
4. Repeat.

Self-play and game logic stay entirely in Rust (single source of truth). Python
only ever sees pre-encoded tensors and never touches game rules.

## Components

### 1. Encoding (`kairnz-encode`)

The only genuinely game-specific work, and the contract everything else depends
on. Lives in its own crate so the encoding is a single source of truth shared by
self-play, inference, and tests.

State to input planes: a stack of 9x9 float32 planes.

- Piece planes: {Stone height 1, Stone height 2, Stone height 3, Keystone} for
  each of 2 players = 8 planes.
- AP remaining: 1 plane (normalized).
- Per-player reserves: 2 planes (scalar broadcast).
- Mid-turn bitboards that affect legality: `capture_locked`, `keystone_moved` =
  2 planes.
- Repetition count of the current position: 1 plane.

Total roughly 14 planes (exact count finalized in implementation; the trainer
reads the channel count from the data, so it is not hard-coded in two places).

Canonical orientation: always orient the board from the side-to-move's
perspective (mirror when P2 moves) so the network is side-agnostic. Value targets
are always from the perspective of the side to move.

Action to fixed index: one flat vector of size 6723.

- Move: `from * 81 + to`, indices 0 to 6560.
- Place: `6561 + to`, indices 6561 to 6641.
- Stack: `6642 + target`, indices 6642 to 6722.

Provide `action_to_index`, `index_to_action`, and `legal_mask(pos) -> [bool;
6723]`. Illegal entries are masked before the softmax.

Decision granularity (important subtlety): a turn is up to 2 action points (2
actions, or one Stack that costs 2). The search branches at each action, not each
turn. `to_move` only flips at turn end, and `Position` already fully captures
mid-turn state (AP remaining, the two bitboards). The network is queried per
`Position`. The engine's own `apply` and turn-advance logic is the only mover, so
MCTS never re-implements rules. Within a turn, `to_move` is constant, so all
samples collected during one turn share the same `z` sign.

### 2. Neural MCTS (`AzMctsPolicy` in `kairnz-policy`)

Upgrade the existing uniform-rollout `MctsPolicy` into a neural-guided policy.

- At a leaf, query the network for a prior policy over legal actions and a scalar
  value `v`; back up `v` (no random rollout).
- PUCT selection: `Q + c_puct * P * sqrt(sum_N) / (1 + N)`.
- Dirichlet noise added at the root during self-play for exploration.
- Temperature: sample proportional to visit counts early in a game, then switch
  to argmax after a configurable move number.
- Leaf evaluations are batched to the GPU. Virtual loss allows multiple
  in-flight leaves per search.
- Inference via `ort` (ONNX Runtime) with the CUDA execution provider.

### 3. Self-Play Driver (`kairnz-selfplay`)

A new crate and binary that:

- Plays games with `AzMctsPolicy`, recording per move: encoded planes, the MCTS
  visit-count policy `pi`, and the side to move.
- At game end, assigns the outcome `z` to each recorded state from that state's
  perspective.
- Runs many games in parallel via `rayon` across the 16 cores, sharing the GPU
  through batched inference.
- Writes samples to disk as `.safetensors` shards (typed, zero-copy, read by both
  Rust and PyTorch).

A replay buffer of several million samples fits within the usable 40 to 50 GB of
system RAM; larger buffers spill to disk shards rather than growing in memory.

### 4. Training (`train/`, Python + PyTorch)

A standalone Python project (its own `uv` environment, separate from the pnpm and
Cargo builds).

- Network: ResNet. Conv stem, then K residual blocks of F filters, then a policy
  head (to 6723 logits) and a value head (to a tanh scalar).
- Starting size: F = 128, K = 10 (strong but fast on a 4080 Super 16 GB),
  scalable to 256 / 20 later. All sizes via YAML config.
- Loss: policy cross-entropy (target is the MCTS visit distribution) plus value
  MSE plus L2 weight decay.
- Replay buffer: sliding window over the most recent samples.
- Export ONNX with a dynamic batch dimension.
- Parity test: Rust `ort` output must equal PyTorch output on identical input
  within tolerance. This guards the export step.

### 5. Orchestration

A loop runner (a Taskfile target plus a driver script) alternating self-play,
train, and gate. It manages the `best.onnx` pointer and per-iteration artifacts
on disk:

- `models/iter_N.onnx`
- `samples/iter_N/*.safetensors`
- logs and metrics per iteration.

### 6. App and Bench Inference (`OnnxPolicy` in `kairnz-policy`)

A `Policy` implementation that loads `best.onnx` via `ort` and plays through
`AzMctsPolicy` (or raw-policy argmax for a fast, weak mode). This is what the
Tauri app and `kairnz-bench` consume. Difficulty tiers (sims, temperature,
smaller nets) are a later spec.

## Data Flow

```
RuleConfig
  -> self-play (Rust, ort + CUDA)
  -> .safetensors shards on disk
  -> trainer (PyTorch)
  -> candidate.onnx
  -> eval gate (Rust bench)
  -> promote -> best.onnx
  -> next iteration
App loads best.onnx via OnnxPolicy.
```

## Interfaces and Contracts

- Encoding contract: the plane layout and the action index map. Defined once in
  `kairnz-encode`. Python never encodes game state; it consumes pre-encoded
  planes from disk. This keeps a single source of encoding truth in Rust.
- ONNX contract: input shape `[B, C, 9, 9]` float32; outputs are policy logits
  `[B, 6723]` and a value `[B, 1]` in `[-1, 1]` (tanh).
- Sample shard schema (`.safetensors`): `planes [N, C, 9, 9]` f32, `policy [N,
  6723]` f32, `value [N]` f32, and `legal_mask [N, 6723]` bool or u8.

## Testing

Rust:
- Action index round-trip: `index_to_action(action_to_index(a)) == a` over all
  legal actions in sampled positions.
- `legal_mask` agrees exactly with `legal_actions`.
- Canonical-perspective symmetry: a position and its mirror encode consistently.
- AzMCTS finds a mate-in-one keystone capture when one exists.

Parity:
- The same `Position` produces matching Rust `ort` and PyTorch outputs within
  tolerance.

Python:
- Forward pass produces the expected output shapes.
- Loss decreases on a small fixed toy batch.
- ONNX export then reload reproduces the PyTorch output.

End-to-end smoke:
- One tiny iteration (a few games, a few sims, one train step) produces an ONNX
  that loads and plays legal moves in Rust.

## Build-Order Milestones

1. Encoding and masking plus tests (no ML yet).
2. ONNX seam proven: `OnnxPolicy` loads a randomly initialized exported network
   and plays legal moves. This validates the cross-language contract before any
   training is attempted.
3. AzMCTS (PUCT) with batched inference.
4. Self-play driver emitting shards.
5. PyTorch network, trainer, ONNX export, and parity test.
6. Orchestration loop and eval gating in bench.
7. First real training run; plot strength vs Random, Greedy, and plain MCTS.

## Dependencies Added

Rust:
- `ort` (ONNX Runtime, CUDA feature).
- `safetensors`.
- `ndarray`.
- `rayon`.
- A progress bar crate such as `indicatif` (optional).

Python (in `train/`, managed by `uv`):
- `torch` (CUDA build).
- `numpy`.
- `safetensors`.
- `onnx`.
- `onnxruntime-gpu` (for the parity check).
- `pyyaml`.

## Decisions and Assumptions

- Canonical perspective encoding (mirror the board for P2).
- `dragon-default` ruleset trained first; pipeline parameterized for others.
- `.safetensors` as the sample interchange format.
- `ort` plus CUDA for Rust-side inference; PyTorch for training.
- Network starts at F = 128, K = 10.
- Python environment via `uv`; the `train/` directory is its own project.
- Milestone 2 (prove the ONNX seam with a random network before training) is
  retained to de-risk the cross-language contract early.

## Open Questions for Implementation

- Final exact plane count and ordering (the trainer reads channel count from the
  data, so this is not duplicated).
- Mid-turn positions with action points remaining but no legal action: confirm
  the engine's behavior (turn advance) and that the self-play driver routes all
  moves through the engine's `apply` path so no rule logic is duplicated.
- Batched-inference design detail: virtual-loss count and batch size tuning for
  the 4080 Super.
