# AlphaZero Plan 4: Self-Play Data Generation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate AlphaZero self-play training data: play games with the neural MCTS (with exploration), record each position's encoded planes, the MCTS visit-count policy target, the legal mask, and the final game outcome from that position's perspective, and write the samples to `.safetensors` shards the PyTorch trainer (Plan 5) will consume.

**Architecture:** A new `kairnz-selfplay` crate (library plus a `selfplay` binary) depending on `kairnz-onnx` (the `AzMcts` search and `OnnxEvaluator`), `kairnz-encode` (planes, action index, legal mask), and `kairnz-core`. Self-play runs the search with Dirichlet root noise and temperature-based move sampling for diversity, while the recorded policy target is always the raw visit distribution. Version 1 plays games sequentially with one reused evaluator; parallel/batched self-play is a deferred throughput optimization. Samples are written as stacked `.safetensors` tensors via the `safetensors` crate.

**Tech Stack:** Rust; `kairnz-onnx`/`kairnz-encode`/`kairnz-core`, `safetensors`, `bytemuck`, `rand`/`rand_pcg`, `clap`, `serde`/`serde_yaml`.

## Global Constraints

- Shard schema (the contract with the Plan 5 trainer), one shard = one `.safetensors` file with these tensors, `N` = sample count:
  - `planes`: `[N, 14, 9, 9]` f32 (from `encode_planes`).
  - `policy`: `[N, 6723]` f32, the normalized MCTS visit distribution (0 on non-searched/illegal entries).
  - `value`: `[N]` f32, the game outcome from each sample's side-to-move perspective: win `+1`, loss `-1`, draw `0`.
  - `legal_mask`: `[N, 6723]` u8, `1` for legal actions at that position else `0`.
- The policy target is the raw visit distribution `N(a) / sum_b N(b)`, indexed via `action_to_index(a, pos.to_move)`. Move SELECTION uses temperature (sample proportional to `N(a)` for the first `temperature_cutoff` plies, then argmax); selection temperature never changes the recorded target.
- Self-play search enables root Dirichlet noise (`dirichlet_epsilon = 0.25`, `dirichlet_alpha = 0.3` by default).
- Plane/policy sizes come from `kairnz_encode::{NUM_PLANES, POLICY_SIZE}`; never re-hardcode 14 or 6723.
- `f32` tensors are written little-endian (x86 native) via `bytemuck::cast_slice`.
- Rust: named constants, doc comments on public items, comprehensive error handling (`Result` on fallible paths), no em dashes, files well under 300 lines.
- GPU runs go through a Taskfile target that puts torch's lib (cuDNN) on PATH; `cargo test` runs on CPU.

---

## File Structure

- Create: `crates/kairnz-selfplay/Cargo.toml`
- Create: `crates/kairnz-selfplay/src/lib.rs` — module wiring, `SelfPlayConfig`.
- Create: `crates/kairnz-selfplay/src/sample.rs` — `Sample`, `policy_target`, `outcome_value`.
- Create: `crates/kairnz-selfplay/src/play.rs` — `play_game`.
- Create: `crates/kairnz-selfplay/src/shard.rs` — `write_shard`.
- Create: `crates/kairnz-selfplay/src/bin/selfplay.rs` — CLI.
- Modify: `Cargo.toml` — add the crate to workspace members.
- Modify: `Taskfile.yml` — add a `selfplay` GPU target.

---

### Task 1: Crate scaffold and `SelfPlayConfig`

**Files:**
- Create: `crates/kairnz-selfplay/Cargo.toml`, `crates/kairnz-selfplay/src/lib.rs`
- Modify: `Cargo.toml`

**Interfaces:**
- Produces: `SelfPlayConfig` (serde-deserializable, with `Default`) holding the search and self-play parameters consumed by later tasks.

- [ ] **Step 1: Add the crate to the workspace**

In root `Cargo.toml`, add `crates/kairnz-selfplay` to the `members` list.

- [ ] **Step 2: Create the manifest**

Create `crates/kairnz-selfplay/Cargo.toml`:

```toml
[package]
name = "kairnz-selfplay"
version = "0.1.0"
edition = "2021"

[dependencies]
kairnz-core = { path = "../kairnz-core" }
kairnz-encode = { path = "../kairnz-encode" }
kairnz-onnx = { path = "../kairnz-onnx" }
safetensors = "0.4"
bytemuck = "1"
rand = { workspace = true }
rand_pcg = { workspace = true }
clap = { workspace = true }
serde = { workspace = true }
serde_yaml = { workspace = true }
```

- [ ] **Step 3: Create the crate root with `SelfPlayConfig`**

Create `crates/kairnz-selfplay/src/lib.rs`:

```rust
//! Self-play data generation for the Kairnz AlphaZero pipeline.

pub mod sample;
pub mod play;
pub mod shard;

use serde::{Deserialize, Serialize};

/// Default number of MCTS simulations per move during self-play.
const DEFAULT_SIMULATIONS: u32 = 200;
/// Default PUCT exploration constant.
const DEFAULT_C_PUCT: f64 = 1.5;
/// Default Dirichlet root-noise weight (exploration is ON for self-play).
const DEFAULT_DIRICHLET_EPSILON: f64 = 0.25;
/// Default Dirichlet concentration.
const DEFAULT_DIRICHLET_ALPHA: f64 = 0.3;
/// Default number of opening plies that sample moves proportional to visits
/// before switching to argmax.
const DEFAULT_TEMPERATURE_CUTOFF: u32 = 20;
/// Default number of self-play games to generate.
const DEFAULT_GAMES: u32 = 64;

/// Parameters controlling a self-play run.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SelfPlayConfig {
    /// MCTS simulations per move.
    pub simulations: u32,
    /// PUCT exploration constant.
    pub c_puct: f64,
    /// Dirichlet root-noise weight.
    pub dirichlet_epsilon: f64,
    /// Dirichlet concentration.
    pub dirichlet_alpha: f64,
    /// Plies of visit-proportional sampling before switching to argmax.
    pub temperature_cutoff: u32,
    /// Number of games to play.
    pub games: u32,
}

impl Default for SelfPlayConfig {
    fn default() -> Self {
        SelfPlayConfig {
            simulations: DEFAULT_SIMULATIONS,
            c_puct: DEFAULT_C_PUCT,
            dirichlet_epsilon: DEFAULT_DIRICHLET_EPSILON,
            dirichlet_alpha: DEFAULT_DIRICHLET_ALPHA,
            temperature_cutoff: DEFAULT_TEMPERATURE_CUTOFF,
            games: DEFAULT_GAMES,
        }
    }
}

impl SelfPlayConfig {
    /// Builds the MCTS search config from these self-play parameters.
    pub fn mcts_config(&self) -> kairnz_onnx::AzMctsConfig {
        kairnz_onnx::AzMctsConfig {
            simulations: self.simulations,
            c_puct: self.c_puct,
            dirichlet_alpha: self.dirichlet_alpha,
            dirichlet_epsilon: self.dirichlet_epsilon,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_enables_root_noise_for_exploration() {
        let c = SelfPlayConfig::default();
        assert!(c.dirichlet_epsilon > 0.0, "self-play must explore");
        assert_eq!(c.mcts_config().dirichlet_epsilon, c.dirichlet_epsilon);
    }
}
```

Note: this file declares `pub mod sample; pub mod play; pub mod shard;` which do not exist until later tasks. Create empty placeholder files `src/sample.rs`, `src/play.rs`, `src/shard.rs` each containing a single line `// implemented in a later task` so Task 1 compiles; later tasks replace them.

- [ ] **Step 4: Run the test**

Run: `cargo test -p kairnz-selfplay`
Expected: PASS (1 test). The crate builds within the workspace.

- [ ] **Step 5: Commit**

```bash
git add Cargo.toml crates/kairnz-selfplay/Cargo.toml crates/kairnz-selfplay/src/lib.rs crates/kairnz-selfplay/src/sample.rs crates/kairnz-selfplay/src/play.rs crates/kairnz-selfplay/src/shard.rs
git commit -m "feat(selfplay): scaffold kairnz-selfplay crate with SelfPlayConfig"
```

---

### Task 2: `Sample` and target construction

**Files:**
- Create (replace placeholder): `crates/kairnz-selfplay/src/sample.rs`

**Interfaces:**
- Produces:
  - `Sample { planes: Vec<f32>, policy: Vec<f32>, value: f32, legal_mask: Vec<u8> }`
  - `policy_target(visits: &[(Action, u32)], to_move: Player) -> Vec<f32>` — a length-`POLICY_SIZE` normalized visit distribution.
  - `outcome_value(player: Player, result: GameResult) -> f32` — `+1`/`-1`/`0` from `player`'s perspective.

- [ ] **Step 1: Write the module and tests**

Replace `crates/kairnz-selfplay/src/sample.rs`:

```rust
//! A single training sample and helpers to build its policy and value targets.

use kairnz_core::actions::Action;
use kairnz_core::outcome::GameResult;
use kairnz_core::piece::Player;
use kairnz_encode::{action_to_index, POLICY_SIZE};

/// One training example: input planes, the MCTS policy target, the game-outcome
/// value target, and the legal-action mask, all for a single position.
#[derive(Clone, Debug, PartialEq)]
pub struct Sample {
    /// Encoded input planes (`NUM_PLANES * 81` floats).
    pub planes: Vec<f32>,
    /// Normalized visit-count policy target (length `POLICY_SIZE`).
    pub policy: Vec<f32>,
    /// Game outcome from this position's side-to-move perspective.
    pub value: f32,
    /// Legal-action mask (length `POLICY_SIZE`), `1` legal else `0`.
    pub legal_mask: Vec<u8>,
}

/// Builds the normalized visit-distribution policy target over `POLICY_SIZE`.
///
/// Each searched action contributes `visits / total_visits` at its policy index.
/// Returns an all-zero vector if there were no visits.
pub fn policy_target(visits: &[(Action, u32)], to_move: Player) -> Vec<f32> {
    let mut policy = vec![0.0f32; POLICY_SIZE];
    let total: u32 = visits.iter().map(|(_, v)| *v).sum();
    if total == 0 {
        return policy;
    }
    let total = total as f32;
    for (action, count) in visits {
        policy[action_to_index(action, to_move)] = *count as f32 / total;
    }
    policy
}

/// Game outcome from `player`'s perspective: win `+1`, loss `-1`, draw `0`.
pub fn outcome_value(player: Player, result: GameResult) -> f32 {
    match result {
        GameResult::Win(winner) if winner == player => 1.0,
        GameResult::Win(_) => -1.0,
        GameResult::Draw(_) => 0.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kairnz_core::outcome::DrawReason;
    use kairnz_core::square::Sq;

    #[test]
    fn policy_target_normalizes_visits() {
        let a = Action::Place { to: Sq(0) };
        let b = Action::Place { to: Sq(1) };
        let policy = policy_target(&[(a, 3), (b, 1)], Player::P1);
        assert_eq!(policy.len(), POLICY_SIZE);
        assert!((policy[action_to_index(&a, Player::P1)] - 0.75).abs() < 1e-6);
        assert!((policy[action_to_index(&b, Player::P1)] - 0.25).abs() < 1e-6);
        let sum: f32 = policy.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "distribution sums to one");
    }

    #[test]
    fn policy_target_empty_is_all_zero() {
        let policy = policy_target(&[], Player::P1);
        assert!(policy.iter().all(|p| *p == 0.0));
    }

    #[test]
    fn outcome_value_is_perspective_relative() {
        assert_eq!(outcome_value(Player::P1, GameResult::Win(Player::P1)), 1.0);
        assert_eq!(outcome_value(Player::P1, GameResult::Win(Player::P2)), -1.0);
        assert_eq!(outcome_value(Player::P1, GameResult::Draw(DrawReason::MaxPlies)), 0.0);
    }
}
```

- [ ] **Step 2: Run the tests**

Run: `cargo test -p kairnz-selfplay sample`
Expected: PASS (3 tests).

- [ ] **Step 3: Commit**

```bash
git add crates/kairnz-selfplay/src/sample.rs
git commit -m "feat(selfplay): add Sample and policy/value target builders"
```

---

### Task 3: The self-play game loop

**Files:**
- Create (replace placeholder): `crates/kairnz-selfplay/src/play.rs`

**Interfaces:**
- Consumes: `kairnz_onnx::AzMcts`, `kairnz_encode::{encode_planes, legal_mask}`, `kairnz_core::{game::Game, config::RuleConfig}`, `Sample`, `policy_target`, `outcome_value`.
- Produces: `play_game(mcts: &mut AzMcts, config: RuleConfig, temperature_cutoff: u32, rng: &mut Pcg64) -> Vec<Sample>` — plays one game from the standard opening and returns its samples with values filled in.

- [ ] **Step 1: Write the game loop and tests**

Replace `crates/kairnz-selfplay/src/play.rs`:

```rust
//! Plays one self-play game and records its training samples.

use kairnz_core::actions::Action;
use kairnz_core::config::RuleConfig;
use kairnz_core::game::Game;
use kairnz_core::piece::Player;
use kairnz_encode::{encode_planes, legal_mask};
use kairnz_onnx::AzMcts;
use rand::Rng;
use rand_pcg::Pcg64;

use crate::sample::{outcome_value, policy_target, Sample};

/// A partially-built sample: everything except the final value, plus the side
/// to move (needed to assign the perspective-relative value at game end).
struct PendingSample {
    planes: Vec<f32>,
    policy: Vec<f32>,
    legal_mask: Vec<u8>,
    to_move: Player,
}

/// Plays one self-play game from the standard opening using `mcts`, returning the
/// recorded samples with values assigned from the final result.
///
/// Moves are sampled proportional to visit counts for the first
/// `temperature_cutoff` plies (exploration), then chosen greedily (argmax). The
/// recorded policy target is always the raw visit distribution.
pub fn play_game(
    mcts: &mut AzMcts,
    config: RuleConfig,
    temperature_cutoff: u32,
    rng: &mut Pcg64,
) -> Vec<Sample> {
    let mut game = Game::new_standard(config);
    let mut pending: Vec<PendingSample> = Vec::new();
    let mut ply = 0u32;

    while game.terminal_result().is_none() {
        let visits = mcts.search(&game);
        if visits.is_empty() {
            break;
        }

        let to_move = game.pos.to_move;
        pending.push(PendingSample {
            planes: encode_planes(&game.pos, game.repetition_count()),
            policy: policy_target(&visits, to_move),
            legal_mask: legal_mask(&game.pos).iter().map(|b| *b as u8).collect(),
            to_move,
        });

        let action = select_move(&visits, ply < temperature_cutoff, rng);
        let _ = game.apply(action);
        ply += 1;
    }

    let result = game.terminal_result();
    pending
        .into_iter()
        .map(|p| {
            let value = match result {
                Some(r) => outcome_value(p.to_move, r),
                None => 0.0,
            };
            Sample { planes: p.planes, policy: p.policy, value, legal_mask: p.legal_mask }
        })
        .collect()
}

/// Selects a move from visit counts: proportional sampling when `explore` is
/// true, otherwise the most-visited action.
fn select_move(visits: &[(Action, u32)], explore: bool, rng: &mut Pcg64) -> Action {
    if explore {
        let total: u32 = visits.iter().map(|(_, v)| *v).sum();
        if total > 0 {
            let mut pick = rng.gen_range(0..total);
            for (action, count) in visits {
                if pick < *count {
                    return *action;
                }
                pick -= *count;
            }
        }
    }
    // Fallback and the post-cutoff path: most-visited action.
    visits
        .iter()
        .max_by_key(|(_, v)| *v)
        .map(|(a, _)| *a)
        .expect("visits is non-empty")
}

#[cfg(test)]
mod tests {
    use super::*;
    use kairnz_encode::{NUM_PLANES, POLICY_SIZE};
    use kairnz_onnx::{AzMctsConfig, OnnxEvaluator};
    use rand::SeedableRng;
    use std::path::PathBuf;

    fn fixture_mcts() -> AzMcts {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../kairnz-onnx/tests/fixtures/random_init.onnx");
        let evaluator = OnnxEvaluator::from_path(&path).expect("fixture loads");
        // Small simulation count keeps the test fast.
        let config = AzMctsConfig { simulations: 16, ..AzMctsConfig::default() };
        AzMcts::new(evaluator, config, 1)
    }

    #[test]
    fn play_game_produces_well_formed_samples() {
        let mut mcts = fixture_mcts();
        let mut rng = Pcg64::seed_from_u64(42);
        let samples = play_game(&mut mcts, RuleConfig::default(), 4, &mut rng);

        assert!(!samples.is_empty(), "a game produces at least one sample");
        for s in &samples {
            assert_eq!(s.planes.len(), NUM_PLANES * 81);
            assert_eq!(s.policy.len(), POLICY_SIZE);
            assert_eq!(s.legal_mask.len(), POLICY_SIZE);
            let policy_sum: f32 = s.policy.iter().sum();
            assert!((policy_sum - 1.0).abs() < 1e-4, "policy row sums to one");
            assert!(s.value == -1.0 || s.value == 0.0 || s.value == 1.0, "value in {{-1,0,1}}");
            assert!(s.legal_mask.iter().all(|m| *m == 0 || *m == 1), "mask is binary");
        }
    }
}
```

Note: this test plays a full game with the fixture network on CPU; keep `simulations` small (16) so it stays fast. If it is slow, lower `simulations`, not the assertions.

- [ ] **Step 2: Run the tests**

Run: `cargo test -p kairnz-selfplay play`
Expected: PASS (1 test).

- [ ] **Step 3: Commit**

```bash
git add crates/kairnz-selfplay/src/play.rs
git commit -m "feat(selfplay): add self-play game loop recording samples"
```

---

### Task 4: safetensors shard writer

**Files:**
- Create (replace placeholder): `crates/kairnz-selfplay/src/shard.rs`

**Interfaces:**
- Consumes: `Sample`, `kairnz_encode::{NUM_PLANES, POLICY_SIZE}`, `safetensors`, `bytemuck`.
- Produces: `write_shard(samples: &[Sample], path: &Path) -> Result<(), ShardError>` writing the four stacked tensors to a `.safetensors` file. `ShardError` wraps IO and safetensors errors.

- [ ] **Step 1: Write the shard writer and a roundtrip test**

Replace `crates/kairnz-selfplay/src/shard.rs`:

```rust
//! Writes self-play samples to a `.safetensors` shard.

use std::path::Path;

use kairnz_encode::{NUM_PLANES, POLICY_SIZE};
use safetensors::tensor::{Dtype, TensorView};
use safetensors::SafeTensorError;

use crate::sample::Sample;

/// Number of board cells per plane.
const BOARD_CELLS: usize = 81;

/// Errors writing a shard.
#[derive(Debug)]
pub enum ShardError {
    /// A safetensors serialization error.
    SafeTensors(SafeTensorError),
}

impl std::fmt::Display for ShardError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ShardError::SafeTensors(e) => write!(f, "safetensors error: {e}"),
        }
    }
}

impl std::error::Error for ShardError {}

impl From<SafeTensorError> for ShardError {
    fn from(e: SafeTensorError) -> Self {
        ShardError::SafeTensors(e)
    }
}

/// Writes `samples` to `path` as a `.safetensors` file with tensors
/// `planes [N,14,9,9] f32`, `policy [N,6723] f32`, `value [N] f32`, and
/// `legal_mask [N,6723] u8`.
pub fn write_shard(samples: &[Sample], path: &Path) -> Result<(), ShardError> {
    let n = samples.len();

    // Flatten each field into one contiguous buffer (row-major over samples).
    let mut planes: Vec<f32> = Vec::with_capacity(n * NUM_PLANES * BOARD_CELLS);
    let mut policy: Vec<f32> = Vec::with_capacity(n * POLICY_SIZE);
    let mut value: Vec<f32> = Vec::with_capacity(n);
    let mut legal_mask: Vec<u8> = Vec::with_capacity(n * POLICY_SIZE);
    for s in samples {
        planes.extend_from_slice(&s.planes);
        policy.extend_from_slice(&s.policy);
        value.push(s.value);
        legal_mask.extend_from_slice(&s.legal_mask);
    }

    let planes_view = TensorView::new(
        Dtype::F32,
        vec![n, NUM_PLANES, 9, 9],
        bytemuck::cast_slice(&planes),
    )?;
    let policy_view =
        TensorView::new(Dtype::F32, vec![n, POLICY_SIZE], bytemuck::cast_slice(&policy))?;
    let value_view = TensorView::new(Dtype::F32, vec![n], bytemuck::cast_slice(&value))?;
    let mask_view = TensorView::new(Dtype::U8, vec![n, POLICY_SIZE], &legal_mask)?;

    safetensors::serialize_to_file(
        [
            ("planes".to_string(), planes_view),
            ("policy".to_string(), policy_view),
            ("value".to_string(), value_view),
            ("legal_mask".to_string(), mask_view),
        ],
        None,
        path,
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sample::Sample;

    fn tiny_sample(value: f32) -> Sample {
        Sample {
            planes: vec![0.0; NUM_PLANES * BOARD_CELLS],
            policy: vec![0.0; POLICY_SIZE],
            value,
            legal_mask: vec![1u8; POLICY_SIZE],
        }
    }

    #[test]
    fn write_shard_roundtrips_shapes_and_values() {
        let samples = vec![tiny_sample(1.0), tiny_sample(-1.0)];
        let dir = std::env::temp_dir();
        let path = dir.join("kairnz_selfplay_test_shard.safetensors");
        write_shard(&samples, &path).expect("shard writes");

        let bytes = std::fs::read(&path).expect("read shard");
        let st = safetensors::SafeTensors::deserialize(&bytes).expect("deserialize");

        let planes = st.tensor("planes").expect("planes tensor");
        assert_eq!(planes.shape(), &[2, NUM_PLANES, 9, 9]);
        assert_eq!(planes.dtype(), Dtype::F32);

        let value = st.tensor("value").expect("value tensor");
        assert_eq!(value.shape(), &[2]);
        // Decode the two f32 values from the raw little-endian bytes. Decode via
        // from_le_bytes rather than bytemuck::cast_slice, because the file buffer
        // is not guaranteed to be 4-byte aligned (cast_slice would panic).
        let decoded: Vec<f32> = value
            .data()
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(decoded, vec![1.0, -1.0]);

        let mask = st.tensor("legal_mask").expect("mask tensor");
        assert_eq!(mask.dtype(), Dtype::U8);
        assert_eq!(mask.shape(), &[2, POLICY_SIZE]);

        let _ = std::fs::remove_file(&path);
    }
}
```

**Integration note (safetensors/bytemuck API):** the plan targets `safetensors = "0.4"` and `bytemuck = "1"`. If `TensorView::new` or `serialize_to_file` signatures differ in the resolved version (for example a `&[u8]` lifetime or an iterator-type bound), adapt minimally to compile while keeping the four tensor names, dtypes, and shapes exactly as specified, and note the adaptation in your report. The byte buffers (`planes`, `policy`, `value`, `legal_mask`) must outlive the `serialize_to_file` call, which they do here.

- [ ] **Step 2: Run the test**

Run: `cargo test -p kairnz-selfplay shard`
Expected: PASS (1 test).

- [ ] **Step 3: Commit**

```bash
git add crates/kairnz-selfplay/src/shard.rs
git commit -m "feat(selfplay): add safetensors shard writer"
```

---

### Task 5: Self-play CLI and Taskfile target

**Files:**
- Create: `crates/kairnz-selfplay/src/bin/selfplay.rs`
- Modify: `Taskfile.yml`

**Interfaces:**
- Consumes: `SelfPlayConfig`, `AzMcts`/`OnnxEvaluator`, `play_game`, `write_shard`.
- Produces: a `selfplay` binary that plays `games` games and writes one `.safetensors` shard, and a `task selfplay` target that runs it with cuDNN on PATH.

- [ ] **Step 1: Write the CLI**

Create `crates/kairnz-selfplay/src/bin/selfplay.rs`:

```rust
//! Self-play CLI: plays games with the neural MCTS and writes a training shard.

use std::path::PathBuf;
use std::process::ExitCode;

use clap::Parser;
use kairnz_core::config::RuleConfig;
use kairnz_onnx::{AzMcts, OnnxEvaluator};
use kairnz_selfplay::play::play_game;
use kairnz_selfplay::shard::write_shard;
use kairnz_selfplay::SelfPlayConfig;
use rand::SeedableRng;
use rand_pcg::Pcg64;

/// Command-line arguments for a self-play run.
#[derive(Parser)]
#[command(about = "Generate Kairnz self-play training shards.")]
struct Args {
    /// Path to the ONNX model to play with.
    #[arg(long)]
    model: PathBuf,
    /// Output shard path (.safetensors).
    #[arg(long)]
    out: PathBuf,
    /// Number of games to play.
    #[arg(long, default_value_t = 8)]
    games: u32,
    /// MCTS simulations per move.
    #[arg(long, default_value_t = 200)]
    simulations: u32,
    /// Base RNG seed.
    #[arg(long, default_value_t = 0)]
    seed: u64,
}

fn main() -> ExitCode {
    let args = Args::parse();
    let config = SelfPlayConfig {
        simulations: args.simulations,
        games: args.games,
        ..SelfPlayConfig::default()
    };

    let evaluator = match OnnxEvaluator::from_path(&args.model) {
        Ok(e) => e,
        Err(error) => {
            eprintln!("failed to load model: {error}");
            return ExitCode::FAILURE;
        }
    };
    println!("self-play backend: {:?}", evaluator.backend());

    let mut mcts = AzMcts::new(evaluator, config.mcts_config(), args.seed);
    let mut rng = Pcg64::seed_from_u64(args.seed);

    let mut samples = Vec::new();
    for g in 0..config.games {
        let game_samples = play_game(&mut mcts, RuleConfig::default(), config.temperature_cutoff, &mut rng);
        println!("game {g}: {} samples", game_samples.len());
        samples.extend(game_samples);
    }

    match write_shard(&samples, &args.out) {
        Ok(()) => {
            println!("wrote {} samples to {}", samples.len(), args.out.display());
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("failed to write shard: {error}");
            ExitCode::FAILURE
        }
    }
}
```

For the binary to access `play` and `shard`, ensure `lib.rs` declares `pub mod play; pub mod shard; pub mod sample;` (Task 1 already did). The binary refers to them as `kairnz_selfplay::play` / `kairnz_selfplay::shard`.

- [ ] **Step 2: Add the Taskfile target**

In `Taskfile.yml`, add a target that runs self-play on the GPU (reusing the `TORCH_LIB` var and PowerShell pattern from `onnx-check`):

```yaml
  # Generate a self-play shard on the GPU. Override vars, e.g.
  #   task selfplay MODEL=path/to/model.onnx OUT=data/shard0.safetensors GAMES=16
  selfplay:
    vars:
      MODEL: '{{.MODEL | default "crates/kairnz-onnx/tests/fixtures/random_init.onnx"}}'
      OUT: '{{.OUT | default "selfplay-shard.safetensors"}}'
      GAMES: '{{.GAMES | default 8}}'
    cmds:
      - powershell -NoProfile -Command '$env:PATH = "{{.TORCH_LIB}};$env:PATH"; cargo run -p kairnz-selfplay --bin selfplay --release -- --model "{{.MODEL}}" --out "{{.OUT}}" --games {{.GAMES}}'
```

- [ ] **Step 3: Build and smoke-test the CLI**

Run: `cargo build -p kairnz-selfplay`
Expected: builds warning-free.

Run a tiny CPU smoke run directly (no GPU needed to prove it works end to end):

```bash
cargo run -p kairnz-selfplay --bin selfplay -- --model crates/kairnz-onnx/tests/fixtures/random_init.onnx --out selfplay-smoke.safetensors --games 2 --simulations 16
```
Expected: prints the backend, two `game N: M samples` lines, and `wrote K samples to selfplay-smoke.safetensors`; the file exists. Then delete it: `rm selfplay-smoke.safetensors`.

- [ ] **Step 4: Run the full crate suite**

Run: `cargo test -p kairnz-selfplay`
Expected: PASS (config, sample, play, shard tests), warning-free.

Run: `cargo build --workspace`
Expected: the workspace builds.

- [ ] **Step 5: Commit**

```bash
git add crates/kairnz-selfplay/src/bin/selfplay.rs Taskfile.yml
git commit -m "feat(selfplay): add self-play CLI and GPU task target"
```

---

## Self-Review Notes

- **Spec coverage:** Implements the spec's Milestone 4 (self-play data generation). The `.safetensors` shard schema is the contract the Plan 5 PyTorch trainer consumes. The trainer, evaluation gating, and orchestration loop are later plans.
- **Targets are correct AlphaZero:** the policy target is the raw normalized visit distribution; the value target is the game outcome from each position's side-to-move perspective; move selection uses temperature only for exploration diversity and never alters the recorded target. Root Dirichlet noise is on for self-play.
- **Deferred (flagged):** Version 1 plays games sequentially with one reused evaluator. Parallel/batched self-play (per-thread evaluators or a batched inference server for GPU throughput) is the obvious next optimization and is intentionally out of scope here so the data pipeline lands correct and testable first. The `selfplay` task already runs on the GPU, so scaling up is a throughput change, not a correctness change.
- **Single-sourcing:** plane and policy sizes come from `kairnz_encode`. The shard tensor names/dtypes/shapes are the single contract; the Plan 5 trainer reads them by name via `safetensors`.
- **Type consistency:** `SelfPlayConfig` (+ `mcts_config`), `Sample`, `policy_target`, `outcome_value`, `play_game`, and `write_shard` signatures are referenced identically across tasks and the CLI.
