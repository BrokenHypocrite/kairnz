# AlphaZero Plan 3: Neural MCTS (PUCT) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `AzMctsPolicy`, a neural-guided PUCT Monte Carlo Tree Search that uses the ONNX model's policy priors and value estimate (instead of random rollouts) to choose Kairnz moves, returning per-action visit counts that later become self-play training targets.

**Architecture:** A new `mcts` module in the existing `kairnz-onnx` crate (which already owns the `OnnxEvaluator` and `ort`). The search reuses the proven structure of `kairnz-policy`'s `MctsPolicy`: a flat arena of nodes, each storing its value from its own `to_move`'s perspective, with a sign flip on turn handover. It differs in two ways: at a leaf it queries the network for (policy priors, value) rather than rolling out, and it selects children by PUCT rather than UCB1. Terminal leaves use the true game result, never the network, which keeps tactics (winning captures) exact. Root Dirichlet-noise machinery is included but inert by default (epsilon 0), so Plan 4 self-play can enable exploration without touching the search.

**Tech Stack:** Rust; `kairnz-onnx` (`OnnxEvaluator`), `kairnz-encode` (`action_to_index`), `kairnz-core` (`Game`, `Action`), `rand`/`rand_pcg`/`rand_distr` (Dirichlet noise).

## Global Constraints

- Value convention (CRITICAL, mirrors `kairnz-policy::mcts`): a node's `value_sum` accumulates the leaf value `v` from the perspective of that node's `to_move`. During backprop, a node on the path receives `+v` if its `to_move` equals the evaluated leaf's `to_move`, else `-v`. During PUCT selection, a child's mean `Q` (stored in the child's perspective) is read from the parent's perspective: used directly when `child.to_move == parent.to_move` (within one multi-AP turn), negated otherwise (turn handover).
- Value range is `[-1, 1]` (the network's tanh value head). Terminal value from a node's `to_move`: `Win(to_move) = +1`, `Win(opponent) = -1`, `Draw = 0`.
- Terminal leaves are scored by `Game::terminal_result`, never by the network.
- PUCT score for a child: `Q_parent_perspective + c_puct * prior * sqrt(parent.visits) / (1 + child.visits)`. Unvisited children use `Q = 0`.
- Priors are a softmax over ONLY the legal actions' logits, indexed by `action_to_index(action, pos.to_move)`; illegal actions get no child.
- The encoder needs a repetition count; obtain it from `Game::repetition_count` (added in Task 1), passed to `OnnxEvaluator::evaluate`.
- Default config: `simulations = 400`, `c_puct = 1.5`, `dirichlet_alpha = 0.3`, `dirichlet_epsilon = 0.0` (no root noise). With `epsilon = 0.0` the search is fully deterministic given a model.
- Rust: named constants, doc comments on public items, comprehensive error handling (`.expect()` only on invariants), no em dashes, files well under 300 lines.

---

## File Structure

- Modify: `crates/kairnz-core/src/game.rs` — add `Game::repetition_count`.
- Modify: `crates/kairnz-onnx/Cargo.toml` — add `rand`, `rand_pcg`, `rand_distr`.
- Create: `crates/kairnz-onnx/src/mcts.rs` — `AzMctsConfig`, helper functions, the search, and `AzMctsPolicy`.
- Modify: `crates/kairnz-onnx/src/lib.rs` — `pub mod mcts;` and re-exports.

---

### Task 1: `Game::repetition_count` in kairnz-core

**Files:**
- Modify: `crates/kairnz-core/src/game.rs`

**Interfaces:**
- Produces: `Game::repetition_count(&self) -> u8` — how many times the current position's Zobrist hash appears in the turn-boundary history, saturating at `u8::MAX`. Consumed by the MCTS evaluation step (Task 3) to feed the encoder's repetition plane.

- [ ] **Step 1: Write the failing test**

In `crates/kairnz-core/src/game.rs`, inside the existing `#[cfg(test)] mod tests`, add:

```rust
    #[test]
    fn repetition_count_counts_history_occurrences() {
        // A fresh standard game has seeded its opening hash once.
        let game = Game::new_standard(RuleConfig::default());
        assert_eq!(game.repetition_count(), 1, "opening hash seeded once");
    }

    #[test]
    fn repetition_count_reflects_injected_repeats() {
        let mut game = Game::new_standard(RuleConfig::default());
        let h = game.pos.zobrist;
        game.history.push(h);
        game.history.push(h);
        assert_eq!(game.repetition_count(), 3, "opening hash now appears three times");
    }
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test -p kairnz-core repetition_count`
Expected: FAIL (`no method named repetition_count`).

- [ ] **Step 3: Implement the method**

In `crates/kairnz-core/src/game.rs`, add this method inside `impl Game` (after `to_move`):

```rust
    /// Returns how many times the current position's Zobrist hash appears in the
    /// turn-boundary history, saturating at `u8::MAX`.
    ///
    /// This is the repetition signal fed to the neural-network encoder. It counts
    /// the same occurrences the repetition draw rule uses.
    pub fn repetition_count(&self) -> u8 {
        let count = self.history.iter().filter(|&&h| h == self.pos.zobrist).count();
        u8::try_from(count).unwrap_or(u8::MAX)
    }
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo test -p kairnz-core repetition_count`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add crates/kairnz-core/src/game.rs
git commit -m "feat(core): add Game::repetition_count accessor"
```

---

### Task 2: MCTS config and pure helper functions

**Files:**
- Modify: `crates/kairnz-onnx/Cargo.toml`
- Create: `crates/kairnz-onnx/src/mcts.rs`
- Modify: `crates/kairnz-onnx/src/lib.rs`

**Interfaces:**
- Produces:
  - `AzMctsConfig { simulations: u32, c_puct: f64, dirichlet_alpha: f64, dirichlet_epsilon: f64 }` with `Default`.
  - `pub(crate) fn terminal_value(to_move: Player, result: GameResult) -> f64`
  - `pub(crate) fn legal_priors(logits: &[f32], legal: &[Action], to_move: Player) -> Vec<f32>` — softmax over the legal actions' logits, aligned to `legal` order, summing to ~1.
  These are consumed by the search in Task 3.

- [ ] **Step 1: Add dependencies**

In `crates/kairnz-onnx/Cargo.toml`, add to `[dependencies]` (keep the existing ones):

```toml
rand = "0.8"
rand_pcg = "0.3"
rand_distr = "0.4"
```

- [ ] **Step 2: Write the module with config and helpers plus their tests**

Create `crates/kairnz-onnx/src/mcts.rs`:

```rust
//! Neural-guided PUCT Monte Carlo Tree Search over Kairnz positions.

use kairnz_core::actions::Action;
use kairnz_core::outcome::GameResult;
use kairnz_core::piece::Player;
use kairnz_encode::action_to_index;

/// Default number of simulations per move.
const DEFAULT_SIMULATIONS: u32 = 400;
/// Default PUCT exploration constant.
const DEFAULT_C_PUCT: f64 = 1.5;
/// Default Dirichlet concentration for root exploration noise.
const DEFAULT_DIRICHLET_ALPHA: f64 = 0.3;
/// Default root-noise weight. Zero disables noise, making search deterministic.
const DEFAULT_DIRICHLET_EPSILON: f64 = 0.0;

/// Terminal value of a win from the winning side's perspective.
const WIN_VALUE: f64 = 1.0;
/// Terminal value of a loss from the losing side's perspective.
const LOSS_VALUE: f64 = -1.0;
/// Terminal value of a draw.
const DRAW_VALUE: f64 = 0.0;

/// Search parameters for [`AzMctsPolicy`].
#[derive(Clone, Copy, Debug)]
pub struct AzMctsConfig {
    /// Number of simulations performed per move.
    pub simulations: u32,
    /// PUCT exploration constant.
    pub c_puct: f64,
    /// Dirichlet concentration parameter for root noise.
    pub dirichlet_alpha: f64,
    /// Root-noise mixing weight in `[0, 1]`; `0.0` disables noise.
    pub dirichlet_epsilon: f64,
}

impl Default for AzMctsConfig {
    fn default() -> Self {
        AzMctsConfig {
            simulations: DEFAULT_SIMULATIONS,
            c_puct: DEFAULT_C_PUCT,
            dirichlet_alpha: DEFAULT_DIRICHLET_ALPHA,
            dirichlet_epsilon: DEFAULT_DIRICHLET_EPSILON,
        }
    }
}

/// Terminal value of `result` from `to_move`'s perspective, in `[-1, 1]`.
pub(crate) fn terminal_value(to_move: Player, result: GameResult) -> f64 {
    match result {
        GameResult::Win(winner) if winner == to_move => WIN_VALUE,
        GameResult::Win(_) => LOSS_VALUE,
        GameResult::Draw(_) => DRAW_VALUE,
    }
}

/// Softmax priors over only the legal actions, aligned to `legal`'s order.
///
/// Each legal action's logit is read from the policy vector via
/// `action_to_index`, then a numerically stable softmax is applied. The result
/// sums to approximately 1 and is used as the PUCT prior for each child.
pub(crate) fn legal_priors(logits: &[f32], legal: &[Action], to_move: Player) -> Vec<f32> {
    if legal.is_empty() {
        return Vec::new();
    }
    let raw: Vec<f32> = legal
        .iter()
        .map(|a| logits[action_to_index(a, to_move)])
        .collect();
    let max = raw.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = raw.iter().map(|x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|e| e / sum).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use kairnz_core::square::Sq;
    use kairnz_encode::POLICY_SIZE;

    #[test]
    fn terminal_value_is_perspective_relative() {
        assert_eq!(terminal_value(Player::P1, GameResult::Win(Player::P1)), 1.0);
        assert_eq!(terminal_value(Player::P1, GameResult::Win(Player::P2)), -1.0);
        assert_eq!(
            terminal_value(Player::P1, GameResult::Draw(kairnz_core::outcome::DrawReason::MaxPlies)),
            0.0
        );
    }

    #[test]
    fn legal_priors_softmax_sums_to_one_over_legal() {
        let mut logits = vec![0.0f32; POLICY_SIZE];
        let a = Action::Place { to: Sq(0) };
        let b = Action::Place { to: Sq(1) };
        logits[action_to_index(&a, Player::P1)] = 2.0;
        logits[action_to_index(&b, Player::P1)] = 0.0;

        let priors = legal_priors(&logits, &[a, b], Player::P1);
        assert_eq!(priors.len(), 2);
        let sum: f32 = priors.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "priors sum to one");
        assert!(priors[0] > priors[1], "higher logit gets higher prior");
    }

    #[test]
    fn legal_priors_empty_for_no_actions() {
        assert!(legal_priors(&[0.0; POLICY_SIZE], &[], Player::P1).is_empty());
    }
}
```

- [ ] **Step 3: Wire the module into `lib.rs`**

In `crates/kairnz-onnx/src/lib.rs`, add after the existing `pub mod policy;`:

```rust
pub mod mcts;

pub use mcts::{AzMctsConfig, AzMctsPolicy};
```

Note: `AzMctsPolicy` does not exist until Task 4. For this task, re-export only `AzMctsConfig`:

```rust
pub mod mcts;

pub use mcts::AzMctsConfig;
```

Task 4 adds the `AzMctsPolicy` re-export.

- [ ] **Step 4: Run the tests**

Run: `cargo test -p kairnz-onnx mcts`
Expected: PASS (3 helper tests). The crate builds with the new deps.

- [ ] **Step 5: Commit**

```bash
git add crates/kairnz-onnx/Cargo.toml crates/kairnz-onnx/src/mcts.rs crates/kairnz-onnx/src/lib.rs
git commit -m "feat(onnx): add AzMctsConfig and MCTS helper functions"
```

---

### Task 3: The PUCT search

**Files:**
- Modify: `crates/kairnz-onnx/src/mcts.rs`

**Interfaces:**
- Consumes: `OnnxEvaluator`, `AzMctsConfig`, `terminal_value`, `legal_priors`, `Game::repetition_count`, `kairnz_core::actions::legal_actions`.
- Produces: `AzMcts` with `AzMcts::new(evaluator: OnnxEvaluator, config: AzMctsConfig, seed: u64) -> AzMcts` and `AzMcts::search(&mut self, game: &Game) -> Vec<(Action, u32)>` returning each root child's `(action, visit_count)`. Consumed by `AzMctsPolicy` (Task 4) and Plan 4 self-play.

- [ ] **Step 1: Add the search implementation and tests**

In `crates/kairnz-onnx/src/mcts.rs`, add these imports at the top (merge with existing `use` lines):

```rust
use kairnz_core::actions::legal_actions;
use kairnz_core::game::Game;
use rand::SeedableRng;
use rand_distr::Distribution;
use rand_pcg::Pcg64;

use crate::OnnxEvaluator;
```

Then add the node type and search (place above the `#[cfg(test)]` module):

```rust
/// A node in the search tree, stored in a flat arena addressed by `usize`.
///
/// VALUE PERSPECTIVE: `value_sum` accumulates leaf values from the perspective of
/// this node's `to_move` (see the crate's value convention). `prior` is this
/// node's PUCT prior as a child of its parent.
struct Node {
    game: Game,
    to_move: Player,
    action_from_parent: Option<Action>,
    parent: Option<usize>,
    prior: f32,
    children: Vec<usize>,
    expanded: bool,
    visits: u32,
    value_sum: f64,
}

impl Node {
    fn new(game: Game, parent: Option<usize>, action_from_parent: Option<Action>, prior: f32) -> Node {
        let to_move = game.pos.to_move;
        Node {
            game,
            to_move,
            action_from_parent,
            parent,
            prior,
            children: Vec::new(),
            expanded: false,
            visits: 0,
            value_sum: 0.0,
        }
    }
}

/// A neural-guided PUCT search. Owns the model evaluator and a seeded RNG used
/// only for root Dirichlet noise (inert when `dirichlet_epsilon` is 0).
pub struct AzMcts {
    evaluator: OnnxEvaluator,
    config: AzMctsConfig,
    rng: Pcg64,
}

impl AzMcts {
    /// Builds a search owning `evaluator`, seeded for reproducible root noise.
    pub fn new(evaluator: OnnxEvaluator, config: AzMctsConfig, seed: u64) -> AzMcts {
        AzMcts { evaluator, config, rng: Pcg64::seed_from_u64(seed) }
    }

    /// Runs the configured number of simulations from `game` and returns each
    /// root child's `(action, visit_count)`. Empty if the root is terminal or has
    /// no legal action.
    pub fn search(&mut self, game: &Game) -> Vec<(Action, u32)> {
        if game.terminal_result().is_some() {
            return Vec::new();
        }
        let mut arena: Vec<Node> = vec![Node::new(game.clone(), None, None, 0.0)];

        for _ in 0..self.config.simulations {
            self.simulate(&mut arena);
        }

        arena[0]
            .children
            .iter()
            .map(|&c| {
                let child = &arena[c];
                (child.action_from_parent.expect("child has an action"), child.visits)
            })
            .collect()
    }

    /// One selection -> evaluation/expansion -> backpropagation cycle.
    fn simulate(&mut self, arena: &mut Vec<Node>) {
        // Selection: descend by PUCT through expanded, non-terminal nodes.
        let mut path: Vec<usize> = vec![0];
        let mut current = 0usize;
        while arena[current].expanded && arena[current].game.terminal_result().is_none() {
            match self.best_child(arena, current) {
                Some(child) => {
                    current = child;
                    path.push(current);
                }
                None => break,
            }
        }

        // Evaluation: terminal leaves use the true result; others query the net
        // and expand all legal children with their priors.
        let leaf = current;
        let value = if let Some(result) = arena[leaf].game.terminal_result() {
            terminal_value(arena[leaf].to_move, result)
        } else {
            self.expand_and_evaluate(arena, leaf)
        };

        // Backpropagation in the leaf's perspective.
        let leaf_to_move = arena[leaf].to_move;
        for &idx in &path {
            arena[idx].visits += 1;
            let signed = if arena[idx].to_move == leaf_to_move { value } else { -value };
            arena[idx].value_sum += signed;
        }
    }

    /// Evaluates `leaf` with the network, creates a child per legal action with
    /// its softmax prior (root noise mixed in when configured), marks the leaf
    /// expanded, and returns the leaf value in `leaf.to_move`'s perspective.
    fn expand_and_evaluate(&mut self, arena: &mut Vec<Node>, leaf: usize) -> f64 {
        let to_move = arena[leaf].to_move;
        let legal = legal_actions(&arena[leaf].game.pos);
        if legal.is_empty() {
            arena[leaf].expanded = true;
            return 0.0;
        }

        let rep = arena[leaf].game.repetition_count();
        let (logits, value) = match self.evaluator.evaluate(&arena[leaf].game.pos, rep) {
            Ok(out) => out,
            Err(error) => {
                // A failed evaluation is treated as a neutral leaf rather than a
                // panic; Plan 4 self-play surfaces inference errors explicitly.
                eprintln!("AzMcts evaluation failed: {error}");
                arena[leaf].expanded = true;
                return 0.0;
            }
        };

        let mut priors = legal_priors(&logits, &legal, to_move);
        if leaf == 0 && self.config.dirichlet_epsilon > 0.0 {
            self.apply_root_noise(&mut priors);
        }

        for (action, prior) in legal.iter().zip(priors.iter()) {
            let mut child_game = arena[leaf].game.clone();
            let _ = child_game.apply(*action);
            let child = Node::new(child_game, Some(leaf), Some(*action), *prior);
            let child_idx = arena.len();
            arena.push(child);
            arena[leaf].children.push(child_idx);
        }
        arena[leaf].expanded = true;
        value as f64
    }

    /// Mixes symmetric Dirichlet noise into root priors: `p = (1-eps)*p + eps*noise`.
    ///
    /// The noise is built from Gamma(alpha, 1) samples normalized to sum to 1,
    /// which is exactly Dirichlet(alpha) and uses only the stable `Gamma` API
    /// (rand_distr's `Dirichlet` type has a version-sensitive signature).
    fn apply_root_noise(&mut self, priors: &mut [f32]) {
        let gamma = match rand_distr::Gamma::new(self.config.dirichlet_alpha, 1.0) {
            Ok(g) => g,
            Err(_) => return,
        };
        let mut noise: Vec<f32> = (0..priors.len())
            .map(|_| gamma.sample(&mut self.rng) as f32)
            .collect();
        let sum: f32 = noise.iter().sum();
        if sum <= 0.0 {
            return;
        }
        for n in noise.iter_mut() {
            *n /= sum;
        }
        let eps = self.config.dirichlet_epsilon as f32;
        for (p, n) in priors.iter_mut().zip(noise.iter()) {
            *p = (1.0 - eps) * *p + eps * *n;
        }
    }

    /// Returns the PUCT-best child of `parent`, or `None` if it has no children.
    fn best_child(&self, arena: &[Node], parent: usize) -> Option<usize> {
        let parent_node = &arena[parent];
        let sqrt_total = (parent_node.visits.max(1) as f64).sqrt();

        let mut best: Option<usize> = None;
        let mut best_score = f64::NEG_INFINITY;
        for &child_idx in &parent_node.children {
            let child = &arena[child_idx];
            let q_child = if child.visits == 0 {
                0.0
            } else {
                child.value_sum / child.visits as f64
            };
            // Read the child's mean in the PARENT's perspective.
            let q = if child.to_move == parent_node.to_move { q_child } else { -q_child };
            let u = self.config.c_puct * child.prior as f64 * sqrt_total / (1.0 + child.visits as f64);
            let score = q + u;
            if score > best_score {
                best_score = score;
                best = Some(child_idx);
            }
        }
        best
    }
}
```

- [ ] **Step 2: Add search behavior tests**

In the `#[cfg(test)] mod tests` of `mcts.rs`, add (and add the imports they need at the top of the test module: `use kairnz_core::config::RuleConfig; use kairnz_core::game::Game; use crate::OnnxEvaluator; use std::path::PathBuf;`):

```rust
    fn fixture_evaluator() -> OnnxEvaluator {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/random_init.onnx");
        OnnxEvaluator::from_path(&path).expect("fixture loads")
    }

    fn small_config() -> AzMctsConfig {
        AzMctsConfig { simulations: 64, ..AzMctsConfig::default() }
    }

    #[test]
    fn search_returns_visits_over_legal_actions() {
        let game = Game::new_standard(RuleConfig::default());
        let legal = kairnz_core::actions::legal_actions(&game.pos);
        let mut mcts = AzMcts::new(fixture_evaluator(), small_config(), 1);

        let result = mcts.search(&game);
        assert_eq!(result.len(), legal.len(), "one root child per legal action");
        for (action, _visits) in &result {
            assert!(legal.contains(action), "every searched action is legal");
        }
        let total: u32 = result.iter().map(|(_, v)| *v).sum();
        assert!(total > 0, "simulations recorded visits");
    }

    #[test]
    fn search_is_deterministic_without_root_noise() {
        let game = Game::new_standard(RuleConfig::default());
        let mut a = AzMcts::new(fixture_evaluator(), small_config(), 1);
        let mut b = AzMcts::new(fixture_evaluator(), small_config(), 2);
        // dirichlet_epsilon is 0, so the seed is irrelevant: identical results.
        assert_eq!(a.search(&game), b.search(&game), "epsilon 0 search is deterministic");
    }
```

- [ ] **Step 3: Run the tests**

Run: `cargo test -p kairnz-onnx mcts`
Expected: PASS (helper tests plus the two search tests). Output warning-free.

- [ ] **Step 4: Commit**

```bash
git add crates/kairnz-onnx/src/mcts.rs
git commit -m "feat(onnx): add neural PUCT search core"
```

---

### Task 4: `AzMctsPolicy` and the tactical sign-convention guard

**Files:**
- Modify: `crates/kairnz-onnx/src/mcts.rs`
- Modify: `crates/kairnz-onnx/src/lib.rs`

**Interfaces:**
- Consumes: `AzMcts`, `AzMctsConfig`, `kairnz_policy::policy::Policy`.
- Produces: `AzMctsPolicy` implementing `Policy` (`choose` returns the most-visited root action; `name` is `"az-mcts"`), with `AzMctsPolicy::new(evaluator: OnnxEvaluator, config: AzMctsConfig, seed: u64) -> AzMctsPolicy` and `AzMctsPolicy::from_path(path: &Path, config: AzMctsConfig, seed: u64) -> ort::Result<AzMctsPolicy>`.

- [ ] **Step 1: Add the policy and tests**

In `crates/kairnz-onnx/src/mcts.rs`, add these imports (merge at top): `use std::path::Path; use kairnz_policy::policy::Policy;`. Then add, above the test module:

```rust
/// A `Policy` that plays the most-visited move from a neural PUCT search.
pub struct AzMctsPolicy {
    search: AzMcts,
}

impl AzMctsPolicy {
    /// Builds a policy owning `evaluator`.
    pub fn new(evaluator: OnnxEvaluator, config: AzMctsConfig, seed: u64) -> AzMctsPolicy {
        AzMctsPolicy { search: AzMcts::new(evaluator, config, seed) }
    }

    /// Loads a model from `path` and builds a policy.
    pub fn from_path(path: &Path, config: AzMctsConfig, seed: u64) -> ort::Result<AzMctsPolicy> {
        Ok(AzMctsPolicy::new(OnnxEvaluator::from_path(path)?, config, seed))
    }
}

impl Policy for AzMctsPolicy {
    fn choose(&mut self, game: &Game) -> Option<Action> {
        self.search
            .search(game)
            .into_iter()
            .max_by_key(|(_, visits)| *visits)
            .map(|(action, _)| action)
    }

    fn name(&self) -> &str {
        "az-mcts"
    }
}
```

Then add these tests to the test module:

```rust
    use kairnz_core::piece::{Piece, PieceKind, Player};
    use kairnz_core::position::{Position, TurnState};
    use kairnz_core::square::{BitBoard81, NUM_SQUARES};

    fn sq(file: u8, rank: u8) -> kairnz_core::square::Sq {
        kairnz_core::square::Sq::new(file, rank).expect("in bounds")
    }

    fn place(pos: &mut Position, file: u8, rank: u8, piece: Piece) {
        pos.board[sq(file, rank).0 as usize] = Some(piece);
    }

    fn minimal_pos(to_move: Player, ap: u8) -> Position {
        Position {
            board: [None; NUM_SQUARES],
            reserves: [0, 0],
            to_move,
            turn: TurnState {
                ap_remaining: ap,
                capture_locked: BitBoard81::default(),
                keystone_moved: BitBoard81::default(),
                enemy_checked_at_start: BitBoard81::default(),
            },
            config: RuleConfig::default(),
            zobrist: 0,
            ply: 0,
        }
    }

    fn game_from_pos(pos: Position) -> Game {
        let mut game = Game::new_standard(RuleConfig::default());
        game.pos = pos;
        game
    }

    #[test]
    fn policy_chooses_a_legal_action_at_opening() {
        let game = Game::new_standard(RuleConfig::default());
        let legal = kairnz_core::actions::legal_actions(&game.pos);
        let mut policy = AzMctsPolicy::new(fixture_evaluator(), small_config(), 1);
        let action = policy.choose(&game).expect("opening has a move");
        assert!(legal.contains(&action));
    }

    #[test]
    fn policy_name_is_az_mcts() {
        let policy = AzMctsPolicy::new(fixture_evaluator(), small_config(), 0);
        assert_eq!(policy.name(), "az-mcts");
    }

    /// Sign-convention guard: with a winning keystone capture available, the
    /// search must choose it. The capture creates a terminal child whose true
    /// value (+1 for the capturing side) must back up to make that move the most
    /// visited, even though the fixture network is random. A backprop or PUCT
    /// sign error would steer the search away from the win.
    #[test]
    fn policy_prefers_an_immediate_winning_capture() {
        let mut pos = minimal_pos(Player::P1, 2);
        place(&mut pos, 4, 3, Piece::new(Player::P1, PieceKind::Stone, 2));
        place(&mut pos, 4, 4, Piece::new(Player::P2, PieceKind::Keystone, 1));
        place(&mut pos, 0, 0, Piece::new(Player::P1, PieceKind::Keystone, 1));
        place(&mut pos, 0, 8, Piece::new(Player::P1, PieceKind::Stone, 1));
        pos.recompute_zobrist();

        let winning = Action::Move { from: sq(4, 3), to: sq(4, 4) };
        let game = game_from_pos(pos);
        assert!(kairnz_core::actions::legal_actions(&game.pos).contains(&winning));

        // Enough simulations for the terminal win signal to dominate the random net.
        let config = AzMctsConfig { simulations: 256, ..AzMctsConfig::default() };
        let mut policy = AzMctsPolicy::new(fixture_evaluator(), config, 7);
        assert_eq!(policy.choose(&game), Some(winning), "must take the winning capture");
    }
```

- [ ] **Step 2: Add the `AzMctsPolicy` re-export**

In `crates/kairnz-onnx/src/lib.rs`, update the mcts re-export line to:

```rust
pub use mcts::{AzMctsConfig, AzMctsPolicy};
```

- [ ] **Step 3: Run the tests**

Run: `cargo test -p kairnz-onnx mcts`
Expected: PASS (all helper, search, and policy tests, including the winning-capture guard). If `policy_prefers_an_immediate_winning_capture` is flaky, raise its `simulations` (the terminal signal strengthens with more visits); do not weaken the assertion.

- [ ] **Step 4: Run the full crate suite and build**

Run: `cargo test -p kairnz-onnx`
Expected: PASS (seam, policy, and mcts tests), warning-free.

Run: `cargo build --workspace`
Expected: the workspace builds.

- [ ] **Step 5: Commit**

```bash
git add crates/kairnz-onnx/src/mcts.rs crates/kairnz-onnx/src/lib.rs
git commit -m "feat(onnx): add AzMctsPolicy with tactical sign-convention test"
```

---

## Self-Review Notes

- **Spec coverage:** Implements the spec's Milestone 3 (neural MCTS / PUCT) as `AzMctsPolicy`. Self-play data generation (Milestone 4), the PyTorch trainer (Milestone 5), and orchestration (Milestones 6-7) are deferred to later plans.
- **Sign conventions** are the load-bearing risk and are mirrored from the tested `kairnz-policy::mcts`: value stored per node in `to_move`'s perspective, flipped on turn handover in both backprop and PUCT. The winning-capture test is the guard, exactly as it guards the existing UCT search.
- **Terminal correctness:** terminal leaves are scored by `Game::terminal_result`, never the network, so tactics remain exact even with an untrained model. This is what lets the random-fixture mate-in-one test pass.
- **Forward design for Plan 4:** `search` returns raw visit counts (the self-play policy target is the normalized visit distribution; move-selection temperature is a self-play concern layered on top). Root Dirichlet noise is implemented but inert at `epsilon = 0`, so self-play enables exploration purely via config. Inference errors degrade to a neutral leaf with a log here; Plan 4 will surface them in the self-play worker rather than silently continue.
- **Performance note:** this search evaluates one leaf per simulation (single inference). Batched/parallel leaf evaluation for self-play throughput is a Plan 4 concern; tests run on CPU and use small simulation counts to stay fast.
- **Type consistency:** `AzMcts::{new, search}`, `AzMctsPolicy::{new, from_path, choose, name}`, `AzMctsConfig`, `terminal_value`, and `legal_priors` are referenced identically across tasks.
