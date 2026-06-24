# AlphaZero Plan 1: Encoding and Masking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a `kairnz-encode` crate that converts a Kairnz `Position` into neural-network input planes and converts between game `Action`s and fixed policy-vector indices, with legal-move masking, all in a canonical side-to-move perspective.

**Architecture:** A new pure-Rust workspace crate depending only on `kairnz-core`. It defines the encoding contract that every later AlphaZero plan (self-play, ONNX inference, training data) relies on. No machine-learning dependencies are introduced in this plan. The encoding is perspective-canonical: the board is always oriented so the side to move is "us," achieved by flipping ranks when P2 is to move.

**Tech Stack:** Rust, `kairnz-core` (game types), `cargo` workspace.

## Global Constraints

- Encoding is the single source of truth shared by all later plans. Plane layout and action-index map must not be duplicated elsewhere.
- `NUM_PLANES = 14`, `POLICY_SIZE = 6723` (Move `from*81+to` = 0..6560, Place = 6561..6641, Stack = 6642..6722).
- Tensor layout is channel-major: value for channel `c` at square index `s` lives at `c * 81 + s`, where `s = rank * 9 + file` in canonical coordinates.
- Canonical perspective: orient from `pos.to_move`. P1 is identity; P2 flips rank to `8 - rank`. The transform is its own inverse (an involution).
- Named constants for all magic values. Every public function gets a doc comment.
- No em dashes anywhere in code or comments.
- Keep each source file focused and well under 300 lines.
- Match existing `kairnz-core` style: `#[cfg(test)] mod tests` inline for unit tests; a `tests/` file for cross-module property tests.

---

## File Structure

- Create: `crates/kairnz-encode/Cargo.toml` — crate manifest, depends on `kairnz-core`.
- Create: `crates/kairnz-encode/src/lib.rs` — module declarations, public re-exports, shared constants.
- Create: `crates/kairnz-encode/src/canonical.rs` — `canonical_sq` perspective transform.
- Create: `crates/kairnz-encode/src/action_index.rs` — `action_to_index` / `index_to_action`.
- Create: `crates/kairnz-encode/src/mask.rs` — `legal_mask`.
- Create: `crates/kairnz-encode/src/planes.rs` — `encode_planes`.
- Create: `crates/kairnz-encode/tests/properties.rs` — perspective-invariance property test.
- Modify: `Cargo.toml:3` — add `crates/kairnz-encode` to workspace members.

---

### Task 1: Scaffold the `kairnz-encode` crate

**Files:**
- Create: `crates/kairnz-encode/Cargo.toml`
- Create: `crates/kairnz-encode/src/lib.rs`
- Modify: `Cargo.toml:3`

**Interfaces:**
- Consumes: nothing (first task).
- Produces: public constants `NUM_PLANES: usize = 14`, `POLICY_SIZE: usize = 6723`, and crate-internal `MOVE_BASE`, `PLACE_BASE`, `STACK_BASE` for later tasks.

- [ ] **Step 1: Add the crate to the workspace**

Modify `Cargo.toml` line 3 from:

```toml
members = ["crates/kairnz-core", "crates/kairnz-policy", "crates/kairnz-bench", "src-tauri"]
```

to:

```toml
members = ["crates/kairnz-core", "crates/kairnz-encode", "crates/kairnz-policy", "crates/kairnz-bench", "src-tauri"]
```

- [ ] **Step 2: Create the crate manifest**

Create `crates/kairnz-encode/Cargo.toml`:

```toml
[package]
name = "kairnz-encode"
version = "0.1.0"
edition = "2021"

[dependencies]
kairnz-core = { path = "../kairnz-core" }
```

- [ ] **Step 3: Write the constants and a test asserting the index layout is internally consistent**

Create `crates/kairnz-encode/src/lib.rs`:

```rust
//! Neural-network encoding for Kairnz: position planes and action indices.
//!
//! This crate is the single source of truth for the AlphaZero encoding contract.
//! All values are produced in a canonical perspective oriented to the side to move.

pub mod canonical;

/// Number of 9x9 input planes produced for each position.
pub const NUM_PLANES: usize = 14;

/// Number of squares on the board (9x9).
pub const BOARD_CELLS: usize = kairnz_core::square::NUM_SQUARES;

/// Flat index of the first Move entry in the policy vector.
pub(crate) const MOVE_BASE: usize = 0;
/// Flat index of the first Place entry in the policy vector.
pub(crate) const PLACE_BASE: usize = BOARD_CELLS * BOARD_CELLS;
/// Flat index of the first Stack entry in the policy vector.
pub(crate) const STACK_BASE: usize = PLACE_BASE + BOARD_CELLS;

/// Total length of the policy vector (Move, then Place, then Stack regions).
pub const POLICY_SIZE: usize = STACK_BASE + BOARD_CELLS;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn policy_layout_constants_are_consistent() {
        assert_eq!(BOARD_CELLS, 81);
        assert_eq!(MOVE_BASE, 0);
        assert_eq!(PLACE_BASE, 6561);
        assert_eq!(STACK_BASE, 6642);
        assert_eq!(POLICY_SIZE, 6723);
    }
}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cargo test -p kairnz-encode`
Expected: PASS (1 test in `lib`). The crate compiles and is part of the workspace.

Note: `src/lib.rs` declares `pub mod canonical;` so it will not compile until Task 2 creates that file. Create an empty `crates/kairnz-encode/src/canonical.rs` first if you want this task to build standalone, or proceed directly to Task 2 and run this test there. Recommended: create `canonical.rs` with a single placeholder line `// implemented in Task 2` so Task 1 builds; Task 2 replaces it.

- [ ] **Step 5: Commit**

```bash
git add Cargo.toml crates/kairnz-encode/Cargo.toml crates/kairnz-encode/src/lib.rs crates/kairnz-encode/src/canonical.rs
git commit -m "feat(encode): scaffold kairnz-encode crate with policy-layout constants"
```

---

### Task 2: Canonical perspective transform

**Files:**
- Create (replace placeholder): `crates/kairnz-encode/src/canonical.rs`

**Interfaces:**
- Consumes: `kairnz_core::square::{Sq, BOARD_SIZE}`, `kairnz_core::piece::Player`.
- Produces: `pub fn canonical_sq(s: Sq, me: Player) -> Sq`. Identity for P1; flips rank to `BOARD_SIZE - 1 - rank` for P2. Used by `action_index` and `planes`.

- [ ] **Step 1: Write the failing tests**

Replace `crates/kairnz-encode/src/canonical.rs` with:

```rust
//! Canonical board orientation relative to the side to move.

use kairnz_core::piece::Player;
use kairnz_core::square::{Sq, BOARD_SIZE};

/// Maps a board square into the canonical perspective of `me`.
///
/// P1 is the identity. P2 flips the rank to `BOARD_SIZE - 1 - rank` so that the
/// side to move always sees its home rows at the bottom. The transform is its
/// own inverse, so applying it twice returns the original square.
pub fn canonical_sq(s: Sq, me: Player) -> Sq {
    match me {
        Player::P1 => s,
        Player::P2 => {
            let flipped_rank = (BOARD_SIZE - 1) - s.rank();
            Sq::new(s.file(), flipped_rank).expect("rank flip stays in bounds")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kairnz_core::square::NUM_SQUARES;

    #[test]
    fn identity_for_p1() {
        for i in 0..NUM_SQUARES as u8 {
            assert_eq!(canonical_sq(Sq(i), Player::P1), Sq(i));
        }
    }

    #[test]
    fn involution_for_p2() {
        for i in 0..NUM_SQUARES as u8 {
            let s = Sq(i);
            assert_eq!(canonical_sq(canonical_sq(s, Player::P2), Player::P2), s);
        }
    }

    #[test]
    fn flips_rank_keeps_file_for_p2() {
        let s = Sq::new(2, 1).unwrap();
        let c = canonical_sq(s, Player::P2);
        assert_eq!((c.file(), c.rank()), (2, 7));
    }
}
```

- [ ] **Step 2: Run the tests to verify they fail then pass**

Run: `cargo test -p kairnz-encode canonical`
Expected: PASS (3 tests). If `BOARD_SIZE` is not re-exported, confirm the import path `kairnz_core::square::BOARD_SIZE` resolves (it is `pub const` in `square.rs`).

- [ ] **Step 3: Commit**

```bash
git add crates/kairnz-encode/src/canonical.rs
git commit -m "feat(encode): add canonical_sq perspective transform"
```

---

### Task 3: Action to index and back

**Files:**
- Create: `crates/kairnz-encode/src/action_index.rs`
- Modify: `crates/kairnz-encode/src/lib.rs` (add `pub mod action_index;` and re-exports)

**Interfaces:**
- Consumes: `canonical_sq`, the `MOVE_BASE`/`PLACE_BASE`/`STACK_BASE`/`POLICY_SIZE` constants, `kairnz_core::actions::Action`, `kairnz_core::square::Sq`, `kairnz_core::piece::Player`.
- Produces:
  - `pub fn action_to_index(a: &Action, me: Player) -> usize`
  - `pub fn index_to_action(index: usize, me: Player) -> Option<Action>` (None if `index >= POLICY_SIZE`)
  These form a bijection over `0..POLICY_SIZE`, used by `mask` and all later plans.

- [ ] **Step 1: Write the failing tests and implementation**

Create `crates/kairnz-encode/src/action_index.rs`:

```rust
//! Bijection between game actions and fixed policy-vector indices.

use kairnz_core::actions::Action;
use kairnz_core::piece::Player;
use kairnz_core::square::{Sq, NUM_SQUARES};

use crate::canonical::canonical_sq;
use crate::{MOVE_BASE, PLACE_BASE, POLICY_SIZE, STACK_BASE};

/// Maps a legal-or-illegal action to its flat policy-vector index.
///
/// Squares are taken into the canonical perspective of `me` first, so the index
/// space is perspective-invariant.
pub fn action_to_index(a: &Action, me: Player) -> usize {
    match a {
        Action::Move { from, to } => {
            let f = canonical_sq(*from, me).0 as usize;
            let t = canonical_sq(*to, me).0 as usize;
            MOVE_BASE + f * NUM_SQUARES + t
        }
        Action::Place { to } => PLACE_BASE + canonical_sq(*to, me).0 as usize,
        Action::Stack { target } => STACK_BASE + canonical_sq(*target, me).0 as usize,
    }
}

/// Inverse of [`action_to_index`]; returns `None` for out-of-range indices.
pub fn index_to_action(index: usize, me: Player) -> Option<Action> {
    if index >= POLICY_SIZE {
        return None;
    }
    if index >= STACK_BASE {
        let canonical = Sq((index - STACK_BASE) as u8);
        Some(Action::Stack { target: canonical_sq(canonical, me) })
    } else if index >= PLACE_BASE {
        let canonical = Sq((index - PLACE_BASE) as u8);
        Some(Action::Place { to: canonical_sq(canonical, me) })
    } else {
        let rel = index - MOVE_BASE;
        let from_canonical = Sq((rel / NUM_SQUARES) as u8);
        let to_canonical = Sq((rel % NUM_SQUARES) as u8);
        Some(Action::Move {
            from: canonical_sq(from_canonical, me),
            to: canonical_sq(to_canonical, me),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn place_and_stack_land_in_expected_ranges() {
        let me = Player::P1;
        assert_eq!(action_to_index(&Action::Place { to: Sq(0) }, me), 6561);
        assert_eq!(action_to_index(&Action::Stack { target: Sq(0) }, me), 6642);
        assert_eq!(index_to_action(6561, me), Some(Action::Place { to: Sq(0) }));
        assert_eq!(index_to_action(6642, me), Some(Action::Stack { target: Sq(0) }));
    }

    #[test]
    fn out_of_range_index_is_none() {
        assert_eq!(index_to_action(POLICY_SIZE, Player::P1), None);
    }

    #[test]
    fn full_bijection_both_perspectives() {
        for me in [Player::P1, Player::P2] {
            for idx in 0..POLICY_SIZE {
                let action = index_to_action(idx, me).expect("index in range decodes");
                assert_eq!(action_to_index(&action, me), idx, "idx {idx} for {me:?}");
            }
        }
    }
}
```

- [ ] **Step 2: Wire the module into `lib.rs`**

In `crates/kairnz-encode/src/lib.rs`, add after `pub mod canonical;`:

```rust
pub mod action_index;

pub use action_index::{action_to_index, index_to_action};
pub use canonical::canonical_sq;
```

- [ ] **Step 3: Run the tests**

Run: `cargo test -p kairnz-encode action_index`
Expected: PASS (3 tests, including the exhaustive `full_bijection_both_perspectives` over all 6723 indices for both perspectives).

- [ ] **Step 4: Commit**

```bash
git add crates/kairnz-encode/src/action_index.rs crates/kairnz-encode/src/lib.rs
git commit -m "feat(encode): add action/index bijection with canonical perspective"
```

---

### Task 4: Legal-move mask

**Files:**
- Create: `crates/kairnz-encode/src/mask.rs`
- Modify: `crates/kairnz-encode/src/lib.rs` (add `pub mod mask;` and re-export)

**Interfaces:**
- Consumes: `action_to_index`, `POLICY_SIZE`, `kairnz_core::actions::legal_actions`, `kairnz_core::position::Position`.
- Produces: `pub fn legal_mask(pos: &Position) -> Vec<bool>` of length `POLICY_SIZE`, `true` exactly at indices of currently legal actions.

- [ ] **Step 1: Write the implementation and tests**

Create `crates/kairnz-encode/src/mask.rs`:

```rust
//! Legal-action mask over the policy vector.

use kairnz_core::actions::legal_actions;
use kairnz_core::position::Position;

use crate::action_index::action_to_index;
use crate::POLICY_SIZE;

/// Builds a boolean mask of length `POLICY_SIZE` marking every legal action.
///
/// Indices are computed in the canonical perspective of `pos.to_move`, matching
/// the policy head's output orientation.
pub fn legal_mask(pos: &Position) -> Vec<bool> {
    let mut mask = vec![false; POLICY_SIZE];
    for action in legal_actions(pos) {
        mask[action_to_index(&action, pos.to_move)] = true;
    }
    mask
}

#[cfg(test)]
mod tests {
    use super::*;
    use kairnz_core::actions::legal_actions;
    use kairnz_core::config::RuleConfig;
    use kairnz_core::position::Position;

    use crate::action_index::action_to_index;

    #[test]
    fn mask_has_policy_size_length() {
        let pos = Position::new_standard(RuleConfig::default());
        assert_eq!(legal_mask(&pos).len(), POLICY_SIZE);
    }

    #[test]
    fn mask_matches_legal_actions_without_collisions() {
        let pos = Position::new_standard(RuleConfig::default());
        let mask = legal_mask(&pos);
        let actions = legal_actions(&pos);

        // Every distinct legal action occupies a distinct index, so the count of
        // set bits equals the number of legal actions.
        let set_count = mask.iter().filter(|bit| **bit).count();
        assert_eq!(set_count, actions.len());

        for action in &actions {
            assert!(mask[action_to_index(action, pos.to_move)]);
        }
    }
}
```

- [ ] **Step 2: Wire the module into `lib.rs`**

In `crates/kairnz-encode/src/lib.rs`, add:

```rust
pub mod mask;

pub use mask::legal_mask;
```

- [ ] **Step 3: Run the tests**

Run: `cargo test -p kairnz-encode mask`
Expected: PASS (2 tests). The collision-free assertion confirms the action-index map is injective over real legal actions.

- [ ] **Step 4: Commit**

```bash
git add crates/kairnz-encode/src/mask.rs crates/kairnz-encode/src/lib.rs
git commit -m "feat(encode): add legal-move mask over the policy vector"
```

---

### Task 5: Position to input planes

**Files:**
- Create: `crates/kairnz-encode/src/planes.rs`
- Modify: `crates/kairnz-encode/src/lib.rs` (add `pub mod planes;` and re-export)

**Interfaces:**
- Consumes: `canonical_sq`, `NUM_PLANES`, `BOARD_CELLS`, `kairnz_core::position::Position`, `kairnz_core::piece::{PieceKind, Player}`, `kairnz_core::square::Sq`.
- Produces: `pub fn encode_planes(pos: &Position, repetition_count: u8) -> Vec<f32>` of length `NUM_PLANES * BOARD_CELLS = 1134`, channel-major, canonical perspective. `repetition_count` is supplied by the caller (self-play computes it from history) because a `Position` alone has no history.

Channel layout (all canonical):
- 0,1,2: my Stone height 1,2,3
- 3: my Keystone
- 4,5,6: opponent Stone height 1,2,3
- 7: opponent Keystone
- 8: AP remaining, normalized by 2.0
- 9: my reserves, normalized by 18.0
- 10: opponent reserves, normalized by 18.0
- 11: capture-locked squares
- 12: keystone-moved squares
- 13: repetition count, normalized by 3.0

- [ ] **Step 1: Write the implementation and tests**

Create `crates/kairnz-encode/src/planes.rs`:

```rust
//! Encodes a position into channel-major neural-network input planes.

use kairnz_core::piece::{PieceKind, Player};
use kairnz_core::position::Position;
use kairnz_core::square::Sq;

use crate::canonical::canonical_sq;
use crate::{BOARD_CELLS, NUM_PLANES};

/// Normalizer for the action-points plane (max AP per turn).
const AP_NORM: f32 = 2.0;
/// Normalizer for reserve-count planes (an upper bound on a player's stones).
const RESERVE_NORM: f32 = 18.0;
/// Normalizer for the repetition plane (default repetition-fold threshold).
const REPETITION_NORM: f32 = 3.0;

/// Channel offset for the first "my Stone" plane.
const CH_MY_STONE: usize = 0;
/// Channel for the "my Keystone" plane.
const CH_MY_KEYSTONE: usize = 3;
/// Channel offset for the first "opponent Stone" plane.
const CH_OPP_STONE: usize = 4;
/// Channel for the "opponent Keystone" plane.
const CH_OPP_KEYSTONE: usize = 7;
/// Channel for the action-points plane.
const CH_AP: usize = 8;
/// Channel for the "my reserves" plane.
const CH_MY_RESERVE: usize = 9;
/// Channel for the "opponent reserves" plane.
const CH_OPP_RESERVE: usize = 10;
/// Channel for the capture-locked plane.
const CH_CAPTURE_LOCKED: usize = 11;
/// Channel for the keystone-moved plane.
const CH_KEYSTONE_MOVED: usize = 12;
/// Channel for the repetition-count plane.
const CH_REPETITION: usize = 13;

/// Encodes `pos` into `NUM_PLANES * BOARD_CELLS` floats in channel-major order.
///
/// Planes are oriented to the canonical perspective of `pos.to_move`. The caller
/// supplies `repetition_count` (how many times the current position has occurred)
/// because a `Position` alone carries no history.
pub fn encode_planes(pos: &Position, repetition_count: u8) -> Vec<f32> {
    let me = pos.to_move;
    let opp = me.opponent();
    let mut planes = vec![0.0f32; NUM_PLANES * BOARD_CELLS];

    // Piece planes.
    for i in 0..BOARD_CELLS {
        let piece = match pos.board[i] {
            Some(pc) => pc,
            None => continue,
        };
        let cs = canonical_sq(Sq(i as u8), me);
        let is_me = piece.owner == me;
        let channel = match (is_me, piece.kind) {
            (true, PieceKind::Stone) => CH_MY_STONE + (piece.height as usize - 1),
            (true, PieceKind::Keystone) => CH_MY_KEYSTONE,
            (false, PieceKind::Stone) => CH_OPP_STONE + (piece.height as usize - 1),
            (false, PieceKind::Keystone) => CH_OPP_KEYSTONE,
        };
        planes[channel * BOARD_CELLS + cs.0 as usize] = 1.0;
    }

    // Scalar planes broadcast across every cell.
    let ap_val = pos.turn.ap_remaining as f32 / AP_NORM;
    let my_reserve = pos.reserves[me.index()] as f32 / RESERVE_NORM;
    let opp_reserve = pos.reserves[opp.index()] as f32 / RESERVE_NORM;
    let rep_val = repetition_count as f32 / REPETITION_NORM;
    for cell in 0..BOARD_CELLS {
        planes[CH_AP * BOARD_CELLS + cell] = ap_val;
        planes[CH_MY_RESERVE * BOARD_CELLS + cell] = my_reserve;
        planes[CH_OPP_RESERVE * BOARD_CELLS + cell] = opp_reserve;
        planes[CH_REPETITION * BOARD_CELLS + cell] = rep_val;
    }

    // Turn-state bitboards, transformed to canonical squares.
    for s in pos.turn.capture_locked.iter() {
        let cs = canonical_sq(s, me);
        planes[CH_CAPTURE_LOCKED * BOARD_CELLS + cs.0 as usize] = 1.0;
    }
    for s in pos.turn.keystone_moved.iter() {
        let cs = canonical_sq(s, me);
        planes[CH_KEYSTONE_MOVED * BOARD_CELLS + cs.0 as usize] = 1.0;
    }

    planes
}

#[cfg(test)]
mod tests {
    use super::*;
    use kairnz_core::config::RuleConfig;

    fn plane_sum(planes: &[f32], channel: usize) -> f32 {
        planes[channel * BOARD_CELLS..(channel + 1) * BOARD_CELLS].iter().sum()
    }

    #[test]
    fn output_has_expected_length() {
        let pos = Position::new_standard(RuleConfig::default());
        assert_eq!(encode_planes(&pos, 0).len(), NUM_PLANES * BOARD_CELLS);
    }

    #[test]
    fn opening_piece_plane_counts() {
        let pos = Position::new_standard(RuleConfig::default());
        let planes = encode_planes(&pos, 0);
        assert_eq!(plane_sum(&planes, CH_MY_STONE), 18.0);
        assert_eq!(plane_sum(&planes, CH_MY_KEYSTONE), 2.0);
        assert_eq!(plane_sum(&planes, CH_OPP_STONE), 18.0);
        assert_eq!(plane_sum(&planes, CH_OPP_KEYSTONE), 2.0);
        // No height-2 or height-3 stones exist at the opening.
        assert_eq!(plane_sum(&planes, CH_MY_STONE + 1), 0.0);
        assert_eq!(plane_sum(&planes, CH_MY_STONE + 2), 0.0);
    }

    #[test]
    fn ap_plane_is_normalized_and_uniform() {
        let pos = Position::new_standard(RuleConfig::default());
        let planes = encode_planes(&pos, 0);
        // Default first-turn AP is 2, normalized to 1.0 across all cells.
        assert!(planes[CH_AP * BOARD_CELLS..(CH_AP + 1) * BOARD_CELLS]
            .iter()
            .all(|v| (*v - 1.0).abs() < 1e-6));
    }

    #[test]
    fn my_keystones_sit_on_canonical_home_squares() {
        let pos = Position::new_standard(RuleConfig::default());
        let planes = encode_planes(&pos, 0);
        // P1 keystones are at rank 1, files 2 and 6; canonical is identity for P1.
        for file in [2usize, 6] {
            let cell = 1 * 9 + file;
            assert_eq!(planes[CH_MY_KEYSTONE * BOARD_CELLS + cell], 1.0);
        }
    }
}
```

- [ ] **Step 2: Wire the module into `lib.rs`**

In `crates/kairnz-encode/src/lib.rs`, add:

```rust
pub mod planes;

pub use planes::encode_planes;
```

- [ ] **Step 3: Run the tests**

Run: `cargo test -p kairnz-encode planes`
Expected: PASS (4 tests).

- [ ] **Step 4: Commit**

```bash
git add crates/kairnz-encode/src/planes.rs crates/kairnz-encode/src/lib.rs
git commit -m "feat(encode): add position-to-planes encoder"
```

---

### Task 6: Perspective-invariance property test

**Files:**
- Create: `crates/kairnz-encode/tests/properties.rs`

**Interfaces:**
- Consumes: the crate's public API (`encode_planes`).
- Produces: a cross-module integration test asserting the canonical encoding is perspective-invariant on the symmetric opening.

- [ ] **Step 1: Write the property test**

Create `crates/kairnz-encode/tests/properties.rs`:

```rust
//! Cross-module properties of the encoding.

use kairnz_core::config::RuleConfig;
use kairnz_core::piece::Player;
use kairnz_core::position::Position;
use kairnz_encode::{encode_planes, BOARD_CELLS};

/// The standard opening is symmetric under a rank flip plus a side swap, so its
/// canonical piece planes must be identical whether P1 or P2 is to move.
#[test]
fn opening_is_perspective_invariant_on_piece_planes() {
    let p1_to_move = Position::new_standard(RuleConfig::default());
    let mut p2_to_move = p1_to_move.clone();
    p2_to_move.to_move = Player::P2;

    let e1 = encode_planes(&p1_to_move, 0);
    let e2 = encode_planes(&p2_to_move, 0);

    // Channels 0..8 are the eight piece planes (my/opp x Stone h1,h2,h3 + Keystone).
    for channel in 0..8 {
        for cell in 0..BOARD_CELLS {
            let i = channel * BOARD_CELLS + cell;
            assert!(
                (e1[i] - e2[i]).abs() < 1e-6,
                "channel {channel} cell {cell} differs: {} vs {}",
                e1[i],
                e2[i]
            );
        }
    }
}
```

- [ ] **Step 2: Run the test**

Run: `cargo test -p kairnz-encode --test properties`
Expected: PASS (1 test).

- [ ] **Step 3: Run the full crate test suite and the workspace build**

Run: `cargo test -p kairnz-encode`
Expected: PASS (all unit and integration tests).

Run: `cargo build --workspace`
Expected: the whole workspace builds with the new crate.

- [ ] **Step 4: Commit**

```bash
git add crates/kairnz-encode/tests/properties.rs
git commit -m "test(encode): add perspective-invariance property test"
```

---

## Self-Review Notes

- **Spec coverage:** This plan implements the spec's "Encoding (`kairnz-encode`)" component and Milestone 1. State-to-planes, action-to-index, legal masking, and canonical perspective are all covered. ONNX, MCTS, self-play, and training are explicitly deferred to later plans.
- **Repetition plane:** The spec lists a repetition plane; `Position` has no history, so `encode_planes` takes `repetition_count` as a caller-supplied argument. This keeps encoding pure and is the documented contract for the self-play plan.
- **Type consistency:** `canonical_sq`, `action_to_index`, `index_to_action`, `legal_mask`, and `encode_planes` signatures are referenced identically across tasks. Constants `NUM_PLANES`, `BOARD_CELLS`, `POLICY_SIZE`, `MOVE_BASE`, `PLACE_BASE`, `STACK_BASE` are defined once in `lib.rs` and reused.
- **Open follow-ups for later plans:** the exact 14-plane layout is now fixed; the PyTorch trainer reads channel count from data shape, so it is not duplicated. The mid-turn no-legal-action edge case is a self-play concern, not an encoding concern, and is deferred.
