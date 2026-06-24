# Kairnz Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver a rule-correct Kairnz rules engine in Rust, a Human-vs-Human Tauri desktop app with an SVG board, and a headless benchmarking harness driving random/greedy/UCT policies — Phase 1 of `KAIRNZ_SPEC.md`.

**Architecture:** A pure `kairnz-core` crate is an action-level state machine: the atomic unit is one Action Point's worth of action (Move/Place/Stack), and an in-turn `TurnState` makes the AP budget, the turn-ending check rule, and the four §7 toggles local, testable transitions. `kairnz-policy` (random/greedy/UCT) and `kairnz-bench` (headless CLI) consume core. `src-tauri` wraps core in thin commands; a Svelte 5 + Vite SPA renders the board in SVG with Shogi-style directional pieces.

**Tech Stack:** Rust (workspace, edition 2021), Tauri 2, Svelte 5 + Vite + TypeScript, `serde`/`serde_yaml`/`serde_json`, `rand` (`StdRng`), `proptest`, `clap`. Build orchestration via `Taskfile.yml`; frontend deps via `pnpm`.

## Global Constraints

- Board fixed at 9×9 (81 squares). Each player: 18 Stones + 2 Keystones. (`KAIRNZ_SPEC.md` §2)
- Normal turn = 2 AP. Move=1 AP, Place=1 AP, Stack=2 AP (whole turn). (§5)
- Keystone never stacks, never promotes, never placed from Reserve. (§3)
- Capturing a Stone/stack banks every token to the capturer's Reserve (Spire = 3 tokens). Capturing a Keystone removes it permanently (not banked) and counts toward the win. Anti-runaway rule is core, never a toggle. (§4)
- Turn ends when AP reach 0, when there is no legal action, or **immediately** when an action puts an enemy Keystone in check that was not in check at the start of the turn. (§5, §6)
- Win = capture both enemy Keystones. No legal action on your turn = you lose. Effectively drawless; safeguard via configurable max-ply and N-fold repetition; draws are reported. (§6)
- Four configurable toggles (§7): Spire = Dragon(default)/Queen; first-player first-turn AP (integer, default test values 1 and 2); capture-lock (default off); keystone-single-move (default off).
- No `unwrap()`/`panic!` in library code; all fallible paths return `Result`. Named constants for all values; user-facing display names and rule presets live in `config/*.yaml`. No em dashes in code/comments/docs. Files target < ~300 lines.
- All policy/bench randomness is seeded; same seed → identical output.

## File Structure

```
kairnz/
  Cargo.toml                       # workspace manifest
  Taskfile.yml                     # build/test/run orchestration
  config/
    names.yaml                     # display names (§10)
    presets.yaml                   # default RuleConfig presets
  crates/
    kairnz-core/
      src/
        lib.rs                     # re-exports
        square.rs                  # Sq, File/Rank, BitBoard81
        piece.rs                   # Player, PieceKind, Piece
        config.rs                  # RuleConfig, SpireMode, defaults
        position.rs                # Position, TurnState, setup
        zobrist.rs                 # Zobrist keys + incremental hash
        movement.rs                # geometry: step/slide targets per piece
        actions.rs                 # Action, IllegalAction, legal-action gen
        apply.rs                   # apply_action, capture, check rule, turn advance
        check.rs                   # in-check detection
        outcome.rs                 # GameResult, terminal detection
    kairnz-policy/
      src/
        lib.rs
        policy.rs                  # Policy trait
        random.rs                  # RandomPolicy
        eval.rs                    # heuristic evaluation
        greedy.rs                  # GreedyPolicy
        mcts.rs                    # plain UCT MCTS
    kairnz-bench/
      src/
        main.rs                    # clap CLI
        runner.rs                  # play one game, record GameRecord
        metrics.rs                 # six §8 metrics
        report.rs                  # human + JSON report, multi-config table
        spec.rs                    # YAML run-spec parsing
  src-tauri/
    Cargo.toml
    tauri.conf.json
    src/
      main.rs
      state.rs                     # GameStore (Mutex<HashMap<GameId, Game>>)
      view.rs                      # GameView, ApplyResult DTOs
      commands.rs                  # new_game/get_view/legal_actions/apply_action/undo
  ui/
    package.json                   # pnpm
    vite.config.ts
    src/
      main.ts
      lib/
        api.ts                     # typed Tauri command wrappers
        types.ts                   # GameView/Action TS mirrors
        names.ts                   # loads config names
      components/
        Board.svelte               # SVG board + squares
        Piece.svelte               # oriented directional glyph
        ConfigPanel.svelte         # rule toggles
        Sidebar.svelte             # AP, reserves, status banner
      App.svelte
```

---

## Milestone A — Workspace scaffolding

### Task 1: Cargo workspace + crate skeletons

**Files:**
- Create: `Cargo.toml`, `crates/kairnz-core/Cargo.toml`, `crates/kairnz-core/src/lib.rs`, `crates/kairnz-policy/Cargo.toml`, `crates/kairnz-policy/src/lib.rs`, `crates/kairnz-bench/Cargo.toml`, `crates/kairnz-bench/src/main.rs`, `Taskfile.yml`, `.gitignore`

**Interfaces:**
- Produces: a buildable workspace with three crates.

- [ ] **Step 1: Write workspace manifest**

```toml
# Cargo.toml
[workspace]
resolver = "2"
members = ["crates/kairnz-core", "crates/kairnz-policy", "crates/kairnz-bench", "src-tauri"]

[workspace.dependencies]
serde = { version = "1", features = ["derive"] }
serde_json = "1"
serde_yaml = "0.9"
rand = "0.8"
rand_pcg = "0.3"
```

- [ ] **Step 2: Write crate manifests and stub lib/main**

```toml
# crates/kairnz-core/Cargo.toml
[package]
name = "kairnz-core"
version = "0.1.0"
edition = "2021"
[dependencies]
serde = { workspace = true }
[dev-dependencies]
proptest = "1"
```

```rust
// crates/kairnz-core/src/lib.rs
#[cfg(test)]
mod smoke {
    #[test]
    fn workspace_builds() {
        assert_eq!(2 + 2, 4);
    }
}
```

Repeat analogous `Cargo.toml`/stub for `kairnz-policy` (deps: `kairnz-core`, `rand`, `rand_pcg`) and `kairnz-bench` (deps: `kairnz-core`, `kairnz-policy`, `serde`, `serde_yaml`, `serde_json`, `clap = { version = "4", features = ["derive"] }`). `kairnz-bench/src/main.rs`: `fn main() {}`.

- [ ] **Step 3: Write Taskfile and .gitignore**

```yaml
# Taskfile.yml
version: '3'
tasks:
  build: { cmds: ["cargo build --workspace"] }
  test: { cmds: ["cargo test --workspace"] }
  bench: { cmds: ["cargo run -p kairnz-bench --"] }
  ui: { dir: ui, cmds: ["pnpm install", "pnpm tauri dev"] }
```

`.gitignore`: `/target`, `ui/node_modules`, `ui/dist`, `src-tauri/target`.

- [ ] **Step 4: Verify build and test**

Run: `cargo test --workspace`
Expected: PASS (`workspace_builds` green), zero warnings.

- [ ] **Step 5: Commit**

```bash
git init && git add -A && git commit -m "chore: scaffold kairnz cargo workspace"
```

---

## Milestone B — Core primitives & state

### Task 2: Square, bitboard, player, piece primitives

**Files:**
- Create: `crates/kairnz-core/src/square.rs`, `crates/kairnz-core/src/piece.rs`
- Modify: `crates/kairnz-core/src/lib.rs` (add `pub mod square; pub mod piece;`)

**Interfaces:**
- Produces:
  - `Sq(pub u8)` with `Sq::new(file: u8, rank: u8) -> Option<Sq>`, `fn file(self)->u8`, `fn rank(self)->u8`, `const BOARD_SIZE: u8 = 9`, `const NUM_SQUARES: usize = 81`.
  - `BitBoard81(u128)` with `set/clear/contains/is_empty/iter`.
  - `enum Player { P1, P2 }` with `fn opponent(self)->Player`, `fn index(self)->usize`.
  - `enum PieceKind { Stone, Keystone }`, `struct Piece { owner: Player, kind: PieceKind, height: u8 }`.

- [ ] **Step 1: Write failing tests**

```rust
// in square.rs #[cfg(test)]
#[test]
fn sq_roundtrips_file_rank() {
    let s = Sq::new(3, 1).unwrap();
    assert_eq!((s.file(), s.rank()), (3, 1));
    assert_eq!(s.0, 1 * 9 + 3);
}
#[test]
fn sq_rejects_out_of_range() {
    assert!(Sq::new(9, 0).is_none());
}
#[test]
fn bitboard_set_contains_clear() {
    let mut b = BitBoard81::default();
    let s = Sq::new(7, 2).unwrap();
    b.set(s); assert!(b.contains(s));
    b.clear(s); assert!(!b.contains(s));
}
#[test]
fn player_opponent_is_involutive() {
    assert_eq!(Player::P1.opponent().opponent(), Player::P1);
}
```

- [ ] **Step 2: Run, verify failure** — Run: `cargo test -p kairnz-core square::` Expected: FAIL (unresolved names).

- [ ] **Step 3: Implement** `Sq`, `BitBoard81` (back with `u128`, mask low 81 bits), `Player`, `PieceKind`, `Piece`. All constructors validate ranges and return `Option`/`Result`; no `unwrap` in non-test code.

- [ ] **Step 4: Run, verify pass** — Run: `cargo test -p kairnz-core` Expected: PASS.

- [ ] **Step 5: Commit** — `git commit -am "feat(core): square, bitboard, player, piece primitives"`

### Task 3: RuleConfig

**Files:**
- Create: `crates/kairnz-core/src/config.rs`; Modify `lib.rs`.

**Interfaces:**
- Produces:
  - `enum SpireMode { Dragon, Queen }`
  - `struct RuleConfig { spire: SpireMode, first_turn_ap: u8, capture_lock: bool, keystone_single_move: bool, max_plies: u32, repetition_fold: u8 }`, `#[derive(Serialize, Deserialize, Clone)]`.
  - `impl Default for RuleConfig` with named consts: `DEFAULT_AP = 2`, `DEFAULT_FIRST_TURN_AP = 2`, `DEFAULT_MAX_PLIES = 400`, `DEFAULT_REPETITION_FOLD = 3`, Spire=Dragon, both toggles off.

- [ ] **Step 1: Failing test**

```rust
#[test]
fn default_config_matches_spec_defaults() {
    let c = RuleConfig::default();
    assert!(matches!(c.spire, SpireMode::Dragon));
    assert_eq!(c.first_turn_ap, 2);
    assert!(!c.capture_lock && !c.keystone_single_move);
}
#[test]
fn config_roundtrips_yaml() {
    let c = RuleConfig::default();
    let y = serde_yaml::to_string(&c).unwrap();
    let back: RuleConfig = serde_yaml::from_str(&y).unwrap();
    assert_eq!(back.first_turn_ap, c.first_turn_ap);
}
```
(Add `serde_yaml` to core dev-dependencies.)

- [ ] **Step 2: Run, verify fail.** Run: `cargo test -p kairnz-core config::`
- [ ] **Step 3: Implement `config.rs`.**
- [ ] **Step 4: Run, verify pass.**
- [ ] **Step 5: Commit** — `git commit -am "feat(core): RuleConfig with spec defaults"`

### Task 4: Position type + standard setup

**Files:**
- Create: `crates/kairnz-core/src/position.rs`; Modify `lib.rs`.

**Interfaces:**
- Consumes: `Sq`, `Piece`, `Player`, `RuleConfig`.
- Produces:
  - `struct TurnState { ap_remaining: u8, capture_locked: BitBoard81, keystone_moved: BitBoard81, enemy_checked_at_start: BitBoard81 }`. (`enemy_checked_at_start` is square-anchored: the set of enemy-Keystone squares in check at the start of the current turn. See Task 8 for why this is a `BitBoard81`, not `[bool; 2]`.)
  - `struct Position { board: [Option<Piece>; 81], reserves: [u8; 2], to_move: Player, turn: TurnState, config: RuleConfig, zobrist: u64, ply: u32 }`.
  - `Position::new_standard(config: RuleConfig) -> Position` building the §2 start: ranks 1 & 3 (rank index 0 & 2) full of that player's Stones, Keystones on files 3 & 7 of rank 2 (index 1), mirrored for P2 on the far ranks. Empty reserves. `to_move = P1`, `ap_remaining = config.first_turn_ap`.
  - `fn keystones_of(&self, p: Player) -> impl Iterator<Item=Sq>`, `fn piece_at(&self, s: Sq) -> Option<Piece>`.

- [ ] **Step 1: Failing tests**

```rust
#[test]
fn standard_setup_has_correct_material() {
    let p = Position::new_standard(RuleConfig::default());
    let count = |owner, kind| (0..81).filter(|&i| {
        p.board[i].map_or(false, |pc| pc.owner == owner && pc.kind == kind)
    }).count();
    assert_eq!(count(Player::P1, PieceKind::Stone), 18);
    assert_eq!(count(Player::P1, PieceKind::Keystone), 2);
    assert_eq!(count(Player::P2, PieceKind::Stone), 18);
    assert_eq!(count(Player::P2, PieceKind::Keystone), 2);
}
#[test]
fn keystones_on_files_3_and_7_rank_2() {
    let p = Position::new_standard(RuleConfig::default());
    for f in [3u8, 7] {
        let s = Sq::new(f, 1).unwrap();
        assert!(matches!(p.piece_at(s), Some(pc) if pc.kind == PieceKind::Keystone && pc.owner == Player::P1));
    }
}
#[test]
fn first_turn_ap_respects_config() {
    let mut cfg = RuleConfig::default(); cfg.first_turn_ap = 1;
    assert_eq!(Position::new_standard(cfg).turn.ap_remaining, 1);
}
#[test]
fn reserves_start_empty() {
    let p = Position::new_standard(RuleConfig::default());
    assert_eq!(p.reserves, [0, 0]);
}
```

- [ ] **Step 2: Run, verify fail.**
- [ ] **Step 3: Implement `position.rs`** (P2 mirrored: P2 stones on ranks 7 & 9 (index 6 & 8), keystones rank 8 (index 7) files 3 & 7). `zobrist` set to 0 here; wired in Task 5.
- [ ] **Step 4: Run, verify pass.**
- [ ] **Step 5: Commit** — `git commit -am "feat(core): Position and standard setup"`

### Task 5: Zobrist hashing

**Files:**
- Create: `crates/kairnz-core/src/zobrist.rs`; Modify `position.rs` (compute hash in setup; expose `fn recompute_zobrist`), `lib.rs`.

**Interfaces:**
- Produces: `fn zobrist_full(pos: &Position) -> u64` keyed by (square, owner, kind, height), side-to-move, and per-player reserve counts. Deterministic fixed key table (seeded const generation, not RNG-at-runtime).

- [ ] **Step 1: Failing tests**

```rust
#[test]
fn identical_positions_hash_equal() {
    let a = Position::new_standard(RuleConfig::default());
    let b = Position::new_standard(RuleConfig::default());
    assert_eq!(zobrist_full(&a), zobrist_full(&b));
}
#[test]
fn changing_side_to_move_changes_hash() {
    let mut a = Position::new_standard(RuleConfig::default());
    let h0 = zobrist_full(&a);
    a.to_move = Player::P2;
    assert_ne!(zobrist_full(&a), h0);
}
```

- [ ] **Step 2: Run, verify fail.**
- [ ] **Step 3: Implement** a fixed key table generated by a small splitmix64 over const seeds. `new_standard` sets `pos.zobrist = zobrist_full(&pos)`.
- [ ] **Step 4: Run, verify pass.**
- [ ] **Step 5: Commit** — `git commit -am "feat(core): zobrist hashing for repetition detection"`

---

## Milestone C — Movement & legal action generation

### Task 6: Movement geometry

**Files:**
- Create: `crates/kairnz-core/src/movement.rs`; Modify `lib.rs`.

**Interfaces:**
- Consumes: `Position`, `Sq`, `Piece`, `SpireMode`.
- Produces: `fn move_targets(pos: &Position, from: Sq) -> Vec<Sq>` returning every square the piece at `from` could move to (empty or enemy-occupied; never friendly), honoring height, kind, and the Spire toggle. Pure geometry; does not consult AP or toggles.

- [ ] **Step 1: Failing tests** (one per movement rule, §3)

```rust
// Helpers: build an otherwise-empty Position, drop a piece, assert target set.
#[test]
fn stone_h1_steps_one_orthogonally() { /* place P1 Stone h1 at (4,4); targets == 4 orthogonal neighbors */ }
#[test]
fn pillar_h2_steps_one_in_eight_directions() { /* 8 neighbors */ }
#[test]
fn spire_dragon_slides_orthogonal_and_steps_diagonal() {
    // Dragon at (4,4) on empty board: full orthogonal rays + the 4 adjacent diagonals only.
}
#[test]
fn spire_queen_slides_all_eight() {
    // With SpireMode::Queen: full rays in all 8 directions.
}
#[test]
fn keystone_steps_one_in_eight() {}
#[test]
fn slide_stops_at_first_piece_and_may_capture_enemy() {
    // friendly blocker: target excluded. enemy blocker: target included, ray stops there.
}
#[test]
fn never_moves_onto_friendly() {}
```

- [ ] **Step 2: Run, verify fail.**
- [ ] **Step 3: Implement** direction tables (`ORTHO`, `DIAG`, `ALL8`), a `step` helper and a `slide` helper, and dispatch by `(kind, height, spire)`.
- [ ] **Step 4: Run, verify pass** — Run: `cargo test -p kairnz-core movement::`
- [ ] **Step 5: Commit** — `git commit -am "feat(core): piece movement geometry incl Dragon/Queen"`

### Task 7: Action types + legal action generation

**Files:**
- Create: `crates/kairnz-core/src/actions.rs`; Modify `lib.rs`.

**Interfaces:**
- Consumes: `move_targets`, `Position`, `TurnState`.
- Produces:
  - `enum Action { Move { from: Sq, to: Sq }, Place { to: Sq }, Stack { target: Sq } }` (`Serialize/Deserialize/Clone/Copy/PartialEq`).
  - `enum IllegalAction { NoAp, NotYourPiece, BadGeometry, FriendlyOccupied, EmptyReserve, TargetNotEmpty, NotStackable, CaptureLocked, KeystoneAlreadyMoved, NeedsTwoAp }`.
  - `fn legal_actions(pos: &Position) -> Vec<Action>` honoring AP budget and (capture-lock, keystone-single-move) toggles.
  - `fn action_cost(a: &Action) -> u8` (Move=1, Place=1, Stack=2).

- [ ] **Step 1: Failing tests**

```rust
#[test]
fn move_requires_at_least_one_ap() {
    // ap_remaining = 0 -> legal_actions contains no Move.
}
#[test]
fn place_requires_reserve_and_empty_square() {
    // reserve 0 -> no Place; reserve 1 + a full board minus one empty -> Place only to that empty sq.
}
#[test]
fn stack_only_with_two_ap_and_stackable_stone() {
    // ap_remaining = 1 -> no Stack; ap 2 + a height<3 own Stone -> Stack{target} present; keystone never a Stack target.
}
#[test]
fn capture_locked_piece_cannot_move_again() {
    // toggle on, mark from-square locked -> its Moves excluded; toggle off -> included.
}
#[test]
fn moved_keystone_cannot_move_again_when_toggle_on() {}
```

- [ ] **Step 2: Run, verify fail.**
- [ ] **Step 3: Implement `legal_actions`** iterating own pieces (Move via `move_targets`, filtered by `capture_locked`/`keystone_moved` when toggles on), Place over empty squares when reserve > 0 and `ap_remaining >= 1`, Stack when `ap_remaining >= 2` and reserve > 0 over own non-keystone stones with height < 3.
- [ ] **Step 4: Run, verify pass.**
- [ ] **Step 5: Commit** — `git commit -am "feat(core): action types and legal action generation"`

---

## Milestone D — Apply, capture, check, turn flow

### Task 8: In-check detection

**Files:**
- Create: `crates/kairnz-core/src/check.rs`, `crates/kairnz-core/src/outcome.rs`; Modify `lib.rs`.

**Interfaces:**
- Produces: `fn is_in_check(pos: &Position, keystone_sq: Sq, by: Player) -> bool` — true if any piece owned by `by` has `keystone_sq` in its `move_targets`. `fn checked_enemy_keystone_squares(pos: &Position, attacker: Player) -> BitBoard81` returning the set of squares holding `attacker.opponent()`'s Keystones that are currently in check by `attacker`. (Square-anchored, not a positional `[bool; 2]` slot array: within a mover's turn the defender's Keystone squares are fixed, so anchoring "checked at start" by square is stable under capture, whereas slot indices shift when a Keystone is removed and would mis-fire the turn-ending rule.) `TurnState.enemy_checked_at_start` is therefore a `BitBoard81`, not `[bool; 2]`.
- Also lands the result enums up front so later tasks compile: `enum GameResult { Win(Player), Draw(DrawReason) }`, `enum DrawReason { MaxPlies, Repetition }` in `outcome.rs` (Task 13 adds `terminal_result` and the `Game` wrapper to this module).

- [ ] **Step 1: Failing tests**

```rust
#[test]
fn keystone_threatened_by_adjacent_pillar_is_in_check() {}
#[test]
fn keystone_not_threatened_is_not_in_check() {}
#[test]
fn dragon_slides_to_threaten_keystone_across_empty_rank() {}
```

- [ ] **Step 2–4:** implement and pass.
- [ ] **Step 5: Commit** — `git commit -am "feat(core): keystone in-check detection"`

### Task 9: apply_action — Move, capture, Reserve banking, Keystone removal

**Files:**
- Create: `crates/kairnz-core/src/apply.rs`; Modify `lib.rs`.

**Interfaces:**
- Consumes: `legal_actions`, `is_in_check`, `checked_enemy_keystone_squares`, `zobrist_full`.
- Produces:
  - `struct ActionOutcome { captured: Option<CapturedInfo>, turn_ended: bool, ended_on_check: bool, result: Option<GameResult> }` (GameResult defined in Task 13; forward-declare minimal here or land Task 13 first — sequence Task 13's enum before this).
  - `fn apply_action(pos: &mut Position, a: Action) -> Result<ActionOutcome, IllegalAction>` — validates against `legal_actions`, mutates board/reserves, banks captured Stone tokens (height count) to the mover's reserve, removes a captured Keystone permanently, decrements AP by `action_cost`, updates per-turn bitboards, recomputes zobrist, increments `ply`.

> Sequencing note: implement `GameResult` (Task 13) and the turn-ending check logic (Task 11) interfaces before wiring their use here; this task lands the Move/capture/banking mechanics with `turn_ended` computed only from AP reaching 0, then Task 11 extends it with the check rule.

- [ ] **Step 1: Failing tests**

```rust
#[test]
fn capturing_a_pillar_banks_two_tokens() {
    // P1 Stone captures a P2 height-2 Pillar -> P1 reserve += 2, square now P1 piece.
}
#[test]
fn capturing_a_spire_banks_three_tokens() {}
#[test]
fn capturing_keystone_removes_it_permanently_not_banked() {
    // reserve unchanged; captured keystone count for P2 decreases by 1.
}
#[test]
fn move_decrements_ap_by_one() {}
#[test]
fn token_conservation_holds_after_capture() {
    // sum(board tokens) + sum(reserves) + permanently_removed_keystones == constant.
}
```

- [ ] **Step 2–4:** implement and pass.
- [ ] **Step 5: Commit** — `git commit -am "feat(core): apply Move with capture and reserve banking"`

### Task 10: apply_action — Place and Stack

**Files:**
- Modify: `crates/kairnz-core/src/apply.rs`

**Interfaces:**
- Produces: Place/Stack arms of `apply_action`. Place: reserve -= 1, new P1/P2 Stone height 1 at `to`, AP -= 1. Stack: reserve -= 1, target height += 1, AP -= 2 (turn ends), never onto Keystone.

- [ ] **Step 1: Failing tests**

```rust
#[test]
fn place_consumes_reserve_and_creates_height1_stone() {}
#[test]
fn stack_raises_height_and_costs_whole_turn() {
    // ap 2 -> after Stack ap 0, turn_ended true, height +1.
}
#[test]
fn stack_onto_keystone_is_illegal() {
    assert!(matches!(apply_action(&mut p, Action::Stack { target: ks }), Err(IllegalAction::NotStackable)));
}
```

- [ ] **Step 2–4:** implement and pass.
- [ ] **Step 5: Commit** — `git commit -am "feat(core): apply Place and Stack actions"`

### Task 11: Turn-ending check rule + turn advance (correctness-critical)

**Files:**
- Modify: `crates/kairnz-core/src/apply.rs`; Create `crates/kairnz-core/src/turn.rs` (advance/reset helpers); Modify `lib.rs`.

**Interfaces:**
- Produces:
  - In `apply_action`, after mutation: compute `now = checked_enemy_keystone_squares(pos, mover)`. If `(now & !pos.turn.enemy_checked_at_start)` is non-empty (any enemy-Keystone square in check now that was not in check at turn start) → set `ended_on_check = true`, `turn_ended = true`.
  - `fn advance_turn(pos: &mut Position)` — flips `to_move`, sets `ap_remaining` to 2 (always, after the first turn), clears `capture_locked` and `keystone_moved`, recomputes `enemy_checked_at_start = checked_enemy_keystone_squares(pos, new_mover)`.
  - `apply_action` calls `advance_turn` when `turn_ended`.

- [ ] **Step 1: Failing tests** (the heaviest suite)

```rust
#[test]
fn newly_threatening_a_keystone_by_move_ends_turn_immediately() {
    // ap 2; move a piece so it now attacks an enemy keystone not previously attacked.
    // outcome.ended_on_check == true; to_move flipped; the second AP is forfeit.
}
#[test]
fn newly_threatening_by_place_ends_turn() {}
#[test]
fn newly_threatening_by_stack_ends_turn() {
    // promote a stone so its new movement now attacks an enemy keystone.
}
#[test]
fn capturing_an_already_checked_keystone_does_not_end_on_check_and_may_continue() {
    // keystone in check at turn start; capture it with action 1; ended_on_check == false; ap == 1; turn continues.
}
#[test]
fn threatening_the_second_keystone_ends_turn_even_if_first_already_checked() {
    // enemy_checked_at_start contains keystone A's square (already in check); action newly checks keystone B -> B's square enters the checked set -> turn ends.
}
#[test]
fn leaving_own_keystone_in_check_is_legal() {
    // an action that exposes the mover's own keystone is still applied (no forced resolution).
}
#[test]
fn turn_ends_when_ap_reaches_zero_without_check() {
    // two quiet Moves -> after second, turn_ended true via AP, ended_on_check false.
}
```

- [ ] **Step 2: Run, verify fail.**
- [ ] **Step 3: Implement** the post-action recheck and `advance_turn`; ensure `enemy_checked_at_start` is recomputed for the new mover on every advance and in `new_standard`.
- [ ] **Step 4: Run, verify pass** — Run: `cargo test -p kairnz-core` (whole crate).
- [ ] **Step 5: Commit** — `git commit -am "feat(core): turn-ending check rule and turn advance"`

### Task 12: Toggle integration tests (capture-lock, keystone single-move, first-turn AP, Spire)

**Files:**
- Modify: `crates/kairnz-core/src/apply.rs` (set `capture_locked` on capture; set `keystone_moved` on keystone move). Tests across `actions.rs`/`apply.rs`.

**Interfaces:**
- Produces: capture sets `capture_locked` at the destination square; a Keystone Move sets `keystone_moved` at the destination square. (Generation already filters in Task 7.)

- [ ] **Step 1: Failing paired tests**

```rust
#[test]
fn capture_lock_on_blocks_second_move_of_capturing_piece() {
    // toggle on, ap 2: capture with piece X (now at dest D). legal_actions has no Move{from: D}.
}
#[test]
fn capture_lock_off_allows_chained_capture() {
    // toggle off: same piece may capture again with the second AP.
}
#[test]
fn keystone_single_move_on_blocks_second_keystone_move() {}
#[test]
fn keystone_single_move_off_allows_two_keystone_moves() {}
#[test]
fn first_turn_ap_one_forbids_stack_on_first_turn() {
    // first_turn_ap = 1 -> Stack not in legal_actions on P1's first turn.
}
#[test]
fn spire_queen_toggle_changes_legal_targets() {}
```

- [ ] **Step 2–4:** implement the two `set` hooks and pass.
- [ ] **Step 5: Commit** — `git commit -am "feat(core): wire capture-lock and keystone-single-move toggles"`

### Task 13: Terminal detection — win, loss, draw

**Files:**
- Create: `crates/kairnz-core/src/outcome.rs`; Modify `apply.rs`/`turn.rs`, `lib.rs`. Add a `position_history: Vec<u64>` to `Position` (or a sibling `Game` wrapper — use a `Game { pos, history: Vec<u64> }` in `position.rs` to keep `Position` lean).

**Interfaces:**
- Produces:
  - `enum GameResult { Win(Player), Draw(DrawReason) }`, `enum DrawReason { MaxPlies, Repetition }`.
  - `fn terminal_result(game: &Game) -> Option<GameResult>` checked at each turn start: both of a player's keystones gone → other player Wins; mover has no `legal_actions` → opponent Wins; `ply >= max_plies` → Draw(MaxPlies); zobrist seen `repetition_fold` times → Draw(Repetition).
  - `struct Game { pos: Position, history: Vec<u64> }` with `fn apply(&mut self, a: Action) -> Result<ActionOutcome, IllegalAction>` pushing zobrist after each advance and surfacing `result`.

- [ ] **Step 1: Failing tests**

```rust
#[test]
fn capturing_both_keystones_wins() {}
#[test]
fn no_legal_action_at_turn_start_loses() {
    // construct a position where mover has zero legal actions -> opponent wins.
}
#[test]
fn max_ply_cap_reports_draw() {}
#[test]
fn threefold_repetition_reports_draw() {}
```

- [ ] **Step 2–4:** implement and pass.
- [ ] **Step 5: Commit** — `git commit -am "feat(core): terminal detection win/loss/draw"`

### Task 14: Property tests + perft anchors

**Files:**
- Create: `crates/kairnz-core/tests/properties.rs`, `crates/kairnz-core/tests/perft.rs`.

**Interfaces:**
- Consumes: public core API.

- [ ] **Step 1: Write tests**

```rust
// properties.rs (proptest): from the standard start, apply N random legal actions;
// assert after every step: ap_remaining <= 2; reserves sum + board tokens + removed keystones == 80 (40 per side);
// no friendly square ever doubly occupied.
proptest! {
    #[test]
    fn invariants_hold_under_random_play(seed in any::<u64>()) { /* ... */ }
}
// perft.rs: legal_actions count from the standard opening for first_turn_ap = 1 and = 2 are fixed numbers;
// record them as regression anchors (compute once, assert equality thereafter).
```

- [ ] **Step 2: Run** — Run: `cargo test -p kairnz-core --test properties --test perft` Expected: PASS.
- [ ] **Step 3: Commit** — `git commit -am "test(core): property invariants and perft anchors"`

---

## Milestone E — Policies

### Task 15: Policy trait + Random policy

**Files:**
- Create: `crates/kairnz-policy/src/policy.rs`, `crates/kairnz-policy/src/random.rs`; Modify `lib.rs`.

**Interfaces:**
- Consumes: `kairnz_core::{Game, Action, legal_actions}`.
- Produces:
  - `trait Policy { fn choose(&mut self, game: &Game) -> Option<Action>; fn name(&self) -> &str; }`
  - `struct RandomPolicy { rng: Pcg64 }` with `RandomPolicy::seeded(seed: u64)`.

- [ ] **Step 1: Failing tests**

```rust
#[test]
fn random_policy_is_deterministic_for_a_seed() {
    let g = Game::new_standard(RuleConfig::default());
    let a = RandomPolicy::seeded(42).choose(&g);
    let b = RandomPolicy::seeded(42).choose(&g);
    assert_eq!(a, b);
}
#[test]
fn random_policy_only_returns_legal_actions() {}
```

- [ ] **Step 2–4:** implement and pass.
- [ ] **Step 5: Commit** — `git commit -am "feat(policy): Policy trait and seeded random policy"`

### Task 16: Heuristic evaluation + Greedy policy

**Files:**
- Create: `crates/kairnz-policy/src/eval.rs`, `crates/kairnz-policy/src/greedy.rs`; Modify `lib.rs`.

**Interfaces:**
- Produces:
  - `fn evaluate(pos: &Position, perspective: Player) -> i32` with named-constant weights: `KEYSTONE_VALUE`, `STONE_BASE`, `HEIGHT_BONUS`, `RESERVE_VALUE`, `MOBILITY_WEIGHT`.
  - `struct GreedyPolicy { rng: Pcg64 }` choosing the legal action whose resulting position maximizes `evaluate(_, mover)`, ties broken by seeded RNG.

- [ ] **Step 1: Failing tests**

```rust
#[test]
fn greedy_prefers_a_free_capture_over_a_quiet_move() {}
#[test]
fn greedy_values_keystone_capture_highest() {}
#[test]
fn greedy_is_deterministic_for_a_seed() {}
```

- [ ] **Step 2–4:** implement and pass (clone the game, apply each candidate action, score).
- [ ] **Step 5: Commit** — `git commit -am "feat(policy): heuristic eval and greedy policy"`

### Task 17: Plain UCT MCTS

**Files:**
- Create: `crates/kairnz-policy/src/mcts.rs`; Modify `lib.rs`.

**Interfaces:**
- Produces:
  - `struct MctsPolicy { iterations: u32, exploration: f64, rollout_cap: u32, rng: Pcg64 }`, `MctsPolicy::new(iterations, seed)`.
  - Internal node stores `to_move`, visit/value sums; UCB1 selection; random rollout to terminal or `rollout_cap`; backprop signs reward by each node's `to_move` (handles multi-AP same-mover turns).

- [ ] **Step 1: Failing tests**

```rust
#[test]
fn mcts_is_deterministic_for_a_seed() {}
#[test]
fn mcts_returns_a_legal_action() {}
#[test]
fn mcts_beats_random_over_a_short_match() {
    // play 20 seeded games MCTS(200 iters) vs Random; assert MCTS win-rate > 0.6.
}
```

- [ ] **Step 2–4:** implement and pass (keep `iterations` low in tests).
- [ ] **Step 5: Commit** — `git commit -am "feat(policy): plain UCT MCTS"`

---

## Milestone F — Benchmark harness

### Task 18: Game runner + GameRecord

**Files:**
- Create: `crates/kairnz-bench/src/runner.rs`; Modify `main.rs`.

**Interfaces:**
- Produces:
  - `struct GameRecord { result: GameResult, plies: u32, first_capture_by: Option<Player>, first_keystone_loss_by: Option<Player>, max_stack_height: u8 }`.
  - `fn play_game(config: RuleConfig, p1: &mut dyn Policy, p2: &mut dyn Policy, seed: u64) -> GameRecord` running until terminal, recording the metrics' raw signals.

- [ ] **Step 1: Failing tests**

```rust
#[test]
fn play_game_is_deterministic_for_a_seed() {
    let r1 = play_game(cfg(), &mut RandomPolicy::seeded(1), &mut RandomPolicy::seeded(2), 7);
    let r2 = play_game(cfg(), &mut RandomPolicy::seeded(1), &mut RandomPolicy::seeded(2), 7);
    assert_eq!(r1.plies, r2.plies);
}
#[test]
fn play_game_records_first_capture_side() {}
```

- [ ] **Step 2–4:** implement and pass.
- [ ] **Step 5: Commit** — `git commit -am "feat(bench): game runner and per-game record"`

### Task 19: Metrics aggregation (all six §8 metrics)

**Files:**
- Create: `crates/kairnz-bench/src/metrics.rs`.

**Interfaces:**
- Produces: `struct Metrics { p1_win_rate, p2_win_rate, draw_rate, ply_median, ply_histogram, snowball_rate, comeback_rate, avg_max_stack }` and `fn aggregate(records: &[GameRecord]) -> Metrics`.

- [ ] **Step 1: Failing tests** (each metric from a hand-built `Vec<GameRecord>`)

```rust
#[test]
fn win_rate_by_side_counts_correctly() {}
#[test]
fn snowball_rate_is_first_capture_then_win_fraction() {}
#[test]
fn comeback_rate_is_lost_keystone_first_then_win_fraction() {}
#[test]
fn avg_max_stack_height_averages_records() {}
#[test]
fn ply_median_handles_even_and_odd_counts() {}
```

- [ ] **Step 2–4:** implement and pass.
- [ ] **Step 5: Commit** — `git commit -am "feat(bench): six section-8 balance metrics"`

### Task 20: CLI + YAML run-spec + report

**Files:**
- Create: `crates/kairnz-bench/src/spec.rs`, `crates/kairnz-bench/src/report.rs`; Modify `main.rs`.

**Interfaces:**
- Produces:
  - `struct RunSpec { configs: Vec<NamedConfig>, games_per_config: u32, seed: u64, p1_policy: PolicySpec, p2_policy: PolicySpec }` parsed from YAML; `enum PolicySpec { Random, Greedy, Mcts { iterations: u32 } }`; `fn build_policy(&PolicySpec, seed) -> Box<dyn Policy>`.
  - `fn render_human(&[(String, Metrics)]) -> String` (side-by-side table) and `fn render_json(...) -> String`.
  - `main` (clap): `kairnz-bench --spec run.yaml [--json out.json]`.

- [ ] **Step 1: Failing tests**

```rust
#[test]
fn runspec_parses_from_yaml() {}
#[test]
fn same_spec_and_seed_produce_identical_report() {
    // run twice -> byte-identical human report.
}
#[test]
fn multi_config_report_has_one_column_per_config() {}
```

- [ ] **Step 2–4:** implement and pass; add an example `config/example-run.yaml` comparing Dragon vs Queen and capture-lock on/off.
- [ ] **Step 5: Commit** — `git commit -am "feat(bench): CLI, YAML run-spec, comparison report"`

---

## Milestone G — Tauri + Svelte Human vs Human

### Task 21: Tauri app skeleton + GameStore + view DTOs

**Files:**
- Create: `src-tauri/Cargo.toml`, `src-tauri/tauri.conf.json`, `src-tauri/src/main.rs`, `src-tauri/src/state.rs`, `src-tauri/src/view.rs`, `src-tauri/src/commands.rs`.

**Interfaces:**
- Produces:
  - `struct GameStore(Mutex<HashMap<GameId, Game>>)`; `type GameId = u64`.
  - `struct GameView { board: Vec<Option<PieceView>>, reserves: [u8;2], to_move: Player, ap_remaining: u8, result: Option<GameResult>, names: NameTable }`, `struct PieceView { owner, kind, height }`, `struct ApplyResult { view: GameView, turn_ended_on_check: bool, last_capture: Option<CapturedInfo>, result: Option<GameResult> }`.
  - Commands: `new_game(config) -> GameView`, `get_view(id) -> GameView`, `legal_actions(id, from: Option<Sq>) -> Vec<Action>`, `apply_action(id, action) -> Result<ApplyResult, String>`, `undo(id) -> GameView`.

- [ ] **Step 1: Failing tests** (Rust unit tests on command logic, store-level)

```rust
#[test]
fn new_game_then_get_view_returns_starting_material() {}
#[test]
fn apply_illegal_action_returns_err_without_mutating() {}
#[test]
fn legal_actions_for_selected_square_filters_to_that_piece() {}
```

- [ ] **Step 2–4:** implement and pass (`cargo test -p kairnz-tauri` for the pure logic; Tauri runtime not needed for these).
- [ ] **Step 5: Commit** — `git commit -am "feat(app): tauri commands and game store"`

### Task 22: Svelte SPA scaffold + typed API + names loading

**Files:**
- Create: `ui/package.json`, `ui/vite.config.ts`, `ui/tsconfig.json`, `ui/src/main.ts`, `ui/src/App.svelte`, `ui/src/lib/api.ts`, `ui/src/lib/types.ts`, `ui/src/lib/names.ts`; `config/names.yaml`, `config/presets.yaml`.

**Interfaces:**
- Produces: TS mirrors of `GameView`/`Action`/`RuleConfig`; `api.ts` wrapping each Tauri command via `@tauri-apps/api/core invoke`.

- [ ] **Step 1: Scaffold** Svelte 5 + Vite + TS via pnpm; add `@tauri-apps/api`. `App.svelte` calls `newGame(defaultConfig)` and renders `to_move` + AP as text.
- [ ] **Step 2: Run** — Run: `cd ui && pnpm install && pnpm tauri dev` Expected: window shows starting `to_move: P1`, `AP: 2`.
- [ ] **Step 3: Commit** — `git commit -am "feat(ui): svelte+vite scaffold and typed tauri api"`

### Task 23: SVG board + oriented directional pieces

**Files:**
- Create: `ui/src/components/Board.svelte`, `ui/src/components/Piece.svelte`.

**Interfaces:**
- Consumes: `GameView`.
- Produces: a 9×9 SVG grid; `Piece.svelte` renders an asymmetric upward-tapering stone-pile wedge glyph rotated 180° for P2 (Shogi-style ownership orientation, not color alone); stack height shown as layered tiers (Stone/Pillar/Spire); Keystone a distinct silhouette sharing the orientation.

- [ ] **Step 1: Build** `Board.svelte` mapping 81 squares to SVG `<g>` cells, placing `Piece.svelte` from `view.board`. Define the wedge path once; apply `transform="rotate(180 cx cy)"` for P2.
- [ ] **Step 2: Run** — `pnpm tauri dev`; visually confirm the starting position renders with P1 pieces pointing up, P2 pieces pointing down, keystones distinct.
- [ ] **Step 3: Commit** — `git commit -am "feat(ui): SVG board with shogi-style oriented pieces"`

### Task 24: Interaction, config panel, status, full rule enforcement

**Files:**
- Create: `ui/src/components/ConfigPanel.svelte`, `ui/src/components/Sidebar.svelte`; Modify `Board.svelte`, `App.svelte`.

**Interfaces:**
- Consumes: `legal_actions`, `apply_action`.

- [ ] **Step 1: Build** click-to-select a piece → call `legal_actions(id, sq)` → render legal-target dots; click a target → `apply_action`; on `turn_ended_on_check` show a banner; `Sidebar.svelte` shows AP-remaining and both Reserve counts; `ConfigPanel.svelte` edits `RuleConfig` and starts a new game (Spire mode, first-turn AP, both toggles); Place/Stack via a small action chooser (token from reserve → click empty square for Place; select own stone → Stack). Game-over overlay shows `GameResult`.
- [ ] **Step 2: Run** — `pnpm tauri dev`; manually verify a full HvH game: illegal targets never offered, check auto-end banner fires, capturing both keystones ends the game, toggles change available actions.
- [ ] **Step 3: Commit** — `git commit -am "feat(ui): interaction, config panel, status, enforcement"`

### Task 25: Wire display names from YAML + final Phase 1 verification

**Files:**
- Modify: `ui/src/lib/names.ts`, `src-tauri/src/view.rs` (load `config/names.yaml`).

- [ ] **Step 1:** Load `names.yaml` so all piece/term labels come from config (§10), not literals.
- [ ] **Step 2: Full-suite verification** — Run: `cargo test --workspace` (all green) and `cargo run -p kairnz-bench -- --spec config/example-run.yaml` (report renders, deterministic on rerun). Manually play one HvH game per the §8 mode-1 requirements.
- [ ] **Step 3: Commit** — `git commit -am "feat: wire display names and finalize Phase 1"`

---

## Phases 2 & 3 (planned at high level only — not part of this plan's execution)

- **Phase 2 (AlphaZero self-play, §9):** add a `kairnz-az` crate consuming the same `Game`/`Action` types. State/action encoding reuses the per-action Move/Place/Stack space already in `actions.rs`; the MCTS in `mcts.rs` is generalized to accept a policy/value evaluator (swap random rollouts for network priors + value). Training path (Hybrid Rust+PyTorch/ONNX vs all-Rust tch/candle/burn) decided at Phase 2 start. No change to `kairnz-core`.
- **Phase 3 (Human-vs-AI, AI-vs-AI, §8 modes 2–3):** a `NetworkPolicy` implementing the existing `Policy` trait loads a Phase 2 checkpoint; the Tauri layer adds mode selection and an AI-move command. Reuses `kairnz-policy` and the existing board UI; no engine change.

## Self-Review notes

- **Spec coverage:** §2 setup → Task 4; §3 movement → Task 6; §4 capture/reserve → Tasks 9–10; §5 AP turn/actions → Tasks 7, 9–11; §6 check/win/draw → Tasks 8, 11, 13; §7 toggles → Tasks 3, 7, 12; §8 mode 1 → Tasks 21–25; §8 mode 4 + six metrics → Tasks 18–20; §10 names → Tasks 22, 25. Random/greedy/UCT → Tasks 15–17.
- **Sequencing (resolved):** `GameResult`/`DrawReason` are defined in Task 8's `outcome.rs` so `ActionOutcome` (Task 9) compiles; Task 13 adds `terminal_result` and the `Game` wrapper to the same module.
