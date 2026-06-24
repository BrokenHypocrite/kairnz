# Kairnz Phase 1 — Architecture & Design

Status: approved (design); pending implementation plan
Date: 2026-06-23
Source of truth for rules: `KAIRNZ_SPEC.md`

## Scope

Phase 1 only, per `KAIRNZ_SPEC.md` §12:

1. The Rust rules engine enforcing the complete rule set (§2–§7).
2. Human vs Human play with full rule enforcement and a graphical SVG board (§8 mode 1).
3. The headless Training & Benchmarking harness (§8 mode 4) using non-learned policies: random, greedy, and plain UCT MCTS.

No neural network and no GPU in Phase 1. Phases 2 (AlphaZero self-play) and 3 (AI-backed interactive modes) are sketched only at the level needed to ensure the Phase 1 architecture leaves room for them.

Stack is fixed by the spec (§13): Rust + Tauri cross-platform desktop. Frontend within the Tauri webview: **Svelte 5 + Vite + TypeScript as an SPA** (not SvelteKit — its routing/SSR is dead weight for a desktop webview; Tauri's recommended path is a Vite SPA). Board rendering: **SVG**.

## Key decisions (with rationale)

| Decision | Choice | Why |
|---|---|---|
| Engine atomic unit | **Action-level state machine** (one AP action) | The AP turn, the turn-ending check rule, and the four toggles all become local, unit-testable transitions. Plies = actions, which is what §8 measures. Matches the per-action Move/Place/Stack space Phase 2's encoder needs. |
| Board rendering | SVG | Crisp scalable 9×9, easy per-square hit targets, clean legal-move dots and stack-height glyphs. |
| Frontend | Svelte 5 + Vite + TS, SPA | UI is thin (Rust is the single source of truth); reactive stores map onto Rust-pushed state with minimal boilerplate. |
| State representation | `[Option<Piece>; 81]` array board | Correctness-first and readable. Optimize (bitboards) only if MCTS proves too slow. |
| Draw safeguard | Configurable max-ply + N-fold repetition via Zobrist | §6 requires an anti-unending-game safeguard; draws reported when they occur. |

## Section 1 — Crate / workspace structure

```
kairnz/
  Cargo.toml                 # workspace
  crates/
    kairnz-core/              # rules, state, move-gen, RuleConfig, Zobrist. NO I/O, NO Tauri, NO rng dependency.
    kairnz-policy/            # random, greedy, UCT MCTS. Depends on core + rng.
    kairnz-bench/             # headless CLI binary. Depends on core + policy. Emits metrics/report.
  src-tauri/                 # Tauri app crate: thin commands wrapping core, holds Game state.
  ui/                        # Svelte 5 + Vite + TS SPA.
  config/                    # YAML: display names (§10) + default rule presets.
  docs/superpowers/specs/    # design docs
```

`kairnz-core` stays free of randomness, threads, and policy logic so it is trivially testable and reusable by every consumer (bench and Tauri now, AlphaZero in Phase 2). Policies live in a separate crate because they are consumers of the engine, shared by bench (Phase 1) and the Tauri app (Phase 3).

## Section 2 — Engine core (`kairnz-core`)

### Core types

```
Player = P1 | P2

Piece { owner: Player, kind: Stone | Keystone, height: 1..=3 }   // Keystone always height 1, never stacks

Position {
  board: [Option<Piece>; 81],
  reserves: [u8; 2],
  to_move: Player,
  turn: TurnState,
  config: RuleConfig,
  zobrist: u64,
}

TurnState {
  ap_remaining: u8,                    // 2 normally; first-turn-AP for P1's first turn
  capture_locked: BitBoard81,          // squares whose piece captured this turn (capture-lock toggle)
  keystone_moved: BitBoard81,          // keystone squares that already moved (single-move toggle)
  enemy_checked_at_start: [bool; 2],   // which of the mover's two target Keystones were in check when this turn began
}

Action = Move { from: Sq, to: Sq } | Place { to: Sq } | Stack { target: Sq }
```

### Movement (§3)

- Stone h1: step 1 orthogonally.
- Stone h2 (Pillar): step 1 in any of 8 directions.
- Stone h3 (Spire): config toggle — **Dragon** (default: slide orthogonally any distance + step 1 diagonally) or **Queen** (slide any distance in all 8 directions).
- Keystone: step 1 in any of 8 directions; never stacks, never promotes, never placed from Reserve.
- Never move onto a friendly piece. Slide stops at the first piece and may capture it if enemy.

### Actions & the AP turn (§5)

- `Move` (1 AP), `Place` (1 AP, new h1 Stone on any empty square from Reserve), `Stack` (2 AP = whole turn; add a Reserve token onto own Stone with height < 3, raising height and upgrading movement; never onto a Keystone).
- `apply_action(&mut Position, Action) -> Result<ActionOutcome, IllegalAction>` is the single atomic operation.
- Legal-action generation reads `TurnState`, so AP budget and toggles are enforced at generation time:
  - `Move` legal iff `ap_remaining >= 1`, from-square not in `capture_locked` (if toggle on), and not a keystone already in `keystone_moved` (if toggle on).
  - `Place` legal iff `ap_remaining >= 1`, Reserve > 0, and an empty square exists.
  - `Stack` legal iff `ap_remaining >= 2` (a fresh normal turn), Reserve > 0, and a stackable own Stone exists.

### Capture, Reserve, reuse (§4)

- Capture by displacement: moving onto an enemy piece removes it; your piece occupies the square.
- Capturing a Stone/stack: every token in it goes to the capturer's Reserve (a captured Spire yields 3 tokens). Core anti-runaway rule; not a toggle.
- Capturing a Keystone: removed from the game permanently (not banked), counts toward the win.
- Reserve tokens are generic (used by Place or Stack).

### The turn-ending check rule (§6) — correctness-critical

After **every** action the engine recomputes which of the mover's two target (enemy) Keystones are currently in check (in check = some enemy piece could capture it). If any target Keystone is in check now that was **not** in `enemy_checked_at_start`, the turn ends immediately and remaining AP is forfeit.

This one mechanism produces the spec guarantee: you can only *capture* a Keystone if it was already in check at the start of your turn. To capture you must land on it, which requires it be capturable; the act of first creating that threat ends the turn that creates it, always giving the defender a turn to respond. `Place` and `Stack` that create a new threat end the turn too — all three action types route through the same post-action recheck. Capturing an already-checked Keystone does not create a new check (it removes the piece), so the turn may legally continue with remaining AP.

There is no forced check resolution: any otherwise-legal action is legal even if it leaves your own Keystone in check (§6).

### The four toggles (§7), each localized

- **Spire movement** (Dragon/Queen): a single branch in height-3 move generation.
- **First-player first-turn AP** (integer, e.g. 1 or 2): sets `ap_remaining` when P1's first turn begins; everything else reads `ap_remaining`. Every other turn uses 2 AP.
- **Capture-lock** (default off): on capture, set the destination square in `capture_locked`; `Move` generation skips locked from-squares. Does not restrict Place/Stack.
- **Keystone single-move** (default off): on a Keystone move, set its new square in `keystone_moved`; `Move` generation skips a keystone already in that set.

### RuleConfig

A serde-serializable struct carrying: spire mode, first-turn AP, capture-lock on/off, keystone-single-move on/off, max-ply cap, repetition-fold count. Named-constant defaults in Rust; user-facing default presets and display names (§10) live in `config/*.yaml`.

### Turn advance & terminal detection

Turn advances when `ap_remaining == 0`, the mover has no legal action, or the check rule fired. On advance: grant the opponent a fresh turn, recompute `enemy_checked_at_start`, reset the per-turn bitboards. Terminal states:

- **Win**: both of a player's Keystones captured.
- **Loss**: a player has no legal action at the start of their turn.
- **Draw** (reported): max-ply cap reached or N-fold repetition (Zobrist). Expected to be a rare edge case.

### Error handling

Engine returns `Result` with a typed `IllegalAction` enum; no panics, no `unwrap()`. Token conservation (board + reserves + permanently-removed keystones) is an invariant.

## Section 3 — Policies (`kairnz-policy`)

All operate on the action-level state, RNG-seeded for reproducibility.

- **Random**: uniform over legal actions.
- **Greedy**: 1-ply over legal actions using a simple, named-constant evaluation (material weighted by stack height, Keystone count weighted heavily, Reserve count, light mobility term). Explicitly a baseline, not tuned.
- **Plain UCT MCTS**: tree of action-level Positions, UCB1 selection, random rollouts to terminal-or-ply-cap, configurable iteration budget. Because one turn spans several same-mover nodes, reward is backpropagated by each node's stored `to_move`, not by alternating depth.

## Section 4 — Tauri app + Svelte UI (Human vs Human, §8 mode 1)

Rust holds all game state (`Mutex<HashMap<GameId, Game>>`); the UI holds only a view copy and never computes legality.

Command surface (thin wrappers over `kairnz-core`):

- `new_game(config) -> GameView`
- `legal_actions(game_id, from?) -> Vec<Action>`  (for highlighting a selected piece)
- `apply_action(game_id, action) -> ApplyResult { view, turn_ended_on_check, last_capture, result }`
- `get_view(game_id) -> GameView`
- `undo(game_id) -> GameView`  (optional; history is tracked for repetition detection anyway)

Errors map to serde-serializable responses; normally the UI prevents illegal actions by only offering legal ones.

UI requirements (§8): SVG board showing stack heights; rule-toggle config panel driving `RuleConfig`; readouts for AP-remaining, both Reserve counts; legal-move dots on piece select; a clear banner when a turn auto-ends on check. Display names from `config/*.yaml`.

**Ownership orientation (Shogi-style).** Because Reserve tokens change hands and a captured token re-enters play as the capturer's piece, on-board ownership must be unmistakable at a glance. Pieces use a directional, asymmetric glyph — an upward-tapering stone-pile wedge appropriate to the game's theme — that points toward its owning player's side of the board (P1 pieces point one way, P2 pieces are rotated 180° to point the other), exactly as Shogi pieces face their owner. Orientation, not just color, conveys ownership, so the signal survives colorblindness and a glance. Stack height is layered within the same oriented glyph (Stone → Pillar → Spire), and Keystones get a distinct silhouette but share the directional orientation. The glyph shape is defined once and reused across all pieces.

## Section 5 — Benchmark harness (`kairnz-bench`, §8 mode 4)

Headless CLI taking a run spec (YAML): one or more `RuleConfig`s, games-per-config, seed, and the policy (with params) per side. Runs games deterministically and emits **both** a human-readable summary and machine-readable JSON, with a side-by-side comparison table when given multiple configs (directly serving "compare the §7 toggles").

All six §8 metrics:

- win rate by side (first vs second player)
- draw rate
- median and distribution of game length in plies
- snowball strength: how often the player with the first capture goes on to win
- comeback rate: how often a player who loses a Keystone first still wins
- average highest stack height reached per game

Same seed → identical report (a tested invariant).

## Section 6 — Testing strategy

TDD throughout.

- `kairnz-core`:
  - Per-movement-type tests: step/slide, Dragon vs Queen, friendly-block, displacement capture.
  - Capture → Reserve conservation: Spire yields 3 tokens; Keystone removed, not banked.
  - **Turn-ending-check suite** (heaviest): new threat via Move / Place / Stack each ends the turn; capturing an already-checked Keystone is allowed and may continue the turn; threatening the second Keystone ends the turn; own-Keystone-in-check is legal.
  - Each toggle: on/off paired tests.
  - Win / no-legal-action-loss / draw (max-ply, repetition).
  - Perft-style move-generation counts as regression anchors.
  - Property tests (proptest): AP never negative; token conservation.
- `kairnz-bench`: same seed → identical report.

## Phases 2 & 3 (high-level — Phase 1 leaves room)

- **Phase 2 (AlphaZero self-play, §9):** consumes the same action-level `Position` and `Action` types. The per-action Move/Place/Stack space is the network's action encoding, and `to_move`-aware backprop is already the MCTS shape, so swapping random rollouts for a policy/value net is additive. Hybrid (Rust self-play + PyTorch training, ONNX export) vs all-Rust (tch/candle/burn) is deferred to Phase 2.
- **Phase 3 (Human-vs-AI, AI-vs-AI, §8 modes 2–3):** reuses `kairnz-policy` as the agent interface; a network-backed policy loads the Phase 2 checkpoint and drops into the existing Tauri modes with no engine change.
