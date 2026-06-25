# Plan 10: Play Against the Trained AI in the App

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Let a human play against a trained Kairnz model in the Tauri/Svelte desktop app, choosing which side the AI plays and how strong it is.

**Architecture:** The Tauri backend gains an `ai_move` command backed by an `AiEngine` that lazily loads an `AzMctsPolicy` from a model path and reuses it. The command clones the current game (brief store lock), runs the MCTS search WITHOUT holding the store lock, then applies the chosen move through the existing apply path. The Svelte UI tracks an AI mode (off / AI plays P1 / AI plays P2) plus difficulty and model path; after each human move (and at new-game when the AI is first), it drives the AI to move, looping while it remains the AI's turn (multi-AP turns). AI inference runs on CPU in the app, which is fine for interactive turn-based play.

**Tech Stack:** Rust (`src-tauri`, adding `kairnz-onnx` + `kairnz-policy`), Svelte 5 + TS (`ui`).

## Global Constraints

- The AI uses `AzMctsPolicy` with `dirichlet_epsilon = 0.0` (deterministic best-move play, no self-play exploration noise). Difficulty maps to the MCTS simulation count.
- The `ai_move` command must NOT hold the `GameStore` lock during the (slow) MCTS search: clone the game under the lock, search lock-free, then apply.
- The AI policy is loaded once and cached; it reloads only when the model path or simulation count changes.
- Mirror the existing command/DTO conventions exactly (`ApplyResult` return, `Action` externally-tagged JSON, the `api.ts` invoke wrapper pattern, the `refreshAfterAction` flow).
- Frontend tasks verify by `pnpm build` (svelte-check, 0 errors); visual confirmation is a manual user step.
- Rust: named constants, doc comments, `Result<_, String>` on the command path (no panics/unwrap on model load or inference), no em dashes.

---

## File Structure

- Modify: `src-tauri/Cargo.toml` — add `kairnz-onnx`, `kairnz-policy`.
- Create: `src-tauri/src/ai.rs` — `AiEngine` (lazy policy cache) + `choose`.
- Modify: `src-tauri/src/state.rs` — add `clone_game(id) -> Result<Game, String>`.
- Modify: `src-tauri/src/commands.rs` — add the `ai_move` command.
- Modify: `src-tauri/src/main.rs` — `mod ai;`, `.manage(AiEngine::default())`, register `ai_move`.
- Modify: `ui/src/lib/api.ts` — `aiMove(...)`.
- Modify: `ui/src/App.svelte` — AI mode state + move-driving loop.
- Modify: `ui/src/components/ConfigPanel.svelte` (or a new AI section) — AI controls.

---

### Task 1: Backend `ai_move` command

**Files:**
- Modify: `src-tauri/Cargo.toml`, `src-tauri/src/state.rs`, `src-tauri/src/commands.rs`, `src-tauri/src/main.rs`
- Create: `src-tauri/src/ai.rs`

**Interfaces:**
- Produces: `AiEngine` (Tauri-managed state) with `choose(&self, game: &Game, model_path: &Path, simulations: u32) -> Result<Action, String>`; `GameStore::clone_game(id) -> Result<Game, String>`; a `#[tauri::command] ai_move(id, model: String, simulations: u32, store, ai) -> Result<ApplyResult, String>`.

- [ ] **Step 1: Add dependencies**

In `src-tauri/Cargo.toml` `[dependencies]`, add (the AI policy and the `Policy` trait):

```toml
kairnz-onnx = { path = "../crates/kairnz-onnx" }
kairnz-policy = { path = "../crates/kairnz-policy" }
```

- [ ] **Step 2: Write the AI engine**

Create `src-tauri/src/ai.rs`:

```rust
//! In-app AI opponent: a lazily-loaded neural MCTS policy.

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use kairnz_core::actions::Action;
use kairnz_core::game::Game;
use kairnz_onnx::{AzMctsConfig, AzMctsPolicy};
use kairnz_policy::policy::Policy;

/// Fixed seed for the in-app AI (deterministic; epsilon 0 means the seed is inert).
const AI_SEED: u64 = 0;

/// A loaded policy plus the model/strength it was built for.
struct Loaded {
    model: PathBuf,
    simulations: u32,
    policy: AzMctsPolicy,
}

/// Tauri-managed state holding a lazily-loaded, reusable AI policy.
#[derive(Default)]
pub struct AiEngine {
    inner: Mutex<Option<Loaded>>,
}

impl AiEngine {
    /// Chooses the AI's move for `game`, loading or reusing a policy for the given
    /// model path and simulation budget. Returns an error string on model-load
    /// failure or if no legal move exists.
    pub fn choose(&self, game: &Game, model_path: &Path, simulations: u32) -> Result<Action, String> {
        let mut guard = self.inner.lock().map_err(|_| "AI engine lock poisoned".to_string())?;

        let needs_load = match guard.as_ref() {
            Some(loaded) => loaded.model != model_path || loaded.simulations != simulations,
            None => true,
        };
        if needs_load {
            let config = AzMctsConfig {
                simulations,
                dirichlet_epsilon: 0.0,
                ..AzMctsConfig::default()
            };
            let policy = AzMctsPolicy::from_path(model_path, config, AI_SEED)
                .map_err(|e| format!("failed to load AI model: {e}"))?;
            *guard = Some(Loaded { model: model_path.to_path_buf(), simulations, policy });
        }

        let loaded = guard.as_mut().expect("policy was just loaded");
        loaded
            .policy
            .choose(game)
            .ok_or_else(|| "AI found no legal move".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kairnz_core::actions::legal_actions;
    use kairnz_core::config::RuleConfig;
    use std::path::PathBuf;

    fn fixture() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../crates/kairnz-onnx/tests/fixtures/random_init.onnx")
    }

    #[test]
    fn ai_chooses_a_legal_move_at_the_opening() {
        let engine = AiEngine::default();
        let game = Game::new_standard(RuleConfig::default());
        let action = engine.choose(&game, &fixture(), 16).expect("ai chooses");
        assert!(legal_actions(&game.pos).contains(&action), "AI move must be legal");
    }
}
```

- [ ] **Step 3: Add `clone_game` to the store**

Read `src-tauri/src/state.rs`. Add a method to `GameStore` that returns a clone of a game (so the AI search runs without holding the store lock). Match the existing field/lock names:

```rust
    /// Returns a clone of the game with `id`, or an error if it does not exist.
    pub fn clone_game(&self, id: GameId) -> Result<Game, String> {
        let games = self.games.lock().map_err(|_| "game store lock poisoned".to_string())?;
        games
            .get(&id)
            .map(|entry| entry.game.clone())
            .ok_or_else(|| "no such game".to_string())
    }
```

(Adjust `self.games` / `entry.game` to the real field names in `state.rs`. `Game` derives `Clone`.)

- [ ] **Step 4: Add the `ai_move` command**

Read `src-tauri/src/commands.rs` and the existing `apply_action` command to match its exact apply path / return mapping. Add:

```rust
#[tauri::command]
pub fn ai_move(
    id: GameId,
    model: String,
    simulations: u32,
    store: tauri::State<GameStore>,
    ai: tauri::State<crate::ai::AiEngine>,
) -> Result<ApplyResult, String> {
    // Clone the game (brief lock), search lock-free, then apply.
    let game = store.clone_game(id)?;
    let action = ai.choose(&game, std::path::Path::new(&model), simulations)?;
    store.apply_action(id, action)
}
```

Match the `store.apply_action(...)` call to whatever the existing `apply_action` command uses to apply and produce `ApplyResult` (same method, same error mapping). If the existing command maps an `IllegalAction` error, mirror that mapping here.

- [ ] **Step 5: Wire the module, state, and command in `main.rs`**

In `src-tauri/src/main.rs`: add `mod ai;`, register the managed state and the command:

```rust
        .manage(ai::AiEngine::default())
```
and add `commands::ai_move` to the `tauri::generate_handler![...]` list (keep the existing commands).

- [ ] **Step 6: Build and test**

Run: `cargo build -p kairnz-tauri`
Expected: builds (first build downloads the ONNX Runtime binary via `ort`). Warning-free.

Run: `cargo test -p kairnz-tauri ai`
Expected: PASS (the AI chooses a legal opening move with the fixture model).

- [ ] **Step 7: Commit**

```bash
git add src-tauri/Cargo.toml src-tauri/src/ai.rs src-tauri/src/state.rs src-tauri/src/commands.rs src-tauri/src/main.rs
git commit -m "feat(app): add ai_move command backed by AzMctsPolicy"
```

---

### Task 2: Frontend API and AI move flow

**Files:**
- Modify: `ui/src/lib/api.ts`, `ui/src/App.svelte`

**Interfaces:**
- Produces: `aiMove(id, model, simulations)` in `api.ts`; AI-mode state and an AI-move-driving loop in `App.svelte`.

- [ ] **Step 1: Add the API wrapper**

In `ui/src/lib/api.ts`, mirroring the existing `applyAction` wrapper, add:

```typescript
export function aiMove(id: GameId, model: string, simulations: number): Promise<ApplyResult> {
  return invoke('ai_move', { id, model, simulations });
}
```

- [ ] **Step 2: Add AI state and the move loop in `App.svelte`**

Read `ui/src/App.svelte`. Add reactive state (Svelte 5 runes, matching the file's style) near the other state:

```typescript
  let aiEnabled = $state(false);
  let aiSide: Player = $state('P2');
  let aiSims = $state(200);
  let aiModel = $state('models/best.onnx');
```

Import `aiMove` from the api module. Add an AI-driving function that loops while it is the AI's turn (handles multi-AP turns) and a small guard against runaway loops:

```typescript
  async function driveAi() {
    let guard = 0;
    while (
      aiEnabled && gameId !== null && view !== null &&
      view.result === null && view.to_move === aiSide
    ) {
      busy = true;
      try {
        const result = await aiMove(gameId, aiModel, aiSims);
        view = result.view;
        legal = await legalActions(gameId);
      } catch (e) {
        error = String(e);
        break;
      } finally {
        busy = false;
      }
      if (++guard > 1000) break;
    }
  }
```

Trigger `driveAi()` (a) after a human move's `refreshAfterAction` completes, and (b) at the end of `handleNewGame` (so the AI moves first when it plays P1). Call it fire-and-forget (`void driveAi();`) so the human's `await` returns promptly; the `busy` flag shows "AI thinking" in the UI.

Adjust the exact integration points to match the file (the post-`applyAction` path in `dispatch`/`refreshAfterAction`, and the new-game handler). Keep the human-input gating unchanged - `busy` already blocks clicks during the AI's turn.

- [ ] **Step 3: Build**

Run: `cd ui && pnpm build`
Expected: svelte-check passes with 0 errors (types align; `aiMove` returns `ApplyResult`).

- [ ] **Step 4: Commit**

```bash
git add ui/src/lib/api.ts ui/src/App.svelte
git commit -m "feat(ui): drive AI opponent moves after human turns"
```

---

### Task 3: AI controls UI and final verification

**Files:**
- Modify: `ui/src/components/ConfigPanel.svelte` (or add an AI section in `App.svelte`)

**Interfaces:**
- Produces: UI controls bound to `App.svelte`'s `aiEnabled`/`aiSide`/`aiSims`/`aiModel`.

- [ ] **Step 1: Add the AI controls**

Add an "AI opponent" section (in `ConfigPanel.svelte`, passing values up via props/bindings, or directly in `App.svelte` near the config). Controls:
- A checkbox: "Play vs AI" -> `aiEnabled`.
- A side selector (radio or select): "AI plays P1 / P2" -> `aiSide`.
- A difficulty select mapping to `aiSims`: e.g. Easy = 50, Medium = 200, Hard = 800.
- A text input for the model path -> `aiModel` (default `models/best.onnx`; the user can paste an absolute path to their trained model).

Use the existing component's binding conventions (Svelte 5 `$bindable`/props or event callbacks, matching how `ConfigPanel` already passes `RuleConfig` to `App`). Keep styling consistent with the existing panel (CSS custom properties, no inline color literals).

Note: the model-path field default is relative; if it does not resolve at runtime (the app's working directory may differ), the user pastes an absolute path. A file picker is a possible later enhancement.

- [ ] **Step 2: Build and full check**

Run: `cd ui && pnpm build`
Expected: svelte-check passes, 0 errors.

Run: `cargo build -p kairnz-tauri`
Expected: builds warning-free.

- [ ] **Step 3: Commit**

```bash
git add ui/src/components/ConfigPanel.svelte ui/src/App.svelte
git commit -m "feat(ui): add AI opponent controls (mode, side, difficulty, model)"
```

- [ ] **Step 4: Manual visual verification (user step)**

This is a manual check by the user (not an agent step): run `pnpm tauri dev`, enable "Play vs AI", point the model path at a trained `best.onnx`, start a game, and confirm the AI responds with legal moves on its turn. The "AI thinking" state shows while it searches.

---

## Self-Review Notes

- **Scope:** adds a neural AI opponent to the existing human-vs-human app. The AI reuses `AzMctsPolicy` (deterministic, epsilon 0); difficulty is the simulation count. No change to the game rules or the existing commands.
- **Concurrency:** `ai_move` clones the game under a brief lock and searches lock-free, so the UI's `get_view`/`legal_actions` are not blocked during the AI's (slow) think. The policy is cached and reused; it reloads only on model/strength change.
- **Multi-AP turns:** the UI loops `ai_move` while `to_move` stays the AI's side, so the AI plays all its action points before handing back, with a runaway guard.
- **CPU inference:** the app runs `ort` on CPU (no cuDNN PATH), which is fine for interactive play; GPU in-app is a future enhancement.
- **Verification:** the backend AI choice is unit-tested with the fixture model; the frontend is build/type-checked; the end-to-end play is a manual visual step (consistent with the app's established frontend verification approach).
- **Type/name consistency:** `AiEngine::choose`, `GameStore::clone_game`, the `ai_move` command, and the `aiMove` API wrapper are referenced identically across tasks. Model-path/simulation values flow UI -> `aiMove` -> `ai_move` -> `AiEngine::choose`.
