# Kairnz: Game and Engine Specification (v0.1)

## 1. Project overview and goals

Build a configurable, rules-accurate implementation of **Kairnz**, an original two-player abstract strategy game, together with a visual board for self-play, a self-play AI for analysis and practice, and a benchmarking harness for tuning the rules toward competitive balance.

Kairnz is a game design still in progress. The aim is a rule set that is deep, effectively drawless, and clean enough to support an Elo rating. Several rules are still being tuned, so the engine must treat key rules as configurable switches, and we will use self-play to measure how each switch affects balance and game character.

High-level deliverables:

1. A single rules engine that enforces the complete rule set in sections 2 through 6.
2. A graphical client with three interactive play modes and one headless training and benchmarking mode (section 8).
3. An AlphaZero-style self-play system (you design the architecture) used both to analyze rule changes and to provide a competent AI opponent for human play and end-to-end testing (section 9).

This document specifies what to build and the exact rules, not the low-level implementation. The high-level technology direction and build order are now chosen and are described in sections 12 and 13. Lower-level choices (frontend framework, rendering approach, crate and library selection, network architecture, and training details) remain yours. Target hardware for training is a single consumer GPU (NVIDIA RTX 4080 Super).

## 2. Board and setup

- Board: a 9 by 9 grid of squares. Fixed at 9 by 9 for now.
- Each player has 20 pieces: 18 Stones and 2 Keystones.
- Starting position for each player, mirrored across the board:
  - Rank 1 (back rank) and Rank 3: nine Stones each, for 18 Stones total.
  - Rank 2: two Keystones on files 3 and 7 (deliberately not adjacent). The rest of Rank 2 starts empty.
- Each player starts with an empty Reserve. Because Reserves begin empty and only fill through captures, the opening naturally plays as single moves and accelerates as captures accumulate.

## 3. Pieces and movement

All Stones start identical. A Stone gains power only by being stacked (section 4). A piece's movement is set by its stack height and type.

Vocabulary: a **step** moves exactly one square; a **slide** moves any number of empty squares in a straight line, stopping at the first piece, which it may capture if that piece is an enemy. Orthogonal means the four cardinal directions. Diagonal means the four diagonals.

- Stone at height 1: step one square orthogonally.
- Stone at height 2 (a Pillar): step one square in any of the eight directions.
- Stone at height 3 (a Spire): configurable, one of:
  - **Dragon** (default): slide orthogonally any distance, and also step one square diagonally.
  - **Queen**: slide any distance in all eight directions.
- Keystone: step one square in any of the eight directions. A Keystone never stacks, never promotes, and is never placed from the Reserve.

A piece may never move onto a square occupied by a friendly piece.

## 4. Capturing, Reserve, and reuse

- Capture is by displacement: moving a piece onto a square holding an enemy piece removes that enemy piece, and your piece occupies the square.
- Capturing a Stone or stack: every token in it goes to the capturing player's Reserve. A captured height-3 Spire therefore yields three Reserve tokens at once. This is the core anti-runaway rule: tall pieces are powerful, but a large loss of material when captured.
- Capturing a Keystone: the Keystone is removed from the game permanently. It does not enter any Reserve. It counts toward the win condition.
- Reserve tokens are generic. They have two possible uses, Place and Stack, defined in section 5.

## 5. Actions and the action-point turn

A turn is a budget of Action Points (AP). A normal turn has 2 AP. The three actions are:

- **Move** (cost 1 AP): move one of your pieces or stacks according to its movement rule.
- **Place** (cost 1 AP): take one token from your Reserve and place it on any empty square as a new height-1 Stone.
- **Stack / Promote** (cost 2 AP, the whole turn): take one token from your Reserve and add it onto one of your own Stones whose height is below 3, raising its height by 1 and upgrading its movement. You may not Stack onto a Keystone.

Within a turn you may spend AP on any combination that fits the budget, including two Moves. The same piece may be moved twice, unless the optional Keystone rule in section 7 is enabled.

A turn ends when AP reach zero, when you have no legal action, or immediately when an action gives check (section 6).

The first player's first turn uses a configurable AP amount (section 7). Every other turn uses 2 AP.

## 6. Check, Keystone safety, and the win condition

- A Keystone is **in check** when an enemy piece could capture it.
- Turn-ending check rule: the instant any of your actions puts an enemy Keystone in check that was not already in check at the start of your turn, your turn ends immediately and any remaining AP is forfeited. The consequence is that you can capture a Keystone only if it was already in check at the start of your turn, which always gives the defender a turn to respond.
- There is no forced check resolution. Any otherwise-legal action is legal even if it leaves your own Keystone in check. Losing a Keystone is the player's own risk.
- Win condition: capture both of the opponent's Keystones.
- A player who has no legal action on their turn loses.
- Kairnz is designed to be effectively drawless. Implement a safeguard against unending games, for example a configurable maximum ply count and/or a position-repetition rule. True draws should be rare edge cases and should be reported when they happen.

## 7. Configurable rules

These must be switchable both in the interactive UI and for benchmarking runs.

1. **Spire movement**: Dragon (default) or Queen.
2. **First player's first-turn AP**: an integer. We are testing values such as 1 and 2. The balanced value is expected to fall between them.
3. **Keystone single-move rule** (on or off, default off): when on, the same Keystone may be moved at most once per turn, so you may not spend both AP moving one Keystone twice. We want to measure gameplay with this on and off.
4. **Capture-lock rule** (on or off, default off): when on, a piece that captures with one action may not be moved again on the same turn. A remaining action point must be spent on a different piece or on another action. This prevents a single piece from chaining multiple captures in one turn. We want to measure gameplay with this on and off.

The anti-runaway capture rule in section 4 is core and is not a toggle.

## 8. Application modes

The same engine powers every mode. The three interactive modes must render the board graphically and must show stack heights, each player's Reserve count, the AP remaining in the current turn, and the legal actions for a selected piece. They must prevent illegal actions and clearly signal when a turn auto-ends because of check.

1. **Human vs Human**: local hot-seat play with full rule enforcement.
2. **Human vs AI**: the human plays against the self-play AI.
3. **AI vs AI**: two AI agents play, shown graphically for observation.
4. **Training and Benchmarking** (headless): runs self-play for a configured number of games under a chosen rule configuration, for both learning and balance measurement, then produces a clear summary report. The operator must be able to set the number of games and the full rule configuration for the run. Suggested metrics:
   - win rate by side (first player vs second player)
   - draw rate
   - median and distribution of game length in plies
   - snowball strength: how often the player who lands the first capture goes on to win
   - comeback rate: how often a player who loses a Keystone first still wins
   - average highest stack height reached per game

## 9. The AlphaZero-style self-play system

Design and implement a self-play reinforcement learning system in the spirit of AlphaZero, meaning a policy and value network guiding a tree search, trained on its own games. The architecture, the state and action encoding, the search, and the training loop are all your decisions. It serves two purposes:

1. **Rule analysis**: train or search under different rule configurations and compare the resulting play, so we can see how changes (Dragon vs Queen, first-turn AP, the optional Keystone rule) affect balance and game character. Where feasible, compare configurations at equal compute, following the approach used to assess game balance with AlphaZero in published chess research.
2. **A playable opponent**: export a checkpoint that the Human vs AI and AI vs AI modes can load, giving a competent opponent for end-to-end testing.

Properties of Kairnz relevant to your design: perfect information, no chance, effectively drawless, two players, a 9 by 9 board, and an action space spanning Move (from-square to to-square), Place (to-square), and Stack (target square), with turns composed of up to 2 AP and a turn that can end early on check. Training must be feasible on a single RTX 4080 Super.

## 10. Naming (proposed, treat as renameable constants)

- Game: **Kairnz**.
- **Stone**: the base height-1 piece.
- **Pillar**: a height-2 stack.
- **Spire**: a height-3 stack (its movement is the Dragon or Queen toggle).
- **Keystone**: the precious piece. Each player has two, and both must be captured to win.

Keep all display names in one place so they are easy to change.

## 11. Non-goals and decisions left to you

- Lower-level choices within the chosen stack: frontend framework, rendering approach, and crate or library selection (see section 13).
- Network architecture, state and action encoding, search, and training details.
- File and project structure.
- Anything not stated as a rule in sections 2 through 7.

## 12. Build phasing

Deliver in three phases. Each phase should be usable on its own.

1. **Phase 1 (first delivery)**: the Rust rules engine, Human vs Human play with full rule enforcement and a graphical board, and the headless Training and Benchmarking harness (section 8, mode 4) using non-learned policies: random, greedy, and plain UCT MCTS. No neural network and no GPU are required for this phase. This delivers a rule-correct, playable board and real balance data early, including side-by-side comparisons of the section 7 toggles.
2. **Phase 2**: the AlphaZero-style self-play system (section 9), producing trained checkpoints and richer balance analysis.
3. **Phase 3**: the Human vs AI and AI vs AI interactive modes (section 8, modes 2 and 3), which load a trained checkpoint from Phase 2.

## 13. Technical direction (chosen)

- The application is a cross-platform desktop app for Windows, Linux, and macOS, built with **Rust and Tauri**. The interactive UI runs in Tauri's webview (web frontend of your choice). The Rust backend is the single source of truth for all rules and game state.
- The rules engine, move generation, the benchmarking harness, plain MCTS, self-play game generation, and in-app model inference all live in Rust, so the shipped desktop binary is self-contained and fast.
- For the Phase 2 network training step, choose one of two paths:
  - **Hybrid**: keep self-play and MCTS in Rust (the parallel, compute-heavy part) and train the network in Python with PyTorch (the mature path used by nearly all reference AlphaZero implementations), exporting the trained network to a portable format such as ONNX or TorchScript.
  - **All-Rust**: train within Rust using tch-rs (LibTorch bindings, CUDA-capable) or a native framework such as candle or burn, trading some ecosystem maturity for a single toolchain.
  In both cases the trained network must export to a portable file that the Rust app loads for inference, so the distributed desktop app does not require a Python runtime.
- Hardware and expectations: training targets a single RTX 4080 Super. On a 9 by 9 board with this action space, a single GPU can produce a competent sparring opponent and a directional balance signal, not a quickly solved or superhuman agent. For rule-comparison research, prefer plain MCTS at equal search budget (available in Phase 1) and lightly trained networks; reaching strong play would require substantially more compute.
