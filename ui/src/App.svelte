<script lang="ts">
  import { onMount } from 'svelte';
  import { newGame, legalActions, applyAction, undo } from './lib/api.js';
  import { defaultConfig } from './lib/types.js';
  import type { Action, GameId, GameView, Player, RuleConfig, Sq } from './lib/types.js';
  import Board from './components/Board.svelte';
  import ConfigPanel from './components/ConfigPanel.svelte';
  import Sidebar from './components/Sidebar.svelte';
  import MoveHistory from './components/MoveHistory.svelte';
  import { names } from './lib/names.js';
  import { actionToNotation } from './lib/notation.js';

  // ---------------------------------------------------------------------------
  // Core state
  // ---------------------------------------------------------------------------

  let gameId = $state<GameId | null>(null);
  let view = $state<GameView | null>(null);
  let legal = $state<Action[]>([]);
  let selected = $state<Sq | null>(null);
  let banner = $state<string | null>(null);
  let pendingPlace = $state(false);
  let config = $state<RuleConfig>({ ...defaultConfig });
  let error = $state<string | null>(null);
  let busy = $state(false);

  // ---------------------------------------------------------------------------
  // Move history
  // ---------------------------------------------------------------------------
  interface HistoryEntry { ply: number; player: Player; text: string; }
  let history = $state<HistoryEntry[]>([]);
  let plyCounter = $state(0);

  // ---------------------------------------------------------------------------
  // Affordance derivation -- all membership checks run against `legal`
  // so it is structurally impossible to submit an action not present there.
  // ---------------------------------------------------------------------------

  /** to values for Move actions from a given square. */
  function moveTargetsFrom(sq: Sq): Sq[] {
    const targets: Sq[] = [];
    for (const a of legal) {
      if ('Move' in a && a.Move.from === sq) targets.push(a.Move.to);
    }
    return targets;
  }

  /** Squares that are valid Stack targets. */
  const stackableSquares = $derived(
    legal.filter((a): a is { Stack: { target: Sq } } => 'Stack' in a)
         .map((a) => a.Stack.target)
  );

  /** Squares that are valid Place destinations. */
  const placeTargets = $derived(
    legal.filter((a): a is { Place: { to: Sq } } => 'Place' in a)
         .map((a) => a.Place.to)
  );

  const canPlace = $derived(placeTargets.length > 0);

  /** Move-target dots for the currently selected square. */
  const currentMoveTargets = $derived(
    selected !== null ? moveTargetsFrom(selected) : []
  );

  /** Whether the selected square is a valid stack target. */
  const selectedIsStackable = $derived(
    selected !== null && stackableSquares.includes(selected)
  );

  const gameOver = $derived(view !== null && view.result !== null);

  // ---------------------------------------------------------------------------
  // Helpers: action membership guards
  // ---------------------------------------------------------------------------

  function isMoveInLegal(from: Sq, to: Sq): boolean {
    return legal.some((a) => 'Move' in a && a.Move.from === from && a.Move.to === to);
  }

  function isStackInLegal(target: Sq): boolean {
    return legal.some((a) => 'Stack' in a && a.Stack.target === target);
  }

  function isPlaceInLegal(to: Sq): boolean {
    return legal.some((a) => 'Place' in a && a.Place.to === to);
  }

  // ---------------------------------------------------------------------------
  // State refresh after any action or new game
  // ---------------------------------------------------------------------------

  async function refreshAfterAction(newView: GameView, newGameId: GameId, checkBanner: boolean) {
    view = newView;
    if (newView.result === null) {
      legal = await legalActions(newGameId);
    } else {
      legal = [];
    }
    selected = null;
    pendingPlace = false;
    if (checkBanner) {
      // banner already set by caller
    }
  }

  // ---------------------------------------------------------------------------
  // Action dispatch -- the single pathway for all game actions
  // ---------------------------------------------------------------------------

  async function dispatch(id: GameId, action: Action) {
    if (busy) return;
    busy = true;
    banner = null;
    try {
      const mover: Player = view!.to_move;
      const result = await applyAction(id, action);
      if (result.turn_ended_on_check) {
        banner = names.check_banner;
      }
      plyCounter += 1;
      const notation = actionToNotation(action, {
        capture: result.last_capture !== null,
        checkEnd: result.turn_ended_on_check,
        gameOver: result.result !== null,
      });
      history = [...history, { ply: plyCounter, player: mover, text: notation }];
      await refreshAfterAction(result.view, id, true);
    } catch (e) {
      error = String(e);
    } finally {
      busy = false;
    }
  }

  // ---------------------------------------------------------------------------
  // Square click handler (wired to Board)
  // ---------------------------------------------------------------------------

  function handleSquareClick(sq: Sq) {
    if (!view || !gameId || gameOver || busy) return;

    // While a Place is pending, any click on a valid place target applies it.
    if (pendingPlace) {
      if (isPlaceInLegal(sq)) {
        const action: Action = { Place: { to: sq } };
        void dispatch(gameId, action);
      } else {
        // Click outside a place target cancels pending place.
        pendingPlace = false;
      }
      return;
    }

    // If a square is already selected:
    if (selected !== null) {
      // Clicking a valid move target applies the Move.
      if (isMoveInLegal(selected, sq)) {
        const from = selected;
        const action: Action = { Move: { from, to: sq } };
        void dispatch(gameId, action);
        return;
      }
      // Clicking the already-selected square deselects.
      if (sq === selected) {
        selected = null;
        return;
      }
    }

    // Try to select a new piece owned by the current player.
    const piece = view.board[sq];
    if (piece !== null && piece.owner === view.to_move) {
      selected = sq;
      return;
    }

    // Clicking anything else clears selection.
    selected = null;
  }

  // ---------------------------------------------------------------------------
  // Stack handler (triggered by button in template)
  // ---------------------------------------------------------------------------

  function handleStack() {
    if (!gameId || selected === null || !isStackInLegal(selected)) return;
    const target = selected;
    void dispatch(gameId, { Stack: { target } });
  }

  // ---------------------------------------------------------------------------
  // Undo handler
  // ---------------------------------------------------------------------------

  async function handleUndo() {
    if (!gameId || busy || gameOver) return;
    busy = true;
    banner = null;
    try {
      const newView = await undo(gameId);
      legal = await legalActions(gameId);
      view = newView;
      selected = null;
      pendingPlace = false;
      if (history.length > 0) history = history.slice(0, -1);
      if (plyCounter > 0) plyCounter -= 1;
    } catch (e) {
      error = String(e);
    } finally {
      busy = false;
    }
  }

  // ---------------------------------------------------------------------------
  // New game handler
  // ---------------------------------------------------------------------------

  async function handleNewGame(cfg: RuleConfig) {
    busy = true;
    banner = null;
    error = null;
    selected = null;
    pendingPlace = false;
    history = [];
    plyCounter = 0;
    try {
      const [id, initialView] = await newGame(cfg);
      gameId = id;
      view = initialView;
      legal = await legalActions(id);
    } catch (e) {
      error = String(e);
    } finally {
      busy = false;
    }
  }

  // ---------------------------------------------------------------------------
  // Initial load
  // ---------------------------------------------------------------------------

  onMount(() => {
    void handleNewGame(defaultConfig);
  });
</script>

<main>
  <h1>{names.game}</h1>

  {#if error}
    <p class="error">Error: {error}</p>
  {/if}

  <div class="layout">
    <ConfigPanel bind:config onNewGame={handleNewGame} disabled={busy} />

    <div class="board-area">
      {#if view}
        <!-- Action buttons above the board -->
        <div class="action-bar">
          {#if !gameOver && canPlace}
            <button
              class="btn-action"
              class:active={pendingPlace}
              onclick={() => { pendingPlace = !pendingPlace; selected = null; }}
              disabled={busy}
            >
              {pendingPlace ? 'Cancel Place' : 'Place from reserve'}
            </button>
          {/if}
          {#if !gameOver && selectedIsStackable}
            <button
              class="btn-action"
              onclick={handleStack}
              disabled={busy || selected === null || !isStackInLegal(selected)}
            >
              Stack (2 AP)
            </button>
          {/if}
        </div>

        <Board
          {view}
          selectedSq={selected}
          legalTargets={currentMoveTargets}
          stackable={stackableSquares}
          placeTargets={placeTargets}
          pendingPlace={pendingPlace}
          onSquareClick={handleSquareClick}
        />
      {:else}
        <p class="loading">Loading...</p>
      {/if}
    </div>

    {#if view}
      <Sidebar
        {view}
        {banner}
        onUndo={handleUndo}
        gameOver={gameOver}
      />
      <MoveHistory {history} />
    {/if}
  </div>
</main>

<style>
  :global(:root) {
    --board-light: #f0d9b5;
    --board-dark: #b58863;
    --board-border: #5a3e28;
    --grid-line: #5a3e2855;
    --piece-p1: #e8d5a3;
    --piece-p2: #8b4513;
    --piece-stroke: #2a2a2a;
    --piece-stroke-w: 1px;
    --coord: #5a3e28;
  }

  main {
    font-family: sans-serif;
    padding: 2rem;
    display: flex;
    flex-direction: column;
    gap: 1rem;
  }

  h1 {
    margin: 0;
  }

  .layout {
    display: flex;
    flex-direction: row;
    align-items: flex-start;
    gap: 1.5rem;
    flex-wrap: wrap;
  }

  .board-area {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
    align-items: flex-start;
  }

  .action-bar {
    display: flex;
    gap: 0.5rem;
    min-height: 2rem;
  }

  .btn-action {
    padding: 0.35rem 0.8rem;
    background: var(--board-border);
    color: #fff;
    border: none;
    border-radius: 3px;
    font-size: 0.88rem;
    cursor: pointer;
    transition: opacity 0.15s;
  }

  .btn-action:hover:not(:disabled) {
    opacity: 0.85;
  }

  .btn-action:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .btn-action.active {
    background: #0066cc;
  }

  .error {
    color: red;
    margin: 0;
  }

  .loading {
    margin: 0;
    color: #666;
  }
</style>
