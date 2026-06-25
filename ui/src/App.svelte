<script lang="ts">
  import { onMount } from 'svelte';
  import { newGame, legalActions, applyAction, undo, pieceMoves, aiMove } from './lib/api.js';
  import { defaultConfig } from './lib/types.js';
  import type { Action, GameId, GameView, Player, RuleConfig, Sq } from './lib/types.js';
  import Board from './components/Board.svelte';
  import ConfigPanel from './components/ConfigPanel.svelte';
  import Sidebar from './components/Sidebar.svelte';
  import MoveHistory from './components/MoveHistory.svelte';
  import { names } from './lib/names.js';
  import { actionToNotation, actionSquares, sqToCoord } from './lib/notation.js';

  // ---------------------------------------------------------------------------
  // Core state
  // ---------------------------------------------------------------------------

  let gameId = $state<GameId | null>(null);
  let view = $state<GameView | null>(null);
  let legal = $state<Action[]>([]);
  let selected = $state<Sq | null>(null);
  let banner = $state<string | null>(null);
  let config = $state<RuleConfig>({ ...defaultConfig });
  let error = $state<string | null>(null);
  let busy = $state(false);

  // ---------------------------------------------------------------------------
  // AI opponent state
  // ---------------------------------------------------------------------------

  let aiEnabled = $state(false);
  let aiSide: Player = $state('P2');
  let aiSims = $state(200);
  let aiModel = $state('models/best.onnx');

  /** Middle-click preview: the inspected square and its geometric move targets. */
  let inspect = $state<{ sq: number; targets: number[] } | null>(null);

  /** Right-click confirmation prompt. */
  let prompt = $state<{ kind: 'place' | 'promote'; sq: number } | null>(null);

  // ---------------------------------------------------------------------------
  // Move history
  // ---------------------------------------------------------------------------
  interface HistoryEntry { ply: number; player: Player; text: string; squares: number[]; }
  let history = $state<HistoryEntry[]>([]);
  let plyCounter = $state(0);

  // ---------------------------------------------------------------------------
  // Display options
  // ---------------------------------------------------------------------------
  /** When true, highlights the opponent's most recent completed turn on the board. */
  let showPrevMove = $state(true);

  /** Helper: returns the opponent of the given player. */
  function opponentOf(p: Player): Player { return p === 'P1' ? 'P2' : 'P1'; }

  /**
   * Derives the squares to highlight for the previous-move indicator.
   * Walks history from the end:
   *  1. Skips any trailing entries by the current player (their own in-progress AP actions).
   *  2. Collects the contiguous run of entries by the opponent (their last completed turn).
   * Returns a de-duplicated list of squares, or [] at game start / when no opponent turn exists.
   */
  const prevMoveSquares = $derived((): number[] => {
    if (!view) return [];
    const current = view.to_move;
    const opponent = opponentOf(current);
    let i = history.length - 1;
    // Skip trailing current-player entries (in-progress AP actions for the current turn).
    while (i >= 0 && history[i].player === current) i--;
    // Collect the opponent's most recent contiguous run.
    const collected: number[] = [];
    while (i >= 0 && history[i].player === opponent) {
      for (const sq of history[i].squares) collected.push(sq);
      i--;
    }
    // De-duplicate while preserving order.
    return [...new Set(collected)];
  });

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

  /** Move-target dots for the currently selected square. */
  const currentMoveTargets = $derived(
    selected !== null ? moveTargetsFrom(selected) : []
  );

  const gameOver = $derived(view !== null && view.result !== null);

  /** Whether each action type has at least one legal instance right now. */
  const canMove = $derived(legal.some((a) => 'Move' in a));
  const canPlace = $derived(legal.some((a) => 'Place' in a));
  const canPromote = $derived(legal.some((a) => 'Stack' in a));

  /** Keystone squares belonging to the side to move that are currently in check. */
  const myCheckedKeystones = $derived(
    view !== null
      ? view.checked_keystones.filter((sq) => {
          const pc = view!.board[sq];
          return pc !== null && pc.owner === view!.to_move;
        })
      : []
  );

  /** Alert text shown at turn start when the current player's keystones are in check. */
  const checkAlert = $derived(
    myCheckedKeystones.length > 0 && !gameOver
      ? `Check! Your Keystone${myCheckedKeystones.length > 1 ? 's' : ''} at ${myCheckedKeystones.map(sqToCoord).join(', ')} ${myCheckedKeystones.length > 1 ? 'are' : 'is'} threatened.`
      : null
  );

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

  async function refreshAfterAction(newView: GameView, newGameId: GameId) {
    view = newView;
    if (newView.result === null) {
      legal = await legalActions(newGameId);
    } else {
      legal = [];
    }
    selected = null;
    inspect = null;
    prompt = null;
  }

  // ---------------------------------------------------------------------------
  // AI move driver -- loops while it is the AI's turn (handles multi-AP turns)
  // ---------------------------------------------------------------------------

  /**
   * Drives AI moves until the turn passes to the human or the game ends.
   * Called fire-and-forget after each human move and at new-game when the AI
   * plays the starting side. The `busy` flag blocks human input during AI turns.
   */
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

  // ---------------------------------------------------------------------------
  // Action dispatch -- the single pathway for all game actions
  // ---------------------------------------------------------------------------

  async function dispatch(id: GameId, action: Action) {
    if (busy) return;
    busy = true;
    banner = null;
    try {
      const mover: Player = view!.to_move;
      const movingPiece = 'Move' in action ? (view!.board[action.Move.from] ?? null) : null;
      const result = await applyAction(id, action);
      if (result.turn_ended_on_check) {
        banner = names.check_banner;
      }
      plyCounter += 1;
      const notation = actionToNotation(
        action,
        {
          capture: result.last_capture !== null,
          checkEnd: result.turn_ended_on_check,
          gameOver: result.result !== null,
        },
        names.piece_codes,
        movingPiece
      );
      history = [...history, { ply: plyCounter, player: mover, text: notation, squares: actionSquares(action) }];
      await refreshAfterAction(result.view, id);
      void driveAi();
    } catch (e) {
      error = String(e);
    } finally {
      busy = false;
    }
  }

  // ---------------------------------------------------------------------------
  // Left-click: select/move
  // ---------------------------------------------------------------------------

  function handleSquareClick(sq: Sq) {
    if (!view || !gameId || gameOver || busy) return;
    inspect = null;
    prompt = null;

    if (selected !== null) {
      if (isMoveInLegal(selected, sq)) {
        const from = selected;
        const action: Action = { Move: { from, to: sq } };
        void dispatch(gameId, action);
        return;
      }
      if (sq === selected) {
        selected = null;
        return;
      }
    }

    const piece = view.board[sq];
    if (piece !== null && piece.owner === view.to_move) {
      selected = sq;
      return;
    }

    selected = null;
  }

  // ---------------------------------------------------------------------------
  // Middle-click: inspect any piece's geometric moves
  // ---------------------------------------------------------------------------

  async function handleInspect(sq: Sq) {
    if (!view || !gameId) return;
    const piece = view.board[sq];
    if (piece === null) {
      inspect = null;
      return;
    }
    try {
      const targets = await pieceMoves(gameId, sq);
      inspect = { sq, targets };
      selected = null;
      prompt = null;
    } catch (e) {
      error = String(e);
    }
  }

  // ---------------------------------------------------------------------------
  // Right-click: promote or place prompt
  // ---------------------------------------------------------------------------

  function handleContext(sq: Sq) {
    if (!view || !gameId || gameOver) return;
    const piece = view.board[sq];
    if (piece !== null && piece.owner === view.to_move && isStackInLegal(sq)) {
      prompt = { kind: 'promote', sq };
    } else if (piece === null && isPlaceInLegal(sq)) {
      prompt = { kind: 'place', sq };
    } else {
      prompt = null;
    }
  }

  // ---------------------------------------------------------------------------
  // Prompt confirm / cancel
  // ---------------------------------------------------------------------------

  function confirmPrompt() {
    if (!prompt || !gameId) return;
    const { kind, sq } = prompt;
    prompt = null;
    if (kind === 'promote') {
      void dispatch(gameId, { Stack: { target: sq } });
    } else {
      void dispatch(gameId, { Place: { to: sq } });
    }
  }

  function cancelPrompt() {
    prompt = null;
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
      inspect = null;
      prompt = null;
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
    inspect = null;
    prompt = null;
    history = [];
    plyCounter = 0;
    try {
      const [id, initialView] = await newGame(cfg);
      gameId = id;
      view = initialView;
      legal = await legalActions(id);
      void driveAi();
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
    {#if view}
      <MoveHistory {history} />
    {/if}

    <div class="board-area">
      {#if view}
        <Board
          {view}
          selectedSq={selected}
          legalTargets={currentMoveTargets}
          stackable={stackableSquares}
          inspectTargets={inspect?.targets ?? []}
          {prompt}
          checkedKeystones={view.checked_keystones}
          prevMoveSquares={showPrevMove ? prevMoveSquares() : []}
          onSquareClick={handleSquareClick}
          onInspect={handleInspect}
          onContext={handleContext}
          onPromptConfirm={confirmPrompt}
          onPromptCancel={cancelPrompt}
        />
      {:else}
        <p class="loading">Loading...</p>
      {/if}
    </div>

    <div class="side-col">
      {#if view}
        <Sidebar
          {view}
          {banner}
          {checkAlert}
          {canMove}
          {canPlace}
          {canPromote}
          onUndo={handleUndo}
          gameOver={gameOver}
          bind:showPrevMove
        />
      {/if}
      <ConfigPanel bind:config onNewGame={handleNewGame} disabled={busy} />
    </div>
  </div>
</main>

<style>
  :global(:root) {
    --board-light: #f0d9b5;
    --board-dark: #b58863;
    --board-border: #5a3e28;
    --grid-line: #5a3e2855;
    --piece-p1: #1a1a1a;
    --piece-p2: #f5f0e8;
    --piece-stroke: #2a2a2a;
    --piece-stroke-w: 1px;
    --coord: #5a3e28;
    --check: #cc2200;
    --inspect-dot: #7c3aed;
    --inspect-dot-stroke: #5b21b6;
    --last-move: rgba(210, 160, 30, 0.35);
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

  .side-col {
    display: flex;
    flex-direction: column;
    gap: 1rem;
    align-self: flex-start;
  }

  .board-area {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
    align-items: flex-start;
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
