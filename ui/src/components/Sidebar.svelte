<!--
  Sidebar.svelte -- Game status, player info, actions, banners, and undo.

  Shows: whose turn, AP remaining, actions availability list, both reserves,
  check alert, check-auto-end banner, game-over message. Undo button calls
  the provided handler.
-->
<script lang="ts">
  import type { GameView } from '../lib/types.js';
  import { names } from '../lib/names.js';

  /** AP cost constants -- fixed game rules, not data. */
  const AP_MOVE = 1;
  const AP_PLACE = 1;
  const AP_PROMOTE = 2;

  interface Props {
    view: GameView;
    banner: string | null;
    checkAlert: string | null;
    canMove: boolean;
    canPlace: boolean;
    canPromote: boolean;
    onUndo: () => void;
    gameOver: boolean;
    showPrevMove?: boolean;
  }

  let {
    view,
    banner,
    checkAlert,
    canMove,
    canPlace,
    canPromote,
    onUndo,
    gameOver,
    showPrevMove = $bindable(true),
  }: Props = $props();

  const toMoveLabel = $derived(
    view.to_move === 'P1'
      ? `${names.side_symbols.P1} ${names.players.P1}`
      : `${names.side_symbols.P2} ${names.players.P2}`
  );

  const resultLabel = $derived(
    !view.result
      ? null
      : 'Win' in view.result
        ? `${view.result.Win === 'P1' ? names.side_symbols.P1 : names.side_symbols.P2} ${view.result.Win === 'P1' ? names.players.P1 : names.players.P2} wins!`
        : 'Draw' in view.result
          ? `Draw: ${names.draw_reasons[view.result.Draw]}`
          : null
  );
</script>

<aside class="sidebar">
  <div class="section">
    <div class="turn-label">
      {#if gameOver}
        <span class="game-over">{resultLabel}</span>
      {:else}
        <span class="to-move">{toMoveLabel} to move</span>
        <span class="ap">AP remaining: <strong>{view.ap_remaining}</strong></span>
      {/if}
    </div>
  </div>

  {#if !gameOver}
    <div class="section actions">
      <div class="actions-title">Actions</div>
      <div class="action-row" class:available={canMove} class:unavailable={!canMove}>
        <span class="action-name">{names.action_labels.Move}</span>
        <span class="action-cost">{AP_MOVE} AP</span>
        <span class="action-status">{canMove ? '✓' : '✗'}</span>
      </div>
      <div class="action-row" class:available={canPlace} class:unavailable={!canPlace}>
        <span class="action-name">{names.action_labels.Place}</span>
        <span class="action-cost">{AP_PLACE} AP</span>
        <span class="action-status">{canPlace ? '✓' : '✗'}</span>
      </div>
      <div class="action-row" class:available={canPromote} class:unavailable={!canPromote}>
        <span class="action-name">{names.action_labels.Promote}</span>
        <span class="action-cost">{AP_PROMOTE} AP</span>
        <span class="action-status">{canPromote ? '✓' : '✗'}</span>
      </div>
    </div>
  {/if}

  <div class="section reserves">
    <div class="reserve-row">
      <span class="reserve-label">{names.side_symbols.P1} {names.players.P1} reserve:</span>
      <span class="reserve-count">{view.reserves[0]}</span>
    </div>
    <div class="reserve-row">
      <span class="reserve-label">{names.side_symbols.P2} {names.players.P2} reserve:</span>
      <span class="reserve-count">{view.reserves[1]}</span>
    </div>
  </div>

  {#if checkAlert}
    <div class="section check-alert" role="alert">
      {checkAlert}
    </div>
  {/if}

  {#if banner}
    <div class="section banner" role="alert">
      {banner}
    </div>
  {/if}

  <div class="section display-opts">
    <label class="opt-label">
      <input type="checkbox" bind:checked={showPrevMove} />
      {names.show_prev_move}
    </label>
  </div>

  {#if !gameOver}
    <button class="btn-undo" onclick={onUndo}>Undo</button>
  {/if}
</aside>

<style>
  .sidebar {
    display: flex;
    flex-direction: column;
    gap: 1rem;
    padding: 1rem;
    border: 1px solid var(--board-border);
    border-radius: 4px;
    background: #faf7f2;
    min-width: 180px;
    align-self: flex-start;
  }

  .section {
    display: flex;
    flex-direction: column;
    gap: 0.3rem;
  }

  .turn-label {
    display: flex;
    flex-direction: column;
    gap: 0.2rem;
  }

  .to-move {
    font-weight: 600;
    font-size: 1rem;
    color: var(--board-border);
  }

  .ap {
    font-size: 0.9rem;
    color: #555;
  }

  .game-over {
    font-weight: 700;
    font-size: 1.05rem;
    color: #b00;
  }

  /* Actions section */
  .actions {
    border-top: 1px solid #e0d8ce;
    padding-top: 0.5rem;
    gap: 0.25rem;
  }

  .actions-title {
    font-size: 0.78rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: #888;
    margin-bottom: 0.15rem;
  }

  .action-row {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    font-size: 0.85rem;
    padding: 0.15rem 0.3rem;
    border-radius: 3px;
  }

  .action-row.available {
    color: var(--board-border);
    font-weight: 600;
  }

  .action-row.unavailable {
    color: #bbb;
  }

  .action-name {
    flex: 1;
  }

  .action-cost {
    font-size: 0.78rem;
    color: inherit;
    opacity: 0.75;
  }

  .action-status {
    font-size: 0.9rem;
    min-width: 1ch;
    text-align: center;
  }

  .action-row.available .action-status {
    color: #2a7a2a;
  }

  .action-row.unavailable .action-status {
    color: #ccc;
  }

  /* Reserves */
  .reserves {
    gap: 0.25rem;
  }

  .reserve-row {
    display: flex;
    justify-content: space-between;
    gap: 0.5rem;
    font-size: 0.88rem;
    color: #444;
  }

  .reserve-count {
    font-weight: 600;
  }

  /* Check alert -- prominent, inside the box */
  .check-alert {
    background: #ffeaea;
    border: 2px solid var(--check);
    border-radius: 3px;
    padding: 0.45rem 0.6rem;
    font-size: 0.85rem;
    font-weight: 600;
    color: var(--check);
  }

  /* Turn-ended-on-check notice */
  .banner {
    background: #fff3cd;
    border: 1px solid #ffc107;
    border-radius: 3px;
    padding: 0.5rem 0.6rem;
    font-size: 0.85rem;
    color: #664d03;
  }

  .display-opts {
    border-top: 1px solid #e0d8ce;
    padding-top: 0.5rem;
  }

  .opt-label {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    font-size: 0.85rem;
    color: #444;
    cursor: pointer;
    user-select: none;
  }

  .btn-undo {
    padding: 0.4rem 1rem;
    background: #6c757d;
    color: #fff;
    border: none;
    border-radius: 3px;
    font-size: 0.88rem;
    cursor: pointer;
    transition: opacity 0.15s;
  }

  .btn-undo:hover {
    opacity: 0.85;
  }
</style>
