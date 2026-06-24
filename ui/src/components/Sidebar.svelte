<!--
  Sidebar.svelte -- Game status, player info, banners, and undo.

  Shows: whose turn, AP remaining, both reserves, check-auto-end banner,
  game-over message. Undo button calls the provided handler.
-->
<script lang="ts">
  import type { GameView } from '../lib/types.js';
  import { names } from '../lib/names.js';

  interface Props {
    view: GameView;
    banner: string | null;
    onUndo: () => void;
    gameOver: boolean;
  }

  let { view, banner, onUndo, gameOver }: Props = $props();

  const toMoveLabel = $derived(
    view.to_move === 'P1' ? names.players.P1 : names.players.P2
  );

  const resultLabel = $derived(
    !view.result
      ? null
      : 'Win' in view.result
        ? `${view.result.Win === 'P1' ? names.players.P1 : names.players.P2} wins!`
        : 'Draw' in view.result
          ? `Draw: ${view.result.Draw === 'MaxPlies' ? 'move limit reached' : 'repetition'}`
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

  <div class="section reserves">
    <div class="reserve-row">
      <span class="reserve-label">{names.players.P1} reserve:</span>
      <span class="reserve-count">{view.reserves[0]}</span>
    </div>
    <div class="reserve-row">
      <span class="reserve-label">{names.players.P2} reserve:</span>
      <span class="reserve-count">{view.reserves[1]}</span>
    </div>
  </div>

  {#if banner}
    <div class="section banner" role="alert">
      {banner}
    </div>
  {/if}

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

  .banner {
    background: #fff3cd;
    border: 1px solid #ffc107;
    border-radius: 3px;
    padding: 0.5rem 0.6rem;
    font-size: 0.85rem;
    color: #664d03;
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
