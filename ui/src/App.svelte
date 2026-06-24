<script lang="ts">
  import { onMount } from 'svelte';
  import { newGame } from './lib/api.js';
  import { defaultConfig } from './lib/types.js';
  import { names } from './lib/names.js';
  import type { GameView } from './lib/types.js';
  import Board from './components/Board.svelte';

  let view = $state<GameView | null>(null);
  let error = $state<string | null>(null);

  onMount(async () => {
    try {
      const [_id, initialView] = await newGame(defaultConfig);
      view = initialView;
    } catch (e) {
      error = String(e);
    }
  });
</script>

<main>
  <h1>{names.game}</h1>
  {#if error}
    <p class="error">Error: {error}</p>
  {:else if view}
    <div class="status-bar">
      <span>To move: <strong>{view.to_move}</strong></span>
      <span>AP remaining: <strong>{view.ap_remaining}</strong></span>
      <span>Reserves: P1 {view.reserves[0]} &middot; P2 {view.reserves[1]}</span>
    </div>
    <Board {view} />
  {:else}
    <p>Loading...</p>
  {/if}
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
  }

  main {
    font-family: sans-serif;
    padding: 2rem;
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    gap: 1rem;
  }

  h1 {
    margin: 0;
  }

  .status-bar {
    display: flex;
    gap: 1.5rem;
    font-size: 0.95rem;
    color: #333;
  }

  .error {
    color: red;
  }
</style>
