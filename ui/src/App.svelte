<script lang="ts">
  import { onMount } from 'svelte';
  import { newGame } from './lib/api.js';
  import { defaultConfig } from './lib/types.js';
  import { names } from './lib/names.js';
  import type { GameView } from './lib/types.js';

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
    <p>To move: {view.to_move}</p>
    <p>AP remaining: {view.ap_remaining}</p>
  {:else}
    <p>Loading...</p>
  {/if}
</main>

<style>
  main {
    font-family: sans-serif;
    padding: 2rem;
  }

  .error {
    color: red;
  }
</style>
