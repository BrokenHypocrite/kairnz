<!--
  MoveHistory.svelte -- Scrollable panel listing every move in notation order.
  Groups moves by ply, showing player label and notation text.
  Legend explains notation symbols.
-->
<script lang="ts">
  import type { Player } from '../lib/types.js';
  import { names } from '../lib/names.js';

  interface HistoryEntry {
    ply: number;
    player: Player;
    text: string;
  }

  interface Props {
    history: HistoryEntry[];
  }

  let { history }: Props = $props();
</script>

<aside class="move-history">
  <h2 class="panel-title">History</h2>

  <div class="legend">
    <span title="Stone">S stone</span>
    <span title="Pillar">P pillar</span>
    <span title="Spire">T spire</span>
    <span title="Keystone">K keystone</span>
    <span title="move">- move</span>
    <span title="capture">x capture</span>
    <span title="place">+ place</span>
    <span title="check end">+ check</span>
    <span title="stack">^ stack</span>
    <span title="game over"># win</span>
  </div>

  <div class="entries" role="log" aria-label="Move history">
    {#if history.length === 0}
      <p class="empty">No moves yet.</p>
    {:else}
      {#each history as entry (entry.ply)}
        <div class="entry" class:p1={entry.player === 'P1'} class:p2={entry.player === 'P2'}>
          <span class="ply">{entry.ply}.</span>
          <span class="player-label">{entry.player === 'P1' ? names.side_symbols.P1 : names.side_symbols.P2}</span>
          <span class="notation">{entry.text}</span>
        </div>
      {/each}
    {/if}
  </div>
</aside>

<style>
  .move-history {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
    padding: 1rem;
    border: 1px solid var(--board-border);
    border-radius: 4px;
    background: #faf7f2;
    min-width: 180px;
    max-width: 220px;
    align-self: flex-start;
  }

  .panel-title {
    margin: 0 0 0.2rem;
    font-size: 1rem;
    font-weight: 600;
    color: var(--board-border);
  }

  .legend {
    display: flex;
    flex-wrap: wrap;
    gap: 0.3rem 0.6rem;
    font-size: 0.72rem;
    color: #666;
    border-bottom: 1px solid #e0d8ce;
    padding-bottom: 0.4rem;
  }

  .entries {
    display: flex;
    flex-direction: column;
    gap: 0.15rem;
    max-height: 420px;
    overflow-y: auto;
  }

  .empty {
    font-size: 0.82rem;
    color: #888;
    margin: 0;
  }

  .entry {
    display: flex;
    align-items: baseline;
    gap: 0.3rem;
    font-size: 0.85rem;
    padding: 0.1rem 0.3rem;
    border-radius: 2px;
  }

  .entry.p1 {
    background: color-mix(in srgb, var(--piece-p1) 25%, transparent);
  }

  .entry.p2 {
    background: color-mix(in srgb, var(--piece-p2) 20%, transparent);
  }

  .ply {
    color: #888;
    font-size: 0.75rem;
    min-width: 1.6rem;
  }

  .player-label {
    font-weight: 600;
    font-size: 0.75rem;
    min-width: 2.2rem;
    color: var(--board-border);
  }

  .notation {
    font-family: monospace;
    font-size: 0.88rem;
  }
</style>
