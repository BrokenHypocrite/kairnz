<!--
  ConfigPanel.svelte -- Rule configuration and new-game launcher.

  Binds to a RuleConfig; emits "newgame" when the user clicks "New Game".
  Labels come from names.ts where applicable; other labels match Rust field names.
-->
<script lang="ts">
  import type { RuleConfig, SpireMode } from '../lib/types.js';

  interface Props {
    config: RuleConfig;
    onNewGame: (cfg: RuleConfig) => void;
    disabled?: boolean;
  }

  let { config = $bindable(), onNewGame, disabled = false }: Props = $props();

  const SPIRE_OPTIONS: SpireMode[] = ['Dragon', 'Queen'];

  function handleNewGame() {
    onNewGame({ ...config });
  }
</script>

<section class="config-panel">
  <h2 class="panel-title">Rules</h2>

  <div class="field">
    <label for="spire-mode">Spire mode</label>
    <select id="spire-mode" bind:value={config.spire} {disabled}>
      {#each SPIRE_OPTIONS as mode}
        <option value={mode}>{mode}</option>
      {/each}
    </select>
  </div>

  <div class="field">
    <label for="first-turn-ap">First-turn AP</label>
    <input
      id="first-turn-ap"
      type="number"
      min="1"
      max="4"
      bind:value={config.first_turn_ap}
      {disabled}
    />
  </div>

  <div class="field field-check">
    <input
      id="capture-lock"
      type="checkbox"
      bind:checked={config.capture_lock}
      {disabled}
    />
    <label for="capture-lock">Capture lock</label>
  </div>

  <div class="field field-check">
    <input
      id="keystone-single-move"
      type="checkbox"
      bind:checked={config.keystone_single_move}
      {disabled}
    />
    <label for="keystone-single-move">Keystone single move</label>
  </div>

  <button class="btn-new-game" onclick={handleNewGame} {disabled}>
    New Game
  </button>
</section>

<style>
  .config-panel {
    display: flex;
    flex-direction: column;
    gap: 0.6rem;
    padding: 1rem;
    border: 1px solid var(--board-border);
    border-radius: 4px;
    background: #faf7f2;
    min-width: 180px;
  }

  .panel-title {
    margin: 0 0 0.4rem;
    font-size: 1rem;
    font-weight: 600;
    color: var(--board-border);
  }

  .field {
    display: flex;
    flex-direction: column;
    gap: 0.2rem;
  }

  .field-check {
    flex-direction: row;
    align-items: center;
    gap: 0.4rem;
  }

  label {
    font-size: 0.85rem;
    color: #444;
  }

  select,
  input[type='number'] {
    padding: 0.25rem 0.4rem;
    border: 1px solid #ccc;
    border-radius: 3px;
    font-size: 0.9rem;
    background: #fff;
  }

  input[type='number'] {
    width: 4rem;
  }

  .btn-new-game {
    margin-top: 0.4rem;
    padding: 0.45rem 1rem;
    background: var(--board-border);
    color: #fff;
    border: none;
    border-radius: 3px;
    font-size: 0.9rem;
    cursor: pointer;
    transition: opacity 0.15s;
  }

  .btn-new-game:hover:not(:disabled) {
    opacity: 0.85;
  }

  .btn-new-game:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
</style>
