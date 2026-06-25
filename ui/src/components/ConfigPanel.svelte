<!--
  ConfigPanel.svelte -- Rule configuration, AI opponent settings, and new-game launcher.

  Binds to a RuleConfig and four AI state values; emits "newgame" when the user
  clicks "New Game". The panel is collapsible (default collapsed). The "New Game"
  button is always visible in the header area so a new game can be started without
  expanding. Labels come from names.ts where applicable; other labels match Rust
  field names.
-->
<script lang="ts">
  import type { Player, RuleConfig, SpireMode } from '../lib/types.js';
  import { names } from '../lib/names.js';

  /** Difficulty label to simulation-count mapping. */
  const DIFFICULTY_OPTIONS: { label: string; sims: number }[] = [
    { label: 'Easy', sims: 100 },
    { label: 'Medium', sims: 400 },
    { label: 'Hard', sims: 1600 },
  ];

  interface Props {
    config: RuleConfig;
    onNewGame: (cfg: RuleConfig) => void;
    disabled?: boolean;
    aiEnabled: boolean;
    aiSide: Player;
    aiSims: number;
    aiModel: string;
  }

  let {
    config = $bindable(),
    onNewGame,
    disabled = false,
    aiEnabled = $bindable(),
    aiSide = $bindable(),
    aiSims = $bindable(),
    aiModel = $bindable(),
  }: Props = $props();

  /** Whether the rule controls are expanded. Defaults to collapsed. */
  let expanded = $state(false);

  const SPIRE_OPTIONS: SpireMode[] = ['Dragon', 'Queen'];

  function handleNewGame() {
    onNewGame({ ...config });
  }

  function toggleExpanded() {
    expanded = !expanded;
  }
</script>

<section class="config-panel">
  <div class="panel-header">
    <button
      class="header-toggle"
      onclick={toggleExpanded}
      aria-expanded={expanded}
      aria-controls="config-controls"
    >
      <span class="chevron" class:chevron-down={expanded}>&#9654;</span>
      <span class="panel-title">{names.rules_title}</span>
    </button>
    <button class="btn-new-game" onclick={handleNewGame} {disabled}>
      New Game
    </button>
  </div>

  {#if expanded}
    <div id="config-controls" class="controls">
      <div class="field">
        <label for="spire-mode">Spire mode</label>
        <select id="spire-mode" bind:value={config.spire} {disabled}>
          {#each SPIRE_OPTIONS as mode}
            <option value={mode}>{names.spire_modes[mode]}</option>
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

      <div class="section-divider"></div>

      <div class="section-label">AI Opponent</div>

      <div class="field field-check">
        <input
          id="ai-enabled"
          type="checkbox"
          bind:checked={aiEnabled}
          {disabled}
        />
        <label for="ai-enabled">Play vs AI</label>
      </div>

      {#if aiEnabled}
        <fieldset class="field field-radio">
          <legend class="field-legend">AI plays</legend>
          <div class="radio-group">
            <label class="radio-label">
              <input
                type="radio"
                name="ai-side"
                value="P1"
                bind:group={aiSide}
                {disabled}
              />
              P1
            </label>
            <label class="radio-label">
              <input
                type="radio"
                name="ai-side"
                value="P2"
                bind:group={aiSide}
                {disabled}
              />
              P2
            </label>
          </div>
        </fieldset>

        <div class="field">
          <label for="ai-difficulty">Difficulty</label>
          <select id="ai-difficulty" bind:value={aiSims} {disabled}>
            {#each DIFFICULTY_OPTIONS as opt}
              <option value={opt.sims}>{opt.label}</option>
            {/each}
          </select>
        </div>

        <div class="field">
          <label for="ai-model">Model path</label>
          <input
            id="ai-model"
            type="text"
            bind:value={aiModel}
            placeholder="models/best.onnx"
            {disabled}
          />
        </div>
      {/if}
    </div>
  {/if}
</section>

<style>
  .config-panel {
    display: flex;
    flex-direction: column;
    gap: 0;
    padding: 0;
    border: 1px solid var(--board-border);
    border-radius: 4px;
    background: var(--panel-bg);
    min-width: 180px;
  }

  .panel-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 0.5rem;
    padding: 0.6rem 1rem;
  }

  .header-toggle {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    background: none;
    border: none;
    padding: 0;
    cursor: pointer;
    color: var(--board-border);
    font: inherit;
    flex: 1;
    text-align: left;
  }

  .header-toggle:focus-visible {
    outline: 2px solid var(--board-border);
    outline-offset: 2px;
    border-radius: 2px;
  }

  .chevron {
    display: inline-block;
    font-size: 0.65rem;
    color: var(--board-border);
    transition: transform 0.18s ease;
    transform: rotate(0deg);
  }

  .chevron-down {
    transform: rotate(90deg);
  }

  .panel-title {
    font-size: 1rem;
    font-weight: 600;
    color: var(--board-border);
  }

  .controls {
    display: flex;
    flex-direction: column;
    gap: 0.6rem;
    padding: 0 1rem 1rem;
    border-top: 1px solid var(--panel-divider);
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
    color: var(--text-muted);
  }

  select,
  input[type='number'] {
    padding: 0.25rem 0.4rem;
    border: 1px solid var(--input-border);
    border-radius: 3px;
    font-size: 0.9rem;
    background: var(--input-bg);
  }

  input[type='number'] {
    width: 4rem;
  }

  .btn-new-game {
    padding: 0.35rem 0.75rem;
    background: var(--board-border);
    color: var(--input-bg);
    border: none;
    border-radius: 3px;
    font-size: 0.85rem;
    cursor: pointer;
    transition: opacity 0.15s;
    white-space: nowrap;
  }

  .btn-new-game:hover:not(:disabled) {
    opacity: 0.85;
  }

  .btn-new-game:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .section-divider {
    height: 1px;
    background: var(--panel-divider);
    margin: 0.2rem 0;
  }

  .section-label {
    font-size: 0.8rem;
    font-weight: 600;
    color: var(--board-border);
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }

  .field-radio {
    border: none;
    padding: 0;
    margin: 0;
  }

  .field-legend {
    font-size: 0.85rem;
    color: var(--text-muted);
    padding: 0;
    margin-bottom: 0.2rem;
  }

  .radio-group {
    display: flex;
    gap: 1rem;
  }

  .radio-label {
    display: flex;
    align-items: center;
    gap: 0.25rem;
    font-size: 0.85rem;
    color: var(--text-muted);
    cursor: pointer;
  }

  input[type='text'] {
    padding: 0.25rem 0.4rem;
    border: 1px solid var(--input-border);
    border-radius: 3px;
    font-size: 0.85rem;
    background: var(--input-bg);
    width: 100%;
    box-sizing: border-box;
  }
</style>
