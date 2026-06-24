<!--
  Board.svelte -- Renders a 9x9 SVG board for Cairn.

  Interaction:
    - Left-click: onSquareClick(sq) for selection/move
    - Middle-click: onInspect(sq) for preview of any piece's geometric moves
    - Right-click: onContext(sq) for place/promote prompt (preventDefault always called)
    - onPromptConfirm / onPromptCancel: prompt popover callbacks

  Colors/theming are CSS custom properties (no hardcoded hex in attributes).
-->
<script lang="ts">
  import type { GameView, Sq } from '../lib/types.js';
  import Piece from './Piece.svelte';
  import { names } from '../lib/names.js';

  interface Props {
    view: GameView;
    selectedSq?: Sq | null;
    legalTargets?: Sq[];
    stackable?: Sq[];
    inspectTargets?: Sq[];
    prompt?: { kind: 'place' | 'promote'; sq: number } | null;
    checkedKeystones?: number[];
    onSquareClick?: (sq: Sq) => void;
    onInspect?: (sq: Sq) => void;
    onContext?: (sq: Sq) => void;
    onPromptConfirm?: () => void;
    onPromptCancel?: () => void;
  }

  let {
    view,
    selectedSq = null,
    legalTargets = [],
    stackable = [],
    inspectTargets = [],
    prompt = null,
    checkedKeystones = [],
    onSquareClick,
    onInspect,
    onContext,
    onPromptConfirm,
    onPromptCancel,
  }: Props = $props();

  const GRID = 9;
  const CELL = 60;
  const BOARD_SIZE = GRID * CELL;
  const LABEL_MARGIN = 20;

  const files = Array.from({ length: GRID }, (_, i) => i);
  const ranks = Array.from({ length: GRID }, (_, i) => i);
  const squares = Array.from({ length: GRID * GRID }, (_, i) => i);

  function cellPos(i: number): { x: number; y: number } {
    const file = i % GRID;
    const rank = Math.floor(i / GRID);
    return { x: file * CELL, y: (GRID - 1 - rank) * CELL };
  }

  function isLight(i: number): boolean {
    const file = i % GRID;
    const rank = Math.floor(i / GRID);
    return (file + rank) % 2 === 0;
  }

  const occupied = $derived(
    squares
      .map((i) => ({ i, piece: view.board[i] }))
      .filter((s): s is { i: number; piece: NonNullable<(typeof view.board)[number]> } =>
        s.piece !== null
      )
  );

  const legalSet = $derived(new Set(legalTargets));
  const stackableSet = $derived(new Set(stackable));
  const inspectSet = $derived(new Set(inspectTargets));
  const checkSet = $derived(new Set(checkedKeystones));

  /** SVG coordinate of the prompt popover anchor (top-left of cell). */
  const promptPos = $derived(
    prompt !== null ? cellPos(prompt.sq) : null
  );

  /** Prompt display text. */
  const promptText = $derived(
    prompt?.kind === 'promote' ? names.prompt_promote : names.prompt_place
  );

  function handleCellClick(sq: Sq) {
    onSquareClick?.(sq);
  }

  function handleCellContext(e: MouseEvent, sq: Sq) {
    e.preventDefault();
    onContext?.(sq);
  }

  function handleCellAuxClick(e: MouseEvent, sq: Sq) {
    if (e.button === 1) {
      onInspect?.(sq);
    }
  }

  function handleCellMouseDown(e: MouseEvent, _sq: Sq) {
    if (e.button === 1) {
      e.preventDefault();
    }
  }
</script>

<div class="board-wrapper">
  <svg
    width={BOARD_SIZE + LABEL_MARGIN}
    height={BOARD_SIZE + LABEL_MARGIN}
    viewBox="{-LABEL_MARGIN} 0 {BOARD_SIZE + LABEL_MARGIN} {BOARD_SIZE + LABEL_MARGIN}"
    class="board-svg"
    role="img"
    aria-label="Cairn game board"
  >
    <!-- Checkered grid cells -->
    {#each squares as i}
      {@const pos = cellPos(i)}
      {@const isSelected = selectedSq === i}
      {@const isTarget = legalSet.has(i)}
      <!-- svelte-ignore a11y_click_events_have_key_events -->
      <!-- svelte-ignore a11y_interactive_supports_focus -->
      <rect
        x={pos.x}
        y={pos.y}
        width={CELL}
        height={CELL}
        class={isLight(i) ? 'cell cell-light' : 'cell cell-dark'}
        class:cell-selected={isSelected}
        class:cell-target={isTarget}
        role="button"
        onclick={() => handleCellClick(i)}
        oncontextmenu={(e) => handleCellContext(e, i)}
        onauxclick={(e) => handleCellAuxClick(e, i)}
        onmousedown={(e) => handleCellMouseDown(e, i)}
      />
    {/each}

    <!-- Grid lines -->
    {#each Array.from({ length: GRID + 1 }, (_, k) => k) as k}
      <line x1={k * CELL} y1={0} x2={k * CELL} y2={BOARD_SIZE} class="grid-line" />
      <line x1={0} y1={k * CELL} x2={BOARD_SIZE} y2={k * CELL} class="grid-line" />
    {/each}

    <!-- Stackable-square ring indicators -->
    {#each squares as i}
      {#if stackableSet.has(i)}
        {@const pos = cellPos(i)}
        <rect
          x={pos.x + 3} y={pos.y + 3}
          width={CELL - 6} height={CELL - 6}
          class="stack-ring"
          pointer-events="none"
        />
      {/if}
    {/each}

    <!-- Check highlight rings -->
    {#each squares as i}
      {#if checkSet.has(i)}
        {@const pos = cellPos(i)}
        <rect
          x={pos.x + 2} y={pos.y + 2}
          width={CELL - 4} height={CELL - 4}
          class="check-ring"
          pointer-events="none"
        />
      {/if}
    {/each}

    <!-- Actionable move-target dots for empty squares (left-click selection) -->
    {#each squares as i}
      {#if legalSet.has(i) && view.board[i] === null}
        {@const pos = cellPos(i)}
        <circle
          cx={pos.x + CELL / 2}
          cy={pos.y + CELL / 2}
          r={CELL * 0.14}
          class="move-dot"
          pointer-events="none"
        />
      {/if}
    {/each}

    <!-- Inspect preview dots for empty squares (middle-click, read-only, visually distinct) -->
    {#each squares as i}
      {#if inspectSet.has(i) && view.board[i] === null}
        {@const pos = cellPos(i)}
        <circle
          cx={pos.x + CELL / 2}
          cy={pos.y + CELL / 2}
          r={CELL * 0.14}
          class="inspect-dot"
          pointer-events="none"
        />
      {/if}
    {/each}

    <!-- Pieces -->
    {#each occupied as { i, piece }}
      {@const pos = cellPos(i)}
      {@const cx = pos.x + CELL / 2}
      {@const cy = pos.y + CELL / 2}
      <g
        transform="translate({cx} {cy})"
        style="cursor: pointer;"
        role="button"
        aria-label="piece at square {i}"
        onclick={() => handleCellClick(i)}
        oncontextmenu={(e) => handleCellContext(e, i)}
        onauxclick={(e) => handleCellAuxClick(e, i)}
        onmousedown={(e) => handleCellMouseDown(e, i)}
        onkeydown={(e) => { if (e.key === 'Enter' || e.key === ' ') handleCellClick(i); }}
        tabindex="0"
      >
        <Piece {piece} cellSize={CELL} />
      </g>
    {/each}

    <!-- Capture rings for occupied actionable targets -- rendered above pieces -->
    {#each squares as i}
      {#if legalSet.has(i) && view.board[i] !== null}
        {@const pos = cellPos(i)}
        <circle
          cx={pos.x + CELL / 2}
          cy={pos.y + CELL / 2}
          r={CELL * 0.44}
          class="capture-ring"
          pointer-events="none"
        />
      {/if}
    {/each}

    <!-- Capture rings for occupied inspect targets -- rendered above pieces -->
    {#each squares as i}
      {#if inspectSet.has(i) && view.board[i] !== null}
        {@const pos = cellPos(i)}
        <circle
          cx={pos.x + CELL / 2}
          cy={pos.y + CELL / 2}
          r={CELL * 0.44}
          class="inspect-capture-ring"
          pointer-events="none"
        />
      {/if}
    {/each}

    <!-- File labels -->
    {#each files as file}
      <text
        x={file * CELL + CELL / 2}
        y={BOARD_SIZE + LABEL_MARGIN * 0.7}
        class="coord-label"
        text-anchor="middle"
        dominant-baseline="middle"
      >{String.fromCharCode(97 + file)}</text>
    {/each}

    <!-- Rank labels -->
    {#each ranks as rank}
      <text
        x={-LABEL_MARGIN * 0.65}
        y={(GRID - 1 - rank) * CELL + CELL / 2 + 5}
        class="coord-label"
        text-anchor="middle"
        dominant-baseline="middle"
      >{rank + 1}</text>
    {/each}

    <!-- Right-click prompt popover (rendered in SVG foreignObject) -->
    {#if prompt !== null && promptPos !== null}
      <foreignObject
        x={promptPos.x}
        y={promptPos.y - 54}
        width="120"
        height="50"
      >
        <div class="prompt-box" xmlns="http://www.w3.org/1999/xhtml">
          <span class="prompt-text">{promptText}</span>
          <div class="prompt-buttons">
            <button class="prompt-yes" onclick={onPromptConfirm}>Yes</button>
            <button class="prompt-no" onclick={onPromptCancel}>No</button>
          </div>
        </div>
      </foreignObject>
    {/if}
  </svg>
</div>

<style>
  .board-wrapper {
    display: inline-block;
    border: 3px solid var(--board-border);
    border-radius: 2px;
    box-shadow: 0 4px 16px #0004;
  }

  .board-svg {
    display: block;
    overflow: visible;
  }

  .cell {
    stroke: none;
    cursor: pointer;
  }

  .cell-light { fill: var(--board-light); }
  .cell-dark { fill: var(--board-dark); }
  .cell-selected { fill: #aef2a8 !important; }
  .cell-target { fill: #d4f5d0 !important; }

  .grid-line {
    stroke: var(--grid-line);
    stroke-width: 0.5;
  }

  .move-dot {
    fill: #1b7a1b;
    opacity: 0.7;
  }

  .inspect-dot {
    fill: none;
    stroke: var(--inspect-dot, #7c3aed);
    stroke-width: 2.5;
    opacity: 0.85;
  }

  /* Capture ring: stroked circle framing the cell, no fill, rendered above the piece. */
  .capture-ring {
    fill: none;
    stroke: #1b7a1b;
    stroke-width: 3.5;
    opacity: 0.85;
  }

  /* Inspect capture ring: same shape, purple to match inspect-dot color. */
  .inspect-capture-ring {
    fill: none;
    stroke: var(--inspect-dot, #7c3aed);
    stroke-width: 3;
    opacity: 0.85;
  }

  .stack-ring {
    fill: none;
    stroke: #cc8800;
    stroke-width: 2.5;
    stroke-dasharray: 6 3;
    rx: 2;
  }

  .check-ring {
    fill: none;
    stroke: var(--check, #cc2200);
    stroke-width: 3;
    rx: 2;
  }

  .coord-label {
    fill: var(--coord);
    font-size: 11px;
    font-family: sans-serif;
    pointer-events: none;
    user-select: none;
  }

  .prompt-box {
    background: #fffbeb;
    border: 1.5px solid #92400e;
    border-radius: 4px;
    padding: 4px 6px;
    font-family: sans-serif;
    font-size: 12px;
    display: flex;
    flex-direction: column;
    gap: 3px;
    box-shadow: 0 2px 6px #0003;
    width: 110px;
  }

  .prompt-text {
    font-weight: 600;
    color: #451a03;
  }

  .prompt-buttons {
    display: flex;
    gap: 4px;
  }

  .prompt-yes {
    background: #16a34a;
    color: #fff;
    border: none;
    border-radius: 3px;
    padding: 2px 8px;
    font-size: 11px;
    cursor: pointer;
  }

  .prompt-no {
    background: #6b7280;
    color: #fff;
    border: none;
    border-radius: 3px;
    padding: 2px 8px;
    font-size: 11px;
    cursor: pointer;
  }
</style>
