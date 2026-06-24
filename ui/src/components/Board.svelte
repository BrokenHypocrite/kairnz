<!--
  Board.svelte -- Renders a 9x9 SVG board for Cairn.

  Layout:
    - Board index i maps to file = i % 9, rank = Math.floor(i / 9).
    - Rank 0 (P1's back rank) is rendered at the BOTTOM of the SVG; rank 8 at top.
      This is the standard board convention (P1 nearest the viewer).
      Achieved by flipping the Y axis: svgY = (8 - rank) * cellSize.
    - Each cell is a square of `cellSize` SVG units.
    - Pieces are centered in their cell via an SVG translate.

  Interaction:
    - `onSquareClick(sq)` fires when the user clicks any square.
    - `selectedSq` highlights the selected square.
    - `legalTargets` renders move-target dots on those squares.
    - `stackable` shows a ring on squares the current player can stack.
    - `placeTargets` shows subtle dots for Place destinations when pendingPlace is active.

  Colors/theming are CSS custom properties (no hardcoded hex in attributes).
-->
<script lang="ts">
  import type { GameView, Sq } from '../lib/types.js';
  import Piece from './Piece.svelte';

  interface Props {
    view: GameView;
    selectedSq?: Sq | null;
    legalTargets?: Sq[];
    stackable?: Sq[];
    placeTargets?: Sq[];
    pendingPlace?: boolean;
    onSquareClick?: (sq: Sq) => void;
  }

  let {
    view,
    selectedSq = null,
    legalTargets = [],
    stackable = [],
    placeTargets = [],
    pendingPlace = false,
    onSquareClick,
  }: Props = $props();

  /** Grid dimensions. */
  const GRID = 9;
  const CELL = 60;

  /** Total SVG canvas size. */
  const BOARD_SIZE = GRID * CELL;

  /** Indices for all 81 squares. */
  const squares = Array.from({ length: GRID * GRID }, (_, i) => i);

  /**
   * Maps a board index to the SVG cell top-left corner.
   * Rank 0 is at the bottom (svgY = (GRID-1)*CELL), rank 8 at top (svgY = 0).
   */
  function cellPos(i: number): { x: number; y: number } {
    const file = i % GRID;
    const rank = Math.floor(i / GRID);
    return {
      x: file * CELL,
      y: (GRID - 1 - rank) * CELL,
    };
  }

  /** True for light squares (standard checkerboard). */
  function isLight(i: number): boolean {
    const file = i % GRID;
    const rank = Math.floor(i / GRID);
    return (file + rank) % 2 === 0;
  }

  /** Occupied squares as {index, piece} pairs. */
  const occupied = $derived(
    squares
      .map((i) => ({ i, piece: view.board[i] }))
      .filter((s): s is { i: number; piece: NonNullable<(typeof view.board)[number]> } =>
        s.piece !== null
      )
  );

  const legalSet = $derived(new Set(legalTargets));
  const stackableSet = $derived(new Set(stackable));
  const placeSet = $derived(new Set(placeTargets));

  function handleCellClick(sq: Sq) {
    onSquareClick?.(sq);
  }
</script>

<div class="board-wrapper">
  <svg
    width={BOARD_SIZE}
    height={BOARD_SIZE}
    viewBox="0 0 {BOARD_SIZE} {BOARD_SIZE}"
    class="board-svg"
    role="img"
    aria-label="Cairn game board"
  >
    <!-- Checkered grid cells (clickable) -->
    {#each squares as i}
      {@const pos = cellPos(i)}
      {@const isSelected = selectedSq === i}
      {@const isTarget = legalSet.has(i)}
      {@const isPlaceDst = pendingPlace && placeSet.has(i)}
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
        class:cell-place={isPlaceDst}
        role="button"
        onclick={() => handleCellClick(i)}
      />
    {/each}

    <!-- Grid border lines (column and row) -->
    {#each Array.from({ length: GRID + 1 }, (_, k) => k) as k}
      <line
        x1={k * CELL}
        y1={0}
        x2={k * CELL}
        y2={BOARD_SIZE}
        class="grid-line"
      />
      <line
        x1={0}
        y1={k * CELL}
        x2={BOARD_SIZE}
        y2={k * CELL}
        class="grid-line"
      />
    {/each}

    <!-- Stackable-square ring indicators -->
    {#each squares as i}
      {#if stackableSet.has(i)}
        {@const pos = cellPos(i)}
        <rect
          x={pos.x + 3}
          y={pos.y + 3}
          width={CELL - 6}
          height={CELL - 6}
          class="stack-ring"
          pointer-events="none"
        />
      {/if}
    {/each}

    <!-- Legal-move target dots -->
    {#each squares as i}
      {#if legalSet.has(i)}
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

    <!-- Place-target dots (shown when pendingPlace) -->
    {#each squares as i}
      {#if pendingPlace && placeSet.has(i)}
        {@const pos = cellPos(i)}
        <circle
          cx={pos.x + CELL / 2}
          cy={pos.y + CELL / 2}
          r={CELL * 0.14}
          class="place-dot"
          pointer-events="none"
        />
      {/if}
    {/each}

    <!-- Pieces centered in their cells -->
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
        onkeydown={(e) => { if (e.key === 'Enter' || e.key === ' ') handleCellClick(i); }}
        tabindex="0"
      >
        <Piece {piece} cellSize={CELL} />
      </g>
    {/each}
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
  }

  .cell {
    stroke: none;
    cursor: pointer;
  }

  .cell-light {
    fill: var(--board-light);
  }

  .cell-dark {
    fill: var(--board-dark);
  }

  .cell-selected {
    fill: #aef2a8 !important;
  }

  .cell-target {
    fill: #d4f5d0 !important;
  }

  .cell-place {
    fill: #cce5ff !important;
  }

  .grid-line {
    stroke: var(--grid-line);
    stroke-width: 0.5;
  }

  .move-dot {
    fill: #1b7a1b;
    opacity: 0.7;
  }

  .place-dot {
    fill: #0066cc;
    opacity: 0.7;
  }

  .stack-ring {
    fill: none;
    stroke: #cc8800;
    stroke-width: 2.5;
    stroke-dasharray: 6 3;
    rx: 2;
  }
</style>
