<!--
  Board.svelte -- Renders a 9x9 SVG board for Cairn.

  Layout:
    - Board index i maps to file = i % 9, rank = Math.floor(i / 9).
    - Rank 0 (P1's back rank) is rendered at the BOTTOM of the SVG; rank 8 at top.
      This is the standard board convention (P1 nearest the viewer).
      Achieved by flipping the Y axis: svgY = (8 - rank) * cellSize.
    - Each cell is a square of `cellSize` SVG units.
    - Pieces are centered in their cell via an SVG translate.

  Colors/theming are CSS custom properties (no hardcoded hex in attributes).
-->
<script lang="ts">
  import type { GameView } from '../lib/types.js';
  import Piece from './Piece.svelte';

  interface Props {
    view: GameView;
  }

  let { view }: Props = $props();

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
    <!-- Checkered grid cells -->
    {#each squares as i}
      {@const pos = cellPos(i)}
      <rect
        x={pos.x}
        y={pos.y}
        width={CELL}
        height={CELL}
        class={isLight(i) ? 'cell cell-light' : 'cell cell-dark'}
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

    <!-- Pieces centered in their cells -->
    {#each occupied as { i, piece }}
      {@const pos = cellPos(i)}
      {@const cx = pos.x + CELL / 2}
      {@const cy = pos.y + CELL / 2}
      <g transform="translate({cx} {cy})">
        <Piece {piece} cellSize={CELL} />
      </g>
    {/each}
  </svg>
</div>

<style>
  :root {
    --board-light: #f0d9b5;
    --board-dark: #b58863;
    --board-border: #5a3e28;
    --grid-line: #5a3e2855;
    --piece-p1: #e8d5a3;
    --piece-p2: #8b4513;
    --piece-stroke: #2a2a2a;
  }

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
  }

  .cell-light {
    fill: var(--board-light);
  }

  .cell-dark {
    fill: var(--board-dark);
  }

  .grid-line {
    stroke: var(--grid-line);
    stroke-width: 0.5;
  }
</style>
