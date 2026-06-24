<!--
  Piece.svelte -- Renders a single board piece as an SVG glyph.

  Ownership is encoded PRIMARILY via orientation (Shogi-style):
    - P1 pieces point UPWARD (default orientation)
    - P2 pieces are rotated 180 degrees around the cell center (point DOWNWARD)

  Color is a SECONDARY cue (CSS custom properties --piece-p1 / --piece-p2).

  Stack height for Stones:
    - height 1 (Stone):  plain wedge with no tiers
    - height 2 (Pillar): wedge with one horizontal tier line
    - height 3 (Spire):  wedge with two horizontal tier lines

  Keystones have a DISTINCT silhouette (diamond with a notch at the "front"),
  but share the same directional rotation for P1/P2.

  All glyph paths are defined once and parameterized; nothing is copy-pasted.
-->
<script lang="ts">
  import type { PieceView } from '../lib/types.js';
  import { names } from '../lib/names.js';

  interface Props {
    /** The piece to render. */
    piece: PieceView;
    /** Size of the board cell in SVG units (default 60). */
    cellSize?: number;
  }

  let { piece, cellSize = 60 }: Props = $props();

  /**
   * Wedge (stone/pillar/spire) path:
   * A pentagon pointing UPWARD in a coordinate system centered on (0,0).
   * Wide at the base (bottom), narrowing to a blunt point at the top.
   *
   *         (0, -h)          <-- apex (front/top)
   *      (-w/3, -h*0.2)      <-- shoulder
   *  (-w/2, h/2)  (w/2, h/2) <-- base corners
   *
   * "Up" = negative Y in SVG coordinates.
   */
  const WEDGE_W = $derived(cellSize * 0.7);
  const WEDGE_H = $derived(cellSize * 0.75);

  function wedgePath(w: number, h: number): string {
    const halfW = w / 2;
    const top = -h * 0.55;       // apex: points upward
    const shoulder = -h * 0.1;   // shoulder indent
    const base = h * 0.45;       // base bottom
    const shoulderW = w * 0.3;   // shoulder width (narrow near top)
    return [
      `M 0 ${top}`,
      `L ${shoulderW} ${shoulder}`,
      `L ${halfW} ${base}`,
      `L ${-halfW} ${base}`,
      `L ${-shoulderW} ${shoulder}`,
      `Z`,
    ].join(' ');
  }

  /**
   * Keystone path: a downward-biased diamond with a rectangular notch
   * cut into the FRONT (top, pointing upward) to form a distinct silhouette.
   * Orientation is still rotated 180 for P2.
   */
  function keystonePath(w: number, h: number): string {
    const halfW = w / 2;
    const top = -h * 0.55;
    const base = h * 0.45;
    const notchW = w * 0.18;
    const notchD = h * 0.18;  // notch depth from apex
    // Diamond with a rectangular notch at the top (apex)
    return [
      `M 0 ${top}`,
      `L ${-notchW} ${top + notchD}`,
      `L ${-halfW * 0.5} ${top + notchD}`,
      `L ${-halfW} 0`,
      `L 0 ${base}`,
      `L ${halfW} 0`,
      `L ${halfW * 0.5} ${top + notchD}`,
      `L ${notchW} ${top + notchD}`,
      `Z`,
    ].join(' ');
  }

  /** Tier line Y positions within the wedge, from base upward. */
  function tierLines(h: number, count: number): number[] {
    const base = h * 0.45;
    const top = -h * 0.55;
    const span = base - top;
    // Divide into (count+1) segments; place lines at segment boundaries
    return Array.from({ length: count }, (_, i) => {
      const t = (i + 1) / (count + 1);
      return base - span * t;
    });
  }

  /**
   * X extent of the wedge at a given Y (for clamping tier line width).
   * Linear interpolation from base halfW to shoulderW at shoulder Y.
   */
  function wedgeXatY(w: number, h: number, y: number): number {
    const halfW = w / 2;
    const base = h * 0.45;
    const shoulder = -h * 0.1;
    const shoulderW = w * 0.3;
    if (y >= shoulder) {
      // Between shoulder and base
      const t = (y - shoulder) / (base - shoulder);
      return shoulderW + t * (halfW - shoulderW);
    }
    // Above shoulder (near apex)
    const top = -h * 0.55;
    const t = (y - top) / (shoulder - top);
    return t * shoulderW;
  }

  const isKeystone = $derived(piece.kind === 'Keystone');
  const glyphPath = $derived(
    isKeystone
      ? keystonePath(WEDGE_W, WEDGE_H)
      : wedgePath(WEDGE_W, WEDGE_H)
  );

  // Tier lines apply only to Stones (height 2 = 1 line, height 3 = 2 lines)
  const tierCount = $derived(isKeystone ? 0 : Math.max(0, piece.height - 1));
  const tiers = $derived(tierLines(WEDGE_H, tierCount));

  // Rotation: P2 pieces point downward (180 degrees around cell center 0,0)
  const rotate = $derived(piece.owner === 'P2' ? 'rotate(180 0 0)' : undefined);

  // Tooltip from names.yaml
  const heightLabel = $derived(() => {
    if (piece.kind === 'Keystone') return names.pieces.keystone;
    if (piece.height >= 3) return names.pieces.spire;
    if (piece.height === 2) return names.pieces.pillar;
    return names.pieces.stone;
  });
  const ownerLabel = $derived(
    piece.owner === 'P1' ? names.players.P1 : names.players.P2
  );
  const tooltipText = $derived(`${ownerLabel} ${heightLabel()}`);

  // Stroke width scales with cell
  const strokeW = $derived(cellSize * 0.03);
  const tierStrokeW = $derived(cellSize * 0.025);
</script>

<!--
  The <g> is centered at (cx, cy) in the parent's coordinate system.
  The caller (Board.svelte) translates to the cell center before rendering.
-->
<g transform={rotate} aria-label={tooltipText}>
  <title>{tooltipText}</title>
  <path
    d={glyphPath}
    class="piece-fill"
    class:p1={piece.owner === 'P1'}
    class:p2={piece.owner === 'P2'}
    class:keystone={isKeystone}
    stroke-width={strokeW}
  />
  {#each tiers as y}
    {@const xExtent = wedgeXatY(WEDGE_W, WEDGE_H, y) * 0.85}
    <line
      x1={-xExtent}
      y1={y}
      x2={xExtent}
      y2={y}
      class="tier-line"
      stroke-width={tierStrokeW}
    />
  {/each}
</g>

<style>
  .piece-fill {
    stroke: var(--piece-stroke, #2a2a2a);
  }

  .piece-fill.p1 {
    fill: var(--piece-p1, #e8d5a3);
  }

  .piece-fill.p2 {
    fill: var(--piece-p2, #8b4513);
  }

  .piece-fill.keystone {
    stroke-width: calc(var(--piece-stroke-w, 1px) * 1.4);
  }

  .tier-line {
    stroke: var(--piece-stroke, #2a2a2a);
    stroke-linecap: round;
  }
</style>
