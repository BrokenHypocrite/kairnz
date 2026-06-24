/**
 * Move-notation helpers for Cairn.
 */
import type { Action, PieceView } from './types.js';

/**
 * Returns the board squares involved in a single action.
 * Used to record which squares to highlight for the previous-move indicator.
 */
export function actionSquares(action: Action): number[] {
  if ('Move' in action) return [action.Move.from, action.Move.to];
  if ('Place' in action) return [action.Place.to];
  return [action.Stack.target];
}

/** Converts a square index to algebraic coordinate (e.g. 12 -> "d2"). */
export function sqToCoord(sq: number): string {
  const file = sq % 9;
  const rank = Math.floor(sq / 9);
  return `${String.fromCharCode(97 + file)}${rank + 1}`;
}

/** Options controlling suffix appended to move notation. */
export interface NotationOpts {
  capture: boolean;
  checkEnd: boolean;
  gameOver: boolean;
}

/**
 * Returns the single-letter piece code for a piece.
 * Keystone -> "K"; by height: 1 -> "S", 2 -> "P", 3 -> "T".
 */
export function pieceCode(piece: PieceView, codes: { Stone: string; Pillar: string; Spire: string; Keystone: string }): string {
  if (piece.kind === 'Keystone') return codes.Keystone;
  if (piece.height >= 3) return codes.Spire;
  if (piece.height === 2) return codes.Pillar;
  return codes.Stone;
}

/**
 * Converts an Action to algebraic notation string.
 * Move: "<code><from>-<to>" or "<code><from>x<to>" if capture
 * Place: "+<to>"
 * Stack: "^<target>"
 * Suffix "+" if checkEnd, "#" if gameOver.
 *
 * `movingPiece` must be the piece at `from` from the PRE-APPLY view (Move only).
 */
export function actionToNotation(
  action: Action,
  opts: NotationOpts,
  codes: { Stone: string; Pillar: string; Spire: string; Keystone: string },
  movingPiece?: PieceView | null
): string {
  let text: string;
  if ('Move' in action) {
    const sep = opts.capture ? 'x' : '-';
    const code = movingPiece ? pieceCode(movingPiece, codes) : '';
    text = `${code}${sqToCoord(action.Move.from)}${sep}${sqToCoord(action.Move.to)}`;
  } else if ('Place' in action) {
    text = `+${sqToCoord(action.Place.to)}`;
  } else {
    text = `^${sqToCoord(action.Stack.target)}`;
  }
  if (opts.gameOver) return text + '#';
  if (opts.checkEnd) return text + '+';
  return text;
}
