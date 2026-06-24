/**
 * Move-notation helpers for Cairn.
 */
import type { Action } from './types.js';

/** Converts a square index to algebraic coordinate (e.g. 12 → "d2"). */
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
 * Converts an Action to algebraic notation string.
 * Move: "<from>-<to>" or "<from>x<to>" if capture
 * Place: "+<to>"
 * Stack: "^<target>"
 * Suffix "+" if checkEnd, "#" if gameOver.
 */
export function actionToNotation(action: Action, opts: NotationOpts): string {
  let text: string;
  if ('Move' in action) {
    const sep = opts.capture ? 'x' : '-';
    text = `${sqToCoord(action.Move.from)}${sep}${sqToCoord(action.Move.to)}`;
  } else if ('Place' in action) {
    text = `+${sqToCoord(action.Place.to)}`;
  } else {
    text = `^${sqToCoord(action.Stack.target)}`;
  }
  if (opts.gameOver) return text + '#';
  if (opts.checkEnd) return text + '+';
  return text;
}
