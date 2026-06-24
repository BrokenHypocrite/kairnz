/**
 * Typed wrappers around @tauri-apps/api/core invoke.
 * Each function mirrors a #[tauri::command] in src-tauri/src/commands.rs.
 * Argument object keys must match the Rust parameter names exactly.
 */
import { invoke } from '@tauri-apps/api/core';
import type { Action, ApplyResult, GameId, GameView, RuleConfig, Sq } from './types.js';

/** Creates a new game with the given rule configuration. */
export async function newGame(config: RuleConfig): Promise<[GameId, GameView]> {
  return invoke('new_game', { config });
}

/** Returns the current view for an existing game. */
export async function getView(id: GameId): Promise<GameView> {
  return invoke('get_view', { id });
}

/**
 * Returns legal actions for a game.
 * When `from` is provided, only Move actions originating at that square are returned.
 */
export async function legalActions(id: GameId, from?: Sq): Promise<Action[]> {
  return invoke('legal_actions', { id, from: from ?? null });
}

/** Applies an action to the game and returns the updated state. */
export async function applyAction(id: GameId, action: Action): Promise<ApplyResult> {
  return invoke('apply_action', { id, action });
}

/** Undoes the last action and returns the restored view. */
export async function undo(id: GameId): Promise<GameView> {
  return invoke('undo', { id });
}
