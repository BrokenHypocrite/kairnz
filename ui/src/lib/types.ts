/**
 * TypeScript mirrors of the Rust serde JSON shapes exposed by cairn-tauri.
 * All types must match the serialized forms exactly; mismatches break runtime.
 */

/** Opaque game session identifier (Rust u64, safe JS number range for our scale). */
export type GameId = number;

/** The two players. Serializes as the bare string "P1" or "P2". */
export type Player = 'P1' | 'P2';

/** Piece kind. Serializes as "Stone" or "Keystone". */
export type PieceKind = 'Stone' | 'Keystone';

/** Spire movement mode. Serializes as "Dragon" or "Queen". */
export type SpireMode = 'Dragon' | 'Queen';

/**
 * A board square index (Rust newtype Sq(pub u8)).
 * Serde serializes this as a bare number, e.g. 12.
 */
export type Sq = number;

/**
 * An action a player can take.
 * Externally tagged enum: one key whose value is the payload object.
 */
export type Action =
  | { Move: { from: Sq; to: Sq } }
  | { Place: { to: Sq } }
  | { Stack: { target: Sq } };

/** Reason a game ended in a draw. */
export type DrawReason = 'MaxPlies' | 'Repetition';

/**
 * Terminal game result.
 * Win(Player) -> {"Win":"P1"} or {"Win":"P2"}.
 * Draw(DrawReason) -> {"Draw":"MaxPlies"} or {"Draw":"Repetition"}.
 */
export type GameResult =
  | { Win: Player }
  | { Draw: DrawReason };

/** A piece as rendered by the UI. */
export interface PieceView {
  owner: Player;
  kind: PieceKind;
  /** Stack height of this piece. */
  height: number;
}

/** Full snapshot of game state sent to the UI. Board has exactly 81 entries. */
export interface GameView {
  /** 81 entries in square-index order; null means empty. */
  board: (PieceView | null)[];
  /** Reserve token counts as [P1, P2]. */
  reserves: [number, number];
  to_move: Player;
  ap_remaining: number;
  result: GameResult | null;
  /** Square indices (0..80) of all Keystones currently in check. */
  checked_keystones: number[];
}

/** Info about a captured piece returned in ApplyResult. */
export interface CapturedInfo {
  owner: Player;
  kind: PieceKind;
  height: number;
  tokens_banked: number;
}

/** The result of applying an action, returned to the UI. */
export interface ApplyResult {
  view: GameView;
  turn_ended_on_check: boolean;
  last_capture: CapturedInfo | null;
  result: GameResult | null;
}

/** Rule configuration passed to new_game. Matches Rust RuleConfig defaults. */
export interface RuleConfig {
  spire: SpireMode;
  first_turn_ap: number;
  capture_lock: boolean;
  keystone_single_move: boolean;
  max_plies: number;
  repetition_fold: number;
}

/** Default rule config matching Rust RuleConfig::default(). */
export const defaultConfig: RuleConfig = {
  spire: 'Dragon',
  first_turn_ap: 1,
  capture_lock: true,
  keystone_single_move: true,
  max_plies: 400,
  repetition_fold: 3,
};
