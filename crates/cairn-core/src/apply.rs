use serde::{Deserialize, Serialize};

use crate::actions::{action_cost, Action, IllegalAction};
use crate::movement::move_targets;
use crate::outcome::GameResult;
use crate::piece::{PieceKind, Player};
use crate::position::Position;
use crate::square::Sq;
use crate::zobrist::zobrist_full;

/// Information about a piece captured during an action.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CapturedInfo {
    /// The original owner of the captured piece.
    pub owner: Player,
    /// The kind of the captured piece.
    pub kind: PieceKind,
    /// The height of the captured piece at time of capture.
    pub height: u8,
    /// Tokens banked to the mover's reserve (height for Stones; 0 for Keystones).
    pub tokens_banked: u8,
}

/// The outcome of a successfully applied action.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ActionOutcome {
    /// Information about any piece captured by this action.
    pub captured: Option<CapturedInfo>,
    /// Whether this action exhausted the turn's AP, ending the turn.
    pub turn_ended: bool,
    /// Whether the turn ended due to a check rule (always false until Task 11).
    pub ended_on_check: bool,
    /// The game result if this action ended the game.
    pub result: Option<GameResult>,
}

/// Applies `action` to `pos`, mutating board/reserves/ap/zobrist/ply.
///
/// Only `Action::Move` is implemented in Task 9. The `Place` and `Stack` arms
/// route to private stubs that will be completed in Task 10.
///
/// Returns an `ActionOutcome` describing what happened, or an `IllegalAction`
/// if the action is not currently legal.
pub fn apply_action(pos: &mut Position, action: Action) -> Result<ActionOutcome, IllegalAction> {
    match action {
        Action::Move { from, to } => apply_move(pos, from, to),
        Action::Place { to } => apply_place(pos, to),
        Action::Stack { target } => apply_stack(pos, target),
    }
}

/// Executes a Move action: validates, captures if applicable, relocates the piece,
/// decrements AP, recomputes the Zobrist hash, increments ply, and determines
/// whether the game ended.
fn apply_move(pos: &mut Position, from: Sq, to: Sq) -> Result<ActionOutcome, IllegalAction> {
    // --- Validation ---

    // Require at least 1 AP.
    if pos.turn.ap_remaining < 1 {
        return Err(IllegalAction::NoAp);
    }

    // Source must hold a piece owned by the active player.
    let moving_piece = match pos.piece_at(from) {
        Some(pc) if pc.owner == pos.to_move => pc,
        _ => return Err(IllegalAction::NotYourPiece),
    };

    // Destination must be in the piece's geometric move targets.
    let targets = move_targets(pos, from);
    if !targets.contains(&to) {
        return Err(IllegalAction::BadGeometry);
    }

    // Capture-lock: if enabled and the source square is locked, refuse.
    if pos.config.capture_lock && pos.turn.capture_locked.contains(from) {
        return Err(IllegalAction::CaptureLocked);
    }

    // Keystone single-move: if enabled and this keystone already moved, refuse.
    if pos.config.keystone_single_move
        && moving_piece.kind == PieceKind::Keystone
        && pos.turn.keystone_moved.contains(from)
    {
        return Err(IllegalAction::KeystoneAlreadyMoved);
    }

    // --- Capture by displacement ---

    let mover = pos.to_move;
    let captured = if let Some(occupant) = pos.piece_at(to) {
        // move_targets guarantees destination is enemy-only, but guard anyway.
        debug_assert_ne!(occupant.owner, mover, "move_targets must exclude friendly squares");

        let tokens_banked = match occupant.kind {
            // Stone captures: bank the stack height to the mover's reserve.
            PieceKind::Stone => {
                pos.reserves[mover.index()] += occupant.height;
                occupant.height
            }
            // Keystone captures: remove permanently; nothing banked.
            PieceKind::Keystone => 0,
        };

        Some(CapturedInfo {
            owner: occupant.owner,
            kind: occupant.kind,
            height: occupant.height,
            tokens_banked,
        })
    } else {
        None
    };

    // --- Relocate the piece ---

    pos.board[to.0 as usize] = Some(moving_piece);
    pos.board[from.0 as usize] = None;

    // --- Decrement AP ---

    let cost = action_cost(&Action::Move { from, to });
    pos.turn.ap_remaining = pos.turn.ap_remaining.saturating_sub(cost);

    // --- Update bookkeeping ---

    pos.zobrist = zobrist_full(pos);
    pos.ply += 1;

    // --- Outcome ---

    // Turn ends when AP reaches zero (check rule added in Task 11).
    let turn_ended = pos.turn.ap_remaining == 0;

    // Win condition: opponent has no Keystones remaining on the board.
    let result = if captured.map_or(false, |c| c.kind == PieceKind::Keystone) {
        let opponent_keystones = pos.keystones_of(mover.opponent()).count();
        if opponent_keystones == 0 {
            Some(GameResult::Win(mover))
        } else {
            None
        }
    } else {
        None
    };

    Ok(ActionOutcome {
        captured,
        turn_ended,
        ended_on_check: false,
        result,
    })
}

/// Placeholder for Task 10: Place action is not yet implemented.
///
/// This stub keeps the public signature of `apply_action` stable so Task 10
/// can fill this function body without touching the `Move` path.
fn apply_place(_pos: &mut Position, _to: Sq) -> Result<ActionOutcome, IllegalAction> {
    // TODO(task-10): implement Place action (place a piece from reserve onto an empty square).
    Err(IllegalAction::EmptyReserve)
}

/// Placeholder for Task 10: Stack action is not yet implemented.
///
/// This stub keeps the public signature of `apply_action` stable so Task 10
/// can fill this function body without touching the `Move` path.
fn apply_stack(_pos: &mut Position, _target: Sq) -> Result<ActionOutcome, IllegalAction> {
    // TODO(task-10): implement Stack action (place from reserve onto an own Stone, increasing height).
    Err(IllegalAction::NotStackable)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::RuleConfig;
    use crate::piece::{Piece, PieceKind, Player};
    use crate::position::{Position, TurnState};
    use crate::square::{BitBoard81, NUM_SQUARES};

    // --- Helpers ---

    fn empty_pos_with_ap(ap: u8) -> Position {
        Position {
            board: [None; NUM_SQUARES],
            reserves: [0, 0],
            to_move: Player::P1,
            turn: TurnState {
                ap_remaining: ap,
                capture_locked: BitBoard81::default(),
                keystone_moved: BitBoard81::default(),
                enemy_checked_at_start: BitBoard81::default(),
            },
            config: RuleConfig::default(),
            zobrist: 0,
            ply: 0,
        }
    }

    fn sq(file: u8, rank: u8) -> Sq {
        Sq::new(file, rank).unwrap()
    }

    fn place(pos: &mut Position, file: u8, rank: u8, piece: Piece) {
        let s = sq(file, rank);
        pos.board[s.0 as usize] = Some(piece);
    }

    /// Counts total stone-tokens on the board (sum of Stone heights) for both players.
    fn board_stone_tokens(pos: &Position) -> u32 {
        pos.board
            .iter()
            .filter_map(|cell| *cell)
            .filter(|pc| pc.kind == PieceKind::Stone)
            .map(|pc| pc.height as u32)
            .sum()
    }

    // --- Tests ---

    #[test]
    fn capturing_a_pillar_banks_two_tokens() {
        let mut pos = empty_pos_with_ap(2);
        // P1 Stone (height 1) at (4, 4) -- will be the mover.
        place(&mut pos, 4, 4, Piece::new(Player::P1, PieceKind::Stone, 1));
        // P2 Pillar (Stone, height 2) one step north -- will be captured.
        place(&mut pos, 4, 5, Piece::new(Player::P2, PieceKind::Stone, 2));

        let outcome = apply_action(&mut pos, Action::Move { from: sq(4, 4), to: sq(4, 5) })
            .expect("move must be legal");

        // Reserve banked 2 tokens.
        assert_eq!(pos.reserves[Player::P1.index()], 2);
        // Captured info carries the correct token count.
        let cap = outcome.captured.expect("capture must be reported");
        assert_eq!(cap.tokens_banked, 2);
        assert_eq!(cap.kind, PieceKind::Stone);
        // Destination now holds the P1 Stone.
        let dest = pos.piece_at(sq(4, 5)).expect("destination must be occupied");
        assert_eq!(dest.owner, Player::P1);
    }

    #[test]
    fn capturing_a_spire_banks_three_tokens() {
        let mut pos = empty_pos_with_ap(2);
        // P1 Stone (height 1) adjacent to a P2 Spire (Stone, height 3).
        // A height-1 stone moves orthogonally; place it one step south of the spire.
        place(&mut pos, 4, 4, Piece::new(Player::P1, PieceKind::Stone, 1));
        place(&mut pos, 4, 5, Piece::new(Player::P2, PieceKind::Stone, 3));

        let outcome = apply_action(&mut pos, Action::Move { from: sq(4, 4), to: sq(4, 5) })
            .expect("move must be legal");

        assert_eq!(pos.reserves[Player::P1.index()], 3);
        let cap = outcome.captured.expect("capture must be reported");
        assert_eq!(cap.tokens_banked, 3);
    }

    #[test]
    fn capturing_keystone_removes_it_permanently_not_banked() {
        let mut pos = empty_pos_with_ap(2);
        // P1 Pillar (height 2) can step diagonally to reach the adjacent P2 Keystone.
        place(&mut pos, 4, 4, Piece::new(Player::P1, PieceKind::Stone, 2));
        place(&mut pos, 5, 5, Piece::new(Player::P2, PieceKind::Keystone, 1));

        let reserve_before = pos.reserves[Player::P1.index()];
        let outcome = apply_action(&mut pos, Action::Move { from: sq(4, 4), to: sq(5, 5) })
            .expect("move must be legal");

        // Reserve must be unchanged for a Keystone capture.
        assert_eq!(pos.reserves[Player::P1.index()], reserve_before);
        let cap = outcome.captured.expect("capture must be reported");
        assert_eq!(cap.tokens_banked, 0);
        assert_eq!(cap.kind, PieceKind::Keystone);
        // Destination holds P1's piece.
        let dest = pos.piece_at(sq(5, 5)).expect("destination must be occupied");
        assert_eq!(dest.owner, Player::P1);
    }

    #[test]
    fn move_decrements_ap_by_one() {
        let mut pos = empty_pos_with_ap(2);
        place(&mut pos, 4, 4, Piece::new(Player::P1, PieceKind::Stone, 1));

        let ap_before = pos.turn.ap_remaining;
        apply_action(&mut pos, Action::Move { from: sq(4, 4), to: sq(4, 5) })
            .expect("move must be legal");

        assert_eq!(pos.turn.ap_remaining, ap_before - 1);
    }

    #[test]
    fn token_conservation_holds_after_capture() {
        let mut pos = empty_pos_with_ap(2);
        // P1 Stone (h1) captures P2 Stone (h2).
        place(&mut pos, 4, 4, Piece::new(Player::P1, PieceKind::Stone, 1));
        place(&mut pos, 4, 5, Piece::new(Player::P2, PieceKind::Stone, 2));

        let tokens_before =
            board_stone_tokens(&pos) + pos.reserves.iter().map(|&r| r as u32).sum::<u32>();

        apply_action(&mut pos, Action::Move { from: sq(4, 4), to: sq(4, 5) })
            .expect("move must be legal");

        let tokens_after =
            board_stone_tokens(&pos) + pos.reserves.iter().map(|&r| r as u32).sum::<u32>();

        assert_eq!(
            tokens_before, tokens_after,
            "stone-token total must be conserved across a capture"
        );
    }

    #[test]
    fn illegal_move_returns_specific_error_without_mutating() {
        let mut pos = empty_pos_with_ap(2);
        // Empty source square -- should return NotYourPiece.
        let before_board = pos.board;
        let before_reserves = pos.reserves;
        let before_ap = pos.turn.ap_remaining;

        let err = apply_action(&mut pos, Action::Move { from: sq(4, 4), to: sq(4, 5) })
            .expect_err("moving from an empty square must be illegal");

        assert_eq!(err, IllegalAction::NotYourPiece);
        assert_eq!(pos.board, before_board, "board must be unchanged");
        assert_eq!(pos.reserves, before_reserves, "reserves must be unchanged");
        assert_eq!(pos.turn.ap_remaining, before_ap, "AP must be unchanged");
    }

    #[test]
    fn winning_capture_sets_result_to_win() {
        let mut pos = empty_pos_with_ap(2);
        // P1 Pillar adjacent to the LAST P2 Keystone (no other P2 Keystones on board).
        place(&mut pos, 4, 4, Piece::new(Player::P1, PieceKind::Stone, 2));
        place(&mut pos, 5, 5, Piece::new(Player::P2, PieceKind::Keystone, 1));
        // Verify no other P2 Keystones exist (the helper position has none by construction).

        let outcome = apply_action(&mut pos, Action::Move { from: sq(4, 4), to: sq(5, 5) })
            .expect("move must be legal");

        assert_eq!(
            outcome.result,
            Some(GameResult::Win(Player::P1)),
            "removing the last opponent Keystone must yield a Win result"
        );
    }

    #[test]
    fn non_final_keystone_capture_does_not_win() {
        let mut pos = empty_pos_with_ap(2);
        place(&mut pos, 4, 4, Piece::new(Player::P1, PieceKind::Stone, 2));
        // Capture one of two P2 Keystones.
        place(&mut pos, 5, 5, Piece::new(Player::P2, PieceKind::Keystone, 1));
        // Second P2 Keystone elsewhere -- game not over.
        place(&mut pos, 0, 0, Piece::new(Player::P2, PieceKind::Keystone, 1));

        let outcome = apply_action(&mut pos, Action::Move { from: sq(4, 4), to: sq(5, 5) })
            .expect("move must be legal");

        assert_eq!(
            outcome.result, None,
            "one remaining opponent Keystone means the game is not over"
        );
    }

    #[test]
    fn no_ap_returns_no_ap_error() {
        let mut pos = empty_pos_with_ap(0);
        place(&mut pos, 4, 4, Piece::new(Player::P1, PieceKind::Stone, 1));

        let err = apply_action(&mut pos, Action::Move { from: sq(4, 4), to: sq(4, 5) })
            .expect_err("must fail with no AP");

        assert_eq!(err, IllegalAction::NoAp);
    }

    #[test]
    fn bad_geometry_returns_error() {
        let mut pos = empty_pos_with_ap(2);
        // Height-1 stone can only step orthogonally; diagonal is bad geometry.
        place(&mut pos, 4, 4, Piece::new(Player::P1, PieceKind::Stone, 1));

        let err = apply_action(&mut pos, Action::Move { from: sq(4, 4), to: sq(5, 5) })
            .expect_err("diagonal move for h1 stone must be illegal");

        assert_eq!(err, IllegalAction::BadGeometry);
    }

    #[test]
    fn ply_increments_after_move() {
        let mut pos = empty_pos_with_ap(2);
        place(&mut pos, 4, 4, Piece::new(Player::P1, PieceKind::Stone, 1));
        let ply_before = pos.ply;

        apply_action(&mut pos, Action::Move { from: sq(4, 4), to: sq(4, 5) })
            .expect("move must be legal");

        assert_eq!(pos.ply, ply_before + 1);
    }

    #[test]
    fn zobrist_changes_after_move() {
        let mut pos = empty_pos_with_ap(2);
        place(&mut pos, 4, 4, Piece::new(Player::P1, PieceKind::Stone, 1));
        let hash_before = pos.zobrist;

        apply_action(&mut pos, Action::Move { from: sq(4, 4), to: sq(4, 5) })
            .expect("move must be legal");

        assert_ne!(pos.zobrist, hash_before, "zobrist must change after a move");
    }

    #[test]
    fn turn_ended_false_when_ap_remains() {
        let mut pos = empty_pos_with_ap(2);
        place(&mut pos, 4, 4, Piece::new(Player::P1, PieceKind::Stone, 1));

        let outcome = apply_action(&mut pos, Action::Move { from: sq(4, 4), to: sq(4, 5) })
            .expect("move must be legal");

        // Started with 2 AP, spent 1 -> 1 AP left -> turn not ended.
        assert!(!outcome.turn_ended);
    }

    #[test]
    fn turn_ended_true_when_ap_exhausted() {
        let mut pos = empty_pos_with_ap(1);
        place(&mut pos, 4, 4, Piece::new(Player::P1, PieceKind::Stone, 1));

        let outcome = apply_action(&mut pos, Action::Move { from: sq(4, 4), to: sq(4, 5) })
            .expect("move must be legal");

        // Started with 1 AP, spent 1 -> 0 AP left -> turn ended.
        assert!(outcome.turn_ended);
    }

    #[test]
    fn capture_locked_piece_returns_error() {
        let mut cfg = RuleConfig::default();
        cfg.capture_lock = true;
        let mut pos = empty_pos_with_ap(2);
        pos.config = cfg;

        let from = sq(4, 4);
        place(&mut pos, 4, 4, Piece::new(Player::P1, PieceKind::Stone, 1));
        pos.turn.capture_locked.set(from);

        let err = apply_action(&mut pos, Action::Move { from, to: sq(4, 5) })
            .expect_err("capture-locked piece must not be movable");

        assert_eq!(err, IllegalAction::CaptureLocked);
    }

    #[test]
    fn keystone_already_moved_returns_error() {
        let mut cfg = RuleConfig::default();
        cfg.keystone_single_move = true;
        let mut pos = empty_pos_with_ap(2);
        pos.config = cfg;

        let from = sq(4, 4);
        place(&mut pos, 4, 4, Piece::new(Player::P1, PieceKind::Keystone, 1));
        pos.turn.keystone_moved.set(from);

        let err = apply_action(&mut pos, Action::Move { from, to: sq(4, 5) })
            .expect_err("keystone that already moved must not move again");

        assert_eq!(err, IllegalAction::KeystoneAlreadyMoved);
    }
}
