use serde::{Deserialize, Serialize};

use crate::actions::{action_cost, legal_actions, Action, IllegalAction};
use crate::check::checked_enemy_keystone_squares;
use crate::movement::move_targets;
use crate::outcome::GameResult;
use crate::piece::{PieceKind, Player};
use crate::position::Position;
use crate::square::Sq;
use crate::turn::advance_turn;
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
    /// Whether this action ended the turn (via AP exhaustion or the check rule).
    pub turn_ended: bool,
    /// Whether the turn ended because this action newly placed an enemy Keystone
    /// in check that was not already in check at the start of the turn.
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

    // --- Populate toggle bitboards ---

    // If this move was a capture, lock the destination square.
    // The generation side gates on config.capture_lock; the bitboard is cleared every turn,
    // so populating it unconditionally is always safe.
    if captured.is_some() {
        pos.turn.capture_locked.set(to);
    }

    // If the moved piece is a Keystone, record its new square.
    // Unconditional: generation gates on config.keystone_single_move.
    if moving_piece.kind == PieceKind::Keystone {
        pos.turn.keystone_moved.set(to);
    }

    // --- Decrement AP ---

    let cost = action_cost(&Action::Move { from, to });
    debug_assert!(pos.turn.ap_remaining >= cost, "AP underflow: validation must ensure ap_remaining >= cost before decrement");
    pos.turn.ap_remaining -= cost;

    // --- Update bookkeeping ---

    pos.zobrist = zobrist_full(pos);
    pos.ply += 1;

    // Win condition: opponent has no Keystones remaining on the board.
    let result = if captured.is_some_and(|c| c.kind == PieceKind::Keystone) {
        let opponent_keystones = pos.keystones_of(mover.opponent()).count();
        if opponent_keystones == 0 {
            Some(GameResult::Win(mover))
        } else {
            None
        }
    } else {
        None
    };

    Ok(finalize(pos, mover, captured, result))
}

/// Shared post-mutation step for every action arm.
///
/// `mover` is the player who just acted; it MUST be captured before any turn
/// advance flips `to_move`. Determines whether the action newly placed an enemy
/// Keystone in check (the turn-ending check rule), whether the turn ended (check
/// or AP exhaustion), advances the turn when appropriate, and assembles the
/// `ActionOutcome`.
///
/// The check rule is square-anchored: `enemy_checked_at_start` records the enemy
/// Keystone squares already in check at the start of the turn. Because the
/// defender cannot move during the mover's turn, those squares are fixed, and the
/// only way one leaves the live set is via capture. Capturing an already-checked
/// Keystone therefore does NOT register as a new check (its square drops out of
/// `now`), while newly threatening any other enemy Keystone does.
fn finalize(
    pos: &mut Position,
    mover: Player,
    captured: Option<CapturedInfo>,
    result: Option<GameResult>,
) -> ActionOutcome {
    let now = checked_enemy_keystone_squares(pos, mover);
    let newly_checked = now.difference(pos.turn.enemy_checked_at_start);
    let ended_on_check = !newly_checked.is_empty();
    let mut turn_ended = ended_on_check || pos.turn.ap_remaining == 0;

    // §5: if the game is still live and the turn hasn't ended yet, end it
    // when the acting player has no remaining legal action (AP > 0 but nothing
    // they can actually do). This prevents the game loop from ever seeing an
    // empty mid-turn action list.
    if result.is_none() && !turn_ended && legal_actions(pos).is_empty() {
        turn_ended = true;
    }

    if result.is_some() {
        // The action won the game: it is over. Force turn_ended and do not advance.
        turn_ended = true;
    } else if turn_ended {
        advance_turn(pos);
    }

    ActionOutcome {
        captured,
        turn_ended,
        ended_on_check,
        result,
    }
}

/// Maximum height a Stone must be at or below to accept a stack token.
const STACK_MAX_SOURCE_HEIGHT: u8 = 2;

/// Executes a Place action: validates reserve and target vacancy, places a new
/// height-1 Stone owned by the mover, decrements AP by 1, recomputes the Zobrist
/// hash, and increments ply.
fn apply_place(pos: &mut Position, to: Sq) -> Result<ActionOutcome, IllegalAction> {
    // Require at least 1 AP.
    if pos.turn.ap_remaining < 1 {
        return Err(IllegalAction::NoAp);
    }

    let mover = pos.to_move;

    // Reserve must be non-empty.
    if pos.reserves[mover.index()] == 0 {
        return Err(IllegalAction::EmptyReserve);
    }

    // Target square must be vacant.
    if pos.piece_at(to).is_some() {
        return Err(IllegalAction::TargetNotEmpty);
    }

    // --- Mutation ---

    pos.reserves[mover.index()] -= 1;
    pos.board[to.0 as usize] = Some(crate::piece::Piece::new(mover, PieceKind::Stone, 1));

    let cost = action_cost(&Action::Place { to });
    debug_assert!(
        pos.turn.ap_remaining >= cost,
        "AP underflow: validation must ensure ap_remaining >= cost before decrement"
    );
    pos.turn.ap_remaining -= cost;

    pos.zobrist = zobrist_full(pos);
    pos.ply += 1;

    Ok(finalize(pos, mover, None, None))
}

/// Executes a Stack action: validates AP (needs 2), reserve, and that the target
/// holds the mover's own Stone with height <= STACK_MAX_SOURCE_HEIGHT. Increments
/// the Stone's height by 1, decrements reserve, costs 2 AP (exhausting the turn),
/// recomputes the Zobrist hash, and increments ply.
fn apply_stack(pos: &mut Position, target: Sq) -> Result<ActionOutcome, IllegalAction> {
    // Stack costs the whole 2-AP turn budget.
    if pos.turn.ap_remaining < 2 {
        return Err(IllegalAction::NeedsTwoAp);
    }

    let mover = pos.to_move;

    // Reserve must be non-empty.
    if pos.reserves[mover.index()] == 0 {
        return Err(IllegalAction::EmptyReserve);
    }

    // Target must hold the mover's own Stone with height <= STACK_MAX_SOURCE_HEIGHT.
    // Rejects: empty square, enemy piece, Keystone, or height-3 Stone.
    let occupant = match pos.piece_at(target) {
        Some(pc)
            if pc.owner == mover
                && pc.kind == PieceKind::Stone
                && pc.height <= STACK_MAX_SOURCE_HEIGHT =>
        {
            pc
        }
        _ => return Err(IllegalAction::NotStackable),
    };

    // --- Mutation ---

    pos.reserves[mover.index()] -= 1;
    pos.board[target.0 as usize] = Some(crate::piece::Piece::new(mover, PieceKind::Stone, occupant.height + 1));

    let cost = action_cost(&Action::Stack { target });
    debug_assert!(
        pos.turn.ap_remaining >= cost,
        "AP underflow: validation must ensure ap_remaining >= cost before decrement"
    );
    pos.turn.ap_remaining -= cost;

    pos.zobrist = zobrist_full(pos);
    pos.ply += 1;

    Ok(finalize(pos, mover, None, None))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::check::checked_enemy_keystone_squares;
    use crate::config::{RuleConfig, DEFAULT_AP};
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
    fn opponent_piece_at_source_returns_not_your_piece_without_mutating() {
        let mut pos = empty_pos_with_ap(2);
        // P2 Stone at source; P1 to_move (default) tries to move it.
        let from = sq(4, 4);
        let to = sq(4, 5);
        place(&mut pos, 4, 4, Piece::new(Player::P2, PieceKind::Stone, 1));
        let before_board = pos.board;
        let before_reserves = pos.reserves;
        let before_ap = pos.turn.ap_remaining;

        let err = apply_action(&mut pos, Action::Move { from, to })
            .expect_err("moving opponent's piece must be illegal");

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

    // ---------------------------------------------------------------------------
    // Place tests
    // ---------------------------------------------------------------------------

    #[test]
    fn place_consumes_reserve_and_creates_height1_stone() {
        let mut pos = empty_pos_with_ap(2);
        pos.reserves[Player::P1.index()] = 3;
        let target = sq(4, 4);

        let outcome = apply_action(&mut pos, Action::Place { to: target })
            .expect("place must be legal");

        // Reserve decremented by 1.
        assert_eq!(pos.reserves[Player::P1.index()], 2);
        // A height-1 Stone owned by P1 appears at the target square.
        let piece = pos.piece_at(target).expect("target must be occupied after Place");
        assert_eq!(piece.owner, Player::P1);
        assert_eq!(piece.kind, PieceKind::Stone);
        assert_eq!(piece.height, 1);
        // AP cost is 1.
        assert_eq!(pos.turn.ap_remaining, 1);
        // turn_ended false because 1 AP remains.
        assert!(!outcome.turn_ended);
        // Place cannot capture.
        assert!(outcome.captured.is_none());
    }

    #[test]
    fn place_on_occupied_square_is_illegal() {
        let mut pos = empty_pos_with_ap(2);
        pos.reserves[Player::P1.index()] = 1;
        let target = sq(4, 4);
        place(&mut pos, 4, 4, Piece::new(Player::P2, PieceKind::Stone, 1));

        let err = apply_action(&mut pos, Action::Place { to: target })
            .expect_err("placing on an occupied square must be illegal");

        assert_eq!(err, IllegalAction::TargetNotEmpty);
        // Position must be unchanged.
        assert_eq!(pos.reserves[Player::P1.index()], 1);
    }

    #[test]
    fn place_with_empty_reserve_is_illegal() {
        let mut pos = empty_pos_with_ap(2);
        pos.reserves[Player::P1.index()] = 0;
        let target = sq(4, 4);

        let err = apply_action(&mut pos, Action::Place { to: target })
            .expect_err("placing with empty reserve must be illegal");

        assert_eq!(err, IllegalAction::EmptyReserve);
    }

    #[test]
    fn place_with_no_ap_is_illegal() {
        let mut pos = empty_pos_with_ap(0);
        pos.reserves[Player::P1.index()] = 1;

        let err = apply_action(&mut pos, Action::Place { to: sq(4, 4) })
            .expect_err("placing with 0 AP must be illegal");

        assert_eq!(err, IllegalAction::NoAp);
    }

    #[test]
    fn place_ply_increments_and_zobrist_changes() {
        let mut pos = empty_pos_with_ap(2);
        pos.reserves[Player::P1.index()] = 1;
        let ply_before = pos.ply;
        let hash_before = pos.zobrist;

        apply_action(&mut pos, Action::Place { to: sq(4, 4) })
            .expect("place must be legal");

        assert_eq!(pos.ply, ply_before + 1);
        assert_ne!(pos.zobrist, hash_before);
    }

    #[test]
    fn place_with_last_ap_ends_turn() {
        let mut pos = empty_pos_with_ap(1);
        pos.reserves[Player::P1.index()] = 1;

        let outcome = apply_action(&mut pos, Action::Place { to: sq(4, 4) })
            .expect("place must be legal");

        // Spending the last AP ends the turn, which advances: side flips and the
        // new turn resets AP to DEFAULT_AP.
        assert!(outcome.turn_ended);
        assert!(!outcome.ended_on_check, "a quiet place threatens nothing");
        assert_eq!(pos.to_move, Player::P2, "turn must advance to P2");
        assert_eq!(pos.turn.ap_remaining, DEFAULT_AP, "new turn resets AP");
    }

    // ---------------------------------------------------------------------------
    // Stack tests
    // ---------------------------------------------------------------------------

    #[test]
    fn stack_raises_height_and_costs_whole_turn() {
        let mut pos = empty_pos_with_ap(2);
        pos.reserves[Player::P1.index()] = 2;
        let target = sq(4, 4);
        place(&mut pos, 4, 4, Piece::new(Player::P1, PieceKind::Stone, 1));

        let outcome = apply_action(&mut pos, Action::Stack { target })
            .expect("stack must be legal");

        // Height incremented from 1 to 2.
        let piece = pos.piece_at(target).expect("target must still be occupied");
        assert_eq!(piece.height, 2);
        // Reserve decremented by 1.
        assert_eq!(pos.reserves[Player::P1.index()], 1);
        // Stack costs the full 2-AP budget, so the turn ends and advances:
        // side flips and the new turn's AP resets to DEFAULT_AP.
        assert!(outcome.turn_ended);
        assert!(!outcome.ended_on_check, "this stack threatens nothing");
        assert_eq!(pos.to_move, Player::P2, "turn must advance to P2");
        assert_eq!(pos.turn.ap_remaining, DEFAULT_AP, "new turn resets AP");
        // No capture.
        assert!(outcome.captured.is_none());
    }

    #[test]
    fn stack_onto_keystone_is_illegal() {
        let mut pos = empty_pos_with_ap(2);
        pos.reserves[Player::P1.index()] = 1;
        let target = sq(4, 4);
        place(&mut pos, 4, 4, Piece::new(Player::P1, PieceKind::Keystone, 1));

        let err = apply_action(&mut pos, Action::Stack { target })
            .expect_err("stacking onto a Keystone must be illegal");

        assert_eq!(err, IllegalAction::NotStackable);
    }

    #[test]
    fn stack_with_one_ap_is_illegal() {
        let mut pos = empty_pos_with_ap(1);
        pos.reserves[Player::P1.index()] = 1;
        let target = sq(4, 4);
        place(&mut pos, 4, 4, Piece::new(Player::P1, PieceKind::Stone, 1));

        let err = apply_action(&mut pos, Action::Stack { target })
            .expect_err("stacking with only 1 AP must be illegal");

        assert_eq!(err, IllegalAction::NeedsTwoAp);
    }

    #[test]
    fn stack_onto_height_3_is_illegal() {
        let mut pos = empty_pos_with_ap(2);
        pos.reserves[Player::P1.index()] = 1;
        let target = sq(4, 4);
        place(&mut pos, 4, 4, Piece::new(Player::P1, PieceKind::Stone, 3));

        let err = apply_action(&mut pos, Action::Stack { target })
            .expect_err("stacking onto a height-3 Stone must be illegal");

        assert_eq!(err, IllegalAction::NotStackable);
    }

    #[test]
    fn stack_onto_enemy_is_illegal() {
        let mut pos = empty_pos_with_ap(2);
        pos.reserves[Player::P1.index()] = 1;
        let target = sq(4, 4);
        place(&mut pos, 4, 4, Piece::new(Player::P2, PieceKind::Stone, 1));

        let err = apply_action(&mut pos, Action::Stack { target })
            .expect_err("stacking onto an enemy piece must be illegal");

        assert_eq!(err, IllegalAction::NotStackable);
    }

    #[test]
    fn stack_onto_empty_is_illegal() {
        let mut pos = empty_pos_with_ap(2);
        pos.reserves[Player::P1.index()] = 1;
        let target = sq(4, 4);

        let err = apply_action(&mut pos, Action::Stack { target })
            .expect_err("stacking onto an empty square must be illegal");

        assert_eq!(err, IllegalAction::NotStackable);
    }

    #[test]
    fn stack_with_empty_reserve_is_illegal() {
        let mut pos = empty_pos_with_ap(2);
        pos.reserves[Player::P1.index()] = 0;
        let target = sq(4, 4);
        place(&mut pos, 4, 4, Piece::new(Player::P1, PieceKind::Stone, 1));

        let err = apply_action(&mut pos, Action::Stack { target })
            .expect_err("stacking with empty reserve must be illegal");

        assert_eq!(err, IllegalAction::EmptyReserve);
    }

    #[test]
    fn stack_ply_increments_and_zobrist_changes() {
        let mut pos = empty_pos_with_ap(2);
        pos.reserves[Player::P1.index()] = 1;
        let target = sq(4, 4);
        place(&mut pos, 4, 4, Piece::new(Player::P1, PieceKind::Stone, 1));
        let ply_before = pos.ply;
        let hash_before = pos.zobrist;

        apply_action(&mut pos, Action::Stack { target })
            .expect("stack must be legal");

        assert_eq!(pos.ply, ply_before + 1);
        assert_ne!(pos.zobrist, hash_before);
    }

    // ---------------------------------------------------------------------------
    // Turn-ending check rule
    // ---------------------------------------------------------------------------

    #[test]
    fn newly_threatening_a_keystone_by_move_ends_turn_immediately() {
        let mut pos = empty_pos_with_ap(2);
        // P2 Keystone at (4, 4), not yet attacked.
        place(&mut pos, 4, 4, Piece::new(Player::P2, PieceKind::Keystone, 1));
        // P1 Pillar (h2) at (4, 2): one orthogonal step to (4, 3) brings it
        // adjacent to the Keystone, newly threatening it.
        place(&mut pos, 4, 2, Piece::new(Player::P1, PieceKind::Stone, 2));

        let outcome = apply_action(&mut pos, Action::Move { from: sq(4, 2), to: sq(4, 3) })
            .expect("move must be legal");

        assert!(outcome.ended_on_check, "newly threatening a Keystone ends on check");
        assert!(outcome.turn_ended, "an ended-on-check action ends the turn");
        // The second AP is forfeit: the turn advanced to P2 with a fresh budget.
        assert_eq!(pos.to_move, Player::P2, "turn advanced after the check");
        assert_eq!(pos.turn.ap_remaining, DEFAULT_AP, "remaining AP forfeit; new turn budget");
    }

    #[test]
    fn newly_threatening_by_place_ends_turn() {
        let mut pos = empty_pos_with_ap(2);
        pos.reserves[Player::P1.index()] = 1;
        // P2 Keystone at (4, 4).
        place(&mut pos, 4, 4, Piece::new(Player::P2, PieceKind::Keystone, 1));

        // Place a height-1 Stone at (4, 3): a h1 Stone steps orthogonally, so it
        // newly threatens the Keystone one square north.
        let outcome = apply_action(&mut pos, Action::Place { to: sq(4, 3) })
            .expect("place must be legal");

        assert!(outcome.ended_on_check, "placing an attacker newly checks the Keystone");
        assert!(outcome.turn_ended);
        assert_eq!(pos.to_move, Player::P2);
        assert_eq!(pos.turn.ap_remaining, DEFAULT_AP);
    }

    #[test]
    fn newly_threatening_by_stack_ends_turn() {
        let mut pos = empty_pos_with_ap(2);
        pos.reserves[Player::P1.index()] = 1;
        // P2 Keystone at (5, 5), diagonally adjacent to (4, 4).
        place(&mut pos, 5, 5, Piece::new(Player::P2, PieceKind::Keystone, 1));
        // P1 height-1 Stone at (4, 4): a h1 Stone moves only orthogonally, so it
        // does NOT yet threaten the diagonal Keystone.
        place(&mut pos, 4, 4, Piece::new(Player::P1, PieceKind::Stone, 1));
        assert!(
            checked_enemy_keystone_squares(&pos, Player::P1).is_empty(),
            "precondition: h1 Stone does not threaten the diagonal Keystone"
        );

        // Stack to height 2: a Pillar steps in all 8 directions and now threatens
        // the diagonal Keystone. Stack also costs 2 AP, but the end must be
        // reflected via ended_on_check.
        let outcome = apply_action(&mut pos, Action::Stack { target: sq(4, 4) })
            .expect("stack must be legal");

        assert!(outcome.ended_on_check, "the upgraded movement newly threatens the Keystone");
        assert!(outcome.turn_ended);
        assert_eq!(pos.to_move, Player::P2);
        assert_eq!(pos.turn.ap_remaining, DEFAULT_AP);
    }

    #[test]
    fn capturing_already_checked_keystone_does_not_end_on_check_and_may_continue() {
        let mut pos = empty_pos_with_ap(2);
        // P2 Keystone A at (4, 4), already in check at the start of the turn.
        let keystone_a = sq(4, 4);
        place(&mut pos, 4, 4, Piece::new(Player::P2, PieceKind::Keystone, 1));
        // A SECOND P2 Keystone elsewhere and safe, so capturing A is not a win.
        place(&mut pos, 0, 8, Piece::new(Player::P2, PieceKind::Keystone, 1));
        // P1 Pillar (h2) adjacent at (4, 3).
        place(&mut pos, 4, 3, Piece::new(Player::P1, PieceKind::Stone, 2));
        // Model "already in check at start": A is in the set.
        pos.turn.enemy_checked_at_start = checked_enemy_keystone_squares(&pos, Player::P1);
        assert!(pos.turn.enemy_checked_at_start.contains(keystone_a), "precondition: A in check at start");

        // Capture A with the first AP.
        let outcome = apply_action(&mut pos, Action::Move { from: sq(4, 3), to: keystone_a })
            .expect("capturing move must be legal");

        let cap = outcome.captured.expect("a Keystone was captured");
        assert_eq!(cap.kind, PieceKind::Keystone);
        assert!(!outcome.ended_on_check, "capturing an already-checked Keystone is not a new check");
        assert!(!outcome.turn_ended, "AP remains, so the turn continues");
        assert_eq!(pos.to_move, Player::P1, "turn did NOT advance");
        assert_eq!(pos.turn.ap_remaining, 1, "one AP spent of two");
    }

    #[test]
    fn threatening_the_second_keystone_ends_turn_even_if_first_already_checked() {
        let mut pos = empty_pos_with_ap(2);
        // Keystone A at (0, 0), already in check at start (modeled below).
        let keystone_a = sq(0, 0);
        place(&mut pos, 0, 0, Piece::new(Player::P2, PieceKind::Keystone, 1));
        // The piece that checks A: a P1 Pillar adjacent at (1, 0).
        place(&mut pos, 1, 0, Piece::new(Player::P1, PieceKind::Stone, 2));
        // Keystone B at (4, 4), NOT yet in check.
        place(&mut pos, 4, 4, Piece::new(Player::P2, PieceKind::Keystone, 1));
        // A separate P1 Pillar at (4, 2) that will newly threaten B by stepping to (4, 3).
        place(&mut pos, 4, 2, Piece::new(Player::P1, PieceKind::Stone, 2));

        pos.turn.enemy_checked_at_start = checked_enemy_keystone_squares(&pos, Player::P1);
        assert!(pos.turn.enemy_checked_at_start.contains(keystone_a), "precondition: A already checked");
        assert!(!pos.turn.enemy_checked_at_start.contains(sq(4, 4)), "precondition: B not yet checked");

        // Newly threaten B.
        let outcome = apply_action(&mut pos, Action::Move { from: sq(4, 2), to: sq(4, 3) })
            .expect("move must be legal");

        assert!(outcome.ended_on_check, "newly threatening B is a new check even with A already checked");
        assert!(outcome.turn_ended);
        assert_eq!(pos.to_move, Player::P2);
    }

    #[test]
    fn leaving_own_keystone_in_check_is_legal() {
        let mut pos = empty_pos_with_ap(2);
        // P1's OWN Keystone at (4, 4).
        place(&mut pos, 4, 4, Piece::new(Player::P1, PieceKind::Keystone, 1));
        // A P2 Dragon Spire (h3) at (4, 8): slides orthogonally down file 4, but is
        // currently blocked by the P1 shield.
        place(&mut pos, 4, 8, Piece::new(Player::P2, PieceKind::Stone, 3));
        // P1 Stone shield at (4, 6) on the Spire's ray to the Keystone.
        place(&mut pos, 4, 6, Piece::new(Player::P1, PieceKind::Stone, 1));

        // Move the shield east off the file: this opens the Spire's ray and leaves
        // P1's own Keystone attacked. The action must still apply; there is no
        // forced check resolution for the mover's own Keystone.
        let outcome = apply_action(&mut pos, Action::Move { from: sq(4, 6), to: sq(5, 6) })
            .expect("exposing your own Keystone is fully legal");

        // The mover (P1) threatened no ENEMY Keystone, so no check end.
        assert!(!outcome.ended_on_check, "own-Keystone exposure is not an enemy check");
        // P1's own Keystone is indeed now attacked by P2 (sanity check on the setup).
        assert!(
            checked_enemy_keystone_squares(&pos, Player::P2).contains(sq(4, 4)),
            "the Spire now attacks P1's own Keystone"
        );
        // The shield piece relocated successfully.
        assert!(pos.piece_at(sq(5, 6)).is_some(), "the shield moved");
        assert!(pos.piece_at(sq(4, 6)).is_none(), "the shield's old square is vacant");
    }

    // ---------------------------------------------------------------------------
    // §7 toggle end-to-end tests (Task 12)
    // ---------------------------------------------------------------------------

    /// Returns true if any Move action in `actions` has `from` equal to `sq`.
    fn has_move_from(actions: &[Action], sq: Sq) -> bool {
        actions.iter().any(|a| matches!(a, Action::Move { from, .. } if *from == sq))
    }

    #[test]
    fn capture_lock_on_blocks_second_move_of_capturing_piece() {
        // ap=2, capture_lock=true. Piece X at (4,4) captures enemy Stone at (4,5).
        // Turn continues (1 AP left). The capturing piece is now at (4,5) and must be locked.
        let mut cfg = RuleConfig::default();
        cfg.capture_lock = true;
        let mut pos = empty_pos_with_ap(2);
        pos.config = cfg;
        // P1 Pillar (h2) is the mover -- h2 can step diagonally but we use ortho here.
        place(&mut pos, 4, 4, Piece::new(Player::P1, PieceKind::Stone, 1));
        // Enemy Stone: simple target one step north.
        place(&mut pos, 4, 5, Piece::new(Player::P2, PieceKind::Stone, 1));
        // Keep P2 Keystone off board so capture does not end the game.
        // Keep board free of any enemy Keystone so check rule does not fire.

        let outcome = apply_action(&mut pos, Action::Move { from: sq(4, 4), to: sq(4, 5) })
            .expect("capture move must be legal");
        assert!(!outcome.turn_ended, "1 AP must remain after capture");
        assert_eq!(pos.turn.ap_remaining, 1);

        let dest = sq(4, 5);
        let actions = crate::actions::legal_actions(&pos);
        assert!(
            !has_move_from(&actions, dest),
            "capture-locked piece must not appear as Move source when toggle is on"
        );
    }

    #[test]
    fn capture_lock_off_allows_chained_capture() {
        // Same setup but capture_lock=false: the piece may move again.
        let mut cfg = RuleConfig::default();
        cfg.capture_lock = false;
        let mut pos = empty_pos_with_ap(2);
        pos.config = cfg;
        place(&mut pos, 4, 4, Piece::new(Player::P1, PieceKind::Stone, 1));
        place(&mut pos, 4, 5, Piece::new(Player::P2, PieceKind::Stone, 1));

        let outcome = apply_action(&mut pos, Action::Move { from: sq(4, 4), to: sq(4, 5) })
            .expect("capture move must be legal");
        assert!(!outcome.turn_ended, "1 AP must remain");

        let dest = sq(4, 5);
        let actions = crate::actions::legal_actions(&pos);
        assert!(
            has_move_from(&actions, dest),
            "toggle off: capturing piece must still be a legal Move source"
        );
    }

    #[test]
    fn keystone_single_move_on_blocks_second_keystone_move() {
        // ap=2, keystone_single_move=true. Move a Keystone from (4,4) to (4,5).
        // The Keystone is isolated; the move does not newly threaten any enemy Keystone.
        let mut cfg = RuleConfig::default();
        cfg.keystone_single_move = true;
        let mut pos = empty_pos_with_ap(2);
        pos.config = cfg;
        // P1 Keystone at (4,4). Place it away from any enemy Keystone.
        place(&mut pos, 4, 4, Piece::new(Player::P1, PieceKind::Keystone, 1));
        // A second P1 Stone at (6,6) ensures that after the Keystone moves, P1 still
        // has at least one legal action (the Stone can move), so the turn does not end
        // via the no-legal-action rule. This is necessary because the test is checking
        // that the moved Keystone is excluded from legal_actions, not that the turn ends.
        place(&mut pos, 6, 6, Piece::new(Player::P1, PieceKind::Stone, 1));

        let outcome = apply_action(&mut pos, Action::Move { from: sq(4, 4), to: sq(4, 5) })
            .expect("keystone move must be legal");
        assert!(!outcome.turn_ended, "1 AP must remain; Stone at (6,6) keeps legal actions non-empty");

        let dest = sq(4, 5);
        let actions = crate::actions::legal_actions(&pos);
        assert!(
            !has_move_from(&actions, dest),
            "moved keystone must not be a Move source when keystone_single_move is on"
        );
    }

    #[test]
    fn keystone_single_move_off_allows_two_keystone_moves() {
        // Same setup but toggle off: the Keystone may move again.
        let mut cfg = RuleConfig::default();
        cfg.keystone_single_move = false;
        let mut pos = empty_pos_with_ap(2);
        pos.config = cfg;
        place(&mut pos, 4, 4, Piece::new(Player::P1, PieceKind::Keystone, 1));

        let outcome = apply_action(&mut pos, Action::Move { from: sq(4, 4), to: sq(4, 5) })
            .expect("keystone move must be legal");
        assert!(!outcome.turn_ended, "1 AP must remain");

        let dest = sq(4, 5);
        let actions = crate::actions::legal_actions(&pos);
        assert!(
            has_move_from(&actions, dest),
            "toggle off: moved keystone must still be a legal Move source"
        );
    }

    #[test]
    fn first_turn_ap_one_forbids_stack_on_first_turn() {
        // Build the standard position with first_turn_ap=1. P1's first turn has only 1 AP,
        // so Stack (costs 2 AP) must not appear in legal_actions.
        let mut cfg = RuleConfig::default();
        cfg.first_turn_ap = 1;
        let pos = crate::position::Position::new_standard(cfg);
        assert_eq!(pos.turn.ap_remaining, 1, "precondition: ap=1 on first turn");
        let actions = crate::actions::legal_actions(&pos);
        let stacks: Vec<_> = actions.iter().filter(|a| matches!(a, Action::Stack { .. })).collect();
        assert!(stacks.is_empty(), "first_turn_ap=1 must forbid Stack actions");
    }

    #[test]
    fn spire_queen_toggle_changes_legal_targets() {
        // A height-3 Spire at center (4,4). Dragon mode gives 20 targets; Queen gives 32.
        // Assert the Queen set strictly contains the Dragon set and has more targets.
        use crate::config::SpireMode;

        let make_pos = |mode: SpireMode| -> crate::position::Position {
            let mut cfg = RuleConfig::default();
            cfg.spire = mode;
            let mut pos = empty_pos_with_ap(2);
            pos.config = cfg;
            place(&mut pos, 4, 4, Piece::new(Player::P1, PieceKind::Stone, 3));
            pos
        };

        let pos_dragon = make_pos(SpireMode::Dragon);
        let pos_queen = make_pos(SpireMode::Queen);

        let from = sq(4, 4);
        let dragon_actions: Vec<_> = crate::actions::legal_actions(&pos_dragon)
            .into_iter()
            .filter(|a| matches!(a, Action::Move { from: f, .. } if *f == from))
            .collect();
        let queen_actions: Vec<_> = crate::actions::legal_actions(&pos_queen)
            .into_iter()
            .filter(|a| matches!(a, Action::Move { from: f, .. } if *f == from))
            .collect();

        // Dragon: 16 ortho + 4 diag steps = 20. Queen: 16 + 16 = 32.
        assert_eq!(dragon_actions.len(), 20, "Dragon Spire at center must have 20 targets");
        assert_eq!(queen_actions.len(), 32, "Queen Spire at center must have 32 targets");

        // Every Dragon target must also be in Queen.
        for a in &dragon_actions {
            assert!(
                queen_actions.contains(a),
                "Queen must include all Dragon targets; missing {a:?}"
            );
        }
    }

    #[test]
    fn quiet_two_moves_end_turn_on_ap_zero_without_check() {
        let mut pos = empty_pos_with_ap(2);
        // A lone P1 Stone with no enemy Keystones anywhere: nothing can be checked.
        place(&mut pos, 4, 4, Piece::new(Player::P1, PieceKind::Stone, 1));

        // First quiet Move: 2 -> 1 AP, turn continues.
        let first = apply_action(&mut pos, Action::Move { from: sq(4, 4), to: sq(4, 5) })
            .expect("first move must be legal");
        assert!(!first.turn_ended, "one AP remains after the first move");
        assert!(!first.ended_on_check);
        assert_eq!(pos.turn.ap_remaining, 1);
        assert_eq!(pos.to_move, Player::P1, "turn has not advanced yet");

        // Second quiet Move: 1 -> 0 AP, turn ends via AP exhaustion, not check.
        let second = apply_action(&mut pos, Action::Move { from: sq(4, 5), to: sq(4, 6) })
            .expect("second move must be legal");
        assert!(second.turn_ended, "the turn ends when AP reaches zero");
        assert!(!second.ended_on_check, "AP exhaustion is not a check end");
        assert_eq!(pos.to_move, Player::P2, "turn advanced to P2");
        assert_eq!(pos.turn.ap_remaining, DEFAULT_AP, "new turn resets AP");
    }
}
