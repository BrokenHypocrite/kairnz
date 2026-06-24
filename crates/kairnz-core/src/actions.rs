use serde::{Deserialize, Serialize};

use crate::movement::move_targets;
use crate::piece::PieceKind;
use crate::position::Position;
use crate::square::{Sq, NUM_SQUARES};

/// AP cost of a Move action.
const COST_MOVE: u8 = 1;
/// AP cost of a Place action.
const COST_PLACE: u8 = 1;
/// AP cost of a Stack action.
const COST_STACK: u8 = 2;

/// Maximum height at which a stone can be stacked upon.
const MAX_STACKABLE_HEIGHT: u8 = 2;

/// An action a player can take on their turn.
#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Debug)]
pub enum Action {
    /// Move a piece from one square to another.
    Move { from: Sq, to: Sq },
    /// Place a piece from reserve onto an empty square.
    Place { to: Sq },
    /// Stack a piece from reserve onto an own Stone, increasing its height.
    Stack { target: Sq },
}

/// Reasons a proposed action may be illegal.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum IllegalAction {
    /// Not enough action points remaining.
    NoAp,
    /// The piece at the source square does not belong to the active player.
    NotYourPiece,
    /// The destination is not reachable by the piece's movement rules.
    BadGeometry,
    /// The destination is occupied by a friendly piece.
    FriendlyOccupied,
    /// No pieces remain in reserve.
    EmptyReserve,
    /// Place target is not an empty square.
    TargetNotEmpty,
    /// The target piece cannot be stacked (is a Keystone or already at max height).
    NotStackable,
    /// The piece is capture-locked and cannot move this turn.
    CaptureLocked,
    /// The Keystone has already moved this turn.
    KeystoneAlreadyMoved,
    /// Stack requires two action points; only one remains.
    NeedsTwoAp,
}

/// Returns the AP cost of an action.
pub fn action_cost(a: &Action) -> u8 {
    match a {
        Action::Move { .. } => COST_MOVE,
        Action::Place { .. } => COST_PLACE,
        Action::Stack { .. } => COST_STACK,
    }
}

/// Returns every legal action available to the player to move in `pos`.
///
/// Honors the AP budget and the `capture_lock` / `keystone_single_move` config toggles.
/// This function only reads `pos.turn.capture_locked` and `pos.turn.keystone_moved`;
/// a later task is responsible for populating those bitboards.
pub fn legal_actions(pos: &Position) -> Vec<Action> {
    let mover = pos.to_move;
    let ap = pos.turn.ap_remaining;
    let reserve = pos.reserves[mover.index()];
    let mut actions: Vec<Action> = Vec::new();

    // Move: costs 1 AP; iterate every square owned by the mover.
    if ap >= COST_MOVE {
        for i in 0..NUM_SQUARES {
            let from = Sq(i as u8);
            let piece = match pos.piece_at(from) {
                Some(pc) if pc.owner == mover => pc,
                _ => continue,
            };

            // capture_lock filter: skip if toggle on and square is locked.
            if pos.config.capture_lock && pos.turn.capture_locked.contains(from) {
                continue;
            }

            // keystone_single_move filter: skip if toggle on, piece is Keystone, and already moved.
            if pos.config.keystone_single_move
                && piece.kind == PieceKind::Keystone
                && pos.turn.keystone_moved.contains(from)
            {
                continue;
            }

            for to in move_targets(pos, from) {
                actions.push(Action::Move { from, to });
            }
        }
    }

    // Place: costs 1 AP and requires at least one reserve piece.
    if ap >= COST_PLACE && reserve > 0 {
        for i in 0..NUM_SQUARES {
            let to = Sq(i as u8);
            if pos.piece_at(to).is_none() {
                actions.push(Action::Place { to });
            }
        }
    }

    // Stack: costs 2 AP and requires at least one reserve piece.
    // Target must be an own Stone (not Keystone) with height < 3.
    if ap >= COST_STACK && reserve > 0 {
        for i in 0..NUM_SQUARES {
            let target = Sq(i as u8);
            if let Some(pc) = pos.piece_at(target) {
                if pc.owner == mover
                    && pc.kind == PieceKind::Stone
                    && pc.height <= MAX_STACKABLE_HEIGHT
                {
                    actions.push(Action::Stack { target });
                }
            }
        }
    }

    actions
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::RuleConfig;
    use crate::piece::{Piece, PieceKind, Player};
    use crate::position::{Position, TurnState};
    use crate::square::{BitBoard81, NUM_SQUARES};

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

    fn has_move(actions: &[Action], from: Sq, to: Sq) -> bool {
        actions.contains(&Action::Move { from, to })
    }

    fn has_stack(actions: &[Action], target: Sq) -> bool {
        actions.iter().any(|a| matches!(a, Action::Stack { target: t } if *t == target))
    }

    #[test]
    fn move_requires_at_least_one_ap() {
        let mut pos = empty_pos_with_ap(0);
        place(&mut pos, 4, 4, Piece::new(Player::P1, PieceKind::Stone, 1));

        let actions = legal_actions(&pos);
        let moves: Vec<_> = actions.iter().filter(|a| matches!(a, Action::Move { .. })).collect();
        assert!(moves.is_empty(), "ap=0 should yield no Move actions");
    }

    #[test]
    fn move_allowed_with_one_ap() {
        let mut pos = empty_pos_with_ap(1);
        place(&mut pos, 4, 4, Piece::new(Player::P1, PieceKind::Stone, 1));

        let actions = legal_actions(&pos);
        let moves: Vec<_> = actions.iter().filter(|a| matches!(a, Action::Move { .. })).collect();
        assert!(!moves.is_empty(), "ap=1 should yield Move actions for a placed stone");
    }

    #[test]
    fn place_requires_reserve_and_empty_square() {
        // reserve 0 -> no Place
        let mut pos = empty_pos_with_ap(2);
        pos.reserves[Player::P1.index()] = 0;
        let actions = legal_actions(&pos);
        let places: Vec<_> = actions.iter().filter(|a| matches!(a, Action::Place { .. })).collect();
        assert!(places.is_empty(), "reserve=0 should yield no Place actions");

        // reserve 1 + one empty square -> exactly one Place
        let mut pos2 = empty_pos_with_ap(2);
        pos2.reserves[Player::P1.index()] = 1;
        // Fill all but one square.
        for i in 0..(NUM_SQUARES - 1) {
            pos2.board[i] = Some(Piece::new(Player::P1, PieceKind::Stone, 1));
        }
        // Last square (index 80) is the only empty one.
        pos2.board[80] = None;
        let actions2 = legal_actions(&pos2);
        let places2: Vec<_> = actions2.iter().filter(|a| matches!(a, Action::Place { .. })).collect();
        assert_eq!(places2.len(), 1, "one empty square with reserve=1 should yield exactly one Place");
        assert!(matches!(places2[0], Action::Place { to } if to.0 == 80));
    }

    #[test]
    fn stack_only_with_two_ap_and_stackable_stone() {
        // ap=1 -> no Stack even with reserve and a stackable stone
        let mut pos = empty_pos_with_ap(1);
        pos.reserves[Player::P1.index()] = 1;
        place(&mut pos, 4, 4, Piece::new(Player::P1, PieceKind::Stone, 1));
        let actions = legal_actions(&pos);
        let stacks: Vec<_> = actions.iter().filter(|a| matches!(a, Action::Stack { .. })).collect();
        assert!(stacks.is_empty(), "ap=1 should yield no Stack actions");

        // ap=2 + reserve + height-1 stone -> Stack present
        let mut pos2 = empty_pos_with_ap(2);
        pos2.reserves[Player::P1.index()] = 1;
        place(&mut pos2, 4, 4, Piece::new(Player::P1, PieceKind::Stone, 1));
        let actions2 = legal_actions(&pos2);
        assert!(has_stack(&actions2, sq(4, 4)), "ap=2 + reserve + h1 stone should yield Stack");

        // ap=2 + reserve + height-2 stone -> Stack present
        let mut pos3 = empty_pos_with_ap(2);
        pos3.reserves[Player::P1.index()] = 1;
        place(&mut pos3, 3, 3, Piece::new(Player::P1, PieceKind::Stone, 2));
        let actions3 = legal_actions(&pos3);
        assert!(has_stack(&actions3, sq(3, 3)), "ap=2 + reserve + h2 stone should yield Stack");

        // height-3 stone -> NOT stackable
        let mut pos4 = empty_pos_with_ap(2);
        pos4.reserves[Player::P1.index()] = 1;
        place(&mut pos4, 3, 3, Piece::new(Player::P1, PieceKind::Stone, 3));
        let actions4 = legal_actions(&pos4);
        assert!(!has_stack(&actions4, sq(3, 3)), "h3 stone should not be a Stack target");

        // Keystone -> NOT a Stack target
        let mut pos5 = empty_pos_with_ap(2);
        pos5.reserves[Player::P1.index()] = 1;
        place(&mut pos5, 4, 4, Piece::new(Player::P1, PieceKind::Keystone, 1));
        let actions5 = legal_actions(&pos5);
        let stacks5: Vec<_> = actions5.iter().filter(|a| matches!(a, Action::Stack { .. })).collect();
        assert!(stacks5.is_empty(), "Keystone must never be a Stack target");
    }

    #[test]
    fn capture_locked_piece_cannot_move_again() {
        let from = sq(4, 4);
        let mut cfg = RuleConfig::default();

        // Toggle ON: locked piece must not appear in Move list.
        cfg.capture_lock = true;
        let mut pos = empty_pos_with_ap(2);
        pos.config = cfg.clone();
        place(&mut pos, 4, 4, Piece::new(Player::P1, PieceKind::Stone, 1));
        pos.turn.capture_locked.set(from);

        let actions = legal_actions(&pos);
        let moves_from: Vec<_> = actions
            .iter()
            .filter(|a| matches!(a, Action::Move { from: f, .. } if *f == from))
            .collect();
        assert!(moves_from.is_empty(), "capture-locked piece must not generate Move actions when toggle is on");

        // Toggle OFF: same locked square should now generate Moves.
        cfg.capture_lock = false;
        let mut pos2 = empty_pos_with_ap(2);
        pos2.config = cfg;
        place(&mut pos2, 4, 4, Piece::new(Player::P1, PieceKind::Stone, 1));
        pos2.turn.capture_locked.set(from);

        let actions2 = legal_actions(&pos2);
        let moves_from2: Vec<_> = actions2
            .iter()
            .filter(|a| matches!(a, Action::Move { from: f, .. } if *f == from))
            .collect();
        assert!(!moves_from2.is_empty(), "capture-locked piece must still generate Move actions when toggle is off");
    }

    #[test]
    fn moved_keystone_cannot_move_again_when_toggle_on() {
        let from = sq(4, 4);

        // Toggle ON: keystone that already moved must not appear in Move list.
        let mut cfg_on = RuleConfig::default();
        cfg_on.keystone_single_move = true;
        let mut pos = empty_pos_with_ap(2);
        pos.config = cfg_on;
        place(&mut pos, 4, 4, Piece::new(Player::P1, PieceKind::Keystone, 1));
        pos.turn.keystone_moved.set(from);

        let actions = legal_actions(&pos);
        let moves_from: Vec<_> = actions
            .iter()
            .filter(|a| matches!(a, Action::Move { from: f, .. } if *f == from))
            .collect();
        assert!(moves_from.is_empty(), "moved keystone must not generate Move actions when toggle is on");

        // Toggle OFF: same keystone should now generate Moves.
        let mut cfg_off = RuleConfig::default();
        cfg_off.keystone_single_move = false;
        let mut pos2 = empty_pos_with_ap(2);
        pos2.config = cfg_off;
        place(&mut pos2, 4, 4, Piece::new(Player::P1, PieceKind::Keystone, 1));
        pos2.turn.keystone_moved.set(from);

        let actions2 = legal_actions(&pos2);
        let moves_from2: Vec<_> = actions2
            .iter()
            .filter(|a| matches!(a, Action::Move { from: f, .. } if *f == from))
            .collect();
        assert!(!moves_from2.is_empty(), "moved keystone should still generate Move actions when toggle is off");
    }

    #[test]
    fn action_cost_values() {
        assert_eq!(action_cost(&Action::Move { from: sq(0, 0), to: sq(0, 1) }), 1);
        assert_eq!(action_cost(&Action::Place { to: sq(0, 0) }), 1);
        assert_eq!(action_cost(&Action::Stack { target: sq(0, 0) }), 2);
    }

    #[test]
    fn only_own_pieces_generate_moves() {
        let mut pos = empty_pos_with_ap(2);
        // Place an enemy stone; it should not generate any Move for P1.
        place(&mut pos, 4, 4, Piece::new(Player::P2, PieceKind::Stone, 1));
        let actions = legal_actions(&pos);
        let moves: Vec<_> = actions.iter().filter(|a| matches!(a, Action::Move { .. })).collect();
        assert!(moves.is_empty(), "enemy piece must not generate Move actions for the active player");
    }

    #[test]
    fn move_targets_not_own_pieces_excluded() {
        let mut pos = empty_pos_with_ap(2);
        let from = sq(4, 4);
        place(&mut pos, 4, 4, Piece::new(Player::P1, PieceKind::Stone, 1));
        // Place a friendly piece directly north; that square must not appear as a Move target.
        place(&mut pos, 4, 5, Piece::new(Player::P1, PieceKind::Stone, 1));
        let actions = legal_actions(&pos);
        let friendly_move = has_move(&actions, from, sq(4, 5));
        assert!(!friendly_move, "Move onto friendly square must not be generated");
    }
}
