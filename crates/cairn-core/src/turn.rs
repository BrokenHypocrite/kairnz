use crate::check::checked_enemy_keystone_squares;
use crate::config::DEFAULT_AP;
use crate::position::Position;
use crate::square::BitBoard81;

/// Advances `pos` to the next player's turn.
///
/// Flips `to_move`, refreshes the AP budget to `DEFAULT_AP`, clears the per-turn
/// toggle bitboards, and recomputes `enemy_checked_at_start` from the NEW mover's
/// perspective (which of the new opponent's Keystones are already in check at the
/// start of this turn).
///
/// This does NOT recompute the Zobrist hash. Side-to-move folding for repetition
/// detection is captured by the caller at turn boundaries in a later task; here we
/// only update side-to-move and turn bookkeeping.
pub fn advance_turn(pos: &mut Position) {
    pos.to_move = pos.to_move.opponent();
    pos.turn.ap_remaining = DEFAULT_AP;
    pos.turn.capture_locked = BitBoard81::default();
    pos.turn.keystone_moved = BitBoard81::default();
    pos.turn.enemy_checked_at_start = checked_enemy_keystone_squares(pos, pos.to_move);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::RuleConfig;
    use crate::piece::{Piece, PieceKind, Player};
    use crate::position::{Position, TurnState};
    use crate::square::{BitBoard81, Sq, NUM_SQUARES};

    fn empty_pos() -> Position {
        Position {
            board: [None; NUM_SQUARES],
            reserves: [0, 0],
            to_move: Player::P1,
            turn: TurnState {
                ap_remaining: 0,
                capture_locked: BitBoard81::default(),
                keystone_moved: BitBoard81::default(),
                enemy_checked_at_start: BitBoard81::default(),
            },
            config: RuleConfig::default(),
            zobrist: 0,
            ply: 0,
        }
    }

    fn place(pos: &mut Position, file: u8, rank: u8, piece: Piece) {
        let sq = Sq::new(file, rank).unwrap();
        pos.board[sq.0 as usize] = Some(piece);
    }

    #[test]
    fn advance_turn_resets_ap_and_clears_toggle_bitboards_and_flips_side() {
        let mut pos = empty_pos();
        // Dirty the per-turn state so we can observe it being cleared.
        let locked = Sq::new(0, 0).unwrap();
        let moved = Sq::new(1, 1).unwrap();
        pos.turn.ap_remaining = 0;
        pos.turn.capture_locked.set(locked);
        pos.turn.keystone_moved.set(moved);

        advance_turn(&mut pos);

        assert_eq!(pos.to_move, Player::P2, "side to move must flip");
        assert_eq!(pos.turn.ap_remaining, DEFAULT_AP, "AP must reset to DEFAULT_AP");
        assert!(pos.turn.capture_locked.is_empty(), "capture_locked must be cleared");
        assert!(pos.turn.keystone_moved.is_empty(), "keystone_moved must be cleared");
    }

    #[test]
    fn advance_turn_recomputes_enemy_checked_for_new_mover() {
        // After advance, the new mover is P2; P2's opponent (P1) has a Keystone
        // already attacked by a P2 piece -> that square enters enemy_checked_at_start.
        let mut pos = empty_pos();
        pos.to_move = Player::P1;
        let p1_keystone = Sq::new(4, 4).unwrap();
        place(&mut pos, 4, 4, Piece::new(Player::P1, PieceKind::Keystone, 1));
        // P2 Pillar adjacent to the P1 Keystone.
        place(&mut pos, 4, 5, Piece::new(Player::P2, PieceKind::Stone, 2));

        advance_turn(&mut pos);

        assert_eq!(pos.to_move, Player::P2);
        assert!(
            pos.turn.enemy_checked_at_start.contains(p1_keystone),
            "new mover P2 already checks P1's Keystone at turn start"
        );
    }
}
