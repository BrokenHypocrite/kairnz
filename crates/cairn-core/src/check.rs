use crate::piece::Player;
use crate::position::Position;
use crate::square::{BitBoard81, Sq, NUM_SQUARES};
use crate::movement::move_targets;

// WHY square-anchored: during the mover's turn the defender cannot move, so
// the defender's Keystone squares are fixed. Anchoring "in check at turn start"
// to SQUARES is stable across the whole turn (the only change is removal via
// capture). A positional [bool; 2] array would shift slot meaning when one
// Keystone is captured and could mis-fire the turn-ending rule.

/// Returns true if any piece owned by `by` has `keystone_sq` among its move targets.
///
/// Only pieces belonging to `by` are considered as attackers. This means calling
/// `is_in_check(pos, ks, enemy)` correctly ignores the keystone owner's own pieces.
pub fn is_in_check(pos: &Position, keystone_sq: Sq, by: Player) -> bool {
    for i in 0..NUM_SQUARES {
        if let Some(piece) = pos.board[i] {
            if piece.owner != by {
                continue;
            }
            // `i` is always < NUM_SQUARES here, so from_index never returns None; the guard keeps us panic-free.
            if let Some(attacker_sq) = Sq::from_index(i as u8) {
                if move_targets(pos, attacker_sq).contains(&keystone_sq) {
                    return true;
                }
            }
        }
    }
    false
}

/// Returns the set of squares holding `attacker.opponent()`'s Keystones that
/// are currently in check by `attacker`.
///
/// Iterates the defender's keystone squares and includes each in the result
/// bitboard for which `is_in_check(pos, sq, attacker)` is true.
pub fn checked_enemy_keystone_squares(pos: &Position, attacker: Player) -> BitBoard81 {
    let defender = attacker.opponent();
    let mut result = BitBoard81::default();
    for sq in pos.keystones_of(defender) {
        if is_in_check(pos, sq, attacker) {
            result.set(sq);
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::RuleConfig;
    use crate::piece::{Piece, PieceKind};
    use crate::position::TurnState;

    fn empty_pos() -> Position {
        Position {
            board: [None; NUM_SQUARES],
            reserves: [0, 0],
            to_move: Player::P1,
            turn: TurnState {
                ap_remaining: 2,
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

    /// A height-2 Stone (Pillar) steps in all 8 directions; placing one adjacent
    /// to a Keystone means the Keystone square is among its move targets.
    #[test]
    fn keystone_threatened_by_adjacent_pillar_is_in_check() {
        let mut pos = empty_pos();
        // P2 Keystone at (4, 4).
        let ks_sq = Sq::new(4, 4).unwrap();
        place(&mut pos, 4, 4, Piece::new(Player::P2, PieceKind::Keystone, 1));
        // P1 Pillar (height-2 Stone) one step north at (4, 5) -- can step in all 8 dirs.
        place(&mut pos, 4, 5, Piece::new(Player::P1, PieceKind::Stone, 2));

        assert!(is_in_check(&pos, ks_sq, Player::P1));
    }

    /// An isolated Keystone with no enemy pieces nearby must not be in check.
    #[test]
    fn keystone_not_threatened_is_not_in_check() {
        let mut pos = empty_pos();
        let ks_sq = Sq::new(4, 4).unwrap();
        place(&mut pos, 4, 4, Piece::new(Player::P2, PieceKind::Keystone, 1));
        // No P1 pieces anywhere; P2 cannot check its own Keystone.

        assert!(!is_in_check(&pos, ks_sq, Player::P1));
    }

    /// A Dragon Spire (height-3 Stone, default SpireMode::Dragon) slides
    /// orthogonally; a clear rank between it and the Keystone means in check.
    #[test]
    fn dragon_slides_to_threaten_keystone_across_empty_rank() {
        let mut pos = empty_pos();
        // P2 Keystone at (4, 4).
        let ks_sq = Sq::new(4, 4).unwrap();
        place(&mut pos, 4, 4, Piece::new(Player::P2, PieceKind::Keystone, 1));
        // P1 Dragon Spire at (4, 0) -- clear orthogonal ray north to (4, 4).
        place(&mut pos, 4, 0, Piece::new(Player::P1, PieceKind::Stone, 3));

        assert!(is_in_check(&pos, ks_sq, Player::P1));
    }

    /// checked_enemy_keystone_squares must mark only the threatened Keystone,
    /// not the safe one.
    #[test]
    fn checked_enemy_keystone_squares_marks_only_threatened_keystones() {
        let mut pos = empty_pos();
        // Two P2 Keystones: one in check, one safe.
        let threatened_sq = Sq::new(4, 4).unwrap();
        let safe_sq = Sq::new(0, 8).unwrap();
        place(&mut pos, 4, 4, Piece::new(Player::P2, PieceKind::Keystone, 1));
        place(&mut pos, 0, 8, Piece::new(Player::P2, PieceKind::Keystone, 1));

        // P1 Pillar adjacent to threatened Keystone only.
        place(&mut pos, 4, 5, Piece::new(Player::P1, PieceKind::Stone, 2));

        let checked = checked_enemy_keystone_squares(&pos, Player::P1);
        assert!(checked.contains(threatened_sq), "threatened keystone must be in set");
        assert!(!checked.contains(safe_sq), "safe keystone must not be in set");
    }

    /// A friendly piece threatening your own Keystone square must NOT count
    /// when querying check by the enemy. is_in_check(pos, ks, enemy) only
    /// considers enemy-owned attackers.
    #[test]
    fn friendly_piece_does_not_count_as_enemy_attacker() {
        let mut pos = empty_pos();
        let ks_sq = Sq::new(4, 4).unwrap();
        place(&mut pos, 4, 4, Piece::new(Player::P2, PieceKind::Keystone, 1));
        // P2's own Pillar adjacent -- should NOT cause in-check from P1's perspective.
        place(&mut pos, 4, 5, Piece::new(Player::P2, PieceKind::Stone, 2));

        assert!(!is_in_check(&pos, ks_sq, Player::P1));
    }
}
