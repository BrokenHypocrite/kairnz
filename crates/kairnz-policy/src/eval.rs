use kairnz_core::{
    movement::move_targets,
    piece::{PieceKind, Player},
    position::Position,
    square::NUM_SQUARES,
};

/// Material value of a Keystone. Dominates all other terms so that
/// capturing a Keystone is always preferred.
const KEYSTONE_VALUE: i32 = 1000;

/// Base material value of a Stone at height 1.
const STONE_BASE: i32 = 100;

/// Extra value granted per height level above 1.
/// A height-h Stone is worth `STONE_BASE + (h - 1) * HEIGHT_BONUS`.
const HEIGHT_BONUS: i32 = 40;

/// Value of each piece held in reserve.
const RESERVE_VALUE: i32 = 30;

/// Mobility weight: added once per piece that has at least one geometric
/// move target, regardless of whose turn it is.
const MOBILITY_WEIGHT: i32 = 1;

/// Evaluates a `Position` from `perspective`'s point of view.
///
/// Returns a positive score when the position favours `perspective` and a
/// negative score when it favours the opponent. The evaluation is symmetric:
/// `evaluate(pos, P1) == -evaluate(pos, P2)` for any position where both
/// sides are mirror images of each other.
///
/// Terms included:
/// - Per-piece material (Keystone, Stone at its height).
/// - Reserves held by each side.
/// - Light mobility: count of own pieces that have at least one geometric
///   target minus the same count for the opponent.
pub fn evaluate(pos: &Position, perspective: Player) -> i32 {
    let opponent = perspective.opponent();

    let mut score: i32 = 0;

    // Material and mobility.
    let mut perspective_mobile: i32 = 0;
    let mut opponent_mobile: i32 = 0;

    for i in 0..NUM_SQUARES {
        let sq = kairnz_core::square::Sq(i as u8);
        let piece = match pos.board[i] {
            Some(p) => p,
            None => continue,
        };

        let piece_value = match piece.kind {
            PieceKind::Keystone => KEYSTONE_VALUE,
            PieceKind::Stone => STONE_BASE + (piece.height as i32 - 1) * HEIGHT_BONUS,
        };

        if piece.owner == perspective {
            score += piece_value;
            if !move_targets(pos, sq).is_empty() {
                perspective_mobile += 1;
            }
        } else {
            debug_assert_eq!(piece.owner, opponent);
            score -= piece_value;
            if !move_targets(pos, sq).is_empty() {
                opponent_mobile += 1;
            }
        }
    }

    // Reserves.
    score += RESERVE_VALUE * pos.reserves[perspective.index()] as i32;
    score -= RESERVE_VALUE * pos.reserves[opponent.index()] as i32;

    // Mobility term.
    score += MOBILITY_WEIGHT * (perspective_mobile - opponent_mobile);

    score
}

#[cfg(test)]
mod tests {
    use super::*;
    use kairnz_core::{config::RuleConfig, game::Game, piece::Player};

    #[test]
    fn evaluate_is_symmetric_on_initial_position() {
        let game = Game::new_standard(RuleConfig::default());
        let p1_score = evaluate(&game.pos, Player::P1);
        let p2_score = evaluate(&game.pos, Player::P2);
        assert_eq!(
            p1_score, -p2_score,
            "evaluate(P1) must equal -evaluate(P2) on the mirror-symmetric start"
        );
    }
}
