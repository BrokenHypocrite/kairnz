//! Encodes a position into channel-major neural-network input planes.

use kairnz_core::piece::PieceKind;
use kairnz_core::position::Position;
use kairnz_core::square::Sq;

use crate::canonical::canonical_sq;
use crate::{BOARD_CELLS, NUM_PLANES};

/// Normalizer for the action-points plane (max AP per turn).
const AP_NORM: f32 = 2.0;
/// Soft normalizer for reserve-count planes. Reserves can exceed this in
/// capture-heavy games, so the reserve planes are clamped to [0, 1].
const RESERVE_NORM: f32 = 18.0;
/// Normalizer for the repetition plane (default repetition-fold threshold).
const REPETITION_NORM: f32 = 3.0;

/// Channel offset for the first "my Stone" plane.
const CH_MY_STONE: usize = 0;
/// Channel for the "my Keystone" plane.
const CH_MY_KEYSTONE: usize = 3;
/// Channel offset for the first "opponent Stone" plane.
const CH_OPP_STONE: usize = 4;
/// Channel for the "opponent Keystone" plane.
const CH_OPP_KEYSTONE: usize = 7;
/// Channel for the action-points plane.
const CH_AP: usize = 8;
/// Channel for the "my reserves" plane.
const CH_MY_RESERVE: usize = 9;
/// Channel for the "opponent reserves" plane.
const CH_OPP_RESERVE: usize = 10;
/// Channel for the capture-locked plane.
const CH_CAPTURE_LOCKED: usize = 11;
/// Channel for the keystone-moved plane.
const CH_KEYSTONE_MOVED: usize = 12;
/// Channel for the repetition-count plane.
const CH_REPETITION: usize = 13;

/// Encodes `pos` into `NUM_PLANES * BOARD_CELLS` floats in channel-major order.
///
/// Planes are oriented to the canonical perspective of `pos.to_move`. The caller
/// supplies `repetition_count` (how many times the current position has occurred)
/// because a `Position` alone carries no history.
pub fn encode_planes(pos: &Position, repetition_count: u8) -> Vec<f32> {
    let me = pos.to_move;
    let opp = me.opponent();
    let mut planes = vec![0.0f32; NUM_PLANES * BOARD_CELLS];

    // Piece planes.
    for i in 0..BOARD_CELLS {
        let piece = match pos.board[i] {
            Some(pc) => pc,
            None => continue,
        };
        let cs = canonical_sq(Sq(i as u8), me);
        let is_me = piece.owner == me;
        let channel = match (is_me, piece.kind) {
            (true, PieceKind::Stone) => CH_MY_STONE + (piece.height as usize - 1),
            (true, PieceKind::Keystone) => CH_MY_KEYSTONE,
            (false, PieceKind::Stone) => CH_OPP_STONE + (piece.height as usize - 1),
            (false, PieceKind::Keystone) => CH_OPP_KEYSTONE,
        };
        planes[channel * BOARD_CELLS + cs.0 as usize] = 1.0;
    }

    // Scalar planes broadcast across every cell.
    let ap_val = pos.turn.ap_remaining as f32 / AP_NORM;
    let my_reserve = (pos.reserves[me.index()] as f32 / RESERVE_NORM).min(1.0);
    let opp_reserve = (pos.reserves[opp.index()] as f32 / RESERVE_NORM).min(1.0);
    let rep_val = repetition_count as f32 / REPETITION_NORM;
    for cell in 0..BOARD_CELLS {
        planes[CH_AP * BOARD_CELLS + cell] = ap_val;
        planes[CH_MY_RESERVE * BOARD_CELLS + cell] = my_reserve;
        planes[CH_OPP_RESERVE * BOARD_CELLS + cell] = opp_reserve;
        planes[CH_REPETITION * BOARD_CELLS + cell] = rep_val;
    }

    // Turn-state bitboards, transformed to canonical squares.
    for s in pos.turn.capture_locked.iter() {
        let cs = canonical_sq(s, me);
        planes[CH_CAPTURE_LOCKED * BOARD_CELLS + cs.0 as usize] = 1.0;
    }
    for s in pos.turn.keystone_moved.iter() {
        let cs = canonical_sq(s, me);
        planes[CH_KEYSTONE_MOVED * BOARD_CELLS + cs.0 as usize] = 1.0;
    }

    planes
}

#[cfg(test)]
mod tests {
    use super::*;
    use kairnz_core::config::RuleConfig;

    fn plane_sum(planes: &[f32], channel: usize) -> f32 {
        planes[channel * BOARD_CELLS..(channel + 1) * BOARD_CELLS].iter().sum()
    }

    #[test]
    fn output_has_expected_length() {
        let pos = Position::new_standard(RuleConfig::default());
        assert_eq!(encode_planes(&pos, 0).len(), NUM_PLANES * BOARD_CELLS);
    }

    #[test]
    fn opening_piece_plane_counts() {
        let pos = Position::new_standard(RuleConfig::default());
        let planes = encode_planes(&pos, 0);
        assert_eq!(plane_sum(&planes, CH_MY_STONE), 18.0);
        assert_eq!(plane_sum(&planes, CH_MY_KEYSTONE), 2.0);
        assert_eq!(plane_sum(&planes, CH_OPP_STONE), 18.0);
        assert_eq!(plane_sum(&planes, CH_OPP_KEYSTONE), 2.0);
        // No height-2 or height-3 stones exist at the opening.
        assert_eq!(plane_sum(&planes, CH_MY_STONE + 1), 0.0);
        assert_eq!(plane_sum(&planes, CH_MY_STONE + 2), 0.0);
    }

    #[test]
    fn ap_plane_is_normalized_and_uniform() {
        let pos = Position::new_standard(RuleConfig::default());
        let planes = encode_planes(&pos, 0);
        // Default first-turn AP is 2, normalized to 1.0 across all cells.
        assert!(planes[CH_AP * BOARD_CELLS..(CH_AP + 1) * BOARD_CELLS]
            .iter()
            .all(|v| (*v - 1.0).abs() < 1e-6));
    }

    #[test]
    fn my_keystones_sit_on_canonical_home_squares() {
        let pos = Position::new_standard(RuleConfig::default());
        let planes = encode_planes(&pos, 0);
        // P1 keystones are at rank 1, files 2 and 6; canonical is identity for P1.
        for file in [2usize, 6] {
            let cell = 1 * 9 + file;
            assert_eq!(planes[CH_MY_KEYSTONE * BOARD_CELLS + cell], 1.0);
        }
    }

    #[test]
    fn turn_state_and_repetition_planes_encode_canonically() {
        use kairnz_core::piece::Player;
        use kairnz_core::position::TurnState;
        use kairnz_core::square::{BitBoard81, Sq, NUM_SQUARES};

        // P2 to move so the canonical rank-flip is exercised on the bitboards.
        let locked = Sq::new(1, 0).unwrap(); // canonical (P2) -> rank 8, cell 73
        let moved = Sq::new(3, 0).unwrap(); // canonical (P2) -> rank 8, cell 75
        let mut capture_locked = BitBoard81::default();
        capture_locked.set(locked);
        let mut keystone_moved = BitBoard81::default();
        keystone_moved.set(moved);

        let pos = Position {
            board: [None; NUM_SQUARES],
            reserves: [0, 0],
            to_move: Player::P2,
            turn: TurnState {
                ap_remaining: 2,
                capture_locked,
                keystone_moved,
                enemy_checked_at_start: BitBoard81::default(),
            },
            config: RuleConfig::default(),
            zobrist: 0,
            ply: 0,
        };

        let planes = encode_planes(&pos, 3);

        // capture_locked -> channel 11 at flipped cell 8*9 + 1 = 73.
        assert_eq!(plane_sum(&planes, CH_CAPTURE_LOCKED), 1.0);
        assert_eq!(planes[CH_CAPTURE_LOCKED * BOARD_CELLS + (8 * 9 + 1)], 1.0);
        // keystone_moved -> channel 12 at flipped cell 8*9 + 3 = 75.
        assert_eq!(plane_sum(&planes, CH_KEYSTONE_MOVED), 1.0);
        assert_eq!(planes[CH_KEYSTONE_MOVED * BOARD_CELLS + (8 * 9 + 3)], 1.0);
        // repetition_count 3 normalized by 3.0 -> 1.0 uniformly on channel 13.
        assert!(planes[CH_REPETITION * BOARD_CELLS..(CH_REPETITION + 1) * BOARD_CELLS]
            .iter()
            .all(|v| (*v - 1.0).abs() < 1e-6));
    }

    #[test]
    fn reserve_plane_is_clamped_to_one() {
        use kairnz_core::piece::Player;
        use kairnz_core::position::TurnState;
        use kairnz_core::square::{BitBoard81, NUM_SQUARES};

        let pos = Position {
            board: [None; NUM_SQUARES],
            reserves: [99, 0], // far exceeds RESERVE_NORM
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
        };

        let planes = encode_planes(&pos, 0);
        // 99 / 18 clamps to 1.0 across the whole my-reserve plane.
        assert!(planes[CH_MY_RESERVE * BOARD_CELLS..(CH_MY_RESERVE + 1) * BOARD_CELLS]
            .iter()
            .all(|v| (*v - 1.0).abs() < 1e-6));
    }
}
