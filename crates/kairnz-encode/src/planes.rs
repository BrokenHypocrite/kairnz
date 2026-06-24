//! Encodes a position into channel-major neural-network input planes.

use kairnz_core::piece::PieceKind;
use kairnz_core::position::Position;
use kairnz_core::square::Sq;

use crate::canonical::canonical_sq;
use crate::{BOARD_CELLS, NUM_PLANES};

/// Normalizer for the action-points plane (max AP per turn).
const AP_NORM: f32 = 2.0;
/// Normalizer for reserve-count planes (an upper bound on a player's stones).
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
    let my_reserve = pos.reserves[me.index()] as f32 / RESERVE_NORM;
    let opp_reserve = pos.reserves[opp.index()] as f32 / RESERVE_NORM;
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
}
