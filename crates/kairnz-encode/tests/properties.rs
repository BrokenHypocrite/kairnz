//! Cross-module properties of the encoding.

use kairnz_core::config::RuleConfig;
use kairnz_core::piece::Player;
use kairnz_core::position::Position;
use kairnz_encode::{encode_planes, BOARD_CELLS};

/// The standard opening is symmetric under a rank flip plus a side swap, so its
/// canonical piece planes must be identical whether P1 or P2 is to move.
#[test]
fn opening_is_perspective_invariant_on_piece_planes() {
    let p1_to_move = Position::new_standard(RuleConfig::default());
    let mut p2_to_move = p1_to_move.clone();
    p2_to_move.to_move = Player::P2;

    let e1 = encode_planes(&p1_to_move, 0);
    let e2 = encode_planes(&p2_to_move, 0);

    // Channels 0..8 are the eight piece planes (my/opp x Stone h1,h2,h3 + Keystone).
    for channel in 0..8 {
        for cell in 0..BOARD_CELLS {
            let i = channel * BOARD_CELLS + cell;
            assert!(
                (e1[i] - e2[i]).abs() < 1e-6,
                "channel {channel} cell {cell} differs: {} vs {}",
                e1[i],
                e2[i]
            );
        }
    }
}
