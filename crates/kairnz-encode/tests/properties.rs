//! Cross-module properties of the encoding.

use kairnz_core::config::RuleConfig;
use kairnz_core::piece::{Piece, PieceKind, Player};
use kairnz_core::position::{Position, TurnState};
use kairnz_core::square::{BitBoard81, Sq, NUM_SQUARES};
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

/// A lone P2 stone on an asymmetric square must land in the "my" stone plane
/// (channel 0) at the rank-flipped canonical cell. This discriminates a working
/// perspective flip from an encoder that ignores `to_move`: the symmetric-opening
/// test above would pass either way, but this one fails unless encode_planes both
/// assigns the side-to-move's pieces to the "my" channels and applies the rank flip.
#[test]
fn p2_perspective_flips_own_piece_into_my_channel_at_canonical_cell() {
    let mut board = [None; NUM_SQUARES];
    let raw = Sq::new(1, 0).unwrap(); // file 1, rank 0
    board[raw.0 as usize] = Some(Piece::new(Player::P2, PieceKind::Stone, 1));

    let pos = Position {
        board,
        reserves: [0, 0],
        to_move: Player::P2,
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

    // Canonical square for P2 flips rank 0 to rank 8: index 8*9 + 1 = 73.
    let canonical_cell = 8 * 9 + 1;
    let my_stone_h1_channel = 0;
    let opp_stone_h1_channel = 4;

    // The stone appears in the "my" stone-h1 plane at the flipped cell.
    assert_eq!(planes[my_stone_h1_channel * BOARD_CELLS + canonical_cell], 1.0);
    // It does NOT appear at the raw, unflipped cell (which would mean no flip).
    assert_eq!(planes[my_stone_h1_channel * BOARD_CELLS + raw.0 as usize], 0.0);
    // The opponent stone-h1 plane stays empty, since this is P2's own piece.
    let opp_sum: f32 = planes
        [opp_stone_h1_channel * BOARD_CELLS..(opp_stone_h1_channel + 1) * BOARD_CELLS]
        .iter()
        .sum();
    assert_eq!(opp_sum, 0.0);
}
