//! Canonical board orientation relative to the side to move.

use kairnz_core::piece::Player;
use kairnz_core::square::{Sq, BOARD_SIZE};

/// Maps a board square into the canonical perspective of `me`.
///
/// P1 is the identity. P2 flips the rank to `BOARD_SIZE - 1 - rank` so that the
/// side to move always sees its home rows at the bottom. The transform is its
/// own inverse, so applying it twice returns the original square.
pub fn canonical_sq(s: Sq, me: Player) -> Sq {
    match me {
        Player::P1 => s,
        Player::P2 => {
            let flipped_rank = (BOARD_SIZE - 1) - s.rank();
            Sq::new(s.file(), flipped_rank).expect("rank flip stays in bounds")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kairnz_core::square::NUM_SQUARES;

    #[test]
    fn identity_for_p1() {
        for i in 0..NUM_SQUARES as u8 {
            assert_eq!(canonical_sq(Sq(i), Player::P1), Sq(i));
        }
    }

    #[test]
    fn involution_for_p2() {
        for i in 0..NUM_SQUARES as u8 {
            let s = Sq(i);
            assert_eq!(canonical_sq(canonical_sq(s, Player::P2), Player::P2), s);
        }
    }

    #[test]
    fn flips_rank_keeps_file_for_p2() {
        let s = Sq::new(2, 1).unwrap();
        let c = canonical_sq(s, Player::P2);
        assert_eq!((c.file(), c.rank()), (2, 7));
    }
}
