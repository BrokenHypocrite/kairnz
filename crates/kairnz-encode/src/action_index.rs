//! Bijection between game actions and fixed policy-vector indices.

use kairnz_core::actions::Action;
use kairnz_core::piece::Player;
use kairnz_core::square::{Sq, NUM_SQUARES};

use crate::canonical::canonical_sq;
use crate::{MOVE_BASE, PLACE_BASE, POLICY_SIZE, STACK_BASE};

/// Maps a legal-or-illegal action to its flat policy-vector index.
///
/// Squares are taken into the canonical perspective of `me` first, so the index
/// space is perspective-invariant.
pub fn action_to_index(a: &Action, me: Player) -> usize {
    match a {
        Action::Move { from, to } => {
            let f = canonical_sq(*from, me).0 as usize;
            let t = canonical_sq(*to, me).0 as usize;
            MOVE_BASE + f * NUM_SQUARES + t
        }
        Action::Place { to } => PLACE_BASE + canonical_sq(*to, me).0 as usize,
        Action::Stack { target } => STACK_BASE + canonical_sq(*target, me).0 as usize,
    }
}

/// Inverse of [`action_to_index`]; returns `None` for out-of-range indices.
pub fn index_to_action(index: usize, me: Player) -> Option<Action> {
    if index >= POLICY_SIZE {
        return None;
    }
    if index >= STACK_BASE {
        let canonical = Sq((index - STACK_BASE) as u8);
        Some(Action::Stack { target: canonical_sq(canonical, me) })
    } else if index >= PLACE_BASE {
        let canonical = Sq((index - PLACE_BASE) as u8);
        Some(Action::Place { to: canonical_sq(canonical, me) })
    } else {
        let rel = index - MOVE_BASE;
        let from_canonical = Sq((rel / NUM_SQUARES) as u8);
        let to_canonical = Sq((rel % NUM_SQUARES) as u8);
        Some(Action::Move {
            from: canonical_sq(from_canonical, me),
            to: canonical_sq(to_canonical, me),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn place_and_stack_land_in_expected_ranges() {
        let me = Player::P1;
        assert_eq!(action_to_index(&Action::Place { to: Sq(0) }, me), 6561);
        assert_eq!(action_to_index(&Action::Stack { target: Sq(0) }, me), 6642);
        assert_eq!(index_to_action(6561, me), Some(Action::Place { to: Sq(0) }));
        assert_eq!(index_to_action(6642, me), Some(Action::Stack { target: Sq(0) }));
    }

    #[test]
    fn out_of_range_index_is_none() {
        assert_eq!(index_to_action(POLICY_SIZE, Player::P1), None);
    }

    #[test]
    fn full_bijection_both_perspectives() {
        for me in [Player::P1, Player::P2] {
            for idx in 0..POLICY_SIZE {
                let action = index_to_action(idx, me).expect("index in range decodes");
                assert_eq!(action_to_index(&action, me), idx, "idx {idx} for {me:?}");
            }
        }
    }
}
