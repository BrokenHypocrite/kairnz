//! Neural-network encoding for Kairnz: position planes and action indices.
//!
//! This crate is the single source of truth for the AlphaZero encoding contract.
//! All values are produced in a canonical perspective oriented to the side to move.

pub mod canonical;
pub mod action_index;
pub mod mask;
pub mod planes;

pub use action_index::{action_to_index, index_to_action};
pub use canonical::canonical_sq;
pub use mask::legal_mask;
pub use planes::{encode_planes, CH_REPETITION, REPETITION_NORM};

/// Number of 9x9 input planes produced for each position.
pub const NUM_PLANES: usize = 14;

/// Number of squares on the board (9x9).
pub const BOARD_CELLS: usize = kairnz_core::square::NUM_SQUARES;

/// Flat index of the first Move entry in the policy vector.
pub(crate) const MOVE_BASE: usize = 0;
/// Flat index of the first Place entry in the policy vector.
pub(crate) const PLACE_BASE: usize = BOARD_CELLS * BOARD_CELLS;
/// Flat index of the first Stack entry in the policy vector.
pub(crate) const STACK_BASE: usize = PLACE_BASE + BOARD_CELLS;

/// Total length of the policy vector (Move, then Place, then Stack regions).
pub const POLICY_SIZE: usize = STACK_BASE + BOARD_CELLS;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn policy_layout_constants_are_consistent() {
        assert_eq!(BOARD_CELLS, 81);
        assert_eq!(MOVE_BASE, 0);
        assert_eq!(PLACE_BASE, 6561);
        assert_eq!(STACK_BASE, 6642);
        assert_eq!(POLICY_SIZE, 6723);
    }
}
