//! Legal-action mask over the policy vector.

use kairnz_core::actions::legal_actions;
use kairnz_core::position::Position;

use crate::action_index::action_to_index;
use crate::POLICY_SIZE;

/// Builds a boolean mask of length `POLICY_SIZE` marking every legal action.
///
/// Indices are computed in the canonical perspective of `pos.to_move`, matching
/// the policy head's output orientation.
pub fn legal_mask(pos: &Position) -> Vec<bool> {
    let mut mask = vec![false; POLICY_SIZE];
    for action in legal_actions(pos) {
        mask[action_to_index(&action, pos.to_move)] = true;
    }
    mask
}

#[cfg(test)]
mod tests {
    use super::*;
    use kairnz_core::actions::legal_actions;
    use kairnz_core::config::RuleConfig;
    use kairnz_core::position::Position;

    use crate::action_index::action_to_index;

    #[test]
    fn mask_has_policy_size_length() {
        let pos = Position::new_standard(RuleConfig::default());
        assert_eq!(legal_mask(&pos).len(), POLICY_SIZE);
    }

    #[test]
    fn mask_matches_legal_actions_without_collisions() {
        let pos = Position::new_standard(RuleConfig::default());
        let mask = legal_mask(&pos);
        let actions = legal_actions(&pos);

        // Every distinct legal action occupies a distinct index, so the count of
        // set bits equals the number of legal actions.
        let set_count = mask.iter().filter(|bit| **bit).count();
        assert_eq!(set_count, actions.len());

        for action in &actions {
            assert!(mask[action_to_index(action, pos.to_move)]);
        }
    }
}
