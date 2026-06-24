//! A single training sample and helpers to build its policy and value targets.

use kairnz_core::actions::Action;
use kairnz_core::outcome::GameResult;
use kairnz_core::piece::Player;
use kairnz_encode::{action_to_index, POLICY_SIZE};

/// One training example: input planes, the MCTS policy target, the game-outcome
/// value target, and the legal-action mask, all for a single position.
#[derive(Clone, Debug, PartialEq)]
pub struct Sample {
    /// Encoded input planes (`NUM_PLANES * 81` floats).
    pub planes: Vec<f32>,
    /// Normalized visit-count policy target (length `POLICY_SIZE`).
    pub policy: Vec<f32>,
    /// Game outcome from this position's side-to-move perspective.
    pub value: f32,
    /// Legal-action mask (length `POLICY_SIZE`), `1` legal else `0`.
    pub legal_mask: Vec<u8>,
}

/// Builds the normalized visit-distribution policy target over `POLICY_SIZE`.
///
/// Each searched action contributes `visits / total_visits` at its policy index.
/// Returns an all-zero vector if there were no visits.
pub fn policy_target(visits: &[(Action, u32)], to_move: Player) -> Vec<f32> {
    let mut policy = vec![0.0f32; POLICY_SIZE];
    let total: u32 = visits.iter().map(|(_, v)| *v).sum();
    if total == 0 {
        return policy;
    }
    let total = total as f32;
    for (action, count) in visits {
        policy[action_to_index(action, to_move)] = *count as f32 / total;
    }
    policy
}

/// Game outcome from `player`'s perspective: win `+1`, loss `-1`, draw `0`.
pub fn outcome_value(player: Player, result: GameResult) -> f32 {
    match result {
        GameResult::Win(winner) if winner == player => 1.0,
        GameResult::Win(_) => -1.0,
        GameResult::Draw(_) => 0.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kairnz_core::outcome::DrawReason;
    use kairnz_core::square::Sq;

    #[test]
    fn policy_target_normalizes_visits() {
        let a = Action::Place { to: Sq(0) };
        let b = Action::Place { to: Sq(1) };
        let policy = policy_target(&[(a, 3), (b, 1)], Player::P1);
        assert_eq!(policy.len(), POLICY_SIZE);
        assert!((policy[action_to_index(&a, Player::P1)] - 0.75).abs() < 1e-6);
        assert!((policy[action_to_index(&b, Player::P1)] - 0.25).abs() < 1e-6);
        let sum: f32 = policy.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "distribution sums to one");
    }

    #[test]
    fn policy_target_empty_is_all_zero() {
        let policy = policy_target(&[], Player::P1);
        assert!(policy.iter().all(|p| *p == 0.0));
    }

    #[test]
    fn outcome_value_is_perspective_relative() {
        assert_eq!(outcome_value(Player::P1, GameResult::Win(Player::P1)), 1.0);
        assert_eq!(outcome_value(Player::P1, GameResult::Win(Player::P2)), -1.0);
        assert_eq!(outcome_value(Player::P1, GameResult::Draw(DrawReason::MaxPlies)), 0.0);
    }
}
