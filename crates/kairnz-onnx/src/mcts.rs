//! Neural-guided PUCT Monte Carlo Tree Search over Kairnz positions.

use kairnz_core::actions::Action;
use kairnz_core::outcome::GameResult;
use kairnz_core::piece::Player;
use kairnz_encode::action_to_index;

/// Default number of simulations per move.
const DEFAULT_SIMULATIONS: u32 = 400;
/// Default PUCT exploration constant.
const DEFAULT_C_PUCT: f64 = 1.5;
/// Default Dirichlet concentration for root exploration noise.
const DEFAULT_DIRICHLET_ALPHA: f64 = 0.3;
/// Default root-noise weight. Zero disables noise, making search deterministic.
const DEFAULT_DIRICHLET_EPSILON: f64 = 0.0;

/// Terminal value of a win from the winning side's perspective.
const WIN_VALUE: f64 = 1.0;
/// Terminal value of a loss from the losing side's perspective.
const LOSS_VALUE: f64 = -1.0;
/// Terminal value of a draw.
const DRAW_VALUE: f64 = 0.0;

/// Search parameters for [`AzMctsPolicy`].
#[derive(Clone, Copy, Debug)]
pub struct AzMctsConfig {
    /// Number of simulations performed per move.
    pub simulations: u32,
    /// PUCT exploration constant.
    pub c_puct: f64,
    /// Dirichlet concentration parameter for root noise.
    pub dirichlet_alpha: f64,
    /// Root-noise mixing weight in `[0, 1]`; `0.0` disables noise.
    pub dirichlet_epsilon: f64,
}

impl Default for AzMctsConfig {
    fn default() -> Self {
        AzMctsConfig {
            simulations: DEFAULT_SIMULATIONS,
            c_puct: DEFAULT_C_PUCT,
            dirichlet_alpha: DEFAULT_DIRICHLET_ALPHA,
            dirichlet_epsilon: DEFAULT_DIRICHLET_EPSILON,
        }
    }
}

/// Terminal value of `result` from `to_move`'s perspective, in `[-1, 1]`.
pub(crate) fn terminal_value(to_move: Player, result: GameResult) -> f64 {
    match result {
        GameResult::Win(winner) if winner == to_move => WIN_VALUE,
        GameResult::Win(_) => LOSS_VALUE,
        GameResult::Draw(_) => DRAW_VALUE,
    }
}

/// Softmax priors over only the legal actions, aligned to `legal`'s order.
///
/// Each legal action's logit is read from the policy vector via
/// `action_to_index`, then a numerically stable softmax is applied. The result
/// sums to approximately 1 and is used as the PUCT prior for each child.
pub(crate) fn legal_priors(logits: &[f32], legal: &[Action], to_move: Player) -> Vec<f32> {
    if legal.is_empty() {
        return Vec::new();
    }
    let raw: Vec<f32> = legal
        .iter()
        .map(|a| logits[action_to_index(a, to_move)])
        .collect();
    let max = raw.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = raw.iter().map(|x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|e| e / sum).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use kairnz_core::square::Sq;
    use kairnz_encode::POLICY_SIZE;

    #[test]
    fn terminal_value_is_perspective_relative() {
        assert_eq!(terminal_value(Player::P1, GameResult::Win(Player::P1)), 1.0);
        assert_eq!(terminal_value(Player::P1, GameResult::Win(Player::P2)), -1.0);
        assert_eq!(
            terminal_value(Player::P1, GameResult::Draw(kairnz_core::outcome::DrawReason::MaxPlies)),
            0.0
        );
    }

    #[test]
    fn legal_priors_softmax_sums_to_one_over_legal() {
        let mut logits = vec![0.0f32; POLICY_SIZE];
        let a = Action::Place { to: Sq(0) };
        let b = Action::Place { to: Sq(1) };
        logits[action_to_index(&a, Player::P1)] = 2.0;
        logits[action_to_index(&b, Player::P1)] = 0.0;

        let priors = legal_priors(&logits, &[a, b], Player::P1);
        assert_eq!(priors.len(), 2);
        let sum: f32 = priors.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "priors sum to one");
        assert!(priors[0] > priors[1], "higher logit gets higher prior");
    }

    #[test]
    fn legal_priors_empty_for_no_actions() {
        assert!(legal_priors(&[0.0; POLICY_SIZE], &[], Player::P1).is_empty());
    }
}
