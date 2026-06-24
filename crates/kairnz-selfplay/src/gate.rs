//! Model-vs-model gating: play a candidate ONNX against a best ONNX and score it.

use std::path::Path;

use kairnz_core::config::RuleConfig;
use kairnz_core::game::Game;
use kairnz_core::outcome::{DrawReason, GameResult};
use kairnz_core::piece::Player;
use kairnz_onnx::{AzMctsConfig, AzMctsPolicy};
use kairnz_policy::policy::Policy;

/// Plays one game from the standard opening between `p1` (P1) and `p2` (P2),
/// returning the terminal result. Never panics: a `None` choice or an illegal
/// action defensively ends the game.
pub fn play_match(config: RuleConfig, p1: &mut dyn Policy, p2: &mut dyn Policy) -> GameResult {
    let mut game = Game::new_standard(config);
    while game.terminal_result().is_none() {
        let mover = game.pos.to_move;
        let policy: &mut dyn Policy = if mover == Player::P1 { p1 } else { p2 };
        match policy.choose(&game) {
            Some(action) => {
                if game.apply(action).is_err() {
                    break;
                }
            }
            None => break,
        }
    }
    game.terminal_result().unwrap_or(GameResult::Draw(DrawReason::MaxPlies))
}

/// Tally of a gate match from model A's perspective.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GateResult {
    /// Games won by model A.
    pub a_wins: u32,
    /// Games won by model B.
    pub b_wins: u32,
    /// Drawn games.
    pub draws: u32,
}

impl GateResult {
    /// Model A's score: wins plus half-credit for draws, over all games. Returns
    /// 0.0 when no games were played.
    pub fn a_score(&self) -> f64 {
        let total = self.a_wins + self.b_wins + self.draws;
        if total == 0 {
            return 0.0;
        }
        (self.a_wins as f64 + 0.5 * self.draws as f64) / total as f64
    }
}

/// Plays `games` gate games between the models at `model_a` and `model_b`,
/// alternating which model plays P1 to cancel first-player bias, and returns the
/// tally from model A's perspective.
///
/// Both sides use `AzMctsPolicy` with `config` (which should carry a small
/// Dirichlet epsilon so games vary). The models are loaded once and reused.
pub fn run_gate(
    model_a: &Path,
    model_b: &Path,
    games: u32,
    config: AzMctsConfig,
    rule: RuleConfig,
    base_seed: u64,
) -> ort::Result<GateResult> {
    let mut policy_a = AzMctsPolicy::from_path(model_a, config, base_seed)?;
    let mut policy_b = AzMctsPolicy::from_path(model_b, config, base_seed.wrapping_add(1))?;

    let mut result = GateResult { a_wins: 0, b_wins: 0, draws: 0 };
    for g in 0..games {
        let a_is_p1 = g % 2 == 0;
        let outcome = if a_is_p1 {
            play_match(rule.clone(), &mut policy_a, &mut policy_b)
        } else {
            play_match(rule.clone(), &mut policy_b, &mut policy_a)
        };
        match outcome {
            GameResult::Win(winner) => {
                let a_player = if a_is_p1 { Player::P1 } else { Player::P2 };
                if winner == a_player {
                    result.a_wins += 1;
                } else {
                    result.b_wins += 1;
                }
            }
            GameResult::Draw(_) => result.draws += 1,
        }
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn fixture() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../kairnz-onnx/tests/fixtures/random_init.onnx")
    }

    fn gate_config() -> AzMctsConfig {
        // Small sims keep the test fast; epsilon > 0 makes games vary by seed.
        AzMctsConfig { simulations: 8, dirichlet_epsilon: 0.15, ..AzMctsConfig::default() }
    }

    fn fast_rule() -> RuleConfig {
        // Short games keep the gate test fast; the gate logic is unchanged.
        RuleConfig { max_plies: 30, ..RuleConfig::default() }
    }

    #[test]
    fn a_score_counts_draws_as_half() {
        let r = GateResult { a_wins: 3, b_wins: 1, draws: 2 };
        assert!((r.a_score() - (3.0 + 1.0) / 6.0).abs() < 1e-9);
    }

    #[test]
    fn gate_tally_sums_to_games_and_is_reproducible() {
        let path = fixture();
        let games = 2;
        let r1 = run_gate(&path, &path, games, gate_config(), fast_rule(), 7)
            .expect("gate runs");
        assert_eq!(r1.a_wins + r1.b_wins + r1.draws, games, "tally sums to games");
        assert!((0.0..=1.0).contains(&r1.a_score()));

        let r2 = run_gate(&path, &path, games, gate_config(), fast_rule(), 7)
            .expect("gate runs");
        assert_eq!(r1, r2, "same seed yields the same gate result");
    }
}
