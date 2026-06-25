//! Strength-vs-baselines harness: plays a trained ONNX model against each of
//! three fixed baselines (`random`, `greedy`, `mcts`) over N games each,
//! alternating colors to cancel first-player bias, and prints one JSON line per
//! baseline with the model's win/draw/loss tally and score.

use std::path::PathBuf;
use std::process::ExitCode;

use clap::Parser;
use kairnz_core::config::RuleConfig;
use kairnz_core::game::Game;
use kairnz_core::outcome::{DrawReason, GameResult};
use kairnz_core::piece::Player;
use kairnz_onnx::{AzMctsConfig, AzMctsPolicy};
use kairnz_policy::{
    greedy::GreedyPolicy, mcts::MctsPolicy, policy::Policy, random::RandomPolicy,
};

/// Default MCTS iterations for the MctsPolicy baseline.
const DEFAULT_MCTS_BASELINE_ITERS: u32 = 50;

/// Command-line arguments for the strength harness.
#[derive(Parser)]
#[command(about = "Evaluate a trained Kairnz model against baseline policies.")]
struct Args {
    /// Path to the ONNX model under evaluation.
    #[arg(long)]
    model: PathBuf,
    /// Number of games to play against each baseline.
    #[arg(long, default_value_t = 50)]
    games: u32,
    /// MCTS simulations per move for the neural model.
    #[arg(long, default_value_t = 400)]
    simulations: u32,
    /// Base RNG seed (each game and baseline offsets from this).
    #[arg(long, default_value_t = 0)]
    seed: u64,
}

/// Per-baseline tally from the model's perspective.
#[derive(Debug, Default)]
struct MatchResult {
    wins: u32,
    draws: u32,
    losses: u32,
}

impl MatchResult {
    /// Model's score: wins + 0.5 * draws over total games.
    ///
    /// Returns 0.0 when no games were played.
    fn score(&self) -> f64 {
        let total = self.wins + self.draws + self.losses;
        if total == 0 {
            return 0.0;
        }
        (self.wins as f64 + 0.5 * self.draws as f64) / total as f64
    }
}

/// Drives a single game between `p1` (Player 1) and `p2` (Player 2) using the
/// `Policy` interface. Returns the terminal `GameResult`. Applies a max-ply
/// guard via `rule.max_plies` (already encoded in the `Game`); additionally
/// guards against a policy returning `None` with no terminal result.
fn play_game(rule: RuleConfig, p1: &mut dyn Policy, p2: &mut dyn Policy) -> GameResult {
    let mut game = Game::new_standard(rule);
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

/// Plays `games` matches between `model_policy` and `baseline`, alternating
/// colors, and returns the tally from the model's perspective.
///
/// For game index `g`:
/// - even `g`: model is P1, baseline is P2
/// - odd `g`: baseline is P1, model is P2
///
/// The seed offsets ensure each game gets a unique RNG starting point.
fn run_match(
    games: u32,
    rule: RuleConfig,
    make_model: &mut dyn FnMut(u64) -> Box<dyn Policy>,
    make_baseline: &mut dyn FnMut(u64) -> Box<dyn Policy>,
    base_seed: u64,
) -> MatchResult {
    let mut result = MatchResult::default();
    for g in 0..games {
        let model_is_p1 = g % 2 == 0;
        let seed_model = base_seed.wrapping_add(g as u64 * 2);
        let seed_baseline = base_seed.wrapping_add(g as u64 * 2 + 1);

        let mut model = make_model(seed_model);
        let mut baseline = make_baseline(seed_baseline);

        let outcome = if model_is_p1 {
            play_game(rule.clone(), model.as_mut(), baseline.as_mut())
        } else {
            play_game(rule.clone(), baseline.as_mut(), model.as_mut())
        };

        match outcome {
            GameResult::Win(winner) => {
                let model_player = if model_is_p1 { Player::P1 } else { Player::P2 };
                if winner == model_player {
                    result.wins += 1;
                } else {
                    result.losses += 1;
                }
            }
            GameResult::Draw(_) => result.draws += 1,
        }
    }
    result
}

fn main() -> ExitCode {
    let args = Args::parse();

    let model_config = AzMctsConfig {
        simulations: args.simulations,
        dirichlet_epsilon: 0.0,
        ..AzMctsConfig::default()
    };

    // Verify the model loads before starting any games.
    if let Err(e) = AzMctsPolicy::from_path(&args.model, model_config, 0) {
        eprintln!("failed to load model: {e}");
        return ExitCode::FAILURE;
    }

    let rule = RuleConfig::default();
    let model_path = args.model.clone();
    let games = args.games;
    let base_seed = args.seed;

    // -- Random baseline --
    {
        let mut make_model = |seed: u64| -> Box<dyn Policy> {
            Box::new(
                AzMctsPolicy::from_path(&model_path, model_config, seed)
                    .expect("model loads"),
            )
        };
        let mut make_baseline = |seed: u64| -> Box<dyn Policy> {
            Box::new(RandomPolicy::seeded(seed))
        };
        let r = run_match(
            games,
            rule.clone(),
            &mut make_model,
            &mut make_baseline,
            base_seed,
        );
        println!(
            "{}",
            serde_json::json!({
                "baseline": "random",
                "wins": r.wins,
                "draws": r.draws,
                "losses": r.losses,
                "score": r.score()
            })
        );
    }

    // -- Greedy baseline --
    {
        let mut make_model = |seed: u64| -> Box<dyn Policy> {
            Box::new(
                AzMctsPolicy::from_path(&model_path, model_config, seed)
                    .expect("model loads"),
            )
        };
        let mut make_baseline = |seed: u64| -> Box<dyn Policy> {
            Box::new(GreedyPolicy::seeded(seed))
        };
        let r = run_match(
            games,
            rule.clone(),
            &mut make_model,
            &mut make_baseline,
            base_seed.wrapping_add(100_000),
        );
        println!(
            "{}",
            serde_json::json!({
                "baseline": "greedy",
                "wins": r.wins,
                "draws": r.draws,
                "losses": r.losses,
                "score": r.score()
            })
        );
    }

    // -- MCTS baseline --
    {
        let mut make_model = |seed: u64| -> Box<dyn Policy> {
            Box::new(
                AzMctsPolicy::from_path(&model_path, model_config, seed)
                    .expect("model loads"),
            )
        };
        let mut make_baseline = |seed: u64| -> Box<dyn Policy> {
            Box::new(MctsPolicy::new(DEFAULT_MCTS_BASELINE_ITERS, seed))
        };
        let r = run_match(
            games,
            rule.clone(),
            &mut make_model,
            &mut make_baseline,
            base_seed.wrapping_add(200_000),
        );
        println!(
            "{}",
            serde_json::json!({
                "baseline": "mcts",
                "wins": r.wins,
                "draws": r.draws,
                "losses": r.losses,
                "score": r.score()
            })
        );
    }

    ExitCode::SUCCESS
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    /// Path to the fixture model used by other crate tests.
    fn fixture_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../kairnz-onnx/tests/fixtures/random_init.onnx")
    }

    /// Fast rule: very short games so the test runs in a couple of seconds.
    fn fast_rule() -> RuleConfig {
        RuleConfig { max_plies: 30, ..RuleConfig::default() }
    }

    /// Fast model config: minimal simulations so each move is near-instant.
    fn fast_model_config() -> AzMctsConfig {
        AzMctsConfig {
            simulations: 8,
            dirichlet_epsilon: 0.0,
            ..AzMctsConfig::default()
        }
    }

    #[test]
    fn strength_harness_runs_and_score_in_range_random() {
        let path = fixture_path();
        let config = fast_model_config();
        let rule = fast_rule();
        let games = 2u32;

        let mut make_model = |seed: u64| -> Box<dyn Policy> {
            Box::new(AzMctsPolicy::from_path(&path, config, seed).expect("fixture loads"))
        };
        let mut make_baseline = |seed: u64| -> Box<dyn Policy> {
            Box::new(RandomPolicy::seeded(seed))
        };

        let result = run_match(games, rule, &mut make_model, &mut make_baseline, 42);

        assert_eq!(
            result.wins + result.draws + result.losses,
            games,
            "tally must sum to games played"
        );
        let score = result.score();
        assert!(
            (0.0..=1.0).contains(&score),
            "score must be in [0, 1], got {score}"
        );
    }

    #[test]
    fn strength_harness_runs_and_score_in_range_greedy() {
        let path = fixture_path();
        let config = fast_model_config();
        let rule = fast_rule();
        let games = 2u32;

        let mut make_model = |seed: u64| -> Box<dyn Policy> {
            Box::new(AzMctsPolicy::from_path(&path, config, seed).expect("fixture loads"))
        };
        let mut make_baseline = |seed: u64| -> Box<dyn Policy> {
            Box::new(GreedyPolicy::seeded(seed))
        };

        let result = run_match(games, rule, &mut make_model, &mut make_baseline, 43);

        assert_eq!(result.wins + result.draws + result.losses, games);
        assert!((0.0..=1.0).contains(&result.score()));
    }

    #[test]
    fn strength_harness_runs_and_score_in_range_mcts() {
        let path = fixture_path();
        let config = fast_model_config();
        let rule = fast_rule();
        let games = 2u32;

        let mut make_model = |seed: u64| -> Box<dyn Policy> {
            Box::new(AzMctsPolicy::from_path(&path, config, seed).expect("fixture loads"))
        };
        let mut make_baseline = |seed: u64| -> Box<dyn Policy> {
            Box::new(MctsPolicy::new(4, seed))
        };

        let result = run_match(games, rule, &mut make_model, &mut make_baseline, 44);

        assert_eq!(result.wins + result.draws + result.losses, games);
        assert!((0.0..=1.0).contains(&result.score()));
    }

    #[test]
    fn match_result_score_is_correct() {
        let r = MatchResult { wins: 3, draws: 2, losses: 1 };
        let expected = (3.0 + 0.5 * 2.0) / 6.0;
        assert!((r.score() - expected).abs() < 1e-9);
    }

    #[test]
    fn match_result_score_zero_when_no_games() {
        let r = MatchResult::default();
        assert_eq!(r.score(), 0.0);
    }
}
