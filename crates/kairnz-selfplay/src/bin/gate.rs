//! Gate CLI: play model A against model B and print the tally as JSON.

use std::path::PathBuf;
use std::process::ExitCode;

use clap::Parser;
use kairnz_core::config::RuleConfig;
use kairnz_onnx::AzMctsConfig;
use kairnz_selfplay::gate::run_gate;

/// Default Dirichlet root-noise weight for gate variety.
const GATE_DIRICHLET_EPSILON: f64 = 0.15;

#[derive(Parser)]
#[command(about = "Gate a candidate Kairnz model against a best model.")]
struct Args {
    /// Candidate model (scored as A).
    #[arg(long)]
    model_a: PathBuf,
    /// Reference model (B).
    #[arg(long)]
    model_b: PathBuf,
    /// Number of games to play.
    #[arg(long, default_value_t = 40)]
    games: u32,
    /// MCTS simulations per move.
    #[arg(long, default_value_t = 100)]
    simulations: u32,
    /// Base RNG seed.
    #[arg(long, default_value_t = 0)]
    seed: u64,
}

fn main() -> ExitCode {
    let args = Args::parse();
    let config = AzMctsConfig {
        simulations: args.simulations,
        dirichlet_epsilon: GATE_DIRICHLET_EPSILON,
        ..AzMctsConfig::default()
    };

    let result = match run_gate(
        &args.model_a,
        &args.model_b,
        args.games,
        config,
        RuleConfig::default(),
        args.seed,
    ) {
        Ok(r) => r,
        Err(error) => {
            eprintln!("gate failed: {error}");
            return ExitCode::FAILURE;
        }
    };

    println!(
        "{{\"a_wins\":{},\"b_wins\":{},\"draws\":{},\"a_score\":{:.4}}}",
        result.a_wins, result.b_wins, result.draws, result.a_score()
    );
    ExitCode::SUCCESS
}
