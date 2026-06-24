//! Self-play CLI: plays games with the neural MCTS and writes a training shard.

use std::path::PathBuf;
use std::process::ExitCode;

use clap::Parser;
use kairnz_core::config::RuleConfig;
use kairnz_onnx::mcts::AzMcts;
use kairnz_onnx::OnnxEvaluator;
use kairnz_selfplay::play::play_game;
use kairnz_selfplay::shard::write_shard;
use kairnz_selfplay::SelfPlayConfig;
use rand::SeedableRng;
use rand_pcg::Pcg64;

/// Command-line arguments for a self-play run.
#[derive(Parser)]
#[command(about = "Generate Kairnz self-play training shards.")]
struct Args {
    /// Path to the ONNX model to play with.
    #[arg(long)]
    model: PathBuf,
    /// Output shard path (.safetensors).
    #[arg(long)]
    out: PathBuf,
    /// Number of games to play.
    #[arg(long, default_value_t = 8)]
    games: u32,
    /// MCTS simulations per move.
    #[arg(long, default_value_t = 200)]
    simulations: u32,
    /// Base RNG seed.
    #[arg(long, default_value_t = 0)]
    seed: u64,
}

fn main() -> ExitCode {
    let args = Args::parse();
    let config = SelfPlayConfig {
        simulations: args.simulations,
        games: args.games,
        ..SelfPlayConfig::default()
    };

    let evaluator = match OnnxEvaluator::from_path(&args.model) {
        Ok(e) => e,
        Err(error) => {
            eprintln!("failed to load model: {error}");
            return ExitCode::FAILURE;
        }
    };
    println!("self-play backend: {:?}", evaluator.backend());

    let mut mcts = AzMcts::new(evaluator, config.mcts_config(), args.seed);
    let mut rng = Pcg64::seed_from_u64(args.seed);

    let mut samples = Vec::new();
    for g in 0..config.games {
        let game_samples = play_game(&mut mcts, RuleConfig::default(), config.temperature_cutoff, &mut rng);
        println!("game {g}: {} samples", game_samples.len());
        samples.extend(game_samples);
    }

    match write_shard(&samples, &args.out) {
        Ok(()) => {
            println!("wrote {} samples to {}", samples.len(), args.out.display());
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("failed to write shard: {error}");
            ExitCode::FAILURE
        }
    }
}
