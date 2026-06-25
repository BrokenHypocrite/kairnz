//! Self-play CLI: plays games with the neural MCTS and writes a training shard.

use std::path::PathBuf;
use std::process::ExitCode;

use clap::Parser;
use kairnz_core::config::RuleConfig;
use kairnz_onnx::{OnnxEvaluator, DEFAULT_MAX_BATCH};
use kairnz_selfplay::parallel::parallel_self_play;
use kairnz_selfplay::shard::write_shard;
use kairnz_selfplay::SelfPlayConfig;

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
    /// Worker threads (0 = auto-detect available parallelism).
    #[arg(long, default_value_t = 0)]
    threads: usize,
    /// Use a single shared inference server (batched GPU path).
    /// By default (without this flag), each thread runs its own ONNX session,
    /// keeping all CPU cores busy -- optimal when the net is small and CPU is
    /// the bottleneck.  Pass --batched when the GPU is the bottleneck and you
    /// want to maximise batch utilisation.
    #[arg(long, default_value_t = false)]
    batched: bool,
    /// Maximum positions per GPU batch (only used with --batched).
    #[arg(long, default_value_t = DEFAULT_MAX_BATCH)]
    max_batch: usize,
    /// Leaves collected and evaluated per batched MCTS step (only used with --batched).
    #[arg(long, default_value_t = 8)]
    leaves_per_step: usize,
}

fn main() -> ExitCode {
    let args = Args::parse();
    let config = SelfPlayConfig {
        simulations: args.simulations,
        games: args.games,
        ..SelfPlayConfig::default()
    };

    let threads = if args.threads == 0 {
        std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
    } else {
        args.threads
    };

    // Load once just to report the execution backend.
    match OnnxEvaluator::from_path(&args.model) {
        Ok(evaluator) => {
            println!(
                "self-play backend: {:?}, threads: {threads}, mode: {}",
                evaluator.backend(),
                if args.batched { "batched" } else { "per-thread" }
            );
        }
        Err(error) => {
            eprintln!("failed to load model: {error}");
            return ExitCode::FAILURE;
        }
    }

    let samples = match parallel_self_play(
        &args.model,
        args.games,
        threads,
        config.mcts_config(),
        RuleConfig::default(),
        config.temperature_cutoff,
        args.seed,
        args.batched,
        args.max_batch,
        args.leaves_per_step,
    ) {
        Ok(s) => s,
        Err(error) => {
            eprintln!("self-play failed: {error}");
            return ExitCode::FAILURE;
        }
    };
    println!("played {} games on {threads} threads -> {} samples", args.games, samples.len());

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
