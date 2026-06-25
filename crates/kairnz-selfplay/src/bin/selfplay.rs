//! Self-play CLI: plays games with the neural MCTS and writes a training shard.
//!
//! In coordinator mode (default), `--threads N` spawns N worker processes, each
//! running this same binary with `--worker`, then merges their fragment shards.
//! In worker mode (`--worker`), one single-threaded self-play run is performed
//! and the shard is written directly to `--out`.

use std::path::PathBuf;
use std::process::ExitCode;

use clap::Parser;
use kairnz_core::config::RuleConfig;
use kairnz_onnx::DEFAULT_MAX_BATCH;
use kairnz_selfplay::parallel::parallel_self_play;
use kairnz_selfplay::shard::{read_shard, write_shard};
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
    /// Number of worker processes to spawn (coordinator) or ignored in worker mode.
    /// 0 = auto-detect available parallelism.
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
    /// Temperature cutoff: plies of proportional sampling before argmax.
    #[arg(long, default_value_t = 20)]
    temperature_cutoff: u32,
    /// Run as a single-threaded worker process; writes its fragment shard to --out.
    /// This flag is set by the coordinator -- callers should not set it manually.
    #[arg(long, default_value_t = false)]
    worker: bool,
}

fn main() -> ExitCode {
    let args = Args::parse();

    if args.worker {
        run_worker(args)
    } else if args.batched {
        run_batched(args)
    } else {
        run_coordinator(args)
    }
}

/// Batched path: a single in-process [`InferenceServer`] coalesces leaf
/// evaluations from `threads` search threads into large GPU batches, instead of
/// each worker firing one-position calls. Use when the GPU is the bottleneck
/// (a capable GPU with a non-trivial net): far fewer kernel launches and real
/// batch utilisation. Writes the shard directly; no worker subprocesses or
/// fragment files.
fn run_batched(args: Args) -> ExitCode {
    let n = if args.threads == 0 {
        std::thread::available_parallelism().map(|p| p.get()).unwrap_or(1)
    } else {
        args.threads
    }
    .max(1);

    let config = SelfPlayConfig {
        simulations: args.simulations,
        temperature_cutoff: args.temperature_cutoff,
        ..SelfPlayConfig::default()
    };

    println!(
        "self-play (batched): {n} threads, {} games, max_batch {}, leaves/step {}",
        args.games, args.max_batch, args.leaves_per_step
    );

    let samples = match parallel_self_play(
        &args.model,
        args.games,
        n,
        config.mcts_config(),
        RuleConfig::default(),
        config.temperature_cutoff,
        args.seed,
        true,
        args.max_batch,
        args.leaves_per_step,
    ) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("batched self-play failed: {e}");
            return ExitCode::FAILURE;
        }
    };

    match write_shard(&samples, &args.out) {
        Ok(()) => {
            println!(
                "self-play: {n} threads, {} samples -> {}",
                samples.len(),
                args.out.display()
            );
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("batched self-play failed to write shard: {e}");
            ExitCode::FAILURE
        }
    }
}

/// Worker path: single-threaded self-play writing one fragment shard to `--out`.
fn run_worker(args: Args) -> ExitCode {
    let config = SelfPlayConfig {
        simulations: args.simulations,
        temperature_cutoff: args.temperature_cutoff,
        ..SelfPlayConfig::default()
    };

    let samples = match parallel_self_play(
        &args.model,
        args.games,
        1, // workers are always single-threaded
        config.mcts_config(),
        RuleConfig::default(),
        config.temperature_cutoff,
        args.seed,
        false, // batched mode is not used inside workers
        args.max_batch,
        args.leaves_per_step,
    ) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("worker self-play failed: {e}");
            return ExitCode::FAILURE;
        }
    };

    match write_shard(&samples, &args.out) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("worker failed to write shard: {e}");
            ExitCode::FAILURE
        }
    }
}

/// Coordinator path: spawn N worker processes, wait for all concurrently, then merge.
fn run_coordinator(args: Args) -> ExitCode {
    let n = if args.threads == 0 {
        std::thread::available_parallelism().map(|p| p.get()).unwrap_or(1)
    } else {
        args.threads
    }
    .max(1);

    println!("self-play coordinator: {n} workers, {} games total", args.games);

    let exe = match std::env::current_exe() {
        Ok(p) => p,
        Err(e) => {
            eprintln!("failed to resolve current executable: {e}");
            return ExitCode::FAILURE;
        }
    };

    let model_str = args.model.to_string_lossy().into_owned();
    let base_games = args.games as usize / n;
    let remainder = args.games as usize % n;

    // Spawn all workers before waiting for any (true concurrency).
    let mut children: Vec<(std::process::Child, PathBuf)> = Vec::with_capacity(n);
    for k in 0..n {
        let games_k = base_games + if k < remainder { 1 } else { 0 };
        if games_k == 0 {
            continue;
        }

        let frag = args.out.with_extension(format!("part{k}.safetensors"));
        let frag_str = frag.to_string_lossy().into_owned();

        let child = std::process::Command::new(&exe)
            .arg("--worker")
            .args(["--model", &model_str])
            .args(["--out", &frag_str])
            .args(["--games", &games_k.to_string()])
            .args(["--simulations", &args.simulations.to_string()])
            .args(["--seed", &(args.seed + k as u64).to_string()])
            .args(["--threads", "1"])
            .args(["--max-batch", &args.max_batch.to_string()])
            .args(["--leaves-per-step", &args.leaves_per_step.to_string()])
            .args(["--temperature-cutoff", &args.temperature_cutoff.to_string()])
            .spawn();

        match child {
            Ok(c) => children.push((c, frag)),
            Err(e) => {
                eprintln!("failed to spawn worker {k}: {e}");
                // Clean up any fragments already queued, then quit.
                cleanup_frags(&children.iter().map(|(_, p)| p.clone()).collect::<Vec<_>>());
                return ExitCode::FAILURE;
            }
        }
    }

    // Wait for all workers; collect any failures.
    let mut failed = Vec::new();
    let frags: Vec<PathBuf> = children.iter().map(|(_, p)| p.clone()).collect();
    for (k, (mut child, frag)) in children.into_iter().enumerate() {
        match child.wait() {
            Ok(status) if status.success() => {}
            Ok(status) => {
                failed.push(format!("worker {k} ({}) exited with {status}", frag.display()));
            }
            Err(e) => {
                failed.push(format!("worker {k} wait error: {e}"));
            }
        }
    }

    if !failed.is_empty() {
        for msg in &failed {
            eprintln!("{msg}");
        }
        cleanup_frags(&frags);
        return ExitCode::FAILURE;
    }

    // Merge fragment shards into the final output file.
    let mut all_samples = Vec::new();
    for frag in &frags {
        match read_shard(frag) {
            Ok(samples) => all_samples.extend(samples),
            Err(e) => {
                eprintln!("failed to read fragment {}: {e}", frag.display());
                cleanup_frags(&frags);
                return ExitCode::FAILURE;
            }
        }
    }

    match write_shard(&all_samples, &args.out) {
        Ok(()) => {}
        Err(e) => {
            eprintln!("failed to write merged shard: {e}");
            cleanup_frags(&frags);
            return ExitCode::FAILURE;
        }
    }

    // Best-effort cleanup of fragment files.
    cleanup_frags(&frags);

    println!(
        "self-play: {n} workers, {} samples -> {}",
        all_samples.len(),
        args.out.display()
    );
    ExitCode::SUCCESS
}

/// Removes fragment shard files on a best-effort basis (ignores errors).
fn cleanup_frags(frags: &[PathBuf]) {
    for frag in frags {
        let _ = std::fs::remove_file(frag);
    }
}
