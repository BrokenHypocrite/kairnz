//! Gate CLI: play model A against model B and print the tally as JSON.
//!
//! In coordinator mode (default), `--threads N` spawns N worker processes,
//! each running this same binary with `--worker`, then sums their tallies.
//! In worker mode (`--worker`), one single-threaded gate run is performed
//! and the W/D/L JSON is written to stdout.

use std::path::PathBuf;
use std::process::ExitCode;

use clap::Parser;
use kairnz_core::config::RuleConfig;
use kairnz_onnx::AzMctsConfig;
use kairnz_selfplay::gate::{run_gate, GateResult};

/// Default Dirichlet root-noise weight for gate variety.
const GATE_DIRICHLET_EPSILON: f64 = 0.15;

/// Maximum worker count when auto-detecting available parallelism.
const MAX_AUTO_WORKERS: usize = 64;

/// Command-line arguments for a gate run.
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
    /// Number of worker processes to spawn (coordinator) or ignored in worker
    /// mode. 0 = auto-detect available parallelism.
    #[arg(long, default_value_t = 0)]
    threads: usize,
    /// Run as a single-threaded worker process; prints its W/D/L JSON to
    /// stdout. This flag is set by the coordinator -- callers should not set
    /// it manually.
    #[arg(long, default_value_t = false)]
    worker: bool,
    /// Global game index of the first game this worker should play. Ensures
    /// per-game seeds and color assignments match the single-process formula.
    /// Only meaningful when `--worker` is set.
    #[arg(long, default_value_t = 0)]
    game_offset: u32,
}

fn main() -> ExitCode {
    let args = Args::parse();

    if args.worker {
        run_worker(args)
    } else {
        run_coordinator(args)
    }
}

/// Worker path: plays `args.games` games starting at `args.game_offset` and
/// prints the tally as JSON. Only the JSON line goes to stdout; diagnostics
/// go to stderr.
fn run_worker(args: Args) -> ExitCode {
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
        args.game_offset,
    ) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("gate worker failed: {e}");
            return ExitCode::FAILURE;
        }
    };

    // Print only the JSON line; the coordinator parses this from stdout.
    println!(
        "{{\"a_wins\":{},\"b_wins\":{},\"draws\":{}}}",
        result.a_wins, result.b_wins, result.draws
    );
    ExitCode::SUCCESS
}

/// Coordinator path: spawn N worker processes, wait for all concurrently, then
/// aggregate their tallies and print the final JSON.
fn run_coordinator(args: Args) -> ExitCode {
    let n = if args.threads == 0 {
        std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(1)
            .min(MAX_AUTO_WORKERS)
    } else {
        args.threads
    }
    .max(1);

    eprintln!("gate coordinator: {n} workers, {} games total", args.games);

    let exe = match std::env::current_exe() {
        Ok(p) => p,
        Err(e) => {
            eprintln!("failed to resolve current executable: {e}");
            return ExitCode::FAILURE;
        }
    };

    let model_a_str = args.model_a.to_string_lossy().into_owned();
    let model_b_str = args.model_b.to_string_lossy().into_owned();

    let base_games = args.games as usize / n;
    let remainder = args.games as usize % n;

    // Spawn all workers before waiting for any (true concurrency).
    let mut children: Vec<(std::process::Child, usize)> = Vec::with_capacity(n);
    let mut offset: u32 = 0;

    for k in 0..n {
        let games_k = (base_games + if k < remainder { 1 } else { 0 }) as u32;
        if games_k == 0 {
            continue;
        }

        let child = std::process::Command::new(&exe)
            .arg("--worker")
            .args(["--model-a", &model_a_str])
            .args(["--model-b", &model_b_str])
            .args(["--games", &games_k.to_string()])
            .args(["--simulations", &args.simulations.to_string()])
            .args(["--seed", &args.seed.to_string()])
            .args(["--game-offset", &offset.to_string()])
            .stdout(std::process::Stdio::piped())
            .spawn();

        match child {
            Ok(c) => children.push((c, k)),
            Err(e) => {
                eprintln!("failed to spawn gate worker {k}: {e}");
                return ExitCode::FAILURE;
            }
        }

        offset += games_k;
    }

    // Wait for all workers and aggregate their tallies.
    let mut total = GateResult { a_wins: 0, b_wins: 0, draws: 0 };
    let mut any_failed = false;

    for (child, k) in children {
        match child.wait_with_output() {
            Err(e) => {
                eprintln!("gate worker {k} wait error: {e}");
                any_failed = true;
            }
            Ok(out) if !out.status.success() => {
                eprintln!(
                    "gate worker {k} exited with {}:\n{}",
                    out.status,
                    String::from_utf8_lossy(&out.stderr)
                );
                any_failed = true;
            }
            Ok(out) => {
                let stdout = String::from_utf8_lossy(&out.stdout);
                // Worker prints exactly one JSON line to stdout.
                let json_line = stdout.lines().find(|l| l.trim_start().starts_with('{'));
                match json_line {
                    None => {
                        eprintln!("gate worker {k} produced no JSON output");
                        any_failed = true;
                    }
                    Some(line) => match parse_worker_result(line) {
                        Ok(r) => {
                            total.a_wins += r.a_wins;
                            total.b_wins += r.b_wins;
                            total.draws += r.draws;
                        }
                        Err(e) => {
                            eprintln!(
                                "gate worker {k} JSON parse error: {e} (line: {line})"
                            );
                            any_failed = true;
                        }
                    },
                }
            }
        }
    }

    if any_failed {
        return ExitCode::FAILURE;
    }

    // Print the final result in the same format as the single-process gate so
    // loop.py's parsing is unchanged.
    println!(
        "{{\"a_wins\":{},\"b_wins\":{},\"draws\":{},\"a_score\":{:.4}}}",
        total.a_wins,
        total.b_wins,
        total.draws,
        total.a_score()
    );
    ExitCode::SUCCESS
}

/// Parses a worker's W/D/L JSON line into a [`GateResult`].
///
/// Extracts `a_wins`, `b_wins`, and `draws` by key-scanning the JSON string.
/// This avoids a serde_json dependency for three integer fields.
fn parse_worker_result(line: &str) -> Result<GateResult, String> {
    let extract = |key: &str| -> Result<u32, String> {
        let needle = format!("\"{}\":", key);
        let pos = line
            .find(&needle)
            .ok_or_else(|| format!("key '{key}' not found in JSON"))?;
        let after = &line[pos + needle.len()..];
        let digits: String = after
            .trim_start()
            .chars()
            .take_while(|c| c.is_ascii_digit())
            .collect();
        digits
            .parse::<u32>()
            .map_err(|e| format!("parse error for '{key}': {e}"))
    };

    Ok(GateResult {
        a_wins: extract("a_wins")?,
        b_wins: extract("b_wins")?,
        draws: extract("draws")?,
    })
}
