//! Parallel self-play: play many games concurrently, one ONNX session per thread.

use std::path::Path;

use kairnz_core::config::RuleConfig;
use kairnz_onnx::mcts::AzMcts;
use kairnz_onnx::{AzMctsConfig, OnnxEvaluator};
use rand::SeedableRng;
use rand_pcg::Pcg64;

use crate::play::play_game;
use crate::sample::Sample;

/// Plays `total_games` self-play games across `threads` worker threads, each with
/// its own ONNX session (reused across its chunk of games) sharing the GPU, and
/// returns every sample.
///
/// Games are partitioned contiguously across threads (the first `remainder`
/// threads play one extra game). Samples are returned in thread order, so a run
/// is reproducible for a fixed (model, total_games, threads, base_seed). Returns
/// the first session-load error if any thread fails to load the model.
pub fn parallel_self_play(
    model_path: &Path,
    total_games: u32,
    threads: usize,
    mcts_config: AzMctsConfig,
    rule: RuleConfig,
    temperature_cutoff: u32,
    base_seed: u64,
) -> ort::Result<Vec<Sample>> {
    let threads = threads.max(1);
    let total = total_games as usize;
    let base = total / threads;
    let remainder = total % threads;

    let chunk_results: Vec<ort::Result<Vec<Sample>>> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..threads)
            .map(|t| {
                let games_for_t = base + if t < remainder { 1 } else { 0 };
                let thread_rule = rule.clone();
                scope.spawn(move || -> ort::Result<Vec<Sample>> {
                    let evaluator = OnnxEvaluator::from_path(model_path)?;
                    let mut mcts = AzMcts::new(evaluator, mcts_config, base_seed + t as u64);
                    let mut rng = Pcg64::seed_from_u64(base_seed ^ ((t as u64) << 32));
                    let mut samples = Vec::new();
                    for _ in 0..games_for_t {
                        samples.extend(play_game(
                            &mut mcts,
                            thread_rule.clone(),
                            temperature_cutoff,
                            &mut rng,
                        ));
                    }
                    Ok(samples)
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("self-play thread panicked"))
            .collect()
    });

    let mut all = Vec::new();
    for result in chunk_results {
        all.extend(result?);
    }
    Ok(all)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn fixture() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../kairnz-onnx/tests/fixtures/random_init.onnx")
    }

    fn small_config() -> AzMctsConfig {
        AzMctsConfig { simulations: 8, ..AzMctsConfig::default() }
    }

    fn fast_rule() -> RuleConfig {
        // Short games keep these CPU tests fast (each thread loads a session).
        RuleConfig { max_plies: 30, ..RuleConfig::default() }
    }

    #[test]
    fn parallel_self_play_produces_samples_and_is_reproducible() {
        let path = fixture();
        let a = parallel_self_play(&path, 2, 2, small_config(), fast_rule(), 4, 7)
            .expect("parallel self-play runs");
        assert!(!a.is_empty(), "self-play produces samples");

        let b = parallel_self_play(&path, 2, 2, small_config(), fast_rule(), 4, 7)
            .expect("parallel self-play runs");
        assert_eq!(a, b, "same (games, threads, seed) is reproducible");
    }

    #[test]
    fn more_threads_than_games_is_handled() {
        let path = fixture();
        // 2 games across 4 threads: two threads play one game, two play none.
        let samples = parallel_self_play(&path, 2, 4, small_config(), fast_rule(), 4, 1)
            .expect("runs with idle threads");
        assert!(!samples.is_empty(), "the two played games still produce samples");
    }
}
