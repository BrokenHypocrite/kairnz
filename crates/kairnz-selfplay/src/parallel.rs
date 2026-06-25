//! Parallel self-play: play many games concurrently, with two execution modes.
//!
//! * `batched == false` (default): each thread loads its own ONNX session and
//!   runs a sequential [`AzMcts`].  All cores are kept busy; optimal when the
//!   CPU MCTS is the bottleneck (small nets, many cores).
//! * `batched == true`: a single shared [`InferenceServer`] services all
//!   threads via [`BatchedAzMcts`].  Use when the GPU is the bottleneck and
//!   you want to maximise batch utilisation.

use std::path::Path;

use kairnz_core::config::RuleConfig;
use kairnz_onnx::batched_mcts::BatchedAzMcts;
use kairnz_onnx::mcts::AzMcts;
use kairnz_onnx::{AzMctsConfig, InferenceServer, OnnxEvaluator};
use rand::SeedableRng;
use rand_pcg::Pcg64;

use crate::play::play_game;
use crate::sample::Sample;

/// Plays `total_games` self-play games across `threads` worker threads and
/// returns every sample.
///
/// Games are partitioned contiguously across threads (the first `remainder`
/// threads play one extra game). Samples are returned in thread order, so a run
/// is reproducible for a fixed `(model, total_games, threads, base_seed)`.
///
/// When `batched` is `false` (the default), each thread loads its own ONNX
/// session and runs a sequential [`AzMcts`]; all CPU cores stay busy.  When
/// `batched` is `true`, a single shared [`InferenceServer`] is used with
/// [`BatchedAzMcts`] on each thread; `max_batch` and `leaves_per_step` control
/// the batching behaviour.
///
/// Returns the first inference error encountered by any thread.
pub fn parallel_self_play(
    model_path: &Path,
    total_games: u32,
    threads: usize,
    mcts_config: AzMctsConfig,
    rule: RuleConfig,
    temperature_cutoff: u32,
    base_seed: u64,
    batched: bool,
    max_batch: usize,
    leaves_per_step: usize,
) -> ort::Result<Vec<Sample>> {
    let threads = threads.max(1);
    let total = total_games as usize;
    let base = total / threads;
    let remainder = total % threads;

    if batched {
        // --- Batched path: one shared InferenceServer, per-thread BatchedAzMcts ---
        let evaluator = OnnxEvaluator::from_path(model_path)?;
        let server = InferenceServer::new(evaluator, max_batch);

        struct ThreadParams {
            t: usize,
            games_for_t: usize,
            thread_config: AzMctsConfig,
            thread_rule: RuleConfig,
        }
        let params: Vec<ThreadParams> = (0..threads)
            .map(|t| ThreadParams {
                t,
                games_for_t: base + if t < remainder { 1 } else { 0 },
                thread_config: AzMctsConfig { leaves_per_step, ..mcts_config },
                thread_rule: rule.clone(),
            })
            .collect();

        let chunk_results: Vec<ort::Result<Vec<Sample>>> = std::thread::scope(|scope| {
            let handles: Vec<_> = params
                .iter()
                .map(|p| {
                    scope.spawn(|| -> ort::Result<Vec<Sample>> {
                        let mut mcts =
                            BatchedAzMcts::new(&server, p.thread_config, base_seed + p.t as u64);
                        let mut rng = Pcg64::seed_from_u64(base_seed ^ ((p.t as u64) << 32));
                        let mut samples = Vec::new();
                        for _ in 0..p.games_for_t {
                            samples.extend(play_game(
                                &mut mcts,
                                p.thread_rule.clone(),
                                temperature_cutoff,
                                &mut rng,
                            )?);
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
    } else {
        // --- Per-thread path: each thread loads its own OnnxEvaluator + AzMcts ---
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
                            )?);
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use kairnz_onnx::DEFAULT_MAX_BATCH;
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
        let a = parallel_self_play(
            &path, 2, 2, small_config(), fast_rule(), 4, 7, false, DEFAULT_MAX_BATCH, 4,
        )
        .expect("parallel self-play runs");
        assert!(!a.is_empty(), "self-play produces samples");

        let b = parallel_self_play(
            &path, 2, 2, small_config(), fast_rule(), 4, 7, false, DEFAULT_MAX_BATCH, 4,
        )
        .expect("parallel self-play runs");
        assert_eq!(a, b, "same (games, threads, seed) is reproducible");
    }

    #[test]
    fn more_threads_than_games_is_handled() {
        let path = fixture();
        // 2 games across 4 threads: two threads play one game, two play none.
        let samples = parallel_self_play(
            &path, 2, 4, small_config(), fast_rule(), 4, 1, false, DEFAULT_MAX_BATCH, 4,
        )
        .expect("runs with idle threads");
        assert!(!samples.is_empty(), "the two played games still produce samples");
    }

    #[test]
    fn batched_mode_produces_samples() {
        let path = fixture();
        let samples = parallel_self_play(
            &path, 2, 2, small_config(), fast_rule(), 4, 7, true, DEFAULT_MAX_BATCH, 4,
        )
        .expect("batched self-play runs");
        assert!(!samples.is_empty(), "batched self-play produces samples");
    }
}
