//! Concurrency + correctness tests for `InferenceServer`.
//!
//! 8 threads each submit a batch of positions; every result must match a
//! direct `evaluate` reference within 1e-4. The server must shut down cleanly
//! on drop.

use std::path::PathBuf;
use std::sync::Arc;
use std::thread;

use kairnz_core::config::RuleConfig;
use kairnz_core::position::Position;
use kairnz_encode::{encode_planes, POLICY_SIZE};
use kairnz_onnx::{BatchEvaluator, InferenceServer, OnnxEvaluator, DEFAULT_MAX_BATCH};

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/random_init.onnx")
}

/// Sampled policy indices spread across the full range.
const SAMPLED_POLICY_INDICES: &[usize] = &[0, 100, 500, 1000, 3000, 6000, 6722];

/// Number of worker threads.
const NUM_THREADS: usize = 8;

/// Positions per thread (two per thread gives 16 total requests).
const POSITIONS_PER_THREAD: usize = 2;

#[test]
fn server_concurrent_results_match_direct_evaluate_within_tolerance() {
    // Build reference results via direct evaluate (single-threaded, before the server).
    let pos = Position::new_standard(RuleConfig::default());
    let rep_counts: Vec<u8> = (0..POSITIONS_PER_THREAD as u8).collect();

    let mut ref_evaluator =
        OnnxEvaluator::from_path(&fixture_path()).expect("fixture loads for reference");
    let mut reference: Vec<(Vec<f32>, f32)> = Vec::new();
    for &rep in &rep_counts {
        let result = ref_evaluator
            .evaluate(&pos, rep)
            .expect("reference evaluate succeeds");
        reference.push(result);
    }
    let reference = Arc::new(reference);

    // Build the server.
    let server_evaluator =
        OnnxEvaluator::from_path(&fixture_path()).expect("fixture loads for server");
    let server = Arc::new(InferenceServer::new(server_evaluator, DEFAULT_MAX_BATCH));

    // Spawn 8 threads; each submits POSITIONS_PER_THREAD positions and checks results.
    let planes: Vec<Vec<f32>> = rep_counts.iter().map(|&r| encode_planes(&pos, r)).collect();
    let planes = Arc::new(planes);

    let handles: Vec<_> = (0..NUM_THREADS)
        .map(|thread_id| {
            let server = Arc::clone(&server);
            let planes = Arc::clone(&planes);
            let rep_counts = rep_counts.clone();
            let reference = Arc::clone(&reference);
            thread::spawn(move || {
                let results = server
                    .evaluate_batch(&planes, &rep_counts)
                    .expect("server evaluate_batch succeeds");

                assert_eq!(
                    results.len(),
                    POSITIONS_PER_THREAD,
                    "thread {thread_id}: wrong result count"
                );

                for (i, ((policy, value), (ref_policy, ref_value))) in
                    results.iter().zip(reference.iter()).enumerate()
                {
                    assert_eq!(
                        policy.len(),
                        POLICY_SIZE,
                        "thread {thread_id} row {i}: policy length"
                    );

                    let value_diff = (value - ref_value).abs();
                    assert!(
                        value_diff < 1e-4,
                        "thread {thread_id} row {i}: value diff {value_diff} exceeds 1e-4 \
                         (got={value}, ref={ref_value})"
                    );

                    for &idx in SAMPLED_POLICY_INDICES {
                        let diff = (policy[idx] - ref_policy[idx]).abs();
                        assert!(
                            diff < 1e-4,
                            "thread {thread_id} row {i}: policy[{idx}] diff {diff} exceeds 1e-4 \
                             (got={}, ref={})",
                            policy[idx],
                            ref_policy[idx]
                        );
                    }
                }
            })
        })
        .collect();

    for h in handles {
        h.join().expect("worker thread did not panic");
    }

    // Drop the Arc to trigger InferenceServer::drop; the batcher thread must join cleanly.
    drop(server);
}

#[test]
fn server_shuts_down_cleanly_on_drop() {
    let evaluator = OnnxEvaluator::from_path(&fixture_path()).expect("fixture loads");
    let server = InferenceServer::new(evaluator, DEFAULT_MAX_BATCH);
    // Dropping immediately (no requests submitted) must not hang or panic.
    drop(server);
}
