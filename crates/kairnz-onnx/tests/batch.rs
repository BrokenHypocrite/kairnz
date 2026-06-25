//! Tolerance tests for the batched evaluator: batch output must match
//! per-position `evaluate` output within 1e-4 on value and sampled policy entries.

use std::path::PathBuf;

use kairnz_core::config::RuleConfig;
use kairnz_core::position::Position;
use kairnz_encode::{encode_planes, POLICY_SIZE};
use kairnz_onnx::{BatchEvaluator, DirectBatchEvaluator, OnnxEvaluator};

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/random_init.onnx")
}

/// Sampled policy indices to compare (spread across the three regions).
const SAMPLED_POLICY_INDICES: &[usize] = &[0, 100, 500, 1000, 3000, 6000, 6722];

#[test]
fn batch_matches_single_evaluate_within_tolerance() {
    let mut evaluator = OnnxEvaluator::from_path(&fixture_path()).expect("fixture loads");
    let pos = Position::new_standard(RuleConfig::default());

    // Build a small set of test cases: 3 positions with different rep counts.
    let rep_counts: &[u8] = &[0, 1, 2];
    let positions: Vec<&Position> = vec![&pos, &pos, &pos];

    // Collect single-evaluate reference outputs.
    let mut reference: Vec<(Vec<f32>, f32)> = Vec::new();
    for (&rep, &p) in rep_counts.iter().zip(positions.iter()) {
        let result = evaluator.evaluate(p, rep).expect("single evaluate succeeds");
        reference.push(result);
    }

    // Build pre-encoded plane vectors (with reps=0; reps[i] will be applied by evaluate_batch).
    // We pass planes encoded with rep=0, then supply the actual rep counts via reps[].
    // evaluate_batch overwrites the repetition channel, so the result should match evaluate().
    let planes: Vec<Vec<f32>> = positions
        .iter()
        .map(|p| encode_planes(p, 0))
        .collect();

    // Run batched inference.
    let batch_results = evaluator
        .evaluate_batch(&planes, rep_counts)
        .expect("batch evaluate succeeds");

    assert_eq!(batch_results.len(), 3, "batch returns one result per input");

    for (i, ((batch_policy, batch_value), (ref_policy, ref_value))) in
        batch_results.iter().zip(reference.iter()).enumerate()
    {
        // Policy length.
        assert_eq!(
            batch_policy.len(),
            POLICY_SIZE,
            "row {i}: policy length should be POLICY_SIZE (6723)"
        );
        assert_eq!(
            ref_policy.len(),
            POLICY_SIZE,
            "row {i}: reference policy length should be POLICY_SIZE (6723)"
        );

        // Value tolerance.
        let value_diff = (batch_value - ref_value).abs();
        assert!(
            value_diff < 1e-4,
            "row {i}: value diff {value_diff} exceeds 1e-4 (batch={batch_value}, ref={ref_value})"
        );

        // Sampled policy tolerance.
        for &idx in SAMPLED_POLICY_INDICES {
            let diff = (batch_policy[idx] - ref_policy[idx]).abs();
            assert!(
                diff < 1e-4,
                "row {i}: policy[{idx}] diff {diff} exceeds 1e-4 (batch={}, ref={})",
                batch_policy[idx],
                ref_policy[idx]
            );
        }
    }
}

#[test]
fn batch_policy_length_is_6723() {
    let mut evaluator = OnnxEvaluator::from_path(&fixture_path()).expect("fixture loads");
    let pos = Position::new_standard(RuleConfig::default());
    let planes = vec![encode_planes(&pos, 0)];
    let reps = vec![0u8];

    let results = evaluator
        .evaluate_batch(&planes, &reps)
        .expect("batch evaluate succeeds");

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].0.len(), POLICY_SIZE);
    assert_eq!(POLICY_SIZE, 6723);
}

#[test]
fn direct_batch_evaluator_via_trait_object() {
    let evaluator = OnnxEvaluator::from_path(&fixture_path()).expect("fixture loads");
    let direct = DirectBatchEvaluator::new(evaluator);

    let pos = Position::new_standard(RuleConfig::default());
    let planes = vec![encode_planes(&pos, 0), encode_planes(&pos, 0)];
    let reps = vec![0u8, 1u8];

    let results = direct
        .evaluate_batch(&planes, &reps)
        .expect("direct batch evaluate succeeds");

    assert_eq!(results.len(), 2);
    for (policy, value) in &results {
        assert_eq!(policy.len(), POLICY_SIZE);
        assert!(policy.iter().all(|v| v.is_finite()), "policy logits finite");
        assert!(*value >= -1.0 && *value <= 1.0, "value {value} in [-1, 1]");
    }
}

#[test]
fn evaluate_batch_empty_input_returns_empty() {
    let mut evaluator = OnnxEvaluator::from_path(&fixture_path()).expect("fixture loads");
    let results = evaluator
        .evaluate_batch(&[], &[])
        .expect("empty batch succeeds");
    assert!(results.is_empty());
}
