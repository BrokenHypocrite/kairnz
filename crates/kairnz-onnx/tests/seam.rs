//! Hermetic seam test: load the committed ONNX fixture and evaluate a position.

use std::path::PathBuf;

use kairnz_core::config::RuleConfig;
use kairnz_core::position::Position;
use kairnz_encode::POLICY_SIZE;
use kairnz_onnx::OnnxEvaluator;

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/random_init.onnx")
}

#[test]
fn evaluator_loads_fixture_and_returns_contract_shapes() {
    let mut evaluator = OnnxEvaluator::from_path(&fixture_path()).expect("fixture loads");
    let pos = Position::new_standard(RuleConfig::default());

    let (policy, value) = evaluator.evaluate(&pos, 0).expect("evaluation succeeds");

    assert_eq!(policy.len(), POLICY_SIZE, "policy vector length");
    assert!(value >= -1.0 && value <= 1.0, "value {value} in [-1, 1]");
    assert!(policy.iter().all(|v| v.is_finite()), "policy logits are finite");
}
