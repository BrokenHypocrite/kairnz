//! Reports which ONNX Runtime backend (CUDA or CPU) engages for the fixture
//! model and runs one evaluation. Non-fatal if CUDA is unavailable.

use std::path::PathBuf;

use kairnz_core::config::RuleConfig;
use kairnz_core::position::Position;
use kairnz_onnx::{Backend, OnnxEvaluator};

fn main() -> ort::Result<()> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/random_init.onnx");

    let mut evaluator = OnnxEvaluator::from_path(&path)?;
    match evaluator.backend() {
        Backend::Cuda => println!("backend: CUDA"),
        Backend::Cpu => println!("backend: CPU (CUDA unavailable or not built in)"),
    }

    let pos = Position::new_standard(RuleConfig::default());
    let (policy, value) = evaluator.evaluate(&pos, 0)?;
    println!("policy length: {}, value: {value:.4}", policy.len());

    Ok(())
}
