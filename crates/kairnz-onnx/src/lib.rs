//! ONNX Runtime inference for Kairnz: load an exported model and evaluate
//! positions into policy logits and a value, and play via `OnnxPolicy`.
//!
//! This crate isolates the native `ort` (ONNX Runtime) dependency from the
//! lightweight game crates.

pub mod evaluator;
pub mod policy;
pub mod mcts;

pub use evaluator::OnnxEvaluator;
pub use policy::OnnxPolicy;
pub use mcts::AzMctsConfig;

/// The execution backend a session is running on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    /// NVIDIA CUDA execution provider.
    Cuda,
    /// CPU execution provider (the default and fallback).
    Cpu,
}
