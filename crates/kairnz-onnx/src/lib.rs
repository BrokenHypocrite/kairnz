//! ONNX Runtime inference for Kairnz: load an exported model and evaluate
//! positions into policy logits and a value, and play via `OnnxPolicy`.
//!
//! This crate isolates the native `ort` (ONNX Runtime) dependency from the
//! lightweight game crates.

pub mod evaluator;
pub mod policy;
pub mod mcts;
pub mod batch;
pub mod batched_mcts;

pub use evaluator::OnnxEvaluator;
pub use policy::OnnxPolicy;
pub use mcts::{AzMcts, AzMctsConfig, AzMctsPolicy};
pub use batch::{BatchEvaluator, DEFAULT_MAX_BATCH, DirectBatchEvaluator, InferenceServer};
pub use batched_mcts::BatchedAzMcts;

/// The execution backend a session is running on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    /// NVIDIA CUDA execution provider.
    Cuda,
    /// CPU execution provider (the default and fallback).
    Cpu,
}

/// A common interface for MCTS engines used during self-play.
///
/// Implemented by both [`AzMcts`] (single-threaded, infallible search) and
/// [`BatchedAzMcts`] (batched, shared-server search). Callers that are generic
/// over `impl Searcher` avoid duplicating game-loop logic.
pub trait Searcher {
    /// Runs a search from `game` and returns each root child's
    /// `(action, visit_count)`. Empty for terminal positions.
    fn search(&mut self, game: &kairnz_core::game::Game)
        -> ort::Result<Vec<(kairnz_core::actions::Action, u32)>>;
}

impl Searcher for AzMcts {
    /// Wraps the infallible [`AzMcts::search`] in `Ok(...)`.
    fn search(
        &mut self,
        game: &kairnz_core::game::Game,
    ) -> ort::Result<Vec<(kairnz_core::actions::Action, u32)>> {
        Ok(AzMcts::search(self, game))
    }
}

impl<'a> Searcher for BatchedAzMcts<'a> {
    fn search(
        &mut self,
        game: &kairnz_core::game::Game,
    ) -> ort::Result<Vec<(kairnz_core::actions::Action, u32)>> {
        BatchedAzMcts::search(self, game)
    }
}
