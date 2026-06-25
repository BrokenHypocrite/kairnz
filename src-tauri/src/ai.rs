//! In-app AI opponent: a lazily-loaded neural MCTS policy.

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use kairnz_core::actions::Action;
use kairnz_core::game::Game;
use kairnz_onnx::{AzMctsConfig, BatchedAzMcts, DirectBatchEvaluator, OnnxEvaluator};

/// Fixed seed for the in-app AI (deterministic; epsilon 0 means the seed is inert).
const AI_SEED: u64 = 0;

/// Number of leaves collected per batched MCTS step.
const LEAVES_PER_STEP: usize = 8;

/// A loaded evaluator plus the model path it was built for.
struct Loaded {
    model: PathBuf,
    evaluator: DirectBatchEvaluator,
}

/// Tauri-managed state holding a lazily-loaded, reusable AI evaluator.
#[derive(Default)]
pub struct AiEngine {
    inner: Mutex<Option<Loaded>>,
}

impl AiEngine {
    /// Chooses the AI's move for `game`, loading or reusing an evaluator for the
    /// given model path, then running a `BatchedAzMcts` search with the requested
    /// simulation budget. Returns an error string on model-load failure or if no
    /// legal move exists. With cuDNN on PATH the `OnnxEvaluator` uses the CUDA EP
    /// automatically; otherwise it falls back to CPU transparently.
    pub fn choose(&self, game: &Game, model_path: &Path, simulations: u32) -> Result<Action, String> {
        let mut guard = self.inner.lock().map_err(|_| "AI engine lock poisoned".to_string())?;

        // Reload the evaluator only when the model path changes; simulations is
        // applied per-call via AzMctsConfig and does not affect the cached session.
        let needs_load = match guard.as_ref() {
            Some(loaded) => loaded.model != model_path,
            None => true,
        };
        if needs_load {
            let onnx = OnnxEvaluator::from_path(model_path)
                .map_err(|e| format!("failed to load AI model: {e}"))?;
            let evaluator = DirectBatchEvaluator::new(onnx);
            *guard = Some(Loaded { model: model_path.to_path_buf(), evaluator });
        }

        let loaded = guard.as_mut().expect("evaluator was just loaded");

        // Build and run BatchedAzMcts while holding the lock so the &dyn
        // BatchEvaluator borrow into loaded.evaluator remains valid.
        let config = AzMctsConfig {
            simulations,
            dirichlet_epsilon: 0.0,
            leaves_per_step: LEAVES_PER_STEP,
            ..AzMctsConfig::default()
        };
        let mut mcts = BatchedAzMcts::new(&loaded.evaluator, config, AI_SEED);
        let visits = mcts.search(game).map_err(|e| format!("MCTS search failed: {e}"))?;

        visits
            .into_iter()
            .max_by_key(|&(_, v)| v)
            .map(|(action, _)| action)
            .ok_or_else(|| "AI found no legal move".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kairnz_core::actions::legal_actions;
    use kairnz_core::config::RuleConfig;
    use std::path::PathBuf;

    fn fixture() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../crates/kairnz-onnx/tests/fixtures/random_init.onnx")
    }

    #[test]
    fn ai_chooses_a_legal_move_at_the_opening() {
        let engine = AiEngine::default();
        let game = Game::new_standard(RuleConfig::default());
        let action = engine.choose(&game, &fixture(), 16).expect("ai chooses");
        assert!(legal_actions(&game.pos).contains(&action), "AI move must be legal");
    }
}
