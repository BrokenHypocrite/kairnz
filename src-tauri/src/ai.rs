//! In-app AI opponent: a lazily-loaded neural MCTS policy.

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use kairnz_core::actions::Action;
use kairnz_core::game::Game;
use kairnz_onnx::{AzMctsConfig, AzMctsPolicy};
use kairnz_policy::policy::Policy;

/// Fixed seed for the in-app AI (deterministic; epsilon 0 means the seed is inert).
const AI_SEED: u64 = 0;

/// A loaded policy plus the model/strength it was built for.
struct Loaded {
    model: PathBuf,
    simulations: u32,
    policy: AzMctsPolicy,
}

/// Tauri-managed state holding a lazily-loaded, reusable AI policy.
#[derive(Default)]
pub struct AiEngine {
    inner: Mutex<Option<Loaded>>,
}

impl AiEngine {
    /// Chooses the AI's move for `game`, loading or reusing a policy for the given
    /// model path and simulation budget. Returns an error string on model-load
    /// failure or if no legal move exists.
    pub fn choose(&self, game: &Game, model_path: &Path, simulations: u32) -> Result<Action, String> {
        let mut guard = self.inner.lock().map_err(|_| "AI engine lock poisoned".to_string())?;

        let needs_load = match guard.as_ref() {
            Some(loaded) => loaded.model != model_path || loaded.simulations != simulations,
            None => true,
        };
        if needs_load {
            let config = AzMctsConfig {
                simulations,
                dirichlet_epsilon: 0.0,
                ..AzMctsConfig::default()
            };
            let policy = AzMctsPolicy::from_path(model_path, config, AI_SEED)
                .map_err(|e| format!("failed to load AI model: {e}"))?;
            *guard = Some(Loaded { model: model_path.to_path_buf(), simulations, policy });
        }

        let loaded = guard.as_mut().expect("policy was just loaded");
        loaded
            .policy
            .choose(game)
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
