//! A `Policy` that selects the highest-logit legal action from a loaded model.

use std::cmp::Ordering;
use std::path::Path;

use kairnz_core::actions::{legal_actions, Action};
use kairnz_core::game::Game;
use kairnz_encode::action_to_index;
use kairnz_policy::policy::Policy;

use crate::OnnxEvaluator;

/// Plays Kairnz by evaluating the current position with an ONNX model and
/// choosing the legal action with the highest policy logit (no search).
pub struct OnnxPolicy {
    evaluator: OnnxEvaluator,
}

impl OnnxPolicy {
    /// Loads a model from `path` for raw-policy play.
    pub fn from_path(path: &Path) -> ort::Result<OnnxPolicy> {
        Ok(OnnxPolicy { evaluator: OnnxEvaluator::from_path(path)? })
    }
}

impl Policy for OnnxPolicy {
    fn choose(&mut self, game: &Game) -> Option<Action> {
        let pos = &game.pos;
        let actions = legal_actions(pos);
        if actions.is_empty() {
            return None;
        }

        let (policy, _value) = match self.evaluator.evaluate(pos, 0) {
            Ok(output) => output,
            Err(error) => {
                eprintln!("OnnxPolicy inference failed: {error}");
                return None;
            }
        };

        let mover = pos.to_move;
        actions.into_iter().max_by(|a, b| {
            let la = policy[action_to_index(a, mover)];
            let lb = policy[action_to_index(b, mover)];
            la.partial_cmp(&lb).unwrap_or(Ordering::Equal)
        })
    }

    fn name(&self) -> &str {
        "onnx"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    use kairnz_core::actions::legal_actions;
    use kairnz_core::config::RuleConfig;
    use kairnz_core::game::Game;

    fn fixture_policy() -> OnnxPolicy {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/random_init.onnx");
        OnnxPolicy::from_path(&path).expect("fixture loads")
    }

    #[test]
    fn chooses_a_legal_action_at_the_opening() {
        let mut policy = fixture_policy();
        let game = Game::new_standard(RuleConfig::default());

        let action = policy.choose(&game).expect("a legal action exists at the opening");
        assert!(
            legal_actions(&game.pos).contains(&action),
            "chosen action must be legal"
        );
    }

    #[test]
    fn plays_a_full_game_with_only_legal_actions() {
        let mut policy = fixture_policy();
        let mut game = Game::new_standard(RuleConfig::default());

        // Drive the game to a terminal state; every chosen action must be legal.
        // terminal_result is valid here: after every apply the game is either at a
        // fresh turn boundary, or mid-turn with at least one legal action remaining
        // (a mid-turn dead end auto-ends the turn inside apply).
        let mut guard = 0;
        while game.terminal_result().is_none() {
            let legal = legal_actions(&game.pos);
            let action = match policy.choose(&game) {
                Some(a) => a,
                None => break,
            };
            assert!(legal.contains(&action), "every chosen action must be legal");
            game.apply(action).expect("legal action applies");
            guard += 1;
            assert!(guard < 2000, "game should terminate within the ply cap");
        }
    }
}
