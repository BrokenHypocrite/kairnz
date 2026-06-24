//! Plays one self-play game and records its training samples.

use kairnz_core::actions::Action;
use kairnz_core::config::RuleConfig;
use kairnz_core::game::Game;
use kairnz_core::piece::Player;
use kairnz_encode::{encode_planes, legal_mask};
use kairnz_onnx::mcts::AzMcts;
use rand::Rng;
use rand_pcg::Pcg64;

use crate::sample::{outcome_value, policy_target, Sample};

/// A partially-built sample: everything except the final value, plus the side
/// to move (needed to assign the perspective-relative value at game end).
struct PendingSample {
    planes: Vec<f32>,
    policy: Vec<f32>,
    legal_mask: Vec<u8>,
    to_move: Player,
}

/// Plays one self-play game from the standard opening using `mcts`, returning the
/// recorded samples with values assigned from the final result.
///
/// Moves are sampled proportional to visit counts for the first
/// `temperature_cutoff` plies (exploration), then chosen greedily (argmax). The
/// recorded policy target is always the raw visit distribution.
pub fn play_game(
    mcts: &mut AzMcts,
    config: RuleConfig,
    temperature_cutoff: u32,
    rng: &mut Pcg64,
) -> Vec<Sample> {
    let mut game = Game::new_standard(config);
    let mut pending: Vec<PendingSample> = Vec::new();
    let mut ply = 0u32;

    while game.terminal_result().is_none() {
        let visits = mcts.search(&game);
        // Defensive: search only returns empty for terminal positions, which the
        // while condition above already excludes. This branch is unreachable per
        // the engine invariant and exists purely as a safety guard.
        if visits.is_empty() {
            break;
        }

        let to_move = game.pos.to_move;
        pending.push(PendingSample {
            planes: encode_planes(&game.pos, game.repetition_count()),
            policy: policy_target(&visits, to_move),
            legal_mask: legal_mask(&game.pos).iter().map(|b| *b as u8).collect(),
            to_move,
        });

        let action = select_move(&visits, ply < temperature_cutoff, rng);
        let _ = game.apply(action);
        ply += 1;
    }

    let result = game.terminal_result();
    pending
        .into_iter()
        .map(|p| {
            let value = match result {
                Some(r) => outcome_value(p.to_move, r),
                None => 0.0,
            };
            Sample { planes: p.planes, policy: p.policy, value, legal_mask: p.legal_mask }
        })
        .collect()
}

/// Selects a move from visit counts: proportional sampling when `explore` is
/// true, otherwise the most-visited action.
fn select_move(visits: &[(Action, u32)], explore: bool, rng: &mut Pcg64) -> Action {
    if explore {
        let total: u32 = visits.iter().map(|(_, v)| *v).sum();
        if total > 0 {
            let mut pick = rng.gen_range(0..total);
            for (action, count) in visits {
                if pick < *count {
                    return *action;
                }
                pick -= *count;
            }
        }
    }
    // Fallback and the post-cutoff path: most-visited action.
    visits
        .iter()
        .max_by_key(|(_, v)| *v)
        .map(|(a, _)| *a)
        .expect("visits is non-empty")
}

#[cfg(test)]
mod tests {
    use super::*;
    use kairnz_encode::{NUM_PLANES, POLICY_SIZE};
    use kairnz_onnx::mcts::AzMcts;
    use kairnz_onnx::{AzMctsConfig, OnnxEvaluator};
    use rand::SeedableRng;
    use std::path::PathBuf;

    fn fixture_mcts() -> AzMcts {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../kairnz-onnx/tests/fixtures/random_init.onnx");
        let evaluator = OnnxEvaluator::from_path(&path).expect("fixture loads");
        // Small simulation count keeps the test fast.
        let config = AzMctsConfig { simulations: 16, ..AzMctsConfig::default() };
        AzMcts::new(evaluator, config, 1)
    }

    #[test]
    fn play_game_produces_well_formed_samples() {
        let mut mcts = fixture_mcts();
        let mut rng = Pcg64::seed_from_u64(42);
        let samples = play_game(&mut mcts, RuleConfig::default(), 4, &mut rng);

        assert!(!samples.is_empty(), "a game produces at least one sample");
        for s in &samples {
            assert_eq!(s.planes.len(), NUM_PLANES * 81);
            assert_eq!(s.policy.len(), POLICY_SIZE);
            assert_eq!(s.legal_mask.len(), POLICY_SIZE);
            let policy_sum: f32 = s.policy.iter().sum();
            assert!((policy_sum - 1.0).abs() < 1e-4, "policy row sums to one");
            assert!(s.value == -1.0 || s.value == 0.0 || s.value == 1.0, "value in {{-1,0,1}}");
            assert!(s.legal_mask.iter().all(|m| *m == 0 || *m == 1), "mask is binary");
        }
    }
}
