use kairnz_core::{actions::{legal_actions, Action}, game::Game};
use rand::Rng;
use rand::SeedableRng;
use rand_pcg::Pcg64;

use crate::policy::Policy;

/// A policy that selects a uniformly random legal action using a seeded PRNG.
///
/// Determinism is guaranteed: two instances constructed with the same seed
/// will make identical choices from the same position.
pub struct RandomPolicy {
    rng: Pcg64,
}

impl RandomPolicy {
    /// Construct a `RandomPolicy` with a reproducible seed.
    ///
    /// Two policies created with the same `seed` will choose identically.
    pub fn seeded(seed: u64) -> RandomPolicy {
        RandomPolicy {
            rng: Pcg64::seed_from_u64(seed),
        }
    }
}

impl Policy for RandomPolicy {
    /// Choose a uniformly random legal action, or `None` if there are none.
    fn choose(&mut self, game: &Game) -> Option<Action> {
        let actions = legal_actions(&game.pos);
        if actions.is_empty() {
            return None;
        }
        let idx = self.rng.gen_range(0..actions.len());
        Some(actions[idx])
    }

    /// Returns `"random"`.
    fn name(&self) -> &str {
        "random"
    }
}
