//! Self-play data generation for the Kairnz AlphaZero pipeline.

pub mod gate;
pub mod parallel;
pub mod play;
pub mod sample;
pub mod shard;

use serde::{Deserialize, Serialize};

/// Default number of MCTS simulations per move during self-play.
const DEFAULT_SIMULATIONS: u32 = 200;
/// Default PUCT exploration constant.
const DEFAULT_C_PUCT: f64 = 1.5;
/// Default Dirichlet root-noise weight (exploration is ON for self-play).
const DEFAULT_DIRICHLET_EPSILON: f64 = 0.25;
/// Default Dirichlet concentration.
const DEFAULT_DIRICHLET_ALPHA: f64 = 0.3;
/// Default number of opening plies that sample moves proportional to visits
/// before switching to argmax.
const DEFAULT_TEMPERATURE_CUTOFF: u32 = 20;
/// Default number of self-play games to generate.
const DEFAULT_GAMES: u32 = 64;

/// Parameters controlling a self-play run.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SelfPlayConfig {
    /// MCTS simulations per move.
    pub simulations: u32,
    /// PUCT exploration constant.
    pub c_puct: f64,
    /// Dirichlet root-noise weight.
    pub dirichlet_epsilon: f64,
    /// Dirichlet concentration.
    pub dirichlet_alpha: f64,
    /// Plies of visit-proportional sampling before switching to argmax.
    pub temperature_cutoff: u32,
    /// Number of games to play.
    pub games: u32,
}

impl Default for SelfPlayConfig {
    fn default() -> Self {
        SelfPlayConfig {
            simulations: DEFAULT_SIMULATIONS,
            c_puct: DEFAULT_C_PUCT,
            dirichlet_epsilon: DEFAULT_DIRICHLET_EPSILON,
            dirichlet_alpha: DEFAULT_DIRICHLET_ALPHA,
            temperature_cutoff: DEFAULT_TEMPERATURE_CUTOFF,
            games: DEFAULT_GAMES,
        }
    }
}

impl SelfPlayConfig {
    /// Builds the MCTS search config from these self-play parameters.
    pub fn mcts_config(&self) -> kairnz_onnx::AzMctsConfig {
        kairnz_onnx::AzMctsConfig {
            simulations: self.simulations,
            c_puct: self.c_puct,
            dirichlet_alpha: self.dirichlet_alpha,
            dirichlet_epsilon: self.dirichlet_epsilon,
            ..kairnz_onnx::AzMctsConfig::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_enables_root_noise_for_exploration() {
        let c = SelfPlayConfig::default();
        assert!(c.dirichlet_epsilon > 0.0, "self-play must explore");
        assert_eq!(c.mcts_config().dirichlet_epsilon, c.dirichlet_epsilon);
    }
}
