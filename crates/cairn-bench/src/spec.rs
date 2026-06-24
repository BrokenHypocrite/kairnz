use cairn_core::config::RuleConfig;
use cairn_policy::{greedy::GreedyPolicy, mcts::MctsPolicy, policy::Policy, random::RandomPolicy};
use serde::Deserialize;

/// Default rollout cap for MCTS in benchmark mode (modest for speed).
const BENCH_DEFAULT_ROLLOUT_CAP: u32 = 100;

/// A named rule configuration pairing a human-readable label with a RuleConfig.
#[derive(Deserialize, Clone)]
pub struct NamedConfig {
    pub name: String,
    pub config: RuleConfig,
}

/// Describes which policy to use and its parameters.
/// YAML shapes:
///   p1_policy: Random
///   p1_policy: Greedy
///   p1_policy: !Mcts { iterations: 100 }
///   p1_policy: !Mcts { iterations: 200, rollout_cap: 50 }
#[derive(Deserialize, Clone)]
pub enum PolicySpec {
    Random,
    Greedy,
    Mcts {
        iterations: u32,
        rollout_cap: Option<u32>,
    },
}

/// A complete benchmark run specification parsed from YAML.
#[derive(Deserialize, Clone)]
pub struct RunSpec {
    pub configs: Vec<NamedConfig>,
    pub games_per_config: u32,
    pub seed: u64,
    pub p1_policy: PolicySpec,
    pub p2_policy: PolicySpec,
}

/// Constructs a boxed policy from a spec and a seed.
pub fn build_policy(spec: &PolicySpec, seed: u64) -> Box<dyn Policy> {
    match spec {
        PolicySpec::Random => Box::new(RandomPolicy::seeded(seed)),
        PolicySpec::Greedy => Box::new(GreedyPolicy::seeded(seed)),
        PolicySpec::Mcts {
            iterations,
            rollout_cap,
        } => Box::new(MctsPolicy::with_params(
            *iterations,
            std::f64::consts::SQRT_2,
            rollout_cap.unwrap_or(BENCH_DEFAULT_ROLLOUT_CAP),
            seed,
        )),
    }
}

/// Loads and parses a RunSpec from a YAML file at `path`.
pub fn load_run_spec(path: &str) -> Result<RunSpec, String> {
    let contents = std::fs::read_to_string(path)
        .map_err(|e| format!("failed to read spec file '{}': {}", path, e))?;
    serde_yaml::from_str(&contents).map_err(|e| format!("failed to parse spec YAML: {}", e))
}

#[cfg(test)]
mod tests {
    use super::*;
    use cairn_core::config::SpireMode;

    #[test]
    fn runspec_parses_from_yaml() {
        let yaml = r#"
configs:
  - name: default
    config:
      spire: Dragon
      first_turn_ap: 2
      capture_lock: false
      keystone_single_move: false
      max_plies: 400
      repetition_fold: 3
  - name: queen
    config:
      spire: Queen
      first_turn_ap: 2
      capture_lock: false
      keystone_single_move: false
      max_plies: 400
      repetition_fold: 3
games_per_config: 4
seed: 99
p1_policy: Random
p2_policy: Greedy
"#;
        let spec: RunSpec = serde_yaml::from_str(yaml).expect("parse failed");
        assert_eq!(spec.configs.len(), 2);
        assert_eq!(spec.configs[0].name, "default");
        assert_eq!(spec.configs[1].name, "queen");
        assert!(matches!(spec.configs[1].config.spire, SpireMode::Queen));
        assert_eq!(spec.games_per_config, 4);
        assert_eq!(spec.seed, 99);
        assert!(matches!(spec.p1_policy, PolicySpec::Random));
        assert!(matches!(spec.p2_policy, PolicySpec::Greedy));
    }

    #[test]
    fn build_policy_constructs_each_variant() {
        let r = build_policy(&PolicySpec::Random, 1);
        assert_eq!(r.name(), "random");
        let g = build_policy(&PolicySpec::Greedy, 2);
        assert_eq!(g.name(), "greedy");
        let m = build_policy(
            &PolicySpec::Mcts {
                iterations: 10,
                rollout_cap: Some(20),
            },
            3,
        );
        assert_eq!(m.name(), "mcts");
    }
}
