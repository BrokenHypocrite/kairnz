use serde::{Deserialize, Serialize};

const DEFAULT_FIRST_TURN_AP: u8 = 2;
const DEFAULT_MAX_PLIES: u32 = 400;
const DEFAULT_REPETITION_FOLD: u8 = 3;

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub enum SpireMode {
    Dragon,
    Queen,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct RuleConfig {
    pub spire: SpireMode,
    pub first_turn_ap: u8,
    pub capture_lock: bool,
    pub keystone_single_move: bool,
    pub max_plies: u32,
    pub repetition_fold: u8,
}

impl Default for RuleConfig {
    fn default() -> Self {
        Self {
            spire: SpireMode::Dragon,
            first_turn_ap: DEFAULT_FIRST_TURN_AP,
            capture_lock: false,
            keystone_single_move: false,
            max_plies: DEFAULT_MAX_PLIES,
            repetition_fold: DEFAULT_REPETITION_FOLD,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_matches_spec_defaults() {
        let c = RuleConfig::default();
        assert!(matches!(c.spire, SpireMode::Dragon));
        assert_eq!(c.first_turn_ap, 2);
        assert!(!c.capture_lock && !c.keystone_single_move);
    }

    #[test]
    fn config_roundtrips_yaml() {
        let c = RuleConfig::default();
        let y = serde_yaml::to_string(&c).unwrap();
        let back: RuleConfig = serde_yaml::from_str(&y).unwrap();
        assert_eq!(back.first_turn_ap, c.first_turn_ap);
    }
}
