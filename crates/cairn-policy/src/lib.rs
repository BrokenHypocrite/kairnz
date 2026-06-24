pub mod policy;
pub mod random;

#[cfg(test)]
mod tests {
    use cairn_core::{actions::legal_actions, config::RuleConfig, game::Game};

    use crate::policy::Policy;
    use crate::random::RandomPolicy;

    #[test]
    fn random_policy_is_deterministic_for_a_seed() {
        let g = Game::new_standard(RuleConfig::default());
        let a = RandomPolicy::seeded(42).choose(&g);
        let b = RandomPolicy::seeded(42).choose(&g);
        assert_eq!(a, b, "same seed must produce same action from same position");
    }

    #[test]
    fn random_policy_only_returns_legal_actions() {
        let g = Game::new_standard(RuleConfig::default());
        let legal = legal_actions(&g.pos);
        let action = RandomPolicy::seeded(7).choose(&g);
        assert!(
            action.is_some(),
            "standard opening position must have legal actions"
        );
        assert!(
            legal.contains(&action.unwrap()),
            "chosen action must be in the legal set"
        );
    }

    #[test]
    fn random_policy_name_is_random() {
        let p = RandomPolicy::seeded(0);
        assert_eq!(p.name(), "random");
    }
}
