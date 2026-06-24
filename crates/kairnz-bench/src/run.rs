use crate::metrics::{aggregate, Metrics};
use crate::runner::play_game;
use crate::spec::{build_policy, NamedConfig, PolicySpec};

/// Seed-mixing constant for per-game seed derivation.
const SEED_MIX_A: u64 = 0x9E3779B97F4A7C15;
/// Seed-mixing constant to differentiate P2's stream from P1's.
const SEED_MIX_B: u64 = 0xD1B54A32D192ED03;

/// Runs `games` games of `named.config` with `p1_spec` always as P1 and
/// `p2_spec` always as P2, using deterministic per-game seeds derived from
/// `base_seed`. Returns aggregated metrics.
///
/// P1 is always first player and P2 is always second -- no side alternation --
/// so p1_win_rate / p2_win_rate directly measure first-vs-second-player balance.
pub fn run_config(
    named: &NamedConfig,
    games: u32,
    base_seed: u64,
    p1_spec: &PolicySpec,
    p2_spec: &PolicySpec,
) -> Metrics {
    let records: Vec<_> = (0..games)
        .map(|g| {
            let s = base_seed ^ (g as u64).wrapping_mul(SEED_MIX_A);
            let mut p1 = build_policy(p1_spec, s);
            let mut p2 = build_policy(p2_spec, s ^ SEED_MIX_B);
            play_game(named.config.clone(), p1.as_mut(), p2.as_mut())
        })
        .collect();
    aggregate(&records)
}
