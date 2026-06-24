use cairn_core::actions::legal_actions;
use cairn_core::config::RuleConfig;
use cairn_core::game::Game;
use cairn_core::position::Position;

/// Returns the number of legal actions from the standard opening with the given first_turn_ap.
fn opening_count(first_turn_ap: u8) -> usize {
    let mut cfg = RuleConfig::default();
    cfg.first_turn_ap = first_turn_ap;
    let pos = Position::new_standard(cfg);
    legal_actions(&pos).len()
}

/// Returns the depth-2 sum for first_turn_ap = 2: sum over each legal first action
/// of the number of legal actions available immediately after applying it.
fn depth2_sum_ap2() -> usize {
    let mut cfg = RuleConfig::default();
    cfg.first_turn_ap = 2;
    let game = Game::new_standard(cfg);
    let first_actions = legal_actions(&game.pos);
    let mut total = 0usize;
    for action in &first_actions {
        let mut branch = Game::new_standard({
            let mut cfg = RuleConfig::default();
            cfg.first_turn_ap = 2;
            cfg
        });
        branch.apply(*action).expect("first action must be legal");
        // Count legal actions at the resulting position.
        total += legal_actions(&branch.pos).len();
    }
    total
}

// --- Regression anchors ---
// These values were obtained by running the test once with placeholder panics
// and reading the actual output, then hardcoded here. Recompute only when
// movement or generation rules intentionally change.

#[test]
fn opening_legal_count_first_turn_ap1() {
    // AP=1 means no Stack actions on the first turn (Stack costs 2 AP).
    // Regression anchor for first_turn_ap = 1.
    let count = opening_count(1);
    assert_eq!(count, EXPECTED_AP1, "opening legal count for first_turn_ap=1 changed; recompute if rules changed");
}

#[test]
fn opening_legal_count_first_turn_ap2() {
    // AP=2 enables Stack (costs 2 AP), so more actions are available.
    // Regression anchor for first_turn_ap = 2.
    let count = opening_count(2);
    assert_eq!(count, EXPECTED_AP2, "opening legal count for first_turn_ap=2 changed; recompute if rules changed");
}

#[test]
fn depth2_sum_first_turn_ap2() {
    // Sum of legal action counts at depth 2 from the AP=2 opening.
    // Regression anchor; recompute only if movement or generation rules change.
    let sum = depth2_sum_ap2();
    assert_eq!(sum, EXPECTED_DEPTH2_SUM, "depth-2 sum for first_turn_ap=2 changed; recompute if rules changed");
}

// These values were obtained by running the tests once with placeholder zeroes
// and recording the actual counts from the failure output, then hardcoded here.
// Recompute only when movement or generation rules intentionally change.
//
// AP=1 and AP=2 yield the same opening count (27) because the standard position
// has no reserves, so Stack actions (which require reserve) never appear at depth 1
// regardless of AP budget.
//
// Recomputed after correcting keystone placement to symmetric files 2/6
// (the 3rd and 7th files, 0-indexed). Depth-2 sum changed from 787 to 788.
const EXPECTED_AP1: usize = 27;
const EXPECTED_AP2: usize = 27;
const EXPECTED_DEPTH2_SUM: usize = 788;
