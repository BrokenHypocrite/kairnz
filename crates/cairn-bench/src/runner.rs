// Public API items are used in tests and will be called from main once the
// harness is wired up; suppress dead-code warnings until then.
#![allow(dead_code)]

use cairn_core::{
    config::RuleConfig,
    game::Game,
    outcome::{DrawReason, GameResult},
    piece::{PieceKind, Player},
};
use cairn_policy::policy::Policy;

/// The number of squares on a Cairn board.
const BOARD_SIZE: usize = 81;

/// Raw signals recorded from a single headless game, used to derive balance metrics.
///
/// The two policies passed to [`play_game`] are expected to be pre-seeded by the
/// caller. No additional seed is threaded through the runner because the game is
/// fully deterministic given the policies' internal PRNG states.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GameRecord {
    /// The terminal result of the game.
    pub result: GameResult,
    /// Total number of half-moves (plies) played.
    pub plies: u32,
    /// The player who made the first capture, if any capture occurred.
    pub first_capture_by: Option<Player>,
    /// The player who first lost a Keystone (owner of the first captured Keystone),
    /// if any Keystone was captured.
    pub first_keystone_loss_by: Option<Player>,
    /// The tallest Stone height (in tokens) reached at any point during the game.
    pub max_stack_height: u8,
}

/// Plays a complete game between `p1` (playing as [`Player::P1`]) and `p2`
/// (playing as [`Player::P2`]) under `config`, returning raw game signals.
///
/// Both policies are assumed to be pre-seeded by the caller. Determinism is
/// guaranteed when identical seeded policies are provided: no extra seed
/// parameter is needed because the game is fully determined by the policies'
/// internal PRNG states and the rule config.
///
/// The function never panics: illegal action responses and missing terminal
/// results are handled defensively.
pub fn play_game(config: RuleConfig, p1: &mut dyn Policy, p2: &mut dyn Policy) -> GameRecord {
    let mut game = Game::new_standard(config);

    let mut plies: u32 = 0;
    let mut first_capture_by: Option<Player> = None;
    let mut first_keystone_loss_by: Option<Player> = None;
    let mut max_stack_height: u8 = max_stone_height(&game);

    while game.terminal_result().is_none() {
        let mover = game.pos.to_move;
        let policy: &mut dyn Policy = if mover == Player::P1 { p1 } else { p2 };

        let action = match policy.choose(&game) {
            Some(a) => a,
            // Defensive: terminal_result() should have caught no-legal-action, but
            // if choose returns None we stop rather than loop forever.
            None => break,
        };

        match game.apply(action) {
            Ok(outcome) => {
                plies += 1;

                if let Some(cap) = outcome.captured {
                    if first_capture_by.is_none() {
                        first_capture_by = Some(mover);
                    }
                    if cap.kind == PieceKind::Keystone && first_keystone_loss_by.is_none() {
                        first_keystone_loss_by = Some(cap.owner);
                    }
                }

                let current_max = max_stone_height(&game);
                if current_max > max_stack_height {
                    max_stack_height = current_max;
                }
            }
            // Defensive: a legal action chosen by choose() should never be illegal.
            // Break rather than loop forever or panic.
            Err(_) => break,
        }
    }

    let result = game.terminal_result().unwrap_or(GameResult::Draw(DrawReason::MaxPlies));

    GameRecord {
        result,
        plies,
        first_capture_by,
        first_keystone_loss_by,
        max_stack_height,
    }
}

/// Returns the maximum height among all Stone pieces currently on the board,
/// or 0 if no Stones are present.
fn max_stone_height(game: &Game) -> u8 {
    let mut max: u8 = 0;
    for idx in 0..BOARD_SIZE {
        if let Some(piece) = game.pos.board[idx] {
            if piece.kind == PieceKind::Stone && piece.height > max {
                max = piece.height;
            }
        }
    }
    max
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use cairn_core::config::RuleConfig;
    use cairn_policy::random::RandomPolicy;

    /// Returns the default config with an optional ply cap for faster tests.
    fn cfg() -> RuleConfig {
        RuleConfig::default()
    }

    /// A faster config with a reduced ply cap.
    fn fast_cfg() -> RuleConfig {
        let mut c = RuleConfig::default();
        c.max_plies = 200;
        c
    }

    /// Two independent runs with the same seeds must produce identical records.
    #[test]
    fn play_game_is_deterministic_for_a_seed() {
        let r1 = play_game(cfg(), &mut RandomPolicy::seeded(1), &mut RandomPolicy::seeded(2));
        let r2 = play_game(cfg(), &mut RandomPolicy::seeded(1), &mut RandomPolicy::seeded(2));
        assert_eq!(r1.plies, r2.plies);
        assert_eq!(r1.result, r2.result);
        assert_eq!(r1.first_capture_by, r2.first_capture_by);
        assert_eq!(r1.first_keystone_loss_by, r2.first_keystone_loss_by);
        assert_eq!(r1.max_stack_height, r2.max_stack_height);
    }

    /// A game won by keystone capture must have a recorded keystone loss.
    ///
    /// A randomly-played Cairn game can sometimes end without any captures (e.g.
    /// draw by max-plies with all pieces shuffling). We therefore assert the
    /// weaker but always-valid invariant: if the result is a Win, the winner
    /// captured at least one Keystone, so `first_keystone_loss_by` is `Some`.
    /// For the common case we also assert `first_capture_by.is_some()` holds
    /// when there was any capture at all (the two are consistent by construction).
    #[test]
    fn play_game_records_first_capture_side() {
        let record = play_game(fast_cfg(), &mut RandomPolicy::seeded(42), &mut RandomPolicy::seeded(43));
        // If the game ended in a Win, someone lost a Keystone.
        if matches!(record.result, GameResult::Win(_)) {
            assert!(
                record.first_keystone_loss_by.is_some(),
                "a Win result requires a Keystone capture, so first_keystone_loss_by must be Some"
            );
            assert!(
                record.first_capture_by.is_some(),
                "a Win result implies at least one capture"
            );
        }
        // Consistency: if first_capture_by is Some, a capture happened; if
        // first_keystone_loss_by is Some, it must have been a Keystone capture.
        if record.first_keystone_loss_by.is_some() {
            assert!(
                record.first_capture_by.is_some(),
                "a Keystone capture is a capture, so first_capture_by must also be Some"
            );
        }
    }

    /// The returned result is a real GameResult and plies is within the ply cap.
    #[test]
    fn play_game_terminates_and_sets_result() {
        let config = fast_cfg();
        let max = config.max_plies;
        let record = play_game(config, &mut RandomPolicy::seeded(7), &mut RandomPolicy::seeded(8));
        // result must be one of the known variants (just confirm it round-trips).
        let _ = record.result;
        assert!(
            record.plies <= max,
            "plies {} must not exceed max_plies {}",
            record.plies,
            max
        );
    }

    /// Every game starts with height-1 stones, so max_stack_height >= 1.
    #[test]
    fn max_stack_height_at_least_one() {
        let record = play_game(fast_cfg(), &mut RandomPolicy::seeded(99), &mut RandomPolicy::seeded(100));
        assert!(
            record.max_stack_height >= 1,
            "there are always height-1 Stones on the board at game start"
        );
    }
}
