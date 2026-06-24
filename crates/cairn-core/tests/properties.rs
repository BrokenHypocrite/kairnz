use cairn_core::actions::legal_actions;
use cairn_core::config::RuleConfig;
use cairn_core::game::Game;
use cairn_core::piece::{PieceKind, Player};
use cairn_core::square::NUM_SQUARES;
use proptest::prelude::*;

/// Maximum steps to play before stopping the random playout.
const MAX_STEPS: usize = 300;

/// Computes total stone tokens in the position: sum of Stone heights on the board
/// for both players, plus both reserve counts.
fn stone_token_total(game: &Game) -> u32 {
    let board_tokens: u32 = game
        .pos
        .board
        .iter()
        .filter_map(|cell| *cell)
        .filter(|pc| pc.kind == PieceKind::Stone)
        .map(|pc| pc.height as u32)
        .sum();
    let reserve_tokens: u32 = game.pos.reserves.iter().map(|&r| r as u32).sum();
    board_tokens + reserve_tokens
}

/// Asserts that all invariants hold for the current game state.
fn assert_invariants(game: &Game, expected_token_total: u32) {
    // Token conservation: stone tokens are conserved.
    assert_eq!(
        stone_token_total(game),
        expected_token_total,
        "stone-token total must be conserved"
    );

    // AP in range.
    assert!(
        game.pos.turn.ap_remaining <= 2,
        "ap_remaining must be <= 2, got {}",
        game.pos.turn.ap_remaining
    );

    // Height constraints: all board pieces must satisfy height rules.
    for i in 0..NUM_SQUARES {
        if let Some(pc) = game.pos.board[i] {
            match pc.kind {
                PieceKind::Stone => {
                    assert!(
                        pc.height >= 1 && pc.height <= 3,
                        "Stone at index {} must have height in 1..=3, got {}",
                        i,
                        pc.height
                    );
                }
                PieceKind::Keystone => {
                    assert_eq!(
                        pc.height, 1,
                        "Keystone at index {} must have height == 1, got {}",
                        i,
                        pc.height
                    );
                }
            }
        }
    }
}

/// Minimal inline xorshift64 PRNG. Returns the next state and a derived value.
fn xorshift_next(mut s: u64) -> (u64, u64) {
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    (s, s)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]
    #[test]
    fn invariants_hold_under_random_play(seed in any::<u64>()) {
        let config = RuleConfig::default();
        let game = Game::new_standard(config);

        // Derive the expected token total from the initial position so this is
        // not a hardcoded magic number.
        let expected_token_total = stone_token_total(&game);

        // Assert invariants on the initial position.
        assert_invariants(&game, expected_token_total);

        // Track keystone counts: they must be monotonically non-increasing.
        let mut ks_p1 = game.pos.keystones_of(Player::P1).count();
        let mut ks_p2 = game.pos.keystones_of(Player::P2).count();

        let mut game = game;
        let mut prng_state = if seed == 0 { 1 } else { seed };

        for _ in 0..MAX_STEPS {
            // Stop if the game has ended.
            if game.terminal_result().is_some() {
                break;
            }

            let actions = legal_actions(&game.pos);
            if actions.is_empty() {
                break;
            }

            // Pick a legal action deterministically from the PRNG.
            let (next_state, val) = xorshift_next(prng_state);
            prng_state = next_state;
            let chosen = &actions[(val as usize) % actions.len()];
            game.apply(*chosen).expect("chosen legal action must apply without error");

            // Assert invariants after every action.
            assert_invariants(&game, expected_token_total);

            // Keystone counts must be non-increasing.
            let new_ks_p1 = game.pos.keystones_of(Player::P1).count();
            let new_ks_p2 = game.pos.keystones_of(Player::P2).count();
            assert!(
                new_ks_p1 <= ks_p1,
                "P1 keystone count must not increase: was {}, now {}",
                ks_p1,
                new_ks_p1
            );
            assert!(
                new_ks_p2 <= ks_p2,
                "P2 keystone count must not increase: was {}, now {}",
                ks_p2,
                new_ks_p2
            );
            ks_p1 = new_ks_p1;
            ks_p2 = new_ks_p2;
        }
    }
}
