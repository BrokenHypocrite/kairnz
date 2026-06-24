use kairnz_core::{actions::legal_actions, actions::Action, game::Game};
use rand::Rng;
use rand::SeedableRng;
use rand_pcg::Pcg64;

use crate::eval::evaluate;
use crate::policy::Policy;

/// A 1-ply greedy agent that picks the legal action yielding the highest
/// immediate evaluation score, breaking ties uniformly at random.
///
/// Determinism is guaranteed: two instances constructed with the same seed
/// will make identical choices from the same position.
pub struct GreedyPolicy {
    rng: Pcg64,
}

impl GreedyPolicy {
    /// Construct a `GreedyPolicy` with a reproducible seed.
    ///
    /// Two policies created with the same `seed` will choose identically.
    pub fn seeded(seed: u64) -> GreedyPolicy {
        GreedyPolicy {
            rng: Pcg64::seed_from_u64(seed),
        }
    }
}

impl Policy for GreedyPolicy {
    /// Choose the legal action that maximises `evaluate` after the action is
    /// applied. Ties are broken uniformly at random using the seeded RNG.
    ///
    /// Returns `None` only if there are no legal actions.
    fn choose(&mut self, game: &Game) -> Option<Action> {
        let actions = legal_actions(&game.pos);
        if actions.is_empty() {
            return None;
        }

        // The side that is currently choosing; we score from this perspective
        // even after the action flips to_move.
        let perspective = game.pos.to_move;

        let mut best_score = i32::MIN;
        let mut best_actions: Vec<Action> = Vec::new();

        for action in actions {
            let mut clone = game.clone();
            // Ignore illegal-action errors: legal_actions guarantees legality.
            let _ = clone.apply(action);
            let score = evaluate(&clone.pos, perspective);

            if score > best_score {
                best_score = score;
                best_actions.clear();
                best_actions.push(action);
            } else if score == best_score {
                best_actions.push(action);
            }
        }

        // Defensive fallback: unreachable in practice since actions is non-empty.
        if best_actions.is_empty() {
            return None;
        }

        let idx = self.rng.gen_range(0..best_actions.len());
        Some(best_actions[idx])
    }

    /// Returns `"greedy"`.
    fn name(&self) -> &str {
        "greedy"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kairnz_core::{
        actions::Action,
        config::RuleConfig,
        game::Game,
        piece::{Piece, PieceKind, Player},
        position::{Position, TurnState},
        square::{BitBoard81, Sq, NUM_SQUARES},
    };

    fn sq(file: u8, rank: u8) -> Sq {
        Sq::new(file, rank).unwrap()
    }

    /// Build a minimal Game from a raw Position.
    fn game_from_pos(pos: Position) -> Game {
        // Game::new_standard is the only public constructor, but we can reproduce
        // any position by building a standard game and replacing its fields.
        // Since Position and Vec<u64> are Clone, we reconstruct via new_standard
        // then swap pos -- but that would corrupt history. Instead, we rely on
        // the fact that greedy only calls game.apply() for scoring and the
        // terminal check is self-contained per-clone. We need a proper Game.
        //
        // The only safe public path: use a helper that mirrors game_from_pos
        // from the game tests. Since Game has no public from_pos constructor we
        // use new_standard and then manipulate the public pos field in-place.
        // This is safe for tests that construct clean positions.
        // SAFETY NOTE: We rely on kairnz_core::game::Game having public `pos`
        // field and Clone derive added in this task. We construct via new_standard
        // to get a valid shell, then overwrite pos. History is seeded manually.
        // In tests this approach is acceptable per the existing test pattern in game.rs.
        let mut game = Game::new_standard(RuleConfig::default());
        game.pos = pos;
        // History mismatch won't affect greedy correctness because greedy only
        // reads game.pos and clones from it.
        game
    }

    fn minimal_pos(to_move: Player, ap: u8) -> Position {
        Position {
            board: [None; NUM_SQUARES],
            reserves: [0, 0],
            to_move,
            turn: TurnState {
                ap_remaining: ap,
                capture_locked: BitBoard81::default(),
                keystone_moved: BitBoard81::default(),
                enemy_checked_at_start: BitBoard81::default(),
            },
            config: RuleConfig::default(),
            zobrist: 0,
            ply: 0,
        }
    }

    fn place(pos: &mut Position, file: u8, rank: u8, piece: Piece) {
        let s = sq(file, rank);
        pos.board[s.0 as usize] = Some(piece);
    }

    /// Greedy should capture an undefended enemy Stone rather than make a quiet move.
    #[test]
    fn greedy_prefers_a_free_capture_over_a_quiet_move() {
        // Board (P1 to move, 1 AP):
        //   P1 Stone h2 at (4,4)  -- can step to all 8 neighbours
        //   P2 Stone h1 at (5,4)  -- adjacent enemy (capture)
        //   P1 Keystone at (0,0)  -- so the game is not over on keystone count
        //   P2 Keystone at (8,8)
        let mut pos = minimal_pos(Player::P1, 1);
        place(&mut pos, 4, 4, Piece::new(Player::P1, PieceKind::Stone, 2));
        place(&mut pos, 5, 4, Piece::new(Player::P2, PieceKind::Stone, 1));
        place(&mut pos, 0, 0, Piece::new(Player::P1, PieceKind::Keystone, 1));
        place(&mut pos, 8, 8, Piece::new(Player::P2, PieceKind::Keystone, 1));
        pos.recompute_zobrist();

        let game = game_from_pos(pos);
        let action = GreedyPolicy::seeded(0).choose(&game);

        assert_eq!(
            action,
            Some(Action::Move { from: sq(4, 4), to: sq(5, 4) }),
            "greedy must capture the adjacent enemy Stone"
        );
    }

    /// When both an enemy Stone capture and a Keystone capture are available,
    /// Greedy must prefer the Keystone because KEYSTONE_VALUE dominates.
    #[test]
    fn greedy_values_keystone_capture_highest() {
        // Board (P1 to move, 1 AP):
        //   P1 Stone h2 at (4,4)  -- can reach (5,4) and (4,5)
        //   P2 Stone h1 at (5,4)  -- enemy Stone capture
        //   P2 Keystone h1 at (4,5) -- enemy Keystone capture (dominant)
        //   P1 Keystone at (0,0)
        //   P2 has one more Keystone at (8,8) so the game doesn't end mid-eval
        let mut pos = minimal_pos(Player::P1, 1);
        place(&mut pos, 4, 4, Piece::new(Player::P1, PieceKind::Stone, 2));
        place(&mut pos, 5, 4, Piece::new(Player::P2, PieceKind::Stone, 1));
        place(&mut pos, 4, 5, Piece::new(Player::P2, PieceKind::Keystone, 1));
        place(&mut pos, 0, 0, Piece::new(Player::P1, PieceKind::Keystone, 1));
        place(&mut pos, 8, 8, Piece::new(Player::P2, PieceKind::Keystone, 1));
        pos.recompute_zobrist();

        let game = game_from_pos(pos);
        let action = GreedyPolicy::seeded(0).choose(&game);

        assert_eq!(
            action,
            Some(Action::Move { from: sq(4, 4), to: sq(4, 5) }),
            "greedy must capture the Keystone over the Stone"
        );
    }

    /// Two `GreedyPolicy` instances with the same seed must choose the same action.
    #[test]
    fn greedy_is_deterministic_for_a_seed() {
        let game = Game::new_standard(RuleConfig::default());
        let a = GreedyPolicy::seeded(99).choose(&game);
        let b = GreedyPolicy::seeded(99).choose(&game);
        assert_eq!(a, b, "same seed must produce the same action from the same position");
    }

    /// Prove that the RNG genuinely drives tie-break selection among equal-scoring moves.
    /// The standard opening position is symmetric, so many opening moves score equally
    /// under evaluate, causing tie-breaking to fire. If tie-breaking regressed to
    /// "always pick the first max-scoring action", every seed would yield the same action.
    /// This test verifies that different seeds choose different actions, proving the RNG
    /// is actually used for tie-breaking.
    #[test]
    fn greedy_tiebreak_varies_with_seed() {
        let game = Game::new_standard(RuleConfig::default());
        let mut chosen_actions: Vec<Action> = Vec::new();

        // Try many different seeds and collect the chosen actions.
        for seed in 0..30 {
            if let Some(action) = GreedyPolicy::seeded(seed).choose(&game) {
                chosen_actions.push(action);
            }
        }

        // Count distinct actions by checking if we see more than one unique action.
        // If tie-breaking were broken (always picking first), all seeds would choose
        // the same action. Assert that we see more than one distinct action,
        // proving the RNG selects among ties.
        let has_distinct = chosen_actions.windows(2).any(|w| w[0] != w[1]);
        assert!(
            has_distinct,
            "RNG must break ties; expected distinct actions across different seeds"
        );
    }

    /// The name method returns the expected identifier.
    #[test]
    fn greedy_name_is_greedy() {
        assert_eq!(GreedyPolicy::seeded(0).name(), "greedy");
    }
}
