use crate::piece::Player;

/// The reason a game ended in a draw.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum DrawReason {
    /// The game reached the configured ply limit without a decisive result.
    MaxPlies,
    /// The same position occurred enough times to trigger the repetition rule.
    Repetition,
}

/// The result of a finished game.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum GameResult {
    /// One player won the game.
    Win(Player),
    /// The game ended in a draw for the given reason.
    Draw(DrawReason),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn game_result_win_roundtrips_player() {
        let r = GameResult::Win(Player::P1);
        assert_eq!(r, GameResult::Win(Player::P1));
        assert_ne!(r, GameResult::Win(Player::P2));
    }

    #[test]
    fn game_result_draw_variants_differ() {
        assert_ne!(
            GameResult::Draw(DrawReason::MaxPlies),
            GameResult::Draw(DrawReason::Repetition)
        );
    }

    #[test]
    fn game_result_roundtrips_through_json() {
        for r in [
            GameResult::Win(Player::P1),
            GameResult::Win(Player::P2),
            GameResult::Draw(DrawReason::MaxPlies),
            GameResult::Draw(DrawReason::Repetition),
        ] {
            let s = serde_json::to_string(&r).unwrap();
            let back: GameResult = serde_json::from_str(&s).unwrap();
            assert_eq!(back, r);
        }
    }
}
