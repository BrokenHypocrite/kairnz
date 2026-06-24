/// The two players in a game.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum Player {
    P1,
    P2,
}

impl Player {
    /// Returns the opposing player.
    pub fn opponent(self) -> Player {
        match self {
            Player::P1 => Player::P2,
            Player::P2 => Player::P1,
        }
    }

    /// Returns a zero-based index for this player (P1 = 0, P2 = 1).
    pub fn index(self) -> usize {
        match self {
            Player::P1 => 0,
            Player::P2 => 1,
        }
    }
}

/// The kind of a piece on the board.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PieceKind {
    Stone,
    Keystone,
}

/// A piece owned by a player, with a kind and stack height.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Piece {
    pub owner: Player,
    pub kind: PieceKind,
    pub height: u8,
}

impl Piece {
    /// Constructs a new piece.
    pub fn new(owner: Player, kind: PieceKind, height: u8) -> Piece {
        Piece { owner, kind, height }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn player_opponent_is_involutive() {
        assert_eq!(Player::P1.opponent().opponent(), Player::P1);
    }

    #[test]
    fn player_p2_opponent_is_involutive() {
        assert_eq!(Player::P2.opponent().opponent(), Player::P2);
    }

    #[test]
    fn player_index_values() {
        assert_eq!(Player::P1.index(), 0);
        assert_eq!(Player::P2.index(), 1);
    }

    #[test]
    fn player_opponents_differ() {
        assert_ne!(Player::P1.opponent(), Player::P1);
        assert_ne!(Player::P2.opponent(), Player::P2);
    }

    #[test]
    fn piece_fields_accessible() {
        let p = Piece::new(Player::P1, PieceKind::Keystone, 3);
        assert_eq!(p.owner, Player::P1);
        assert_eq!(p.kind, PieceKind::Keystone);
        assert_eq!(p.height, 3);
    }
}
