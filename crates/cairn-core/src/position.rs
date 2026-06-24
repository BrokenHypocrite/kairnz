use crate::config::RuleConfig;
use crate::piece::{Piece, PieceKind, Player};
use crate::square::{BitBoard81, Sq, NUM_SQUARES};
use crate::zobrist::zobrist_full;

/// File index of the left keystone starting position.
const KEYSTONE_FILE_LEFT: u8 = 3;
/// File index of the right keystone starting position.
const KEYSTONE_FILE_RIGHT: u8 = 7;

/// Rank index where P1's front row of stones starts.
const P1_RANK_FRONT: u8 = 0;
/// Rank index where P1's keystones are placed.
const P1_RANK_KEYSTONE: u8 = 1;
/// Rank index where P1's back row of stones starts.
const P1_RANK_BACK: u8 = 2;

/// Rank index where P2's back row of stones starts (mirrored).
const P2_RANK_BACK: u8 = 6;
/// Rank index where P2's keystones are placed (mirrored).
const P2_RANK_KEYSTONE: u8 = 7;
/// Rank index where P2's front row of stones starts (mirrored).
const P2_RANK_FRONT: u8 = 8;

/// Starting height for all pieces placed at game start.
const STARTING_HEIGHT: u8 = 1;

/// Per-turn state tracking action points and move restrictions.
#[derive(Clone, Debug)]
pub struct TurnState {
    /// Action points remaining this turn.
    pub ap_remaining: u8,
    /// Squares whose top piece is locked from capture this turn.
    pub capture_locked: BitBoard81,
    /// Squares from which a keystone has already moved this turn.
    pub keystone_moved: BitBoard81,
    /// Whether each player's keystone was in check at the start of this turn.
    pub enemy_checked_at_start: [bool; 2],
}

/// The full mutable game state.
#[derive(Clone, Debug)]
pub struct Position {
    /// Pieces occupying each of the 81 board squares.
    pub board: [Option<Piece>; NUM_SQUARES],
    /// Pieces held in reserve for each player (indexed by `Player::index()`).
    pub reserves: [u8; 2],
    /// The player whose turn it is to move.
    pub to_move: Player,
    /// State for the current turn.
    pub turn: TurnState,
    /// Rule configuration for this game.
    pub config: RuleConfig,
    /// Zobrist hash of the current position (0 until Task 5).
    pub zobrist: u64,
    /// Number of half-moves played since the game started.
    pub ply: u32,
}

impl Position {
    /// Constructs the standard §2 opening position.
    pub fn new_standard(config: RuleConfig) -> Position {
        let mut board = [None; NUM_SQUARES];

        // Place a full row of stones for a player at the given rank.
        let place_stone_row = |board: &mut [Option<Piece>; NUM_SQUARES], owner: Player, rank: u8| {
            for file in 0..crate::square::BOARD_SIZE {
                let sq = Sq::new(file, rank).expect("file and rank are in bounds");
                board[sq.0 as usize] = Some(Piece::new(owner, PieceKind::Stone, STARTING_HEIGHT));
            }
        };

        // Place keystones at the two designated files for a player at the given rank.
        let place_keystones = |board: &mut [Option<Piece>; NUM_SQUARES], owner: Player, rank: u8| {
            for &file in &[KEYSTONE_FILE_LEFT, KEYSTONE_FILE_RIGHT] {
                let sq = Sq::new(file, rank).expect("file and rank are in bounds");
                board[sq.0 as usize] = Some(Piece::new(owner, PieceKind::Keystone, STARTING_HEIGHT));
            }
        };

        // P1 setup: stones on ranks 0 and 2, keystones on rank 1.
        place_stone_row(&mut board, Player::P1, P1_RANK_FRONT);
        place_keystones(&mut board, Player::P1, P1_RANK_KEYSTONE);
        place_stone_row(&mut board, Player::P1, P1_RANK_BACK);

        // P2 setup (mirrored): stones on ranks 8 and 6, keystones on rank 7.
        place_stone_row(&mut board, Player::P2, P2_RANK_FRONT);
        place_keystones(&mut board, Player::P2, P2_RANK_KEYSTONE);
        place_stone_row(&mut board, Player::P2, P2_RANK_BACK);

        let ap = config.first_turn_ap;
        let mut pos = Position {
            board,
            reserves: [0, 0],
            to_move: Player::P1,
            turn: TurnState {
                ap_remaining: ap,
                capture_locked: BitBoard81::default(),
                keystone_moved: BitBoard81::default(),
                enemy_checked_at_start: [false, false],
            },
            config,
            zobrist: 0,
            ply: 0,
        };
        pos.zobrist = zobrist_full(&pos);
        pos
    }

    /// Recomputes and stores the Zobrist hash from scratch.
    ///
    /// Call this after any batch of board mutations to resync `self.zobrist`.
    pub fn recompute_zobrist(&mut self) {
        self.zobrist = zobrist_full(self);
    }

    /// Returns the piece at square `s`, if any.
    pub fn piece_at(&self, s: Sq) -> Option<Piece> {
        self.board[s.0 as usize]
    }

    /// Returns an iterator over all squares occupied by keystones belonging to `p`.
    pub fn keystones_of(&self, p: Player) -> impl Iterator<Item = Sq> + '_ {
        (0..NUM_SQUARES).filter_map(move |i| {
            self.board[i].and_then(|pc| {
                if pc.owner == p && pc.kind == PieceKind::Keystone {
                    Sq::from_index(i as u8)
                } else {
                    None
                }
            })
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::RuleConfig;
    use crate::piece::{PieceKind, Player};
    use crate::square::Sq;
    use crate::zobrist::zobrist_full;

    #[test]
    fn standard_setup_has_correct_material() {
        let p = Position::new_standard(RuleConfig::default());
        let count = |owner, kind| {
            (0..81)
                .filter(|&i| p.board[i].map_or(false, |pc| pc.owner == owner && pc.kind == kind))
                .count()
        };
        assert_eq!(count(Player::P1, PieceKind::Stone), 18);
        assert_eq!(count(Player::P1, PieceKind::Keystone), 2);
        assert_eq!(count(Player::P2, PieceKind::Stone), 18);
        assert_eq!(count(Player::P2, PieceKind::Keystone), 2);
    }

    #[test]
    fn keystones_on_files_3_and_7_rank_index_1() {
        let p = Position::new_standard(RuleConfig::default());
        for f in [3u8, 7] {
            let s = Sq::new(f, 1).unwrap();
            assert!(
                matches!(p.piece_at(s), Some(pc) if pc.kind == PieceKind::Keystone && pc.owner == Player::P1)
            );
        }
    }

    #[test]
    fn p2_keystones_on_files_3_and_7_rank_index_7() {
        let p = Position::new_standard(RuleConfig::default());
        for f in [3u8, 7] {
            let s = Sq::new(f, 7).unwrap();
            assert!(matches!(p.piece_at(s), Some(pc) if pc.kind == PieceKind::Keystone && pc.owner == Player::P2));
        }
    }

    #[test]
    fn first_turn_ap_respects_config() {
        let mut cfg = RuleConfig::default();
        cfg.first_turn_ap = 1;
        assert_eq!(Position::new_standard(cfg).turn.ap_remaining, 1);
    }

    #[test]
    fn reserves_start_empty() {
        let p = Position::new_standard(RuleConfig::default());
        assert_eq!(p.reserves, [0, 0]);
    }

    #[test]
    fn identical_positions_hash_equal() {
        let a = Position::new_standard(RuleConfig::default());
        let b = Position::new_standard(RuleConfig::default());
        assert_eq!(zobrist_full(&a), zobrist_full(&b));
    }

    #[test]
    fn changing_side_to_move_changes_hash() {
        let mut a = Position::new_standard(RuleConfig::default());
        let h0 = zobrist_full(&a);
        a.to_move = Player::P2;
        assert_ne!(zobrist_full(&a), h0);
    }
}
