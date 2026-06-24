use crate::config::SpireMode;
use crate::piece::{Piece, PieceKind};
use crate::position::Position;
use crate::square::{Sq, BOARD_SIZE};

// Direction tables as (file_delta, rank_delta) pairs.
const ORTHO: [(i8, i8); 4] = [(0, 1), (0, -1), (1, 0), (-1, 0)];
const DIAG: [(i8, i8); 4] = [(1, 1), (1, -1), (-1, 1), (-1, -1)];
const ALL8: [(i8, i8); 8] = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)];

/// Applies a single (df, dr) step from a square, returning `None` if out of bounds.
fn step_sq(sq: Sq, df: i8, dr: i8) -> Option<Sq> {
    let f = (sq.file() as i8) + df;
    let r = (sq.rank() as i8) + dr;
    if f < 0 || r < 0 || f >= BOARD_SIZE as i8 || r >= BOARD_SIZE as i8 {
        return None;
    }
    Sq::new(f as u8, r as u8)
}

/// Collects one step in direction (df, dr) as a target if empty or enemy-occupied.
fn collect_step(pos: &Position, from: Sq, df: i8, dr: i8, mover: Piece, targets: &mut Vec<Sq>) {
    if let Some(dest) = step_sq(from, df, dr) {
        match pos.piece_at(dest) {
            None => targets.push(dest),
            Some(occupant) => {
                if occupant.owner != mover.owner {
                    targets.push(dest);
                }
            }
        }
    }
}

/// Slides in direction (df, dr) until blocked, collecting empty squares and stopping
/// at the first piece (including it as a target if it is an enemy).
fn collect_slide(pos: &Position, from: Sq, df: i8, dr: i8, mover: Piece, targets: &mut Vec<Sq>) {
    let mut cur = from;
    loop {
        match step_sq(cur, df, dr) {
            None => break,
            Some(dest) => {
                match pos.piece_at(dest) {
                    None => {
                        targets.push(dest);
                        cur = dest;
                    }
                    Some(occupant) => {
                        if occupant.owner != mover.owner {
                            targets.push(dest);
                        }
                        break;
                    }
                }
            }
        }
    }
}

/// Returns every square the piece at `from` can move to under pure geometry rules.
///
/// A square is a target if it is empty or holds an enemy piece; never if friendly.
/// This function does not consult AP, capture-lock, or keystone-single-move toggles.
pub fn move_targets(pos: &Position, from: Sq) -> Vec<Sq> {
    let piece: Piece = match pos.piece_at(from) {
        Some(p) => p,
        None => return Vec::new(),
    };

    let mut targets: Vec<Sq> = Vec::new();

    match piece.kind {
        PieceKind::Keystone => {
            for &(df, dr) in &ALL8 {
                collect_step(pos, from, df, dr, piece, &mut targets);
            }
        }
        PieceKind::Stone => match piece.height {
            1 => {
                for &(df, dr) in &ORTHO {
                    collect_step(pos, from, df, dr, piece, &mut targets);
                }
            }
            2 => {
                for &(df, dr) in &ALL8 {
                    collect_step(pos, from, df, dr, piece, &mut targets);
                }
            }
            3 => {
                // Spire
                match pos.config.spire {
                    SpireMode::Dragon => {
                        for &(df, dr) in &ORTHO {
                            collect_slide(pos, from, df, dr, piece, &mut targets);
                        }
                        for &(df, dr) in &DIAG {
                            collect_step(pos, from, df, dr, piece, &mut targets);
                        }
                    }
                    SpireMode::Queen => {
                        for &(df, dr) in &ALL8 {
                            collect_slide(pos, from, df, dr, piece, &mut targets);
                        }
                    }
                }
            }
            _ => {}
        },
    }

    targets
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{RuleConfig, SpireMode};
    use crate::piece::{Piece, PieceKind, Player};
    use crate::position::{Position, TurnState};
    use crate::square::{BitBoard81, Sq, NUM_SQUARES};

    fn empty_pos(config: RuleConfig) -> Position {
        Position {
            board: [None; NUM_SQUARES],
            reserves: [0, 0],
            to_move: Player::P1,
            turn: TurnState {
                ap_remaining: 2,
                capture_locked: BitBoard81::default(),
                keystone_moved: BitBoard81::default(),
                enemy_checked_at_start: [false, false],
            },
            config,
            zobrist: 0,
            ply: 0,
        }
    }

    fn place(pos: &mut Position, file: u8, rank: u8, piece: Piece) {
        let sq = Sq::new(file, rank).unwrap();
        pos.board[sq.0 as usize] = Some(piece);
    }

    fn sq(file: u8, rank: u8) -> Sq {
        Sq::new(file, rank).unwrap()
    }

    fn sorted_targets(pos: &Position, from: Sq) -> Vec<u8> {
        let mut v: Vec<u8> = move_targets(pos, from).into_iter().map(|s| s.0).collect();
        v.sort_unstable();
        v
    }

    #[test]
    fn stone_h1_steps_one_orthogonally() {
        let mut pos = empty_pos(RuleConfig::default());
        let from = sq(4, 4);
        place(&mut pos, 4, 4, Piece::new(Player::P1, PieceKind::Stone, 1));

        let targets = sorted_targets(&pos, from);
        let mut expected: Vec<u8> = [sq(4, 5), sq(4, 3), sq(5, 4), sq(3, 4)]
            .iter()
            .map(|s| s.0)
            .collect();
        expected.sort_unstable();

        assert_eq!(targets, expected);
    }

    #[test]
    fn stone_h1_no_wraparound_at_left_edge() {
        let mut pos = empty_pos(RuleConfig::default());
        let from = sq(0, 4);
        place(&mut pos, 0, 4, Piece::new(Player::P1, PieceKind::Stone, 1));

        let targets = move_targets(&pos, from);
        let target_squares: Vec<(u8, u8)> = targets.iter().map(|s| (s.file(), s.rank())).collect();
        assert!(!target_squares.contains(&(8, 3)), "wrapped to (8,3) -- index arithmetic bug");
        assert!(!target_squares.contains(&(8, 4)), "wrapped to (8,4) -- index arithmetic bug");
        assert_eq!(targets.len(), 3, "file 0 stone: north, south, east only");
    }

    #[test]
    fn pillar_h2_steps_one_in_eight_directions() {
        let mut pos = empty_pos(RuleConfig::default());
        let from = sq(4, 4);
        place(&mut pos, 4, 4, Piece::new(Player::P1, PieceKind::Stone, 2));

        let targets = sorted_targets(&pos, from);
        let mut expected: Vec<u8> = [
            sq(4, 5), sq(4, 3), sq(5, 4), sq(3, 4),
            sq(5, 5), sq(5, 3), sq(3, 5), sq(3, 3),
        ]
        .iter()
        .map(|s| s.0)
        .collect();
        expected.sort_unstable();

        assert_eq!(targets, expected);
    }

    #[test]
    fn spire_dragon_slides_orthogonal_and_steps_diagonal() {
        let mut pos = empty_pos(RuleConfig::default());
        let from = sq(4, 4);
        place(&mut pos, 4, 4, Piece::new(Player::P1, PieceKind::Stone, 3));

        let targets = move_targets(&pos, from);

        // All orthogonal squares along each ray must be reachable.
        for f in 0..9u8 {
            if f == 4 { continue; }
            assert!(targets.contains(&sq(f, 4)), "Dragon missing ortho ({f},4)");
        }
        for r in 0..9u8 {
            if r == 4 { continue; }
            assert!(targets.contains(&sq(4, r)), "Dragon missing ortho (4,{r})");
        }

        // The 4 adjacent diagonal squares must be reachable (step only, not slide).
        assert!(targets.contains(&sq(5, 5)), "missing diag (5,5)");
        assert!(targets.contains(&sq(5, 3)), "missing diag (5,3)");
        assert!(targets.contains(&sq(3, 5)), "missing diag (3,5)");
        assert!(targets.contains(&sq(3, 3)), "missing diag (3,3)");

        // Non-adjacent diagonals must NOT be reachable for Dragon.
        assert!(!targets.contains(&sq(6, 6)), "Dragon must not slide diagonally to (6,6)");
        assert!(!targets.contains(&sq(2, 2)), "Dragon must not slide diagonally to (2,2)");

        // From (4,4): 4 ortho rays * 4 squares each = 16, plus 4 diagonal steps = 20.
        assert_eq!(targets.len(), 20);
    }

    #[test]
    fn spire_queen_slides_all_eight() {
        let mut cfg = RuleConfig::default();
        cfg.spire = SpireMode::Queen;
        let mut pos = empty_pos(cfg);
        let from = sq(4, 4);
        place(&mut pos, 4, 4, Piece::new(Player::P1, PieceKind::Stone, 3));

        let targets = move_targets(&pos, from);

        // All orthogonal squares.
        for f in 0..9u8 {
            if f == 4 { continue; }
            assert!(targets.contains(&sq(f, 4)), "Queen missing ortho ({f},4)");
        }
        for r in 0..9u8 {
            if r == 4 { continue; }
            assert!(targets.contains(&sq(4, r)), "Queen missing ortho (4,{r})");
        }

        // All diagonal squares (full rays), 4 steps in each diagonal direction from center.
        for d in 1..5u8 {
            assert!(targets.contains(&sq(4 + d, 4 + d)), "Queen missing NE diag +{d}");
            assert!(targets.contains(&sq(4 + d, 4 - d)), "Queen missing SE diag +{d}/-{d}");
            assert!(targets.contains(&sq(4 - d, 4 + d)), "Queen missing NW diag -{d}/+{d}");
            assert!(targets.contains(&sq(4 - d, 4 - d)), "Queen missing SW diag -{d}");
        }

        // ortho 16 + diag (4+4+4+4) = 32.
        assert_eq!(targets.len(), 32);
    }

    #[test]
    fn keystone_steps_one_in_eight() {
        let mut pos = empty_pos(RuleConfig::default());
        let from = sq(4, 4);
        place(&mut pos, 4, 4, Piece::new(Player::P1, PieceKind::Keystone, 1));

        let targets = sorted_targets(&pos, from);
        let mut expected: Vec<u8> = [
            sq(4, 5), sq(4, 3), sq(5, 4), sq(3, 4),
            sq(5, 5), sq(5, 3), sq(3, 5), sq(3, 3),
        ]
        .iter()
        .map(|s| s.0)
        .collect();
        expected.sort_unstable();

        assert_eq!(targets, expected);
    }

    #[test]
    fn slide_stops_at_first_piece_and_may_capture_enemy() {
        let mut cfg = RuleConfig::default();
        cfg.spire = SpireMode::Queen;
        let mut pos = empty_pos(cfg);
        let from = sq(4, 4);
        place(&mut pos, 4, 4, Piece::new(Player::P1, PieceKind::Stone, 3));

        // Friendly blocker on north ray at rank 6.
        place(&mut pos, 4, 6, Piece::new(Player::P1, PieceKind::Stone, 1));
        // Enemy blocker on east ray at file 7.
        place(&mut pos, 7, 4, Piece::new(Player::P2, PieceKind::Stone, 1));

        let targets = move_targets(&pos, from);

        // North ray: rank 5 reachable, rank 6 friendly (not a target), 7 and 8 blocked.
        assert!(targets.contains(&sq(4, 5)), "rank 5 must be reachable");
        assert!(!targets.contains(&sq(4, 6)), "friendly at rank 6 must not be a target");
        assert!(!targets.contains(&sq(4, 7)), "rank 7 must not be reachable past friendly");
        assert!(!targets.contains(&sq(4, 8)), "rank 8 must not be reachable past friendly");

        // East ray: files 5 and 6 empty, file 7 enemy (capture), file 8 unreachable.
        assert!(targets.contains(&sq(5, 4)), "file 5 must be reachable");
        assert!(targets.contains(&sq(6, 4)), "file 6 must be reachable");
        assert!(targets.contains(&sq(7, 4)), "enemy at file 7 must be a capture target");
        assert!(!targets.contains(&sq(8, 4)), "file 8 past enemy must not be reachable");
    }

    #[test]
    fn spire_dragon_blocked_rays() {
        let mut pos = empty_pos(RuleConfig::default());
        let from = sq(4, 4);
        place(&mut pos, 4, 4, Piece::new(Player::P1, PieceKind::Stone, 3));

        // Friendly blocker on north ray (orthogonal) at rank 6.
        place(&mut pos, 4, 6, Piece::new(Player::P1, PieceKind::Stone, 1));
        // Enemy blocker on east ray (orthogonal) at file 7.
        place(&mut pos, 7, 4, Piece::new(Player::P2, PieceKind::Stone, 1));

        let targets = move_targets(&pos, from);

        // North ray: rank 5 reachable, rank 6 friendly (not a target), 7 and 8 blocked.
        assert!(targets.contains(&sq(4, 5)), "north ray: rank 5 must be reachable");
        assert!(!targets.contains(&sq(4, 6)), "north ray: friendly at rank 6 must not be a target");
        assert!(!targets.contains(&sq(4, 7)), "north ray: rank 7 must not be reachable past friendly");

        // East ray: files 5 and 6 empty, file 7 enemy (capture), file 8 unreachable.
        assert!(targets.contains(&sq(5, 4)), "east ray: file 5 must be reachable");
        assert!(targets.contains(&sq(6, 4)), "east ray: file 6 must be reachable");
        assert!(targets.contains(&sq(7, 4)), "east ray: enemy at file 7 must be a capture target");
        assert!(!targets.contains(&sq(8, 4)), "east ray: file 8 past enemy must not be reachable");
    }

    #[test]
    fn never_moves_onto_friendly() {
        let mut pos = empty_pos(RuleConfig::default());
        let from = sq(4, 4);
        place(&mut pos, 4, 4, Piece::new(Player::P1, PieceKind::Stone, 2));

        // Surround with friendly pieces in all 8 directions.
        for &(df, dr) in &ALL8 {
            let f = (4i8 + df) as u8;
            let r = (4i8 + dr) as u8;
            place(&mut pos, f, r, Piece::new(Player::P1, PieceKind::Stone, 1));
        }

        let targets = move_targets(&pos, from);
        assert!(targets.is_empty(), "no targets when all neighbors are friendly");
    }

    #[test]
    fn spire_dragon_at_board_corner() {
        let mut pos = empty_pos(RuleConfig::default());
        let from = sq(0, 0);
        place(&mut pos, 0, 0, Piece::new(Player::P1, PieceKind::Stone, 3));

        let targets = move_targets(&pos, from);

        // Orthogonal slides terminate at board edge: (8,0) and (0,8).
        assert!(targets.contains(&sq(8, 0)), "corner dragon must slide east to file 8");
        assert!(targets.contains(&sq(0, 8)), "corner dragon must slide north to rank 8");

        // Check for no wraparound: wrapped squares don't exist on a corner.
        let target_squares: Vec<(u8, u8)> = targets.iter().map(|s| (s.file(), s.rank())).collect();
        // (0,0) is the source, so no negative or wrapped indices should appear.
        for (f, r) in target_squares {
            assert!(f <= 8 && r <= 8, "target ({f},{r}) is out of bounds");
        }
    }

    #[test]
    fn empty_square_returns_no_targets() {
        let pos = empty_pos(RuleConfig::default());
        let from = sq(4, 4);
        assert!(move_targets(&pos, from).is_empty());
    }
}
