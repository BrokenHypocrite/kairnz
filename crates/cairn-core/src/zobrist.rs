//! Zobrist hashing for position identity and repetition detection.
//!
//! # What is hashed
//! We key on: each occupied square's (index, owner, kind, height), the side to
//! move, and each player's reserve count. Transient per-turn state
//! (ap_remaining, capture_locked, keystone_moved, enemy_checked_at_start) is
//! intentionally excluded. Repetition is only ever checked at turn boundaries
//! where that state is fully reset, so including it would make logically equal
//! positions hash differently.

use crate::piece::{PieceKind, Player};
use crate::position::Position;
use crate::square::NUM_SQUARES;

// ---------------------------------------------------------------------------
// Sizing constants
// ---------------------------------------------------------------------------

/// Number of players.
const NUM_PLAYERS: usize = 2;

/// Number of piece kinds (Stone, Keystone).
const NUM_PIECE_KINDS: usize = 2;

/// Maximum meaningful piece stack height. Heights above this are unusual but
/// we clamp the index rather than panicking.
const MAX_HEIGHT: usize = 40;

/// Maximum reserve count per player. Reserve counts above this are clamped.
const MAX_RESERVE: usize = 40;

/// Number of reserve keys per player (one per count 0..=MAX_RESERVE).
const RESERVE_TABLE_SIZE: usize = MAX_RESERVE + 1;

// ---------------------------------------------------------------------------
// Seed constants (arbitrary fixed values; must never change across versions)
// ---------------------------------------------------------------------------

const SEED_PIECE_BASE: u64 = 0x9e37_79b9_7f4a_7c15;
const SEED_SIDE_TO_MOVE: u64 = 0x6c62_272e_07bb_0142;
const SEED_RESERVE_P1: u64 = 0xbf58_476d_1ce4_e5b9;
const SEED_RESERVE_P2: u64 = 0x94d0_49bb_1331_11eb;

// ---------------------------------------------------------------------------
// splitmix64 -- deterministic, non-crypto mixer used only at table init time
// ---------------------------------------------------------------------------

/// One step of splitmix64: mixes `x` and returns the next state and output.
const fn splitmix64(x: u64) -> (u64, u64) {
    let x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let z = x;
    let z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    let z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    (x, z ^ (z >> 31))
}

/// Hashes a u64 value into a key by mixing with a per-slot seed.
const fn mix(seed: u64, slot: u64) -> u64 {
    let (_, out) = splitmix64(seed.wrapping_add(slot.wrapping_mul(0x517c_c1b7_2722_0a95)));
    out
}

// ---------------------------------------------------------------------------
// Key tables (generated at compile time via const fn)
// ---------------------------------------------------------------------------

/// Keys for piece placement: indexed [square][player][kind][height_clamped].
///
/// Dimensions: NUM_SQUARES x NUM_PLAYERS x NUM_PIECE_KINDS x (MAX_HEIGHT + 1).
struct PieceKeys([[[[u64; MAX_HEIGHT + 1]; NUM_PIECE_KINDS]; NUM_PLAYERS]; NUM_SQUARES]);

/// Keys for side-to-move: index 0 = P1, 1 = P2.
struct SideKeys([u64; NUM_PLAYERS]);

/// Keys for reserve counts: indexed [player][count_clamped].
struct ReserveKeys([[u64; RESERVE_TABLE_SIZE]; NUM_PLAYERS]);

/// The complete Zobrist key table.
pub struct ZobristTable {
    piece: PieceKeys,
    side: SideKeys,
    reserve: ReserveKeys,
}

impl ZobristTable {
    /// Constructs the table deterministically from fixed seeds.
    ///
    /// Each (square, player, kind, height) triple gets a unique key derived
    /// from its position in the table via splitmix64 mixing. No runtime RNG is
    /// used: the output is identical across runs and across `Position` instances.
    const fn new() -> Self {
        // Build piece keys.
        // We need a mutable array; Rust const fn allows this with array literals.
        let mut piece = [[[[0u64; MAX_HEIGHT + 1]; NUM_PIECE_KINDS]; NUM_PLAYERS]; NUM_SQUARES];

        let mut sq = 0usize;
        while sq < NUM_SQUARES {
            let mut pl = 0usize;
            while pl < NUM_PLAYERS {
                let mut ki = 0usize;
                while ki < NUM_PIECE_KINDS {
                    let mut ht = 0usize;
                    while ht <= MAX_HEIGHT {
                        // Combine all four axes into a unique slot index.
                        let slot = (sq as u64)
                            .wrapping_mul(1_000_003)
                            .wrapping_add((pl as u64).wrapping_mul(1_000_033))
                            .wrapping_add((ki as u64).wrapping_mul(1_000_037))
                            .wrapping_add(ht as u64);
                        piece[sq][pl][ki][ht] = mix(SEED_PIECE_BASE, slot);
                        ht += 1;
                    }
                    ki += 1;
                }
                pl += 1;
            }
            sq += 1;
        }

        // Build side-to-move keys.
        let side = [
            mix(SEED_SIDE_TO_MOVE, 0),
            mix(SEED_SIDE_TO_MOVE, 1),
        ];

        // Build reserve keys.
        let mut reserve = [[0u64; RESERVE_TABLE_SIZE]; NUM_PLAYERS];
        let mut cnt = 0usize;
        while cnt < RESERVE_TABLE_SIZE {
            reserve[0][cnt] = mix(SEED_RESERVE_P1, cnt as u64);
            reserve[1][cnt] = mix(SEED_RESERVE_P2, cnt as u64);
            cnt += 1;
        }

        ZobristTable {
            piece: PieceKeys(piece),
            side: SideKeys(side),
            reserve: ReserveKeys(reserve),
        }
    }
}

/// The global key table, constructed once at program start from fixed seeds.
static TABLE: std::sync::OnceLock<ZobristTable> = std::sync::OnceLock::new();

/// Returns a reference to the shared key table.
fn table() -> &'static ZobristTable {
    TABLE.get_or_init(ZobristTable::new)
}

// ---------------------------------------------------------------------------
// Public interface
// ---------------------------------------------------------------------------

/// Returns the Zobrist hash for `pos`.
///
/// Hashes the board, side to move, and per-player reserve counts. Transient
/// turn state is excluded; see module-level comment for rationale.
pub fn zobrist_full(pos: &Position) -> u64 {
    let t = table();
    let mut hash = 0u64;

    // XOR in a key for each occupied square.
    for sq in 0..NUM_SQUARES {
        if let Some(pc) = pos.board[sq] {
            let pl = pc.owner.index();
            let ki = kind_index(pc.kind);
            let ht = (pc.height as usize).min(MAX_HEIGHT);
            hash ^= t.piece.0[sq][pl][ki][ht];
        }
    }

    // XOR in the side-to-move key.
    hash ^= t.side.0[side_index(pos.to_move)];

    // XOR in reserve keys for both players.
    for pl in 0..NUM_PLAYERS {
        let count = (pos.reserves[pl] as usize).min(MAX_RESERVE);
        hash ^= t.reserve.0[pl][count];
    }

    hash
}

// ---------------------------------------------------------------------------
// Index helpers
// ---------------------------------------------------------------------------

/// Maps a `PieceKind` to a table index.
fn kind_index(k: PieceKind) -> usize {
    match k {
        PieceKind::Stone => 0,
        PieceKind::Keystone => 1,
    }
}

/// Maps a `Player` to a side-to-move table index.
fn side_index(p: Player) -> usize {
    match p {
        Player::P1 => 0,
        Player::P2 => 1,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::RuleConfig;
    use crate::piece::{Piece, PieceKind, Player};
    use crate::position::Position;

    #[test]
    fn table_is_nonzero() {
        let t = table();
        // Spot-check: key at square 0 / P1 / Stone / height 1 must be nonzero.
        assert_ne!(t.piece.0[0][0][0][1], 0);
    }

    #[test]
    fn piece_keys_are_distinct_across_squares() {
        let t = table();
        // Keys for the same piece at different squares must differ.
        assert_ne!(t.piece.0[0][0][0][1], t.piece.0[1][0][0][1]);
    }

    #[test]
    fn side_keys_differ() {
        let t = table();
        assert_ne!(t.side.0[0], t.side.0[1]);
    }

    #[test]
    fn reserve_keys_differ_by_count() {
        let t = table();
        assert_ne!(t.reserve.0[0][0], t.reserve.0[0][1]);
    }

    #[test]
    fn flipping_reserve_changes_hash() {
        let mut pos = Position::new_standard(RuleConfig::default());
        let h0 = zobrist_full(&pos);
        pos.reserves[0] += 1;
        assert_ne!(zobrist_full(&pos), h0);
    }

    #[test]
    fn adding_piece_changes_hash() {
        let mut pos = Position::new_standard(RuleConfig::default());
        let h0 = zobrist_full(&pos);
        // Place a stone on an empty square (rank 4, file 4 is guaranteed empty).
        let sq = crate::square::Sq::new(4, 4).unwrap();
        pos.board[sq.0 as usize] = Some(Piece::new(Player::P1, PieceKind::Stone, 1));
        assert_ne!(zobrist_full(&pos), h0);
    }
}
