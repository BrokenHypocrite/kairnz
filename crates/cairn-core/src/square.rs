/// Board size constant: 9x9.
pub const BOARD_SIZE: u8 = 9;

/// Total number of squares on the board.
pub const NUM_SQUARES: usize = 81;

/// Mask covering only the low 81 bits of a u128.
const BOARD_MASK: u128 = (1u128 << NUM_SQUARES) - 1;

/// A board square, stored as `file + rank * BOARD_SIZE` in `0..NUM_SQUARES`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Sq(pub u8);

impl Sq {
    /// Constructs a square from file and rank, returning `None` if either is out of `0..BOARD_SIZE`.
    pub fn new(file: u8, rank: u8) -> Option<Sq> {
        if file < BOARD_SIZE && rank < BOARD_SIZE {
            Some(Sq(rank * BOARD_SIZE + file))
        } else {
            None
        }
    }

    /// Constructs a square from a raw board index, returning `None` if it is out of range.
    pub fn from_index(index: u8) -> Option<Sq> {
        if (index as usize) < NUM_SQUARES {
            Some(Sq(index))
        } else {
            None
        }
    }

    /// Returns the file (column) index of this square.
    pub fn file(self) -> u8 {
        self.0 % BOARD_SIZE
    }

    /// Returns the rank (row) index of this square.
    pub fn rank(self) -> u8 {
        self.0 / BOARD_SIZE
    }
}

/// An 81-bit set of squares, backed by a `u128` (only bits 0..81 are valid).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BitBoard81(u128);

impl BitBoard81 {
    /// Sets the bit corresponding to `sq`.
    pub fn set(&mut self, sq: Sq) {
        self.0 = (self.0 | (1u128 << sq.0)) & BOARD_MASK;
    }

    /// Clears the bit corresponding to `sq`.
    pub fn clear(&mut self, sq: Sq) {
        self.0 &= !(1u128 << sq.0) & BOARD_MASK;
    }

    /// Returns `true` if the bit for `sq` is set.
    pub fn contains(self, sq: Sq) -> bool {
        (self.0 >> sq.0) & 1 == 1
    }

    /// Returns `true` if no bits are set.
    pub fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// Iterates over all squares whose bits are set.
    pub fn iter(self) -> BitBoard81Iter {
        BitBoard81Iter(self.0 & BOARD_MASK)
    }
}

/// Iterator over the set squares in a `BitBoard81`.
pub struct BitBoard81Iter(u128);

impl Iterator for BitBoard81Iter {
    type Item = Sq;

    fn next(&mut self) -> Option<Sq> {
        if self.0 == 0 {
            return None;
        }
        let idx = self.0.trailing_zeros() as u8;
        self.0 &= self.0 - 1;
        Some(Sq(idx))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sq_roundtrips_file_rank() {
        let s = Sq::new(3, 1).unwrap();
        assert_eq!((s.file(), s.rank()), (3, 1));
        assert_eq!(s.0, 1 * 9 + 3);
    }

    #[test]
    fn sq_rejects_out_of_range() {
        assert!(Sq::new(9, 0).is_none());
        assert!(Sq::new(0, 9).is_none());
    }

    #[test]
    fn sq_from_index_roundtrips_and_rejects_out_of_range() {
        assert_eq!(Sq::from_index(80), Some(Sq(80)));
        assert!(Sq::from_index(81).is_none());
    }

    #[test]
    fn bitboard_set_contains_clear() {
        let mut b = BitBoard81::default();
        let s = Sq::new(7, 2).unwrap();
        b.set(s);
        assert!(b.contains(s));
        b.clear(s);
        assert!(!b.contains(s));
    }

    #[test]
    fn bitboard_is_empty_initially() {
        assert!(BitBoard81::default().is_empty());
    }

    #[test]
    fn bitboard_iter_yields_set_squares() {
        let mut b = BitBoard81::default();
        let s1 = Sq::new(0, 0).unwrap();
        let s2 = Sq::new(8, 8).unwrap();
        b.set(s1);
        b.set(s2);
        let collected: Vec<Sq> = b.iter().collect();
        assert_eq!(collected.len(), 2);
        assert!(collected.contains(&s1));
        assert!(collected.contains(&s2));
    }

    #[test]
    fn bitboard_does_not_set_bits_above_80() {
        let mut b = BitBoard81::default();
        // Set all 81 squares.
        for rank in 0..9u8 {
            for file in 0..9u8 {
                b.set(Sq::new(file, rank).unwrap());
            }
        }
        // The raw u128 must only have the low 81 bits set.
        assert_eq!(b.0, BOARD_MASK);
    }
}
