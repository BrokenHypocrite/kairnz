pub mod actions;
pub mod apply;
pub mod check;
pub mod config;
pub mod movement;
pub mod outcome;
pub mod piece;
pub mod position;
pub mod square;
pub mod zobrist;

#[cfg(test)]
mod smoke {
    #[test]
    fn workspace_builds() {
        assert_eq!(2 + 2, 4);
    }
}
