pub mod config;
pub mod piece;
pub mod position;
pub mod square;

#[cfg(test)]
mod smoke {
    #[test]
    fn workspace_builds() {
        assert_eq!(2 + 2, 4);
    }
}
