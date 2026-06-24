use cairn_core::apply::CapturedInfo;
use cairn_core::game::Game;
use cairn_core::outcome::GameResult;
use cairn_core::piece::{PieceKind, Player};
use cairn_core::square::Sq;
use serde::Serialize;

/// A piece as seen by the UI.
#[derive(Serialize, Clone, Debug, PartialEq)]
pub struct PieceView {
    /// The player who owns this piece.
    pub owner: Player,
    /// The kind of this piece.
    pub kind: PieceKind,
    /// The stack height of this piece.
    pub height: u8,
}

/// A snapshot of game state sent to the UI.
///
/// `board` has exactly 81 entries in square-index order (index = rank*9 + file).
#[derive(Serialize, Clone, Debug, PartialEq)]
pub struct GameView {
    /// Board squares, one entry per square in index order; `None` means empty.
    pub board: Vec<Option<PieceView>>,
    /// Reserve token counts for [P1, P2].
    pub reserves: [u8; 2],
    /// The player whose turn it is.
    pub to_move: Player,
    /// Action points remaining this turn.
    pub ap_remaining: u8,
    /// Terminal result, or `None` while the game is still in progress.
    pub result: Option<GameResult>,
    /// Square indices of all Keystones (either player's) currently in check.
    pub checked_keystones: Vec<Sq>,
}

/// The result of applying an action, returned to the UI.
#[derive(Serialize, Clone, Debug)]
pub struct ApplyResult {
    /// Updated game view after the action.
    pub view: GameView,
    /// Whether the turn ended because a new enemy Keystone was placed in check.
    pub turn_ended_on_check: bool,
    /// Info about a piece captured by the action, if any.
    pub last_capture: Option<CapturedInfo>,
    /// Terminal result, or `None` while the game is still in progress.
    pub result: Option<GameResult>,
}

/// Builds a [`GameView`] from the current state of `game`.
pub fn view_of(game: &Game) -> GameView {
    let pos = &game.pos;

    let board: Vec<Option<PieceView>> = pos
        .board
        .iter()
        .map(|cell| {
            cell.map(|pc| PieceView {
                owner: pc.owner,
                kind: pc.kind,
                height: pc.height,
            })
        })
        .collect();

    GameView {
        board,
        reserves: pos.reserves,
        to_move: pos.to_move,
        ap_remaining: pos.turn.ap_remaining,
        result: game.terminal_result(),
        checked_keystones: cairn_core::check::checked_keystone_squares(&game.pos),
    }
}
