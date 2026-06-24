use cairn_core::actions::Action;
use cairn_core::config::RuleConfig;
use cairn_core::square::Sq;
use tauri::State;

use crate::state::{GameId, GameStore};
use crate::view::{ApplyResult, GameView};

/// Creates a new game with the given rule configuration.
///
/// Returns the game ID and initial view as a tuple encoded as JSON.
#[tauri::command]
pub fn new_game(
    config: RuleConfig,
    store: State<GameStore>,
) -> (GameId, GameView) {
    store.new_game(config)
}

/// Returns the current view for an existing game.
#[tauri::command]
pub fn get_view(id: GameId, store: State<GameStore>) -> Result<GameView, String> {
    store.get_view(id)
}

/// Returns the legal actions for a game.
///
/// When `from` is provided, only `Move` actions originating at that square are
/// returned. When omitted, all legal actions are returned.
#[tauri::command]
pub fn legal_actions(
    id: GameId,
    from: Option<Sq>,
    store: State<GameStore>,
) -> Result<Vec<Action>, String> {
    store.legal_actions(id, from)
}

/// Applies an action to the game and returns the updated state.
#[tauri::command]
pub fn apply_action(
    id: GameId,
    action: Action,
    store: State<GameStore>,
) -> Result<ApplyResult, String> {
    store.apply_action(id, action)
}

/// Undoes the last action and returns the restored view.
#[tauri::command]
pub fn undo(id: GameId, store: State<GameStore>) -> Result<GameView, String> {
    store.undo(id)
}
