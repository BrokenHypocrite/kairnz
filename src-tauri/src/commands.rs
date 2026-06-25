use kairnz_core::actions::Action;
use kairnz_core::config::RuleConfig;
use kairnz_core::square::Sq;
use tauri::State;

use crate::state::{GameId, GameStore};
use crate::view::{AiMoveResult, ApplyResult, GameView};

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

/// Asks the AI to choose and apply a move, returning the chosen action and
/// updated state (so the UI can record the move in its history).
///
/// `model` is the filesystem path to the ONNX model file; `simulations` controls
/// the MCTS search budget. The AI engine is lazily loaded and cached across calls.
#[tauri::command]
pub fn ai_move(
    id: GameId,
    model: String,
    simulations: u32,
    store: State<GameStore>,
    ai: State<crate::ai::AiEngine>,
) -> Result<AiMoveResult, String> {
    // Clone the game (brief lock), search lock-free, then apply.
    let game = store.clone_game(id)?;
    let action = ai.choose(&game, std::path::Path::new(&model), simulations)?;
    let apply = store.apply_action(id, action)?;
    Ok(AiMoveResult { action, apply })
}

/// Returns the geometric move targets for the piece at `from`, ignoring AP/turn rules.
///
/// Returns an empty list if the square is empty.
#[tauri::command]
pub fn piece_moves(
    id: GameId,
    from: Sq,
    store: State<GameStore>,
) -> Result<Vec<Sq>, String> {
    store.piece_moves(id, from)
}
