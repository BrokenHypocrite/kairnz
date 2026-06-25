// Prevents a console window from appearing on Windows in release builds.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod ai;
mod commands;
mod state;
mod view;

use state::GameStore;

fn main() {
    tauri::Builder::default()
        .manage(GameStore::new())
        .manage(ai::AiEngine::default())
        .invoke_handler(tauri::generate_handler![
            commands::new_game,
            commands::get_view,
            commands::legal_actions,
            commands::apply_action,
            commands::undo,
            commands::piece_moves,
            commands::ai_move,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
