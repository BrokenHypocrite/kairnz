use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use cairn_core::actions::{legal_actions, Action};
use cairn_core::config::RuleConfig;
use cairn_core::game::Game;
use cairn_core::square::Sq;

use crate::view::{view_of, ApplyResult, GameView};

/// Opaque identifier for a game session.
pub type GameId = u64;

/// A single game together with its undo history.
pub struct GameEntry {
    /// The live game state.
    pub game: Game,
    /// Snapshots of the game before each applied action, for undo support.
    pub undo_stack: Vec<Game>,
}

/// Application-wide store of active game sessions.
///
/// All game state is protected by a single `Mutex`; undo history is kept
/// per-entry so undo never crosses game boundaries.
pub struct GameStore {
    games: Mutex<HashMap<GameId, GameEntry>>,
    next_id: AtomicU64,
}

impl GameStore {
    /// Creates an empty store.
    pub fn new() -> Self {
        Self {
            games: Mutex::new(HashMap::new()),
            next_id: AtomicU64::new(1),
        }
    }

    /// Starts a new game with `config` and returns its ID together with the
    /// initial [`GameView`].
    pub fn new_game(&self, config: RuleConfig) -> (GameId, GameView) {
        let game = Game::new_standard(config);
        let view = view_of(&game);
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let entry = GameEntry { game, undo_stack: Vec::new() };

        let mut guard = match self.games.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        guard.insert(id, entry);
        (id, view)
    }

    /// Returns the current [`GameView`] for the given game, or an error string
    /// if `id` is unknown.
    pub fn get_view(&self, id: GameId) -> Result<GameView, String> {
        let guard = self.games.lock().map_err(|e| e.to_string())?;
        let entry = guard.get(&id).ok_or_else(|| format!("unknown game id {id}"))?;
        Ok(view_of(&entry.game))
    }

    /// Returns the legal actions for the given game.
    ///
    /// When `from` is `Some(sq)`, only `Move` actions whose `from` field equals
    /// that square are returned. All other action kinds (Place, Stack) and Moves
    /// from other squares are excluded. When `from` is `None`, all legal actions
    /// are returned.
    pub fn legal_actions(&self, id: GameId, from: Option<Sq>) -> Result<Vec<Action>, String> {
        let guard = self.games.lock().map_err(|e| e.to_string())?;
        let entry = guard.get(&id).ok_or_else(|| format!("unknown game id {id}"))?;
        let all = legal_actions(&entry.game.pos);
        let filtered = match from {
            None => all,
            Some(sq) => all
                .into_iter()
                .filter(|a| matches!(a, Action::Move { from: f, .. } if *f == sq))
                .collect(),
        };
        Ok(filtered)
    }

    /// Applies `action` to the game, returning an [`ApplyResult`].
    ///
    /// On success the action is committed and the previous state is pushed onto
    /// the undo stack. On engine error the game state is unchanged.
    pub fn apply_action(&self, id: GameId, action: Action) -> Result<ApplyResult, String> {
        let mut guard = self.games.lock().map_err(|e| e.to_string())?;
        let entry = guard.get_mut(&id).ok_or_else(|| format!("unknown game id {id}"))?;

        let snapshot = entry.game.clone();
        match entry.game.apply(action) {
            Ok(outcome) => {
                entry.undo_stack.push(snapshot);
                let result = entry.game.terminal_result();
                Ok(ApplyResult {
                    view: view_of(&entry.game),
                    turn_ended_on_check: outcome.ended_on_check,
                    last_capture: outcome.captured,
                    result,
                })
            }
            Err(e) => Err(format!("{e:?}")),
        }
    }

    /// Returns the geometric move targets for the piece at `from`, ignoring AP/turn rules.
    ///
    /// Returns an empty Vec if the square is empty. Returns Err if `id` is unknown
    /// or the mutex is poisoned.
    pub fn piece_moves(&self, id: GameId, from: Sq) -> Result<Vec<Sq>, String> {
        use cairn_core::movement::move_targets;
        let guard = self.games.lock().map_err(|e| e.to_string())?;
        let entry = guard.get(&id).ok_or_else(|| format!("unknown game id {id}"))?;
        if entry.game.pos.piece_at(from).is_none() {
            return Ok(Vec::new());
        }
        Ok(move_targets(&entry.game.pos, from))
    }

    /// Restores the previous game state by popping the undo stack.
    ///
    /// Returns an error string if `id` is unknown or the undo stack is empty.
    pub fn undo(&self, id: GameId) -> Result<GameView, String> {
        let mut guard = self.games.lock().map_err(|e| e.to_string())?;
        let entry = guard.get_mut(&id).ok_or_else(|| format!("unknown game id {id}"))?;
        match entry.undo_stack.pop() {
            Some(previous) => {
                entry.game = previous;
                Ok(view_of(&entry.game))
            }
            None => Err(format!("undo stack is empty for game {id}")),
        }
    }
}

impl Default for GameStore {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use cairn_core::actions::Action;
    use cairn_core::square::Sq;

    fn default_store() -> GameStore {
        GameStore::new()
    }

    fn first_legal_move(store: &GameStore, id: GameId) -> Action {
        let actions = store.legal_actions(id, None).expect("legal_actions must succeed");
        *actions
            .iter()
            .find(|a| matches!(a, Action::Move { .. }))
            .expect("there must be at least one legal Move in the opening position")
    }

    /// After `new_game`, the view must show 40 pieces (18+2 per side) and the
    /// turn state must be P1 to move with 2 AP.
    #[test]
    fn new_game_then_get_view_returns_starting_material() {
        let store = default_store();
        let (id, view) = store.new_game(RuleConfig::default());

        // Board has exactly 81 entries.
        assert_eq!(view.board.len(), 81);

        // Count non-empty squares.
        let piece_count = view.board.iter().filter(|c| c.is_some()).count();
        assert_eq!(piece_count, 40, "opening position must have 40 pieces total");

        // Turn state.
        assert_eq!(
            view.to_move,
            cairn_core::piece::Player::P1,
            "P1 moves first"
        );
        assert_eq!(view.ap_remaining, 2, "opening AP must be 2");
        assert_eq!(view.result, None, "game must not be over at start");

        // get_view must agree.
        let view2 = store.get_view(id).expect("get_view must succeed");
        assert_eq!(view2.board.len(), 81);
        assert_eq!(view2.to_move, cairn_core::piece::Player::P1);
    }

    /// Applying a clearly illegal action must return Err, and the game state
    /// must be identical to what it was before the attempt.
    #[test]
    fn apply_illegal_action_returns_err_without_mutating() {
        let store = default_store();
        let (id, _) = store.new_game(RuleConfig::default());

        // Capture the full game view before the illegal action.
        let before = store.get_view(id).expect("get_view must succeed");

        // Move from an empty square is always illegal.
        let empty_sq = Sq::new(4, 4).unwrap();
        let illegal = Action::Move {
            from: empty_sq,
            to: Sq::new(4, 5).unwrap(),
        };

        let result = store.apply_action(id, illegal);
        assert!(result.is_err(), "illegal action must return Err");

        // Capture the game view after the failed action.
        let after = store.get_view(id).expect("get_view must succeed after failed apply");

        // Entire board must be unchanged.
        assert_eq!(before.board, after.board, "board must be unchanged");

        // Turn state and reserves must also be unchanged.
        assert_eq!(
            before.ap_remaining, after.ap_remaining,
            "AP must be unchanged"
        );
        assert_eq!(
            before.to_move, after.to_move,
            "to_move must be unchanged"
        );
        assert_eq!(
            before.reserves, after.reserves,
            "reserves must be unchanged"
        );
    }

    /// `legal_actions(id, Some(sq))` must return only `Move` actions whose
    /// `from` equals `sq`, and that set must be a subset of all legal actions.
    #[test]
    fn legal_actions_for_selected_square_filters_to_that_piece() {
        let store = default_store();
        let (id, _) = store.new_game(RuleConfig::default());

        let all = store.legal_actions(id, None).expect("legal_actions(None) must succeed");
        assert!(!all.is_empty(), "opening position must have legal actions");

        // Pick the first Move action's from-square.
        let first_move_from = all
            .iter()
            .find_map(|a| {
                if let Action::Move { from, .. } = a {
                    Some(*from)
                } else {
                    None
                }
            })
            .expect("there must be at least one Move in the opening position");

        let filtered = store
            .legal_actions(id, Some(first_move_from))
            .expect("legal_actions(Some) must succeed");

        // Every filtered action is a Move from the selected square.
        for a in &filtered {
            match a {
                Action::Move { from, .. } => {
                    assert_eq!(*from, first_move_from, "all returned actions must be from the selected square");
                }
                _ => panic!("filtered result must only contain Move actions"),
            }
        }

        // Filtered set is a strict subset.
        assert!(!filtered.is_empty(), "selected square must have at least one legal move");
        assert!(
            filtered.len() < all.len(),
            "filtered set must be strictly smaller than all legal actions"
        );
        for a in &filtered {
            assert!(all.contains(a), "every filtered action must appear in the full set");
        }
    }

    /// After applying a legal action, undo must restore the exact previous view.
    #[test]
    fn apply_then_undo_restores_previous_view() {
        let store = default_store();
        let (id, view_before) = store.new_game(RuleConfig::default());

        let action = first_legal_move(&store, id);
        store.apply_action(id, action).expect("apply must succeed");

        let view_restored = store.undo(id).expect("undo must succeed");

        assert_eq!(
            view_before.ap_remaining, view_restored.ap_remaining,
            "AP must be restored"
        );
        assert_eq!(
            view_before.to_move, view_restored.to_move,
            "to_move must be restored"
        );
        // Board must match entry-by-entry.
        for (i, (before, restored)) in view_before
            .board
            .iter()
            .zip(view_restored.board.iter())
            .enumerate()
        {
            match (before, restored) {
                (None, None) => {}
                (Some(b), Some(r)) => {
                    assert_eq!(b.owner, r.owner, "square {i} owner must match after undo");
                    assert_eq!(b.kind, r.kind, "square {i} kind must match after undo");
                    assert_eq!(b.height, r.height, "square {i} height must match after undo");
                }
                _ => panic!("square {i} occupancy differs after undo"),
            }
        }
    }

    /// `piece_moves` returns geometric targets for an enemy piece and empty for an empty square.
    #[test]
    fn piece_moves_returns_targets_for_enemy_and_empty_for_vacant() {
        let store = default_store();
        let (id, view) = store.new_game(RuleConfig::default());

        // Find an enemy (P2) piece square on the board.
        let enemy_sq = view.board
            .iter()
            .enumerate()
            .find_map(|(i, pc)| {
                if let Some(p) = pc {
                    if p.owner == cairn_core::piece::Player::P2 {
                        return Some(Sq::new((i % 9) as u8, (i / 9) as u8).unwrap());
                    }
                }
                None
            })
            .expect("opening board must have at least one P2 piece");

        let targets = store.piece_moves(id, enemy_sq).expect("piece_moves must succeed");
        assert!(!targets.is_empty(), "enemy piece at opening must have geometric move targets");

        // An empty square must return empty.
        // Find an empty square (centre area, typically empty at start).
        let empty_sq = view.board
            .iter()
            .enumerate()
            .find_map(|(i, pc)| {
                if pc.is_none() {
                    return Some(Sq::new((i % 9) as u8, (i / 9) as u8).unwrap());
                }
                None
            })
            .expect("opening board must have at least one empty square");

        let empty_targets = store.piece_moves(id, empty_sq).expect("piece_moves on empty must succeed");
        assert!(empty_targets.is_empty(), "empty square must have no move targets");
    }

    /// Operations on an unknown game ID must return an Err.
    #[test]
    fn unknown_game_id_returns_err() {
        let store = default_store();
        let bad_id: GameId = 99999;

        assert!(store.get_view(bad_id).is_err(), "get_view with unknown id must fail");
        assert!(
            store.legal_actions(bad_id, None).is_err(),
            "legal_actions with unknown id must fail"
        );
        assert!(
            store
                .apply_action(bad_id, Action::Move {
                    from: Sq::new(0, 0).unwrap(),
                    to: Sq::new(0, 1).unwrap()
                })
                .is_err(),
            "apply_action with unknown id must fail"
        );
        assert!(store.undo(bad_id).is_err(), "undo with unknown id must fail");
    }
}
