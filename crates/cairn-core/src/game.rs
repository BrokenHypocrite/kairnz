use crate::actions::{legal_actions, Action};
use crate::apply::{apply_action, ActionOutcome};
use crate::actions::IllegalAction;
use crate::config::RuleConfig;
use crate::outcome::{DrawReason, GameResult};
use crate::piece::Player;
use crate::position::Position;

/// A game session wrapping a `Position` with a turn-boundary position history
/// used for repetition detection.
pub struct Game {
    /// The current game position.
    pub pos: Position,
    /// Zobrist hashes recorded at each turn boundary (including the opening position).
    history: Vec<u64>,
}

impl Game {
    /// Creates a new game from the standard §2 opening position.
    ///
    /// The initial position's Zobrist hash is seeded into history so that
    /// repetition counting begins from the very first turn boundary.
    pub fn new_standard(config: RuleConfig) -> Game {
        let pos = Position::new_standard(config);
        let initial_hash = pos.zobrist;
        Game {
            pos,
            history: vec![initial_hash],
        }
    }

    /// Evaluates the terminal condition at the current position.
    ///
    /// Must be called at a turn boundary (after `apply` has advanced the turn).
    /// The checks are ordered so that the strongest signal (keystone capture) is
    /// reported first, then loss-by-no-legal-action, then draw conditions.
    ///
    /// Returns `None` when the game is still in progress.
    pub fn terminal_result(&self) -> Option<GameResult> {
        // 1. P1 lost both Keystones -> P2 wins.
        if self.pos.keystones_of(Player::P1).count() == 0 {
            return Some(GameResult::Win(Player::P2));
        }

        // 2. P2 lost both Keystones -> P1 wins.
        if self.pos.keystones_of(Player::P2).count() == 0 {
            return Some(GameResult::Win(Player::P1));
        }

        // 3. The player to move has no legal action at the start of their turn -> they lose.
        if legal_actions(&self.pos).is_empty() {
            return Some(GameResult::Win(self.pos.to_move.opponent()));
        }

        // 4. Ply cap reached -> draw.
        if self.pos.ply >= self.pos.config.max_plies {
            return Some(GameResult::Draw(DrawReason::MaxPlies));
        }

        // 5. Position has appeared repetition_fold times -> draw.
        let fold = self.pos.config.repetition_fold as usize;
        let occurrences = self.history.iter().filter(|&&h| h == self.pos.zobrist).count();
        if occurrences >= fold {
            return Some(GameResult::Draw(DrawReason::Repetition));
        }

        None
    }

    /// Applies `action` to the current position, records the new turn-boundary
    /// hash when a turn completes, then sets the authoritative terminal result.
    ///
    /// Returns an `ActionOutcome` whose `result` field is the definitive terminal
    /// status (it supersedes any win-by-capture result set internally by `apply_action`).
    pub fn apply(&mut self, action: Action) -> Result<ActionOutcome, IllegalAction> {
        let mut outcome = apply_action(&mut self.pos, action)?;

        if outcome.turn_ended {
            // Record the position at this turn boundary for repetition detection.
            self.history.push(self.pos.zobrist);
        }

        // Authoritative terminal check: subsumes the win-by-capture result and
        // additionally detects loss-by-no-legal-action and draw conditions.
        outcome.result = self.terminal_result();

        Ok(outcome)
    }

    /// Returns the terminal result if the game is over, or `None` if it continues.
    pub fn result(&self) -> Option<GameResult> {
        self.terminal_result()
    }

    /// Returns the player whose turn it is to move.
    pub fn to_move(&self) -> Player {
        self.pos.to_move
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::actions::Action;
    use crate::config::RuleConfig;
    use crate::outcome::{DrawReason, GameResult};
    use crate::piece::{Piece, PieceKind, Player};
    use crate::position::{Position, TurnState};
    use crate::square::{BitBoard81, NUM_SQUARES};

    // --- Helpers ---

    fn sq(file: u8, rank: u8) -> crate::square::Sq {
        crate::square::Sq::new(file, rank).unwrap()
    }

    fn place(pos: &mut Position, file: u8, rank: u8, piece: Piece) {
        let s = sq(file, rank);
        pos.board[s.0 as usize] = Some(piece);
    }

    /// Build a Game from a bare Position and a starting history seeded with the
    /// position's current Zobrist (mirrors `new_standard` but accepts any Position).
    fn game_from_pos(pos: Position) -> Game {
        let initial_hash = pos.zobrist;
        Game { pos, history: vec![initial_hash] }
    }

    fn empty_pos_with_ap(ap: u8) -> Position {
        Position {
            board: [None; NUM_SQUARES],
            reserves: [0, 0],
            to_move: Player::P1,
            turn: TurnState {
                ap_remaining: ap,
                capture_locked: BitBoard81::default(),
                keystone_moved: BitBoard81::default(),
                enemy_checked_at_start: BitBoard81::default(),
            },
            config: RuleConfig::default(),
            zobrist: 0,
            ply: 0,
        }
    }

    // ---------------------------------------------------------------------------
    // Test: capturing both keystones wins
    // ---------------------------------------------------------------------------

    /// Capturing the opponent's last Keystone immediately ends the game in a Win.
    #[test]
    fn capturing_both_keystones_wins() {
        // P1 Pillar (h2) at (4,3), adjacent to the ONLY P2 Keystone at (4,4).
        // No other P2 Keystones exist, so capturing it wins for P1.
        let mut pos = empty_pos_with_ap(2);
        place(&mut pos, 4, 3, Piece::new(Player::P1, PieceKind::Stone, 2));
        place(&mut pos, 4, 4, Piece::new(Player::P2, PieceKind::Keystone, 1));
        // Keep a P1 Keystone so terminal_result doesn't see P1 as having 0 keystones.
        place(&mut pos, 0, 0, Piece::new(Player::P1, PieceKind::Keystone, 1));
        pos.recompute_zobrist();

        let mut game = game_from_pos(pos);
        let outcome = game
            .apply(Action::Move { from: sq(4, 3), to: sq(4, 4) })
            .expect("capture must be legal");

        assert_eq!(
            outcome.result,
            Some(GameResult::Win(Player::P1)),
            "capturing the last opponent Keystone must set Win(P1)"
        );
        assert_eq!(
            game.terminal_result(),
            Some(GameResult::Win(Player::P1)),
            "terminal_result must agree"
        );
    }

    // ---------------------------------------------------------------------------
    // Test: no legal action at turn start -> loss
    // ---------------------------------------------------------------------------

    /// At the start of their turn, the player to move has zero legal actions -> they lose.
    ///
    /// Construction: P1 has a single Keystone with `keystone_single_move=true` and
    /// the Keystone has already-moved bit set (simulating the start of a new turn
    /// where the Keystone was already counted as having moved -- i.e., we set the
    /// bit manually). Reserve=0, no other P1 pieces. P2 has a Keystone elsewhere.
    ///
    /// Note: `keystone_moved` is cleared by `advance_turn`. To simulate a turn
    /// boundary where this matters, we instead use a Keystone surrounded on all
    /// reachable squares by P1 own Stones (all 3 reachable squares from corner (0,0):
    /// (1,0), (0,1), (1,1)), AND we ensure none of those Stones have any legal
    /// moves either. We do that by surrounding all of them with the board edge and
    /// other P1 Stones.
    ///
    /// Full cage: P1 occupies a 2x2 block in the corner:
    ///   (0,0)=Keystone, (1,0)=Stone, (0,1)=Stone, (1,1)=Stone
    ///   Stones at (1,0): can reach (2,0), (1,1)[P1 blocked]. Add P1 Stone at (2,0).
    ///   Stone at (0,1): can reach (0,2), (1,1)[P1 blocked]. Add P1 Stone at (0,2).
    ///   Stone at (1,1): can reach (2,1), (1,2), (0,1)[P1 blocked], (1,0)[P1 blocked].
    ///     Add P1 Stones at (2,1) and (1,2).
    ///   Stone at (2,0): can reach (3,0), (2,1)[P1 blocked]. Add P1 Stone at (3,0).
    ///   Stone at (0,2): can reach (0,3), (1,2)[P1 blocked]. Add P1 Stone at (0,3).
    ///   Stone at (2,1): can reach (3,1), (2,2), (2,0)[P1 blocked], (2,1)-diag blocked by config...
    ///     Wait: h1 Stones only move orthogonally. So (2,1) can reach (3,1),(1,1)[P1],(2,2),(2,0)[P1].
    ///     Add P1 Stones at (3,1) and (2,2).
    ///   Stone at (1,2): can reach (0,2)[P1],(2,2)[P1],(1,3),(1,1)[P1]. Add Stone at (1,3).
    ///   Stone at (3,0): can reach (4,0),(3,1)[P1]. Add Stone at (4,0).
    ///   Stone at (0,3): can reach (1,3)[need to check],(0,4). Add Stones at (1,3) if not
    ///     already there, and (0,4).
    ///
    /// This gets complex. Simpler: use `keystone_single_move=true` AND pre-set the
    /// `keystone_moved` bit to simulate a position where P1's only piece is already
    /// flagged as moved. But `advance_turn` clears this bit, so at the start of a real
    /// turn the bit would be clear. We're calling `terminal_result` directly without
    /// going through `advance_turn`, so manually setting the bit is fine for testing
    /// the LOSS condition itself (which only calls `legal_actions`).
    #[test]
    fn no_legal_action_at_turn_start_loses() {
        let mut cfg = RuleConfig::default();
        // Enable keystone_single_move so a Keystone flagged as moved cannot move again.
        cfg.keystone_single_move = true;

        let mut pos = empty_pos_with_ap(2);
        pos.config = cfg;

        // P1's only piece: a Keystone at (4,4).
        place(&mut pos, 4, 4, Piece::new(Player::P1, PieceKind::Keystone, 1));

        // Pre-set the keystone_moved bit to simulate P1 already having moved their
        // Keystone. With keystone_single_move=true this prevents the Keystone from
        // being a legal Move source. Combined with reserve=0, P1 has zero legal actions.
        pos.turn.keystone_moved.set(sq(4, 4));

        // No reserve -> no Place or Stack.
        pos.reserves[Player::P1.index()] = 0;

        // P2 has a Keystone safely elsewhere so the game is not over by keystone-count.
        place(&mut pos, 8, 8, Piece::new(Player::P2, PieceKind::Keystone, 1));

        pos.to_move = Player::P1;
        pos.recompute_zobrist();

        let game = game_from_pos(pos);

        // Verify the precondition: P1 really has no legal actions.
        assert!(
            legal_actions(&game.pos).is_empty(),
            "precondition: P1 must have zero legal actions"
        );

        assert_eq!(
            game.terminal_result(),
            Some(GameResult::Win(Player::P2)),
            "player with no legal action at turn start must lose"
        );
    }

    // ---------------------------------------------------------------------------
    // Test: max-ply cap reports draw
    // ---------------------------------------------------------------------------

    /// When `pos.ply >= config.max_plies`, the game is a draw by MaxPlies.
    #[test]
    fn max_ply_cap_reports_draw() {
        let mut cfg = RuleConfig::default();
        cfg.max_plies = 10;

        let mut pos = empty_pos_with_ap(2);
        pos.config = cfg;
        // Both players have at least one Keystone so the game is not decided by captures.
        place(&mut pos, 0, 0, Piece::new(Player::P1, PieceKind::Keystone, 1));
        place(&mut pos, 8, 8, Piece::new(Player::P2, PieceKind::Keystone, 1));
        // Place a P1 Stone so legal_actions is non-empty (game is still live).
        place(&mut pos, 4, 4, Piece::new(Player::P1, PieceKind::Stone, 1));

        // Drive ply to the cap.
        pos.ply = 10;
        pos.recompute_zobrist();

        let game = game_from_pos(pos);
        assert_eq!(
            game.terminal_result(),
            Some(GameResult::Draw(DrawReason::MaxPlies)),
            "ply >= max_plies must report Draw(MaxPlies)"
        );
    }

    // ---------------------------------------------------------------------------
    // Test: N-fold repetition reports draw
    // ---------------------------------------------------------------------------

    /// When the current Zobrist appears `repetition_fold` times in history, it is a draw.
    #[test]
    fn threefold_repetition_reports_draw() {
        let mut cfg = RuleConfig::default();
        cfg.repetition_fold = 3;

        let mut pos = empty_pos_with_ap(2);
        pos.config = cfg;
        place(&mut pos, 0, 0, Piece::new(Player::P1, PieceKind::Keystone, 1));
        place(&mut pos, 8, 8, Piece::new(Player::P2, PieceKind::Keystone, 1));
        place(&mut pos, 4, 4, Piece::new(Player::P1, PieceKind::Stone, 1));
        pos.recompute_zobrist();

        let current_hash = pos.zobrist;

        // Build a game with the current hash already appearing 3 times in history
        // (one seeded by game_from_pos + two more injected here).
        let mut game = game_from_pos(pos);
        game.history.push(current_hash); // second occurrence
        game.history.push(current_hash); // third occurrence

        assert_eq!(
            game.terminal_result(),
            Some(GameResult::Draw(DrawReason::Repetition)),
            "three occurrences of the same hash must report Draw(Repetition)"
        );
    }

    // ---------------------------------------------------------------------------
    // Test: no legal action mid-turn ends turn without losing
    // ---------------------------------------------------------------------------

    /// After an action, the acting player still has AP but no further legal action.
    /// The turn should end (advance to the opponent) and the game must NOT be over:
    /// the acting player does NOT lose (the loss rule only applies at turn START).
    ///
    /// Construction: P1 has AP=2. Place a P1 Stone at (0,0). After the first action
    /// (a Move that does something), the only remaining P1 piece has no targets
    /// because it is completely surrounded by P1 friendlies, and the reserve is 0.
    /// We model this by using apply.rs's mid-turn no-legal-action end.
    ///
    /// Simpler direct approach: call `apply_action` on a position where after the
    /// action there are no remaining legal actions for P1, then check that the
    /// turn advanced to P2 and `terminal_result` returns None once the turn has
    /// advanced.
    #[test]
    fn no_legal_action_mid_turn_ends_turn_without_losing() {
        // P1 has AP=2. P1 has a single Stone at (0,0).
        // After that Stone moves to (1,0), it will be blocked on all four sides:
        //   - south (0,0): now empty (the source -- but (1,0) south is (1,-1) which is off-board)
        //   - We'll surround (1,0) with P1 friendlies on every valid neighbor.
        // A height-1 Stone at (1,0) can step to: (0,0), (2,0), (1,1).
        // Block (2,0) and (1,1) with P1 Stones; (0,0) becomes empty after the move.
        // Then block (0,0) with another P1 Stone placed beforehand? No, the moving
        // piece LEAVES (0,0), so (0,0) becomes empty and is a valid return target.
        //
        // Alternative: use a Keystone at (0,0) blocked by friendlies, then have P1
        // place a stone somewhere as the first action (but then (1,0) becomes P1's
        // own piece and might be a valid place to block from above).
        //
        // Simplest approach: construct a position where P1's only piece is a Stone
        // at (4,4) surrounded on all sides (ortho) by P1 Stones. After P1 uses AP1
        // to do a Stack (which costs 2 AP) it would exhaust turn. But that's AP0.
        //
        // Actual correct approach per spec §5: build via apply_action.
        //
        // P1 Stone at (0,0), surrounded by P1 Stones at (1,0) and (0,1).
        // h1 Stone at (0,0) can only step orthogonally: candidates are (1,0) and (0,1)
        // (both occupied by P1) and below/left which are off-board. So it already has no
        // moves even before the turn starts! That's "turn start no legal action" = loss.
        //
        // We need mid-turn: P1 has AP=2, makes one action, then after that action
        // the remaining legal actions are empty.
        //
        // The trick: start with P1 Stone (h2, so it can move diagonally) at (0,0),
        // and friendly stones fully surrounding (0,1) and (1,0). The h2 Stone can also
        // step diagonally to (1,1). We place a P1 Stone at (1,1) too. Then P1 moves
        // from (0,0) to... wait, (1,1) is friendly so that's blocked.
        //
        // Actually: P1 Stone h1 at (0,0) can step to (1,0) and (0,1) (N/E of corner).
        // If we put P1 Stones at (2,0) and (0,2) but leave (1,0) and (0,1) empty,
        // then P1 can move the Stone to either. Once it's at (1,0), it can step to:
        // (0,0) [now empty], (2,0) [P1, blocked], (1,1) [if empty -- blocked if we put one there].
        // So if we add a P1 Stone at (1,1), then after moving to (1,0), P1's legal actions
        // become just Move {(1,0)->(0,0)}. Still not empty.
        //
        // Better: fully surround the destination. h1 Stone at (0,0) can go to (1,0).
        // From (1,0) it can reach: (0,0), (2,0), (1,1). Block all three with P1 Stones.
        // - (0,0): becomes empty when it moves, then we need another P1 Stone here
        //   -- but the piece just left it, so it IS empty. We can't place a P1 piece there
        //   after the fact without an action.
        //
        // Right approach: the Stone can return to (0,0) from (1,0), so that path is always
        // open unless something is placed there.
        //
        // We need the piece to move to a place where all neighbors it can reach are already
        // occupied by friendly pieces BEFORE the move, and the source square will be
        // blocked by some other mechanism.
        //
        // Actually for this test, we can use a Keystone. A Keystone at (0,0) with
        // keystone_single_move=true enabled will be forbidden from moving again
        // after the first action. Then after the Keystone moves away, the source (0,0)
        // is empty, but the Keystone is now stuck (moved once). If no other P1 pieces
        // can act (empty reserve, all blocked), the turn ends.
        //
        // Setup: keystone_single_move=true. P1 Keystone at (1,0). P1 Stone at (0,0) blocked
        // on all sides (right beside the Keystone). Actually let's keep it simpler:
        // P1 only has ONE Keystone at center (4,4). keystone_single_move=true. Reserve=0.
        // P1 moves the Keystone (1 AP used, 1 AP remains). Now the Keystone has the
        // keystone_single_move bit set, so it cannot move again. Reserve is still 0.
        // There are no other P1 pieces. Therefore legal_actions is empty.
        // finalize detects this and ends the turn. P2 takes over. Game is NOT over
        // because both players still have Keystones.

        let mut cfg = RuleConfig::default();
        cfg.keystone_single_move = true;

        let mut pos = empty_pos_with_ap(2);
        pos.config = cfg;

        // P1 Keystone at (4,4) -- the only P1 piece.
        place(&mut pos, 4, 4, Piece::new(Player::P1, PieceKind::Keystone, 1));

        // P2 Keystone somewhere safe -- game must not be won by keystone-count.
        place(&mut pos, 8, 8, Piece::new(Player::P2, PieceKind::Keystone, 1));

        pos.reserves[Player::P1.index()] = 0;
        pos.recompute_zobrist();

        let mut game = game_from_pos(pos);

        // Verify precondition: P1 currently has legal actions (Keystone can move).
        assert!(
            !legal_actions(&game.pos).is_empty(),
            "precondition: P1 must have legal actions before the move"
        );

        // Move the Keystone one step north. This uses 1 AP (1 remains). With
        // keystone_single_move=true the Keystone is now flagged. No other P1 pieces
        // or reserve, so legal_actions becomes empty mid-turn -> finalize ends turn.
        let outcome = game
            .apply(Action::Move { from: sq(4, 4), to: sq(4, 5) })
            .expect("keystone move must be legal");

        assert!(outcome.turn_ended, "turn must end because no further actions available");
        assert_eq!(
            game.pos.to_move,
            Player::P2,
            "turn must have advanced to P2 after forced end"
        );
        assert_eq!(
            outcome.result, None,
            "game must NOT be over: the acting player did not lose"
        );
        assert_eq!(
            game.terminal_result(),
            None,
            "terminal_result must also return None"
        );
    }

    // ---------------------------------------------------------------------------
    // Test: ongoing position returns no terminal result
    // ---------------------------------------------------------------------------

    /// The standard opening position has no terminal result.
    #[test]
    fn ongoing_position_has_no_terminal_result() {
        let game = Game::new_standard(RuleConfig::default());
        assert_eq!(
            game.terminal_result(),
            None,
            "the standard opening position must have no terminal result"
        );
    }
}
