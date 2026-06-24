use cairn_core::{
    actions::{legal_actions, Action},
    game::Game,
    outcome::GameResult,
    piece::Player,
};
use rand::Rng;
use rand::SeedableRng;
use rand_pcg::Pcg64;

use crate::policy::Policy;

/// The UCB1 exploration constant. The classic theoretical value is `sqrt(2)`,
/// balancing exploitation of high-value children against exploration of
/// under-visited ones.
const DEFAULT_EXPLORATION: f64 = std::f64::consts::SQRT_2;

/// Maximum number of plies a single random rollout may play before it is
/// abandoned and scored as a draw. Bounds worst-case rollout cost so that a
/// non-terminating random game cannot stall the search.
const DEFAULT_ROLLOUT_CAP: u32 = 400;

/// Reward awarded to the player to move at a node when the rollout result is a
/// win for that player.
const REWARD_WIN: f64 = 1.0;
/// Reward awarded when the rollout result is a loss for the player to move.
const REWARD_LOSS: f64 = 0.0;
/// Reward awarded for a drawn rollout result.
const REWARD_DRAW: f64 = 0.5;

/// A node in the Monte Carlo search tree, stored in a flat arena and addressed
/// by `usize` index.
///
/// VALUE PERSPECTIVE CONVENTION: a node's `value_sum` accumulates reward from
/// the perspective of `to_move`, i.e. the player who acts FROM this node. This
/// is the single convention used by both backprop and selection. Because the
/// same player may occupy consecutive nodes within one multi-AP turn, reward is
/// anchored to each node's `to_move` rather than alternated by depth.
struct Node {
    /// The game state at this node.
    game: Game,
    /// The player to move at this node (equals `game.pos.to_move`).
    to_move: Player,
    /// Index of the parent node, or `None` for the root.
    parent: Option<usize>,
    /// The action applied at the parent to reach this node, or `None` for the
    /// root. Used to recover the chosen move from the most-visited root child.
    action_from_parent: Option<Action>,
    /// Legal actions from this node that have not yet been expanded.
    untried: Vec<Action>,
    /// Arena indices of expanded child nodes.
    children: Vec<usize>,
    /// Number of times this node has been visited.
    visits: u32,
    /// Sum of rewards from the perspective of `to_move` (see convention above).
    value_sum: f64,
}

impl Node {
    /// Builds a fresh node for `game`, seeding `untried` with its legal actions.
    fn new(game: Game, parent: Option<usize>, action_from_parent: Option<Action>) -> Node {
        let to_move = game.pos.to_move;
        let untried = legal_actions(&game.pos);
        Node {
            game,
            to_move,
            parent,
            action_from_parent,
            untried,
            children: Vec::new(),
            visits: 0,
            value_sum: 0.0,
        }
    }

    /// A node is fully expanded once every legal action has a child.
    fn is_fully_expanded(&self) -> bool {
        self.untried.is_empty()
    }
}

/// Maps a finished game result onto a reward in `[0, 1]` from `player`'s point
/// of view: a win for `player` scores `REWARD_WIN`, a loss scores `REWARD_LOSS`,
/// and a draw scores `REWARD_DRAW`.
fn reward_for(player: Player, result: GameResult) -> f64 {
    match result {
        GameResult::Win(winner) if winner == player => REWARD_WIN,
        GameResult::Win(_) => REWARD_LOSS,
        GameResult::Draw(_) => REWARD_DRAW,
    }
}

/// A plain UCT Monte Carlo Tree Search agent using uniformly random rollouts
/// (no neural network). All randomness flows from a single seeded `Pcg64`, so a
/// given seed, position, and iteration count always yield the same action.
pub struct MctsPolicy {
    /// Number of MCTS iterations performed per `choose` call.
    iterations: u32,
    /// UCB1 exploration constant.
    exploration: f64,
    /// Maximum plies per random rollout before scoring it as a draw.
    rollout_cap: u32,
    /// The single source of randomness for selection, expansion, and rollouts.
    rng: Pcg64,
}

impl MctsPolicy {
    /// Constructs an `MctsPolicy` that runs `iterations` simulations per move,
    /// seeded for reproducibility. Uses `DEFAULT_EXPLORATION` and
    /// `DEFAULT_ROLLOUT_CAP`.
    pub fn new(iterations: u32, seed: u64) -> MctsPolicy {
        Self::with_params(iterations, DEFAULT_EXPLORATION, DEFAULT_ROLLOUT_CAP, seed)
    }

    /// Full control over search parameters (used by tests and the benchmark harness).
    pub fn with_params(iterations: u32, exploration: f64, rollout_cap: u32, seed: u64) -> MctsPolicy {
        MctsPolicy {
            iterations,
            exploration,
            rollout_cap,
            rng: Pcg64::seed_from_u64(seed),
        }
    }

    /// Runs one selection-expansion-simulation-backpropagation iteration over
    /// the arena rooted at index 0.
    fn run_iteration(&mut self, arena: &mut Vec<Node>) {
        let leaf = self.select(arena);
        let expanded = self.expand(arena, leaf);
        let result = self.simulate(arena[expanded].game.clone());
        self.backpropagate(arena, expanded, result);
    }

    /// Descends from the root, choosing the UCB1-best child at each fully
    /// expanded, non-terminal node, until reaching a node that is either not
    /// fully expanded or terminal. Returns that node's index.
    fn select(&self, arena: &[Node]) -> usize {
        let mut current = 0usize;
        loop {
            let node = &arena[current];
            if !node.is_fully_expanded() || node.game.terminal_result().is_some() {
                return current;
            }
            match self.best_uct_child(arena, current) {
                Some(child) => current = child,
                None => return current,
            }
        }
    }

    /// Selects the child of `parent` with the highest UCB1 score.
    ///
    /// A child's mean value is stored from the CHILD's `to_move` perspective. The
    /// parent ranks children by the outcome for the PARENT's mover, so when the
    /// child's mover differs from the parent's (a normal turn handover) the mean
    /// is flipped via `1.0 - mean`; within a multi-AP turn the movers match and
    /// the mean is used directly. This keeps selection consistent with backprop.
    fn best_uct_child(&self, arena: &[Node], parent: usize) -> Option<usize> {
        let parent_node = &arena[parent];
        let parent_visits = parent_node.visits.max(1);
        let ln_parent = (parent_visits as f64).ln();

        let mut best: Option<usize> = None;
        let mut best_score = f64::NEG_INFINITY;

        for &child_idx in &parent_node.children {
            let child = &arena[child_idx];
            let exploit_raw = child.value_sum / child.visits.max(1) as f64;
            let exploit = if child.to_move == parent_node.to_move {
                exploit_raw
            } else {
                1.0 - exploit_raw
            };
            let explore = self.exploration * (ln_parent / child.visits.max(1) as f64).sqrt();
            let score = exploit + explore;
            if score > best_score {
                best_score = score;
                best = Some(child_idx);
            }
        }
        best
    }

    /// If `leaf` is non-terminal and has an untried action, pops one (chosen via
    /// the RNG), applies it on a clone of the leaf's game, appends the resulting
    /// child to the arena, and returns the child's index. Otherwise returns
    /// `leaf` unchanged (a terminal or already-fully-expanded node).
    fn expand(&mut self, arena: &mut Vec<Node>, leaf: usize) -> usize {
        if arena[leaf].game.terminal_result().is_some() {
            return leaf;
        }
        if arena[leaf].untried.is_empty() {
            return leaf;
        }

        let pick = self.rng.gen_range(0..arena[leaf].untried.len());
        let action = arena[leaf].untried.swap_remove(pick);

        let mut child_game = arena[leaf].game.clone();
        // legal_actions guarantees the action is legal; ignore the outcome.
        let _ = child_game.apply(action);

        let child = Node::new(child_game, Some(leaf), Some(action));
        let child_idx = arena.len();
        arena.push(child);
        arena[leaf].children.push(child_idx);
        child_idx
    }

    /// Plays uniformly random legal actions from `game` until it reaches a
    /// terminal result or `rollout_cap` plies elapse. A rollout that hits the cap
    /// (or a dead state with no legal actions and no terminal result) is scored
    /// as a draw.
    fn simulate(&mut self, mut game: Game) -> GameResult {
        if let Some(result) = game.terminal_result() {
            return result;
        }
        for _ in 0..self.rollout_cap {
            let actions = legal_actions(&game.pos);
            if actions.is_empty() {
                return game
                    .terminal_result()
                    .unwrap_or(GameResult::Draw(cairn_core::outcome::DrawReason::MaxPlies));
            }
            let idx = self.rng.gen_range(0..actions.len());
            let _ = game.apply(actions[idx]);
            if let Some(result) = game.terminal_result() {
                return result;
            }
        }
        GameResult::Draw(cairn_core::outcome::DrawReason::MaxPlies)
    }

    /// Walks from `node` back to the root, incrementing `visits` and adding
    /// `reward_for(node.to_move, result)` to each node's `value_sum`. Because
    /// every node's value is stored from its own mover's perspective, the same
    /// reward function applies uniformly regardless of multi-AP turn structure.
    fn backpropagate(&self, arena: &mut Vec<Node>, node: usize, result: GameResult) {
        let mut current = Some(node);
        while let Some(idx) = current {
            arena[idx].visits += 1;
            arena[idx].value_sum += reward_for(arena[idx].to_move, result);
            current = arena[idx].parent;
        }
    }

    /// Returns the root action leading to the most-visited child (the robust
    /// child), the standard final-move criterion for UCT.
    fn most_visited_action(&self, arena: &[Node]) -> Option<Action> {
        let root = &arena[0];
        let mut best_action: Option<Action> = None;
        let mut best_visits = 0u32;

        // Each child stores the action that produced it, so the most-visited
        // child maps directly back to the root action to play.
        for &child_idx in &root.children {
            let child = &arena[child_idx];
            if child.visits > best_visits {
                best_visits = child.visits;
                best_action = child.action_from_parent;
            }
        }
        best_action
    }
}

impl Policy for MctsPolicy {
    /// Runs UCT for `iterations` simulations from `game` and returns the
    /// most-visited root action, or `None` when there is no legal action.
    fn choose(&mut self, game: &Game) -> Option<Action> {
        if game.terminal_result().is_some() {
            return None;
        }
        let root = Node::new(game.clone(), None, None);
        if root.untried.is_empty() && root.children.is_empty() {
            return None;
        }
        let mut arena: Vec<Node> = vec![root];

        for _ in 0..self.iterations {
            self.run_iteration(&mut arena);
        }

        self.most_visited_action(&arena)
    }

    /// Returns `"mcts"`.
    fn name(&self) -> &str {
        "mcts"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::random::RandomPolicy;
    use cairn_core::{
        actions::legal_actions,
        config::RuleConfig,
        outcome::GameResult,
        piece::{Piece, PieceKind, Player},
        position::{Position, TurnState},
        square::{BitBoard81, Sq, NUM_SQUARES},
    };

    /// Number of iterations used by tests that need MCTS to actually converge.
    const CONVERGED_ITERS: u32 = 300;
    /// Iterations used by the match test; kept modest so the test stays fast.
    const MATCH_ITERS: u32 = 20;
    /// Rollout cap used in the match test. Games from the reduced starting
    /// position terminate in well under 80 plies, so rollouts reach terminal
    /// states and give MCTS real signal without running the full 400-ply cap.
    const MATCH_ROLLOUT_CAP: u32 = 40;
    /// Number of games played in the MCTS-vs-random match test.
    const MATCH_GAMES: u32 = 20;
    /// Minimum decisive-game win rate MCTS must achieve against a random opponent.
    const MIN_WIN_RATE: f64 = 0.55;
    /// `max_plies` guard for the match game loop, ensuring no single game can
    /// stall the test even if the starting position is unexpectedly long.
    const MATCH_MAX_PLIES: u32 = 300;

    fn sq(file: u8, rank: u8) -> Sq {
        Sq::new(file, rank).expect("file and rank in bounds")
    }

    /// Builds a `Game` from a bare `Position`. `Game::history` is private, so we
    /// start from a standard game and overwrite its public `pos` field (mirroring
    /// the established pattern in the greedy tests). The seeded history is the
    /// standard opening's, which is irrelevant to these short tactical tests.
    fn game_from_pos(pos: Position) -> Game {
        let mut game = Game::new_standard(RuleConfig::default());
        game.pos = pos;
        game
    }

    fn minimal_pos(to_move: Player, ap: u8) -> Position {
        Position {
            board: [None; NUM_SQUARES],
            reserves: [0, 0],
            to_move,
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

    fn place(pos: &mut Position, file: u8, rank: u8, piece: Piece) {
        pos.board[sq(file, rank).0 as usize] = Some(piece);
    }

    /// On the standard opening, `choose` returns a legal action.
    #[test]
    fn mcts_returns_a_legal_action() {
        let game = Game::new_standard(RuleConfig::default());
        let legal = legal_actions(&game.pos);
        let action = MctsPolicy::new(CONVERGED_ITERS, 1).choose(&game);
        assert!(action.is_some(), "opening position must yield an action");
        assert!(
            legal.contains(&action.expect("action present")),
            "chosen action must be legal"
        );
    }

    /// Two policies with the same seed choose identically from the same position.
    #[test]
    fn mcts_is_deterministic_for_a_seed() {
        let game = Game::new_standard(RuleConfig::default());
        let a = MctsPolicy::new(200, 7).choose(&game);
        let b = MctsPolicy::new(200, 7).choose(&game);
        assert_eq!(a, b, "same seed must produce the same action");
    }

    /// In a position where the side to move can capture the opponent's LAST
    /// Keystone, MCTS must return that winning move. A backprop sign error would
    /// make MCTS avoid the win, so this test guards the perspective convention.
    #[test]
    fn mcts_prefers_an_immediately_winning_move() {
        // P1 to move, 2 AP. A P1 Stone at (4,3) sits adjacent to P2's ONLY
        // Keystone at (4,4); moving onto it captures it and wins. P1 also has
        // a Keystone (so P1 is not already lost) and several quiet alternatives.
        let mut pos = minimal_pos(Player::P1, 2);
        place(&mut pos, 4, 3, Piece::new(Player::P1, PieceKind::Stone, 2));
        place(&mut pos, 4, 4, Piece::new(Player::P2, PieceKind::Keystone, 1));
        place(&mut pos, 0, 0, Piece::new(Player::P1, PieceKind::Keystone, 1));
        // Quiet alternatives: another P1 Stone far away with room to wander.
        place(&mut pos, 0, 8, Piece::new(Player::P1, PieceKind::Stone, 1));
        pos.recompute_zobrist();

        let winning = Action::Move { from: sq(4, 3), to: sq(4, 4) };
        let game = game_from_pos(pos);
        assert!(
            legal_actions(&game.pos).contains(&winning),
            "precondition: the winning capture must be legal"
        );
        assert!(
            legal_actions(&game.pos).len() > 1,
            "precondition: there must be non-winning alternatives too"
        );

        let action = MctsPolicy::new(CONVERGED_ITERS, 3).choose(&game);
        assert_eq!(
            action,
            Some(winning),
            "MCTS must pick the immediately winning keystone capture"
        );
    }

    /// Builds a reduced-piece starting position for the match test.
    ///
    /// Each side has one Keystone and four Stones placed near the center of the
    /// board (but not adjacent to each other). With only ten pieces total, random
    /// rollouts reach a decisive result in roughly 30-80 plies -- fast enough that
    /// MCTS gets real terminal-state signal even with a modest rollout cap.
    fn match_start_game(seed_offset: u32) -> Game {
        let config = RuleConfig { max_plies: MATCH_MAX_PLIES, ..RuleConfig::default() };
        let mut pos = minimal_pos(Player::P1, config.first_turn_ap);
        pos.config = config;
        // P1 pieces: keystone at (4,2), stones spread around it.
        place(&mut pos, 4, 2, Piece::new(Player::P1, PieceKind::Keystone, 2));
        place(&mut pos, 3, 1, Piece::new(Player::P1, PieceKind::Stone, 2));
        place(&mut pos, 5, 1, Piece::new(Player::P1, PieceKind::Stone, 2));
        place(&mut pos, 2, 2, Piece::new(Player::P1, PieceKind::Stone, 2));
        place(&mut pos, 6, 2, Piece::new(Player::P1, PieceKind::Stone, 2));
        // P2 pieces: keystone at (4,6), stones spread around it.
        place(&mut pos, 4, 6, Piece::new(Player::P2, PieceKind::Keystone, 2));
        place(&mut pos, 3, 7, Piece::new(Player::P2, PieceKind::Stone, 2));
        place(&mut pos, 5, 7, Piece::new(Player::P2, PieceKind::Stone, 2));
        place(&mut pos, 2, 6, Piece::new(Player::P2, PieceKind::Stone, 2));
        place(&mut pos, 6, 6, Piece::new(Player::P2, PieceKind::Stone, 2));
        // Vary ply so Zobrist differs per game, breaking potential repetition.
        pos.ply = seed_offset as u32 % 7;
        pos.recompute_zobrist();
        game_from_pos(pos)
    }

    /// MCTS beats a random opponent over a short seeded match, alternating sides
    /// to cancel any first-player bias.
    ///
    /// The match uses a reduced-piece starting position so games reach a terminal
    /// state in tens of plies instead of hundreds, making rollouts informative
    /// even with a small cap and keeping the total test time under 30 seconds.
    #[test]
    fn mcts_beats_random_over_a_short_match() {
        let mut mcts_wins = 0u32;
        let mut random_wins = 0u32;
        let mut decisive = 0u32;

        for g in 0..MATCH_GAMES {
            let mcts_is_p1 = g % 2 == 0;
            let mut game = match_start_game(g);
            let mut mcts = MctsPolicy::with_params(
                MATCH_ITERS,
                DEFAULT_EXPLORATION,
                MATCH_ROLLOUT_CAP,
                1000 + g as u64,
            );
            let mut random = RandomPolicy::seeded(5000 + g as u64);

            while game.terminal_result().is_none() {
                let mover_is_mcts = (game.pos.to_move == Player::P1) == mcts_is_p1;
                let action = if mover_is_mcts {
                    mcts.choose(&game)
                } else {
                    random.choose(&game)
                };
                match action {
                    Some(a) => {
                        let _ = game.apply(a);
                    }
                    None => break,
                }
            }

            if let Some(GameResult::Win(winner)) = game.terminal_result() {
                decisive += 1;
                let mcts_player = if mcts_is_p1 { Player::P1 } else { Player::P2 };
                if winner == mcts_player {
                    mcts_wins += 1;
                } else {
                    random_wins += 1;
                }
            }
        }

        // Require at least some decisive games so the win-rate assertion is
        // meaningful. If zero games were decisive, MATCH_MAX_PLIES is too small.
        assert!(
            decisive >= 2,
            "only {decisive}/{MATCH_GAMES} games were decisive; raise MATCH_MAX_PLIES"
        );

        // Require MCTS to win at least MIN_WIN_RATE of decisive games.
        // Games that ended as max-ply draws are excluded from the denominator.
        let win_rate = mcts_wins as f64 / decisive as f64;
        assert!(
            win_rate > MIN_WIN_RATE,
            "MCTS win rate {win_rate:.2} ({mcts_wins}/{decisive} decisive, {random_wins} random wins) must exceed {MIN_WIN_RATE}"
        );
    }

    /// `name` is the stable identifier `"mcts"`.
    #[test]
    fn mcts_name_is_mcts() {
        assert_eq!(MctsPolicy::new(1, 0).name(), "mcts");
    }
}
