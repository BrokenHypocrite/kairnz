//! Neural-guided PUCT Monte Carlo Tree Search over Kairnz positions.

use std::path::Path;

use kairnz_core::actions::{legal_actions, Action};
use kairnz_core::game::Game;
use kairnz_core::outcome::GameResult;
use kairnz_core::piece::Player;
use kairnz_encode::action_to_index;
use kairnz_policy::policy::Policy;
use rand::SeedableRng;
use rand_distr::Distribution;
use rand_pcg::Pcg64;

use crate::OnnxEvaluator;

/// Default number of simulations per move.
const DEFAULT_SIMULATIONS: u32 = 400;
/// Default PUCT exploration constant.
const DEFAULT_C_PUCT: f64 = 1.5;
/// Default Dirichlet concentration for root exploration noise.
const DEFAULT_DIRICHLET_ALPHA: f64 = 0.3;
/// Default root-noise weight. Zero disables noise, making search deterministic.
const DEFAULT_DIRICHLET_EPSILON: f64 = 0.0;

/// Terminal value of a win from the winning side's perspective.
const WIN_VALUE: f64 = 1.0;
/// Terminal value of a loss from the losing side's perspective.
const LOSS_VALUE: f64 = -1.0;
/// Terminal value of a draw.
const DRAW_VALUE: f64 = 0.0;

/// Search parameters for [`AzMctsPolicy`].
#[derive(Clone, Copy, Debug)]
pub struct AzMctsConfig {
    /// Number of simulations performed per move.
    pub simulations: u32,
    /// PUCT exploration constant.
    pub c_puct: f64,
    /// Dirichlet concentration parameter for root noise.
    pub dirichlet_alpha: f64,
    /// Root-noise mixing weight in `[0, 1]`; `0.0` disables noise.
    pub dirichlet_epsilon: f64,
}

impl Default for AzMctsConfig {
    fn default() -> Self {
        AzMctsConfig {
            simulations: DEFAULT_SIMULATIONS,
            c_puct: DEFAULT_C_PUCT,
            dirichlet_alpha: DEFAULT_DIRICHLET_ALPHA,
            dirichlet_epsilon: DEFAULT_DIRICHLET_EPSILON,
        }
    }
}

/// Terminal value of `result` from `to_move`'s perspective, in `[-1, 1]`.
pub(crate) fn terminal_value(to_move: Player, result: GameResult) -> f64 {
    match result {
        GameResult::Win(winner) if winner == to_move => WIN_VALUE,
        GameResult::Win(_) => LOSS_VALUE,
        GameResult::Draw(_) => DRAW_VALUE,
    }
}

/// Softmax priors over only the legal actions, aligned to `legal`'s order.
///
/// Each legal action's logit is read from the policy vector via
/// `action_to_index`, then a numerically stable softmax is applied. The result
/// sums to approximately 1 and is used as the PUCT prior for each child.
pub(crate) fn legal_priors(logits: &[f32], legal: &[Action], to_move: Player) -> Vec<f32> {
    if legal.is_empty() {
        return Vec::new();
    }
    let raw: Vec<f32> = legal
        .iter()
        .map(|a| logits[action_to_index(a, to_move)])
        .collect();
    let max = raw.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = raw.iter().map(|x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|e| e / sum).collect()
}

/// A node in the search tree, stored in a flat arena addressed by `usize`.
///
/// VALUE PERSPECTIVE: `value_sum` accumulates leaf values from the perspective of
/// this node's `to_move` (see the crate's value convention). `prior` is this
/// node's PUCT prior as a child of its parent.
struct Node {
    game: Game,
    to_move: Player,
    action_from_parent: Option<Action>,
    prior: f32,
    children: Vec<usize>,
    expanded: bool,
    visits: u32,
    value_sum: f64,
}

impl Node {
    fn new(game: Game, action_from_parent: Option<Action>, prior: f32) -> Node {
        let to_move = game.pos.to_move;
        Node {
            game,
            to_move,
            action_from_parent,
            prior,
            children: Vec::new(),
            expanded: false,
            visits: 0,
            value_sum: 0.0,
        }
    }
}

/// A neural-guided PUCT search. Owns the model evaluator and a seeded RNG used
/// only for root Dirichlet noise (inert when `dirichlet_epsilon` is 0).
pub struct AzMcts {
    evaluator: OnnxEvaluator,
    config: AzMctsConfig,
    rng: Pcg64,
}

impl AzMcts {
    /// Builds a search owning `evaluator`, seeded for reproducible root noise.
    pub fn new(evaluator: OnnxEvaluator, config: AzMctsConfig, seed: u64) -> AzMcts {
        AzMcts { evaluator, config, rng: Pcg64::seed_from_u64(seed) }
    }

    /// Runs the configured number of simulations from `game` and returns each
    /// root child's `(action, visit_count)`. Empty if the root is terminal or has
    /// no legal action.
    pub fn search(&mut self, game: &Game) -> Vec<(Action, u32)> {
        if game.terminal_result().is_some() {
            return Vec::new();
        }
        let mut arena: Vec<Node> = vec![Node::new(game.clone(), None, 0.0)];

        for _ in 0..self.config.simulations {
            self.simulate(&mut arena);
        }

        arena[0]
            .children
            .iter()
            .map(|&c| {
                let child = &arena[c];
                (child.action_from_parent.expect("child has an action"), child.visits)
            })
            .collect()
    }

    /// One selection -> evaluation/expansion -> backpropagation cycle.
    fn simulate(&mut self, arena: &mut Vec<Node>) {
        // Selection: descend by PUCT through expanded, non-terminal nodes.
        let mut path: Vec<usize> = vec![0];
        let mut current = 0usize;
        while arena[current].expanded && arena[current].game.terminal_result().is_none() {
            match self.best_child(arena, current) {
                Some(child) => {
                    current = child;
                    path.push(current);
                }
                None => break,
            }
        }

        // Evaluation: terminal leaves use the true result; others query the net
        // and expand all legal children with their priors.
        let leaf = current;
        let value = if let Some(result) = arena[leaf].game.terminal_result() {
            terminal_value(arena[leaf].to_move, result)
        } else {
            self.expand_and_evaluate(arena, leaf)
        };

        // Backpropagation in the leaf's perspective.
        let leaf_to_move = arena[leaf].to_move;
        for &idx in &path {
            arena[idx].visits += 1;
            let signed = if arena[idx].to_move == leaf_to_move { value } else { -value };
            arena[idx].value_sum += signed;
        }
    }

    /// Evaluates `leaf` with the network, creates a child per legal action with
    /// its softmax prior (root noise mixed in when configured), marks the leaf
    /// expanded, and returns the leaf value in `leaf.to_move`'s perspective.
    fn expand_and_evaluate(&mut self, arena: &mut Vec<Node>, leaf: usize) -> f64 {
        let to_move = arena[leaf].to_move;
        let legal = legal_actions(&arena[leaf].game.pos);
        if legal.is_empty() {
            arena[leaf].expanded = true;
            return 0.0;
        }

        let rep = arena[leaf].game.repetition_count();
        let (logits, value) = match self.evaluator.evaluate(&arena[leaf].game.pos, rep) {
            Ok(out) => out,
            Err(error) => {
                // A failed evaluation is treated as a neutral leaf rather than a
                // panic; Plan 4 self-play surfaces inference errors explicitly.
                eprintln!("AzMcts evaluation failed: {error}");
                arena[leaf].expanded = true;
                return 0.0;
            }
        };

        let mut priors = legal_priors(&logits, &legal, to_move);
        if leaf == 0 && self.config.dirichlet_epsilon > 0.0 {
            self.apply_root_noise(&mut priors);
        }

        for (action, prior) in legal.iter().zip(priors.iter()) {
            let mut child_game = arena[leaf].game.clone();
            let _ = child_game.apply(*action);
            let child = Node::new(child_game, Some(*action), *prior);
            let child_idx = arena.len();
            arena.push(child);
            arena[leaf].children.push(child_idx);
        }
        arena[leaf].expanded = true;
        value as f64
    }

    /// Mixes symmetric Dirichlet noise into root priors: `p = (1-eps)*p + eps*noise`.
    ///
    /// The noise is built from Gamma(alpha, 1) samples normalized to sum to 1,
    /// which is exactly Dirichlet(alpha) and uses only the stable `Gamma` API
    /// (rand_distr's `Dirichlet` type has a version-sensitive signature).
    fn apply_root_noise(&mut self, priors: &mut [f32]) {
        let gamma = match rand_distr::Gamma::new(self.config.dirichlet_alpha, 1.0) {
            Ok(g) => g,
            Err(_) => return,
        };
        let mut noise: Vec<f32> = (0..priors.len())
            .map(|_| gamma.sample(&mut self.rng) as f32)
            .collect();
        let sum: f32 = noise.iter().sum();
        if sum <= 0.0 {
            return;
        }
        for n in noise.iter_mut() {
            *n /= sum;
        }
        let eps = self.config.dirichlet_epsilon as f32;
        for (p, n) in priors.iter_mut().zip(noise.iter()) {
            *p = (1.0 - eps) * *p + eps * *n;
        }
    }

    /// Returns the PUCT-best child of `parent`, or `None` if it has no children.
    fn best_child(&self, arena: &[Node], parent: usize) -> Option<usize> {
        let parent_node = &arena[parent];
        let sqrt_total = (parent_node.visits.max(1) as f64).sqrt();

        let mut best: Option<usize> = None;
        let mut best_score = f64::NEG_INFINITY;
        for &child_idx in &parent_node.children {
            let child = &arena[child_idx];
            let q_child = if child.visits == 0 {
                0.0
            } else {
                child.value_sum / child.visits as f64
            };
            // Read the child's mean in the PARENT's perspective.
            let q = if child.to_move == parent_node.to_move { q_child } else { -q_child };
            let u = self.config.c_puct * child.prior as f64 * sqrt_total / (1.0 + child.visits as f64);
            let score = q + u;
            if score > best_score {
                best_score = score;
                best = Some(child_idx);
            }
        }
        best
    }
}

/// A `Policy` that plays the most-visited move from a neural PUCT search.
pub struct AzMctsPolicy {
    search: AzMcts,
}

impl AzMctsPolicy {
    /// Builds a policy owning `evaluator`.
    pub fn new(evaluator: OnnxEvaluator, config: AzMctsConfig, seed: u64) -> AzMctsPolicy {
        AzMctsPolicy { search: AzMcts::new(evaluator, config, seed) }
    }

    /// Loads a model from `path` and builds a policy.
    pub fn from_path(path: &Path, config: AzMctsConfig, seed: u64) -> ort::Result<AzMctsPolicy> {
        Ok(AzMctsPolicy::new(OnnxEvaluator::from_path(path)?, config, seed))
    }
}

impl Policy for AzMctsPolicy {
    fn choose(&mut self, game: &Game) -> Option<Action> {
        self.search
            .search(game)
            .into_iter()
            .max_by_key(|(_, visits)| *visits)
            .map(|(action, _)| action)
    }

    fn name(&self) -> &str {
        "az-mcts"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kairnz_core::config::RuleConfig;
    use kairnz_core::game::Game;
    use kairnz_core::piece::{Piece, PieceKind, Player};
    use kairnz_core::position::{Position, TurnState};
    use kairnz_core::square::{BitBoard81, NUM_SQUARES, Sq};
    use kairnz_encode::POLICY_SIZE;
    use std::path::PathBuf;

    use crate::OnnxEvaluator;

    #[test]
    fn terminal_value_is_perspective_relative() {
        assert_eq!(terminal_value(Player::P1, GameResult::Win(Player::P1)), 1.0);
        assert_eq!(terminal_value(Player::P1, GameResult::Win(Player::P2)), -1.0);
        assert_eq!(
            terminal_value(Player::P1, GameResult::Draw(kairnz_core::outcome::DrawReason::MaxPlies)),
            0.0
        );
    }

    #[test]
    fn legal_priors_softmax_sums_to_one_over_legal() {
        let mut logits = vec![0.0f32; POLICY_SIZE];
        let a = Action::Place { to: Sq(0) };
        let b = Action::Place { to: Sq(1) };
        logits[action_to_index(&a, Player::P1)] = 2.0;
        logits[action_to_index(&b, Player::P1)] = 0.0;

        let priors = legal_priors(&logits, &[a, b], Player::P1);
        assert_eq!(priors.len(), 2);
        let sum: f32 = priors.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "priors sum to one");
        assert!(priors[0] > priors[1], "higher logit gets higher prior");
    }

    #[test]
    fn legal_priors_empty_for_no_actions() {
        assert!(legal_priors(&[0.0; POLICY_SIZE], &[], Player::P1).is_empty());
    }

    fn fixture_evaluator() -> OnnxEvaluator {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/random_init.onnx");
        OnnxEvaluator::from_path(&path).expect("fixture loads")
    }

    fn small_config() -> AzMctsConfig {
        AzMctsConfig { simulations: 64, ..AzMctsConfig::default() }
    }

    #[test]
    fn search_returns_visits_over_legal_actions() {
        let game = Game::new_standard(RuleConfig::default());
        let legal = kairnz_core::actions::legal_actions(&game.pos);
        let mut mcts = AzMcts::new(fixture_evaluator(), small_config(), 1);

        let result = mcts.search(&game);
        assert_eq!(result.len(), legal.len(), "one root child per legal action");
        for (action, _visits) in &result {
            assert!(legal.contains(action), "every searched action is legal");
        }
        let total: u32 = result.iter().map(|(_, v)| *v).sum();
        assert!(total > 0, "simulations recorded visits");
    }

    #[test]
    fn search_is_deterministic_without_root_noise() {
        let game = Game::new_standard(RuleConfig::default());
        let mut a = AzMcts::new(fixture_evaluator(), small_config(), 1);
        let mut b = AzMcts::new(fixture_evaluator(), small_config(), 2);
        // dirichlet_epsilon is 0, so the seed is irrelevant: identical results.
        assert_eq!(a.search(&game), b.search(&game), "epsilon 0 search is deterministic");
    }

    fn sq(file: u8, rank: u8) -> kairnz_core::square::Sq {
        kairnz_core::square::Sq::new(file, rank).expect("in bounds")
    }

    fn place(pos: &mut Position, file: u8, rank: u8, piece: Piece) {
        pos.board[sq(file, rank).0 as usize] = Some(piece);
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

    fn game_from_pos(pos: Position) -> Game {
        let mut game = Game::new_standard(RuleConfig::default());
        game.pos = pos;
        game
    }

    #[test]
    fn policy_chooses_a_legal_action_at_opening() {
        let game = Game::new_standard(RuleConfig::default());
        let legal = kairnz_core::actions::legal_actions(&game.pos);
        let mut policy = AzMctsPolicy::new(fixture_evaluator(), small_config(), 1);
        let action = policy.choose(&game).expect("opening has a move");
        assert!(legal.contains(&action));
    }

    #[test]
    fn policy_name_is_az_mcts() {
        let policy = AzMctsPolicy::new(fixture_evaluator(), small_config(), 0);
        assert_eq!(policy.name(), "az-mcts");
    }

    /// Sign-convention guard: with a winning keystone capture available, the
    /// search must choose it. The capture creates a terminal child whose true
    /// value (+1 for the capturing side) must back up to make that move the most
    /// visited, even though the fixture network is random. A backprop or PUCT
    /// sign error would steer the search away from the win.
    #[test]
    fn policy_prefers_an_immediate_winning_capture() {
        let mut pos = minimal_pos(Player::P1, 2);
        place(&mut pos, 4, 3, Piece::new(Player::P1, PieceKind::Stone, 2));
        place(&mut pos, 4, 4, Piece::new(Player::P2, PieceKind::Keystone, 1));
        place(&mut pos, 0, 0, Piece::new(Player::P1, PieceKind::Keystone, 1));
        place(&mut pos, 0, 8, Piece::new(Player::P1, PieceKind::Stone, 1));
        pos.recompute_zobrist();

        let winning = Action::Move { from: sq(4, 3), to: sq(4, 4) };
        let game = game_from_pos(pos);
        assert!(kairnz_core::actions::legal_actions(&game.pos).contains(&winning));

        // Enough simulations for the terminal win signal to dominate the random net.
        let config = AzMctsConfig { simulations: 256, ..AzMctsConfig::default() };
        let mut policy = AzMctsPolicy::new(fixture_evaluator(), config, 7);
        assert_eq!(policy.choose(&game), Some(winning), "must take the winning capture");
    }
}
