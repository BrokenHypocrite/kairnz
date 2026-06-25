//! Virtual-loss PUCT MCTS that batches K leaf evaluations per search step.
//!
//! This mirrors [`crate::mcts::AzMcts`] exactly (tree/node layout, PUCT
//! selection, masked-and-renormalized priors, leaf-perspective backup, and the
//! value sign convention) and adds virtual loss so that several leaves can be
//! selected before any of them is evaluated. The collected leaves are then sent
//! to the network in a single [`BatchEvaluator::evaluate_batch`] call, which is
//! where the throughput win comes from.

use kairnz_core::actions::{legal_actions, Action};
use kairnz_core::game::Game;
use kairnz_core::piece::Player;
use kairnz_encode::encode_planes;
use rand::SeedableRng;
use rand_distr::Distribution;
use rand_pcg::Pcg64;

use crate::batch::BatchEvaluator;
use crate::mcts::{legal_priors, terminal_value, AzMctsConfig};

/// A node in the batched search tree, stored in a flat arena addressed by
/// `usize`.
///
/// VALUE PERSPECTIVE: `value_sum` accumulates leaf values from the perspective
/// of this node's `to_move`, identical to [`crate::mcts::AzMcts`]. `prior` is
/// this node's PUCT prior as a child of its parent. `vl` counts in-flight
/// virtual visits applied during selection and removed during backup.
struct Node {
    game: Game,
    to_move: Player,
    action_from_parent: Option<Action>,
    prior: f32,
    children: Vec<usize>,
    expanded: bool,
    visits: u32,
    value_sum: f64,
    vl: u32,
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
            vl: 0,
        }
    }
}

/// One non-terminal leaf collected during a batched step, awaiting evaluation.
struct PendingLeaf {
    leaf: usize,
    path: Vec<usize>,
    planes: Vec<f32>,
    rep: u8,
}

/// A neural-guided PUCT search that batches leaf evaluations using virtual loss.
///
/// Borrows a [`BatchEvaluator`] (so the same network session can be shared) and
/// owns a seeded RNG used only for root Dirichlet noise.
pub struct BatchedAzMcts<'a> {
    evaluator: &'a dyn BatchEvaluator,
    config: AzMctsConfig,
    rng: Pcg64,
}

impl<'a> BatchedAzMcts<'a> {
    /// Builds a batched search over `eval`, seeded for reproducible root noise.
    pub fn new(eval: &'a dyn BatchEvaluator, config: AzMctsConfig, seed: u64) -> BatchedAzMcts<'a> {
        BatchedAzMcts { evaluator: eval, config, rng: Pcg64::seed_from_u64(seed) }
    }

    /// Runs the configured number of simulations from `game` and returns each
    /// root child's `(action, visit_count)`. Empty if the root is terminal or
    /// has no legal action. Propagates `evaluate_batch` errors.
    pub fn search(&mut self, game: &Game) -> ort::Result<Vec<(Action, u32)>> {
        if game.terminal_result().is_some() {
            return Ok(Vec::new());
        }
        let mut arena: Vec<Node> = vec![Node::new(game.clone(), None, 0.0)];

        // Expand and evaluate the root first so root priors (with Dirichlet
        // noise) exist before any selection, mirroring how AzMcts applies noise
        // when it first expands leaf 0.
        self.expand_root(&mut arena)?;

        while arena[0].visits < self.config.simulations {
            let want =
                self.config.leaves_per_step.min((self.config.simulations - arena[0].visits) as usize);
            if want == 0 {
                break;
            }

            let mut pending: Vec<PendingLeaf> = Vec::with_capacity(want);
            for _ in 0..want {
                let (leaf, path) = self.select_leaf(&mut arena);
                if let Some(result) = arena[leaf].game.terminal_result() {
                    // Terminal leaf: known value, backup immediately (removes vl).
                    let value = terminal_value(arena[leaf].to_move, result);
                    self.backup(&mut arena, &path, value);
                } else {
                    // Non-terminal leaf: record for the batched evaluation; its
                    // virtual loss stays applied until backup after the eval.
                    let rep = arena[leaf].game.repetition_count();
                    let planes = encode_planes(&arena[leaf].game.pos, rep);
                    pending.push(PendingLeaf { leaf, path, planes, rep });
                }
            }

            if pending.is_empty() {
                continue;
            }

            let planes_vec: Vec<Vec<f32>> = pending.iter().map(|p| p.planes.clone()).collect();
            let reps_vec: Vec<u8> = pending.iter().map(|p| p.rep).collect();
            let results = self.evaluator.evaluate_batch(&planes_vec, &reps_vec)?;

            for (p, (policy, value)) in pending.into_iter().zip(results.into_iter()) {
                // Collisions: a leaf may have been expanded by an earlier row in
                // this same step. Skip the re-expand but still backup so its
                // virtual loss is removed and the visit recorded.
                if !arena[p.leaf].expanded {
                    self.expand(&mut arena, p.leaf, &policy);
                }
                self.backup(&mut arena, &p.path, value as f64);
            }
        }

        Ok(arena[0]
            .children
            .iter()
            .map(|&c| {
                let child = &arena[c];
                (child.action_from_parent.expect("child has an action"), child.visits)
            })
            .collect())
    }

    /// Expands the root via a single-element batch, applies Dirichlet noise,
    /// and records the root's first visit and value -- matching the sequential
    /// `AzMcts` semantics, where leaf 0 is expanded and backed up inside the
    /// first simulation. Recording that visit here keeps the loop's `visits`
    /// accounting in lockstep with `AzMcts`.
    fn expand_root(&mut self, arena: &mut Vec<Node>) -> ort::Result<()> {
        let rep = arena[0].game.repetition_count();
        let planes = encode_planes(&arena[0].game.pos, rep);
        let results = self.evaluator.evaluate_batch(&[planes], &[rep])?;
        let (policy, value) = results.into_iter().next().expect("one row in, one row out");
        self.expand(arena, 0, &policy);
        // Record the root's own first visit/value exactly as AzMcts does when it
        // expands leaf 0 inside the first simulation.
        arena[0].visits += 1;
        arena[0].value_sum += value as f64;
        Ok(())
    }

    /// Walks from the root choosing the PUCT-best child using virtual-loss
    /// adjusted statistics, incrementing each visited node's `vl`, and stops at
    /// the first unexpanded node or a terminal node. Returns `(leaf, path)`.
    fn select_leaf(&self, arena: &mut [Node]) -> (usize, Vec<usize>) {
        let mut path: Vec<usize> = vec![0];
        let mut current = 0usize;
        arena[0].vl += 1;
        while arena[current].expanded && arena[current].game.terminal_result().is_none() {
            match self.best_child(arena, current) {
                Some(child) => {
                    current = child;
                    arena[current].vl += 1;
                    path.push(current);
                }
                None => break,
            }
        }
        (current, path)
    }

    /// Creates a child per legal action of `leaf` with its masked, renormalized
    /// softmax prior (root noise mixed in for leaf 0) and marks `leaf` expanded.
    /// Mirrors `AzMcts::expand_and_evaluate`'s child creation exactly.
    fn expand(&mut self, arena: &mut Vec<Node>, leaf: usize, policy: &[f32]) {
        let to_move = arena[leaf].to_move;
        let legal = legal_actions(&arena[leaf].game.pos);
        if legal.is_empty() {
            arena[leaf].expanded = true;
            return;
        }

        let mut priors = legal_priors(policy, &legal, to_move);
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
    }

    /// Adds one real visit and the leaf `value` (in leaf perspective, signed per
    /// node's `to_move`) to every node on `path`, and removes the virtual loss
    /// applied during selection by decrementing each node's `vl`.
    fn backup(&self, arena: &mut [Node], path: &[usize], value: f64) {
        let leaf = *path.last().expect("path is non-empty");
        let leaf_to_move = arena[leaf].to_move;
        for &idx in path {
            arena[idx].visits += 1;
            let signed = if arena[idx].to_move == leaf_to_move { value } else { -value };
            arena[idx].value_sum += signed;
            arena[idx].vl = arena[idx].vl.saturating_sub(1);
        }
    }

    /// Mixes symmetric Dirichlet noise into root priors: `p = (1-eps)*p + eps*noise`.
    /// Identical to `AzMcts::apply_root_noise`.
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

    /// Returns the PUCT-best child of `parent` using virtual-loss adjusted
    /// statistics, or `None` if it has no children.
    ///
    /// For each child the effective visit count is `N_eff = N + vl` and the
    /// effective mean is `Q_eff = (W_parent - vl*virtual_loss) / max(N_eff, 1)`,
    /// where `W_parent` is the child's accumulated value read in the PARENT's
    /// perspective. The virtual-loss penalty is subtracted in the parent's
    /// perspective so each in-flight visit makes the child look like a
    /// parent-side loss, discouraging picking the same child again within a
    /// batch. `N_eff` is also the child's exploration denominator.
    ///
    /// The parent's exploration numerator uses real visits only
    /// (`sqrt(N_parent.max(1))`), exactly as the sequential engine does, so that
    /// with `leaves_per_step = 1` and `virtual_loss = 0` selection reduces to the
    /// sequential PUCT to within the `N_eff` child-denominator (which equals the
    /// sequential denominator whenever a node has no concurrent in-flight visit,
    /// i.e. always at `leaves_per_step = 1`).
    fn best_child(&self, arena: &[Node], parent: usize) -> Option<usize> {
        let parent_node = &arena[parent];
        let sqrt_total = (parent_node.visits.max(1) as f64).sqrt();
        let vl_weight = self.config.virtual_loss as f64;

        let mut best: Option<usize> = None;
        let mut best_score = f64::NEG_INFINITY;
        for &child_idx in &parent_node.children {
            let child = &arena[child_idx];
            let n_eff = child.visits + child.vl;
            // Child's accumulated value in the PARENT's perspective.
            let w_parent =
                if child.to_move == parent_node.to_move { child.value_sum } else { -child.value_sum };
            let q = if n_eff == 0 {
                0.0
            } else {
                (w_parent - child.vl as f64 * vl_weight) / n_eff as f64
            };
            let u = self.config.c_puct * child.prior as f64 * sqrt_total / (1.0 + n_eff as f64);
            let score = q + u;
            if score > best_score {
                best_score = score;
                best = Some(child_idx);
            }
        }
        best
    }
}
