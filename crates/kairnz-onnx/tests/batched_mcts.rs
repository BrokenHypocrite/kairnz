//! Integration tests for `BatchedAzMcts`: validity of the visit distribution,
//! tactical correctness under virtual-loss batching (mate-in-one), and
//! equivalence to the sequential `AzMcts` engine when batching is disabled.

use std::path::PathBuf;

use kairnz_core::actions::{legal_actions, Action};
use kairnz_core::config::RuleConfig;
use kairnz_core::game::Game;
use kairnz_core::piece::{Piece, PieceKind, Player};
use kairnz_core::position::{Position, TurnState};
use kairnz_core::square::{BitBoard81, Sq, NUM_SQUARES};
use kairnz_onnx::mcts::AzMcts;
use kairnz_onnx::{AzMctsConfig, BatchedAzMcts, DirectBatchEvaluator, OnnxEvaluator};

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/random_init.onnx")
}

fn fixture_evaluator() -> OnnxEvaluator {
    OnnxEvaluator::from_path(&fixture_path()).expect("fixture loads")
}

fn direct_evaluator() -> DirectBatchEvaluator {
    DirectBatchEvaluator::new(fixture_evaluator())
}

fn sq(file: u8, rank: u8) -> Sq {
    Sq::new(file, rank).expect("in bounds")
}

fn place(pos: &mut Position, file: u8, rank: u8, piece: Piece) {
    pos.board[sq(file, rank).0 as usize] = Some(piece);
}

/// An empty-board minimal position, mirroring the `AzMcts` test helper.
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

/// The exact mate-in-one position from `mcts.rs`'s
/// `policy_prefers_an_immediate_winning_capture` test: a P1 stone one move from
/// capturing P2's only keystone, which ends the game as a P1 win.
fn mate_in_one() -> (Game, Action) {
    let mut pos = minimal_pos(Player::P1, 2);
    place(&mut pos, 4, 3, Piece::new(Player::P1, PieceKind::Stone, 2));
    place(&mut pos, 4, 4, Piece::new(Player::P2, PieceKind::Keystone, 1));
    place(&mut pos, 0, 0, Piece::new(Player::P1, PieceKind::Keystone, 1));
    place(&mut pos, 0, 8, Piece::new(Player::P1, PieceKind::Stone, 1));
    pos.recompute_zobrist();

    let winning = Action::Move { from: sq(4, 3), to: sq(4, 4) };
    let game = game_from_pos(pos);
    assert!(legal_actions(&game.pos).contains(&winning), "winning move is legal");
    (game, winning)
}

#[test]
fn batched_search_returns_valid_visit_distribution() {
    let game = Game::new_standard(RuleConfig::default());
    let legal = legal_actions(&game.pos);
    let eval = direct_evaluator();
    let config = AzMctsConfig { simulations: 64, leaves_per_step: 8, ..AzMctsConfig::default() };
    let mut mcts = BatchedAzMcts::new(&eval, config, 1);

    let result = mcts.search(&game).expect("search succeeds");

    assert_eq!(result.len(), legal.len(), "one root child per legal action");
    for (action, _visits) in &result {
        assert!(legal.contains(action), "every searched action is legal");
    }
    let total: u32 = result.iter().map(|(_, v)| *v).sum();
    // Each non-root simulation records one visit at the root's children; the
    // root's own expansion visit does not, so children sum to simulations - 1.
    assert_eq!(
        total,
        config.simulations - 1,
        "child visits sum to simulations minus the root expansion visit"
    );
}

#[test]
fn batched_search_finds_mate_in_one() {
    let (game, winning) = mate_in_one();
    let eval = direct_evaluator();
    // Enough simulations for the terminal win signal to dominate the random net.
    let config = AzMctsConfig { simulations: 256, leaves_per_step: 8, ..AzMctsConfig::default() };
    let mut mcts = BatchedAzMcts::new(&eval, config, 7);

    let result = mcts.search(&game).expect("search succeeds");
    let best = result
        .into_iter()
        .max_by_key(|(_, visits)| *visits)
        .map(|(action, _)| action);

    assert_eq!(best, Some(winning), "batched search must take the winning capture");
}

#[test]
fn leaves_per_step_one_matches_sequential_within_tolerance() {
    let game = Game::new_standard(RuleConfig::default());

    // Sequential reference (owns its own evaluator).
    let mut sequential = AzMcts::new(fixture_evaluator(), small_seq_config(), 1);
    let mut seq_result = sequential.search(&game);

    // Batched with batching disabled and no virtual loss should reduce exactly
    // to the sequential PUCT engine.
    let eval = direct_evaluator();
    let batched_config = AzMctsConfig {
        simulations: 64,
        leaves_per_step: 1,
        virtual_loss: 0.0,
        ..AzMctsConfig::default()
    };
    let mut batched = BatchedAzMcts::new(&eval, batched_config, 1);
    let mut batch_result = batched.search(&game).expect("search succeeds");

    seq_result.sort_by_key(|(a, _)| format!("{a:?}"));
    batch_result.sort_by_key(|(a, _)| format!("{a:?}"));

    // Same set of root actions.
    let seq_actions: Vec<Action> = seq_result.iter().map(|(a, _)| *a).collect();
    let batch_actions: Vec<Action> = batch_result.iter().map(|(a, _)| *a).collect();
    assert_eq!(seq_actions, batch_actions, "same root actions in the same order");

    // Top move identical.
    let seq_top = seq_result.iter().max_by_key(|(_, v)| *v).map(|(a, _)| *a);
    let batch_top = batch_result.iter().max_by_key(|(_, v)| *v).map(|(a, _)| *a);
    assert_eq!(seq_top, batch_top, "top (most-visited) move matches the sequential engine");

    // Visit distributions match. With batching disabled and no virtual loss the
    // batched engine reduces exactly to the sequential PUCT, so counts are
    // identical; a tolerance of 1 keeps the bridge robust without weakening it.
    for ((a, sv), (_, bv)) in seq_result.iter().zip(batch_result.iter()) {
        let diff = (*sv as i64 - *bv as i64).abs();
        assert!(
            diff <= 1,
            "visit count for {a:?} differs by {diff} (seq={sv}, batched={bv})"
        );
    }
}

fn small_seq_config() -> AzMctsConfig {
    AzMctsConfig { simulations: 64, ..AzMctsConfig::default() }
}
