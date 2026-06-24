use std::collections::BTreeMap;

use kairnz_core::{
    outcome::GameResult,
    piece::Player,
};

use crate::runner::GameRecord;

/// Width of each ply-count bucket in the histogram.
const PLY_BUCKET: u32 = 25;

/// Aggregated balance metrics derived from a slice of [`GameRecord`]s.
///
/// All rates are in [0.0, 1.0]. Division-by-zero is guarded: if the relevant
/// denominator is zero the rate is 0.0.
#[derive(Debug, Clone, serde::Serialize)]
pub struct Metrics {
    /// Total number of games in the sample.
    pub games: usize,
    /// Fraction of games won by P1.
    pub p1_win_rate: f64,
    /// Fraction of games won by P2.
    pub p2_win_rate: f64,
    /// Fraction of games that ended in a draw.
    pub draw_rate: f64,
    /// Median ply count across all games. Average of the two middle values for
    /// even-sized samples.
    pub ply_median: f64,
    /// Distribution of ply counts bucketed by [`PLY_BUCKET`]-wide intervals.
    /// Keys are the bucket start (inclusive); only non-empty buckets are present.
    pub ply_histogram: BTreeMap<u32, u32>,
    /// Among decisive games where a first capture is recorded: the fraction
    /// where the winner was also the first to capture.
    pub snowball_rate: f64,
    /// Among decisive games where a first Keystone loss is recorded: the
    /// fraction where the player who FIRST lost a Keystone still won.
    pub comeback_rate: f64,
    /// Mean of `max_stack_height` across all games.
    pub avg_max_stack: f64,
}

/// Returns `Some(player)` for a `Win` result, `None` for any draw.
fn winner(result: &GameResult) -> Option<Player> {
    match result {
        GameResult::Win(p) => Some(*p),
        GameResult::Draw(_) => None,
    }
}

/// Aggregates a slice of [`GameRecord`]s into the six section-8 balance metrics.
///
/// If `records` is empty every rate and average is 0.0 and the histogram is
/// empty; no division by zero occurs.
pub fn aggregate(records: &[GameRecord]) -> Metrics {
    let n = records.len();

    if n == 0 {
        return Metrics {
            games: 0,
            p1_win_rate: 0.0,
            p2_win_rate: 0.0,
            draw_rate: 0.0,
            ply_median: 0.0,
            ply_histogram: BTreeMap::new(),
            snowball_rate: 0.0,
            comeback_rate: 0.0,
            avg_max_stack: 0.0,
        };
    }

    let nf = n as f64;

    // Win/draw counts.
    let p1_wins = records.iter().filter(|r| r.result == GameResult::Win(Player::P1)).count();
    let p2_wins = records.iter().filter(|r| r.result == GameResult::Win(Player::P2)).count();
    let draws = records.iter().filter(|r| matches!(r.result, GameResult::Draw(_))).count();

    // Ply median.
    let mut plies: Vec<u32> = records.iter().map(|r| r.plies).collect();
    plies.sort_unstable();
    let ply_median = if plies.len() % 2 == 1 {
        plies[plies.len() / 2] as f64
    } else {
        let mid = plies.len() / 2;
        (plies[mid - 1] as f64 + plies[mid] as f64) / 2.0
    };

    // Ply histogram.
    let mut ply_histogram: BTreeMap<u32, u32> = BTreeMap::new();
    for &p in &plies {
        let bucket = (p / PLY_BUCKET) * PLY_BUCKET;
        *ply_histogram.entry(bucket).or_insert(0) += 1;
    }

    // Snowball rate: among decisive games WITH a first_capture_by signal,
    // the fraction where winner == first_capture_by.
    let snowball_denom: usize = records
        .iter()
        .filter(|r| winner(&r.result).is_some() && r.first_capture_by.is_some())
        .count();
    let snowball_num: usize = records
        .iter()
        .filter(|r| {
            if let (Some(w), Some(fc)) = (winner(&r.result), r.first_capture_by) {
                w == fc
            } else {
                false
            }
        })
        .count();
    let snowball_rate = if snowball_denom == 0 {
        0.0
    } else {
        snowball_num as f64 / snowball_denom as f64
    };

    // Comeback rate: among decisive games WITH a first_keystone_loss_by signal,
    // the fraction where winner == first_keystone_loss_by (loser of first
    // Keystone still won the game).
    let comeback_denom: usize = records
        .iter()
        .filter(|r| winner(&r.result).is_some() && r.first_keystone_loss_by.is_some())
        .count();
    let comeback_num: usize = records
        .iter()
        .filter(|r| {
            if let (Some(w), Some(kl)) = (winner(&r.result), r.first_keystone_loss_by) {
                w == kl
            } else {
                false
            }
        })
        .count();
    let comeback_rate = if comeback_denom == 0 {
        0.0
    } else {
        comeback_num as f64 / comeback_denom as f64
    };

    // Average max stack height.
    let avg_max_stack: f64 =
        records.iter().map(|r| r.max_stack_height as f64).sum::<f64>() / nf;

    Metrics {
        games: n,
        p1_win_rate: p1_wins as f64 / nf,
        p2_win_rate: p2_wins as f64 / nf,
        draw_rate: draws as f64 / nf,
        ply_median,
        ply_histogram,
        snowball_rate,
        comeback_rate,
        avg_max_stack,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use kairnz_core::outcome::{DrawReason, GameResult};
    use kairnz_core::piece::Player;

    /// Constructs a minimal GameRecord with the given result and plies.
    fn rec(result: GameResult, plies: u32) -> GameRecord {
        GameRecord {
            result,
            plies,
            first_capture_by: None,
            first_keystone_loss_by: None,
            max_stack_height: 1,
        }
    }

    fn win1(plies: u32) -> GameRecord { rec(GameResult::Win(Player::P1), plies) }
    fn win2(plies: u32) -> GameRecord { rec(GameResult::Win(Player::P2), plies) }
    fn draw(plies: u32) -> GameRecord { rec(GameResult::Draw(DrawReason::MaxPlies), plies) }

    #[test]
    fn win_rate_by_side_counts_correctly() {
        // 2 x P1 win, 1 x P2 win, 1 x draw => N=4
        let records = vec![win1(10), win1(20), win2(30), draw(40)];
        let m = aggregate(&records);
        assert_eq!(m.games, 4);
        assert!((m.p1_win_rate - 0.5).abs() < f64::EPSILON, "p1_win_rate should be 0.5");
        assert!((m.p2_win_rate - 0.25).abs() < f64::EPSILON, "p2_win_rate should be 0.25");
        assert!((m.draw_rate - 0.25).abs() < f64::EPSILON, "draw_rate should be 0.25");
        // Rates must sum to 1.0.
        let sum = m.p1_win_rate + m.p2_win_rate + m.draw_rate;
        assert!((sum - 1.0).abs() < 1e-10, "rates sum {sum} != 1.0");
    }

    #[test]
    fn ply_median_handles_even_and_odd_counts() {
        // Odd count: [10, 20, 30] => median = 20.
        let odd = vec![win1(30), win1(10), win1(20)];
        let m = aggregate(&odd);
        assert!((m.ply_median - 20.0).abs() < f64::EPSILON, "odd median should be 20.0");

        // Even count: [10, 20, 30, 40] => median = (20 + 30) / 2 = 25.
        let even = vec![win1(30), win1(10), win1(40), win1(20)];
        let m2 = aggregate(&even);
        assert!((m2.ply_median - 25.0).abs() < f64::EPSILON, "even median should be 25.0");
    }

    #[test]
    fn snowball_rate_is_first_capture_then_win_fraction() {
        // Snowball denominator = decisive games WITH first_capture_by.
        // Build:
        //   - P1 wins, first_capture P1 (snowball) => counted, numerator
        //   - P1 wins, first_capture P1 (snowball) => counted, numerator
        //   - P2 wins, first_capture P1 (no snowball) => counted, NOT numerator
        //   - P1 wins, no first_capture => EXCLUDED from denominator
        //   - draw, first_capture P1 => EXCLUDED (not decisive)
        // => denom = 3, num = 2, rate = 2/3.
        let make = |result: GameResult, first_capture: Option<Player>| GameRecord {
            result,
            plies: 10,
            first_capture_by: first_capture,
            first_keystone_loss_by: None,
            max_stack_height: 1,
        };

        let records = vec![
            make(GameResult::Win(Player::P1), Some(Player::P1)),
            make(GameResult::Win(Player::P1), Some(Player::P1)),
            make(GameResult::Win(Player::P2), Some(Player::P1)),
            make(GameResult::Win(Player::P1), None),
            make(GameResult::Draw(DrawReason::MaxPlies), Some(Player::P1)),
        ];
        let m = aggregate(&records);
        let expected = 2.0 / 3.0;
        assert!(
            (m.snowball_rate - expected).abs() < 1e-10,
            "snowball_rate {:.6} != {:.6}",
            m.snowball_rate,
            expected
        );
    }

    #[test]
    fn comeback_rate_is_lost_keystone_first_then_win_fraction() {
        // Comeback denominator = decisive games WITH first_keystone_loss_by.
        // Build:
        //   - P1 wins, first_keystone_loss P1 (comeback!) => counted, numerator
        //   - P2 wins, first_keystone_loss P1 (no comeback) => counted, NOT numerator
        //   - P1 wins, first_keystone_loss P1 (comeback!) => counted, numerator
        //   - P1 wins, no first_keystone_loss => EXCLUDED
        //   - draw, first_keystone_loss P2 => EXCLUDED (not decisive)
        // => denom = 3, num = 2, rate = 2/3.
        let make = |result: GameResult, first_kl: Option<Player>| GameRecord {
            result,
            plies: 10,
            first_capture_by: None,
            first_keystone_loss_by: first_kl,
            max_stack_height: 1,
        };

        let records = vec![
            make(GameResult::Win(Player::P1), Some(Player::P1)),
            make(GameResult::Win(Player::P2), Some(Player::P1)),
            make(GameResult::Win(Player::P1), Some(Player::P1)),
            make(GameResult::Win(Player::P1), None),
            make(GameResult::Draw(DrawReason::MaxPlies), Some(Player::P2)),
        ];
        let m = aggregate(&records);
        let expected = 2.0 / 3.0;
        assert!(
            (m.comeback_rate - expected).abs() < 1e-10,
            "comeback_rate {:.6} != {:.6}",
            m.comeback_rate,
            expected
        );
    }

    #[test]
    fn avg_max_stack_height_averages_records() {
        let make_height = |h: u8| GameRecord {
            result: GameResult::Win(Player::P1),
            plies: 10,
            first_capture_by: None,
            first_keystone_loss_by: None,
            max_stack_height: h,
        };
        // heights: 2, 4, 6 => mean = 4.0
        let records = vec![make_height(2), make_height(4), make_height(6)];
        let m = aggregate(&records);
        assert!((m.avg_max_stack - 4.0).abs() < f64::EPSILON, "avg_max_stack should be 4.0");
    }

    #[test]
    fn empty_records_produce_zeroed_metrics_without_panic() {
        let m = aggregate(&[]);
        assert_eq!(m.games, 0);
        assert_eq!(m.p1_win_rate, 0.0);
        assert_eq!(m.p2_win_rate, 0.0);
        assert_eq!(m.draw_rate, 0.0);
        assert_eq!(m.ply_median, 0.0);
        assert!(m.ply_histogram.is_empty());
        assert_eq!(m.snowball_rate, 0.0);
        assert_eq!(m.comeback_rate, 0.0);
        assert_eq!(m.avg_max_stack, 0.0);
    }

    #[test]
    fn histogram_buckets_plies_correctly() {
        // PLY_BUCKET = 25.
        // plies 10, 20 => bucket 0 (count 2)
        // plies 30, 40 => bucket 25 (count 2)
        // plies 50     => bucket 50 (count 1)
        let records = vec![
            win1(10),
            win1(20),
            win1(30),
            win1(40),
            win1(50),
        ];
        let m = aggregate(&records);
        assert_eq!(m.ply_histogram.len(), 3, "expected 3 non-empty buckets");
        assert_eq!(m.ply_histogram[&0], 2, "bucket 0 should have count 2");
        assert_eq!(m.ply_histogram[&25], 2, "bucket 25 should have count 2");
        assert_eq!(m.ply_histogram[&50], 1, "bucket 50 should have count 1");
    }
}
