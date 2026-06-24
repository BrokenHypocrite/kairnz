use crate::metrics::Metrics;

#[derive(serde::Serialize)]
struct ConfigMetrics<'a> {
    name: &'a str,
    metrics: &'a Metrics,
}

/// Renders a side-by-side text table with one column per config and one row
/// per metric. Intended for terminal display.
pub fn render_human(results: &[(String, Metrics)]) -> String {
    if results.is_empty() {
        return String::from("(no results)");
    }

    // Column width: max of header lengths and metric values
    const COL_MIN: usize = 12;
    let col_w: Vec<usize> = results
        .iter()
        .map(|(name, _)| name.len().max(COL_MIN))
        .collect();

    let metric_col = 18usize;

    // Separator row
    let col_seps: String = results
        .iter()
        .enumerate()
        .map(|(i, _)| "-".repeat(col_w[i] + 2))
        .collect::<Vec<_>>()
        .join("+");
    let full_sep = format!("+{}+{}+", "-".repeat(metric_col), col_seps);

    let mut out = String::new();

    out.push_str(&full_sep);
    out.push('\n');

    // Header row
    out.push_str(&format!("| {:<width$}", "Metric", width = metric_col - 1));
    for (i, (name, _)) in results.iter().enumerate() {
        out.push_str(&format!("| {:>width$} ", name, width = col_w[i]));
    }
    out.push_str("|\n");
    out.push_str(&full_sep);
    out.push('\n');

    // Metric rows
    let rows: Vec<(&str, Vec<String>)> = vec![
        (
            "games",
            results
                .iter()
                .map(|(_, m)| m.games.to_string())
                .collect(),
        ),
        (
            "p1_win_rate",
            results
                .iter()
                .map(|(_, m)| format!("{:.1}%", m.p1_win_rate * 100.0))
                .collect(),
        ),
        (
            "p2_win_rate",
            results
                .iter()
                .map(|(_, m)| format!("{:.1}%", m.p2_win_rate * 100.0))
                .collect(),
        ),
        (
            "draw_rate",
            results
                .iter()
                .map(|(_, m)| format!("{:.1}%", m.draw_rate * 100.0))
                .collect(),
        ),
        (
            "ply_median",
            results
                .iter()
                .map(|(_, m)| format!("{:.1}", m.ply_median))
                .collect(),
        ),
        (
            "snowball_rate",
            results
                .iter()
                .map(|(_, m)| format!("{:.1}%", m.snowball_rate * 100.0))
                .collect(),
        ),
        (
            "comeback_rate",
            results
                .iter()
                .map(|(_, m)| format!("{:.1}%", m.comeback_rate * 100.0))
                .collect(),
        ),
        (
            "avg_max_stack",
            results
                .iter()
                .map(|(_, m)| format!("{:.2}", m.avg_max_stack))
                .collect(),
        ),
    ];

    for (label, values) in &rows {
        out.push_str(&format!("| {:<width$}", label, width = metric_col - 1));
        for (i, v) in values.iter().enumerate() {
            out.push_str(&format!("| {:>width$} ", v, width = col_w[i]));
        }
        out.push_str("|\n");
    }

    // Histogram row: compact bucket:count pairs, left-aligned within the cell
    out.push_str(&format!(
        "| {:<width$}",
        "ply_histogram",
        width = metric_col - 1
    ));
    for (i, (_, m)) in results.iter().enumerate() {
        let hist: String = m
            .ply_histogram
            .iter()
            .map(|(k, v)| format!("{}:{}", k, v))
            .collect::<Vec<_>>()
            .join(" ");
        let truncated = if hist.len() > col_w[i] {
            format!("{}...", &hist[..col_w[i].saturating_sub(3)])
        } else {
            hist
        };
        // Left-align histogram so partial content is readable when truncated
        out.push_str(&format!("| {:<width$} ", truncated, width = col_w[i]));
    }
    out.push_str("|\n");
    out.push_str(&full_sep);
    out.push('\n');

    out
}

/// Serializes results to pretty-printed JSON pairing each config name with its
/// Metrics. Returns an error message string on failure (no panic).
pub fn render_json(results: &[(String, Metrics)]) -> String {
    let payload: Vec<ConfigMetrics> = results
        .iter()
        .map(|(name, metrics)| ConfigMetrics { name, metrics })
        .collect();
    serde_json::to_string_pretty(&payload)
        .unwrap_or_else(|e| format!("{{\"error\": \"json serialization failed: {}\"}}", e))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::run::run_config;
    use crate::spec::{NamedConfig, PolicySpec};
    use cairn_core::config::RuleConfig;

    fn make_configs() -> Vec<NamedConfig> {
        vec![
            NamedConfig {
                name: "cfg-a".to_string(),
                config: {
                    let mut c = RuleConfig::default();
                    c.max_plies = 100;
                    c
                },
            },
            NamedConfig {
                name: "cfg-b".to_string(),
                config: {
                    let mut c = RuleConfig::default();
                    c.capture_lock = true;
                    c.max_plies = 100;
                    c
                },
            },
        ]
    }

    #[test]
    fn multi_config_report_has_one_column_per_config() {
        let configs = make_configs();
        let results: Vec<(String, Metrics)> = configs
            .iter()
            .map(|nc| {
                let m = run_config(nc, 4, 7, &PolicySpec::Random, &PolicySpec::Random);
                (nc.name.clone(), m)
            })
            .collect();
        let output = render_human(&results);
        assert!(output.contains("cfg-a"), "should contain cfg-a");
        assert!(output.contains("cfg-b"), "should contain cfg-b");
    }

    #[test]
    fn same_spec_and_seed_produce_identical_report() {
        let configs = make_configs();
        let run = |configs: &Vec<NamedConfig>| -> Vec<(String, Metrics)> {
            configs
                .iter()
                .map(|nc| {
                    let m = run_config(nc, 4, 7, &PolicySpec::Random, &PolicySpec::Random);
                    (nc.name.clone(), m)
                })
                .collect()
        };
        let r1 = render_human(&run(&configs));
        let r2 = render_human(&run(&configs));
        assert_eq!(
            r1, r2,
            "identical spec and seed must produce byte-identical reports"
        );
    }
}
