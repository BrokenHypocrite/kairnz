mod metrics;
mod report;
mod run;
mod runner;
mod spec;

use clap::Parser;
use run::run_config;
use spec::load_run_spec;

/// Command-line arguments for the Cairn benchmark harness.
#[derive(Parser)]
#[command(name = "cairn-bench", about = "Run headless Cairn game benchmarks")]
struct Cli {
    /// Path to the YAML run-spec file.
    #[arg(long)]
    spec: String,

    /// If provided, write JSON results to this file path.
    #[arg(long)]
    json: Option<String>,
}

fn main() {
    let cli = Cli::parse();

    let run_spec = match load_run_spec(&cli.spec) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: {}", e);
            std::process::exit(1);
        }
    };

    let mut results: Vec<(String, crate::metrics::Metrics)> = Vec::new();
    for named in &run_spec.configs {
        eprintln!(
            "running config '{}' ({} games)...",
            named.name, run_spec.games_per_config
        );
        let metrics = run_config(
            named,
            run_spec.games_per_config,
            run_spec.seed,
            &run_spec.p1_policy,
            &run_spec.p2_policy,
        );
        results.push((named.name.clone(), metrics));
    }

    let human = report::render_human(&results);
    print!("{}", human);

    if let Some(json_path) = &cli.json {
        let json = report::render_json(&results);
        if let Err(e) = std::fs::write(json_path, &json) {
            eprintln!("error writing JSON to '{}': {}", json_path, e);
            std::process::exit(1);
        }
    }
}
