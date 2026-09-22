//! Reports, for every `dag_<N>.yaml` in a pool, the DAG-Fluid segment
//! decomposition's own maximum concurrency (`max(Segment::concurrency)`
//! across that DAG's whole "infinite processors" timeline — see
//! `dag_fluid::decompose_segments`'s own doc) alongside a pool-wide summary.
//!
//! # Why this exists
//! DAG-Fluid's real-machine dispatch (`scheduler::dp_wrap`) can only ever
//! grant a segment as many *real* CPUs as the target's own DAG-pool core
//! count -- a segment whose own concurrency exceeds that count necessarily
//! serializes part of its "virtual thread" parallelism onto the same
//! physical cores no matter how good the dispatch mechanism is, inflating
//! its real completion time well past the papers' own zero-slack
//! theoretical deadline. RD-Gen's own generation parameters (Maximum
//! branches / Probability of branching / Number of nodes / edge
//! probability) do not directly express "peak concurrent nodes" -- multiple
//! independent branch groups can still overlap in time even with a small
//! per-branch-point cap -- so the only reliable way to check whether a
//! candidate parameter set fits a given target machine is to generate a
//! sample and measure it here, the same way `dag_fluid::decompose_segments`
//! itself would be exercised for real. Used to tune a new sample_config
//! (see `RD-Gen/sample_config/branching/`'s own docs) against this
//! project's own real target (5 worker cores, ~4 DAG-pool cores after the
//! regular-pool split -- see `scheduler::pool`'s own doc).
//!
//! Usage: `segment_concurrency_stats <pool_dir> [target_max_concurrency]`
//! (`target_max_concurrency` only changes the "over target" percentage
//! printed; default 4).

use std::{env, fs, path::Path, process::ExitCode};

use rd_gen_to_dags::dag_fluid::Segment;

fn load_pool_segments(pool_dir: &Path) -> Result<Vec<(String, Vec<Segment>)>, String> {
    let mut yaml_paths: Vec<_> = fs::read_dir(pool_dir)
        .map_err(|e| format!("cannot read directory '{}': {e}", pool_dir.display()))?
        .filter_map(|entry| entry.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|ext| ext == "yaml"))
        .filter(|p| {
            p.file_stem()
                .and_then(|s| s.to_str())
                .is_some_and(|s| s.starts_with("dag_"))
        })
        .collect();
    yaml_paths.sort();

    let names: Vec<String> = yaml_paths
        .iter()
        .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
        .collect();
    let yaml_contents: Vec<String> = yaml_paths
        .iter()
        .map(|p| fs::read_to_string(p).map_err(|e| format!("cannot read {}: {e}", p.display())))
        .collect::<Result<_, _>>()?;
    let yaml_refs: Vec<&str> = yaml_contents.iter().map(String::as_str).collect();

    let all = rd_gen_to_dags::dag_metrics_and_fluid_segments_from_yaml(&yaml_refs)
        .map_err(|e| format!("failed to parse DAG pool in '{}': {e}", pool_dir.display()))?;

    Ok(names
        .into_iter()
        .zip(all)
        .map(|(name, (_config, segments))| (name, segments))
        .collect())
}

fn percentile(sorted: &[u32], p: f64) -> u32 {
    if sorted.is_empty() {
        return 0;
    }
    let idx = ((sorted.len() - 1) as f64 * p).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();
    let (pool_dir, target) = match args.as_slice() {
        [_, pool_dir] => (pool_dir, 4u32),
        [_, pool_dir, target] => match target.parse::<u32>() {
            Ok(t) => (pool_dir, t),
            Err(_) => {
                eprintln!("target_max_concurrency must be a non-negative integer");
                return ExitCode::FAILURE;
            }
        },
        _ => {
            eprintln!("usage: segment_concurrency_stats <pool_dir> [target_max_concurrency]");
            return ExitCode::FAILURE;
        }
    };

    let pool = match load_pool_segments(Path::new(pool_dir)) {
        Ok(pool) => pool,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::FAILURE;
        }
    };
    if pool.is_empty() {
        eprintln!("'{pool_dir}' has no dag_*.yaml files");
        return ExitCode::FAILURE;
    }

    let mut max_concurrency: Vec<u32> = pool
        .iter()
        .map(|(_, segments)| segments.iter().map(|s| s.concurrency).max().unwrap_or(0))
        .collect();
    max_concurrency.sort_unstable();

    let n = max_concurrency.len();
    let sum: u64 = max_concurrency.iter().map(|&c| c as u64).sum();
    let mean = sum as f64 / n as f64;
    let over_target = max_concurrency.iter().filter(|&&c| c > target).count();

    println!("pool: {pool_dir} ({n} DAGs)");
    println!(
        "max concurrency per DAG: min={} p50={} p90={} max={} mean={mean:.2}",
        max_concurrency[0],
        percentile(&max_concurrency, 0.50),
        percentile(&max_concurrency, 0.90),
        max_concurrency[n - 1],
    );
    println!(
        "DAGs with max concurrency > {target} (target DAG-pool core count): {over_target}/{n} ({:.1}%)",
        100.0 * over_target as f64 / n as f64
    );

    // Show the worst few offenders by name, for spot-checking.
    let mut named: Vec<(String, u32)> = pool
        .iter()
        .map(|(name, segments)| (name.clone(), segments.iter().map(|s| s.concurrency).max().unwrap_or(0)))
        .collect();
    named.sort_by(|a, b| b.1.cmp(&a.1));
    println!("worst 5 by max concurrency:");
    for (name, c) in named.iter().take(5) {
        println!("  {name}: {c}");
    }

    ExitCode::SUCCESS
}
