//! One-off diagnostic (not part of the formal pipeline): at a single
//! `U_norm` level, resample task sets exactly as `acceptance_ratio.rs` does
//! and break down Federated's/V-Fed's acceptance rate by how many of the 8
//! DAGs classify as *heavy* (under each policy's own `classify`), to test
//! the hypothesis that Federated's persistent non-zero acceptance ratio at
//! high `U_norm` comes from light-dominated draws (which reduce to
//! Federated's near-ideal, fragmentation-free shared pool), not from any
//! per-DAG core-count difference versus V-Fed.
//!
//! Usage: `diagnose_light_dominance <pool_dir> <u_norm> [trials]`

use std::{env, fs, path::Path, process::ExitCode};

use awkernel_async_lib::dag_sched::{
    metrics::DagMetrics,
    policy::{
        federated::{self, TaskClass as FedTaskClass},
        vfed::{self, PackingStrategy, TaskClass as VFedTaskClass},
    },
};
use rand::{seq::IndexedRandom, Rng};

const DAGS_PER_SET: usize = 8;
const ALPHA: f64 = 0.3;

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();
    let (pool_dir, u_norm, trials) = match args.as_slice() {
        [_, p, u] => (p.as_str(), u.parse::<f64>(), 2000usize),
        [_, p, u, t] => (p.as_str(), u.parse::<f64>(), t.parse().unwrap_or(2000)),
        _ => {
            eprintln!("usage: diagnose_light_dominance <pool_dir> <u_norm> [trials]");
            return ExitCode::from(2);
        }
    };
    let Ok(u_norm) = u_norm else {
        eprintln!("u_norm must be a number");
        return ExitCode::from(2);
    };

    let pool = match load_pool(Path::new(pool_dir)) {
        Ok(p) if !p.is_empty() => p,
        Ok(_) => {
            eprintln!("'{pool_dir}' has no dag_*.yaml files");
            return ExitCode::FAILURE;
        }
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::FAILURE;
        }
    };

    let mut rng = rand::rng();

    // Bucketed by how many of the 8 DAGs are heavy under *Federated's own*
    // classify_dag (the two policies' heavy/light boundary can differ when
    // D > T, but every deadline here is clamped to D <= T, so in practice
    // they agree almost always -- bucketing by one policy's count is enough
    // to test the hypothesis).
    let mut by_heavy_count: Vec<(usize, usize, usize)> = vec![(0, 0, 0); DAGS_PER_SET + 1]; // (n, fed_ok, vfed_ok)

    let mut u_sigma_sum = 0.0f64;
    let mut m_sum = 0u64;
    let mut m_min = u16::MAX;
    let mut m_max = 0u16;
    let mut light_util_sum = 0.0f64;
    let mut light_util_max = 0.0f64;

    for _ in 0..trials {
        let set: Vec<DagMetrics> = (0..DAGS_PER_SET)
            .map(|_| *pool.choose(&mut rng).expect("pool checked non-empty above"))
            .collect();

        let u_sigma: f64 = set.iter().map(|d| d.volume as f64 / d.period as f64).sum();
        let m = ((u_sigma / u_norm).ceil() as u16).max(1);

        let heavy_count = set
            .iter()
            .filter(|c| matches!(federated::classify_dag(c), Ok(FedTaskClass::Heavy { .. })))
            .count();

        let fed_ok = federated_batch_feasible(&set, m);
        let vfed_ok = vfed::is_batch_feasible(&set, m, PackingStrategy::BestFit);

        let bucket = &mut by_heavy_count[heavy_count];
        bucket.0 += 1;
        if fed_ok {
            bucket.1 += 1;
        }
        if vfed_ok {
            bucket.2 += 1;
        }

        u_sigma_sum += u_sigma;
        m_sum += m as u64;
        m_min = m_min.min(m);
        m_max = m_max.max(m);
        let this_light_util: f64 = set
            .iter()
            .filter(|c| matches!(federated::classify_dag(c), Ok(FedTaskClass::Light)))
            .map(|c| c.volume as f64 / c.relative_deadline.min(c.period) as f64)
            .sum();
        light_util_sum += this_light_util;
        light_util_max = light_util_max.max(this_light_util);
    }

    println!(
        "# u_sigma: mean={:.2} | m: mean={:.2} min={} max={} | light_util_of_light_subset: mean={:.2} max={:.2}",
        u_sigma_sum / trials as f64,
        m_sum as f64 / trials as f64,
        m_min,
        m_max,
        light_util_sum / trials as f64,
        light_util_max,
    );

    // A few concrete pool DAGs' own numbers, to see actual scale.
    for config in pool.iter().take(5) {
        println!(
            "# sample pool DAG: C={} L={} T={} D={} m_i={:?} class={:?}",
            config.volume,
            config.critical_path,
            config.period,
            config.relative_deadline,
            config.min_dedicated_cores(),
            federated::classify_dag(config),
        );
    }

    println!("u_norm={u_norm}, trials={trials}");
    println!("heavy_count,n_sampled,fed_accept_pct,vfed_accept_pct");
    for (heavy_count, &(n, fed_ok, vfed_ok)) in by_heavy_count.iter().enumerate() {
        if n == 0 {
            continue;
        }
        println!(
            "{heavy_count},{n},{:.2},{:.2}",
            100.0 * fed_ok as f64 / n as f64,
            100.0 * vfed_ok as f64 / n as f64,
        );
    }

    // Also report V-Fed's own heavy classification distribution, in case it
    // disagrees with Federated's (would show up as a note, not a bucket).
    let mut vfed_heavy_mismatch = 0usize;
    let mut checked = 0usize;
    for config in &pool {
        checked += 1;
        let fed_heavy = matches!(federated::classify_dag(config), Ok(FedTaskClass::Heavy { .. }));
        let vfed_heavy = matches!(vfed::classify(config), Ok(VFedTaskClass::Heavy { .. }));
        if fed_heavy != vfed_heavy {
            vfed_heavy_mismatch += 1;
        }
    }
    println!(
        "# heavy/light classification mismatch between federated and vfed: {vfed_heavy_mismatch}/{checked} pool DAGs"
    );

    ExitCode::SUCCESS
}

fn load_pool(pool_dir: &Path) -> Result<Vec<DagMetrics>, String> {
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

    let yaml_contents: Vec<String> = yaml_paths
        .iter()
        .map(|p| fs::read_to_string(p).map_err(|e| format!("cannot read {}: {e}", p.display())))
        .collect::<Result<_, _>>()?;
    let yaml_refs: Vec<&str> = yaml_contents.iter().map(String::as_str).collect();

    let all = rd_gen_to_dags::dag_metrics_from_yaml(&yaml_refs)
        .map_err(|e| format!("failed to parse DAG pool in '{}': {e}", pool_dir.display()))?;

    let mut rng = rand::rng();
    Ok(all
        .into_iter()
        .map(|config| {
            let config = override_deadline(config, &mut rng);
            let config = override_period(config, &mut rng);
            assign_max_parallelism(config, &mut rng)
        })
        .collect())
}

/// See `acceptance_ratio.rs`'s own `override_deadline`/`override_period`
/// (kept in sync with those; duplicated here rather than shared since both
/// are throwaway examples, not a library).
fn override_deadline(config: DagMetrics, rng: &mut impl Rng) -> DagMetrics {
    let l = config.critical_path as f64;
    let d = l + rng.random_range(0.0..=1.0) * (l / ALPHA - l);
    let d = (d as u64).max(1);
    DagMetrics { relative_deadline: d, ..config }
}

fn override_period(config: DagMetrics, rng: &mut impl Rng) -> DagMetrics {
    let beta = rng.random_range(0.1..=1.0);
    let period = ((config.relative_deadline as f64 / beta) as u64).max(config.relative_deadline);
    DagMetrics { period, ..config }
}

fn assign_max_parallelism(config: DagMetrics, rng: &mut impl Rng) -> DagMetrics {
    let Some(m) = config.min_dedicated_cores() else {
        return config;
    };
    let lo = m.div_ceil(2);
    let max_parallelism = if lo >= m { m } else { rng.random_range(lo..=m) };
    DagMetrics { max_parallelism, ..config }
}

fn federated_batch_feasible(configs: &[DagMetrics], num_cores: u16) -> bool {
    let mut heavy_cores: u32 = 0;
    let mut light_utilization: f64 = 0.0;

    for &config in configs {
        match federated::classify_dag(&config) {
            Ok(FedTaskClass::Heavy { required_cores }) => {
                heavy_cores += required_cores as u32;
            }
            Ok(FedTaskClass::Light) => {
                let window = config.relative_deadline.min(config.period) as f64;
                light_utilization += config.volume as f64 / window;
            }
            Err(_) => return false,
        }
    }

    if heavy_cores > num_cores as u32 {
        return false;
    }
    light_utilization <= (num_cores as u32 - heavy_cores) as f64
}
