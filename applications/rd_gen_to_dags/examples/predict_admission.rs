//! Deterministic, single-batch admission prediction for one concrete
//! RD-Gen-generated DAG directory (as opposed to `acceptance_ratio`'s
//! Monte-Carlo resampling over a *pool* of many independent directories):
//! this is for validating one specific, curated task set that will actually
//! be booted for real — e.g. by `run_realboot_evaluation.py`, which
//! generates a directory with a fixed DAG count via RD-Gen, predicts its
//! admission here *before* spending time on a real boot, then boots it for
//! real and compares outcomes.
//!
//! Prints each DAG's own classification under both policies, then whether
//! the whole set (all of it, exactly as given — no resampling) is jointly
//! admittable under each, mirroring `dag_sched::policy::federated`'s and
//! `dag_sched::policy::vfed`'s own real batch semantics but as pure
//! functions of the parsed `DagMetrics` (see `acceptance_ratio.rs`'s own
//! doc for why: no `resource`/`PASSIVE_POOL` ledger side effects).
//!
//! Usage: `predict_admission <dag_dir> <num_cores>`

use std::{env, fs, process::ExitCode};

use awkernel_async_lib::dag_sched::{
    metrics::DagMetrics,
    policy::{
        federated::{self, TaskClass as FedTaskClass},
        vfed::{self, PackingStrategy},
    },
};

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();
    let [_, dag_dir, num_cores] = args.as_slice() else {
        eprintln!("usage: predict_admission <dag_dir> <num_cores>");
        return ExitCode::from(2);
    };
    let num_cores: u16 = match num_cores.parse() {
        Ok(n) if n > 0 => n,
        _ => {
            eprintln!("num_cores must be a positive integer, got '{num_cores}'");
            return ExitCode::from(2);
        }
    };

    let configs = match load_configs(dag_dir) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::FAILURE;
        }
    };

    println!("# {} DAGs from '{dag_dir}', num_cores={num_cores}", configs.len());
    for (i, config) in configs.iter().enumerate() {
        let fed = federated::classify_dag(config);
        let v = vfed::classify(config);
        println!(
            "DAG#{i}: C={} L={} T={} D={} | federated={fed:?} vfed={v:?}",
            config.volume, config.critical_path, config.period, config.relative_deadline,
        );
    }

    let fed_ok = federated_batch_feasible(&configs, num_cores);
    let vfed_ok = vfed::is_batch_feasible(&configs, num_cores, PackingStrategy::FirstFit);
    println!();
    println!("PREDICTION federated={} vfed={}", verdict(fed_ok), verdict(vfed_ok));

    ExitCode::SUCCESS
}

fn verdict(ok: bool) -> &'static str {
    if ok {
        "ACCEPT"
    } else {
        "REJECT"
    }
}

fn load_configs(dag_dir: &str) -> Result<Vec<DagMetrics>, String> {
    let mut yaml_paths: Vec<_> = fs::read_dir(dag_dir)
        .map_err(|e| format!("cannot read directory '{dag_dir}': {e}"))?
        .filter_map(|entry| entry.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|ext| ext == "yaml"))
        .filter(|p| {
            p.file_stem()
                .and_then(|s| s.to_str())
                .is_some_and(|s| s.starts_with("dag_"))
        })
        .collect();
    yaml_paths.sort();

    if yaml_paths.is_empty() {
        return Err(format!("no dag_*.yaml files found in '{dag_dir}'"));
    }

    let yaml_contents: Vec<String> = yaml_paths
        .iter()
        .map(|p| fs::read_to_string(p).map_err(|e| format!("cannot read {}: {e}", p.display())))
        .collect::<Result<_, _>>()?;
    let yaml_refs: Vec<&str> = yaml_contents.iter().map(String::as_str).collect();

    rd_gen_to_dags::dag_metrics_from_yaml(&yaml_refs)
        .map_err(|e| format!("failed to parse DAG set in '{dag_dir}': {e}"))
}

/// Verbatim copy of `acceptance_ratio.rs`'s own pure Federated batch check
/// (see that file for why it's reimplemented rather than calling the real
/// `resource`-ledger-backed `federated::admit_dag`).
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
