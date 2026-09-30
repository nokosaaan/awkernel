//! Deterministic, single-batch admission prediction for one concrete
//! RD-Gen-generated DAG directory (as opposed to `paper_setting_comparison`'s
//! Monte-Carlo resampling over a *pool* of many independent directories):
//! this is for validating one specific, curated task set that will actually
//! be booted for real — e.g. by `run_realboot_evaluation.py`, which
//! generates a directory with a fixed DAG count via RD-Gen, predicts its
//! admission here *before* spending time on a real boot, then boots it for
//! real and compares outcomes.
//!
//! Prints each DAG's own classification under both policies, then whether
//! the whole set (all of it, exactly as given — no resampling) is jointly
//! admittable under each -- the same pure batch decisions the kernel's own
//! `federated::admit_batch` / `vfed::admit_batch` make at boot
//! (`federated::is_batch_feasible`, `vfed::is_batch_feasible` with
//! best-fit), with no `resource`/`PASSIVE_POOL` ledger side effects.
//!
//! Usage: `predict_admission <dag_dir> <num_cores>`

use std::{env, fs, process::ExitCode};

use awkernel_async_lib::dag_sched::{
    graph::DagGraph,
    metrics::DagMetrics,
    policy::{
        federated::{self, FedTask, FederatedVariant},
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

    let (configs, graphs) = match load_configs(dag_dir) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::FAILURE;
        }
    };

    println!(
        "# {} DAGs from '{dag_dir}', num_cores={num_cores}",
        configs.len()
    );
    for (i, config) in configs.iter().enumerate() {
        let fed = federated::classify_dag(config).map_err(|e| e.with_dag_id(i as u32));
        let v = vfed::classify(config).map_err(|e| e.with_dag_id(i as u32));
        println!(
            "DAG#{i}: C={} L={} T={} D={} | federated={fed:?} vfed={v:?}",
            config.volume, config.critical_path, config.period, config.relative_deadline,
        );
    }

    let fed_tasks: Vec<FedTask<'_>> = configs
        .iter()
        .zip(graphs.iter())
        .map(|(&metrics, graph)| FedTask { metrics, graph })
        .collect();
    let fed_variant = FederatedVariant::for_task_set(configs.iter());
    let fed_ok = federated::is_batch_feasible(&fed_tasks, num_cores);
    let vfed_ok = vfed::is_batch_feasible(&configs, num_cores, PackingStrategy::BestFit);
    println!();
    println!(
        "PREDICTION federated={} ({fed_variant:?}) vfed={}",
        verdict(fed_ok),
        verdict(vfed_ok)
    );

    ExitCode::SUCCESS
}

fn verdict(ok: bool) -> &'static str {
    if ok { "ACCEPT" } else { "REJECT" }
}

fn load_configs(dag_dir: &str) -> Result<(Vec<DagMetrics>, Vec<DagGraph>), String> {
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

    let configs = rd_gen_to_dags::dag_metrics_from_yaml(&yaml_refs)
        .map_err(|e| format!("failed to parse DAG set in '{dag_dir}': {e}"))?;
    let graphs = rd_gen_to_dags::dag_graphs_from_yaml(&yaml_refs)
        .map_err(|e| format!("failed to parse DAG set in '{dag_dir}': {e}"))?;
    Ok((configs, graphs))
}
