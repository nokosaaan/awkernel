//! Diagnostic dump: per-DAG utilization/required_cores for every bin under
//! a `theory_vs_reality.rs`-style `bins_root_dir`, read as-is (RD-Gen's
//! 'Constrained' deadline mode already guarantees `D <= period` by
//! construction, so no post-parse override is needed or applied here --
//! matches `theory_vs_reality.rs`'s own `load_and_prefilter_pool`; see that
//! file's module doc for why an `override_deadline`/`override_period` scheme
//! was tried first and rejected, 2026-09-24).
//!
//! Usage: `dump_bin_stats <bins_root_dir> <real_cores>`, prints CSV
//! (`u_norm,dag,volume,critical_path,period,relative_deadline,utilization,class,required_cores`)
//! to stdout.

use std::{env, fs, path::{Path, PathBuf}, process::ExitCode};

use awkernel_async_lib::dag_sched::{
    metrics::DagMetrics,
    policy::federated::{self, TaskClass as FedTaskClass},
};

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();
    let [_, bins_root_dir, real_cores] = args.as_slice() else {
        eprintln!("usage: dump_bin_stats <bins_root_dir> <real_cores>");
        return ExitCode::from(2);
    };
    let real_cores: u16 = match real_cores.parse() {
        Ok(n) if n > 0 => n,
        _ => {
            eprintln!("real_cores must be a positive integer, got '{real_cores}'");
            return ExitCode::from(2);
        }
    };

    let bins = match discover_bins(Path::new(bins_root_dir)) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::FAILURE;
        }
    };

    println!("u_norm,dag,volume,critical_path,period,relative_deadline,utilization,class,required_cores");
    for (u_norm, bin_dir) in bins {
        let dags_dir = bin_dir.join("DAGs");
        let entries = match load_dags(&dags_dir) {
            Ok(e) => e,
            Err(e) => {
                eprintln!("u_norm={u_norm:.4}: {e}");
                continue;
            }
        };
        for (name, config) in entries {
            let utilization = config.volume as f64 / config.period as f64;
            let (class, required_cores) = match federated::classify_dag(&config) {
                Ok(FedTaskClass::Heavy { required_cores }) => {
                    let over = if required_cores > real_cores { "_over" } else { "" };
                    (format!("heavy{over}"), required_cores.to_string())
                }
                Ok(FedTaskClass::Light) => ("light".to_string(), String::new()),
                Err(_) => ("infeasible".to_string(), String::new()),
            };
            println!(
                "{u_norm:.4},{name},{},{},{},{},{utilization:.6},{class},{required_cores}",
                config.volume, config.critical_path, config.period, config.relative_deadline,
            );
        }
    }

    ExitCode::SUCCESS
}

fn discover_bins(bins_root_dir: &Path) -> Result<Vec<(f64, PathBuf)>, String> {
    let mut bins: Vec<(f64, PathBuf)> = fs::read_dir(bins_root_dir)
        .map_err(|e| format!("cannot read directory '{}': {e}", bins_root_dir.display()))?
        .filter_map(|entry| entry.ok())
        .filter(|entry| entry.path().is_dir())
        .filter_map(|entry| {
            let name = entry.file_name();
            let u_norm: f64 = name.to_str()?.strip_prefix("u_norm_")?.parse().ok()?;
            Some((u_norm, entry.path()))
        })
        .collect();
    bins.sort_by(|a, b| a.0.total_cmp(&b.0));
    Ok(bins)
}

fn load_dags(dags_dir: &Path) -> Result<Vec<(String, DagMetrics)>, String> {
    let mut yaml_paths: Vec<_> = fs::read_dir(dags_dir)
        .map_err(|e| format!("cannot read directory '{}': {e}", dags_dir.display()))?
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

    let configs = rd_gen_to_dags::dag_metrics_from_yaml(&yaml_refs)
        .map_err(|e| format!("failed to parse DAG pool in '{}': {e}", dags_dir.display()))?;

    Ok(names.into_iter().zip(configs).collect())
}
