//! Per-`u_norm`-bin Federated acceptance-ratio sweep against a *fixed* real
//! core count, producing a trial manifest a real-machine run can replay --
//! the offline half of a theory-vs-reality gap study (predicted admission +
//! [`min_dedicated_cores`] preconditions vs. actually measured deadline
//! misses on real hardware for the *same* task sets).
//!
//! [`min_dedicated_cores`]: awkernel_async_lib::dag_sched::metrics::DagMetrics::min_dedicated_cores
//!
//! # How this differs from `paper_setting_comparison.rs`
//! That file (the V-Fed paper's Sec. 7 method) derives `m = ceil(U_Sigma / u_norm)` *per resampled trial*, so
//! `m` is whatever the trial's own draw needs -- never tied to any real
//! machine's actual core count (see its own module doc). This file inverts
//! that: `real_cores` (a CLI argument, e.g. the target machine's actual
//! dag-pool core count) is fixed for the whole run -- the processor set `P`
//! every admission test below is given, matching how V-Fed's own Algorithm 1
//! (TPDS'23) takes `P` as an input of the *test*, not of the task set.
//!
//! # Core-count-independent pool, per-`u_norm` window chosen here
//! The input is ONE RD-Gen pool (`<pool_dir>/DAGs/dag_*.yaml`, i.e.
//! `run_generator.py`'s own `-d` output layout, or `<pool_dir>/dag_*.yaml`
//! directly) whose `Whole-DAG utilization` is spread over a wide range with
//! no core count baked in (see RD-Gen's
//! `sample_config/chain_based/theory_vs_reality_pool.yaml`). For each target
//! `u_norm`, this file computes the per-DAG utilization a set of
//! `dags_per_set` DAGs needs on `real_cores` cores,
//!
//! ```text
//! center = u_norm * real_cores / dags_per_set
//! delta  = WINDOW_FRACTION_OF_STEP * u_norm_step * real_cores / dags_per_set
//! ```
//!
//! and draws only from the DAGs whose *measured* `volume / period` falls in
//! `[center - delta, center + delta]`, so a plain random draw already lands
//! close to the target `u_norm` -- no rejection sampling, no post-hoc
//! rescaling of `C`/`T`, and the same pool serves any `real_cores`. (An
//! earlier version baked `REAL_CORES` into generation instead, one RD-Gen
//! pool per `u_norm` bin; changing the target machine meant regenerating
//! every bin.)
//!
//! # Per-DAG prefilter
//! Before windowing, every DAG that `classify_dag` already calls infeasible
//! (`D < L`) is dropped from the pool: no scheduling policy, however
//! parallel, can finish critical-path work faster than the critical path
//! itself takes, so this is a universal, algorithm-agnostic
//! disqualification -- unlike a plain `Heavy { required_cores }` DAG
//! itself, whose `required_cores > real_cores` would only disqualify it
//! under FEDERATED's own dedicated-cluster math specifically (an earlier
//! version of this file also dropped those here, which biased the shared
//! pool against V-Fed/DAG-Fluid's own, different core-requirement math --
//! see "Per-algorithm admission" below for why the three now share one
//! pool and one resampled draw per trial rather than each filtering its
//! own).
//!
//! # Per-algorithm admission, not a Federated-only proxy
//! Each trial's *same* resampled `dags_per_set`-DAG set (the same real-
//! machine run would boot under any of `--algorithms federated,vfed,
//! dagfluid,laxity`, see `real_machine_trial.py`) is checked against three
//! independent theories, all fixed to the same `real_cores`: Federated's
//! batch admission (`federated::is_batch_feasible` -- Li et al. ECRTS'14,
//! Baruah DATE'15 or Baruah IPDPS'15, whichever matches the drawn set's
//! deadline type; recorded per trial as `federated_variant`), V-Fed's own
//! batch planner (`vfed::is_batch_feasible`, TPDS'23 Algorithms 1-2), and
//! DAG-Fluid's own fluid-capacity check (`dag_fluid::is_batch_feasible`) --
//! the same policies as `paper_setting_comparison.rs`'s CSV, just with
//! `real_cores` fixed instead of a derived `m`. `manifest_jsonl` records all three
//! per-trial booleans (`accepted`/`vfed_accepted`/`dag_fluid_accepted`),
//! (plus `sfs_accepted`, SFS-G of Lendve et al., JSA 2026, `policy::sfs`),
//! so a real-machine comparison can read the ONE column matching whichever
//! algorithm it actually booted, instead of reusing Federated's own
//! decision as a stand-in for every algorithm (which conflates "the real
//! machine deviated from theory" with "this algorithm's own theory was
//! never actually evaluated"). `laxity` has no admission test of its own
//! (its kernel build puts every DAG on global EDF without any admission
//! check), so for it `accepted` -- Federated's decision -- is only a proxy.
//!
//! No file staging, no kernel build, no boot: `manifest_jsonl` records
//! which pool directory and filenames each trial drew and whether it was
//! predicted ACCEPT/REJECT under each theory; a separate script (see
//! `run_realboot_evaluation.py`'s own pattern) reads it, stages the
//! ACCEPTed trials' files, boots them for real under the matching
//! algorithm, and compares the resulting measured deadline-miss rate
//! against this file's own per-bin acceptance ratio.
//!
//! Usage: `theory_vs_reality <pool_dir> <real_cores> <dags_per_set>
//! <trials_per_bin> <manifest_jsonl_path> [<u_norm_min> <u_norm_max>
//! <u_norm_step>]` (defaults `0.10 1.00 0.05`), prints
//! `u_norm,federated_ratio,vfed_ratio,dag_fluid_ratio` CSV to stdout (one
//! row per `u_norm` bin with enough DAGs in its window, ascending `u_norm`).

use std::{
    env, fs,
    io::Write,
    path::{Path, PathBuf},
    process::ExitCode,
};

use awkernel_async_lib::dag_sched::{
    graph::DagGraph,
    metrics::DagMetrics,
    policy::{
        dag_fluid::{self, Segment},
        federated::{self, FedTask, FederatedVariant},
        sfs::{self, SfsTask},
        vfed::{self, PackingStrategy},
    },
};
use rand::seq::IndexedRandom;

const DEFAULT_U_NORM_MIN: f64 = 0.10;
const DEFAULT_U_NORM_MAX: f64 = 1.00;
const DEFAULT_U_NORM_STEP: f64 = 0.05;

/// Half-width of each `u_norm` bin's per-DAG utilization window, as a
/// fraction of the gap between two adjacent bins' centers
/// (`u_norm_step * real_cores / dags_per_set`). Kept below 0.5 so adjacent
/// windows never overlap; 0.3 (the value the earlier generation-side bins
/// were tuned with) still leaves each bin some internal utilization spread.
const WINDOW_FRACTION_OF_STEP: f64 = 0.3;

struct PoolEntry {
    name: String,
    metrics: DagMetrics,
    segments: Vec<Segment>,
    graph: DagGraph,
    utilization: f64,
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();
    let (pool_dir, real_cores, dags_per_set, trials_per_bin, manifest_jsonl_path, u_norm_range) =
        match args.as_slice() {
            [_, p, c, d, t, m] => (p, c, d, t, m, None),
            [_, p, c, d, t, m, lo, hi, step] => (p, c, d, t, m, Some((lo, hi, step))),
            _ => {
                eprintln!(
                    "usage: theory_vs_reality <pool_dir> <real_cores> <dags_per_set> \
                     <trials_per_bin> <manifest_jsonl_path> [<u_norm_min> <u_norm_max> <u_norm_step>]"
                );
                return ExitCode::from(2);
            }
        };

    let real_cores: u16 = match real_cores.parse() {
        Ok(n) if n > 0 => n,
        _ => {
            eprintln!("real_cores must be a positive integer, got '{real_cores}'");
            return ExitCode::from(2);
        }
    };
    let dags_per_set: usize = match dags_per_set.parse() {
        Ok(n) if n > 0 => n,
        _ => {
            eprintln!("dags_per_set must be a positive integer, got '{dags_per_set}'");
            return ExitCode::from(2);
        }
    };
    let trials_per_bin: usize = match trials_per_bin.parse() {
        Ok(n) if n > 0 => n,
        _ => {
            eprintln!("trials_per_bin must be a positive integer, got '{trials_per_bin}'");
            return ExitCode::from(2);
        }
    };
    let (u_norm_min, u_norm_max, u_norm_step) = match u_norm_range {
        None => (DEFAULT_U_NORM_MIN, DEFAULT_U_NORM_MAX, DEFAULT_U_NORM_STEP),
        Some((lo, hi, step)) => match (lo.parse::<f64>(), hi.parse::<f64>(), step.parse::<f64>()) {
            (Ok(lo), Ok(hi), Ok(step)) if lo > 0.0 && hi >= lo && step > 0.0 => (lo, hi, step),
            _ => {
                eprintln!(
                    "u_norm_min/u_norm_max/u_norm_step must satisfy 0 < min <= max and step > 0, \
                     got '{lo}' '{hi}' '{step}'"
                );
                return ExitCode::from(2);
            }
        },
    };

    let dags_dir = resolve_dags_dir(Path::new(pool_dir));
    let (pool, dropped) = match load_and_prefilter_pool(&dags_dir) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::FAILURE;
        }
    };
    eprintln!(
        "pool '{}': {} feasible-by-itself DAGs, {dropped} dropped by prefilter",
        dags_dir.display(),
        pool.len(),
    );

    let mut manifest = match fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(manifest_jsonl_path)
    {
        Ok(f) => f,
        Err(e) => {
            eprintln!("cannot open '{manifest_jsonl_path}' for appending: {e}");
            return ExitCode::FAILURE;
        }
    };

    let mut rng = rand::rng();
    let per_dag_scale = real_cores as f64 / dags_per_set as f64;
    let delta = WINDOW_FRACTION_OF_STEP * u_norm_step * per_dag_scale;

    println!("u_norm,federated_ratio,vfed_ratio,dag_fluid_ratio,sfs_ratio");
    for u_norm in u_norm_values(u_norm_min, u_norm_max, u_norm_step) {
        let center = u_norm * per_dag_scale;
        let (lo, hi) = (center - delta, center + delta);
        let window: Vec<&PoolEntry> = pool
            .iter()
            .filter(|e| e.utilization >= lo && e.utilization <= hi)
            .collect();
        eprintln!(
            "u_norm={u_norm:.4}: per-DAG utilization window [{lo:.4}, {hi:.4}] has {} DAGs",
            window.len(),
        );
        if window.len() < dags_per_set {
            eprintln!(
                "u_norm={u_norm:.4}: window has only {} DAGs (< dags_per_set={dags_per_set}); skipping bin \
                 -- widen the pool's 'Whole-DAG utilization' range or raise its 'Number of DAGs'",
                window.len(),
            );
            continue;
        }

        let mut fed_accepted = 0usize;
        let mut vfed_accepted = 0usize;
        let mut dag_fluid_accepted = 0usize;
        let mut sfs_accepted = 0usize;
        for trial in 0..trials_per_bin {
            let set: Vec<&PoolEntry> = window
                .choose_multiple(&mut rng, dags_per_set)
                .copied()
                .collect();
            let metrics: Vec<DagMetrics> = set.iter().map(|e| e.metrics).collect();

            let fed_tasks: Vec<FedTask<'_>> = set
                .iter()
                .map(|e| FedTask {
                    metrics: e.metrics,
                    graph: &e.graph,
                })
                .collect();
            let fed_variant = FederatedVariant::for_task_set(metrics.iter());
            let fed_ok = federated::is_batch_feasible(&fed_tasks, real_cores);
            if fed_ok {
                fed_accepted += 1;
            }
            let vfed_ok = vfed::is_batch_feasible(&metrics, real_cores, PackingStrategy::BestFit);
            if vfed_ok {
                vfed_accepted += 1;
            }
            let dag_fluid_entries: Vec<(u64, u64, u64, u64, &[Segment])> = set
                .iter()
                .map(|e| {
                    let m = &e.metrics;
                    (
                        m.volume,
                        m.period,
                        m.critical_path,
                        m.relative_deadline,
                        e.segments.as_slice(),
                    )
                })
                .collect();
            let dag_fluid_ok = dag_fluid::is_batch_feasible(&dag_fluid_entries, real_cores);
            if dag_fluid_ok {
                dag_fluid_accepted += 1;
            }
            let sfs_tasks: Vec<SfsTask<'_>> = set
                .iter()
                .map(|e| SfsTask {
                    graph: &e.graph,
                    period: e.metrics.period,
                    deadline: e.metrics.relative_deadline,
                })
                .collect();
            let sfs_ok = sfs::is_schedulable(&sfs_tasks, real_cores);
            if sfs_ok {
                sfs_accepted += 1;
            }

            let u_sigma_actual: f64 = set.iter().map(|e| e.utilization).sum();
            write_trial_record(
                &mut manifest,
                u_norm,
                trial,
                &dags_dir,
                &set,
                fed_variant,
                fed_ok,
                vfed_ok,
                dag_fluid_ok,
                sfs_ok,
                u_sigma_actual,
                real_cores,
            );
        }

        println!(
            "{u_norm:.4},{:.2},{:.2},{:.2},{:.2}",
            100.0 * fed_accepted as f64 / trials_per_bin as f64,
            100.0 * vfed_accepted as f64 / trials_per_bin as f64,
            100.0 * dag_fluid_accepted as f64 / trials_per_bin as f64,
            100.0 * sfs_accepted as f64 / trials_per_bin as f64,
        );
    }

    ExitCode::SUCCESS
}

/// `<pool_dir>/DAGs` if it exists (`run_generator.py -d <pool_dir>`'s own
/// layout), else `pool_dir` itself.
fn resolve_dags_dir(pool_dir: &Path) -> PathBuf {
    let nested = pool_dir.join("DAGs");
    if nested.is_dir() {
        nested
    } else {
        pool_dir.to_path_buf()
    }
}

/// `u_norm_min, u_norm_min + step, ..., u_norm_max` (inclusive), each
/// rounded to 4 decimals so float accumulation doesn't drop the last bin.
fn u_norm_values(u_norm_min: f64, u_norm_max: f64, u_norm_step: f64) -> Vec<f64> {
    let n_steps = ((u_norm_max - u_norm_min) / u_norm_step + 1e-9).floor() as usize;
    (0..=n_steps)
        .map(|i| ((u_norm_min + i as f64 * u_norm_step) * 1e4).round() / 1e4)
        .collect()
}

/// Load every `dag_<N>.yaml` directly inside `dags_dir` as-is -- RD-Gen's own
/// 'Constrained' deadline mode already guarantees `D <= period` by
/// construction (see this file's own module doc for why this file does NOT
/// override deadline/period the way `paper_setting_comparison.rs` does) -- then drop
/// any DAG `classify_dag` already calls infeasible (see this file's own
/// "Per-DAG prefilter" doc for why only that case, not a plain
/// `Heavy { required_cores }`, is dropped here). Returns the survivors
/// with their filename, DAG-Fluid segment decomposition (needed for
/// `dag_fluid::is_batch_feasible`) and measured `volume / period`, plus how
/// many were dropped.
fn load_and_prefilter_pool(dags_dir: &Path) -> Result<(Vec<PoolEntry>, usize), String> {
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
    if yaml_paths.is_empty() {
        return Err(format!("'{}' has no dag_*.yaml files", dags_dir.display()));
    }

    let names: Vec<String> = yaml_paths
        .iter()
        .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
        .collect();
    let yaml_contents: Vec<String> = yaml_paths
        .iter()
        .map(|p| fs::read_to_string(p).map_err(|e| format!("cannot read {}: {e}", p.display())))
        .collect::<Result<_, _>>()?;
    let yaml_refs: Vec<&str> = yaml_contents.iter().map(String::as_str).collect();

    let configs = rd_gen_to_dags::dag_metrics_and_fluid_segments_from_yaml(&yaml_refs)
        .map_err(|e| format!("failed to parse DAG pool in '{}': {e}", dags_dir.display()))?;
    let graphs = rd_gen_to_dags::dag_graphs_from_yaml(&yaml_refs)
        .map_err(|e| format!("failed to parse DAG pool in '{}': {e}", dags_dir.display()))?;

    let mut survivors = Vec::new();
    let mut dropped = 0usize;
    for ((name, (config, segments)), graph) in names.into_iter().zip(configs).zip(graphs) {
        match federated::classify_dag(&config) {
            Err(_) => {
                dropped += 1;
            }
            _ => survivors.push(PoolEntry {
                name,
                utilization: config.volume as f64 / config.period as f64,
                metrics: config,
                segments,
                graph,
            }),
        }
    }
    Ok((survivors, dropped))
}

/// Appends one JSON-Lines record for a single resampled trial: which pool
/// directory and *distinct* filenames (in draw order) were drawn, whether
/// the whole set was predicted ACCEPT under `real_cores` by each of
/// the three independent theories (`accepted` = Federated's batch
/// admission under `federated_variant`, also the proxy used for `laxity`
/// -- see this file's module doc; `vfed_accepted`/`dag_fluid_accepted` =
/// that algorithm's own), the
/// `real_cores` those theories were given, and this particular draw's own
/// *actual* `U_Sigma`/`u_norm` (`u_norm_actual = u_sigma_actual /
/// real_cores`) -- the bin's own `u_norm` is only the window's *target*
/// (see this file's module doc), so a caller that wants the realized
/// density of the exact DAGs it's about to boot, rather than the bin label
/// they were drawn from, reads these two instead.
/// A real-machine run reads `dags_dir`/`dags` to know exactly which files to
/// stage for an ACCEPTed trial (and, for calibration, may also boot a
/// sample of REJECTed ones to confirm they really fail admission), and
/// should read the ONE `*_accepted` column matching whichever algorithm it
/// actually boots.
#[allow(clippy::too_many_arguments)]
fn write_trial_record(
    out: &mut fs::File,
    u_norm: f64,
    trial: usize,
    dags_dir: &Path,
    set: &[&PoolEntry],
    federated_variant: FederatedVariant,
    accepted: bool,
    vfed_accepted: bool,
    dag_fluid_accepted: bool,
    sfs_accepted: bool,
    u_sigma_actual: f64,
    real_cores: u16,
) {
    let dags = set
        .iter()
        .map(|e| format!("\"{}\"", e.name))
        .collect::<Vec<_>>()
        .join(",");
    let u_norm_actual = u_sigma_actual / real_cores as f64;
    let line = format!(
        "{{\"u_norm\":{u_norm:.4},\"trial\":{trial},\"dags_dir\":\"{}\",\"dags\":[{dags}],\
         \"real_cores\":{real_cores},\"federated_variant\":\"{federated_variant:?}\",\"accepted\":{accepted},\"vfed_accepted\":{vfed_accepted},\
         \"dag_fluid_accepted\":{dag_fluid_accepted},\"sfs_accepted\":{sfs_accepted},\"u_sigma_actual\":{u_sigma_actual:.6},\
         \"u_norm_actual\":{u_norm_actual:.6}}}\n",
        dags_dir.display(),
    );
    // A single trial record failing to write isn't worth aborting a
    // long-running sweep over -- the aggregate ratio (this file's stdout
    // CSV) is unaffected either way.
    let _ = out.write_all(line.as_bytes());
}
