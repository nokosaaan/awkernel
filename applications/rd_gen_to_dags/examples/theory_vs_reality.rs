//! Per-`u_norm`-bin Federated acceptance-ratio sweep against a *fixed* real
//! core count, producing a trial manifest a real-machine run can replay --
//! the offline half of a theory-vs-reality gap study (predicted admission +
//! [`min_dedicated_cores`] preconditions vs. actually measured deadline
//! misses on real hardware for the *same* task sets).
//!
//! [`min_dedicated_cores`]: awkernel_async_lib::dag_sched::metrics::DagMetrics::min_dedicated_cores
//!
//! # How this differs from `acceptance_ratio.rs`
//! That file derives `m = ceil(U_Sigma / u_norm)` *per resampled trial*, so
//! `m` is whatever the trial's own draw needs -- never tied to any real
//! machine's actual core count (see its own module doc). This file inverts
//! that: `real_cores` (a CLI argument, e.g. the target machine's actual
//! dag-pool core count) is fixed for the whole run, and `u_norm` is instead
//! achieved on the *generation* side -- each `u_norm` bin is expected to be
//! its own RD-Gen-generated pool directory (`<bins_root_dir>/u_norm_<value>/
//! DAGs/dag_*.yaml`, i.e. `run_generator.py`'s own `-d` output layout) whose
//! `Whole-DAG utilization` range was centered on `u_norm * real_cores /
//! dags_per_set` at generation time, so a plain random draw of `dags_per_set`
//! DAGs from that bin already lands close to the target `u_norm` -- no
//! rejection sampling, no post-hoc rescaling of `C`/`T`.
//!
//! # Per-DAG prefilter
//! Before resampling, every DAG in a bin that `classify_dag` already calls
//! infeasible (`D < L`) is dropped from that bin's pool: no scheduling
//! policy, however parallel, can finish critical-path work faster than the
//! critical path itself takes, so this is a universal, algorithm-agnostic
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
//! own static admission gate (`federated_batch_feasible`), V-Fed's own
//! batch planner (`vfed::is_batch_feasible`), and DAG-Fluid's own fluid-
//! capacity check (`dag_fluid::is_batch_feasible`) -- mirroring
//! `acceptance_ratio.rs`'s own three-column CSV, just with `real_cores`
//! fixed instead of a derived `m`. `manifest_jsonl` records all three
//! per-trial booleans (`accepted`/`vfed_accepted`/`dag_fluid_accepted`),
//! so a real-machine comparison can read the ONE column matching whichever
//! algorithm it actually booted, instead of reusing Federated's own
//! decision as a stand-in for every algorithm (which conflates "the real
//! machine deviated from theory" with "this algorithm's own theory was
//! never actually evaluated"). `laxity` has no admission test of its own
//! here, same as in `acceptance_ratio.rs`: it reuses the SAME static gate
//! as Federated (see `rd_gen_to_dags::build_dag`'s own `laxity` arm --
//! only the in-batch node ordering differs, not the admission math), so
//! `accepted` is the correct, non-proxy value for it too, not merely a
//! stand-in.
//!
//! No file staging, no kernel build, no boot: `manifest_jsonl` records
//! which pool directory and filenames each trial drew and whether it was
//! predicted ACCEPT/REJECT under each theory; a separate script (see
//! `run_realboot_evaluation.py`'s own pattern) reads it, stages the
//! ACCEPTed trials' files, boots them for real under the matching
//! algorithm, and compares the resulting measured deadline-miss rate
//! against this file's own per-bin acceptance ratio.
//!
//! Usage: `theory_vs_reality <bins_root_dir> <real_cores> <dags_per_set>
//! <trials_per_bin> <manifest_jsonl_path>`, prints
//! `u_norm,federated_ratio,vfed_ratio,dag_fluid_ratio` CSV to stdout (one
//! row per discovered bin, ascending `u_norm`).

use std::{
    env, fs,
    io::Write,
    path::{Path, PathBuf},
    process::ExitCode,
};

use awkernel_async_lib::dag_sched::{
    metrics::DagMetrics,
    policy::{
        federated::{self, TaskClass as FedTaskClass},
        vfed::{self, PackingStrategy},
    },
};
use rand::seq::IndexedRandom;
use rd_gen_to_dags::dag_fluid::{self, Segment};

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();
    let [_, bins_root_dir, real_cores, dags_per_set, trials_per_bin, manifest_jsonl_path] =
        args.as_slice()
    else {
        eprintln!(
            "usage: theory_vs_reality <bins_root_dir> <real_cores> <dags_per_set> \
             <trials_per_bin> <manifest_jsonl_path>"
        );
        return ExitCode::from(2);
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

    let bins = match discover_bins(Path::new(bins_root_dir)) {
        Ok(b) if !b.is_empty() => b,
        Ok(_) => {
            eprintln!("'{bins_root_dir}' has no 'u_norm_<value>' subdirectories");
            return ExitCode::FAILURE;
        }
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::FAILURE;
        }
    };

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

    println!("u_norm,federated_ratio,vfed_ratio,dag_fluid_ratio");
    for (u_norm, bin_dir) in bins {
        let dags_dir = bin_dir.join("DAGs");
        let (pool, dropped) = match load_and_prefilter_pool(&dags_dir) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("u_norm={u_norm:.4}: {e}");
                continue;
            }
        };
        eprintln!(
            "u_norm={u_norm:.4}: {} feasible-by-itself DAGs, {dropped} dropped by prefilter",
            pool.len(),
        );
        if pool.len() < dags_per_set {
            eprintln!(
                "u_norm={u_norm:.4}: pool has only {} DAGs after prefilter (< dags_per_set={dags_per_set}); skipping bin"
            , pool.len());
            continue;
        }

        let mut fed_accepted = 0usize;
        let mut vfed_accepted = 0usize;
        let mut dag_fluid_accepted = 0usize;
        for trial in 0..trials_per_bin {
            let set: Vec<&(String, DagMetrics, Vec<Segment>)> =
                pool.choose_multiple(&mut rng, dags_per_set).collect();
            let metrics: Vec<DagMetrics> = set.iter().map(|(_, m, _)| *m).collect();

            let fed_ok = federated_batch_feasible(&metrics, real_cores);
            if fed_ok {
                fed_accepted += 1;
            }
            let vfed_ok = vfed::is_batch_feasible(&metrics, real_cores, PackingStrategy::BestFit);
            if vfed_ok {
                vfed_accepted += 1;
            }
            let dag_fluid_entries: Vec<(u64, u64, u64, u64, &[Segment])> = set
                .iter()
                .map(|(_, m, segments)| (m.volume, m.period, m.critical_path, m.relative_deadline, segments.as_slice()))
                .collect();
            let dag_fluid_ok = dag_fluid::is_batch_feasible(&dag_fluid_entries, real_cores);
            if dag_fluid_ok {
                dag_fluid_accepted += 1;
            }

            let u_sigma_actual: f64 = metrics.iter().map(|d| d.volume as f64 / d.period as f64).sum();
            write_trial_record(
                &mut manifest, u_norm, trial, &dags_dir, &set, fed_ok, vfed_ok, dag_fluid_ok,
                u_sigma_actual, real_cores,
            );
        }

        println!(
            "{u_norm:.4},{:.2},{:.2},{:.2}",
            100.0 * fed_accepted as f64 / trials_per_bin as f64,
            100.0 * vfed_accepted as f64 / trials_per_bin as f64,
            100.0 * dag_fluid_accepted as f64 / trials_per_bin as f64,
        );
    }

    ExitCode::SUCCESS
}

/// `<bins_root_dir>/u_norm_<value>/` subdirectories, sorted ascending by the
/// parsed `u_norm` value. Non-matching entries (e.g. a stray file) are
/// silently skipped.
fn discover_bins(bins_root_dir: &Path) -> Result<Vec<(f64, PathBuf)>, String> {
    let mut bins: Vec<(f64, PathBuf)> = fs::read_dir(bins_root_dir)
        .map_err(|e| format!("cannot read directory '{}': {e}", bins_root_dir.display()))?
        .filter_map(|entry| entry.ok())
        .filter(|entry| entry.path().is_dir())
        .filter_map(|entry| {
            let name = entry.file_name();
            let u_norm = parse_u_norm_from_dir_name(name.to_str()?)?;
            Some((u_norm, entry.path()))
        })
        .collect();
    bins.sort_by(|a, b| a.0.total_cmp(&b.0));
    Ok(bins)
}

fn parse_u_norm_from_dir_name(name: &str) -> Option<f64> {
    let v: f64 = name.strip_prefix("u_norm_")?.parse().ok()?;
    (v > 0.0).then_some(v)
}

/// Load every `dag_<N>.yaml` directly inside `dags_dir` as-is -- RD-Gen's own
/// 'Constrained' deadline mode already guarantees `D <= period` by
/// construction (see this file's own module doc for why this file does NOT
/// override deadline/period the way `acceptance_ratio.rs` does) -- then drop
/// any DAG `classify_dag` already calls infeasible (see this file's own
/// "Per-DAG prefilter" doc for why only that case, not a plain
/// `Heavy { required_cores }`, is dropped here). Returns the survivors
/// paired with their filename and DAG-Fluid segment decomposition (needed
/// for `dag_fluid::is_batch_feasible`), plus how many were dropped.
fn load_and_prefilter_pool(
    dags_dir: &Path,
) -> Result<(Vec<(String, DagMetrics, Vec<Segment>)>, usize), String> {
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

    let configs = rd_gen_to_dags::dag_metrics_and_fluid_segments_from_yaml(&yaml_refs)
        .map_err(|e| format!("failed to parse DAG pool in '{}': {e}", dags_dir.display()))?;

    let mut survivors = Vec::new();
    let mut dropped = 0usize;
    for (name, (config, segments)) in names.into_iter().zip(configs) {
        match federated::classify_dag(&config) {
            Err(_) => {
                dropped += 1;
            }
            _ => survivors.push((name, config, segments)),
        }
    }
    Ok((survivors, dropped))
}

/// Appends one JSON-Lines record for a single resampled trial: which pool
/// directory and *distinct* filenames (in draw order) were drawn, whether
/// whether the whole set was predicted ACCEPT under `real_cores` by each of
/// the three independent theories (`accepted` = Federated's own gate, also
/// the correct value for `laxity` -- see this file's module doc;
/// `vfed_accepted`/`dag_fluid_accepted` = that algorithm's own), and this
/// particular draw's own *actual* `U_Sigma`/`u_norm` (`u_norm_actual =
/// u_sigma_actual / real_cores`) -- the bin's own `u_norm` is only the
/// generation-time *target* (see this file's module doc), so a caller that
/// wants the realized density of the exact 8 DAGs it's about to boot,
/// rather than the bin label they were drawn from, reads these two instead.
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
    set: &[&(String, DagMetrics, Vec<Segment>)],
    accepted: bool,
    vfed_accepted: bool,
    dag_fluid_accepted: bool,
    u_sigma_actual: f64,
    real_cores: u16,
) {
    let dags = set
        .iter()
        .map(|(name, _, _)| format!("\"{name}\""))
        .collect::<Vec<_>>()
        .join(",");
    let u_norm_actual = u_sigma_actual / real_cores as f64;
    let line = format!(
        "{{\"u_norm\":{u_norm:.4},\"trial\":{trial},\"dags_dir\":\"{}\",\"dags\":[{dags}],\
         \"accepted\":{accepted},\"vfed_accepted\":{vfed_accepted},\
         \"dag_fluid_accepted\":{dag_fluid_accepted},\"u_sigma_actual\":{u_sigma_actual:.6},\
         \"u_norm_actual\":{u_norm_actual:.6}}}\n",
        dags_dir.display(),
    );
    // A single trial record failing to write isn't worth aborting a
    // long-running sweep over -- the aggregate ratio (this file's stdout
    // CSV) is unaffected either way.
    let _ = out.write_all(line.as_bytes());
}

/// Verbatim copy of `acceptance_ratio.rs`'s/`predict_admission.rs`'s own
/// pure Federated batch check (see `acceptance_ratio.rs` for why it's
/// reimplemented rather than calling the real `resource`-ledger-backed
/// `federated::admit_dag`).
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
            Err(_) => return false, // Infeasible regardless of resources.
        }
    }

    if heavy_cores > num_cores as u32 {
        return false;
    }
    light_utilization <= (num_cores as u32 - heavy_cores) as f64
}
