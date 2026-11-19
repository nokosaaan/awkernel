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
//! <u_norm_step> [<seed>]]` (defaults `0.10 1.00 0.05`, random seed), or
//! `theory_vs_reality` without arguments to be asked for each value (stdin;
//! prompts go to stderr). Before drawing, it prints each `u_norm`'s window
//! and how many pool DAGs fall in it (a window with fewer than
//! `dags_per_set` is skipped), and writes the run's values, the seed and the
//! equivalent command line to `<manifest_jsonl_path>.params.json`. It prints
//! `u_norm,federated_ratio,vfed_ratio,dag_fluid_ratio,sfs_ratio` CSV to stdout
//! (one row per `u_norm` bin with enough DAGs in its window, ascending
//! `u_norm`) and writes the same CSV to `<manifest_jsonl_path>.ratio.csv`.
//!
//! Results stay out of the awkernel tree: the interactive default manifest is
//! `<results root>/<YYYYmmdd-HHMMSS>_m<m>_n<N>/manifest.jsonl`, where the
//! results root is `$AWKERNEL_TRIALS_DIR` or
//! `~/ws/awkernel_script/rd_gen_trials`. Missing parent directories of the
//! manifest are created, and a leading `~/` in an interactive answer is
//! expanded to `$HOME`.

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
use rand::{SeedableRng, rngs::StdRng, seq::IndexedRandom};

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

/// Everything one run needs: from the command line, or asked on stdin
/// when the program is started without arguments (see [`Params::ask`]).
struct Params {
    pool_dir: String,
    real_cores: u16,
    dags_per_set: usize,
    trials_per_bin: usize,
    manifest_path: String,
    /// Keep the manifest's existing records (always, on the command line).
    append: bool,
    u_norm_min: f64,
    u_norm_max: f64,
    u_norm_step: f64,
    /// Seed of the set drawing; drawn at random (and reported) if not given.
    seed: u64,
}

const USAGE: &str = "usage: theory_vs_reality <pool_dir> <real_cores> <dags_per_set> <trials_per_bin> \
                     <manifest_jsonl_path> [<u_norm_min> <u_norm_max> <u_norm_step> [<seed>]]\n       \
                     theory_vs_reality            (no arguments: asks for each value)";

/// Results root when `$AWKERNEL_TRIALS_DIR` is unset, relative to `$HOME`:
/// next to the trial logs `scripts/real_machine_trial.py` writes for the same
/// manifest, in the awkernel_script repository.
const DEFAULT_TRIALS_DIR_RELATIVE_TO_HOME: &str = "ws/awkernel_script/rd_gen_trials";

/// `$AWKERNEL_TRIALS_DIR`, else `~/ws/awkernel_script/rd_gen_trials`.
fn trials_root() -> PathBuf {
    match env::var_os("AWKERNEL_TRIALS_DIR") {
        Some(dir) if !dir.is_empty() => PathBuf::from(dir),
        _ => PathBuf::from(env::var("HOME").unwrap_or_default()).join(DEFAULT_TRIALS_DIR_RELATIVE_TO_HOME),
    }
}

/// Local time as `YYYYmmdd-HHMMSS` (from `date`, which knows the time zone),
/// or the Unix time in seconds if `date` is unavailable.
fn timestamp() -> String {
    std::process::Command::new("date")
        .arg("+%Y%m%d-%H%M%S")
        .output()
        .ok()
        .filter(|out| out.status.success())
        .and_then(|out| String::from_utf8(out.stdout).ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs().to_string())
                .unwrap_or_default()
        })
}

/// A leading `~` or `~/` replaced by `$HOME` (the shell does this for command
/// line arguments, but not for answers typed at a prompt).
fn expand_tilde(v: &str) -> String {
    match (v.strip_prefix('~'), env::var("HOME")) {
        (Some(rest), Ok(home)) if rest.is_empty() || rest.starts_with('/') => format!("{home}{rest}"),
        _ => v.to_string(),
    }
}

/// Default pool for the interactive mode: RD-Gen's wide pool
/// (`scripts/generate_wide_pool.py`), next to this repository.
const DEFAULT_POOL_RELATIVE_TO_HOME: &str = "azumi-lab/RD-Gen/test/awkernel_wide_pool";

fn parse_positive<T: std::str::FromStr + PartialOrd + Default>(
    name: &str,
    v: &str,
) -> Result<T, String> {
    match v.trim().parse::<T>() {
        Ok(n) if n > T::default() => Ok(n),
        _ => Err(format!("{name} must be a positive number, got '{v}'")),
    }
}

fn check_u_norm_range(lo: f64, hi: f64, step: f64) -> Result<(), String> {
    if lo > 0.0 && hi >= lo && step > 0.0 {
        Ok(())
    } else {
        Err(format!(
            "u_norm_min/u_norm_max/u_norm_step must satisfy 0 < min <= max and step > 0, got {lo} {hi} {step}"
        ))
    }
}

impl Params {
    fn from_args(args: &[String]) -> Result<Self, String> {
        let (p, c, d, t, m, range, seed) = match args {
            [p, c, d, t, m] => (p, c, d, t, m, None, None),
            [p, c, d, t, m, lo, hi, step] => (p, c, d, t, m, Some((lo, hi, step)), None),
            [p, c, d, t, m, lo, hi, step, s] => (p, c, d, t, m, Some((lo, hi, step)), Some(s)),
            _ => return Err(USAGE.to_string()),
        };
        let (u_norm_min, u_norm_max, u_norm_step) = match range {
            None => (DEFAULT_U_NORM_MIN, DEFAULT_U_NORM_MAX, DEFAULT_U_NORM_STEP),
            Some((lo, hi, step)) => (
                lo.parse::<f64>()
                    .map_err(|_| format!("bad u_norm_min '{lo}'"))?,
                hi.parse::<f64>()
                    .map_err(|_| format!("bad u_norm_max '{hi}'"))?,
                step.parse::<f64>()
                    .map_err(|_| format!("bad u_norm_step '{step}'"))?,
            ),
        };
        check_u_norm_range(u_norm_min, u_norm_max, u_norm_step)?;
        let seed = match seed {
            Some(s) => s
                .parse::<u64>()
                .map_err(|_| format!("seed must be a u64, got '{s}'"))?,
            None => rand::random::<u64>(),
        };
        Ok(Self {
            pool_dir: p.clone(),
            real_cores: parse_positive("real_cores", c)?,
            dags_per_set: parse_positive("dags_per_set", d)?,
            trials_per_bin: parse_positive("trials_per_bin", t)?,
            manifest_path: m.clone(),
            append: true,
            u_norm_min,
            u_norm_max,
            u_norm_step,
            seed,
        })
    }

    /// Ask for every value on stderr/stdin (stdout stays the CSV), with the
    /// default in brackets taken on an empty answer.
    fn ask() -> Result<Self, String> {
        let default_pool = env::var("HOME")
            .map(|home| format!("{home}/{DEFAULT_POOL_RELATIVE_TO_HOME}"))
            .unwrap_or_default();
        let pool_dir = ask_until("DAG pool directory", Some(&default_pool), |v| {
            let v = expand_tilde(v);
            if Path::new(&v).is_dir() {
                Ok(v)
            } else {
                Err(format!("'{v}' is not a directory"))
            }
        })?;
        let real_cores: u16 =
            ask_until("real cores m (the DAG-pool core count)", Some("14"), |v| {
                parse_positive("m", v)
            })?;
        let dags_per_set: usize =
            ask_until("DAGs per set N", Some("8"), |v| parse_positive("N", v))?;
        let trials_per_bin: usize = ask_until("trials per u_norm", Some("500"), |v| {
            parse_positive("trials", v)
        })?;
        let (u_norm_min, u_norm_max, u_norm_step) = loop {
            let lo: f64 = ask_until("u_norm min", Some("0.10"), |v| {
                parse_positive("u_norm min", v)
            })?;
            let hi: f64 = ask_until("u_norm max", Some("1.00"), |v| {
                parse_positive("u_norm max", v)
            })?;
            let step: f64 = ask_until("u_norm step", Some("0.05"), |v| {
                parse_positive("u_norm step", v)
            })?;
            match check_u_norm_range(lo, hi, step) {
                Ok(()) => break (lo, hi, step),
                Err(e) => eprintln!("  {e}"),
            }
        };
        let random_seed = rand::random::<u64>().to_string();
        let seed: u64 = ask_until("seed (blank: random)", Some(&random_seed), |v| {
            v.parse::<u64>()
                .map_err(|_| format!("seed must be a u64, got '{v}'"))
        })?;
        let default_manifest = trials_root()
            .join(format!("{}_m{real_cores}_n{dags_per_set}", timestamp()))
            .join("manifest.jsonl")
            .display()
            .to_string();
        let manifest_path = ask_until("manifest JSONL path", Some(&default_manifest), |v| {
            Ok(expand_tilde(v))
        })?;
        let append = if Path::new(&manifest_path).exists() {
            let answer = ask_until(
                "manifest exists: [a]ppend / [o]verwrite / [q]uit",
                Some("q"),
                |v| match v {
                    "a" | "o" | "q" => Ok(v.to_string()),
                    _ => Err("answer a, o or q".to_string()),
                },
            )?;
            match answer.as_str() {
                "a" => true,
                "o" => false,
                _ => return Err("cancelled".to_string()),
            }
        } else {
            true
        };
        Ok(Self {
            pool_dir,
            real_cores,
            dags_per_set,
            trials_per_bin,
            manifest_path,
            append,
            u_norm_min,
            u_norm_max,
            u_norm_step,
            seed,
        })
    }

    /// The command line that repeats this run (the manifest is appended to).
    fn command_line(&self) -> String {
        format!(
            "theory_vs_reality {} {} {} {} {} {} {} {} {}",
            self.pool_dir,
            self.real_cores,
            self.dags_per_set,
            self.trials_per_bin,
            self.manifest_path,
            self.u_norm_min,
            self.u_norm_max,
            self.u_norm_step,
            self.seed
        )
    }
}

/// Print `question [default]: ` to stderr and read one line until `parse`
/// accepts it (an empty line takes `default`).
fn ask_until<T>(
    question: &str,
    default: Option<&str>,
    parse: impl Fn(&str) -> Result<T, String>,
) -> Result<T, String> {
    let stdin = std::io::stdin();
    loop {
        match default {
            Some(d) if !d.is_empty() => eprint!("{question} [{d}]: "),
            _ => eprint!("{question}: "),
        }
        let _ = std::io::stderr().flush();
        let mut line = String::new();
        if stdin.read_line(&mut line).map_err(|e| e.to_string())? == 0 {
            return Err("stdin closed".to_string());
        }
        let line = line.trim();
        let value = if line.is_empty() {
            default.unwrap_or("")
        } else {
            line
        };
        match parse(value) {
            Ok(v) => return Ok(v),
            Err(e) => eprintln!("  {e}"),
        }
    }
}

/// `u_norm`'s per-DAG utilization window `[lo, hi]`.
fn window_of(u_norm: f64, per_dag_scale: f64, delta: f64) -> (f64, f64) {
    let center = u_norm * per_dag_scale;
    (center - delta, center + delta)
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().skip(1).collect();
    let interactive = args.is_empty();
    let params = if interactive {
        Params::ask()
    } else {
        Params::from_args(&args)
    };
    let params = match params {
        Ok(p) => p,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::from(2);
        }
    };
    let Params {
        real_cores,
        dags_per_set,
        trials_per_bin,
        u_norm_min,
        u_norm_max,
        u_norm_step,
        seed,
        ..
    } = params;
    let manifest_jsonl_path = params.manifest_path.as_str();

    let dags_dir = resolve_dags_dir(Path::new(&params.pool_dir));
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

    let per_dag_scale = real_cores as f64 / dags_per_set as f64;
    let delta = WINDOW_FRACTION_OF_STEP * u_norm_step * per_dag_scale;
    let u_norms = u_norm_values(u_norm_min, u_norm_max, u_norm_step);

    // Coverage check before drawing anything: a window with fewer DAGs
    // than dags_per_set is skipped (no CSV row, no manifest record).
    let window_counts: Vec<usize> = u_norms
        .iter()
        .map(|&u_norm| {
            let (lo, hi) = window_of(u_norm, per_dag_scale, delta);
            pool.iter()
                .filter(|e| e.utilization >= lo && e.utilization <= hi)
                .count()
        })
        .collect();
    let pool_max_u = pool.iter().map(|e| e.utilization).fold(0.0f64, f64::max);
    eprintln!(
        "m={real_cores} N={dags_per_set}: per-DAG utilization = u_norm * {per_dag_scale:.4} \
         +- {delta:.4} (pool reaches {pool_max_u:.3})"
    );
    eprintln!("  u_norm   window             DAGs");
    let mut short_bins = Vec::new();
    for (&u_norm, &count) in u_norms.iter().zip(&window_counts) {
        let (lo, hi) = window_of(u_norm, per_dag_scale, delta);
        let mark = if count < dags_per_set {
            short_bins.push(u_norm);
            "  <- too few, skipped"
        } else {
            ""
        };
        eprintln!("  {u_norm:.4}   [{lo:.4}, {hi:.4}]   {count:5}{mark}");
    }
    if !short_bins.is_empty() {
        eprintln!(
            "{} of {} u_norm points have fewer than N={dags_per_set} DAGs in their window and \
             will be skipped (the pool must reach u_norm_max * m / N = {:.3})",
            short_bins.len(),
            u_norms.len(),
            u_norm_max * per_dag_scale + delta,
        );
        if interactive {
            match ask_until("continue anyway? [y/N]", Some("n"), |v| match v {
                "y" | "n" => Ok(v == "y"),
                _ => Err("answer y or n".to_string()),
            }) {
                Ok(true) => {}
                _ => return ExitCode::from(2),
            }
        }
    }

    if let Some(parent) = Path::new(manifest_jsonl_path).parent() {
        if !parent.as_os_str().is_empty() {
            if let Err(e) = fs::create_dir_all(parent) {
                eprintln!("cannot create '{}': {e}", parent.display());
                return ExitCode::FAILURE;
            }
        }
    }
    let mut manifest = match fs::OpenOptions::new()
        .create(true)
        .write(true)
        .append(params.append)
        .truncate(!params.append)
        .open(manifest_jsonl_path)
    {
        Ok(f) => f,
        Err(e) => {
            eprintln!("cannot open '{manifest_jsonl_path}' for writing: {e}");
            return ExitCode::FAILURE;
        }
    };

    // The values of this run next to the manifest, so it can be repeated.
    let params_path = format!("{manifest_jsonl_path}.params.json");
    let params_json = format!(
        "{{\"pool_dir\":\"{}\",\"real_cores\":{real_cores},\"dags_per_set\":{dags_per_set},\
         \"trials_per_bin\":{trials_per_bin},\"u_norm_min\":{u_norm_min},\"u_norm_max\":{u_norm_max},\
         \"u_norm_step\":{u_norm_step},\"seed\":{seed},\"window_fraction_of_step\":{WINDOW_FRACTION_OF_STEP},\
         \"pool_dags\":{},\"skipped_u_norm\":{:?},\"command\":\"{}\"}}\n",
        dags_dir.display(),
        pool.len(),
        short_bins,
        params.command_line(),
    );
    if let Err(e) = fs::write(&params_path, params_json) {
        eprintln!("cannot write '{params_path}': {e}");
    }
    eprintln!("seed {seed}; repeat with: {}", params.command_line());
    eprintln!("manifest: {manifest_jsonl_path}");

    let mut rng = StdRng::seed_from_u64(seed);
    let per_dag_scale = real_cores as f64 / dags_per_set as f64;
    let delta = WINDOW_FRACTION_OF_STEP * u_norm_step * per_dag_scale;

    // The CSV goes to stdout and, so it is kept with the manifest, to a file.
    let ratio_path = format!("{manifest_jsonl_path}.ratio.csv");
    let mut ratio_csv = String::from("u_norm,federated_ratio,vfed_ratio,dag_fluid_ratio,sfs_ratio\n");
    print!("{ratio_csv}");
    for &u_norm in &u_norms {
        let (lo, hi) = window_of(u_norm, per_dag_scale, delta);
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

        let row = format!(
            "{u_norm:.4},{:.2},{:.2},{:.2},{:.2}\n",
            100.0 * fed_accepted as f64 / trials_per_bin as f64,
            100.0 * vfed_accepted as f64 / trials_per_bin as f64,
            100.0 * dag_fluid_accepted as f64 / trials_per_bin as f64,
            100.0 * sfs_accepted as f64 / trials_per_bin as f64,
        );
        print!("{row}");
        ratio_csv.push_str(&row);
    }

    if let Err(e) = fs::write(&ratio_path, ratio_csv) {
        eprintln!("cannot write '{ratio_path}': {e}");
    }
    eprintln!("wrote {manifest_jsonl_path}, {params_path}, {ratio_path}");

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
