//! One-shot acceptance-ratio sweep: classical Federated vs V-Fed, matching
//! the TPDS23 V-Fed paper's own evaluation methodology (Section 7) as
//! closely as RD-Gen allows: a single shared pool of DAGs whose per-vertex
//! WCET is drawn `Uniform[1, 100]` *independent* of any target utilization
//! (RD-Gen's `Execution time: Random` property, not `Whole-DAG utilization`
//! — see `RD-Gen/sample_config/chain_based/my_chain_theory_pool.yaml`), and
//! for each `U_norm` point, `M = ceil(U_Sigma / U_norm)` computed *per
//! resampled task set* from that set's own actual total utilization
//! `U_Sigma = sum(C_i / T_i)` — not a fixed `--cores` swept via generation,
//! which was this file's earlier (pre-paper-reading) design.
//!
//! # Why one shared pool instead of fresh DAGs per `U_norm` point
//! The paper generates ~1000 entirely new task sets per point. Since this
//! pool's WCET generation is already independent of `U_norm` (unlike the
//! old per-point `Whole-DAG utilization` sweep), regenerating per point buys
//! nothing statistically — resampling with replacement from one large shared
//! pool (`--pool-dir`) gives the same distribution at a fraction of the
//! generation cost. This was an explicit, agreed simplification.
//!
//! # Why resampling task sets, not the pool's files as-is
//! Each `dag_<N>.yaml` in the pool is one *independent* DAG. Testing each of
//! the pool's DAGs alone against a fresh core budget was tried first (in an
//! earlier version of this file) and rejected: a solo DAG's admission
//! reduces to the *identical* `required_cores <= M` test for both policies
//! (heavy) or trivially succeeds for both (light) — there is no other DAG's
//! leftover capacity for V-Fed's passive-VP mechanism to draw on, so that
//! methodology can never show any difference between the two policies,
//! correctness aside. Instead, for each `U_norm` level this resamples
//! [`TRIALS_PER_LEVEL`] task sets of [`DAGS_PER_SET`] *distinct* DAGs each
//! (without replacement *within* a trial, though of course the same pool
//! entry can and does recur *across* different trials), and checks whether
//! the *whole set* is jointly admittable (all-or-nothing) under each policy
//! at that set's own computed `M`. The paper's own methodology draws with
//! replacement (so one draw's 8 DAGs need not be distinct); this file
//! deviates from strict reproduction here because a distinct-per-trial set
//! is also the input to a real-machine trial (see `write_trial_record`'s
//! own doc), which stages each drawn DAG as a same-named `dag_<N>.yaml`
//! file -- a repeated draw would silently collapse to one file, embedding
//! fewer than `DAGS_PER_SET` distinct DAGs while this file's own admission
//! check still (correctly) counted it twice.
//!
//! # Deadline: overridden post-parse, not RD-Gen's own generated value
//! The paper draws `D_i` uniform in `[L_i, L_i / ALPHA]` (`ALPHA = 0.3` for
//! its basic configuration) — a different distribution from RD-Gen's own
//! `Constrained` deadline mode (uniform in `(L, T]`). Rather than fight
//! RD-Gen's `deadline_setter`, the pool config uses `Deadline mode:
//! 'Arbitrary'` — the only mode that never rejects a DAG for an L-vs-T
//! reason (`Implicit` and `Constrained` both raise `BuildFailedError`
//! whenever the critical path alone exceeds the period, which would bias
//! the pool against high-critical-path draws) — and this file overrides
//! `relative_deadline` directly from each DAG's own parsed `critical_path`,
//! once per pooled DAG at load time (not re-randomized on later resample
//! draws of the same DAG — matches "D fixed per DAG when its task set is
//! generated"). See [`ALPHA`] and [`override_deadline`].
//!
//! Both feasibility checks are pure functions of the sampled `DagMetrics`
//! batch and a computed `m` — neither touches `dag_sched::resource`'s or
//! `dag_sched::policy::vfed`'s real, process-global ledgers, so many
//! resampled trials run safely in one process with no reset/rollback
//! machinery needed.
//!
//! Usage: `acceptance_ratio <pool_dir> <u_norm_min> <u_norm_max> <u_norm_step>
//! [trials_jsonl_path]`, prints `u_norm,federated_ratio,vfed_ratio,
//! dag_fluid_ratio` CSV to stdout.
//!
//! # Recording individual trials (`trials_jsonl_path`)
//! The ratios above are an aggregate over resampled task sets that are
//! otherwise generated and discarded. When a real-machine trial needs to
//! reproduce *one specific* resampled task set (to compare this file's own
//! admission prediction against actual real-machine boot/spawn behavior on
//! the identical 8 DAGs -- see `scripts/real_machine_trial.py` and the
//! Notion "実機ホスト環境移行の引き継ぎ" page's design), that draw must be
//! recoverable after the fact. Passing `trials_jsonl_path` appends one JSON
//! object per trial (JSON Lines, so a run can be interrupted/resumed by
//! just appending), e.g.:
//! `{"u_norm":0.5000,"trial":37,"dags":["dag_1819.yaml","dag_2643.yaml",...],"federated_accepted":true,"vfed_accepted":true,"dag_fluid_accepted":false}`
//! A specific reproducible group is then just a `jq` filter away, e.g.
//! `jq 'select(.federated_accepted and .vfed_accepted)' trials.jsonl | head -1`.
//!
//! # DAG-Fluid's third column
//! DAG-Fluid (Guan, Qiao, Han, IEEE TC 2020/2021) is included here as a
//! third, *static-only* schedulability test (segment decomposition +
//! Algorithm 2's execution-rate assignment, see `rd_gen_to_dags::dag_fluid`'s
//! own doc) — no runtime/dispatch layer exists for it in this codebase, so
//! this column is a pure admission-math comparison, exactly like the other
//! two. Segments are structure-derived and unaffected by
//! `override_deadline`/`override_period`/`assign_max_parallelism` (which
//! only touch each DAG's `DagMetrics` half), so they're computed once at
//! pool-load time and carried alongside each `DagMetrics` through every
//! later resample.

use std::{
    env, fs,
    io::Write,
    path::Path,
    process::ExitCode,
};

use awkernel_async_lib::dag_sched::{
    metrics::DagMetrics,
    policy::{
        federated::{self, TaskClass as FedTaskClass},
        vfed::{self, PackingStrategy},
    },
};
use rand::{seq::IndexedRandom, Rng};
use rd_gen_to_dags::dag_fluid::{self, Segment};

/// N in the paper's own "N=8 DAGs per task set".
const DAGS_PER_SET: usize = 8;

/// Resampled task sets per `U_norm` level (matches the paper's own "1000
/// task sets generated per configuration", Section 7).
const TRIALS_PER_LEVEL: usize = 1000;

/// The paper's basic-configuration deadline-generation shape parameter:
/// `D_i ~ Uniform[L_i, L_i / ALPHA]`.
const ALPHA: f64 = 0.3;

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();
    let (pool_dir, u_norm_min, u_norm_max, u_norm_step, trials_jsonl_path) = match args.as_slice()
    {
        [_, pool_dir, u_norm_min, u_norm_max, u_norm_step] => {
            (pool_dir, u_norm_min, u_norm_max, u_norm_step, None)
        }
        [_, pool_dir, u_norm_min, u_norm_max, u_norm_step, trials_jsonl_path] => {
            (pool_dir, u_norm_min, u_norm_max, u_norm_step, Some(trials_jsonl_path))
        }
        _ => {
            eprintln!(
                "usage: acceptance_ratio <pool_dir> <u_norm_min> <u_norm_max> <u_norm_step> [trials_jsonl_path]"
            );
            return ExitCode::from(2);
        }
    };
    let (u_norm_min, u_norm_max, u_norm_step) =
        match (parse_positive_f64(u_norm_min), parse_positive_f64(u_norm_max), parse_positive_f64(u_norm_step)) {
            (Some(a), Some(b), Some(c)) if a <= b => (a, b, c),
            _ => {
                eprintln!(
                    "u_norm_min/u_norm_max/u_norm_step must be positive numbers with min <= max, \
                     got '{u_norm_min}', '{u_norm_max}', '{u_norm_step}'"
                );
                return ExitCode::from(2);
            }
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

    let mut trials_jsonl = trials_jsonl_path.map(|path| {
        fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .unwrap_or_else(|e| panic!("cannot open '{path}' for appending: {e}"))
    });

    println!("u_norm,federated_ratio,vfed_ratio,dag_fluid_ratio");

    let mut u_norm = u_norm_min;
    while u_norm <= u_norm_max + u_norm_step / 2.0 {
        let mut fed_accepted = 0usize;
        let mut vfed_accepted = 0usize;
        let mut dag_fluid_accepted = 0usize;

        for trial in 0..TRIALS_PER_LEVEL {
            let set: Vec<&(String, DagMetrics, Vec<Segment>)> =
                pool.choose_multiple(&mut rng, DAGS_PER_SET).collect();
            let metrics: Vec<DagMetrics> = set.iter().map(|(_, m, _)| *m).collect();

            let u_sigma: f64 = metrics.iter().map(|d| d.volume as f64 / d.period as f64).sum();
            // u_sigma > 0 always (every WCET is >= 1), so this is never 0/0;
            // ceil() can still round down to 0 when u_norm is large enough
            // that a fractional core would suffice, which we round up to 1
            // since M=0 cores can admit nothing.
            let m = ((u_sigma / u_norm).ceil() as u16).max(1);

            let fed_ok = federated_batch_feasible(&metrics, m);
            if fed_ok {
                fed_accepted += 1;
            }
            let vfed_ok = vfed::is_batch_feasible(&metrics, m, PackingStrategy::BestFit);
            if vfed_ok {
                vfed_accepted += 1;
            }
            let dag_fluid_entries: Vec<(u64, u64, u64, u64, &[Segment])> = set
                .iter()
                .map(|(_, metrics, segments)| {
                    (
                        metrics.volume,
                        metrics.period,
                        metrics.critical_path,
                        metrics.relative_deadline,
                        segments.as_slice(),
                    )
                })
                .collect();
            let dag_fluid_ok = dag_fluid::is_batch_feasible(&dag_fluid_entries, m);
            if dag_fluid_ok {
                dag_fluid_accepted += 1;
            }

            if let Some(f) = trials_jsonl.as_mut() {
                write_trial_record(f, u_norm, trial, &set, fed_ok, vfed_ok, dag_fluid_ok);
            }
        }

        println!(
            "{u_norm:.4},{:.2},{:.2},{:.2}",
            100.0 * fed_accepted as f64 / TRIALS_PER_LEVEL as f64,
            100.0 * vfed_accepted as f64 / TRIALS_PER_LEVEL as f64,
            100.0 * dag_fluid_accepted as f64 / TRIALS_PER_LEVEL as f64,
        );

        u_norm += u_norm_step;
    }

    ExitCode::SUCCESS
}

fn parse_positive_f64(s: &str) -> Option<f64> {
    let v: f64 = s.parse().ok()?;
    (v > 0.0).then_some(v)
}

/// Load every `dag_<N>.yaml` directly inside `pool_dir` (skipping RD-Gen's
/// own `combination_log.yaml`), overriding each one's `relative_deadline`
/// per [`override_deadline`], and pairing each resulting [`DagMetrics`] with
/// its DAG-Fluid [`Segment`] decomposition (structure-derived, so computed
/// once here and left untouched by the later per-`DagMetrics` overrides).
fn load_pool(pool_dir: &Path) -> Result<Vec<(String, DagMetrics, Vec<Segment>)>, String> {
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

    let mut rng = rand::rng();
    Ok(names
        .into_iter()
        .zip(all)
        .map(|(name, (config, segments))| {
            let config = override_deadline(config, &mut rng);
            let config = override_period(config, &mut rng);
            let config = assign_max_parallelism(config, &mut rng);
            (name, config, segments)
        })
        .collect())
}

/// Appends one JSON-Lines record for a single resampled trial: which 8
/// *distinct* pool entries (by filename, in draw order) were drawn, and
/// whether each admission policy accepted that exact set. See the module
/// doc's "Why resampling task sets" section for why a trial's own 8 draws
/// are distinct (unlike the paper's own with-replacement methodology), and
/// its "Recording individual trials" section for why this exists and how
/// to filter it.
fn write_trial_record(
    out: &mut fs::File,
    u_norm: f64,
    trial: usize,
    set: &[&(String, DagMetrics, Vec<Segment>)],
    federated_accepted: bool,
    vfed_accepted: bool,
    dag_fluid_accepted: bool,
) {
    let dags = set
        .iter()
        .map(|(name, _, _)| format!("\"{name}\""))
        .collect::<Vec<_>>()
        .join(",");
    let line = format!(
        "{{\"u_norm\":{u_norm:.4},\"trial\":{trial},\"dags\":[{dags}],\
         \"federated_accepted\":{federated_accepted},\"vfed_accepted\":{vfed_accepted},\
         \"dag_fluid_accepted\":{dag_fluid_accepted}}}\n"
    );
    // A single trial record failing to write isn't worth aborting a
    // long-running sweep over -- the aggregate ratios (this file's main
    // output) are unaffected either way.
    let _ = out.write_all(line.as_bytes());
}

/// Replace `config.relative_deadline` with the paper's own
/// `D ~ Uniform[L, L/ALPHA]`. No clamp to `period` here — see
/// [`override_period`], which runs right after and derives a *new* period
/// from this `D` instead, guaranteeing `D <= period` by construction rather
/// than by clamping `D` down (which produced artificially tiny `D -
/// critical_path` gaps, and thus wildly inflated `min_dedicated_cores()`
/// values, whenever RD-Gen's own generated period happened to sit close to
/// `critical_path` — confirmed empirically: a `C=1500, L=983` DAG clamped to
/// `period=1000` got `D=1000`, i.e. a `D - L` gap of just 17, giving
/// `m_i = ceil(517/17) = 31` dedicated cores for one single DAG).
fn override_deadline(config: DagMetrics, rng: &mut impl Rng) -> DagMetrics {
    let l = config.critical_path as f64;
    let d = l + rng.random_range(0.0..=1.0) * (l / ALPHA - l);
    let d = (d as u64).max(1);
    DagMetrics { relative_deadline: d, ..config }
}

/// The paper's own generation order for its constrained-deadline figures
/// (Fig. 9/10): period derived *from* the (already-set) deadline, not the
/// other way around — `T = D / beta`, `beta ~ Uniform[0.1, 1]` — so
/// `T >= D` holds unconditionally (`beta <= 1`), with no clamp and no risk
/// of the tiny-`D - L`-gap blowup `override_deadline`'s doc describes.
/// Discards whatever period RD-Gen itself generated (it was only ever a
/// UUniFast sizing input for `volume`, with no meaning after admission,
/// mirroring how `critical_path`/`relative_deadline` are already treated —
/// see `DagMetrics::critical_path`'s own doc). Must run after
/// `override_deadline`.
fn override_period(config: DagMetrics, rng: &mut impl Rng) -> DagMetrics {
    let beta = rng.random_range(0.1..=1.0);
    let period = ((config.relative_deadline as f64 / beta) as u64).max(config.relative_deadline);
    DagMetrics { period, ..config }
}

/// Set `config.max_parallelism` for V-Fed's OURS2 (`Π'`) refinement —
/// verbatim the TPDS23 paper's own evaluation methodology: `Uniform[ceil(m/2), m]`
/// where `m = config.min_dedicated_cores()` (must be run *after*
/// `override_deadline`, since `m` depends on the just-overridden deadline).
/// Left at the `DagMetrics::from_static` sentinel (`u16::MAX`, "unknown") if
/// `min_dedicated_cores()` is `None` — an infeasible DAG that will never
/// admit either way, not worth a synthetic parallelism value.
fn assign_max_parallelism(config: DagMetrics, rng: &mut impl Rng) -> DagMetrics {
    let Some(m) = config.min_dedicated_cores() else {
        return config;
    };
    let lo = m.div_ceil(2);
    let max_parallelism = if lo >= m { m } else { rng.random_range(lo..=m) };
    DagMetrics { max_parallelism, ..config }
}

/// Federated's all-or-nothing batch verdict for `configs` against
/// `num_cores`, reimplemented as a pure function of `classify_dag` (no
/// `resource` ledger side effects, mirroring `vfed::is_batch_feasible`):
/// every heavy DAG's `required_cores` must sum to at most `num_cores`, and
/// every light DAG's utilization must sum to at most whatever cores that
/// leaves — exactly the two checks `resource::allocate_cluster`/
/// `reserve_light_utilization` enforce for real, restated without the
/// shared ledger.
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
