//! Offline schedulability comparison: classical Federated scheduling
//! (`dag_sched::policy::federated`) vs V-Fed
//! (`dag_sched::policy::vfed`) — acceptance ratio across a utilization
//! sweep, the standard evaluation methodology both policies' own papers
//! use. Pure host-side computation: no QEMU, no real cores, no scheduler
//! mechanism involved, just admission-time math over synthetic task sets.
//!
//! Run with:
//! ```sh
//! cargo run --example vfed_vs_federated_acceptance --features std --release
//! ```
//! prints a `utilization,federated_ratio,vfed_ratio` CSV to stdout.
//!
//! # Method
//! - [`uunifast`] (Bini & Buttazzo) splits a target total utilization
//!   among [`NUM_TASKS`] DAGs reasonably uniformly over the simplex. Unlike
//!   classical single-core task generation, a per-task value can exceed 1
//!   here — that is exactly a heavy DAG (needs more than one core's worth),
//!   not an invalid input.
//! - [`generate_task_set`] turns each per-task utilization into a
//!   [`DagMetrics`]: `period` random, `volume = u * period`, `critical_path`
//!   a random fraction of `volume` (a "parallelism" factor), and
//!   `deadline` uniform in `(critical_path, period]` — always a
//!   *constrained* deadline (`L < D <= T`), which V-Fed's own theorems
//!   require (see `vfed.rs`'s module doc); an arbitrary-deadline task would
//!   fail vfed's `classify` outright regardless of resources, which would
//!   bias the comparison rather than measure it.
//! - [`federated_batch_feasible`] reproduces an all-or-nothing batch
//!   verdict for Federated — which only has a per-DAG `admit_dag`, no
//!   batch API — by admitting every DAG through the real, `resource`
//!   -ledger-backed `admit_dag` (heavy DAGs first: a heavy cluster's core
//!   claim shrinks what `resource::reserve_light_utilization`'s live
//!   `light_pool_size` counts as available to the light side, so admitting
//!   light DAGs first would let them race a bigger pool than actually
//!   remains once every heavy DAG has staked its claim) and unwinding every
//!   claim the trial made afterward, so the shared ledger returns to
//!   baseline for the next trial.
//! - `vfed::is_batch_feasible` is the pure, side-effect-free counterpart
//!   already built into the library for exactly this kind of experiment.

use awkernel_async_lib::dag_sched::{
    metrics::DagMetrics,
    policy::{
        federated::{self, TaskClass as FedTaskClass},
        vfed::{self, PackingStrategy},
    },
    provision::Provision,
    resource,
};
use rand::Rng;

/// Cores available to DAG scheduling. `resource`'s ledger additionally
/// reserves cpu 0 (primary) and one worker for the regular pool (see
/// `scheduler::pool::is_dag_pool_core`), so the real cpu count must be
/// `NUM_CORES + 2` for Federated's ledger-backed path to see this many.
const NUM_CORES: u16 = 16;
/// DAGs per generated task set. Fixed so only utilization varies across the
/// sweep, matching how these comparisons are usually plotted.
const NUM_TASKS: usize = 32;
/// Independent random task sets per utilization point.
const NUM_TRIALS: usize = 2000;
/// Utilization sweep resolution: `1/UTIL_STEPS .. 1.0` in equal steps, as a
/// fraction of `NUM_CORES`.
const UTIL_STEPS: usize = 40;

fn main() {
    // SAFETY: called once, before any other awkernel_lib::cpu use, from a
    // single-threaded `main` — exactly the "during kernel initialization"
    // contract `set_num_cpu` documents, just for a host process instead of
    // a real boot.
    unsafe {
        awkernel_lib::cpu::set_num_cpu(NUM_CORES as usize + 2);
    }

    let mut rng = rand::rng();

    println!("utilization,federated_ratio,vfed_ratio");

    for step in 1..=UTIL_STEPS {
        let utilization_fraction = step as f64 / UTIL_STEPS as f64;
        let target_util = NUM_CORES as f64 * utilization_fraction;

        let mut fed_accepted = 0usize;
        let mut vfed_accepted = 0usize;

        for _ in 0..NUM_TRIALS {
            let configs = generate_task_set(&mut rng, NUM_TASKS, target_util);

            if federated_batch_feasible(&configs) {
                fed_accepted += 1;
            }
            if vfed::is_batch_feasible(&configs, NUM_CORES, PackingStrategy::FirstFit) {
                vfed_accepted += 1;
            }
        }

        println!(
            "{utilization_fraction:.2},{:.4},{:.4}",
            fed_accepted as f64 / NUM_TRIALS as f64,
            vfed_accepted as f64 / NUM_TRIALS as f64,
        );
    }
}

/// UUniFast (Bini & Buttazzo, "Measuring the Performance of Schedulability
/// Tests"): split `target_util` into `n` values distributed reasonably
/// uniformly over the simplex summing to it, rather than naively dividing
/// evenly (which would never generate the skewed, some-tasks-heavy mixes a
/// real task set has).
fn uunifast(rng: &mut impl Rng, n: usize, target_util: f64) -> Vec<f64> {
    let mut utils = Vec::with_capacity(n);
    let mut sum_u = target_util;
    for i in 1..n {
        let next_sum_u = sum_u * rng.random::<f64>().powf(1.0 / (n - i) as f64);
        utils.push(sum_u - next_sum_u);
        sum_u = next_sum_u;
    }
    utils.push(sum_u);
    utils
}

/// Turn `n` UUniFast-split utilizations targeting `target_util` combined
/// into `DagMetrics`. See the module doc for why `critical_path` is always
/// clamped strictly below `period` and `deadline` always constrained.
fn generate_task_set(rng: &mut impl Rng, n: usize, target_util: f64) -> Vec<DagMetrics> {
    uunifast(rng, n, target_util)
        .into_iter()
        .map(|u| {
            let period = rng.random_range(1_000..=100_000u64);
            let volume = ((u * period as f64).round() as u64).max(1);
            // How much of the DAG's volume is unavoidably sequential.
            // parallelism=1 => fully sequential (critical_path == volume);
            // larger values simulate a DAG with more exploitable
            // parallelism per unit of work.
            let parallelism = rng.random_range(1..=8u64);
            let critical_path = (volume / parallelism).clamp(1, period - 1);
            let deadline = rng.random_range((critical_path + 1)..=period);
            DagMetrics::from_static(volume, critical_path, period, deadline)
        })
        .collect()
}

/// Federated's all-or-nothing batch verdict, reconstructed from its
/// per-DAG `admit_dag` (see the module doc for the heavy-first ordering and
/// why the whole trial's claims get unwound afterward either way).
fn federated_batch_feasible(configs: &[DagMetrics]) -> bool {
    let mut heavy = Vec::new();
    let mut light = Vec::new();
    for &config in configs {
        match federated::classify_dag(&config) {
            Ok(FedTaskClass::Heavy { .. }) => heavy.push(config),
            Ok(FedTaskClass::Light) => light.push(config),
            Err(_) => return false, // Infeasible regardless of resources.
        }
    }

    let mut committed: Vec<(Provision, DagMetrics)> = Vec::with_capacity(configs.len());
    let mut feasible = true;
    for config in heavy.into_iter().chain(light) {
        match federated::admit_dag(config) {
            Ok(assignment) => committed.push((assignment.provision, config)),
            Err(_) => {
                feasible = false;
                break;
            }
        }
    }

    for (provision, config) in committed {
        match provision {
            Provision::Dedicated { cores } => resource::release_cluster(cores),
            Provision::Shared { .. } => {
                let window = config.relative_deadline.min(config.period);
                resource::release_light_utilization(config.volume, window);
            }
        }
    }

    feasible
}
