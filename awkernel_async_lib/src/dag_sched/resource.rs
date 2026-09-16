//! The DAG scheduling core/utilization ledger.
//!
//! Bundled behind one lock so a claim that shrinks the light pool (an
//! exclusive cluster) and a check against it (a light-pool reservation) can
//! never interleave inconsistently.
//!
//! Algorithm-agnostic: any admission policy that needs an exclusive cluster
//! of cores, or a share of the remaining pool tracked by utilization, draws
//! from this same ledger rather than keeping a private one of its own.
//! Today that is only [`crate::dag_sched::policy::federated`], but this
//! module has no Federated-specific knowledge — it never sees `u = C/T`,
//! only core counts and already-scaled utilization — so a future policy with
//! the same resource shape (e.g. V-Fed's classical-Federated-equivalent
//! cluster/utilization split) can reuse it unchanged.

use awkernel_lib::{
    cpu::{num_cpu, CpuSet},
    sync::mutex::{MCSNode, Mutex},
};

use alloc::vec::Vec;

use crate::scheduler::pool::is_dag_pool_core;

/// An admission policy's resource request could not be satisfied by the
/// shared ledger.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResourceError {
    /// Fewer than `required` DAG-pool cores are currently free.
    InsufficientCores { required: u16, available: u16 },
    /// Committing this much utilization would push the shared light pool's
    /// total over its capacity. Both fields are scaled by
    /// `UTILIZATION_SCALE`.
    LightPoolOversubscribed {
        additional_utilization_scaled: u64,
        available_capacity_scaled: u64,
    },
}

impl core::fmt::Display for ResourceError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            ResourceError::InsufficientCores {
                required,
                available,
            } => write!(
                f,
                "needs {required} dedicated core(s) but only {available} are free"
            ),
            ResourceError::LightPoolOversubscribed {
                additional_utilization_scaled,
                available_capacity_scaled,
            } => write!(
                f,
                "light pool oversubscribed: this DAG needs {}.{:02}% more utilization but only {}.{:02}% is free",
                additional_utilization_scaled / 10_000,
                (additional_utilization_scaled % 10_000) / 100,
                available_capacity_scaled / 10_000,
                (available_capacity_scaled % 10_000) / 100,
            ),
        }
    }
}

/// Utilization is tracked as an integer scaled by this factor (parts per
/// million) rather than a float, so the light-pool admission check below
/// stays exact and panic-free. `u = 1.0` (100%) is represented as
/// `1_000_000`. `pub(crate)` so other admission policies needing the same
/// scaled representation (e.g. `policy::vfed`'s partitioned-EDF density
/// bin-packing) share one scale factor instead of each picking their own.
pub(crate) const UTILIZATION_SCALE: u64 = 1_000_000;

/// `volume / window`, scaled. `window` is whatever interval the caller wants
/// `volume`'s demand spread over for the purpose of this bound — the period
/// for an implicit/lenient (`D >= T`) DAG, or the shorter of its relative
/// deadline and period otherwise (see [`crate::dag_sched::policy::federated`]).
/// This module stays agnostic to which: it only ever sees the resulting
/// scaled ratio. `pub(crate)` so other policies can reuse this exact scaling
/// (e.g. `policy::vfed`'s density-based partitioned-EDF bin-packing) instead
/// of duplicating it.
pub(crate) fn utilization_scaled(volume: u64, window: u64) -> u64 {
    // `volume`/`window` are WCET sums/time bounds in the caller's time unit,
    // orders of magnitude below `u64::MAX / UTILIZATION_SCALE` for any DAG
    // anyone would actually declare, so this cannot overflow in practice.
    // `window.max(1)` guards the degenerate `window == 0` input.
    volume.saturating_mul(UTILIZATION_SCALE) / window.max(1)
}

/// Shared state for the ledger, bundled behind one lock so a cluster claim
/// (which shrinks the light pool) and a light-pool reservation (which checks
/// against it) can never interleave inconsistently.
struct PoolLedger {
    /// Cores currently claimed by some DAG's exclusive cluster. Disjoint
    /// from (and unrelated to) `NUM_CLUSTERED_TASKS_ALIVE`: that counter
    /// tracks live *tasks* per CPU once spawned, while this tracks which
    /// worker CPUs this module has already promised to a cluster, so two
    /// exclusive clusters never get the same core.
    claimed_cores: CpuSet,
    /// Sum of every admitted light-pool DAG's utilization, scaled by
    /// [`UTILIZATION_SCALE`].
    light_utilization_scaled: u64,
}

static POOL: Mutex<PoolLedger> = Mutex::new(PoolLedger {
    claimed_cores: CpuSet::empty(),
    light_utilization_scaled: 0,
});

/// DAG-pool worker CPUs not currently claimed by any exclusive cluster —
/// i.e. the cores the light pool has to share. Excludes the regular-pool
/// core (see [`is_dag_pool_core`]), so it is never counted as light-pool
/// capacity nor handed out to a cluster.
fn light_pool_size(pool: &PoolLedger) -> usize {
    (1..num_cpu())
        .filter(|&cpu| is_dag_pool_core(cpu) && !pool.claimed_cores.contains(cpu))
        .count()
}

/// Number of DAG-pool worker CPUs not currently claimed by any exclusive
/// cluster, i.e. the most [`allocate_cluster`] could hand out right now.
/// Exposed for admission policies that need to search over candidate
/// cluster sizes (e.g. `dag_sched::policy::vfed`'s search over how many
/// cores to dedicate to heavy tasks as a whole) before committing to one.
pub fn free_core_count() -> u16 {
    let mut node = MCSNode::new();
    let pool = POOL.lock(&mut node);
    // `light_pool_size` is bounded by `num_cpu() <= NUM_MAX_CPU` (512), fits u16.
    light_pool_size(&pool) as u16
}

/// Claim `required_cores` DAG-pool worker CPUs (`1..num_cpu()`, excluding
/// CPU 0 and the regular-pool core; see [`is_dag_pool_core`]) not already
/// claimed by another cluster.
pub fn allocate_cluster(required_cores: u16) -> Result<CpuSet, ResourceError> {
    let required = required_cores as usize;

    let mut node = MCSNode::new();
    let mut pool = POOL.lock(&mut node);

    let free_workers: Vec<usize> = (1..num_cpu())
        .filter(|&cpu| is_dag_pool_core(cpu) && !pool.claimed_cores.contains(cpu))
        .collect();

    if free_workers.len() < required {
        return Err(ResourceError::InsufficientCores {
            required: required_cores,
            available: free_workers.len() as u16, // free_workers.len() < num_cpu() <= NUM_MAX_CPU (512), fits u16
        });
    }

    let mut cluster = CpuSet::empty();
    for cpu in free_workers.into_iter().take(required) {
        cluster.insert(cpu);
    }

    pool.claimed_cores = pool.claimed_cores.union(cluster);

    // Reserve these cores against `crate::task::is_cpu_reserved` immediately,
    // not just in this module's own ledger: this DAG's tasks are not spawned
    // until later (`finish_create_dags`), and an already-admitted DAG spawned
    // first can otherwise dispatch a global/regular task onto a core this
    // cluster has already claimed but not yet spawned into, in the window
    // between this admission and that spawn. Paired with `task::release_cpu`
    // in `release_cluster`.
    for cpu in cluster.iter() {
        crate::task::reserve_cpu(cpu);
    }

    Ok(cluster)
}

/// Release a cluster previously returned by [`allocate_cluster`], making its
/// cores available to the next admission that needs one.
pub fn release_cluster(cluster: CpuSet) {
    let mut node = MCSNode::new();
    let mut pool = POOL.lock(&mut node);
    for cpu in cluster.iter() {
        pool.claimed_cores.remove(cpu);
        crate::task::release_cpu(cpu);
    }
}

/// Commit `volume/window`'s utilization against the shared light pool,
/// returning the scaled utilization actually reserved (release it later with
/// the same `volume`/`window` via [`release_light_utilization`]).
///
/// This is the necessary condition every scheduling algorithm requires
/// (`Σ light utilization <= light pool core count`); it is not by itself a
/// sufficient schedulability proof for any particular global scheduler, but
/// admitting past it is certain to be unschedulable, so it is enforced as a
/// hard gate.
pub fn reserve_light_utilization(volume: u64, window: u64) -> Result<u64, ResourceError> {
    let additional = utilization_scaled(volume, window);

    let mut node = MCSNode::new();
    let mut pool = POOL.lock(&mut node);

    let capacity = (light_pool_size(&pool) as u64).saturating_mul(UTILIZATION_SCALE);
    let committed = pool.light_utilization_scaled;

    if committed.saturating_add(additional) > capacity {
        return Err(ResourceError::LightPoolOversubscribed {
            additional_utilization_scaled: additional,
            available_capacity_scaled: capacity.saturating_sub(committed),
        });
    }

    pool.light_utilization_scaled = committed + additional;
    Ok(additional)
}

/// Release utilization previously committed by [`reserve_light_utilization`].
pub fn release_light_utilization(volume: u64, window: u64) {
    let released = utilization_scaled(volume, window);
    let mut node = MCSNode::new();
    let mut pool = POOL.lock(&mut node);
    pool.light_utilization_scaled = pool.light_utilization_scaled.saturating_sub(released);
}

