//! The common release origin `t0` of DAG-Fluid's offline switch table
//! (`dag_sched::fluid_table`).
//!
//! The offline table is a plan of one hyperperiod in time relative to a
//! synchronous release of every DAG. To place it on the clock, every DAG's
//! job `k` must be released at `t0 + k * T`. The application sets `t0`
//! once, after admission and before it spawns any DAG node; each source
//! node then starts its interval at `t0` instead of at its own spawn time
//! (`dag::spawn_periodic_reactor`), and the planner adds `t0 + n * H` to the
//! table's times.
//!
//! Unset (the default) means "no common origin": sources start their
//! intervals when they are spawned, as before.

use awkernel_lib::time::Instant;
use core::{
    sync::atomic::{AtomicU64, Ordering},
    time::Duration,
};

/// `t0` in uptime nanoseconds; `u64::MAX` while unset.
static T0_NS: AtomicU64 = AtomicU64::new(u64::MAX);

/// Set the common release origin. Call once, before any DAG node is
/// spawned.
pub fn set_t0(t0: Instant) {
    let ns = u64::try_from(t0.uptime().as_nanos()).unwrap_or(u64::MAX - 1);
    T0_NS.store(ns, Ordering::Release);
}

/// The common release origin, if set.
pub fn t0() -> Option<Instant> {
    match T0_NS.load(Ordering::Acquire) {
        u64::MAX => None,
        ns => Some(Instant::zero() + Duration::from_nanos(ns)),
    }
}
