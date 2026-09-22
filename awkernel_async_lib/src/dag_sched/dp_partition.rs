//! System-wide Deadline-Partition boundary tracking (real-machine
//! DAG-Fluid Phase 2): every DAG-Fluid task's own per-segment absolute
//! deadline (computed at admission time — see
//! `rd_gen_to_dags::dag_fluid`'s own module doc for Algorithm 1 lines
//! 16–21, the 2022 paper's Section 8 thread-list conversion) is
//! registered here once, before dispatch begins. This module then arms
//! `awkernel_lib::timer`'s `TimerRequestId::DpBoundary` for the next one
//! and logs each firing — computation/measurement only. No dispatch
//! action is taken when a boundary fires, continuing Phase 0/1's
//! placeholder-dispatch design (DAG-Fluid tasks still run under an
//! ordinary `ClusteredEDF` cluster, unaffected by this module).
//!
//! # Scope: a measurement probe, not WCET-proven dispatch
//! The papers' true DP-Fair dynamic re-partitioning (Section 8) is out of
//! scope entirely here, by explicit user direction: this module exists to
//! measure the *actual* timer-interrupt latency between a boundary's
//! theoretical (scheduled) time and when the callback actually observes
//! it — a concrete number for the paper's theory-vs-reality discussion,
//! not a scheduling guarantee. Accordingly, this module makes no attempt
//! at a proven WCET bound on the callback's own execution time (though it
//! stays allocation-free and panic-free at runtime regardless — see
//! [`on_dp_boundary`]'s own doc).
//!
//! # Placement: whichever CPU happens to register/arm it
//! An earlier design discussion favored running this on the primary CPU
//! (CPU#0), architecturally idle since neither the DAG-pool nor
//! regular-pool split ever dispatches a task there (see
//! `crate::scheduler::pool::is_dag_pool_core`/`is_regular_pool_core`,
//! both `cpu_id != 0`). This module does not *enforce* that placement —
//! `awkernel_lib::timer::request_at` arms whichever CPU calls it (that
//! module's own per-CPU multiplexer design) — it only logs which CPU
//! actually did the arming/firing, so the assumption is empirically
//! checkable from the trace rather than silently assumed.
//!
//! # Absolute-time baseline
//! Segment deadlines are computed relative to a task's own release time
//! (the papers' `r_i,j`, Section 4.2), but this experimental setup admits
//! every DAG-Fluid task once at boot, before dispatch
//! (`finish_create_dags`) begins — there is no live "job release" event
//! yet to anchor to. [`register_segment`]'s caller-supplied `release`
//! time stands in for it (by convention, `Time::now()` at admission —
//! see `rd_gen_to_dags::build_dag`'s `dagfluid` arm); this is a
//! documented approximation, not the papers' own semantics.

use alloc::{boxed::Box, vec::Vec};
use awkernel_lib::{
    sync::mutex::{MCSNode, Mutex},
    time::Time,
    timer::{self, TimerRequestId},
};
use core::time::Duration;

/// One outstanding segment deadline, not yet fired.
struct BoundaryEntry {
    dag_id: u32,
    segment_index: usize,
    /// The absolute time this boundary is scheduled for.
    scheduled: Time,
}

static PENDING: Mutex<Vec<BoundaryEntry>> = Mutex::new(Vec::new());

/// Convert a segment's relative-deadline/release-offset value (`f64`,
/// milliseconds — the unit `rd_gen_to_dags`'s own `time_unit` default
/// feature uses) into a `Duration`. Truncates towards zero rather than
/// rounding: `f64::round`/`ceil` need `std`/`libm`, unavailable in this
/// crate's `no_std` build — the same constraint
/// `rd_gen_to_dags::dag_fluid` documents for its own float-to-integer
/// conversions (`ceil_capacity_to_cores`). Negative input (should not
/// occur for a real offset/deadline) clamps to zero rather than
/// wrapping.
fn millis_f64_to_duration(ms: f64) -> Duration {
    Duration::from_millis(ms.max(0.0) as u64)
}

/// Register one segment's own absolute deadline (`release +
/// offset_ms + duration_ms`, both in milliseconds — `offset_ms` from
/// `rd_gen_to_dags::dag_fluid::segment_release_offsets`, `duration_ms`
/// from that same segment's own `SegmentSchedule::relative_deadline`)
/// for `dag_id`. Called once per segment at DAG-Fluid admission time
/// (boot, non-RT), before [`arm_next`]/dispatch begins. `Vec::push`
/// (possibly reallocating) is fine here: this only ever runs during boot
/// admission, never from [`on_dp_boundary`].
pub fn register_segment(dag_id: u32, segment_index: usize, release: Time, offset_ms: f64, duration_ms: f64) {
    let scheduled = release + millis_f64_to_duration(offset_ms + duration_ms);
    let mut node = MCSNode::new();
    let mut pending = PENDING.lock(&mut node);
    pending.push(BoundaryEntry {
        dag_id,
        segment_index,
        scheduled,
    });
}

/// Arm the timer for the earliest still-pending boundary, if any. Safe to
/// call repeatedly (e.g. once after every DAG's admission, or once at the
/// end of the whole batch) — a no-op if nothing is pending, and
/// idempotent otherwise (re-arming for the same soonest deadline is
/// harmless, `awkernel_lib::timer::request_at`'s own contract).
pub fn arm_next() {
    let next = {
        let mut node = MCSNode::new();
        let pending = PENDING.lock(&mut node);
        pending.iter().map(|e| e.scheduled).min()
    };
    if let Some(deadline) = next {
        log::info!(
            "dp_partition: arming DpBoundary on CPU#{} for t={:?}",
            awkernel_lib::cpu::cpu_id(),
            deadline
        );
        timer::request_at(TimerRequestId::DpBoundary, deadline);
    }
}

/// The `TimerRequestId::DpBoundary` callback (see [`install`]). Pops
/// whichever pending boundary has actually reached its scheduled time,
/// logs the measured latency (`Time::now()` at fire time minus its own
/// `scheduled` time) alongside which CPU observed it, and re-arms for
/// whatever remains. No dispatch action — see this module's own doc.
///
/// WCET note (see module doc's "not WCET-proven" scope): bounded by the
/// number of DAG-Fluid segments still pending, which only ever shrinks
/// after boot admission ends (never regrows: [`register_segment`] is
/// never called after [`arm_next`]/dispatch begins, by this module's own
/// documented boot-time-only contract); no allocation
/// (`Vec::swap_remove` never grows the backing storage); no panic
/// (`Time: Ord` — a plain integer comparison, no `partial_cmp().unwrap()`
/// NaN hazard; `.get`-free index handled via `enumerate().min_by_key`
/// rather than direct indexing).
fn on_dp_boundary() {
    let now = Time::now();
    let fired = {
        let mut node = MCSNode::new();
        let mut pending = PENDING.lock(&mut node);
        let due = pending
            .iter()
            .enumerate()
            .filter(|(_, e)| e.scheduled <= now)
            .min_by_key(|(_, e)| e.scheduled)
            .map(|(i, _)| i);
        due.map(|i| pending.swap_remove(i))
    };

    if let Some(entry) = fired {
        let latency = now.saturating_duration_since(entry.scheduled);
        log::info!(
            "dp_partition: DAG#{} segment[{}] DP boundary fired on CPU#{}, scheduled={:?} actual={now:?} latency={latency:?}",
            entry.dag_id,
            entry.segment_index,
            awkernel_lib::cpu::cpu_id(),
            entry.scheduled,
        );
    }

    arm_next();
}

/// Register [`on_dp_boundary`] as `TimerRequestId::DpBoundary`'s
/// callback. Call once at boot, before the first
/// [`register_segment`]/[`arm_next`].
pub fn install() {
    timer::register_timer_callback(TimerRequestId::DpBoundary, Box::new(on_dp_boundary));
}
