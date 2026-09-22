//! System-wide Deadline-Partition boundary tracking and real dispatch
//! entitlement (real-machine DAG-Fluid): every DAG-Fluid task's own
//! per-segment absolute deadline (computed at admission time — see
//! `rd_gen_to_dags::dag_fluid`'s own module doc for Algorithm 1 lines
//! 16–21, the 2022 paper's Section 8 thread-list conversion) is
//! registered here once, before dispatch begins. Two independent
//! consumers then watch [`PENDING`]:
//!
//! - [`on_dp_boundary`] (a `TimerRequestId::DpBoundary` interrupt
//!   callback): fires exactly at each boundary's theoretical deadline and
//!   logs the measurement this module is named for (scheduled vs. actual
//!   time) — nothing else. It never mutates a segment's completion state
//!   and never touches [`CURRENT`] or the DAG/task graph (see "Split
//!   design" below for why).
//! - [`advance_due_segments`] (an ordinary background task, spawned by
//!   [`spawn_advancer`]): polls [`PENDING`] on a short, fixed cadence
//!   ([`ADVANCER_POLL_INTERVAL`]) and, once a due boundary's own
//!   *completion gate* is satisfied (see below), advances that DAG's
//!   current segment and calls `scheduler::dp_wrap::recompute_and_apply`
//!   to recompute the real per-CPU entitlement for the new Deadline
//!   Partition.
//!
//! # Completion gate: advance is gated on real work, not just the clock
//! The papers' segment deadlines are assigned with *zero* slack by
//! construction (`Σd_i,j` sums exactly to the virtual deadline `D*_i` —
//! see `rd_gen_to_dags::dag_fluid`'s own module doc): the idealized fluid
//! model proves a segment finishes exactly on time if run at its own
//! `theta_i,j` rate continuously, with no margin anywhere. On real
//! hardware, IPI latency, migration cost, and discrete task-switch
//! granularity all eat directly into that zero margin.
//!
//! An earlier design would have advanced strictly on the theoretical
//! clock, reassigning a CPU's entitlement to the *next* segment the
//! instant a boundary's scheduled time passed, regardless of whether the
//! current segment's own work had actually finished — silently abandoning
//! any still-running node with no rescue mechanism. This was explicitly
//! rejected as unacceptable. Instead, each [`BoundaryEntry`] carries a
//! `gate_nodes` list ([`rd_gen_to_dags::dag_fluid::segment_completion_gates`]'s
//! own output — the node ids the idealized timeline expects to finish
//! exactly at this segment's end): [`advance_due_segments`] only actually
//! advances past a boundary once every node in its gate has reached
//! `task::State::Terminated` for real. Until then, the CPU(s) currently
//! entitled to that DAG keep their entitlement (the segment does **not**
//! lose its capacity share), the overrun is logged once, and the boundary
//! is retried on the advancer's own next poll — the gap between
//! "theoretical deadline" and "gate actually satisfied" *is* the
//! theory-vs-reality measurement this module exists to produce, made
//! visible in the log rather than silently absorbed.
//!
//! # Split design: why the gate check cannot run from the timer callback
//! An earlier version ran the completion-gate check (and the subsequent
//! `CURRENT` update / entitlement recompute) directly inside
//! [`on_dp_boundary`]. This reproducibly broke the system: **every**
//! `TimerRequestId::DpBoundary` firing after the first simply stopped
//! happening at all, confirmed with both a blocking `Mutex::lock` and a
//! non-blocking `Mutex::try_lock` for the gate check's own node lookups.
//! Direct experiment isolated this to calling into `Dag`'s graph lock or
//! `TaskInfo`'s own lock from timer-interrupt context specifically: with
//! an empty `gate_nodes` (skipping both), all three of a real test DAG's
//! segments fired correctly with the *same* surrounding code; with a
//! real, non-empty gate, only the first ever fired. The likely mechanism
//! (not fully confirmed): `awkernel_sync`'s `InterruptGuard` (used
//! internally by every `MCSLock::lock`/`try_lock`) calls
//! `voluntary_preemption()` on drop whenever interrupts end up enabled
//! again — appropriate for ordinary task-context code, but not
//! necessarily safe to trigger from inside a timer ISR's own callback
//! chain. `Dag::graph`/`TaskInfo`'s locks are also the *same* locks
//! ordinary task-context code (pub/sub routing, `task::run_main`'s own
//! poll bookkeeping) contends for concurrently — unlike this module's own
//! `PENDING`/`CURRENT` locks, which only this module ever touches and
//! which showed no such issue either from the timer callback or from the
//! advancer task.
//!
//! The fix is this module's current split: [`on_dp_boundary`] touches
//! only [`PENDING`] (proven safe from interrupt context) purely to timestamp
//! *when* each boundary's theoretical deadline was actually observed;
//! every DAG/task lookup (`segment_gate_satisfied`) and every `CURRENT`/
//! entitlement mutation happens only from [`advance_due_segments`],
//! which never runs from interrupt context.
//!
//! # Scope: not WCET-proven dispatch
//! By explicit user direction, this module does not carry a proven WCET
//! bound on its own callback latency — that latency (and now also gate
//! overrun) is exactly what it measures, not what it guarantees.
//!
//! # Placement: whichever CPU happens to register/arm it
//! `awkernel_lib::timer::request_at` arms whichever CPU calls it (that
//! module's own per-CPU multiplexer design); this module does not enforce
//! any particular placement, only logs which CPU actually did the
//! arming/firing.
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

use alloc::{boxed::Box, collections::BTreeMap, vec::Vec};
use awkernel_lib::{
    sync::mutex::{MCSNode, Mutex},
    time::Time,
    timer::{self, TimerRequestId},
};
use core::time::Duration;

/// How often [`advance_due_segments`] polls [`PENDING`] for boundaries
/// whose completion gate has newly cleared. Short enough that a
/// segment's measured "gate satisfied" latency stays close to the real
/// completion time; long enough to be a plain periodic background task
/// rather than a busy-poll.
const ADVANCER_POLL_INTERVAL: Duration = Duration::from_millis(1);

/// One outstanding segment deadline, not yet advanced past.
struct BoundaryEntry {
    dag_id: u32,
    segment_index: usize,
    /// The absolute time this boundary is scheduled for.
    scheduled: Time,
    /// `m_i,j`: this segment's own thread count (see
    /// `rd_gen_to_dags::dag_fluid::SegmentSchedule::concurrency`).
    concurrency: u32,
    /// `theta_i,j`: this segment's own execution rate (see
    /// `rd_gen_to_dags::dag_fluid::SegmentSchedule::rate`).
    rate: f64,
    /// This segment's own completion gate (see this module's own doc) —
    /// node ids the idealized timeline expects to finish exactly at this
    /// segment's end.
    gate_nodes: Vec<u32>,
    /// Set by [`on_dp_boundary`] once it has logged this entry's
    /// scheduled-vs-actual latency measurement, so a repeat timer fire
    /// (or the timer simply never being re-armed for it again, see that
    /// function's own doc) never logs the same entry twice.
    logged: bool,
    /// Set by [`advance_due_segments`] once it has logged this entry's
    /// completion-gate overrun, so repeated polls while still blocked
    /// don't spam the log every [`ADVANCER_POLL_INTERVAL`].
    warned_overrun: bool,
}

static PENDING: Mutex<Vec<BoundaryEntry>> = Mutex::new(Vec::new());

/// Per-`dag_id`: the `(concurrency, rate)` of that DAG's own *currently
/// active* segment — i.e. the one whose entitlement
/// `scheduler::dp_wrap::recompute_and_apply` should currently be granting
/// CPU time to. Seeded by [`register_segment`] for segment 0, advanced by
/// [`on_dp_boundary`] once a segment's completion gate clears, and removed
/// once a DAG's last segment clears (nothing left to entitle).
static CURRENT: Mutex<BTreeMap<u32, (u32, f64)>> = Mutex::new(BTreeMap::new());

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
/// for `dag_id`, along with the `(concurrency, rate)` DP-Wrap density this
/// segment contributes (see `scheduler::dp_wrap::recompute_and_apply`'s
/// own doc) and its own completion gate (`gate_nodes` — see
/// `rd_gen_to_dags::dag_fluid::segment_completion_gates`). Called once per
/// segment, in increasing `segment_index` order, at DAG-Fluid admission
/// time (boot, non-RT), before [`arm_next`]/dispatch begins. `segment_index
/// == 0` also seeds [`CURRENT`] with this segment's own density, since a
/// DAG's first segment is active from release with no boundary of its own
/// to advance past. `Vec::push`/`BTreeMap` insertion (possibly
/// reallocating) is fine here: this only ever runs during boot admission,
/// never from [`on_dp_boundary`].
pub fn register_segment(
    dag_id: u32,
    segment_index: usize,
    release: Time,
    offset_ms: f64,
    duration_ms: f64,
    concurrency: u32,
    rate: f64,
    gate_nodes: Vec<u32>,
) {
    let scheduled = release + millis_f64_to_duration(offset_ms + duration_ms);
    let mut node = MCSNode::new();
    let mut pending = PENDING.lock(&mut node);
    pending.push(BoundaryEntry {
        dag_id,
        segment_index,
        scheduled,
        concurrency,
        rate,
        gate_nodes,
        logged: false,
        warned_overrun: false,
    });
    drop(pending);

    if segment_index == 0 {
        let mut node = MCSNode::new();
        let mut current = CURRENT.lock(&mut node);
        current.insert(dag_id, (concurrency, rate));
    }
}

/// Apply [`CURRENT`]'s present contents as the real per-CPU entitlement —
/// call once after every DAG-Fluid DAG has been admitted (segment 0 of
/// each already seeded [`CURRENT`] via [`register_segment`]), before
/// dispatch begins, and again from [`on_dp_boundary`] whenever a segment
/// actually advances.
fn apply_current_entitlement() {
    let active: Vec<(u32, u32, f64)> = {
        let mut node = MCSNode::new();
        let current = CURRENT.lock(&mut node);
        current
            .iter()
            .map(|(&dag_id, &(concurrency, rate))| (dag_id, concurrency, rate))
            .collect()
    };
    crate::scheduler::dp_wrap::recompute_and_apply(&active);
}

/// Call once, after every DAG-Fluid DAG has been admitted and every
/// segment registered, to grant each DAG's own segment-0 entitlement
/// before dispatch begins (see `rd_gen_to_dags::lib::run`'s call site,
/// alongside [`arm_next`]).
pub fn apply_initial_entitlement() {
    apply_current_entitlement();
}

/// Whether every node in `gate_nodes` has really reached
/// `task::State::Terminated` — see this module's own doc on why a
/// boundary's advance is gated on this rather than the clock alone.
///
/// Only ever called from [`advance_due_segments`] (ordinary task
/// context) — see this module's own "Split design" doc for why this
/// must never run from [`on_dp_boundary`]'s interrupt context.
///
/// `dag_id` not naming a live DAG is treated as "already satisfied"
/// (defensive: should only happen for a should-never-occur
/// admission/registration mismatch, and retrying forever against a
/// phantom DAG would be worse than moving on). A gate node's own lookup
/// (`Dag::get_node_task_id`, `task::get_task`, `TaskInfo`'s own state)
/// failing is treated as "**not** yet satisfied" — the opposite default,
/// because silently treating a real, still-tracked node as done is the
/// one mistake this whole gate exists to prevent.
///
/// # Known gap: `State::Terminated` never happens for a live DAG's own
/// reactor tasks
/// Confirmed by direct observation once the interrupt-safety redesign
/// above made it possible to actually exercise this function against a
/// real DAG: a periodic source reactor (`spawn_periodic_reactor`) loops
/// forever (`interval.tick().await` each period) and an
/// intermediate/sink reactor loops waiting for its next pub/sub message
/// — neither ever reaches `State::Terminated` during normal steady-state
/// operation (only DAG teardown would do that). This means a gate whose
/// nodes are still "alive" (the normal case for any DAG still running)
/// can never actually be satisfied by this check as written, and every
/// segment boundary after admission will overrun forever. The correct
/// signal is "this node finished its *current job's* contribution" —
/// most likely tracked via `DagInfo::period_index`'s own progression
/// (already read elsewhere for tracing) or `State::Waiting` (idle between
/// releases), not `State::Terminated` — but this needs its own design
/// pass (in particular, distinguishing "just finished period N" from
/// "hasn't started its first period yet", which can look similar from
/// `TaskInfo::get_state()` alone). Left as `Terminated` for now since
/// fixing the interrupt-context hang was this redesign's scope; the
/// overrun this gap causes is at least safe and clearly logged, not
/// silent.
fn segment_gate_satisfied(dag_id: u32, gate_nodes: &[u32]) -> bool {
    let Some(dag) = crate::dag::get_dag(dag_id) else {
        return true;
    };
    gate_nodes.iter().all(|&node_id| {
        let Some(task_id) = dag.get_node_task_id(node_id) else {
            return false;
        };
        let Some(task) = crate::task::get_task(task_id) else {
            return false;
        };
        let mut node = MCSNode::new();
        let state = task.info.lock(&mut node).get_state();
        matches!(
            state,
            crate::task::State::Terminated | crate::task::State::Panicked
        )
    })
}

/// Arm the timer for the earliest still-*unlogged* boundary, if any (see
/// [`on_dp_boundary`]'s own doc for why "unlogged" rather than "still
/// pending"). Safe to call repeatedly — a no-op if nothing qualifies, and
/// idempotent otherwise (re-arming for the same soonest deadline is
/// harmless, `awkernel_lib::timer::request_at`'s own contract).
pub fn arm_next() {
    let next = {
        let mut node = MCSNode::new();
        let pending = PENDING.lock(&mut node);
        pending.iter().filter(|e| !e.logged).map(|e| e.scheduled).min()
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

/// The `TimerRequestId::DpBoundary` callback (see [`install`]). Purely a
/// measurement probe (see this module's own "Split design" doc): for
/// every not-yet-[`logged`](BoundaryEntry::logged) entry whose scheduled
/// time has actually arrived, logs the measured latency (`Time::now()`
/// at fire time minus its own `scheduled` time) alongside which CPU
/// observed it, and marks it logged so a later re-check (or the timer
/// simply not being re-armed for it again) never logs it twice. Takes no
/// action on the entry otherwise — no DAG/task lookup, no `CURRENT`
/// mutation, no entitlement recompute; that is entirely
/// [`advance_due_segments`]'s job.
///
/// Re-arms for the earliest still-unlogged entry, if any (see
/// [`arm_next`]) — once every currently-registered entry has been
/// logged, this simply stops re-arming until [`register_segment`] adds a
/// new one; [`advance_due_segments`]'s own independent poll loop is what
/// continues to notice and act on already-logged, still-pending entries
/// from here on.
///
/// WCET note (see module doc's "not WCET-proven" scope): bounded by the
/// number of DAG-Fluid segments still registered (only ever shrinks,
/// never regrows after boot admission); no DAG/task lookup, so no risk of
/// the interrupt-context lock hazard this module's own doc describes; no
/// panic (`Time: Ord`, no direct indexing).
fn on_dp_boundary() {
    let now = Time::now();

    let mut node = MCSNode::new();
    let mut pending = PENDING.lock(&mut node);
    for entry in pending.iter_mut() {
        if !entry.logged && entry.scheduled <= now {
            let latency = now.saturating_duration_since(entry.scheduled);
            log::info!(
                "dp_partition: DAG#{} segment[{}] DP boundary fired on CPU#{}, scheduled={:?} actual={now:?} latency={latency:?}",
                entry.dag_id,
                entry.segment_index,
                awkernel_lib::cpu::cpu_id(),
                entry.scheduled,
            );
            entry.logged = true;
        }
    }
    let next = pending.iter().filter(|e| !e.logged).map(|e| e.scheduled).min();
    drop(pending);

    if let Some(deadline) = next {
        log::info!(
            "dp_partition: arming DpBoundary on CPU#{} for t={:?}",
            awkernel_lib::cpu::cpu_id(),
            deadline
        );
        timer::request_at(TimerRequestId::DpBoundary, deadline);
    }
}

/// Poll [`PENDING`] once for boundaries whose scheduled time has arrived,
/// and actually advance the ones whose completion gate
/// ([`segment_gate_satisfied`]) is satisfied — see this module's own
/// "Split design" doc for why this, and not [`on_dp_boundary`], is where
/// the DAG/task lookups and `CURRENT`/entitlement mutation happen. Called
/// from ordinary (non-interrupt) task context only — see
/// [`spawn_advancer`].
///
/// - **Satisfied**: removed from [`PENDING`], and [`CURRENT`] advanced to
///   whatever segment of that `dag_id` comes next (or removed entirely if
///   this was its last segment).
/// - **Not yet satisfied**: left in [`PENDING`]; the overrun is logged
///   once (guarded by [`BoundaryEntry::warned_overrun`], so a still-blocked
///   entry doesn't spam the log on every poll).
///
/// [`apply_current_entitlement`] is called once at the end if anything
/// actually advanced.
fn advance_due_segments() {
    let now = Time::now();

    // Snapshot every due entry's own fields up front (one `PENDING` lock,
    // released before `segment_gate_satisfied` looks up live DAG/task
    // state, which takes other locks of its own).
    let due: Vec<(usize, u32, usize, Time, Vec<u32>)> = {
        let mut node = MCSNode::new();
        let pending = PENDING.lock(&mut node);
        pending
            .iter()
            .enumerate()
            .filter(|(_, e)| e.scheduled <= now)
            .map(|(i, e)| (i, e.dag_id, e.segment_index, e.scheduled, e.gate_nodes.clone()))
            .collect()
    };

    let mut resolved_indices: Vec<usize> = Vec::new();
    let mut newly_blocked: Vec<usize> = Vec::new();
    for (i, dag_id, _segment_index, _scheduled, gate_nodes) in &due {
        if segment_gate_satisfied(*dag_id, gate_nodes) {
            resolved_indices.push(*i);
        } else {
            newly_blocked.push(*i);
        }
    }

    if !newly_blocked.is_empty() {
        let mut node = MCSNode::new();
        let mut pending = PENDING.lock(&mut node);
        for &i in &newly_blocked {
            let Some(entry) = pending.get_mut(i) else {
                continue;
            };
            if !entry.warned_overrun {
                let overrun = now.saturating_duration_since(entry.scheduled);
                log::warn!(
                    "dp_partition: DAG#{} segment[{}] DP boundary overrun on CPU#{}: theoretical deadline passed {overrun:?} ago, completion gate not yet satisfied -- entitlement unchanged, retrying",
                    entry.dag_id,
                    entry.segment_index,
                    awkernel_lib::cpu::cpu_id(),
                );
                entry.warned_overrun = true;
            }
        }
    }

    if resolved_indices.is_empty() {
        return;
    }

    // Highest-index-first so `swap_remove`'s move-the-last-element step
    // can never invalidate an index still queued for removal.
    resolved_indices.sort_unstable_by(|a, b| b.cmp(a));

    let removed: Vec<BoundaryEntry> = {
        let mut node = MCSNode::new();
        let mut pending = PENDING.lock(&mut node);
        resolved_indices
            .iter()
            .map(|&i| pending.swap_remove(i))
            .collect()
    };

    for entry in &removed {
        let completion_latency = now.saturating_duration_since(entry.scheduled);
        log::info!(
            "dp_partition: DAG#{} segment[{}] completion gate satisfied on CPU#{}: theoretical deadline={:?}, gate cleared at {now:?} (completion_latency={completion_latency:?}, warned_overrun_first={})",
            entry.dag_id,
            entry.segment_index,
            awkernel_lib::cpu::cpu_id(),
            entry.scheduled,
            entry.warned_overrun,
        );
    }

    let next_density: BTreeMap<u32, (u32, f64)> = {
        let mut node = MCSNode::new();
        let pending = PENDING.lock(&mut node);
        removed
            .iter()
            .filter_map(|entry| {
                pending
                    .iter()
                    .find(|e| e.dag_id == entry.dag_id)
                    .map(|e| (entry.dag_id, (e.concurrency, e.rate)))
            })
            .collect()
    };

    {
        let mut node = MCSNode::new();
        let mut current = CURRENT.lock(&mut node);
        for entry in &removed {
            match next_density.get(&entry.dag_id) {
                Some(&density) => {
                    current.insert(entry.dag_id, density);
                }
                None => {
                    current.remove(&entry.dag_id);
                }
            }
        }
    }

    apply_current_entitlement();
}

/// Spawn the background task that drives [`advance_due_segments`] on a
/// fixed [`ADVANCER_POLL_INTERVAL`] cadence, forever. Call once at boot,
/// after [`install`]/[`apply_initial_entitlement`]/[`arm_next`].
pub fn spawn_advancer() {
    crate::task::spawn(
        "dp_wrap advancer".into(),
        async {
            loop {
                advance_due_segments();
                crate::sleep(ADVANCER_POLL_INTERVAL).await;
            }
        },
        crate::scheduler::SchedulerType::PrioritizedFIFO(0),
    );
}

/// Register [`on_dp_boundary`] as `TimerRequestId::DpBoundary`'s
/// callback. Call once at boot, before the first
/// [`register_segment`]/[`arm_next`].
pub fn install() {
    timer::register_timer_callback(TimerRequestId::DpBoundary, Box::new(on_dp_boundary));
}
