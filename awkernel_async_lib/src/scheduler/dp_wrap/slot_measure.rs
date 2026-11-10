//! One record per per-node slot (the `dagfluid_node` feature, 方式B N7):
//! which node and job the slot belonged to, the node's state when the CPU
//! first looked at the slot, and when the node became ready and started
//! running there. Recorded only while the trace (`task::trace`) is
//! recording, and dumped with it as `TRACE_SLOT` lines.
//!
//! Each CPU keeps one open record:
//! - [`begin`] (the `DpSwitch` handler, own CPU, interrupt context) closes
//!   the previous slot at the new slot's start and opens the new one. Only
//!   atomics are touched.
//! - [`observe`] (the CPU's own decision point: `DpWrapScheduler::get_next`
//!   or the IPI 255 path) stores the state the first time only.
//! - [`ready`] (`DpWrapScheduler::wake_task`, any CPU) stores when the
//!   slot's node was woken. Called from another CPU, it may race with the
//!   slot closing; the record then misses (or, rarely, attributes to the
//!   next slot) one wake time, which the analysis tolerates.
//! - [`task_started`] (the trace's slice start, own CPU) stores when the
//!   slot's node started running.
//!
//! The overrun time of a node is not recorded here: the analysis computes
//! it from the node's execution slices (`TRACE_EV`) inside no-DAG slots.

pub(crate) use imp::*;

// The `state` of a slot record (`task::trace::SlotEvent`).
/// The CPU never looked at the slot (no decision point during it).
pub const SLOT_UNOBSERVED: u8 = 0;
/// The node was ready for the slot's job and was dispatched.
pub const SLOT_READY: u8 = 1;
/// The node was already running on this CPU.
pub const SLOT_RUNNING_HERE: u8 = 2;
/// The node was on the slot's job but not woken yet (inputs not there).
pub const SLOT_NOT_WOKEN: u8 = 3;
/// The node had already finished the slot's job.
pub const SLOT_DONE: u8 = 4;
/// The node was still on an earlier job.
pub const SLOT_BEHIND: u8 = 5;
/// The node was running on another CPU.
pub const SLOT_RUNNING_ELSEWHERE: u8 = 6;
/// A no-DAG slot (overrun nodes only).
pub const SLOT_NO_DAG: u8 = 7;

#[cfg(feature = "perf")]
mod imp {
    use super::{SLOT_NO_DAG, SLOT_UNOBSERVED};
    use crate::{
        scheduler::dp_wrap::switch_table::NodeSlot,
        task::trace::{self, SlotEvent},
    };
    use array_macro::array;
    use awkernel_lib::{cpu::NUM_MAX_CPU, interrupt::InterruptGuard, time::Instant};
    use core::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicU8, Ordering};

    const NONE: u32 = u32::MAX;

    struct Open {
        open: AtomicBool,
        dag: AtomicU32,
        node: AtomicU32,
        job: AtomicU64,
        start: AtomicU64,
        state: AtomicU8,
        observed: AtomicU64,
        ready: AtomicU64,
        started: AtomicU64,
        /// The slot node's task, once known ([`observe`]); 0: unknown.
        task: AtomicU32,
        skipped: AtomicU32,
    }

    impl Open {
        const fn new() -> Self {
            Self {
                open: AtomicBool::new(false),
                dag: AtomicU32::new(NONE),
                node: AtomicU32::new(NONE),
                job: AtomicU64::new(0),
                start: AtomicU64::new(0),
                state: AtomicU8::new(SLOT_UNOBSERVED),
                observed: AtomicU64::new(0),
                ready: AtomicU64::new(0),
                started: AtomicU64::new(0),
                task: AtomicU32::new(0),
                skipped: AtomicU32::new(0),
            }
        }

        fn is(&self, dag_id: u32, node_id: u32) -> bool {
            self.open.load(Ordering::Acquire)
                && self.dag.load(Ordering::Relaxed) == dag_id
                && self.node.load(Ordering::Relaxed) == node_id
        }
    }

    static OPEN: [Open; NUM_MAX_CPU] = array![_ => Open::new(); NUM_MAX_CPU];

    fn ns(t: Instant) -> u64 {
        u64::try_from(t.uptime().as_nanos()).unwrap_or(u64::MAX)
    }

    /// `cpu_id` switches at `at` to `dag_id`'s slot `node` (`None` for both:
    /// a no-DAG slot). Closes the open record at `at` unless the owner is
    /// the same. Must run on `cpu_id`.
    pub(crate) fn begin(
        cpu_id: usize,
        at: Instant,
        dag_id: Option<u32>,
        node: Option<NodeSlot>,
        skipped: u32,
    ) {
        let Some(o) = OPEN.get(cpu_id) else {
            return;
        };
        let dag = dag_id.unwrap_or(NONE);
        let (node_id, job) = node.map_or((NONE, 0), |slot| (slot.node_id, slot.job));
        let at = ns(at);
        if o.open.load(Ordering::Relaxed) {
            if o.dag.load(Ordering::Relaxed) == dag
                && o.node.load(Ordering::Relaxed) == node_id
                && o.job.load(Ordering::Relaxed) == job
            {
                return;
            }
            o.open.store(false, Ordering::Release);
            trace::record_slot(SlotEvent {
                dag_id: o.dag.load(Ordering::Relaxed),
                node_id: o.node.load(Ordering::Relaxed),
                job: o.job.load(Ordering::Relaxed),
                start_ns: o.start.load(Ordering::Relaxed),
                end_ns: at,
                state: o.state.load(Ordering::Relaxed),
                observed_ns: o.observed.load(Ordering::Relaxed),
                ready_ns: o.ready.load(Ordering::Relaxed),
                started_ns: o.started.load(Ordering::Relaxed),
                skipped: o.skipped.load(Ordering::Relaxed),
            });
        }
        if !trace::is_enabled() {
            return;
        }
        o.dag.store(dag, Ordering::Relaxed);
        o.node.store(node_id, Ordering::Relaxed);
        o.job.store(job, Ordering::Relaxed);
        o.start.store(at, Ordering::Relaxed);
        let state = if dag == NONE {
            SLOT_NO_DAG
        } else {
            SLOT_UNOBSERVED
        };
        o.state.store(state, Ordering::Relaxed);
        o.observed.store(0, Ordering::Relaxed);
        o.ready.store(0, Ordering::Relaxed);
        o.started.store(0, Ordering::Relaxed);
        o.task.store(0, Ordering::Relaxed);
        o.skipped.store(skipped, Ordering::Relaxed);
        o.open.store(true, Ordering::Release);
    }

    /// `cpu_id` looked at its slot of `dag_id`'s node `node_id` and found it
    /// in `state` (its task: `task_id`, 0 if unknown). First call per slot
    /// only. Must run on `cpu_id`.
    pub(crate) fn observe(cpu_id: usize, dag_id: u32, node_id: u32, state: u8, task_id: u32) {
        // The handler may open a new record on this CPU at any moment.
        let _guard = InterruptGuard::new();
        let Some(o) = OPEN.get(cpu_id) else {
            return;
        };
        if !o.is(dag_id, node_id) || o.state.load(Ordering::Relaxed) != SLOT_UNOBSERVED {
            return;
        }
        o.state.store(state, Ordering::Relaxed);
        o.observed.store(ns(Instant::now()), Ordering::Relaxed);
        o.task.store(task_id, Ordering::Relaxed);
    }

    /// `dag_id`'s node `node_id` was woken now while `cpu_id` holds its
    /// slot. First wake per slot only. Callable from any CPU.
    pub(crate) fn ready(cpu_id: usize, dag_id: u32, node_id: u32, task_id: u32) {
        let Some(o) = OPEN.get(cpu_id) else {
            return;
        };
        if !o.is(dag_id, node_id) {
            return;
        }
        let _ =
            o.ready
                .compare_exchange(0, ns(Instant::now()), Ordering::Relaxed, Ordering::Relaxed);
        let _ = o
            .task
            .compare_exchange(0, task_id, Ordering::Relaxed, Ordering::Relaxed);
    }

    /// `task_id` starts an execution slice on `cpu_id` (own CPU).
    pub(crate) fn task_started(cpu_id: usize, task_id: u32) {
        let _guard = InterruptGuard::new();
        let Some(o) = OPEN.get(cpu_id) else {
            return;
        };
        if o.open.load(Ordering::Acquire)
            && task_id != 0
            && o.task.load(Ordering::Relaxed) == task_id
            && o.started.load(Ordering::Relaxed) == 0
        {
            o.started.store(ns(Instant::now()), Ordering::Relaxed);
        }
    }
}

/// Without `perf` there is no trace to record into.
#[cfg(not(feature = "perf"))]
mod imp {
    use crate::scheduler::dp_wrap::switch_table::NodeSlot;
    use awkernel_lib::time::Instant;

    pub(crate) fn begin(
        _cpu_id: usize,
        _at: Instant,
        _dag_id: Option<u32>,
        _node: Option<NodeSlot>,
        _skipped: u32,
    ) {
    }
    #[cfg_attr(feature = "no_preempt", allow(dead_code))]
    pub(crate) fn observe(_cpu_id: usize, _dag_id: u32, _node_id: u32, _state: u8, _task_id: u32) {}
    pub(crate) fn ready(_cpu_id: usize, _dag_id: u32, _node_id: u32, _task_id: u32) {}
    #[allow(dead_code)]
    pub(crate) fn task_started(_cpu_id: usize, _task_id: u32) {}
}
