//! Per-CPU task execution trace recorder.
//!
//! Unlike the `perf` module (which accumulates totals), this module records a
//! timeline of events: which task started/ended on which CPU at which time.
//! The dump can be turned into a per-core Gantt chart on the host.
//!
//! Design notes:
//! - Each CPU owns its buffer; the write index is claimed with a per-CPU
//!   `fetch_add`, so recording is lock-free and safe against preemption.
//! - Timestamps are raw CPU counter values (TSC on x86).  A pair of
//!   (cpu_counter, uptime µs) calibration points taken at `start()` and
//!   `stop()` lets the host convert ticks to wall-clock time.
//! - `events()` must only be called after `stop()`; a record racing past the
//!   enabled-check may at worst leave one torn event at the tail.

use crate::atomic_u64::AtomicU64;
use alloc::vec::Vec;
use array_macro::array;
use awkernel_lib::cpu::NUM_MAX_CPU;
use core::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

/// Maximum number of events recorded per CPU.
/// 32768 events x 16 bytes = 512 KiB per CPU, allocated on `start()`.
pub const TRACE_CAP: usize = 32768;

pub const KIND_START: u8 = 0;
pub const KIND_END: u8 = 1;
/// Unused: no producer emits this anymore now that `period_index` is
/// recorded directly on each event (see `TraceEvent::period_index`). Kept
/// only so `dump_to_console`'s event-kind match stays exhaustive against
/// old callers/log readers that still expect the `R` character.
pub const KIND_RELEASE: u8 = 2;

#[derive(Clone, Copy, Default)]
pub struct TraceEvent {
    /// Raw CPU counter value (TSC on x86).
    pub tsc: u64,
    pub task_id: u32,
    /// `KIND_START` or `KIND_END`.
    pub kind: u8,
    /// Which period (job) this slice belongs to, from `DagInfo::period_index`
    /// at the moment this event was recorded. `None` for a non-DAG task, or
    /// when the `period-index-propagation` feature is disabled.
    #[cfg(feature = "period-index-propagation")]
    pub period_index: Option<u32>,
}

/// One applied DAG-Fluid switch point (方式B), with its four timestamps
/// (uptime nanoseconds, the unit switch tables use; 0 = did not happen):
/// scheduled time, `DpSwitch` handler entry, successor decided (the
/// preemption path picked it, or the idle run loop dequeued it), and that
/// successor's first execution slice starting. See
/// `scheduler::dp_wrap::dp_switch::measure`.
#[derive(Clone, Copy, Default)]
pub struct DpSwitchEvent {
    pub scheduled_ns: u64,
    pub handler_ns: u64,
    pub decided_ns: u64,
    pub started_ns: u64,
    /// `u32::MAX` for "no DAG".
    pub from_dag: u32,
    /// `u32::MAX` for "no DAG".
    pub to_dag: u32,
    /// One of the `DPSW_*` paths.
    pub path: u8,
    /// Due entries passed over because the handler ran late.
    pub skipped: u32,
}

pub const DPSW_CAP: usize = 4096;

/// A task of another DAG was running: preempted through IPI 255.
pub const DPSW_PREEMPT: u8 = 0;
/// The CPU was idle: its run loop picks the new DAG up.
pub const DPSW_IDLE: u8 = 1;
/// Nothing to switch: same DAG, or the running task already belongs to
/// the entitled DAG.
pub const DPSW_SAME: u8 = 2;
/// Switched to no DAG.
pub const DPSW_TO_NONE: u8 = 3;
/// The entitled DAG had no ready node, so the running task kept the CPU.
pub const DPSW_NO_SUCCESSOR: u8 = 4;
/// Per-node slots (`dagfluid_node`): the slot's node was not ready, so the
/// running node was stopped (the CPU's placeholder task took over) and the
/// slot is left empty.
pub const DPSW_STOP: u8 = 5;

fn dpsw_path_name(path: u8) -> &'static str {
    match path {
        DPSW_PREEMPT => "preempt",
        DPSW_IDLE => "idle",
        DPSW_SAME => "same",
        DPSW_TO_NONE => "none",
        DPSW_NO_SUCCESSOR => "nosucc",
        DPSW_STOP => "stop",
        _ => "?",
    }
}

static DPSW_IDX: [AtomicUsize; NUM_MAX_CPU] = array![_ => AtomicUsize::new(0); NUM_MAX_CPU];

static mut DPSW_BUFS: [Option<Vec<DpSwitchEvent>>; NUM_MAX_CPU] = [const { None }; NUM_MAX_CPU];

/// Record one switch on the current CPU. Lock-free, like [`record`].
pub(crate) fn record_dp_switch(event: DpSwitchEvent) {
    if !ENABLED.load(Ordering::Acquire) {
        return;
    }
    let cpu_id = awkernel_lib::cpu::cpu_id();
    let i = DPSW_IDX[cpu_id].fetch_add(1, Ordering::Relaxed);
    if i >= DPSW_CAP {
        return;
    }
    unsafe {
        let bufs = &mut *core::ptr::addr_of_mut!(DPSW_BUFS);
        if let Some(slot) = bufs[cpu_id].as_mut().and_then(|buf| buf.get_mut(i)) {
            *slot = event;
        }
    }
}

/// A DAG job event: a release (`DAG_EV_RELEASE`, `node_id` unused) or one
/// node finishing a job (`DAG_EV_NODE_DONE`). Times are uptime
/// nanoseconds.
#[derive(Clone, Copy, Default)]
pub struct DagJobEvent {
    pub kind: u8,
    pub dag_id: u32,
    pub node_id: u32,
    pub job: u64,
    /// Release: the scheduled release time. Node done: when it finished.
    pub at_ns: u64,
    /// Release: when the source node actually woke up for it.
    pub observed_ns: u64,
}

pub const DAG_EV_CAP: usize = 4096;
pub const DAG_EV_RELEASE: u8 = 0;
pub const DAG_EV_NODE_DONE: u8 = 1;

static DAG_EV_IDX: [AtomicUsize; NUM_MAX_CPU] = array![_ => AtomicUsize::new(0); NUM_MAX_CPU];

static mut DAG_EV_BUFS: [Option<Vec<DagJobEvent>>; NUM_MAX_CPU] = [const { None }; NUM_MAX_CPU];

/// Record a DAG job event on the current CPU. Lock-free, like [`record`].
pub(crate) fn record_dag_event(event: DagJobEvent) {
    if !ENABLED.load(Ordering::Acquire) {
        return;
    }
    let cpu_id = awkernel_lib::cpu::cpu_id();
    let i = DAG_EV_IDX[cpu_id].fetch_add(1, Ordering::Relaxed);
    if i >= DAG_EV_CAP {
        return;
    }
    unsafe {
        let bufs = &mut *core::ptr::addr_of_mut!(DAG_EV_BUFS);
        if let Some(slot) = bufs[cpu_id].as_mut().and_then(|buf| buf.get_mut(i)) {
            *slot = event;
        }
    }
}

/// One DAG-Fluid slot as a CPU actually held it (per-node slots, the
/// `dagfluid_node` feature; see `scheduler::dp_wrap::slot_measure`). Times
/// are uptime nanoseconds, 0 for "none".
#[derive(Clone, Copy, Default)]
pub struct SlotEvent {
    /// `u32::MAX` for a no-DAG slot.
    pub dag_id: u32,
    /// `u32::MAX` for a no-DAG slot.
    pub node_id: u32,
    pub job: u64,
    /// The slot's start and end in the CPU's plan.
    pub start_ns: u64,
    pub end_ns: u64,
    /// One of the `SLOT_*` states: the node's state when the CPU first
    /// looked at the slot.
    pub state: u8,
    /// When the CPU first looked at the slot (the state was taken).
    pub observed_ns: u64,
    /// When the slot's node was woken during the slot.
    pub ready_ns: u64,
    /// When the slot's node started running on this CPU during the slot.
    pub started_ns: u64,
    /// Plan entries passed over just before this slot (the handler ran
    /// late).
    pub skipped: u32,
}

pub const SLOT_CAP: usize = 2048;
pub use crate::scheduler::dp_wrap::slot_measure::{
    SLOT_BEHIND, SLOT_DONE, SLOT_NOT_WOKEN, SLOT_NO_DAG, SLOT_READY, SLOT_RUNNING_ELSEWHERE,
    SLOT_RUNNING_HERE, SLOT_UNOBSERVED,
};

fn slot_state_name(state: u8) -> &'static str {
    match state {
        SLOT_UNOBSERVED => "unobserved",
        SLOT_READY => "ready",
        SLOT_RUNNING_HERE => "running_here",
        SLOT_NOT_WOKEN => "not_woken",
        SLOT_DONE => "done",
        SLOT_BEHIND => "behind",
        SLOT_RUNNING_ELSEWHERE => "running_elsewhere",
        SLOT_NO_DAG => "no_dag",
        _ => "?",
    }
}

static SLOT_IDX: [AtomicUsize; NUM_MAX_CPU] = array![_ => AtomicUsize::new(0); NUM_MAX_CPU];

static mut SLOT_BUFS: [Option<Vec<SlotEvent>>; NUM_MAX_CPU] = [const { None }; NUM_MAX_CPU];

/// Record one closed slot on the current CPU. Lock-free, like [`record`].
pub(crate) fn record_slot(event: SlotEvent) {
    if !ENABLED.load(Ordering::Acquire) {
        return;
    }
    let cpu_id = awkernel_lib::cpu::cpu_id();
    let i = SLOT_IDX[cpu_id].fetch_add(1, Ordering::Relaxed);
    if i >= SLOT_CAP {
        return;
    }
    unsafe {
        let bufs = &mut *core::ptr::addr_of_mut!(SLOT_BUFS);
        if let Some(slot) = bufs[cpu_id].as_mut().and_then(|buf| buf.get_mut(i)) {
            *slot = event;
        }
    }
}

static ENABLED: AtomicBool = AtomicBool::new(false);

static IDX: [AtomicUsize; NUM_MAX_CPU] = array![_ => AtomicUsize::new(0); NUM_MAX_CPU];

/// Per CPU: the buffer slot of the latest `KIND_START` event not yet closed
/// by a `KIND_END` (`usize::MAX`: none). A slice's start is recorded before
/// the task's poll learns which period it is working on (a reactor stores
/// `DagInfo::period_index` inside the poll), so its label can be one
/// period behind; the matching end carries the right one, and relabels the
/// start with it (see [`record`]). Starts and ends pair up per CPU: a
/// context switch always records the end first.
#[cfg(feature = "period-index-propagation")]
static OPEN_START: [AtomicUsize; NUM_MAX_CPU] =
    array![_ => AtomicUsize::new(usize::MAX); NUM_MAX_CPU];

static mut BUFS: [Option<Vec<TraceEvent>>; NUM_MAX_CPU] = [const { None }; NUM_MAX_CPU];

static CAL_START_TSC: AtomicU64 = AtomicU64::new(0);
static CAL_START_US: AtomicU64 = AtomicU64::new(0);
static CAL_END_TSC: AtomicU64 = AtomicU64::new(0);
static CAL_END_US: AtomicU64 = AtomicU64::new(0);

/// Record one event.  Called from the task poll boundary in `task::run`.
///
/// Lock-free: only the current CPU writes to its own buffer, and the slot is
/// claimed atomically, so a preempting context on the same CPU cannot corrupt
/// an event nor deadlock.
/// `period_index` is the DAG period this event belongs to (see
/// `DagInfo::period_index`), or `None` for a non-DAG task. Ignored when the
/// `period-index-propagation` feature is off, since `TraceEvent` then has no
/// field to hold it.
#[inline]
pub(crate) fn record(task_id: u32, kind: u8, period_index: Option<u32>) {
    if !ENABLED.load(Ordering::Acquire) {
        return;
    }

    let cpu_id = awkernel_lib::cpu::cpu_id();
    let i = IDX[cpu_id].fetch_add(1, Ordering::Relaxed);
    if i >= TRACE_CAP {
        return; // Buffer full: drop the event (counted via IDX overflow).
    }

    let tsc = awkernel_lib::delay::cpu_counter();

    if kind == KIND_START {
        crate::scheduler::dp_wrap::dp_switch::measure::note_task_start(cpu_id, task_id);
        crate::scheduler::dp_wrap::slot_measure::task_started(cpu_id, task_id);
    }

    unsafe {
        let bufs = &mut *core::ptr::addr_of_mut!(BUFS);
        if let Some(buf) = bufs[cpu_id].as_mut() {
            #[cfg(feature = "period-index-propagation")]
            {
                buf[i] = TraceEvent {
                    tsc,
                    task_id,
                    kind,
                    period_index,
                };
                // Give the slice's start the period its end reports, so
                // both events of one slice carry the same label.
                if kind == KIND_START {
                    OPEN_START[cpu_id].store(i, Ordering::Relaxed);
                } else if kind == KIND_END {
                    let open = OPEN_START[cpu_id].swap(usize::MAX, Ordering::Relaxed);
                    if let Some(start) = buf.get_mut(open) {
                        if start.task_id == task_id && start.kind == KIND_START {
                            start.period_index = period_index;
                        }
                    }
                }
            }
            #[cfg(not(feature = "period-index-propagation"))]
            {
                let _ = period_index;
                buf[i] = TraceEvent { tsc, task_id, kind };
            }
        }
    }
}

/// Allocate buffers (first call only), reset indices, take the start
/// calibration point, and enable recording.
pub fn start() {
    if ENABLED.load(Ordering::Acquire) {
        return;
    }

    let n = awkernel_lib::cpu::num_cpu();

    unsafe {
        let bufs = &mut *core::ptr::addr_of_mut!(BUFS);
        for buf in bufs.iter_mut().take(n) {
            if buf.is_none() {
                *buf = Some(alloc::vec![TraceEvent::default(); TRACE_CAP]);
            }
        }
    }

    for idx in IDX.iter().take(n) {
        idx.store(0, Ordering::Relaxed);
    }
    #[cfg(feature = "period-index-propagation")]
    for open in OPEN_START.iter().take(n) {
        open.store(usize::MAX, Ordering::Relaxed);
    }

    unsafe {
        let bufs = &mut *core::ptr::addr_of_mut!(DPSW_BUFS);
        for buf in bufs.iter_mut().take(n) {
            if buf.is_none() {
                *buf = Some(alloc::vec![DpSwitchEvent::default(); DPSW_CAP]);
            }
        }
    }
    for idx in DPSW_IDX.iter().take(n) {
        idx.store(0, Ordering::Relaxed);
    }

    unsafe {
        let bufs = &mut *core::ptr::addr_of_mut!(SLOT_BUFS);
        for buf in bufs.iter_mut().take(n) {
            if buf.is_none() {
                *buf = Some(alloc::vec![SlotEvent::default(); SLOT_CAP]);
            }
        }
    }
    for idx in SLOT_IDX.iter().take(n) {
        idx.store(0, Ordering::Relaxed);
    }

    unsafe {
        let bufs = &mut *core::ptr::addr_of_mut!(DAG_EV_BUFS);
        for buf in bufs.iter_mut().take(n) {
            if buf.is_none() {
                *buf = Some(alloc::vec![DagJobEvent::default(); DAG_EV_CAP]);
            }
        }
    }
    for idx in DAG_EV_IDX.iter().take(n) {
        idx.store(0, Ordering::Relaxed);
    }

    CAL_START_TSC.store(awkernel_lib::delay::cpu_counter(), Ordering::Relaxed);
    CAL_START_US.store(awkernel_lib::delay::uptime(), Ordering::Relaxed);

    // Start this recording window with an empty pubsub table rather than
    // whatever accumulated the last time recording was on.
    #[cfg(feature = "period-index-propagation")]
    super::perf::reset_pubsub_tables();

    // Buffers must be visible before any CPU starts recording.
    ENABLED.store(true, Ordering::Release);
}

/// Disable recording and take the end calibration point.
pub fn stop() {
    if !ENABLED.swap(false, Ordering::AcqRel) {
        return;
    }

    CAL_END_TSC.store(awkernel_lib::delay::cpu_counter(), Ordering::Relaxed);
    CAL_END_US.store(awkernel_lib::delay::uptime(), Ordering::Relaxed);
}

pub fn is_enabled() -> bool {
    ENABLED.load(Ordering::Acquire)
}

/// (start_tsc, start_us, end_tsc, end_us) for tick-to-time conversion.
pub fn calibration() -> (u64, u64, u64, u64) {
    (
        CAL_START_TSC.load(Ordering::Relaxed),
        CAL_START_US.load(Ordering::Relaxed),
        CAL_END_TSC.load(Ordering::Relaxed),
        CAL_END_US.load(Ordering::Relaxed),
    )
}

/// Number of events dropped on `cpu_id` because the buffer was full.
pub fn num_dropped(cpu_id: usize) -> usize {
    IDX[cpu_id]
        .load(Ordering::Relaxed)
        .saturating_sub(TRACE_CAP)
}

/// Copy of the recorded events on `cpu_id`.  Call only after `stop()`.
pub fn events(cpu_id: usize) -> Vec<TraceEvent> {
    let n = IDX[cpu_id].load(Ordering::Relaxed).min(TRACE_CAP);

    unsafe {
        let bufs = &*core::ptr::addr_of!(BUFS);
        match bufs[cpu_id].as_ref() {
            Some(buf) => buf[..n].to_vec(),
            None => Vec::new(),
        }
    }
}

/// Dump the recorded trace to the console in the `TRACE_*` CSV format
/// consumed by `awkernel_script/trace_view/plot_trace.py`.
///
/// Used both by the shell's `(trace)` command and by the boot-time
/// auto-trace task (real hardware, where no shell input is available:
/// the output is captured on the host with `minicom -C trace.log`).
pub fn dump_to_console() {
    use alloc::{format, string::String};
    use awkernel_lib::console;

    if is_enabled() {
        console::print("trace is running; stop it before dumping\r\n");
        return;
    }

    let (start_tsc, start_us, end_tsc, end_us) = calibration();
    if start_tsc == 0 {
        console::print("no trace recorded\r\n");
        return;
    }

    console::print("TRACE_BEGIN\r\n");
    console::print(&format!("TRACE_CAL,start,{start_tsc},{start_us}\r\n"));
    console::print(&format!("TRACE_CAL,end,{end_tsc},{end_us}\r\n"));

    // TRACE_TASK,<id>,<dag_id|->,<node_id|->,<cpu|cpu|...|->,<name>
    // dag/node identify the DAG membership, the cpu list is the task's
    // `cpu_set` (clustered tasks only).  `-` means "not applicable".
    for t in super::get_tasks() {
        let dag_info = {
            let mut node = awkernel_lib::sync::mutex::MCSNode::new();
            let info = t.info.lock(&mut node);
            info.get_dag_info()
        };
        let (dag_id, node_id) = match dag_info {
            Some(d) => (format!("{}", d.dag_id), format!("{}", d.node_id)),
            None => (String::from("-"), String::from("-")),
        };

        let cpus = match &t.cpu_set {
            Some(set) => {
                let mut s = String::new();
                for cpu in set.iter() {
                    if !s.is_empty() {
                        s.push('|');
                    }
                    s.push_str(&format!("{cpu}"));
                }
                s
            }
            None => String::from("-"),
        };

        console::print(&format!(
            "TRACE_TASK,{},{dag_id},{node_id},{cpus},{}\r\n",
            t.id, t.name
        ));
    }

    // TRACE_DAG,<dag_id>,<src_node_id>,<dst_node_id> — DAG topology, so the
    // host can compute critical paths from the log alone.
    for (dag_id, src, dst) in crate::dag::get_all_dag_edges() {
        console::print(&format!("TRACE_DAG,{dag_id},{src},{dst}\r\n"));
    }

    // TRACE_BUILD_MISS,<dag_id>,<reason> — a DAG that got a dag_id but
    // failed to build (admission rejection, arity error, ...) and so was
    // never spawned and never appears as a TRACE_TASK. Host-side tooling
    // counts these as deadline misses too: the DAG never got a chance to
    // run, which is a miss just as much as a spawned DAG overrunning its
    // deadline. `reason` is free text (a `Display`ed error), so it may
    // contain commas; readers should split on the first two commas only.
    for (dag_id, reason) in crate::dag::take_build_failures() {
        console::print(&format!("TRACE_BUILD_MISS,{dag_id},{reason}\r\n"));
    }

    // TRACE_PUBSUB,<dag_id>,<node_id>,<period>,<publish_ns|->,<subscribe_ns|->
    // Pubsub latency for this same recording window; see
    // `perf::dump_pubsub_to_console` for the format and scoping rationale.
    #[cfg(feature = "period-index-propagation")]
    super::perf::dump_pubsub_to_console();

    for cpu_id in 0..awkernel_lib::cpu::num_cpu() {
        let evs = events(cpu_id);
        let mut out = String::new();
        for e in &evs {
            let kind = match e.kind {
                KIND_START => 'S',
                KIND_END => 'E',
                _ => 'R',
            };

            // TRACE_EV,<cpu>,<task_id>,<S|E>,<tsc>[,<period_index|->]
            // The 6th field is only emitted under `period-index-propagation`,
            // matching `plot_trace.py`'s backward-compatible parsing (an
            // older 5-field line is still accepted).
            #[cfg(feature = "period-index-propagation")]
            {
                match e.period_index {
                    Some(p) => out.push_str(&format!(
                        "TRACE_EV,{cpu_id},{},{kind},{},{p}\r\n",
                        e.task_id, e.tsc
                    )),
                    None => out.push_str(&format!(
                        "TRACE_EV,{cpu_id},{},{kind},{},-\r\n",
                        e.task_id, e.tsc
                    )),
                }
            }
            #[cfg(not(feature = "period-index-propagation"))]
            out.push_str(&format!(
                "TRACE_EV,{cpu_id},{},{kind},{}\r\n",
                e.task_id, e.tsc
            ));

            // Flush in chunks so a single huge String is not required.
            if out.len() > 4096 {
                console::print(&out);
                out.clear();
            }
        }
        console::print(&out);

        // Expected line count so the host can detect serial data loss.
        console::print(&format!("TRACE_COUNT,{cpu_id},{}\r\n", evs.len()));

        let dropped = num_dropped(cpu_id);
        if dropped > 0 {
            console::print(&format!("TRACE_DROPPED,{cpu_id},{dropped}\r\n"));
        }
    }

    // TRACE_DPSW,<cpu>,<scheduled_ns>,<handler_ns>,<decided_ns|->,
    //   <started_ns|->,<from_dag|->,<to_dag|->,<path>,<skipped>
    // One applied DAG-Fluid switch point (see `DpSwitchEvent`).
    let ns_or_dash = |v: u64| if v == 0 { "-".into() } else { format!("{v}") };
    let dag_or_dash = |v: u32| {
        if v == u32::MAX {
            "-".into()
        } else {
            format!("{v}")
        }
    };
    for cpu_id in 0..awkernel_lib::cpu::num_cpu() {
        let n = DPSW_IDX[cpu_id].load(Ordering::Relaxed);
        let mut out = String::new();
        unsafe {
            let bufs = &*core::ptr::addr_of!(DPSW_BUFS);
            if let Some(buf) = bufs[cpu_id].as_ref() {
                for e in buf.iter().take(n.min(DPSW_CAP)) {
                    out.push_str(&format!(
                        "TRACE_DPSW,{cpu_id},{},{},{},{},{},{},{},{}\r\n",
                        e.scheduled_ns,
                        e.handler_ns,
                        ns_or_dash(e.decided_ns),
                        ns_or_dash(e.started_ns),
                        dag_or_dash(e.from_dag),
                        dag_or_dash(e.to_dag),
                        dpsw_path_name(e.path),
                        e.skipped,
                    ));
                    if out.len() > 4096 {
                        console::print(&out);
                        out.clear();
                    }
                }
            }
        }
        console::print(&out);
        if n > DPSW_CAP {
            console::print(&format!("TRACE_DPSW_DROPPED,{cpu_id},{}\r\n", n - DPSW_CAP));
        }
    }

    // TRACE_SLOT,<cpu>,<dag|->,<node|->,<job|->,<start_ns>,<end_ns>,<state>,
    //   <observed_ns|->,<ready_ns|->,<started_ns|->,<skipped>
    // One per-node slot a CPU held (see `SlotEvent`).
    for cpu_id in 0..awkernel_lib::cpu::num_cpu() {
        let n = SLOT_IDX[cpu_id].load(Ordering::Relaxed);
        let mut out = String::new();
        unsafe {
            let bufs = &*core::ptr::addr_of!(SLOT_BUFS);
            if let Some(buf) = bufs[cpu_id].as_ref() {
                for e in buf.iter().take(n.min(SLOT_CAP)) {
                    let no_dag = e.dag_id == u32::MAX;
                    out.push_str(&format!(
                        "TRACE_SLOT,{cpu_id},{},{},{},{},{},{},{},{},{},{}\r\n",
                        dag_or_dash(e.dag_id),
                        dag_or_dash(e.node_id),
                        if no_dag {
                            "-".into()
                        } else {
                            format!("{}", e.job)
                        },
                        e.start_ns,
                        e.end_ns,
                        slot_state_name(e.state),
                        ns_or_dash(e.observed_ns),
                        ns_or_dash(e.ready_ns),
                        ns_or_dash(e.started_ns),
                        e.skipped,
                    ));
                    if out.len() > 4096 {
                        console::print(&out);
                        out.clear();
                    }
                }
            }
        }
        console::print(&out);
        if n > SLOT_CAP {
            console::print(&format!("TRACE_SLOT_DROPPED,{cpu_id},{}\r\n", n - SLOT_CAP));
        }
    }

    // TRACE_REL,<dag_id>,<job>,<release_ns>,<woke_ns>
    //   a job release: its scheduled time and when the source woke for it.
    // TRACE_NDONE,<dag_id>,<node_id>,<job>,<done_ns>
    //   a node finishing a job.
    let mut dropped_dag_events = 0;
    for cpu_id in 0..awkernel_lib::cpu::num_cpu() {
        let n = DAG_EV_IDX[cpu_id].load(Ordering::Relaxed);
        dropped_dag_events += n.saturating_sub(DAG_EV_CAP);
        let mut out = String::new();
        unsafe {
            let bufs = &*core::ptr::addr_of!(DAG_EV_BUFS);
            if let Some(buf) = bufs[cpu_id].as_ref() {
                for e in buf.iter().take(n.min(DAG_EV_CAP)) {
                    match e.kind {
                        DAG_EV_RELEASE => out.push_str(&format!(
                            "TRACE_REL,{},{},{},{}\r\n",
                            e.dag_id, e.job, e.at_ns, e.observed_ns
                        )),
                        _ => out.push_str(&format!(
                            "TRACE_NDONE,{},{},{},{}\r\n",
                            e.dag_id, e.node_id, e.job, e.at_ns
                        )),
                    }
                    if out.len() > 4096 {
                        console::print(&out);
                        out.clear();
                    }
                }
            }
        }
        console::print(&out);
    }
    if dropped_dag_events > 0 {
        console::print(&format!("TRACE_DAG_EV_DROPPED,{dropped_dag_events}\r\n"));
    }

    console::print("TRACE_END\r\n");
}
