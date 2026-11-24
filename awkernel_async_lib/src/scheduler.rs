//! Define types and trait for the Awkernel scheduler.
//! This module contains `SleepingTasks` for sleeping.

use core::sync::atomic::Ordering;
use core::time::Duration;

use crate::{
    task::{get_current_task, get_scheduler_type_by_task_id, State, Task, NUM_TASK_IN_QUEUE},
    time::Instant,
};
use alloc::collections::{binary_heap::BinaryHeap, btree_map::BTreeMap};
use alloc::sync::Arc;
use awkernel_lib::{
    cpu::{num_cpu, CpuSet},
    priority_queue::PriorityQueue,
    sync::mutex::{MCSNode, Mutex},
    verified::delta_list::DeltaList,
};

#[cfg(not(feature = "std"))]
use alloc::boxed::Box;

pub(crate) mod active_vp;
pub mod dp_wrap;
mod edf;
pub(crate) mod mixed_vp;
pub(super) mod panicked;
pub(crate) mod passive_vp;
pub(crate) mod pool;
pub use pool::global_dag_cpu_set;
mod prioritized_fifo;
mod prioritized_rr;

static SLEEPING: Mutex<SleepingTasks> = Mutex::new(SleepingTasks::new());

/// Tasks that request preemption by IPI. The key is the IPI destination CPU ID.
static PREEMPTION_PENDING_TASKS: Mutex<BTreeMap<usize, BinaryHeap<Arc<Task>>>> =
    Mutex::new(BTreeMap::new());

#[inline(always)]
pub fn peek_preemption_pending(cpu_id: usize) -> Option<Arc<Task>> {
    let mut node = MCSNode::new();
    let pending_tasks = PREEMPTION_PENDING_TASKS.lock(&mut node);
    pending_tasks
        .get(&cpu_id)
        .and_then(|heap| heap.peek().cloned())
}

#[inline(always)]
pub fn remove_preemption_pending(cpu_id: usize, task_id: u32) {
    let mut node = MCSNode::new();
    let mut pending_tasks = PREEMPTION_PENDING_TASKS.lock(&mut node);
    if let Some(heap) = pending_tasks.get_mut(&cpu_id) {
        heap.retain(|task| task.id != task_id);
    }
}

#[inline(always)]
pub fn push_preemption_pending(cpu_id: usize, task: Arc<Task>) {
    let mut node = MCSNode::new();
    let mut pending_tasks = PREEMPTION_PENDING_TASKS.lock(&mut node);
    pending_tasks.entry(cpu_id).or_default().push(task);
}

#[inline(always)]
pub fn pop_preemption_pending(cpu_id: usize) -> Option<Arc<Task>> {
    let mut node = MCSNode::new();
    let mut pending_tasks = PREEMPTION_PENDING_TASKS.lock(&mut node);
    pending_tasks.get_mut(&cpu_id).and_then(|heap| heap.pop())
}

#[inline(always)]
pub fn move_preemption_pending(cpu_id: usize) -> Option<BinaryHeap<Arc<Task>>> {
    let mut node = MCSNode::new();
    let mut pending_tasks = PREEMPTION_PENDING_TASKS.lock(&mut node);
    pending_tasks.remove(&cpu_id)
}

/// Type of scheduler.
/// `u8` is the priority of priority based schedulers.
/// 0 is the lowest priority and 31 is the highest priority.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchedulerType {
    /// V-Fed active-VP scheduling: `(cpu_set, leading_cpu)` (see
    /// `dag_sched::policy::vfed` and `scheduler::active_vp`). `cpu_set` is
    /// a heavy DAG's dedicated active-VP group, normalized the same way as
    /// `EDF`'s; `leading_cpu` (a member of `cpu_set`) is the one
    /// core privileged to be tried first for any eligible vertex. Each
    /// core's replenishable per-job execution budget is tracked separately
    /// (`active_vp::set_initial_budget`), not carried here.
    ActiveVp(CpuSet, usize),
    /// V-Fed mixed active+passive scheduling (Theorem 4):
    /// `(active_set, leading_cpu, passive_set)` — a heavy DAG's own
    /// (possibly partial) active-VP group plus other DAGs' leftover
    /// passive-VPs pulled in to top it up (see `dag_sched::policy::vfed`'s
    /// `VFedAssignment::Heavy` with a non-empty `passive`, and
    /// `scheduler::mixed_vp`). `active_set`/`leading_cpu` are normalized and
    /// budget/leading-gated exactly like `ActiveVp`'s; `passive_set` is
    /// normalized and gated exactly like `PassiveVp`'s. Disjoint from
    /// `active_set` by construction (a DAG never borrows its own cores).
    MixedVp(CpuSet, usize, CpuSet),
    /// EDF: `(relative_deadline, cpu_set)`.
    ///
    /// `cpu_set` is the set of cores the task may run on. It is normalized at
    /// spawn time: CPU 0 (the primary core) and any out-of-range bits are
    /// removed. If normalization leaves the set empty — e.g. an empty set, or
    /// one naming only CPU 0 or out-of-range cores — the task is **not**
    /// rejected; it falls back to all worker cores (`1..num_cpu()`) with a
    /// warning.
    ///
    /// `CpuSet::all()` gives global EDF, and a one-core set gives
    /// partitioned EDF. DAG nodes that used to run under the former GEDF
    /// (global over the DAG pool) use the DAG-pool set instead
    /// (`pool::global_dag_cpu_set`).
    EDF(u64, CpuSet),
    /// V-Fed passive-VP scheduling: `cpu_set` (see
    /// `dag_sched::policy::vfed` and `scheduler::passive_vp`) — the set of
    /// other DAGs' leftover active-VP cores this task draws on. Normalized
    /// the same way as `EDF`'s. No leading/budget concept: dispatch
    /// is gated purely by whether each core's active-VP owner is currently
    /// executing there (`passive_vp::active_vp_is_busy`).
    PassiveVp(CpuSet),
    /// DAG-Fluid's real dispatch (`dag_sched::planner` /
    /// `scheduler::dp_wrap`): `dag_id`, the DAG this node belongs to. Unlike
    /// `EDF`/`ActiveVp`, the CPU set a `DpWrap` task may run on is
    /// **not** carried here — it is looked up dynamically at dispatch time
    /// from `scheduler::dp_wrap`'s own per-CPU entitlement table, which
    /// changes at every DP-Wrap switch point (see that module's
    /// own doc). `cpu_set()`/`is_clustered()` therefore both treat this
    /// variant like any non-clustered scheduler.
    DpWrap(u32),
    PrioritizedFIFO(u8),
    PrioritizedRR(u8),
    Panicked,
}

impl SchedulerType {
    pub const fn equals(&self, other: &Self) -> bool {
        matches!(
            (self, other),
            (SchedulerType::ActiveVp(_, _), SchedulerType::ActiveVp(_, _))
                | (
                    SchedulerType::MixedVp(_, _, _),
                    SchedulerType::MixedVp(_, _, _)
                )
                | (SchedulerType::DpWrap(_), SchedulerType::DpWrap(_))
                | (SchedulerType::EDF(_, _), SchedulerType::EDF(_, _))
                | (SchedulerType::PassiveVp(_), SchedulerType::PassiveVp(_))
                | (
                    SchedulerType::PrioritizedFIFO(_),
                    SchedulerType::PrioritizedFIFO(_)
                )
                | (
                    SchedulerType::PrioritizedRR(_),
                    SchedulerType::PrioritizedRR(_)
                )
                | (SchedulerType::Panicked, SchedulerType::Panicked)
        )
    }

    /// Return the CPU affinity set if this is a [`SchedulerType::EDF`],
    /// [`SchedulerType::ActiveVp`], [`SchedulerType::PassiveVp`], or
    /// [`SchedulerType::MixedVp`] scheduler.
    ///
    /// Returns `Some(set)` where `set` is the set of CPU cores (`1..num_cpu()`) the task
    /// may run on. Returns `None` for all other scheduler variants.
    pub const fn cpu_set(&self) -> Option<CpuSet> {
        match self {
            SchedulerType::EDF(_, set) => Some(*set),
            SchedulerType::ActiveVp(set, _) => Some(*set),
            SchedulerType::PassiveVp(set) => Some(*set),
            SchedulerType::MixedVp(active_set, _, passive_set) => {
                Some(active_set.union(*passive_set))
            }
            _ => None,
        }
    }

    /// True if this is a clustered scheduler, i.e. one whose tasks carry a
    /// CPU affinity set. Update this function when adding a new clustered
    /// scheduler; it is the single source of truth for the clustered block
    /// of `PRIORITY_LIST`.
    pub const fn is_clustered(&self) -> bool {
        matches!(
            self,
            SchedulerType::ActiveVp(_, _)
                | SchedulerType::MixedVp(_, _, _)
                | SchedulerType::EDF(_, _)
                | SchedulerType::PassiveVp(_)
        )
    }

    /// True if this scheduler carries DAG light-pool work that is not bound
    /// to a CPU set (DAG-Fluid's `DpWrap`; DAG nodes under `EDF` carry the
    /// DAG-pool set themselves). Update this function when adding a further
    /// DAG-oriented global scheduler; it is the single source of truth for
    /// the DAG-pool prefix of `PRIORITY_LIST` (see `pool::is_dag_pool_core`).
    pub const fn is_dag_pool_scheduler(&self) -> bool {
        matches!(self, SchedulerType::DpWrap(_))
    }
}

/// # Priority
///
/// `priority()` returns the priority of the scheduler for preemption.
///
/// - The highest priority.
///   - DAG-Fluid's `DpWrap` scheduler (the DAG-pool prefix, see
///     `is_dag_pool_scheduler`). Highest so that no other task can take a
///     core away from a DP-Wrap slot: the fluid schedule hands out exact
///     amounts of CPU time per slot, and any task that preempts a slot's
///     node eats into time the plan already gave to that node.
/// - The second highest priority.
///   - Active-VP scheduler (V-Fed).
/// - The third highest priority.
///   - Mixed active+passive-VP scheduler (V-Fed, Theorem 4). Below
///     `ActiveVp` so a mixed task can never preempt a core's *true*
///     active-VP owner when running there as a mere borrower; above
///     the rest because on its own (possibly partial) active-VP cores it is
///     servicing the same heavy-DAG obligation `ActiveVp` does. Relative
///     order versus `EDF`/`PassiveVp` below has no live scenario to matter
///     for: a specific core is claimed by at most one active-VP owner and
///     lent to at most one passive/mixed borrower at a time (see
///     `dag_sched::policy::vfed`'s pool semantics).
/// - The fourth highest priority.
///   - EDF scheduler (clustered, partitioned, and global over a set).
/// - The fifth highest priority.
///   - Passive-VP scheduler (V-Fed). Below `EDF` because a passive-VP's
///     whole premise is "leftover capacity", lower priority than any
///     exclusive cluster reservation elsewhere.
/// - The sixth highest priority.
///   - Prioritized FIFO scheduler.
/// - The seventh highest priority.
///   - Prioritized Round-Robin scheduler.
/// - The lowest priority.
///   - Panicked scheduler.
///
/// The list is laid out as the DAG-pool prefix, then the clustered block,
/// then the regular-pool tail; `get_next_task` relies on that layout (checked
/// at compile time below).
static PRIORITY_LIST: [SchedulerType; 8] = [
    SchedulerType::DpWrap(0),
    SchedulerType::ActiveVp(CpuSet::empty(), 0),
    SchedulerType::MixedVp(CpuSet::empty(), 0, CpuSet::empty()),
    SchedulerType::EDF(0, CpuSet::empty()),
    SchedulerType::PassiveVp(CpuSet::empty()),
    SchedulerType::PrioritizedFIFO(0),
    SchedulerType::PrioritizedRR(0),
    SchedulerType::Panicked,
];

/// Return the number of DAG-pool schedulers (currently just `DpWrap`) at the
/// head of `PRIORITY_LIST`.
const fn get_num_dag_pool_schedulers() -> usize {
    let mut count = 0;
    while count < PRIORITY_LIST.len() && PRIORITY_LIST[count].is_dag_pool_scheduler() {
        count += 1;
    }
    count
}

/// Return the number of clustered schedulers, which follow the DAG-pool
/// prefix of `PRIORITY_LIST`.
const fn get_num_clustered_schedulers() -> usize {
    let start = get_num_dag_pool_schedulers();
    let mut count = 0;
    while start + count < PRIORITY_LIST.len() && PRIORITY_LIST[start + count].is_clustered() {
        count += 1;
    }
    count
}

/// The clustered block of `PRIORITY_LIST`.
fn clustered_schedulers() -> &'static [SchedulerType] {
    let start = get_num_dag_pool_schedulers();
    &PRIORITY_LIST[start..start + get_num_clustered_schedulers()]
}

// `get_next_task` relies on `PRIORITY_LIST` being the DAG-pool prefix, then
// the clustered block, then a tail of neither. Reject any reorder at compile
// time.
const _: () = {
    let mut i = get_num_dag_pool_schedulers() + get_num_clustered_schedulers();
    while i < PRIORITY_LIST.len() {
        assert!(
            !PRIORITY_LIST[i].is_clustered() && !PRIORITY_LIST[i].is_dag_pool_scheduler(),
            "PRIORITY_LIST must be the DAG-pool prefix, then the clustered block, then the rest"
        );
        i += 1;
    }
};

/// When false, no scheduler requests preemption, and `wake_task` enqueues the woken task.
/// `no_preempt` compiles out the IPI handler that runs preemption-pending tasks.
/// Without this flag, a pending task waits until its target CPU goes idle and starves while every worker keeps polling.
pub(crate) const PREEMPTION_ENABLED: bool = cfg!(not(feature = "no_preempt"));

/// For exclusion execution of `wake_task` and `get_next` across all schedulers.
/// In order to resolve priority inversion in multiple priority-based schedulers,
/// the decision to preempt, dequeuing, enqueuing, and updating of RUNNING must be executed exclusively.
static GLOBAL_WAKE_GET_MUTEX: Mutex<()> = Mutex::new(());

/// # Implementing a clustered scheduler
///
/// A clustered scheduler keeps tasks that may run only on a subset of CPUs.
/// Every clustered scheduler must:
///
/// 1. Wrap each entry of its run queue in [`ClusteredTask`] so that
///    `NUM_CLUSTERED_TASKS_IN_QUEUE` is kept consistent.
/// 2. Implement [`Scheduler::queued_cpu_mask`] to report the exact set of
///    CPUs its queued tasks are runnable on (`wake_workers` relies on it).
/// 3. Be placed in the clustered block of `PRIORITY_LIST` and be matched by
///    `SchedulerType::is_clustered` (checked at compile time).
pub(crate) trait Scheduler {
    /// Enqueue an executable task.
    /// The enqueued task will be taken by `get_next()`.
    fn wake_task(&self, task: Arc<Task>);

    /// Get the next executable task.
    fn get_next(&self, execution_ensured: bool) -> Option<Arc<Task>>;

    /// Dequeue the task at the head of the run queue, but only if `pred`
    /// accepts it. A rejected task stays at the head of the run queue.
    /// Unlike `get_next(true)`, the dequeued task is not marked as running.
    ///
    /// The check and the dequeue must be done without releasing the lock of
    /// the run queue, so that the dequeued task is the one `pred` accepted.
    ///
    /// This is used only by `get_next_task_if()`, which runs on the primary
    /// CPU (CPU 0). A scheduler whose tasks must never be taken that way
    /// returns [`ConditionalNext::Empty`].
    fn get_next_if(&self, pred: &dyn Fn(&Task) -> bool) -> ConditionalNext;

    /// Get the scheduler name.
    fn scheduler_name(&self) -> SchedulerType;

    #[allow(dead_code)] // TODO: to be removed
    fn priority(&self) -> u8;

    /// The set of CPUs that have at least one queued task eligible to run on
    /// them. Non-clustered schedulers use the default (empty) implementation;
    /// clustered schedulers must override it.
    fn queued_cpu_mask(&self) -> CpuSet {
        CpuSet::empty()
    }
}

/// Return the union of [`Scheduler::queued_cpu_mask`] over all clustered
/// schedulers: the exact set of CPUs for which a clustered task is queued.
pub(crate) fn clustered_queued_cpu_mask() -> CpuSet {
    let mut mask = CpuSet::empty();
    for scheduler_type in clustered_schedulers() {
        mask = mask.union(get_scheduler(scheduler_type).queued_cpu_mask());
    }
    mask
}

/// Get the next executable task.
#[inline]
pub(crate) fn get_next_task(execution_ensured: bool) -> Option<Arc<Task>> {
    let cpu_id = awkernel_lib::cpu::cpu_id();
    // DAG-Fluid switch latency breakdown (idle path, `perf` only): no-ops
    // unless this CPU has an open switch record.
    dp_wrap::dp_switch::measure::mark(cpu_id, dp_wrap::dp_switch::measure::ENTERED);

    let mut node = MCSNode::new();
    dp_wrap::dp_switch::measure::mark(cpu_id, dp_wrap::dp_switch::measure::PRE_LOCK);
    let _guard = GLOBAL_WAKE_GET_MUTEX.lock(&mut node);
    dp_wrap::dp_switch::measure::mark(cpu_id, dp_wrap::dp_switch::measure::WAKE_LOCK);

    // A CPU reserved by a clustered task's cpu_set (see
    // `task::NUM_CLUSTERED_TASKS_ALIVE`/`task::is_cpu_reserved`) — e.g. a
    // Federated heavy DAG's exclusive cluster — must never run a
    // global-policy (DpWrap/PrioritizedFIFO/PrioritizedRR/Panicked) task, even
    // while its clustered queue is momentarily empty. Otherwise unrelated
    // traffic could occupy a core another DAG's schedulability analysis
    // already counted on being exclusive.
    let reserved = crate::task::is_cpu_reserved(cpu_id);

    // The DAG pool (`DpWrap`, the highest priority) first, on a DAG-pool
    // core (see `pool::is_dag_pool_core`). Its own queue counter
    // (`task::NUM_DAG_POOL_TASK_IN_QUEUE`) skips the lookup in builds and
    // moments without a ready DAG-Fluid node.
    if !reserved
        && pool::is_dag_pool_core(cpu_id)
        && crate::task::NUM_DAG_POOL_TASK_IN_QUEUE.load(Ordering::Relaxed) > 0
    {
        let task = PRIORITY_LIST[..get_num_dag_pool_schedulers()]
            .iter()
            .find_map(|scheduler_type| get_scheduler(scheduler_type).get_next(execution_ensured));
        if task.is_some() {
            return task;
        }
    }

    let num_clustered_tasks = crate::task::NUM_CLUSTERED_TASKS_IN_QUEUE.load(Ordering::Relaxed);

    // The counter is global: it may be non-zero even when no queued clustered
    // task is runnable on this CPU. That is fine because the run queues fail
    // fast (`pop_for_cpu` checks the root affinity mask in O(1)).
    if num_clustered_tasks > 0 {
        let task = clustered_schedulers()
            .iter()
            .find_map(|scheduler_type| get_scheduler(scheduler_type).get_next(execution_ensured));

        // The counter is decremented by the ClusteredTask::take() call inside
        // get_next() (Drop is a no-op once take() has run).
        if task.is_some() {
            return task;
        }
    }

    if reserved || !pool::is_regular_pool_core(cpu_id) {
        return None;
    }

    // The regular pool (PrioritizedFIFO/RR/Panicked), on a regular-pool core.
    // While the pool split is inactive (too few workers), every worker is
    // both a DAG-pool and a regular-pool core — identical to the pre-split
    // behavior. These schedulers count `task::NUM_TASK_IN_QUEUE` at push and
    // pop.
    PRIORITY_LIST[get_num_dag_pool_schedulers() + get_num_clustered_schedulers()..]
        .iter()
        .find_map(|scheduler_type| get_scheduler(scheduler_type).get_next(execution_ensured))
}

/// The result of [`Scheduler::get_next_if`].
pub(crate) enum ConditionalNext {
    /// The run queue has no executable task.
    Empty,

    /// The head of the run queue was rejected and is left in the queue.
    Rejected,

    /// The head of the run queue was accepted and dequeued.
    Taken(Arc<Task>),
}

/// Dequeue the next executable task only if `pred` accepts it.
///
/// This examines the head of the run queues of the non-clustered schedulers
/// in the order of their priorities, except that panicked tasks are never
/// taken. If `pred` returns `false` for it, the run queues are left
/// untouched, so the task keeps its position in the queue, and `false` is
/// returned. The dequeued task is not marked as running.
///
/// If a task is dequeued, it is passed to `on_taken` and `true` is returned.
///
/// `pred` is called with `GLOBAL_WAKE_GET_MUTEX`, the lock of the run queue,
/// and the lock of the task information held. `on_taken` is called with only
/// `GLOBAL_WAKE_GET_MUTEX` held, so what `pred` observed under that mutex
/// still holds, and a worker that takes the mutex afterwards sees the effect
/// of `on_taken`.
///
/// This function must be called on the primary CPU, which can never dequeue
/// a DAG-pool or clustered task, so only the schedulers after those blocks
/// (`PrioritizedFIFO`, `PrioritizedRR`, `Panicked`) are examined.
pub(crate) fn get_next_task_if(
    pred: impl Fn(&Task) -> bool,
    on_taken: impl FnOnce(Arc<Task>),
) -> bool {
    let mut node = MCSNode::new();
    let _guard = GLOBAL_WAKE_GET_MUTEX.lock(&mut node);

    // Skip the DAG-pool prefix too, not only the clustered block: on this
    // branch `PRIORITY_LIST` starts with `DpWrap`, so slicing from
    // `get_num_clustered_schedulers()` alone would include `PassiveVp` and
    // let the primary CPU take a V-Fed task off its reserved cores.
    for scheduler_type in
        &PRIORITY_LIST[get_num_dag_pool_schedulers() + get_num_clustered_schedulers()..]
    {
        match get_scheduler(scheduler_type).get_next_if(&pred) {
            ConditionalNext::Empty => (),
            ConditionalNext::Rejected => return false,
            ConditionalNext::Taken(task) => {
                on_taken(task);
                return true;
            }
        }
    }

    false
}

/// The body of [`Scheduler::get_next_if`] for the schedulers whose run queue
/// is a [`PriorityQueue`]. `task_of` returns the task of a queue entry.
///
/// Finished tasks at the head of the queue are discarded as `get_next()`
/// does. The lock of the run queue must be held by the caller.
fn take_head_if<T>(
    queue: &mut PriorityQueue<T>,
    task_of: impl Fn(&T) -> &Arc<Task>,
    pred: &dyn Fn(&Task) -> bool,
) -> ConditionalNext {
    loop {
        let Some(task) = queue.peek().map(|entry| task_of(entry).clone()) else {
            return ConditionalNext::Empty;
        };

        let mut node = MCSNode::new();
        let mut task_info = task.info.lock(&mut node);

        if matches!(task_info.state, State::Terminated | State::Panicked) {
            drop(task_info);
            queue.pop();
            NUM_TASK_IN_QUEUE.fetch_sub(1, Ordering::Relaxed);
            continue;
        }

        if !pred(&task) {
            return ConditionalNext::Rejected;
        }

        queue.pop();
        NUM_TASK_IN_QUEUE.fetch_sub(1, Ordering::Relaxed);

        if task_info.state == State::Preempted {
            task_info.need_preemption = false;
        }

        drop(task_info);
        return ConditionalNext::Taken(task);
    }
}

/// Get a scheduler.
///
/// Takes `&SchedulerType` so the caller does not copy the full payload (the
/// embedded `CpuSet` makes `SchedulerType` large); only the discriminant is
/// matched.
pub(crate) fn get_scheduler(sched_type: &SchedulerType) -> &'static dyn Scheduler {
    match sched_type {
        SchedulerType::PrioritizedFIFO(_) => &prioritized_fifo::SCHEDULER,
        SchedulerType::PrioritizedRR(_) => &prioritized_rr::SCHEDULER,
        SchedulerType::EDF(_, _) => &edf::SCHEDULER,
        SchedulerType::DpWrap(_) => &dp_wrap::SCHEDULER,
        SchedulerType::ActiveVp(_, _) => &active_vp::SCHEDULER,
        SchedulerType::MixedVp(_, _, _) => &mixed_vp::SCHEDULER,
        SchedulerType::PassiveVp(_) => &passive_vp::SCHEDULER,
        SchedulerType::Panicked => &panicked::SCHEDULER,
    }
}

pub const fn get_priority(sched_type: &SchedulerType) -> u8 {
    let mut index = 0;
    while index < PRIORITY_LIST.len() {
        if PRIORITY_LIST[index].equals(sched_type) {
            return (PRIORITY_LIST.len() - 1 - index) as u8;
        }
        index += 1;
    }
    panic!("Scheduler type not registered in PRIORITY_LIST or equals()")
}

/// RAII wrapper representing one slot in `NUM_CLUSTERED_TASKS_IN_QUEUE`.
///
/// Constructing a `ClusteredTask` increments the counter. The counter is
/// decremented exactly once — either via [`take`] (explicit ownership
/// transfer) or via `Drop` (e.g. when a terminated task is discarded). If
/// `take` has already been called, `Drop` is a no-op.
///
/// [`take`]: ClusteredTask::take
pub(crate) struct ClusteredTask<T> {
    inner: Option<T>,
}

impl<T> ClusteredTask<T> {
    pub(crate) fn new(inner: T) -> Self {
        crate::task::NUM_CLUSTERED_TASKS_IN_QUEUE.fetch_add(1, Ordering::Relaxed);
        Self { inner: Some(inner) }
    }

    /// Take the inner value and decrement the counters.
    ///
    /// Returns `None` if the value has already been taken.
    /// After this call `Drop` will be a no-op.
    pub(crate) fn take(&mut self) -> Option<T> {
        let val = self.inner.take();
        if val.is_some() {
            crate::task::NUM_CLUSTERED_TASKS_IN_QUEUE.fetch_sub(1, Ordering::Relaxed);
        }
        val
    }
}

impl<T> Drop for ClusteredTask<T> {
    fn drop(&mut self) {
        // Decrement only if take() has not been called yet.
        if self.inner.is_some() {
            crate::task::NUM_CLUSTERED_TASKS_IN_QUEUE.fetch_sub(1, Ordering::Relaxed);
        }
    }
}

/// Maintain sleeping tasks by a delta list.
struct SleepingTasks {
    delta_list: DeltaList<Box<dyn FnOnce() + Send>>,
    base_time: Instant,
}

impl SleepingTasks {
    const fn new() -> Self {
        Self {
            delta_list: DeltaList::Nil,
            base_time: Instant::zero(),
        }
    }

    /// `dur` is a Duration.
    fn sleep_task(&mut self, handler: Box<dyn FnOnce() + Send>, mut dur: Duration) {
        if self.delta_list.is_empty() {
            self.base_time = Instant::now();
        } else {
            let diff = self.base_time.elapsed();
            dur += diff;
        }

        self.delta_list.insert(dur.as_nanos() as u64, handler);
    }

    /// Sleep until `next` (no relative duration).
    fn sleep_until_task(&mut self, handler: Box<dyn FnOnce() + Send>, next: Instant) {
        if self.delta_list.is_empty() {
            self.base_time = Instant::now();
        }
        let dur = next - self.base_time;

        self.delta_list.insert(dur.as_nanos() as u64, handler);
    }

    /// Wake tasks up.
    fn wake_task(&mut self) {
        while let Some((dur, _)) = self.delta_list.front() {
            let dur = Duration::from_nanos(dur);
            let elapsed = self.base_time.elapsed();

            if dur <= elapsed {
                // Timed out.
                if let DeltaList::Cons(data) = self.delta_list.pop().unwrap() {
                    let (_, handler, _) = data.into_inner();
                    handler(); // Invoke a handler.

                    self.base_time += dur;
                }
            } else {
                break;
            }
        }
    }

    /// Get the duration of between the current time and the time of the head.
    fn time_to_wait(&self) -> Option<Duration> {
        let (dur, _) = self.delta_list.front()?;
        let elapsed = self.base_time.elapsed().as_nanos() as u64;

        if elapsed >= dur {
            Some(Duration::from_nanos(0))
        } else {
            Some(Duration::from_nanos(dur - elapsed))
        }
    }
}

/// After `dur` time, `sleep_handler` will be invoked.
/// `dur` is microseconds.
pub(crate) fn sleep_task(sleep_handler: Box<dyn FnOnce() + Send>, dur: Duration) {
    {
        let mut node = MCSNode::new();
        let mut guard = SLEEPING.lock(&mut node);
        guard.sleep_task(sleep_handler, dur);
    }

    awkernel_lib::cpu::wake_cpu(0);
}

/// After reaching `next` time, `sleep_until_handler` will be invoked.
pub(crate) fn sleep_until_task(sleep_handler: Box<dyn FnOnce() + Send>, next: Instant) {
    {
        let mut node = MCSNode::new();
        let mut guard = SLEEPING.lock(&mut node);
        guard.sleep_until_task(sleep_handler, next);
    }

    awkernel_lib::cpu::wake_cpu(0);
}

/// Wake executable tasks up.
/// Waked tasks will be enqueued to their scheduler's queue.
///
/// This function should be called from only Awkernel.
/// So, do not call this from userland.
///
/// # Return Value
///
/// If there are sleeping tasks, this function returns the duration to wait.
/// If there are no sleeping tasks, this function returns `None`.
pub fn wake_task() -> Option<Duration> {
    // Check whether each running task exceeds the time quantum (RR) or its
    // active-VP budget (V-Fed) — see `active_vp`'s WCET contract for the
    // bound on this loop's per-cpu work.
    //
    // Only the RR quantum check preempts, so only it is skipped when
    // preemption is disabled. The active-VP budget accounting never sends
    // an IPI itself, and `get_next` gating relies on it either way, so it
    // runs on every call regardless of `PREEMPTION_ENABLED`.
    let mut quantum_wait: Option<Duration> = None;

    for cpu_id in 1..num_cpu() {
        if let Some(task_id) = get_current_task(cpu_id) {
            match get_scheduler_type_by_task_id(task_id) {
                Some(SchedulerType::PrioritizedRR(_)) => {
                    if PREEMPTION_ENABLED {
                        let wait =
                            prioritized_rr::SCHEDULER.invoke_preemption_tick(cpu_id, task_id);
                        quantum_wait = match (quantum_wait, wait) {
                            (Some(a), Some(b)) => Some(a.min(b)),
                            (a, b) => a.or(b),
                        };
                    }
                }
                Some(SchedulerType::ActiveVp(_, _)) => active_vp::tick_budget(cpu_id),
                // Only while running as this group's own active-VP work —
                // running here as a mere borrower of some *other* DAG's
                // passive-VP falls through to the `_` arm below, exactly
                // like a plain `PassiveVp` task already does, which is
                // correct: budget belongs to the core's true owner, not to
                // whoever it's currently lent to.
                Some(SchedulerType::MixedVp(active_set, _, _)) if active_set.contains(cpu_id) => {
                    active_vp::tick_budget(cpu_id)
                }
                _ => active_vp::mark_idle(cpu_id),
            }
        } else {
            active_vp::mark_idle(cpu_id);
        }
    }

    let mut node = MCSNode::new();
    let mut guard = SLEEPING.lock(&mut node);
    guard.wake_task();

    // A queued task may be waiting for a time quantum to expire. Nothing
    // else wakes the primary CPU at that time, so do not sleep beyond it.
    // If no task is queued, the next enqueue wakes the primary CPU.
    //
    // This is checked after `guard.wake_task()`: a sleep handler enqueues
    // its task on this CPU, and `wake_cpu(0)` does nothing here, so such a
    // task must be counted now.
    if NUM_TASK_IN_QUEUE.load(Ordering::Relaxed) == 0 {
        quantum_wait = None;
    }

    match (guard.time_to_wait(), quantum_wait) {
        (Some(a), Some(b)) => Some(a.min(b)),
        (a, b) => a.or(b),
    }
}
