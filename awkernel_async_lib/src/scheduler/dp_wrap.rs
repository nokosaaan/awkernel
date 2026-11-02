//! DP-WRAP's core computation (Funk, Levin, Sadowski, Pye, Brandt, "DP-FAIR:
//! a unifying theory for optimal hard real-time multiprocessor scheduling",
//! Real-Time Systems 2011, Section 4.2): McNaughton's wrap-around algorithm,
//! used here as DAG-Fluid's real dispatch mechanism within one Deadline
//! Partition (DP). `dag_sched::planner` lays out every DP of the coming
//! span with [`wrap_blocks`] and publishes each CPU's switch points to
//! [`switch_table`]; [`dp_switch`] applies them on each CPU at their own
//! times.
//!
//! # The algorithm, transcribed from the primary source (Section 4.2)
//!
//! "To schedule jobs in σj, make a 'block' of length δi for each Ti, and
//! line these blocks up along a number line (in any order), starting at
//! zero. Their total length will be no more than m. Split this stack of
//! blocks into length 1 chunks at 1, 2, ..., m − 1, and assign each chunk
//! to its own processor. Each length 1 chunk of tasks represents the
//! scheduling of tasks on the respective processor; tasks which are sliced
//! in two migrate between their two processors [...] To find the actual
//! timing points of context switches [...] within any σj, multiply each
//! length 1 segment by Lj."
//!
//! For DAG-Fluid, each "task" `Ti` in the quote above is one currently-
//! active segment of one admitted DAG-Fluid DAG, and `δi` is that
//! segment's own execution rate `θ_i,j` (`dag_fluid::SegmentSchedule::rate`,
//! computed in Phase 1) rather than a whole periodic task's density -- the
//! DP-FAIR paper's `δi = ei/min(pi,Di)` and DAG-Fluid's `θ_i,j` play the
//! same structural role (a fraction of one processor a unit of work is
//! entitled to for the duration of one scheduling interval), which is why
//! the same wrap-around construction applies directly.
//!
//! # WCET note
//! By explicit user direction (this is a measurement probe for the paper's
//! theory-vs-reality discussion, not a WCET-proven dispatch mechanism),
//! this module does not carry a proven WCET bound. It stays
//! allocation-bounded and panic-free regardless: [`wrap_blocks`] allocates
//! exactly `m` `Vec`s up front (`(0..m).map(...)`, no further reallocation
//! beyond ordinary `Vec::push` growth bounded by `densities.len()`), and
//! every arithmetic path is on plain `f64` (no indexing that can panic: the
//! `proc >= m` case is checked and simply stops emitting further blocks
//! rather than indexing out of bounds -- reachable only if the *caller*
//! violates the `Σdensities <= m` precondition, at which point silently
//! dropping the remainder here is the same "defensive, not expected"
//! posture `dag_fluid::heavy_capacity_and_deadline` documents for its own
//! analogous precondition).

pub mod dp_switch;
pub mod switch_table;

use super::{
    get_priority, peek_preemption_pending, push_preemption_pending, Scheduler, SchedulerType, Task,
    GLOBAL_WAKE_GET_MUTEX, PREEMPTION_ENABLED,
};
use crate::{
    dag::calculate_and_update_dag_deadline,
    task::{
        get_task, get_task_running, get_tasks_running, set_current_task, set_need_preemption,
        State, MAX_TASK_PRIORITY,
    },
};
use alloc::{
    collections::{BTreeMap, BinaryHeap},
    sync::Arc,
    vec::Vec,
};
use array_macro::array;
use awkernel_lib::{
    cpu::NUM_MAX_CPU,
    sync::mutex::{MCSNode, Mutex},
};
use core::{
    cmp::max,
    sync::atomic::{AtomicU32, Ordering},
};

/// McNaughton's wrap-around algorithm. `densities` are `(BlockId, density)`
/// pairs in the order to lay them along the number line (Funk et al.'s "in
/// any order" -- see this module's own doc for what `density` means for a
/// DAG-Fluid segment); `Σdensity <= m` is the caller's precondition (the
/// same shared-pool admission check `dag_sched::resource::reserve_dagfluid_capacity`
/// already enforces).
///
/// Returns one `Vec` per processor (`result.len() == m`, `result[p]` is
/// processor `p`'s own dispatch order), each an ordered list of `(index
/// into densities, start, end)`. `start`/`end` are offsets *within that
/// processor's own unit-length chunk* (`0.0..=1.0`), not yet multiplied by
/// the DP's own length `Lj` -- the paper's own "except for this last
/// multiplication, all calculations can be done once as a preprocessing
/// step" (Sect. 4.2); the caller scales by `Lj` (see this module's own
/// tests for a worked example of both steps together).
pub fn wrap_blocks(densities: &[(usize, f64)], m: usize) -> Vec<Vec<(usize, f64, f64)>> {
    let mut result: Vec<Vec<(usize, f64, f64)>> = (0..m).map(|_| Vec::new()).collect();
    let mut pos = 0.0_f64;

    for &(id, density) in densities {
        let mut remaining = density;
        while remaining > 0.0 {
            let proc = pos as usize;
            if proc >= m {
                // Caller violated the `Σdensity <= m` precondition -- see
                // this module's own doc for why this is a defensive stop,
                // not a panic.
                break;
            }
            let chunk_offset = pos - proc as f64;
            let space_in_chunk = 1.0 - chunk_offset;
            let take = remaining.min(space_in_chunk);
            result[proc].push((id, chunk_offset, chunk_offset + take));
            remaining -= take;
            pos += take;
        }
    }

    result
}

/// DAG-Fluid's real dispatch (`SchedulerType::DpWrap`): a task's own CPU
/// affinity is not fixed at spawn time (unlike `ClusteredEDF`/`ActiveVp`)
/// -- it is *which DAG a CPU is currently entitled to*, set by
/// [`dp_switch`] at each switch point of the CPU's own switch table. Once
/// a CPU is entitled to `dag_id`, any of that DAG's
/// own ready nodes may run there -- entitlement is per-DAG, not per
/// individual node or per-segment (see this crate's own design discussion
/// in `applications/rd_gen_to_dags/src/dag_fluid.rs`'s
/// `segment_completion_gates` doc for why per-segment tracking was ruled
/// out: a node's own execution can legitimately span more than one
/// segment).
const NO_DAG: u32 = u32::MAX;

/// Per-CPU: the `dag_id` this CPU is currently entitled to, or [`NO_DAG`].
/// Read by [`DpWrapScheduler::get_next`] (a plain array load, O(1),
/// alloc-free, panic-free), written only by [`dp_switch`] on the CPU itself
/// (from its `DpSwitch` timer interrupt, or the IPI 255 path after a kick).
static ENTITLEMENT: [AtomicU32; NUM_MAX_CPU] = array![_ => AtomicU32::new(NO_DAG); NUM_MAX_CPU];

/// The `dag_id` `cpu_id` is currently entitled to, if any.
pub(crate) fn entitlement_of(cpu_id: usize) -> Option<u32> {
    match ENTITLEMENT.get(cpu_id) {
        Some(slot) => match slot.load(Ordering::Relaxed) {
            NO_DAG => None,
            dag_id => Some(dag_id),
        },
        None => None,
    }
}

/// One ready DAG-Fluid node, queued under its own `dag_id` (see
/// [`DpWrapData`]). Ordering is identical to `gedf::GEDFTask`'s (same
/// `(absolute_deadline, node_priority, wake_time)` key, smaller
/// `absolute_deadline` first) -- reused verbatim rather than re-derived,
/// since within one DAG's own ready pool the tie-break rationale is the
/// same one GEDF already documents.
struct DpWrapTask {
    task: Arc<Task>,
    absolute_deadline: u64,
    node_priority: u64,
    wake_time: u64,
}

impl PartialOrd for DpWrapTask {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for DpWrapTask {
    fn eq(&self, other: &Self) -> bool {
        self.absolute_deadline == other.absolute_deadline
            && self.node_priority == other.node_priority
            && self.wake_time == other.wake_time
    }
}

impl Ord for DpWrapTask {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        match other.absolute_deadline.cmp(&self.absolute_deadline) {
            core::cmp::Ordering::Equal => match self.node_priority.cmp(&other.node_priority) {
                core::cmp::Ordering::Equal => other.wake_time.cmp(&self.wake_time),
                ord => ord,
            },
            ord => ord,
        }
    }
}

impl Eq for DpWrapTask {}

/// One ready-queue per `dag_id`, rather than one global queue (GEDF) or one
/// queue per fixed `CpuSet` (ClusteredEDF): the entitled CPU set for a
/// given `dag_id` changes over time (see [`ENTITLEMENT`]), but the queue of
/// *that DAG's own* ready nodes does not need to move when it does.
struct DpWrapData {
    queues: BTreeMap<u32, BinaryHeap<DpWrapTask>>,
}

impl DpWrapData {
    fn new() -> Self {
        Self {
            queues: BTreeMap::new(),
        }
    }
}

pub struct DpWrapScheduler {
    data: Mutex<Option<DpWrapData>>,
    priority: u8,
}

impl Scheduler for DpWrapScheduler {
    fn wake_task(&self, task: Arc<Task>) {
        let (wake_time, absolute_deadline, node_priority, dag_id) = {
            let mut node_inner = MCSNode::new();
            let mut info = task.info.lock(&mut node_inner);
            let dag_info = info.get_dag_info();
            match info.scheduler_type {
                SchedulerType::DpWrap(dag_id) => {
                    let wake_time = awkernel_lib::delay::uptime();
                    let (absolute_deadline, node_priority) = if let Some(ref dag_info) = dag_info {
                        (
                            calculate_and_update_dag_deadline(dag_info, wake_time, info.state),
                            crate::dag::get_node_priority(dag_info.dag_id, dag_info.node_id),
                        )
                    } else {
                        // Every `DpWrap` task is a DAG-Fluid node and should
                        // always carry `dag_info` -- defensive fallback
                        // mirroring GEDF's own "no dag_info" arm rather than
                        // panicking on a should-never-happen case.
                        (wake_time, 0)
                    };

                    task.priority
                        .update_priority_info(self.priority, MAX_TASK_PRIORITY - absolute_deadline);
                    info.update_absolute_deadline(absolute_deadline);

                    (wake_time, absolute_deadline, node_priority, dag_id)
                }
                _ => unreachable!(),
            }
        };

        let mut node = MCSNode::new();
        let _guard = GLOBAL_WAKE_GET_MUTEX.lock(&mut node);
        if !self.invoke_preemption(task.clone(), dag_id) {
            let mut node_inner = MCSNode::new();
            let mut data = self.data.lock(&mut node_inner);
            let internal_data = data.get_or_insert_with(DpWrapData::new);
            internal_data
                .queues
                .entry(dag_id)
                .or_default()
                .push(DpWrapTask {
                    task: task.clone(),
                    absolute_deadline,
                    node_priority,
                    wake_time,
                });
        }
    }

    fn get_next(&self, execution_ensured: bool) -> Option<Arc<Task>> {
        let cpu_id = awkernel_lib::cpu::cpu_id();
        let dag_id = entitlement_of(cpu_id)?;

        let mut node = MCSNode::new();
        let mut data = self.data.lock(&mut node);
        let data = (*data).as_mut()?;
        let queue = data.queues.get_mut(&dag_id)?;

        loop {
            let task = queue.pop()?;

            {
                let mut node = MCSNode::new();
                let mut task_info = task.task.info.lock(&mut node);

                if matches!(task_info.state, State::Terminated | State::Panicked) {
                    continue;
                }

                if task_info.state == State::Preempted {
                    task_info.need_preemption = false;
                }
                if execution_ensured {
                    task_info.state = State::Running;
                    set_current_task(awkernel_lib::cpu::cpu_id(), task.task.id);
                    // The run loop picked the entitled DAG's task up: the
                    // successor of an idle-path switch (`dp_switch::measure`).
                    dp_switch::measure::decided(cpu_id, task.task.id);
                }
            }

            return Some(task.task);
        }
    }

    fn scheduler_name(&self) -> SchedulerType {
        SchedulerType::DpWrap(0)
    }

    fn priority(&self) -> u8 {
        self.priority
    }
}

pub static SCHEDULER: DpWrapScheduler = DpWrapScheduler {
    data: Mutex::new(None),
    priority: get_priority(&SchedulerType::DpWrap(0)),
};

impl DpWrapScheduler {
    /// Same shape as `gedf::GEDFScheduler::invoke_preemption`, restricted to
    /// CPUs currently entitled to `dag_id` (see [`ENTITLEMENT`]) rather than
    /// every running CPU -- a `DpWrap` task must never preempt a CPU that
    /// some *other* DAG currently owns the entitlement for.
    fn invoke_preemption(&self, task: Arc<Task>, dag_id: u32) -> bool {
        if !PREEMPTION_ENABLED {
            return false;
        }

        let tasks_running = get_tasks_running()
            .into_iter()
            .filter(|rt| rt.task_id != 0) // Filter out idle CPUs.
            .collect::<alloc::vec::Vec<_>>();

        if tasks_running.iter().any(|rt| rt.task_id == task.id) {
            return false;
        }

        // An idle CPU currently entitled to `dag_id` will pick this task up
        // on its own next `get_next` poll; no forced preemption needed.
        let entitled_idle_cpu_exists = (1..awkernel_lib::cpu::num_cpu())
            .any(|cpu| entitlement_of(cpu) == Some(dag_id) && get_task_running(cpu).task_id == 0);
        if entitled_idle_cpu_exists {
            return false;
        }

        let preemption_target = tasks_running
            .iter()
            .filter(|rt| {
                !crate::task::is_cpu_reserved(rt.cpu_id)
                    && entitlement_of(rt.cpu_id) == Some(dag_id)
            })
            .filter_map(|rt| {
                get_task(rt.task_id).map(|t| {
                    let highest_pending = peek_preemption_pending(rt.cpu_id).unwrap_or(t.clone());
                    (max(t, highest_pending), rt.cpu_id)
                })
            })
            .min();

        let Some((target_task, target_cpu)) = preemption_target else {
            return false;
        };
        if task.preempts(&target_task) {
            push_preemption_pending(target_cpu, task);
            let preempt_irq = awkernel_lib::interrupt::get_preempt_irq();
            set_need_preemption(target_task.id, target_cpu);
            awkernel_lib::interrupt::send_ipi(preempt_irq, target_cpu as u32);
            return true;
        }

        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    /// Funk et al.'s own worked example (Fig. 7): seven tasks with
    /// densities `[0.3, 0.5, 0.5, 0.6, 0.5, 0.4, 0.2]` (summing to
    /// exactly `3 = m`) on 3 processors, lined up in that order. Hand
    /// re-derived from the paper's own algorithm description (Sect. 4.2)
    /// independently of the figure's pixels, then cross-checked against
    /// the figure's own textual claims: "Task 3 migrates from Processor 2
    /// to Processor 1" and "Task 5 migrates from Processor 3 to Processor
    /// 2" (1-indexed in the paper; tasks 3 and 5 are indices 2 and 4
    /// here) -- i.e. each migrating task runs at the *end* of its first
    /// processor's chunk and the *start* of its second, matching "tasks
    /// which migrate are run at the beginning of the slice on one
    /// processor, and at the end on the other" (Sect. 4.2).
    #[test]
    fn test_wrap_blocks_matches_paper_worked_example() {
        let densities: Vec<(usize, f64)> = vec![
            (0, 0.3),
            (1, 0.5),
            (2, 0.5),
            (3, 0.6),
            (4, 0.5),
            (5, 0.4),
            (6, 0.2),
        ];
        let result = wrap_blocks(&densities, 3);
        assert_eq!(result.len(), 3);

        // `f64` accumulation (`pos += take`) leaves the usual last-bit
        // rounding noise (e.g. `0.30000000000000004`), so compare with a
        // tolerance rather than `assert_eq!` on the tuples directly.
        fn assert_close(actual: &[(usize, f64, f64)], expected: &[(usize, f64, f64)]) {
            assert_eq!(actual.len(), expected.len());
            for (&(aid, astart, aend), &(eid, estart, eend)) in actual.iter().zip(expected) {
                assert_eq!(aid, eid);
                assert!((astart - estart).abs() < 1e-9, "{astart} vs {estart}");
                assert!((aend - eend).abs() < 1e-9, "{aend} vs {eend}");
            }
        }

        assert_close(&result[0], &[(0, 0.0, 0.3), (1, 0.3, 0.8), (2, 0.8, 1.0)]);
        assert_close(&result[1], &[(2, 0.0, 0.3), (3, 0.3, 0.9), (4, 0.9, 1.0)]);
        assert_close(&result[2], &[(4, 0.0, 0.4), (5, 0.4, 0.8), (6, 0.8, 1.0)]);

        // Every block's total assigned time (summed across whichever
        // processor(s) it landed on) equals its own density -- Sect. 4.2's
        // own invariant ("their total length will be no more than m").
        let mut totals = [0.0_f64; 7];
        for proc in &result {
            for &(id, start, end) in proc {
                totals[id] += end - start;
            }
        }
        for (id, &(_, expected_density)) in densities.iter().enumerate() {
            assert!((totals[id] - expected_density).abs() < 1e-9, "block {id}");
        }
    }

    /// A block that fits entirely within one processor's chunk (no
    /// migration) is a single `(id, 0.0, density)` entry there and
    /// nowhere else.
    #[test]
    fn test_wrap_blocks_single_task_no_migration() {
        let densities = vec![(0, 0.5)];
        let result = wrap_blocks(&densities, 2);
        assert_eq!(result[0], vec![(0, 0.0, 0.5)]);
        assert_eq!(result[1], vec![]);
    }

    /// Scaling step (the paper's own "multiply each length 1 segment by
    /// Lj"): with a DP of length `Lj = 10`, processor 0's chunk from the
    /// worked example above becomes absolute-within-DP offsets
    /// `[0, 3), [3, 8), [8, 10)`.
    #[test]
    fn test_scaling_by_slice_length() {
        let densities: Vec<(usize, f64)> = vec![(0, 0.3), (1, 0.5), (2, 0.5)];
        let result = wrap_blocks(&densities, 1);
        let l_j = 10.0_f64;
        let scaled: Vec<(usize, f64, f64)> = result[0]
            .iter()
            .map(|&(id, s, e)| (id, s * l_j, e * l_j))
            .collect();
        assert_eq!(scaled, vec![(0, 0.0, 3.0), (1, 3.0, 8.0), (2, 8.0, 10.0)]);
    }
}
