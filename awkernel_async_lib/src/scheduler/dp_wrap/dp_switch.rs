//! Per-core switching for DAG-Fluid (方式B, Step 3): each DAG-pool CPU's own
//! `DpSwitch` timer applies the next entry of its switch table
//! ([`super::switch_table`]) and, when that takes the CPU away from the DAG
//! it is running, hands the actual context switch to the existing
//! preemption path through a preemption IPI to itself.
//!
//! # Who does what
//! - [`on_dp_switch`] (the `DpSwitch` timer callback, interrupt context,
//!   own CPU): consumes the due entries, stores the new DAG into
//!   `ENTITLEMENT`, re-arms the timer for the next entry, and if a task of
//!   another DAG is running, marks the CPU dirty and sends IPI 255 to
//!   itself. Only atomics and the timer multiplexer's own lock are touched:
//!   taking DAG or task locks from the timer interrupt stopped later timers
//!   from firing before (see `dag_sched::dp_partition`'s "Split design").
//! - [`prepare_preemption_on_cpu`] (start of `do_preemption`, the IPI 255
//!   path): if the CPU is dirty, picks the successor from the entitled DAG's
//!   ready queue and queues it as a preemption-pending task.
//! - [`take_rotation`] (inside `do_preemption`): lets that successor
//!   preempt the running task even without a higher priority. DP-Wrap
//!   gives CPU time by rate, not by priority, so a node of the newly
//!   entitled DAG must win even if its deadline is later -- the same kind
//!   of exception the PrioritizedRR quantum already gets there.
//!
//! # Starting and restarting a CPU's plan
//! A CPU arms its own timer (`timer::request_at` only arms the calling
//! CPU), so the planner, which runs elsewhere, cannot. After publishing the
//! first plan, or a plan for a CPU whose plan had run out, the planner
//! calls [`kick_cpu`]: IPI 255 always reaches the CPU, running a task or
//! not, and [`prepare_preemption_on_cpu`] then polls the table and arms the
//! timer. From then on the CPU keeps itself armed.

use super::{switch_table, ENTITLEMENT, NO_DAG};
use crate::task::{get_current_task, request_preemption};
use alloc::boxed::Box;
use array_macro::array;
use awkernel_lib::{
    cpu::NUM_MAX_CPU,
    time::Instant,
    timer::{self, TimerRequestId},
};
use core::sync::atomic::{AtomicBool, Ordering};

#[cfg(not(feature = "no_preempt"))]
pub(crate) use preemption_side::{prepare_preemption_on_cpu, take_rotation};

/// Set by the timer callback when the CPU's entitlement moved away from the
/// DAG of the task it may be running; cleared by
/// [`prepare_preemption_on_cpu`].
static ENTITLEMENT_DIRTY: [AtomicBool; NUM_MAX_CPU] =
    array![_ => AtomicBool::new(false); NUM_MAX_CPU];

/// Set by [`kick_cpu`]: poll the switch table and arm the timer on the next
/// IPI 255.
static KICK: [AtomicBool; NUM_MAX_CPU] = array![_ => AtomicBool::new(false); NUM_MAX_CPU];

/// What an entitlement change means for the CPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SwitchAction {
    /// Same DAG, or nothing newly entitled to switch to.
    Nothing,
    /// The CPU is idle; its run loop picks the new DAG up by itself.
    PickUpWhenIdle,
    /// A task may be running for another DAG: preempt it.
    Preempt,
}

fn switch_action(old: Option<u32>, new: Option<u32>, running: bool) -> SwitchAction {
    if old == new || new.is_none() {
        // With nothing newly entitled there is no successor to switch to;
        // a running node of the old DAG keeps the CPU until it yields.
        SwitchAction::Nothing
    } else if running {
        SwitchAction::Preempt
    } else {
        SwitchAction::PickUpWhenIdle
    }
}

/// Store `new` as `cpu_id`'s entitlement. Returns true if a running task
/// now has to be preempted (the CPU has been marked dirty).
fn apply_due(cpu_id: usize, new: Option<u32>) -> bool {
    let Some(slot) = ENTITLEMENT.get(cpu_id) else {
        return false;
    };
    let old = slot.swap(new.unwrap_or(NO_DAG), Ordering::AcqRel);
    let old = (old != NO_DAG).then_some(old);
    match switch_action(old, new, get_current_task(cpu_id).is_some()) {
        SwitchAction::Preempt => {
            if let Some(dirty) = ENTITLEMENT_DIRTY.get(cpu_id) {
                dirty.store(true, Ordering::Release);
            }
            true
        }
        SwitchAction::Nothing | SwitchAction::PickUpWhenIdle => false,
    }
}

/// Apply every entry of `cpu_id`'s switch table due at `now`, then arm the
/// timer for the next one. Returns true if a preemption is needed. Must
/// run on `cpu_id`.
fn service_switch_table(cpu_id: usize, now: Instant) -> bool {
    let Ok(result) = switch_table::poll_due_switch(cpu_id, now) else {
        return false;
    };
    // Apply before arming: arming a time that has already passed fires the
    // callback again right here, and that later entry must win.
    let need_preemption = result.due.is_some_and(|due| apply_due(cpu_id, due.dag_id));
    if let Some(next) = result.next_at {
        timer::request_at(TimerRequestId::DpSwitch, next);
    }
    need_preemption
}

/// The `DpSwitch` timer callback. Interrupt context, own CPU only.
fn on_dp_switch() {
    let cpu_id = awkernel_lib::cpu::cpu_id();
    if service_switch_table(cpu_id, Instant::now()) {
        // The IPI is the normal path; the request flag makes the next
        // voluntary preemption point catch it if the IPI were lost.
        request_preemption(cpu_id);
        let preempt_irq = awkernel_lib::interrupt::get_preempt_irq();
        awkernel_lib::interrupt::send_ipi(preempt_irq, cpu_id as u32);
    }
}

/// Register [`on_dp_switch`] as the `DpSwitch` timer callback. Call once at
/// boot, before any plan is published.
pub fn install() {
    timer::register_timer_callback(TimerRequestId::DpSwitch, Box::new(on_dp_switch));
}

/// Planner side: make `cpu_id` poll its switch table and arm its timer
/// (after its first plan, or after its plan had run out).
pub fn kick_cpu(cpu_id: usize) {
    let Some(kick) = KICK.get(cpu_id) else {
        return;
    };
    kick.store(true, Ordering::Release);
    let preempt_irq = awkernel_lib::interrupt::get_preempt_irq();
    awkernel_lib::interrupt::send_ipi(preempt_irq, cpu_id as u32);
}

/// The parts that only exist with preemption (`do_preemption` is compiled
/// out under `no_preempt`, e.g. the `std` build).
#[cfg(not(feature = "no_preempt"))]
mod preemption_side {
    use super::{
        super::{entitlement_of, DpWrapTask, SCHEDULER},
        *,
    };
    use crate::{
        scheduler::{push_preemption_pending, SchedulerType, Task, GLOBAL_WAKE_GET_MUTEX},
        task::{get_task, State},
    };
    use alloc::sync::Arc;
    use awkernel_lib::sync::mutex::MCSNode;
    use core::sync::atomic::AtomicU32;

    /// The task id [`prepare_preemption_on_cpu`] queued as the successor for an
    /// entitlement change (0: none). Only that task gets [`take_rotation`]'s
    /// exception.
    static ROTATION_TASK: [AtomicU32; NUM_MAX_CPU] = array![_ => AtomicU32::new(0); NUM_MAX_CPU];

    /// The DAG a `DpWrap` task belongs to, or `None` for any other scheduler.
    fn dpwrap_dag_of(task: &Task) -> Option<u32> {
        let mut node = MCSNode::new();
        let info = task.info.lock(&mut node);
        match info.scheduler_type {
            SchedulerType::DpWrap(dag_id) => Some(dag_id),
            _ => None,
        }
    }

    /// Pop the first runnable task of `dag_id`'s ready queue.
    fn pop_successor(dag_id: u32) -> Option<Arc<Task>> {
        let mut node = MCSNode::new();
        let _guard = GLOBAL_WAKE_GET_MUTEX.lock(&mut node);
        let mut node_inner = MCSNode::new();
        let mut data = SCHEDULER.data.lock(&mut node_inner);
        let queue = (*data).as_mut()?.queues.get_mut(&dag_id)?;
        loop {
            let DpWrapTask { task, .. } = queue.pop()?;
            let mut node = MCSNode::new();
            let state = task.info.lock(&mut node).state;
            if !matches!(state, State::Terminated | State::Panicked) {
                return Some(task);
            }
        }
    }

    /// Called at the start of `do_preemption` on `cpu_id` (the IPI 255 path).
    /// Handles a pending [`kick_cpu`], then, if the entitlement moved away from
    /// the running task's DAG, queues the entitled DAG's next task as the
    /// preemption-pending successor.
    pub(crate) fn prepare_preemption_on_cpu(cpu_id: usize) {
        if KICK
            .get(cpu_id)
            .is_some_and(|kick| kick.swap(false, Ordering::AcqRel))
        {
            // Sets the dirty flag itself if the kick brings a switch.
            service_switch_table(cpu_id, Instant::now());
        }

        let dirty = ENTITLEMENT_DIRTY
            .get(cpu_id)
            .is_some_and(|dirty| dirty.swap(false, Ordering::AcqRel));
        if !dirty {
            return;
        }
        let Some(dag_id) = entitlement_of(cpu_id) else {
            return;
        };
        let Some(current) = get_current_task(cpu_id).and_then(get_task) else {
            // Went idle meanwhile: the run loop picks the DAG up by itself.
            return;
        };
        if dpwrap_dag_of(&current) == Some(dag_id) {
            return;
        }
        // No ready node of the entitled DAG: the running task keeps the CPU
        // until one wakes up (`DpWrapScheduler::invoke_preemption`).
        let Some(successor) = pop_successor(dag_id) else {
            return;
        };
        if let Some(rotation) = ROTATION_TASK.get(cpu_id) {
            rotation.store(successor.id, Ordering::Release);
        }
        push_preemption_pending(cpu_id, successor);
    }

    /// Called by `do_preemption` before its priority check: true if `next` is
    /// the successor [`prepare_preemption_on_cpu`] queued for an entitlement
    /// change and may therefore preempt `current` regardless of priority.
    pub(crate) fn take_rotation(cpu_id: usize, current: &Task, next: &Task) -> bool {
        let Some(rotation) = ROTATION_TASK.get(cpu_id) else {
            return false;
        };
        if rotation
            .compare_exchange(next.id, 0, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return false;
        }
        rotation_allowed(
            entitlement_of(cpu_id),
            dpwrap_dag_of(current),
            dpwrap_dag_of(next),
        )
    }
}

/// Whether `next` may preempt `current` regardless of priority: the CPU is
/// entitled to `next`'s DAG and `current` belongs to another DAG-Fluid DAG.
/// A running task of any other scheduler keeps the normal priority rule.
// Used by `preemption_side::take_rotation`, which `no_preempt` compiles out.
#[cfg_attr(feature = "no_preempt", allow(dead_code))]
fn rotation_allowed(
    entitled: Option<u32>,
    current_dag: Option<u32>,
    next_dag: Option<u32>,
) -> bool {
    match (entitled, current_dag, next_dag) {
        (Some(entitled), Some(current), Some(next)) => next == entitled && current != entitled,
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::{super::entitlement_of, *};

    #[test]
    fn same_dag_or_nothing_new_needs_no_switch() {
        assert_eq!(switch_action(Some(1), Some(1), true), SwitchAction::Nothing);
        assert_eq!(switch_action(None, None, true), SwitchAction::Nothing);
        // Losing the entitlement with nothing to switch to.
        assert_eq!(switch_action(Some(1), None, true), SwitchAction::Nothing);
        assert_eq!(switch_action(Some(1), None, false), SwitchAction::Nothing);
    }

    #[test]
    fn new_dag_preempts_only_a_running_cpu() {
        assert_eq!(switch_action(Some(1), Some(2), true), SwitchAction::Preempt);
        assert_eq!(switch_action(None, Some(2), true), SwitchAction::Preempt);
        assert_eq!(
            switch_action(Some(1), Some(2), false),
            SwitchAction::PickUpWhenIdle
        );
        assert_eq!(
            switch_action(None, Some(2), false),
            SwitchAction::PickUpWhenIdle
        );
    }

    #[test]
    fn rotation_only_from_another_dag_to_the_entitled_one() {
        // Old DAG 1 -> entitled DAG 2.
        assert!(rotation_allowed(Some(2), Some(1), Some(2)));
        // The successor is not of the entitled DAG.
        assert!(!rotation_allowed(Some(2), Some(1), Some(3)));
        // The running task already belongs to the entitled DAG.
        assert!(!rotation_allowed(Some(2), Some(2), Some(2)));
        // The running task is not a DpWrap task: normal priority rule.
        assert!(!rotation_allowed(Some(2), None, Some(2)));
        // No entitlement at all.
        assert!(!rotation_allowed(None, Some(1), Some(2)));
        // The successor is not a DpWrap task.
        assert!(!rotation_allowed(Some(2), Some(1), None));
    }

    #[test]
    fn apply_due_marks_dirty_only_for_a_running_cpu() {
        // CPU ids far from the ones other tests could touch; no task runs
        // on them in unit tests, so this exercises the idle side.
        let cpu = NUM_MAX_CPU - 1;
        assert!(!apply_due(cpu, Some(7)));
        assert_eq!(entitlement_of(cpu), Some(7));
        assert!(!ENTITLEMENT_DIRTY[cpu].load(Ordering::Acquire));
        assert!(!apply_due(cpu, None));
        assert_eq!(entitlement_of(cpu), None);
        assert!(
            !apply_due(NUM_MAX_CPU, Some(1)),
            "out-of-range CPU is ignored"
        );
    }
}
