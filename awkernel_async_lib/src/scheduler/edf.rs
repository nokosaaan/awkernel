//! Earliest Deadline First, ordered by the real job deadline.

use super::{
    clustered::{ClusteredOrdering, ClusteredScheduler},
    SchedulerType,
};
use crate::{
    dag::{calculate_and_update_dag_deadline, get_node_priority},
    task::TaskInfo,
};
use awkernel_lib::cpu::CpuSet;

pub(super) struct Edf;

pub(super) static SCHEDULER: ClusteredScheduler<Edf> = ClusteredScheduler::new();

impl ClusteredOrdering for Edf {
    const KIND: SchedulerType = SchedulerType::EDF(0, CpuSet::empty());

    fn ready_priority(info: &TaskInfo, wake_time: u64) -> (u64, u64) {
        let SchedulerType::EDF(relative_deadline, _) = info.scheduler_type else {
            unreachable!("non-EDF task routed to EDF");
        };
        let deadline = match &info.dag_info {
            Some(dag_info) => calculate_and_update_dag_deadline(dag_info, wake_time, info.state),
            None => wake_time + relative_deadline,
        };
        (deadline, deadline)
    }

    /// `u64::MAX - `[`get_node_priority`] so that a higher node priority (see
    /// [`crate::dag::Dag::set_node_priority`]) packs to a smaller key and
    /// dequeues first among nodes sharing the same `absolute_deadline` — the
    /// same inversion `MAX_TASK_PRIORITY - absolute_deadline` uses to turn a
    /// larger-is-more-urgent value into a smaller-sorts-first one. Regular
    /// (non-DAG) tasks are treated as node priority `0`.
    fn tie_break(info: &TaskInfo) -> u64 {
        let node_priority = match &info.dag_info {
            Some(dag_info) => get_node_priority(dag_info.dag_id, dag_info.node_id),
            None => 0,
        };
        u64::MAX - node_priority
    }
}
