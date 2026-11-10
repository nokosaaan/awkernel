//! Per-node bookkeeping for DAG-Fluid's per-node dispatch (the
//! `dagfluid_node` feature).
//!
//! The per-node plan names each slot by a node's task id, and the
//! dispatcher needs to know which job a node is working on (to tell an
//! early or late node from one in its own slot). This module records, for
//! every DAG node:
//! - the task id of its reactor (`dag::spawn_reactor` and friends register
//!   it right after spawning), and
//! - how many jobs it has finished (`Dag::finish_job` reports each one).
//!
//! A node processes its jobs in order, one at a time (one task per node),
//! so the job it is working on, or waiting for, is the number of jobs it
//! has finished.
//!
//! Lookups take a `Mutex`: call them from task context (the scheduler, the
//! planner), never from an interrupt handler.

use alloc::{collections::BTreeMap, sync::Arc};
use awkernel_lib::sync::mutex::{MCSNode, Mutex};
use core::sync::atomic::{AtomicU64, Ordering};

/// One DAG node.
#[derive(Debug)]
pub struct NodeEntry {
    pub dag_id: u32,
    pub node_id: u32,
    pub task_id: u32,
    done_jobs: AtomicU64,
}

impl NodeEntry {
    /// How many jobs this node has finished, i.e. the number of the job it
    /// is working on or waiting for next.
    pub fn done_jobs(&self) -> u64 {
        self.done_jobs.load(Ordering::Acquire)
    }
}

struct Tables {
    by_node: BTreeMap<(u32, u32), Arc<NodeEntry>>,
    by_task: BTreeMap<u32, Arc<NodeEntry>>,
}

static TABLES: Mutex<Tables> = Mutex::new(Tables {
    by_node: BTreeMap::new(),
    by_task: BTreeMap::new(),
});

/// Record that `task_id` is the reactor of `dag_id`'s node `node_id`.
/// Re-registering a node replaces the previous entry.
pub fn register_node_task(dag_id: u32, node_id: u32, task_id: u32) {
    let entry = Arc::new(NodeEntry {
        dag_id,
        node_id,
        task_id,
        done_jobs: AtomicU64::new(0),
    });
    let mut node = MCSNode::new();
    let mut tables = TABLES.lock(&mut node);
    if let Some(old) = tables.by_node.insert((dag_id, node_id), Arc::clone(&entry)) {
        tables.by_task.remove(&old.task_id);
    }
    tables.by_task.insert(task_id, entry);
}

/// The entry of `dag_id`'s node `node_id`, if registered.
pub fn node(dag_id: u32, node_id: u32) -> Option<Arc<NodeEntry>> {
    let mut node = MCSNode::new();
    let tables = TABLES.lock(&mut node);
    tables.by_node.get(&(dag_id, node_id)).cloned()
}

/// The entry of the node whose reactor is `task_id`, if registered.
pub fn node_of_task(task_id: u32) -> Option<Arc<NodeEntry>> {
    let mut node = MCSNode::new();
    let tables = TABLES.lock(&mut node);
    tables.by_task.get(&task_id).cloned()
}

/// Record that `dag_id`'s node `node_id` finished job `job` (called from
/// `Dag::finish_job`). Never moves the count backwards.
pub fn note_job_done(dag_id: u32, node_id: u32, job: u64) {
    if let Some(entry) = node(dag_id, node_id) {
        entry
            .done_jobs
            .fetch_max(job.saturating_add(1), Ordering::AcqRel);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // The tables are global: each test uses its own dag ids.

    #[test]
    fn lookup_by_node_and_by_task() {
        register_node_task(9001, 2, 40);
        let by_node = node(9001, 2).unwrap();
        assert_eq!(by_node.task_id, 40);
        let by_task = node_of_task(40).unwrap();
        assert_eq!((by_task.dag_id, by_task.node_id), (9001, 2));
        assert!(node(9001, 3).is_none());
    }

    #[test]
    fn done_jobs_counts_finished_jobs_and_never_goes_back() {
        register_node_task(9002, 0, 41);
        assert_eq!(node(9002, 0).unwrap().done_jobs(), 0);
        note_job_done(9002, 0, 0);
        assert_eq!(node(9002, 0).unwrap().done_jobs(), 1);
        note_job_done(9002, 0, 4);
        note_job_done(9002, 0, 2);
        assert_eq!(node(9002, 0).unwrap().done_jobs(), 5);
        // Unknown nodes are ignored.
        note_job_done(9002, 7, 0);
    }

    #[test]
    fn re_registering_replaces_the_task() {
        register_node_task(9003, 1, 50);
        register_node_task(9003, 1, 51);
        assert_eq!(node(9003, 1).unwrap().task_id, 51);
        assert!(node_of_task(50).is_none());
        assert_eq!(node_of_task(51).unwrap().node_id, 1);
    }
}
