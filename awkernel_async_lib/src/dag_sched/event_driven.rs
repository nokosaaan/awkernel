//! Completion tracking for DAG-Fluid's event-driven dispatch (方式B ③, the
//! `dagfluid_event` feature).
//!
//! In ③ a segment no longer starts at a fixed offset from its job's
//! release: segment `j + 1` of job `k` *arrives* when every node of segment
//! `j`'s completion gate (`SegmentTemplate::gate_nodes`, from
//! `dag_fluid::segment_completion_gates`) has finished job `k`, and must
//! finish by its arrival plus `d_j+1`. Segment 0 still arrives at the job's
//! release. This module records, per DAG and job, which nodes have finished
//! and when each segment completed, and wakes the planner whenever a
//! segment completes so it can lay out the new Deadline Partition at once.
//!
//! Node completions come from `Dag::finish_job` (task context, the node's
//! own CPU). Recording takes this module's lock and allocates; it never
//! runs in interrupt context.

use super::job_table;
use alloc::{
    collections::{BTreeMap, BTreeSet},
    vec::Vec,
};
use awkernel_lib::{
    sync::mutex::{MCSNode, Mutex},
    time::Instant,
};
use core::{
    future::Future,
    pin::Pin,
    sync::atomic::{AtomicBool, Ordering},
    task::{Context, Poll, Waker},
};

#[derive(Default)]
struct JobProgress {
    done_nodes: BTreeSet<u32>,
    /// When each segment completed, in segment order; shorter than the
    /// segment count until the last one completes.
    complete_at: Vec<Option<Instant>>,
}

/// `dag_id` -> job -> progress.
static PROGRESS: Mutex<BTreeMap<u32, BTreeMap<u64, JobProgress>>> = Mutex::new(BTreeMap::new());

/// Mark segments complete, in order, while their whole gate is in `done`.
/// A segment completes no earlier than the one before it. Returns whether
/// any segment newly completed.
pub fn advance_completion(
    gates: &[&[u32]],
    done: &BTreeSet<u32>,
    complete_at: &mut Vec<Option<Instant>>,
    now: Instant,
) -> bool {
    complete_at.resize(gates.len(), None);
    let mut changed = false;
    for (gate, at) in gates.iter().zip(complete_at.iter_mut()) {
        if at.is_some() {
            continue;
        }
        if !gate.iter().all(|node| done.contains(node)) {
            break;
        }
        *at = Some(now);
        changed = true;
    }
    changed
}

/// Record that `node_id` of `dag_id` finished job `job` (called from
/// `Dag::finish_job`). A no-op for a DAG that is not DAG-Fluid.
pub fn notify_node_done(dag_id: u32, node_id: u32, job: u64) {
    let Some(ring) = job_table::job_ring(dag_id) else {
        return;
    };
    let gates: Vec<&[u32]> = ring
        .template()
        .segments()
        .iter()
        .map(|segment| segment.gate_nodes.as_slice())
        .collect();
    let now = Instant::now();
    let changed = {
        let mut node = MCSNode::new();
        let mut progress = PROGRESS.lock(&mut node);
        let job_progress = progress.entry(dag_id).or_default().entry(job).or_default();
        job_progress.done_nodes.insert(node_id);
        advance_completion(
            &gates,
            &job_progress.done_nodes,
            &mut job_progress.complete_at,
            now,
        )
    };
    if changed {
        wake_planner();
    }
}

/// When each segment of `dag_id`'s job `job` completed so far (empty if
/// none has).
pub fn completed_segments(dag_id: u32, job: u64) -> Vec<Option<Instant>> {
    let mut node = MCSNode::new();
    let progress = PROGRESS.lock(&mut node);
    progress
        .get(&dag_id)
        .and_then(|jobs| jobs.get(&job))
        .map(|job| job.complete_at.clone())
        .unwrap_or_default()
}

/// Drop the records of `dag_id`'s jobs before `job` (they are done).
pub fn forget_jobs_before(dag_id: u32, job: u64) {
    let mut node = MCSNode::new();
    let mut progress = PROGRESS.lock(&mut node);
    if let Some(jobs) = progress.get_mut(&dag_id) {
        jobs.retain(|&k, _| k >= job);
    }
}

static NOTIFIED: AtomicBool = AtomicBool::new(false);
static PLANNER_WAKER: Mutex<Option<Waker>> = Mutex::new(None);

fn wake_planner() {
    NOTIFIED.store(true, Ordering::Release);
    let waker = {
        let mut node = MCSNode::new();
        let waker = PLANNER_WAKER.lock(&mut node).take();
        waker
    };
    if let Some(waker) = waker {
        waker.wake();
    }
}

/// Resolves once a segment has completed since the last time it resolved.
pub struct Notified;

impl Future for Notified {
    type Output = ();

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
        if NOTIFIED.swap(false, Ordering::AcqRel) {
            return Poll::Ready(());
        }
        {
            let mut node = MCSNode::new();
            *PLANNER_WAKER.lock(&mut node) = Some(cx.waker().clone());
        }
        // A completion between the first check and storing the waker would
        // otherwise go unnoticed until the next one.
        if NOTIFIED.swap(false, Ordering::AcqRel) {
            Poll::Ready(())
        } else {
            Poll::Pending
        }
    }
}

/// See [`Notified`].
pub fn notified() -> Notified {
    Notified
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::time::Duration;

    fn at_ms(v: u64) -> Instant {
        Instant::zero() + Duration::from_millis(v)
    }

    fn done(nodes: &[u32]) -> BTreeSet<u32> {
        nodes.iter().copied().collect()
    }

    #[test]
    fn segments_complete_in_order_when_their_gate_is_done() {
        let gates: [&[u32]; 3] = [&[0], &[1, 2], &[3]];
        let mut at = Vec::new();
        assert!(!advance_completion(&gates, &done(&[]), &mut at, at_ms(1)));
        assert_eq!(at, [None, None, None]);

        assert!(advance_completion(
            &gates,
            &done(&[0, 1]),
            &mut at,
            at_ms(5)
        ));
        assert_eq!(at, [Some(at_ms(5)), None, None]);

        assert!(advance_completion(
            &gates,
            &done(&[0, 1, 2]),
            &mut at,
            at_ms(9)
        ));
        assert_eq!(at, [Some(at_ms(5)), Some(at_ms(9)), None]);
    }

    #[test]
    fn a_later_gate_done_early_waits_for_the_earlier_segment() {
        // Node 3 (segment 2's gate) finishes before segment 1's gate.
        let gates: [&[u32]; 3] = [&[0], &[1], &[3]];
        let mut at = Vec::new();
        advance_completion(&gates, &done(&[0, 3]), &mut at, at_ms(4));
        assert_eq!(at, [Some(at_ms(4)), None, None]);
        advance_completion(&gates, &done(&[0, 1, 3]), &mut at, at_ms(7));
        assert_eq!(at, [Some(at_ms(4)), Some(at_ms(7)), Some(at_ms(7))]);
    }

    #[test]
    fn an_empty_gate_completes_with_the_segment_before_it() {
        let gates: [&[u32]; 2] = [&[0], &[]];
        let mut at = Vec::new();
        advance_completion(&gates, &done(&[0]), &mut at, at_ms(3));
        assert_eq!(at, [Some(at_ms(3)), Some(at_ms(3))]);
    }

    #[test]
    fn nothing_new_reports_no_change() {
        let gates: [&[u32]; 1] = [&[0]];
        let mut at = Vec::new();
        assert!(advance_completion(&gates, &done(&[0]), &mut at, at_ms(1)));
        assert!(!advance_completion(&gates, &done(&[0]), &mut at, at_ms(2)));
        assert_eq!(at, [Some(at_ms(1))]);
    }
}
