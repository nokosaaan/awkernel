//! Paper-conformance tests for the three federated-scheduling variants.
//! Expected values are numbers printed in the papers; assertions that are
//! consequences of those numbers rather than printed values are marked
//! "derived".
//!
//! - [Li14] J. Li, J.-J. Chen, K. Agrawal, C. Lu, C. Gill, A. Saifullah,
//!   "Analysis of Federated and Global Scheduling for Parallel Real-Time
//!   Tasks", ECRTS 2014.
//! - [DATE15] S. Baruah, "The federated scheduling of constrained-deadline
//!   sporadic DAG task systems", DATE 2015.
//! - [IPDPS15] S. Baruah, "Federated scheduling of sporadic DAG task
//!   systems", IPDPS 2015.

use super::*;

fn chain_plus_parallel(chain: &[u64], parallel: &[u64]) -> DagGraph {
    let mut wcet = chain.to_vec();
    wcet.extend_from_slice(parallel);
    let edges: Vec<(usize, usize)> = (1..chain.len()).map(|i| (i - 1, i)).collect();
    DagGraph::new(wcet, &edges).unwrap()
}

// ---------------------------------------------------------------------
// [Li14] implicit deadlines
// ---------------------------------------------------------------------

/// [Li14] Fig. 1 (p. 6): a high-utilization task with `L = 12, C = 20,
/// T = D = 16, u = 1.25`. Eq. (3): `n = ceil((20-12)/(16-12)) = 2`
/// (derived). Li's `MINPROCS` uses only `C` and `L`, so a graph with the
/// same `C` and `L` (a 2-5-2-3 chain plus two independent 4s) stands in
/// for the figure's DAG.
#[test]
fn li14_fig1_core_count_eq3() {
    let g = chain_plus_parallel(&[2, 5, 2, 3], &[4, 4]);
    let task = FedTask {
        metrics: DagMetrics::from_static(20, 12, 16, 16),
        graph: &g,
    };
    assert!(is_high_density(&task.metrics)); // u = 1.25 >= 1
    let n = min_procs(FederatedVariant::LiImplicit, &task, 16).unwrap();
    assert_eq!(n, 2);

    let c = &task.metrics;
    // Derived, Theorem 2: n (D - L) + L >= C  (2*4 + 12 = 20).
    assert!(u64::from(n) * (c.relative_deadline - c.critical_path) + c.critical_path >= c.volume);
    // Derived, Lemma 4: n < 2u  (2 < 2.5).
    assert!(f64::from(n) < 2.0 * c.volume as f64 / c.period as f64);
}

/// [Li14] Sec. III-A: "The federated scheduling algorithm admits the task
/// set tau, if n_low is non-negative" -- the Fig. 1 task alone needs its
/// 2 dedicated cores.
#[test]
fn li14_fig1_admission_needs_two_cores() {
    let g = chain_plus_parallel(&[2, 5, 2, 3], &[4, 4]);
    let tasks = [FedTask {
        metrics: DagMetrics::from_static(20, 12, 16, 16),
        graph: &g,
    }];
    let plan = plan_batch(&tasks, 2, FederatedVariant::LiImplicit).unwrap();
    assert_eq!(plan.plans, alloc::vec![FedPlan::Heavy { cores: 2 }]);
    assert_eq!(plan.heavy_cores, 2);
    assert!(plan_batch(&tasks, 1, FederatedVariant::LiImplicit).is_err());
}

/// [Li14] Sec. IV: a task with `C = m`, `L = 1`, `D = T = 1` needs speed
/// `> 2 - 1/m`; on unit-speed processors no core count suffices, so
/// Eq. (3) (division by `D - L = 0`) must yield "infeasible".
#[test]
fn li14_sec4_lower_bound_task_is_infeasible_at_unit_speed() {
    let m = 4u64;
    let g = chain_plus_parallel(&[1], &[1, 1, 1]);
    let task = FedTask {
        metrics: DagMetrics::from_static(m, 1, 1, 1),
        graph: &g,
    };
    assert_eq!(min_procs(FederatedVariant::LiImplicit, &task, 64), None);
}

// ---------------------------------------------------------------------
// [DATE15] constrained deadlines
// ---------------------------------------------------------------------

/// [DATE15] Fig. 1: WCETs 1 and 2 feeding a 2 that fans out to 1, 1 and 2
/// (the text says "five vertices", the figure and `vol = 9` have six).
fn date15_fig1_graph() -> DagGraph {
    DagGraph::new(
        alloc::vec![1, 2, 2, 1, 1, 2],
        &[(0, 2), (1, 2), (2, 3), (2, 4), (2, 5)],
    )
    .unwrap()
}

/// [DATE15] Example 1: `len1 = 6`, `vol1 = 9`, `D1 = 16`, `T1 = 20`,
/// `delta1 = 9/16`, `u1 = 9/20`; "since delta1 < 1, task tau1 is a
/// low-density task".
#[test]
fn date15_example1_is_low_density() {
    let g = date15_fig1_graph();
    // len and vol of the graph itself.
    assert_eq!(list_schedule_makespan(&g, 1), Some(9)); // vol on 1 processor
    assert_eq!(list_schedule_makespan(&g, g.len() as u16), Some(6)); // len

    // Derived: on 2 processors, {v0, v1} -> v2 at 2..4 -> {v3, v4} at
    // 4..5 -> v5 at 5..7 (list scheduling in vertex order).
    assert_eq!(list_schedule_makespan(&g, 2), Some(7));
    assert_eq!(list_schedule_makespan(&g, 3), Some(6));
    let tau1 = DagMetrics::from_static(9, 6, 20, 16);
    assert_eq!(density_window(&tau1), 16);
    assert!(!is_high_density(&tau1));

    // Derived: a lone low-density task is partitioned onto one processor.
    let tasks = [FedTask {
        metrics: tau1,
        graph: &g,
    }];
    let plan = plan_batch(&tasks, 1, FederatedVariant::BaruahConstrained).unwrap();
    assert_eq!(plan.plans, alloc::vec![FedPlan::Light { processor: 0 }]);
    assert_eq!(plan.heavy_cores, 0);
}

/// [DATE15] Eq. (1) and Fig. 4 line 3 on Example 1's task
/// `(vol, D, T) = (9, 16, 20)`: `DBF*(tau1, 16) = 9`, so a second task
/// with `D = 16` fits on the same processor iff `vol <= 16 - 9 = 7`
/// (derived). `DBF*(tau1, 36) = 9 + (9/20)(36 - 16) = 18`, so a task with
/// `D = 36` fits iff `vol <= 36 - 18 = 18` (derived).
#[test]
fn date15_eq1_dbf_star_and_fig4_condition() {
    let tau1 = partition::SeqTask {
        volume: 9,
        deadline: 16,
        period: 20,
    };
    let fits = |other: partition::SeqTask| {
        partition::partition(
            &[tau1, other],
            1,
            Condition::Demand,
            PackingStrategy::FirstFit,
        )
        .is_some()
    };
    let t = |volume, deadline, period| partition::SeqTask {
        volume,
        deadline,
        period,
    };
    assert!(fits(t(7, 16, 20)));
    assert!(!fits(t(8, 16, 20)));
    assert!(fits(t(18, 36, 100)));
    assert!(!fits(t(19, 36, 100)));
}

/// [DATE15] Example 2 (and [IPDPS15] Example 1): `n` tasks, each a single
/// vertex with WCET 1, `D = 1`, `T = n`. `Usum = 1`, yet the system needs
/// `n` processors. Each task has density 1 (high-density), so FEDCONS
/// gives each `MINPROCS = ceil(delta) = 1` dedicated processor (derived).
#[test]
fn date15_example2_needs_n_processors() {
    let n = 4usize;
    let g = DagGraph::new(alloc::vec![1], &[]).unwrap();
    let metrics = DagMetrics::from_static(1, 1, n as u64, 1);
    let tasks: Vec<FedTask<'_>> = (0..n).map(|_| FedTask { metrics, graph: &g }).collect();
    let usum: f64 = tasks
        .iter()
        .map(|t| t.metrics.volume as f64 / t.metrics.period as f64)
        .sum();
    assert!((usum - 1.0).abs() < 1e-12);

    let plan = plan_batch(&tasks, n as u16, FederatedVariant::BaruahConstrained).unwrap();
    assert!(plan.plans.iter().all(|p| *p == FedPlan::Heavy { cores: 1 }));
    assert_eq!(plan.heavy_cores, n as u16);
    assert!(plan_batch(&tasks, n as u16 - 1, FederatedVariant::BaruahConstrained).is_err());

    // The automatic variant choice for this set is DATE 2015 (all D <= T).
    assert_eq!(
        FederatedVariant::for_task_set(tasks.iter().map(|t| &t.metrics)),
        FederatedVariant::BaruahConstrained
    );
    assert!(is_batch_feasible(&tasks, n as u16));
    assert!(!is_batch_feasible(&tasks, n as u16 - 1));
}

// ---------------------------------------------------------------------
// [IPDPS15] arbitrary deadlines
// ---------------------------------------------------------------------

/// [IPDPS15] Example 1 is the same system as [DATE15] Example 2; FEDERATED
/// (which also covers constrained deadlines) must reach the same verdict.
#[test]
fn ipdps15_example1_needs_n_processors() {
    let n = 4usize;
    let g = DagGraph::new(alloc::vec![1], &[]).unwrap();
    let metrics = DagMetrics::from_static(1, 1, n as u64, 1);
    let tasks: Vec<FedTask<'_>> = (0..n).map(|_| FedTask { metrics, graph: &g }).collect();
    assert!(plan_batch(&tasks, n as u16, FederatedVariant::BaruahArbitrary).is_ok());
    assert!(plan_batch(&tasks, n as u16 - 1, FederatedVariant::BaruahArbitrary).is_err());
}
