//! Paper-conformance tests for SFS: expected values are numbers printed
//! in S. Lendve, K. Bletsas, P. F. Souto, "Resource-efficient scheduling
//! of parallel DAG tasks on identical multiprocessors", Journal of Systems
//! Architecture 176 (2026) 103775. Assertions that are consequences of
//! those numbers rather than printed values are marked "derived".
//!
//! Node `tau_1,k` of the paper is node `k - 1` here.

use super::*;

/// Fig. 1 (p. 4): 10 nodes, `D = 130`; WCETs and edges as drawn.
fn fig1() -> DagGraph {
    DagGraph::new(
        vec![10, 20, 30, 20, 20, 10, 20, 20, 20, 10],
        &[
            (0, 1),
            (0, 2),
            (0, 3),
            (1, 4),
            (1, 6),
            (2, 4),
            (2, 5),
            (3, 5),
            (4, 6),
            (4, 7),
            (5, 7),
            (5, 8),
            (6, 9),
            (7, 9),
            (8, 9),
        ],
    )
    .unwrap()
}

/// Fig. 3 (p. 6): chains `tau1 (1) -> tau3 (49)` and `tau2 (49) -> tau4 (1)`.
fn fig3() -> DagGraph {
    DagGraph::new(vec![1, 49, 49, 1], &[(0, 2), (1, 3)]).unwrap()
}

fn wcets(g: &DagGraph) -> Vec<u64> {
    (0..g.len()).map(|v| g.wcet(v)).collect()
}

/// Eq. (1) / Fig. 1: segments S1..S5 separated by the dashed lines;
/// "node tau_1,7 belongs to S_1,4 because the longest path to it from
/// tau_1,0 is 4 hops long".
#[test]
fn eq1_segments_of_fig1() {
    assert_eq!(
        segments(&fig1()).unwrap(),
        vec![vec![0], vec![1, 2, 3], vec![4, 5], vec![6, 7, 8], vec![9]]
    );
}

/// Algorithm 1 / Fig. 2 (p. 4): flattening Fig. 1 over 2 processors gives
/// makespan 105 and slack 25 against `D = 130`. Every interval of Fig. 2:
/// P1: t1 0-10, t2 10-30, t3 30-45, t5 45-65, t7 65-85, t8 85-95,
///     t10 95-105; P2: t3 10-25, t4 25-45, t6 45-55, t8 65-75, t9 75-95.
#[test]
fn algorithm1_fig2_flattened_schedule() {
    let g = fig1();
    let (len, ivs) = flatten(&segments(&g).unwrap(), &wcets(&g), 2);
    assert_eq!(len, 105);
    assert_eq!(130 - len, 25);

    let got: Vec<(usize, u64, u64)> = ivs.iter().map(|iv| (iv.node, iv.start, iv.end)).collect();
    let expected = vec![
        (0, 0, 10),   // S1: t1 (P1)
        (1, 10, 30),  // S2: t2 (P1)
        (2, 30, 45),  //     t3 head (P1)
        (2, 10, 25),  //     t3 wrapped tail (P2), runs first in time
        (3, 25, 45),  //     t4 (P2)
        (4, 45, 65),  // S3: t5 (P1)
        (5, 45, 55),  //     t6 (P2)
        (6, 65, 85),  // S4: t7 (P1)
        (7, 85, 95),  //     t8 head (P1)
        (7, 65, 75),  //     t8 wrapped tail (P2)
        (8, 75, 95),  //     t9 (P2)
        (9, 95, 105), // S5: t10 (P1)
    ];
    assert_eq!(got, expected);
}

/// Algorithm 2 (p. 5) on Fig. 1 with `D = 130`: initial `m' =
/// ceil(W / min(D, T)) = ceil(180/130) = 2` already fits (length 105), so
/// it returns `(2, 105)` (derived). Lines 2-6: the lower bound is the sum
/// of each segment's largest WCET, 10 + 30 + 20 + 20 + 10 = 90, so any
/// deadline below 90 is FAILURE (derived).
#[test]
fn algorithm2_on_fig1() {
    let g = fig1();
    let segs = segments(&g).unwrap();
    let rem = wcets(&g);
    assert_eq!(rem.iter().sum::<u64>(), 180);
    assert_eq!(feasibly_max_flatten(&segs, &rem, 130), Some((2, 105)));
    assert!(feasibly_max_flatten(&segs, &rem, 89).is_none());
    assert!(feasibly_max_flatten(&segs, &rem, 90).is_some());
}

/// Fig. 3 (p. 6): "Graham's bound for the makespan of the above DAG over 2
/// processors is 50 + (100 - 50)/2 = 75. The length of the corresponding
/// segmented-and-flattened schedule is 49 + 49 = 98."
#[test]
fn fig3_graham_bound_vs_flattening() {
    let g = fig3();
    let segs = segments(&g).unwrap();
    assert_eq!(critical_path(&g), 50);
    assert_eq!(flattened_length(&segs, &wcets(&g), 2), 98);
    // Eq. (2)/(3) with a deadline of 75: m' = ceil(50/25) = 2, M = 75.
    assert_eq!(graham_cluster(100, 50, 75), Some((2, 75)));
}

/// Fig. 3 with `D = T = 80` (derived): flattening (98) misses the
/// deadline, so Eqs. (2)/(3) give the Graham fall-back `(2, 75)`; the task
/// fits a 2-processor platform but not a single processor.
#[test]
fn fig3_deadline_80_needs_graham_fallback() {
    let g = fig3();
    let prep = Prepared {
        graph: &g,
        period: 80,
        deadline: 80,
        work: 100,
        length: critical_path(&g),
        segs: segments(&g).unwrap(),
    };
    assert!(feasibly_max_flatten(&prep.segs, &wcets(&g), 80).is_none());
    assert_eq!(cluster_size_requirements(&prep), Some((2, 75)));
    let task = SfsTask {
        graph: &g,
        period: 80,
        deadline: 80,
    };
    assert!(is_schedulable(&[task], 2));
    assert!(!is_schedulable(&[task], 1));
}

/// Sec. 4.2 (p. 6) tie rule: "In case of tie, flattening is preferred,
/// unless the resulting schedule length would exceed the upper-bound on
/// the makespan under the fall-back." With `D = 98`, both need 2
/// processors (flattening: 98, Graham: 75), so the fall-back wins
/// (derived).
#[test]
fn sec4_2_tie_rule_on_fig3() {
    let g = fig3();
    let prep = Prepared {
        graph: &g,
        period: 98,
        deadline: 98,
        work: 100,
        length: critical_path(&g),
        segs: segments(&g).unwrap(),
    };
    assert_eq!(
        feasibly_max_flatten(&prep.segs, &wcets(&g), 98),
        Some((2, 98))
    );
    assert_eq!(cluster_size_requirements(&prep), Some((2, 75)));
}

/// Fig. 5 (pp. 7, 10): after the first 60 time units of the Fig. 2
/// schedule on Q1, the rump DAG keeps 5 units of tau_1,5, 0 of tau_1,6 and
/// all of tau_1,7..tau_1,10; flattened on Q2's 3 processors it has length
/// 35, with relative deadline 130 - 60 = 70.
#[test]
fn fig5_rump_dag_after_split_at_60() {
    let g = fig1();
    let segs = segments(&g).unwrap();
    let mut rem = wcets(&g);
    let (_, ivs) = flatten(&segs, &rem, 2);
    for iv in &ivs {
        let done = iv.end.min(60).saturating_sub(iv.start);
        rem[iv.node] -= done.min(rem[iv.node]);
    }
    assert_eq!(rem, vec![0, 0, 0, 0, 5, 0, 20, 20, 20, 10]);
    let rump_len = flatten(&segs, &rem, 3).0;
    assert_eq!(rump_len, 35);
    assert!(rump_len <= 130 - 60);
}

/// Algorithm 4 on Fig. 1 alone (`D = T = 130`): it is heavy (`W = 180 >
/// 130`), flattening needs 2 processors while Graham needs
/// `ceil((180 - 90)/(130 - 90)) = 3`, so it gets a 2-processor cluster
/// with gang WCET 105; one processor is not enough (derived).
#[test]
fn algorithm4_fig1_alone() {
    let g = fig1();
    assert_eq!(critical_path(&g), 90);
    assert_eq!(graham_cluster(180, 90, 130), Some((3, 90 + 30)));
    let task = SfsTask {
        graph: &g,
        period: 130,
        deadline: 130,
    };
    let plan = plan(&[task], 2).unwrap();
    assert_eq!(plan.targets.len(), 1);
    assert_eq!(plan.targets[0].processors, 2);
    assert_eq!(plan.targets[0].pieces[0].wcet, 105);
    assert_eq!(plan.split_dags, 0);
    assert!(!is_schedulable(&[task], 1));
}
