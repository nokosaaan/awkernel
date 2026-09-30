//! Paper-conformance tests for DAG-Fluid's admission algorithm: expected
//! values are numbers printed in F. Guan, L. Peng, J. Qiao, "A Fluid
//! Scheduling Algorithm for DAG Tasks With Constrained or Arbitrary
//! Deadlines", IEEE TC 71(8), 2022. Assertions that are consequences of
//! those numbers rather than printed values are marked "derived".
//!
//! The fixture is Fig. 1 (p. 1864): `C_i = 120` (`c_1..c_5 = 10, 10, 20,
//! 50, 30`), `L_i = 60`, `T_i = 90`, `D_i = 80`. Edges as read from
//! Fig. 1a and confirmed by the infinite-processor diagram of Fig. 1b
//! (tau_i,3 and tau_i,5 start at 20, i.e. after tau_i,2; tau_i,4 starts at
//! 10): 1->2, 1->3, 1->4, 2->3, 2->5. Node `tau_i,k` is graph index
//! `k - 1`. The algorithm tests start from the segment list the paper
//! prints in Fig. 1c, so they do not depend on the decomposition.

use super::*;

const C: u64 = 120;
const L: u64 = 60;
const T: u64 = 90;
const D: u64 = 80;

fn seg(duration: u64, concurrency: u32) -> Segment {
    Segment {
        duration,
        concurrency,
    }
}

/// Fig. 1a.
fn fig1a() -> DagGraph {
    DagGraph::new(
        alloc::vec![10, 10, 20, 50, 30],
        &[(0, 1), (0, 2), (0, 3), (1, 2), (1, 4)],
    )
    .unwrap()
}

/// Fig. 1c.
fn fig1c() -> Vec<Segment> {
    alloc::vec![seg(10, 1), seg(10, 2), seg(20, 3), seg(10, 2), seg(10, 1)]
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-9
}

/// Fig. 1a: `C_i = 120`; Sec. 4.1 / Fig. 1c: five segments with
/// `(l_i,j, m_i,j)` = (10, 1), (10, 2), (20, 3), (10, 2), (10, 1), and
/// `L_i = Σ l_i,j = 60`.
#[test]
fn sec4_1_fig1c_segment_decomposition() {
    let g = fig1a();
    let segments = decompose_segments(&g);
    assert_eq!(segments, fig1c());
    assert_eq!(segments.iter().map(|s| s.duration).sum::<u64>(), L);
    // Derived: every unit of work lies in exactly one segment.
    assert_eq!(
        segments
            .iter()
            .map(|s| s.duration * s.concurrency as u64)
            .sum::<u64>(),
        C
    );
    assert_eq!((0..g.len()).map(|v| g.wcet(v)).sum::<u64>(), C);
}

/// Derived from Fig. 1b: the nodes finishing exactly at each segment's end
/// (the runtime's completion gates): tau_i,1 at 10, tau_i,2 at 20,
/// tau_i,3 at 40, tau_i,5 at 50, tau_i,4 at 60.
#[test]
fn fig1b_segment_completion_gates() {
    assert_eq!(
        segment_completion_gates(&fig1a()),
        alloc::vec![
            alloc::vec![0],
            alloc::vec![1],
            alloc::vec![2],
            alloc::vec![4],
            alloc::vec![3]
        ]
    );
}

/// Table 1, row 1: `T_i/D_i = 90/80 in [1, inf)` gives `D*_i = D_i = 80`
/// (Fig. 1d caption: `D*_i = 80`).
#[test]
fn table1_row1_for_fig1() {
    assert_eq!(virtual_deadline(T, D, L), 80);
}

/// Algorithm 1 lines 1-15 / Fig. 1d caption: `D^H_i = D*_i - d_i,1 -
/// d_i,5 = 60`, `C^H_i = C_i - l_i,1 m_i,1 - l_i,5 m_i,5 = 100`.
#[test]
fn algorithm1_heavy_capacity_and_deadline_for_fig1() {
    let segments = fig1c();
    let (c_h, d_h) = heavy_capacity_and_deadline(&segments, C, 80).unwrap();
    assert!(close(c_h, 100.0));
    assert!(close(d_h, 60.0));
}

/// Algorithm 1 lines 16-21, Definition 4.1 and Fig. 1d: light segments
/// sigma_i,1 and sigma_i,5 get `d = l`; `d_i,j` = 10, 12, 36, 12, 10;
/// `theta_i,j` = 1, 5/6, 5/9, 5/6, 1. Eq. (3): `sum d_i,j = D*_i`.
/// Sec. 4.2: release offsets `r_i,j` at 0, 10, 22, 58, 70 (Fig. 1d).
#[test]
fn algorithm1_segment_deadlines_rates_and_offsets_for_fig1() {
    let segments = fig1c();
    let schedule = assign_segment_deadlines(&segments, C, 80).unwrap();

    let d: Vec<f64> = schedule.iter().map(|s| s.relative_deadline).collect();
    let expected_d = [10.0, 12.0, 36.0, 12.0, 10.0];
    assert!(d.iter().zip(expected_d).all(|(a, b)| close(*a, b)), "{d:?}");

    let theta: Vec<f64> = schedule.iter().map(|s| s.rate).collect();
    let expected_theta = [1.0, 5.0 / 6.0, 5.0 / 9.0, 5.0 / 6.0, 1.0];
    assert!(
        theta.iter().zip(expected_theta).all(|(a, b)| close(*a, b)),
        "{theta:?}"
    );

    assert!(close(d.iter().sum::<f64>(), 80.0)); // Eq. (3)

    let r = segment_release_offsets(&schedule);
    let expected_r = [0.0, 10.0, 22.0, 58.0, 70.0];
    assert!(r.iter().zip(expected_r).all(|(a, b)| close(*a, b)), "{r:?}");
}

/// Lemma 4.2 / Lemma 5.9: the processor usage of tau_i is
/// `ceil(D*_i/T_i) * max(theta_i,j m_i,j) = C^H_i / D^H_i = 100/60`
/// (derived from the Fig. 1d numbers); Algorithm 2 line 12 adds exactly
/// this to `delta`.
#[test]
fn lemma4_2_and_algorithm2_contribution_for_fig1() {
    let segments = fig1c();
    let schedule = assign_segment_deadlines(&segments, C, 80).unwrap();
    let peak = schedule
        .iter()
        .map(|s| s.rate * s.concurrency as f64)
        .fold(0.0f64, f64::max);
    assert!(close(peak, 100.0 / 60.0));

    let required = required_capacity(C, T, L, D, &segments).unwrap();
    assert!(close(required, 100.0 / 60.0));

    // Derived, Algorithm 2 lines 14-18: accepted on 2 processors, not on 1.
    let entry = (C, T, L, D, segments.as_slice());
    assert!(is_batch_feasible(&[entry], 2));
    assert!(!is_batch_feasible(&[entry], 1));
}

/// Lemma 5.2: a task with `C_i > D*_i` always has at least one heavy
/// segment -- for Fig. 1 the heavy set is sigma_i,2..sigma_i,4 (Fig. 1d),
/// i.e. every segment with `m_i,j > 1` is below rate 1.
#[test]
fn lemma5_2_fig1_has_heavy_segments() {
    let segments = fig1c();
    let schedule = assign_segment_deadlines(&segments, C, 80).unwrap();
    let heavy: Vec<bool> = schedule.iter().map(|s| s.rate < 1.0).collect();
    assert_eq!(heavy, alloc::vec![false, true, true, true, false]);
}
