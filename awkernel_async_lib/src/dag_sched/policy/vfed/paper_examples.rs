//! Paper-conformance tests for V-Fed. Expected values are numbers printed
//! in the papers -- not values derived from this implementation;
//! assertions that are consequences of those numbers rather than printed
//! values are marked "derived".
//!
//! - [TPDS23] X. Jiang, H. Liang, N. Guan, Y. Tang, L. Qiao, Y. Wang,
//!   "Scheduling Parallel Real-Time Tasks on Virtual Processors", IEEE
//!   TPDS 34(1), 2023. Unmarked section/equation numbers refer to it.
//! - [RTSS21] X. Jiang, N. Guan, H. Liang, Y. Tang, L. Qiao, W. Yi,
//!   "Virtually-Federated Scheduling of Parallel Real-Time Tasks", RTSS
//!   2021. Same analysis, heavy tasks only, other numbers (see the module
//!   doc of `vfed.rs` for the mapping); it has no worked allocation
//!   example, so its tests use Figs. 5 and 7.
//!
//! The central [TPDS23] fixture is the Sec. 6.2 resource-allocation example
//! (p. 44): five tasks on `m = 7` processors, heavy tasks tau1..tau3 on
//! `M_h = 6` processors, tau4 on a passive-VP, tau5 partitioned onto p7.
//! Passive-VP `pi_k` is the one complementary to the active-VP on
//! processor `p_k`; in this implementation's placeholder slot ids that is
//! slot `k - 1`.

use super::*;

/// Sec. 6.2: tau1..tau5 as `(C, L, T, D)`.
fn tau1() -> DagMetrics {
    DagMetrics::from_static(12, 7, 10, 8)
}
fn tau2() -> DagMetrics {
    DagMetrics::from_static(10, 4, 10, 8)
}
fn tau3() -> DagMetrics {
    DagMetrics::from_static(10, 2, 9, 9)
}
fn tau4() -> DagMetrics {
    DagMetrics::from_static(6, 2, 10, 10)
}
fn tau5() -> DagMetrics {
    DagMetrics::from_static(5, 3, 10, 10)
}

/// A passive-VP complementary to an active-VP with budget `theta` of a
/// task with period `t` and deadline `d` (Lemma 1's parameters).
fn pvp(cpu: usize, theta: u64, t: u64, d: u64) -> PassiveVp {
    PassiveVp {
        cpu,
        active_budget: theta,
        owner_period: t,
        owner_deadline: d,
        group: None,
    }
}

// ---------------------------------------------------------------------
// Sec. 2 / Sec. 3 / Sec. 6.1: classification, eq. (1), active-VP budgets
// ---------------------------------------------------------------------

/// Fig. 1 (p. 34): `C = 11, L = 7, D = 8, T = 10`. Eq. (1):
/// `m = ceil((11 - 7) / (8 - 7)) = 4` (derived); density 11/8 > 1 so heavy.
#[test]
fn fig1_task_is_heavy_with_eq1_core_count() {
    let fig1 = DagMetrics::from_static(11, 7, 10, 8);
    assert_eq!(classify(&fig1), Ok(TaskClass::Heavy { required_cores: 4 }));
}

/// Sec. 2 (p. 34): "A DAG task is called a heavy task if its density is
/// strictly larger than 1, and a light task otherwise" -- density `C/D`.
/// Sec. 6.2: "tau1, tau2 and tau3 are heavy tasks. tau4 and tau5 are light
/// tasks."
#[test]
fn sec6_2_heavy_light_classification() {
    assert!(matches!(classify(&tau1()), Ok(TaskClass::Heavy { .. })));
    assert!(matches!(classify(&tau2()), Ok(TaskClass::Heavy { .. })));
    assert!(matches!(classify(&tau3()), Ok(TaskClass::Heavy { .. })));
    assert_eq!(classify(&tau4()), Ok(TaskClass::Light));
    assert_eq!(classify(&tau5()), Ok(TaskClass::Light));
    // Boundary: density exactly 1 is light (derived from the definition).
    assert_eq!(
        classify(&DagMetrics::from_static(10, 2, 10, 10)),
        Ok(TaskClass::Light)
    );
}

/// Sec. 6.2: "Since (C1 - L1) / (D1 - L1) = 5, we construct active-VP
/// group Theta1 = {8, 1, 1, 1, 1}".
#[test]
fn sec6_2_tau1_active_vp_group() {
    let t = tau1();
    assert_eq!(t.min_dedicated_cores(), Some(5));
    assert_eq!(
        full_active_vp_budgets(t.volume, t.critical_path, t.relative_deadline, 5),
        alloc::vec![8, 1, 1, 1, 1]
    );
}

/// Sec. 6.1 (p. 42): leading budget `D`, non-leading `D - L`, last one
/// `C - D - floor((C-D)/(D-L)) (D-L)`; the group has `m_i` VPs and its
/// budgets sum to `C` (Theorem 1, condition (8)). Fig. 1's task: C-D = 3,
/// D-L = 1 gives {8, 1, 1, 1} (derived).
#[test]
fn sec6_1_budget_rule_and_theorem1_conditions() {
    let (c, l, d) = (11, 7, 8);
    let budgets = full_active_vp_budgets(c, l, d, 4);
    assert_eq!(budgets, alloc::vec![8, 1, 1, 1]);
    assert_eq!(budgets.iter().sum::<u64>(), c); // (8)
    assert!(budgets[0] <= d); // (9)
    assert!(budgets[1..].iter().all(|&b| b <= d - l)); // (10)
}

/// Sec. 6.2: tau2 gets the single remaining processor with leading budget
/// `Theta2 = {8}` (Algorithm 1 lines 11-12, partial group).
#[test]
fn sec6_2_tau2_partial_group() {
    let t = tau2();
    assert_eq!(
        partial_active_vp_budgets(t.critical_path, t.relative_deadline, 1),
        alloc::vec![8]
    );
}

// ---------------------------------------------------------------------
// Lemma 1 (SBF), condition (18), Theorems 2 and 4 on the Sec. 6.2 numbers
// ---------------------------------------------------------------------

/// Sec. 6.2: `sbf_pi2(8) = 6`, `sbf_pi3(9) = 7`, `sbf_pi4(9) = 7`,
/// `sbf_pi5(10) - L4 = 6` (so `sbf_pi5(10) = 8`), and the passive-VP left
/// on p6 has `sbf(10) = 2`.
#[test]
fn lemma1_sbf_values_from_sec6_2() {
    let (t1, d1) = (10, 8); // tau1
    let pi2 = pvp(1, 1, t1, d1);
    let pi3 = pvp(2, 1, t1, d1);
    let pi4 = pvp(3, 1, t1, d1);
    let pi5 = pvp(4, 1, t1, d1);
    let pi6 = pvp(5, 8, 10, 8); // tau2's leading active-VP, theta = 8
    assert_eq!(pi2.sbf(8), 6);
    assert_eq!(pi3.sbf(9), 7);
    assert_eq!(pi4.sbf(9), 7);
    assert_eq!(pi5.sbf(10) - 2, 6);
    assert_eq!(pi6.sbf(10), 2);
    // Derived: pi1 (tau1's leading active-VP, theta = 8) is never useful
    // to a later task -- sbf(8) = 0 for tau2, sbf(9) = 1 < L3 for tau3.
    let pi1 = pvp(0, 8, t1, d1);
    assert_eq!(pi1.sbf(8), 0);
    assert_eq!(pi1.sbf(9), 1);
}

/// Condition (18): `pi` is useful to `tau_j` only if `sbf(D_j) > L_j`.
/// Sec. 6.2: pi5 is picked for tau4; for tau5 "only one passive-VP is left
/// where sbf(10) = 2 < L5".
#[test]
fn condition18_usefulness_from_sec6_2() {
    let pi5 = pvp(4, 1, 10, 8);
    let pi6 = pvp(5, 8, 10, 8);
    assert!(passive_vp_is_useful(&pi5, tau4().critical_path, 10));
    assert!(!passive_vp_is_useful(&pi6, tau5().critical_path, 10));
}

/// Theorem 4, condition (13), Sec. 6.2: for tau2 with Theta2 = {8} and
/// Pi = {pi2}: "10 <= 8 + 2".
#[test]
fn theorem4_tau2_from_sec6_2() {
    let pi2 = pvp(1, 1, 10, 8);
    let t = tau2();
    assert!(PassiveTest::Theorem4 {
        active_budget_sum: 8
    }
    .holds(t.volume, t.critical_path, t.relative_deadline, &[&pi2]));
    // Derived: without pi2 the active budget alone (8) is short of C2 = 10.
    assert!(!PassiveTest::Theorem4 {
        active_budget_sum: 8
    }
    .holds(t.volume, t.critical_path, t.relative_deadline, &[]));
}

/// Theorem 2, condition (12), Sec. 6.2: for tau3, "Theorem 2 is not
/// satisfied, i.e., 8 > 5" with Pi = {pi3}, then "8 < 5 + 5" with
/// Pi = {pi3, pi4}. For tau4 with Pi = {pi5}: satisfied.
#[test]
fn theorem2_tau3_and_tau4_from_sec6_2() {
    let pi3 = pvp(2, 1, 10, 8);
    let pi4 = pvp(3, 1, 10, 8);
    let pi5 = pvp(4, 1, 10, 8);
    let t3 = tau3();
    assert!(!PassiveTest::Theorem2.holds(
        t3.volume,
        t3.critical_path,
        t3.relative_deadline,
        &[&pi3]
    ));
    assert!(PassiveTest::Theorem2.holds(
        t3.volume,
        t3.critical_path,
        t3.relative_deadline,
        &[&pi3, &pi4]
    ));
    let t4 = tau4();
    assert!(PassiveTest::Theorem2.holds(
        t4.volume,
        t4.critical_path,
        t4.relative_deadline,
        &[&pi5]
    ));
}

// ---------------------------------------------------------------------
// Algorithm 1 (AllocH) and Algorithm 2 (Allo_Both) end to end
// ---------------------------------------------------------------------

/// Algorithm 1/2 line 1 sorts heavy tasks by increasing `D - L`; Sec. 6.2:
/// "Since D1 - L1 < D2 - L2 < D3 - L3, we first allocate resource to tau1."
/// Algorithm 2 lines 3-6 / Sec. 6.2: "We omit the enumerating of M_H when
/// M_H <= 5, where Algorithm AllocH(M_h) returns failure."
#[test]
fn algorithm1_fails_below_mh6_and_succeeds_at_mh6() {
    let heavy = [tau1(), tau2(), tau3()];
    for mh in 1..=5 {
        assert!(try_alloc_heavy_within(&heavy, mh).is_none(), "M_h = {mh}");
    }
    assert!(try_alloc_heavy_within(&heavy, 6).is_some());
    // Derived: Algorithm 2 lines 3-6 therefore fail on m = 5.
    assert!(!is_batch_feasible(&heavy, 5, PackingStrategy::BestFit));
}

/// Algorithm 1 with `M_h = 6`, Sec. 6.2: tau1 on active-VPs p1..p5; tau2
/// on the active-VP on p6 plus pi2; tau3 on pi3 and pi4; pi1, pi5, pi6 are
/// left for light tasks ("Passive-VPs on pi5 and pi6 are not allocated";
/// pi1's sbf is useless to every later task).
#[test]
fn algorithm1_allocation_at_mh6_matches_sec6_2() {
    let heavy = [tau1(), tau2(), tau3()];
    let (plans, pool) = try_alloc_heavy_within(&heavy, 6).unwrap();

    assert_eq!(plans[0].cores_used, 5);
    assert_eq!(plans[0].active_budgets, alloc::vec![8, 1, 1, 1, 1]);
    assert!(plans[0].passive_indices.is_empty());

    // V = [pi1..pi5, pi6] when tau2 searches; pi2 is index 1.
    assert_eq!(plans[1].cores_used, 1);
    assert_eq!(plans[1].active_budgets, alloc::vec![8]);
    assert_eq!(plans[1].passive_indices, alloc::vec![1]);

    // V = [pi1, pi3, pi4, pi5, pi6] when tau3 searches; pi3, pi4 = 1, 2.
    assert_eq!(plans[2].cores_used, 0);
    assert!(plans[2].active_budgets.is_empty());
    assert_eq!(plans[2].passive_indices, alloc::vec![1, 2]);

    let left: Vec<usize> = pool.iter().map(|p| p.cpu).collect();
    assert_eq!(left, alloc::vec![0, 4, 5]); // pi1, pi5, pi6
}

/// Algorithm 2 on the whole Sec. 6.2 task set, `m = 7`: `M_h = 6`, light
/// tasks in decreasing `C/D` ("Since C4/D4 > C5/D5, we pick pi5 for
/// tau4"), tau4 on pi5, tau5 partitioned onto the one remaining processor
/// ("We partition the remaining task tau5 on processor pi7").
#[test]
fn algorithm2_whole_example_matches_sec6_2() {
    let set = [tau1(), tau2(), tau3(), tau4(), tau5()];
    let plan = plan_batch(&set, 7, PackingStrategy::BestFit).unwrap();

    assert_eq!(plan.mh, 6);
    let heavy_order: Vec<usize> = plan.heavy.iter().map(|(i, _)| *i).collect();
    assert_eq!(heavy_order, alloc::vec![0, 1, 2]);
    let light_order: Vec<usize> = plan.light.iter().map(|(i, _)| *i).collect();
    assert_eq!(light_order, alloc::vec![3, 4]); // tau4 before tau5

    // Leftover V after AllocH is [pi1, pi5, pi6]; pi5 is index 1.
    match &plan.outcomes[0] {
        LightOutcome::Passive(indices) => assert_eq!(indices, &alloc::vec![1]),
        LightOutcome::Partitioned(_) => panic!("tau4 must be served by pi5"),
    }
    match &plan.outcomes[1] {
        LightOutcome::Partitioned(bin) => assert_eq!(*bin, 0), // M_l = 1
        LightOutcome::Passive(_) => panic!("tau5 must be partitioned"),
    }
    assert_eq!(plan.bins.len(), 1);

    assert!(is_batch_feasible(&set, 7, PackingStrategy::BestFit));
}

/// Derived from Sec. 6.2: tau5 needs the 7th processor (M_l = m - 6), so
/// the same task set is not admitted on 6 processors.
#[test]
fn algorithm2_rejects_example_on_six_processors() {
    let set = [tau1(), tau2(), tau3(), tau4(), tau5()];
    assert!(!is_batch_feasible(&set, 6, PackingStrategy::BestFit));
}

/// The result must not depend on the input order: Algorithm 1 line 1 and
/// Algorithm 2 lines 1 and 8 sort the tasks themselves.
#[test]
fn algorithm2_is_independent_of_input_order() {
    let set = [tau5(), tau3(), tau4(), tau1(), tau2()];
    let plan = plan_batch(&set, 7, PackingStrategy::BestFit).unwrap();
    assert_eq!(plan.mh, 6);
    let heavy_order: Vec<usize> = plan.heavy.iter().map(|(i, _)| *i).collect();
    assert_eq!(heavy_order, alloc::vec![3, 4, 1]); // tau1, tau2, tau3
    let light_order: Vec<usize> = plan.light.iter().map(|(i, _)| *i).collect();
    assert_eq!(light_order, alloc::vec![2, 0]); // tau4, tau5
    assert!(!is_batch_feasible(&set, 6, PackingStrategy::BestFit));
}

/// The incremental [`plan_heavy`]/[`plan_light`] path (used by
/// `admit_one`, not the paper's batch algorithm) reproduces the same
/// Sec. 6.2 allocation when fed the tasks in Algorithm 1's order with the
/// paper's `M_h = 6` split: tau1 on 5 cores, tau2 on 1 core plus pi2, tau3
/// on pi3 and pi4, tau4 on pi5.
#[test]
fn incremental_plan_heavy_path_reproduces_sec6_2() {
    let (t1, t2, t3) = (tau1(), tau2(), tau3());
    let mut pool: Vec<PassiveVp> = Vec::new();

    let plan1 = plan_heavy(&t1, 5, &pool).unwrap();
    assert_eq!(plan1.cores_used, 5);
    assert_eq!(plan1.active_budgets, alloc::vec![8, 1, 1, 1, 1]);
    assert!(plan1.passive_indices.is_empty());
    pool.extend(
        plan1
            .active_budgets
            .iter()
            .enumerate()
            .map(|(k, &b)| pvp(k, b, t1.period, t1.relative_deadline)),
    ); // pi1..pi5

    let plan2 = plan_heavy(&t2, 1, &pool).unwrap();
    assert_eq!(plan2.cores_used, 1);
    assert_eq!(plan2.active_budgets, alloc::vec![8]);
    assert_eq!(plan2.passive_indices, alloc::vec![1]); // pi2
    pool.remove(1);
    pool.push(pvp(5, 8, t2.period, t2.relative_deadline)); // pi6

    // Pool: pi1, pi3, pi4, pi5, pi6.
    let plan3 = plan_heavy(&t3, 0, &pool).unwrap();
    assert_eq!(plan3.cores_used, 0);
    assert!(plan3.active_budgets.is_empty());
    assert_eq!(plan3.passive_indices, alloc::vec![1, 2]); // pi3, pi4
    pool.remove(2);
    pool.remove(1);

    // Pool: pi1, pi5, pi6; "we pick pi5 for tau4".
    assert_eq!(plan_light(&tau4(), &pool).unwrap(), alloc::vec![1]);
    pool.remove(1);
    // "Only one passive-VP is left where sbf(10) = 2 < L5" (pi1 is
    // equally useless, derived).
    assert!(plan_light(&tau5(), &pool).is_err());
}

// ---------------------------------------------------------------------
// [RTSS21]
// ---------------------------------------------------------------------

/// [RTSS21] Fig. 5 (p. 486): job `J_i` with `c(v1) = 2`, `c(v2) = 3`,
/// `c(v3) = 3` (so `C = 8`, `L = 5`), deadline marker at 7 and next
/// release at 8 (`D = 7`, `T = 8`), served by `Theta = {6, 2}`.
const RTSS_FIG5: (u64, u64, u64, u64) = (8, 5, 8, 7); // (C, L, T, D)

/// [RTSS21] Fig. 5's group satisfies Theorem 1, conditions (4)-(6)
/// (derived). Eq. (16) gives `m = ceil(3/2) = 2` VPs, and the allocation
/// heuristic (Sec. VII-A, "as-large-as-possible active-VPs") would build
/// `{7, 1}` rather than the figure's illustrative `{6, 2}` (derived).
#[test]
fn rtss21_fig5_group_satisfies_theorem1() {
    let (c, l, t, d) = RTSS_FIG5;
    let theta = [6u64, 2];
    assert_eq!(theta.iter().sum::<u64>(), c); // (4)
    assert!(theta[0] <= d); // (5)
    assert!(theta[1] <= d - l); // (6)

    let task = DagMetrics::from_static(c, l, t, d);
    assert_eq!(classify(&task), Ok(TaskClass::Heavy { required_cores: 2 }));
    assert_eq!(full_active_vp_budgets(c, l, d, 2), alloc::vec![7, 1]);
}

/// [RTSS21] Fig. 7 (p. 486): the sbf of the passive-VPs complementary to
/// `theta1 = 6` and `theta2 = 2` of Fig. 5, read off the plotted curves at
/// `t = 0..8`. This also pins the third case of eq. (7) to
/// `gamma = beta - theta` (the formula) rather than `T - theta` (the prose
/// under Lemma 3's proof): at `t = 7` on `theta2` the figure shows 3,
/// where `T - theta` would give 6.
#[test]
fn rtss21_fig7_sbf_curves() {
    let (_, _, t, d) = RTSS_FIG5;
    let on_theta1: Vec<u64> = (0..=8).map(|x| sbf(x, 6, t, d)).collect();
    let on_theta2: Vec<u64> = (0..=8).map(|x| sbf(x, 2, t, d)).collect();
    assert_eq!(on_theta1, alloc::vec![0, 0, 0, 0, 0, 0, 0, 1, 1]);
    assert_eq!(on_theta2, alloc::vec![0, 0, 0, 1, 1, 1, 2, 3, 4]);
}

/// [RTSS21] Lemma 6 = [TPDS23] Lemma 9: for a passive-VP complementary to
/// an active-VP with budget `theta = D_i - L_i`, `sbf(D_j) > L_j` implies
/// `D_i - L_i < D_j - L_j`. Checked exhaustively on a small grid of
/// constrained-deadline parameters (derived: a statement of the papers,
/// not a printed number).
#[test]
fn lemma9_rtss21_lemma6_necessary_condition() {
    for ti in 1..=12u64 {
        for di in 1..=ti {
            for li in 0..di {
                let theta = di - li;
                for dj in 1..=16u64 {
                    for lj in 0..=dj {
                        if sbf(dj, theta, ti, di) > lj {
                            assert!(
                                di - li < dj - lj,
                                "T_i={ti} D_i={di} L_i={li} D_j={dj} L_j={lj}"
                            );
                        }
                    }
                }
            }
        }
    }
}

/// [RTSS21] Algorithm 1 has no `M_h` search: it runs on all `M`
/// processors, i.e. it is [TPDS23] `AllocH(M)`. On the Sec. 6.2 heavy
/// tasks with `M = 7` it succeeds too, with tau2 on a full group
/// `{8, 2}` (eq. (16): `ceil(6/4) = 2`) and tau3 reaching lines 10-17 with
/// `|P| = 0`, served by pi2 and pi3 under condition (10) (derived). The
/// batch path, which searches `M_h = 1..=m`, therefore accepts every
/// heavy-only set [RTSS21] accepts.
#[test]
fn rtss21_algorithm1_is_alloch_on_all_processors() {
    let heavy = [tau1(), tau2(), tau3()];
    let (plans, pool) = try_alloc_heavy_within(&heavy, 7).unwrap();
    assert_eq!(plans[0].active_budgets, alloc::vec![8, 1, 1, 1, 1]);
    assert_eq!(plans[1].active_budgets, alloc::vec![8, 2]);
    assert!(plans[1].passive_indices.is_empty());
    assert_eq!(plans[2].cores_used, 0);
    assert_eq!(plans[2].passive_indices, alloc::vec![1, 2]); // pi2, pi3
    let left: Vec<usize> = pool.iter().map(|p| p.cpu).collect();
    assert_eq!(left, alloc::vec![0, 3, 4, 5, 6]);
    assert!(is_batch_feasible(&heavy, 7, PackingStrategy::BestFit));
}
