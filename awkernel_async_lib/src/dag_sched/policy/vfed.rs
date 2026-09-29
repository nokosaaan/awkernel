//! V-Fed ("virtually-federated") admission policy (Jiang, Guan, Liang, Tang,
//! Qiao, Wang — RTSS 2021 / TPDS 2023) for **constrained-deadline**
//! (`D <= T`) DAG tasks.
//!
//! Where classical Federated ([`super::federated`]) grants a heavy DAG
//! exclusive ownership of whole physical cores (wasting whatever capacity
//! the DAG doesn't use on them), V-Fed constructs, on each physical core, an
//! **active-VP** (a budget-limited virtual processor the owning DAG uses
//! with the highest priority) and a complementary **passive-VP** (the
//! leftover capacity, offered to some other task at low priority). A heavy
//! DAG still gets the same minimum core count as classical Federated
//! (`m = ceil((C-L)/(D-L))`, see [`crate::dag_sched::metrics::DagMetrics::min_dedicated_cores`]),
//! but when there are not enough free cores for a full active-VP group, the
//! shortfall can be topped up with *other* DAGs' leftover passive-VPs
//! instead of failing admission outright — and light DAGs (`C <= D`) can be
//! served entirely from passive-VPs, with no dedicated core at all.
//!
//! # Scope of this module (current state)
//!
//! The batch path ([`admit_batch`]/[`is_batch_feasible`], both built on
//! [`plan_batch`]) follows the TPDS 2023 version's Algorithm 2
//! (`Allo_Both`) and Algorithm 1 (`AllocH`) line by line:
//! - [`sbf`]: the supply bound function (Lemma 1).
//! - [`PassiveTest`]: Theorem 2 / condition (12) (passive-VPs only,
//!   `C <= L + Σ max{H_j - L, 0}`) and Theorem 4 / condition (13) (active
//!   plus passive, `C <= Σθ + Σ max{H_j - L, 0}`), kept as two distinct
//!   conditions -- an earlier revision folded Theorem 2 into Theorem 4 with
//!   `Σθ = 0`, silently dropping Theorem 2's `L_i` term.
//! - [`try_alloc_heavy_within`]: Algorithm 1 as written, including the
//!   partial-group branch (lines 10-17, run once, on whatever processors
//!   remain -- possibly none) and the `goto line 19` switch to Theorem 2
//!   for every heavy DAG after it.
//! - [`search_min_heavy_cores`]: Algorithm 2's search over `M_h`.
//! - [`plan_batch`]: Algorithm 2's light-task handling -- passive-VPs by
//!   Theorem 2 first, then `Partition(L, M_l)` as best-fit partitioned EDF
//!   with the DBF* test of the paper's reference \[15\] (see
//!   [`crate::dag_sched::partition`]).
//!
//! [`admit_one`] is an incremental, one-DAG-at-a-time variant kept for
//! callers that admit DAGs individually. It uses the same theorems but is
//! *not* the paper's algorithm (no global sort, no `M_h` search, and a
//! density-based partition test, since DBF* needs deadline-ordered
//! insertion); the kernel's V-Fed build uses [`admit_batch`].
//!
//! The scheduler mechanism itself now exists too —
//! [`crate::scheduler::active_vp`] (budget-gated active-VP dispatch,
//! leading-VP preference), [`crate::scheduler::passive_vp`] ("yield while
//! the owner is busy" passive-VP dispatch), and
//! [`crate::scheduler::mixed_vp`] (Theorem 4's combined role: a single task
//! dispatched on both its own active-VP cores and other DAGs' borrowed
//! passive-VP cores, whichever becomes eligible first) — and
//! [`VFedAssignment::into_scheduler_type`] connects an admission decision to
//! whichever of the three applies, plus the partitioned-EDF fallback for a
//! light DAG that fits no available passive-VP.

use alloc::vec::Vec;

use awkernel_lib::{
    cpu::CpuSet,
    sync::mutex::{MCSNode, Mutex},
};

use crate::{
    dag_sched::{
        metrics::DagMetrics,
        partition,
        resource::{self, ResourceError},
    },
    scheduler::SchedulerType,
};

/// Result of [`classify`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskClass {
    Light,
    Heavy { required_cores: u16 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VFedError {
    /// `relative_deadline < critical_path` (traversing the critical path
    /// alone already exceeds the deadline, so no core count can help), or
    /// `relative_deadline == critical_path` with exploitable parallelism
    /// (`volume > critical_path`, i.e. Heavy): see
    /// [`super::federated::FederatedError::Infeasible`]'s doc for why the
    /// implicit-deadline boundary `D == L` is only feasible when
    /// `volume == critical_path` (Light).
    Infeasible {
        critical_path: u64,
        relative_deadline: u64,
    },
    /// V-Fed's theorems all assume constrained deadlines (`D <= T`); see the
    /// module doc. A DAG with `D > T` must go through
    /// [`super::federated`] instead.
    ArbitraryDeadlineNotSupported { relative_deadline: u64, period: u64 },
    /// No combination of currently-available cores/passive-VPs/partitions
    /// can schedule this DAG (or, for [`admit_batch`], the whole task set).
    NoFeasibleAllocation,
    /// The shared core ledger could not satisfy a core request; see
    /// [`ResourceError`].
    Resource(ResourceError),
}

impl From<ResourceError> for VFedError {
    fn from(e: ResourceError) -> Self {
        VFedError::Resource(e)
    }
}

/// The supply bound function (Lemma 1): the minimum processing time a
/// passive-VP complementary to an active-VP with initial budget `budget`
/// (on a task with period `period` and relative deadline `deadline`) is
/// guaranteed to provide in any interval of length `delta`.
///
/// Preconditions (upheld by every [`PassiveVp`] this module constructs,
/// never by external input): `deadline <= period` (constrained deadline)
/// and `budget <= deadline` (an active-VP's budget is always `deadline` or
/// `deadline - critical_path`, both `<= deadline <= period`).
fn sbf(delta: u64, budget: u64, period: u64, deadline: u64) -> u64 {
    if delta < budget {
        return 0;
    }
    let x = delta - budget;
    let alpha = (x / period) * (period - budget);
    let beta = x % period;
    let gamma = if beta <= period - deadline {
        beta
    } else if beta <= period - deadline + budget {
        period - deadline
    } else {
        beta - budget
    };
    alpha + gamma
}

/// One core dedicated as part of a heavy DAG's active-VP group.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActiveVp {
    pub cpu: usize,
    /// Initial budget `θ_z`: replenished to this value at every job
    /// release, consumed 1:1 with execution, forcing the served task off
    /// this core once exhausted (see the module doc's "not yet
    /// implemented" list — that enforcement is mechanism-layer work).
    pub budget: u64,
}

/// A physical core's leftover capacity once its active-VP has been
/// accounted for. Characterized entirely by [`PassiveVpKind`] (in turn by
/// the owning active-VP group's own budgets/period/deadline, per [`sbf`]) —
/// not by which task the *active* side happens to serve, which is
/// irrelevant to what the *passive* side can guarantee.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PassiveVp {
    pub cpu: usize,
    kind: PassiveVpKind,
}

/// How a [`PassiveVp`]'s supply is characterized. `Independent` is TPDS2023
/// OURS1 (== RTSS2021's Theorem 2/4, mathematically: summing
/// `max(sbf_x(D) - L, 0)` over a set doesn't depend on order, so the naive
/// per-VP sum this variant produces IS the paper's `Π*` construction, not a
/// downgrade from it). `AlwaysFree`/`Shared` together are OURS2 (`Π'`,
/// TPDS2023 Section 4.2): when a heavy DAG's own maximum parallelism `Li` is
/// known and less than how many active-VPs it has (`mi`), at least
/// `mi - Li` of them are *structurally* never used by that DAG itself (its
/// workload can never exceed `Li`-way parallelism), so their complementary
/// passive-VPs are unconditionally available — strictly more than the usual
/// SBF pattern promises. See [`generate_passive_vps`] for the split that
/// produces this.
#[derive(Debug, Clone, PartialEq, Eq)]
enum PassiveVpKind {
    /// OURS1: this VP's own `(budget, period, deadline)`.
    Independent {
        active_budget: u64,
        owner_period: u64,
        owner_deadline: u64,
    },
    /// OURS2: one of a group's `mi - Li` structurally-always-idle slots —
    /// `sbf(Δ) = Δ` unconditionally.
    AlwaysFree,
    /// OURS2: one of a group's `Li` "shared" slots. Every sibling slot in
    /// the same group is identical by construction (same `budgets`/`li`), so
    /// each carries its own copy rather than the group being deduplicated —
    /// pools here are bounded by core count, never large enough for that to
    /// matter.
    Shared {
        /// Every active-VP budget in the *whole* group (not just this
        /// slot's own) — `Π'`'s formula sums over all of them.
        budgets: Vec<u64>,
        owner_period: u64,
        owner_deadline: u64,
        /// This group's `Li`: how many `Shared` slots divide up the
        /// aggregate leftover supply (`budgets.len()` is the group's `mi`).
        li: u16,
    },
}

impl PassiveVp {
    /// The minimum processing time this passive-VP is guaranteed to provide
    /// in any interval of length `delta` — [`PassiveVpKind::Independent`]
    /// via the plain [`sbf`] (Lemma 1); [`PassiveVpKind::AlwaysFree`]
    /// trivially (`delta`); [`PassiveVpKind::Shared`] via TPDS2023's `Π'`
    /// formula: `max((Σ sbf_z(Δ) - Δ·(mi - li)) / li, 0)`, `mi =
    /// budgets.len()` — identical for every sibling slot in the group.
    fn sbf(&self, delta: u64) -> u64 {
        match &self.kind {
            PassiveVpKind::Independent {
                active_budget,
                owner_period,
                owner_deadline,
            } => sbf(delta, *active_budget, *owner_period, *owner_deadline),
            PassiveVpKind::AlwaysFree => delta,
            PassiveVpKind::Shared {
                budgets,
                owner_period,
                owner_deadline,
                li,
            } => {
                if *li == 0 {
                    // Never constructed this way (see `generate_passive_vps`);
                    // defensive rather than a silent divide-by-zero.
                    return 0;
                }
                let raw: u64 = budgets
                    .iter()
                    .map(|&b| sbf(delta, b, *owner_period, *owner_deadline))
                    .sum();
                let mi_minus_li = (budgets.len() as u64).saturating_sub(*li as u64);
                raw.saturating_sub(delta.saturating_mul(mi_minus_li)) / (*li as u64)
            }
        }
    }
}

/// Turn one heavy DAG's active-VP budgets into the passive-VPs left over on
/// those same cores, applying OURS2's split (per [`PassiveVpKind`]'s doc)
/// whenever `max_parallelism` is known and satisfies the paper's own
/// precondition `Li < mi` (`mi = budgets.len()`, the *actual* count this
/// call received — a partial active-VP group works too: the "structurally
/// idle" argument only needs `mi_actual > Li`, not a full group). Falls back
/// to `mi` independent (OURS1) entries otherwise — including the sentinel
/// `u16::MAX` ("unknown") [`crate::dag_sched::metrics::DagMetrics`] sets by
/// default, so every existing caller that never supplies `max_parallelism`
/// is unaffected. One shared helper for the three sites that used to
/// duplicate this loop (`try_alloc_heavy_within`, `admit_heavy`,
/// `admit_batch`'s commit).
fn generate_passive_vps(
    budgets: &[u64],
    cpus: impl Iterator<Item = usize>,
    period: u64,
    deadline: u64,
    max_parallelism: u16,
) -> Vec<PassiveVp> {
    let mi = budgets.len() as u16;
    if max_parallelism < mi {
        let li = max_parallelism;
        let always_free = mi - li;
        cpus.enumerate()
            .map(|(i, cpu)| {
                let kind = if (i as u16) < always_free {
                    PassiveVpKind::AlwaysFree
                } else {
                    PassiveVpKind::Shared {
                        budgets: budgets.to_vec(),
                        owner_period: period,
                        owner_deadline: deadline,
                        li,
                    }
                };
                PassiveVp { cpu, kind }
            })
            .collect()
    } else {
        budgets
            .iter()
            .zip(cpus)
            .map(|(&active_budget, cpu)| PassiveVp {
                cpu,
                kind: PassiveVpKind::Independent {
                    active_budget,
                    owner_period: period,
                    owner_deadline: deadline,
                },
            })
            .collect()
    }
}

/// `density = C/D`: a constrained-deadline DAG is heavy iff its volume
/// exceeds its relative deadline (see the module doc: `min(D,T) = D` always
/// holds here, so this is the general density test specialized to this
/// module's scope).
const fn is_heavy(volume: u64, deadline: u64) -> bool {
    volume > deadline
}

/// Classify a DAG. Does not claim any cores or passive-VPs; see
/// [`plan_heavy`]/[`plan_light`]/[`admit_one`] for that.
pub fn classify(config: &DagMetrics) -> Result<TaskClass, VFedError> {
    if config.relative_deadline > config.period {
        return Err(VFedError::ArbitraryDeadlineNotSupported {
            relative_deadline: config.relative_deadline,
            period: config.period,
        });
    }
    if config.relative_deadline < config.critical_path {
        return Err(VFedError::Infeasible {
            critical_path: config.critical_path,
            relative_deadline: config.relative_deadline,
        });
    }

    if !is_heavy(config.volume, config.relative_deadline) {
        return Ok(TaskClass::Light);
    }

    let Some(required_cores) = config.min_dedicated_cores() else {
        return Err(VFedError::Infeasible {
            critical_path: config.critical_path,
            relative_deadline: config.relative_deadline,
        });
    };

    Ok(TaskClass::Heavy { required_cores })
}

/// Build the initial budgets for a *full* (`cores == m_i`) active-VP group:
/// `θ1 = D` for the leading VP, `D - L` for every non-leading VP except
/// possibly the last, whose budget absorbs the remainder so
/// `Σθz == volume` exactly (Theorem 1's condition (4)). When `C - D` is an
/// exact multiple of `D - L` every non-leading VP gets the full `D - L` and
/// there is no remainder VP at all.
fn full_active_vp_budgets(volume: u64, critical_path: u64, deadline: u64, cores: u16) -> Vec<u64> {
    let mut budgets = Vec::with_capacity(cores as usize);
    budgets.push(deadline); // leading: θ1 = D
    if cores == 1 {
        // Unreachable for an actually-heavy DAG (see `classify`: C > D
        // forces m_i >= 2), kept only so this function is total.
        return budgets;
    }
    let d_minus_l = deadline - critical_path;
    let c_minus_d = volume - deadline;
    let full_non_leading = c_minus_d / d_minus_l;
    let remainder = c_minus_d - full_non_leading * d_minus_l;
    for _ in 0..full_non_leading {
        budgets.push(d_minus_l);
    }
    if remainder > 0 {
        budgets.push(remainder);
    }
    budgets
}

/// Build the initial budgets for a *partial* (`cores < m_i`) active-VP
/// group: every VP — leading included — gets its maximum allowed budget
/// (`D` for the leading one, `D - L` for the rest). The total is
/// deliberately less than `volume`; the shortfall is made up from
/// passive-VPs (Theorem 4).
fn partial_active_vp_budgets(critical_path: u64, deadline: u64, cores: u16) -> Vec<u64> {
    let mut budgets = Vec::with_capacity(cores as usize);
    if cores == 0 {
        return budgets;
    }
    budgets.push(deadline);
    for _ in 1..cores {
        budgets.push(deadline - critical_path);
    }
    budgets
}

/// `Σ_{j=1}^{|Π|} max{H_j(D_i, Π) - L_i, 0}`: the passive-VP term shared by
/// Theorem 2 and Theorem 4. `H_j` of the hypothetical platform (`Π*` or
/// `Π'`) is exactly the multiset of the passive-VPs' own `sbf` values, so
/// the sum is order-independent.
fn passive_supply(critical_path: u64, deadline: u64, passives: &[&PassiveVp]) -> u64 {
    passives
        .iter()
        .map(|p| p.sbf(deadline).saturating_sub(critical_path))
        .sum()
}

/// Which schedulability condition a passive-VP search tests against.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PassiveTest {
    /// Theorem 2, condition (12) (task on a passive-VP group only):
    /// `C_i <= L_i + Σ max{H_j(D_i, Π) - L_i, 0}`.
    Theorem2,
    /// Theorem 4, condition (13) (task on `⟨Θ, Π⟩`), with
    /// `active_budget_sum = Σ_{θ_z ∈ Θ} θ_z`:
    /// `C_i <= Σθ_z + Σ max{H_j(D_i, Π) - L_i, 0}`. Algorithm 1 line 17
    /// applies this even when `Θ` ended up empty (`|P| = 0` at line 11).
    Theorem4 { active_budget_sum: u64 },
}

impl PassiveTest {
    fn holds(
        self,
        volume: u64,
        critical_path: u64,
        deadline: u64,
        passives: &[&PassiveVp],
    ) -> bool {
        let base = match self {
            PassiveTest::Theorem2 => critical_path,
            PassiveTest::Theorem4 { active_budget_sum } => active_budget_sum,
        };
        volume <= base.saturating_add(passive_supply(critical_path, deadline, passives))
    }
}

/// A passive-VP is only ever worth pulling for a task with relative
/// deadline `deadline` and critical path `critical_path` if it clears this
/// bar (paper's condition (17)/(18)); a fixed property of the VP and the
/// consumer, independent of what else is already assigned.
fn passive_vp_is_useful(vp: &PassiveVp, critical_path: u64, deadline: u64) -> bool {
    vp.sbf(deadline) > critical_path
}

/// Pure planning result for a heavy DAG: how many cores its active-VP group
/// should claim, their budgets (parallel to the cores once assigned), and
/// which entries of the passive-VP pool it needs on top (by index into the
/// slice `plan_heavy` was given).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HeavyPlan {
    pub cores_used: u16,
    pub active_budgets: Vec<u64>,
    pub passive_indices: Vec<usize>,
}

/// Plan a heavy DAG's admission against `cores_available` free cores and
/// the current passive-VP `pool`, without mutating anything -- the
/// single-task step of the incremental [`admit_one`] path. (The
/// paper-faithful batch path is [`try_alloc_heavy_within`], which follows
/// Algorithm 1's control flow literally instead.) A full active-VP group if
/// enough cores are free (Theorem 1); otherwise every free core plus as many
/// passive-VPs as needed (Theorem 4), or passive-VPs alone (Theorem 2) if no
/// core is free at all. Passive-VPs are pulled greedily in the pool's order,
/// the first useful one each time (the paper's own Sec. 6.2 choice).
pub fn plan_heavy(
    config: &DagMetrics,
    cores_available: u16,
    pool: &[PassiveVp],
) -> Result<HeavyPlan, VFedError> {
    let Some(required_cores) = config.min_dedicated_cores() else {
        return Err(VFedError::Infeasible {
            critical_path: config.critical_path,
            relative_deadline: config.relative_deadline,
        });
    };

    if cores_available >= required_cores {
        return Ok(HeavyPlan {
            cores_used: required_cores,
            active_budgets: full_active_vp_budgets(
                config.volume,
                config.critical_path,
                config.relative_deadline,
                required_cores,
            ),
            passive_indices: Vec::new(),
        });
    }

    if cores_available == 0 {
        return pull_passives_until_schedulable(config, PassiveTest::Theorem2, pool).map(
            |passive_indices| HeavyPlan {
                cores_used: 0,
                active_budgets: Vec::new(),
                passive_indices,
            },
        );
    }

    let active_budgets = partial_active_vp_budgets(
        config.critical_path,
        config.relative_deadline,
        cores_available,
    );
    let active_budget_sum: u64 = active_budgets.iter().sum();
    let passive_indices =
        pull_passives_until_schedulable(config, PassiveTest::Theorem4 { active_budget_sum }, pool)?;

    Ok(HeavyPlan {
        cores_used: cores_available,
        active_budgets,
        passive_indices,
    })
}

/// The paper's passive-VP `do ... while` loop (Algorithm 1 lines 14-17 /
/// 22-25, Algorithm 2 lines 12-15): repeatedly move the first passive-VP in
/// `pool` satisfying condition (18) into `Π` until `test` holds, failing if
/// no useful passive-VP is left. At least one passive-VP is always taken
/// before `test` is first evaluated, as in the paper's `do`-`while`.
fn pull_passives_until_schedulable(
    config: &DagMetrics,
    test: PassiveTest,
    pool: &[PassiveVp],
) -> Result<Vec<usize>, VFedError> {
    let (volume, critical_path, deadline) = (
        config.volume,
        config.critical_path,
        config.relative_deadline,
    );
    let mut used = Vec::new();
    let mut used_mask = alloc::vec![false; pool.len()];

    loop {
        let chosen = pool
            .iter()
            .enumerate()
            .find(|(i, vp)| !used_mask[*i] && passive_vp_is_useful(vp, critical_path, deadline));
        let Some((idx, _)) = chosen else {
            return Err(VFedError::NoFeasibleAllocation);
        };
        used_mask[idx] = true;
        used.push(idx);

        let chosen_refs: Vec<&PassiveVp> = used.iter().map(|&i| &pool[i]).collect();
        if test.holds(volume, critical_path, deadline, &chosen_refs) {
            return Ok(used);
        }
    }
}

/// Plan a DAG purely from passive-VPs (Theorem 2, condition (12)) -- the
/// light-task step of Algorithm 2 (lines 11-17). Returns
/// [`VFedError::NoFeasibleAllocation`] if the current pool's useful
/// entries can't make it schedulable; the caller then leaves it for the
/// partitioned-EDF step.
pub fn plan_light(config: &DagMetrics, pool: &[PassiveVp]) -> Result<Vec<usize>, VFedError> {
    pull_passives_until_schedulable(config, PassiveTest::Theorem2, pool)
}

/// The paper's Algorithm 1, `AllocH(M_h)`, transcribed line by line for the
/// heavy DAGs in `heavy_sorted` (already sorted ascending by `D - L`, as
/// line 1 does) on `mh` processors (line 2). Runs against a local passive-VP pool `V`
/// addressed by placeholder slot ids, not real cpu ids (only the caller
/// that commits a plan substitutes real cores). Returns `None` where the
/// paper returns `false`.
///
/// Every plan it returns must be replayed as "append this task's own
/// complementary passive-VPs to `V`, *then* remove `passive_indices` from
/// `V`": in the partial branch (lines 11-12) the new passive-VPs enter `V`
/// before the search at line 15 runs, so its indices refer to that
/// extended `V`. (For the other two kinds of plan one of the two steps is
/// empty, so the same replay order is correct for them too.)
fn try_alloc_heavy_within(
    heavy_sorted: &[DagMetrics],
    mh: u16,
) -> Option<(Vec<HeavyPlan>, Vec<PassiveVp>)> {
    let mut processors_left = mh; // |P|
    let mut next_slot: usize = 0;
    let mut pool: Vec<PassiveVp> = Vec::new(); // V
    let mut plans = Vec::with_capacity(heavy_sorted.len());

    let mut tasks = heavy_sorted.iter();

    // Lines 4-18.
    for config in tasks.by_ref() {
        let m_i = config.min_dedicated_cores()?; // line 6, eq. (1)
        let (budgets, active_budget_sum) = if m_i <= processors_left {
            // Lines 8-9: full active-VP group, Theorem 1.
            let budgets = full_active_vp_budgets(
                config.volume,
                config.critical_path,
                config.relative_deadline,
                m_i,
            );
            (budgets, None)
        } else {
            // Lines 11-12: an active-VP group on all |P| remaining
            // processors -- possibly none at all, in which case Θ is empty
            // and line 17's condition (13) degenerates to Σθ = 0.
            let budgets = partial_active_vp_budgets(
                config.critical_path,
                config.relative_deadline,
                processors_left,
            );
            let sum = budgets.iter().sum();
            (budgets, Some(sum))
        };

        let n = budgets.len();
        pool.extend(generate_passive_vps(
            &budgets,
            next_slot..next_slot + n,
            config.period,
            config.relative_deadline,
            config.max_parallelism,
        ));
        next_slot += n;
        // `n` <= `processors_left` in both branches.
        processors_left -= n as u16;

        let Some(active_budget_sum) = active_budget_sum else {
            plans.push(HeavyPlan {
                cores_used: n as u16,
                active_budgets: budgets,
                passive_indices: Vec::new(),
            });
            continue;
        };

        // Lines 13-17.
        let passive_indices = pull_passives_until_schedulable(
            config,
            PassiveTest::Theorem4 { active_budget_sum },
            &pool,
        )
        .ok()?;
        take_indices(&mut pool, &passive_indices);
        plans.push(HeavyPlan {
            cores_used: n as u16,
            active_budgets: budgets,
            passive_indices,
        });
        break; // Line 18: goto line 19.
    }

    // Lines 19-25: every remaining heavy DAG on passive-VPs only, Theorem 2.
    for config in tasks {
        let passive_indices =
            pull_passives_until_schedulable(config, PassiveTest::Theorem2, &pool).ok()?;
        take_indices(&mut pool, &passive_indices);
        plans.push(HeavyPlan {
            cores_used: 0,
            active_budgets: Vec::new(),
            passive_indices,
        });
    }

    Some((plans, pool))
}

/// The outer search from the paper's Algorithm 2 (lines 3-6): try
/// `M_h = 1, 2, ..., max_cores`, returning the *smallest* one for which
/// `AllocH(M_h)` succeeds, together with the resulting plans and leftover
/// (still placeholder-addressed) passive-VP pool. `None` if no `M_h` up to
/// `max_cores` works (line 6).
///
/// One deliberate deviation: with no heavy DAG at all this returns
/// `M_h = 0`. Read literally, line 3 starts at `M_h = 1` and `AllocH(1)`
/// trivially succeeds on an empty queue, which would withhold one processor
/// (hosting no VP of any kind) from the light tasks' `M_l = m - M_h` for no
/// reason -- an artifact of the pseudo-code, not of the method.
fn search_min_heavy_cores(
    heavy_sorted: &[DagMetrics],
    max_cores: u16,
) -> Option<(u16, Vec<HeavyPlan>, Vec<PassiveVp>)> {
    if heavy_sorted.is_empty() {
        return Some((0, Vec::new(), Vec::new()));
    }
    (1..=max_cores).find_map(|mh| {
        try_alloc_heavy_within(heavy_sorted, mh).map(|(plans, pool)| (mh, plans, pool))
    })
}

/// How a partitioned-EDF step picks among processors a light task fits on.
/// Shared with Federated; see [`crate::dag_sched::partition`]. The batch
/// path ([`plan_batch`]) uses it for the paper's `Partition(L, M_l)`
/// (Algorithm 2 line 20, which specifies best-fit); the incremental
/// [`admit_one`] path uses it for [`choose_partition`].
pub use crate::dag_sched::partition::PackingStrategy;

/// One partitioned-EDF core as the incremental [`admit_one`] path tracks
/// it: a bare (non-active-VP) core hosting zero or more light DAGs
/// scheduled together by ordinary single-core EDF, admitted as long as
/// their combined density (`C/D`, scaled by [`resource::UTILIZATION_SCALE`])
/// stays within one core's capacity -- a sufficient uniprocessor-EDF test
/// that, unlike the batch path's DBF* test, doesn't require the tasks to
/// arrive in deadline order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Partition {
    committed_density_scaled: u64,
}

impl Partition {
    fn remaining_scaled(&self) -> u64 {
        resource::UTILIZATION_SCALE.saturating_sub(self.committed_density_scaled)
    }
}

/// Pick which of `partitions` a task needing `density_scaled` should join,
/// per `strategy`. `None` if none has room — the caller must open a new
/// partition (claim one more bare core) instead.
fn choose_partition(
    strategy: PackingStrategy,
    partitions: &[Partition],
    density_scaled: u64,
) -> Option<usize> {
    let fits = |p: &&Partition| p.remaining_scaled() >= density_scaled;
    match strategy {
        PackingStrategy::FirstFit => partitions.iter().position(|p| fits(&p)),
        PackingStrategy::BestFit => partitions
            .iter()
            .enumerate()
            .filter(|(_, p)| fits(p))
            .min_by_key(|(_, p)| p.remaining_scaled())
            .map(|(i, _)| i),
        PackingStrategy::WorstFit => partitions
            .iter()
            .enumerate()
            .filter(|(_, p)| fits(p))
            .max_by_key(|(_, p)| p.remaining_scaled())
            .map(|(i, _)| i),
    }
}

/// The shared pool of passive-VPs generated as a byproduct of active-VP
/// allocation, not yet claimed by any task. Mirrors
/// [`crate::dag_sched::resource`]'s ledger pattern: bundled behind one lock,
/// module-private, exposed only through the `admit_*` functions below.
static PASSIVE_POOL: Mutex<Vec<PassiveVp>> = Mutex::new(Vec::new());

/// Partitioned-EDF cores opened so far by [`admit_light`]'s fallback (and
/// persisted after a successful [`admit_batch`]). Indices here are
/// meaningless outside this module; [`VFedAssignment::LightPartitioned`]
/// reports the real cpu instead.
static PARTITIONS: Mutex<Vec<(usize, Partition)>> = Mutex::new(Vec::new());

/// The outcome of admitting one DAG.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VFedAssignment {
    /// A heavy DAG: which cores its active-VP group claimed, and which
    /// passive-VPs (of other DAGs) it draws on to top up a partial group.
    Heavy {
        active: Vec<ActiveVp>,
        passive: Vec<PassiveVp>,
    },
    /// A light DAG served entirely from other DAGs' leftover passive-VPs
    /// (Theorem 2) — no dedicated core at all.
    LightPassive { passive: Vec<PassiveVp> },
    /// A light DAG that didn't fit any available passive-VP, instead
    /// sharing a partitioned-EDF core with (possibly) other light DAGs.
    LightPartitioned { cpu: usize },
}

impl VFedAssignment {
    /// Turn this admission decision into the `SchedulerType` every node of
    /// the DAG it was decided for should be registered with — mirroring
    /// `Provision::into_scheduler_type` for Federated.
    ///
    /// A [`VFedAssignment::Heavy`] whose active-VP group needed passive-VP
    /// top-up (`passive` non-empty — Theorem 4, including the fully-passive
    /// case where `active` is empty too) becomes
    /// [`SchedulerType::MixedVp`] (`scheduler::mixed_vp`); a pure active-VP
    /// group (`passive` empty) becomes [`SchedulerType::ActiveVp`]; a pure
    /// passive-VP or partitioned-EDF light DAG becomes
    /// [`SchedulerType::PassiveVp`]/[`SchedulerType::ClusteredEDF`]
    /// respectively.
    pub fn into_scheduler_type(&self, relative_deadline: u64) -> Option<SchedulerType> {
        match self {
            VFedAssignment::Heavy { active, passive } => {
                let active_set = active
                    .iter()
                    .fold(CpuSet::empty(), |set, a| set.with(a.cpu));
                if passive.is_empty() {
                    let leading = active.first()?.cpu;
                    Some(SchedulerType::ActiveVp(active_set, leading))
                } else {
                    // `leading` is meaningless when `active` is empty (the
                    // fully-passive-covered Theorem 4 case) — no cpu ever
                    // reads it, since `MixedVpScheduler` only ever checks
                    // `leading_of` for cpus actually in `active_set`, which
                    // would then itself be empty.
                    let leading = active.first().map_or(0, |a| a.cpu);
                    let passive_set = passive
                        .iter()
                        .fold(CpuSet::empty(), |set, p| set.with(p.cpu));
                    Some(SchedulerType::MixedVp(active_set, leading, passive_set))
                }
            }
            VFedAssignment::LightPassive { passive } => {
                let cpu_set = passive
                    .iter()
                    .fold(CpuSet::empty(), |set, p| set.with(p.cpu));
                Some(SchedulerType::PassiveVp(cpu_set))
            }
            VFedAssignment::LightPartitioned { cpu } => Some(SchedulerType::ClusteredEDF(
                relative_deadline,
                CpuSet::empty().with(*cpu),
            )),
        }
    }
}

/// Admit one DAG against whatever is currently free, claiming cores from
/// [`resource`] and consuming/producing passive-VPs/partitions in
/// [`PASSIVE_POOL`]/[`PARTITIONS`] for real. This is DAG-granularity,
/// greedy, single-pass admission (see the module doc): calling it once per
/// DAG in the paper's sorted order (heavy DAGs by ascending `D - L`, then
/// light DAGs by descending `C/D`) reproduces the paper's
/// `AllocH`/`Allo_Both` allocation for a *given* number of cores dedicated
/// to heavy tasks, but each call commits independently — unlike
/// [`admit_batch`], one task failing to place here does not undo earlier
/// ones, and this never searches for a better `M_h`.
pub fn admit_one(
    config: DagMetrics,
    packing: PackingStrategy,
) -> Result<VFedAssignment, VFedError> {
    match classify(&config)? {
        TaskClass::Heavy { .. } => admit_heavy(config),
        TaskClass::Light => admit_light(config, packing),
    }
}

fn admit_heavy(config: DagMetrics) -> Result<VFedAssignment, VFedError> {
    // Held for the whole decide-and-commit sequence below (including the
    // nested `resource::allocate_cluster` call, which takes its own,
    // separate lock), so a concurrent `admit_one` can never plan against a
    // pool snapshot that has since changed underneath `take_indices` —
    // mirroring `resource.rs`'s own single-lock-per-decision pattern.
    let mut node = MCSNode::new();
    let mut pool = PASSIVE_POOL.lock(&mut node);

    let cores_available = resource::free_core_count();
    let plan = plan_heavy(&config, cores_available, &pool)?;

    let cores: CpuSet = if plan.cores_used > 0 {
        resource::allocate_cluster(plan.cores_used)?
    } else {
        CpuSet::empty()
    };

    let active: Vec<ActiveVp> = cores
        .iter()
        .zip(plan.active_budgets.iter().copied())
        .map(|(cpu, budget)| ActiveVp { cpu, budget })
        .collect();

    // Arm each claimed core's active-VP budget immediately, ahead of the
    // scheduler mechanism (`scheduler::active_vp`) that will eventually
    // dispatch onto it — this admission decision is the single source of
    // truth for what each core's budget *should* be, so it is set here
    // rather than left for some later, separate wiring step to get wrong.
    for a in &active {
        crate::scheduler::active_vp::set_initial_budget(a.cpu, a.budget);
    }

    let new_passives = generate_passive_vps(
        &plan.active_budgets,
        active.iter().map(|a| a.cpu),
        config.period,
        config.relative_deadline,
        config.max_parallelism,
    );

    let consumed = take_indices(&mut pool, &plan.passive_indices);
    pool.extend(new_passives);

    Ok(VFedAssignment::Heavy {
        active,
        passive: consumed,
    })
}

fn admit_light(config: DagMetrics, packing: PackingStrategy) -> Result<VFedAssignment, VFedError> {
    {
        let mut node = MCSNode::new();
        let mut pool = PASSIVE_POOL.lock(&mut node);
        if let Ok(indices) = plan_light(&config, &pool) {
            let consumed = take_indices(&mut pool, &indices);
            return Ok(VFedAssignment::LightPassive { passive: consumed });
        }
    }
    admit_light_partitioned(config, packing)
}

/// Fallback for a light DAG that fits no available passive-VP: bin-pack it
/// (by density, `C/D`) onto an already-open partitioned-EDF core per
/// `packing`, or open a fresh bare core if none has room.
fn admit_light_partitioned(
    config: DagMetrics,
    packing: PackingStrategy,
) -> Result<VFedAssignment, VFedError> {
    let density_scaled = resource::utilization_scaled(config.volume, config.relative_deadline);

    let mut node = MCSNode::new();
    let mut partitions = PARTITIONS.lock(&mut node);

    let existing: Vec<Partition> = partitions.iter().map(|(_, p)| *p).collect();
    if let Some(idx) = choose_partition(packing, &existing, density_scaled) {
        partitions[idx].1.committed_density_scaled += density_scaled;
        return Ok(VFedAssignment::LightPartitioned {
            cpu: partitions[idx].0,
        });
    }

    let cores = resource::allocate_cluster(1)?;
    let cpu = cores.iter().next().ok_or(VFedError::NoFeasibleAllocation)?;
    partitions.push((
        cpu,
        Partition {
            committed_density_scaled: density_scaled,
        },
    ));
    Ok(VFedAssignment::LightPartitioned { cpu })
}

/// Remove the entries at `indices` (assumed sorted ascending, as
/// [`pull_passives_until_schedulable`] produces them) from `pool` and
/// return them, preserving the order `indices` named them in.
fn take_indices(pool: &mut Vec<PassiveVp>, indices: &[usize]) -> Vec<PassiveVp> {
    let mut taken = Vec::with_capacity(indices.len());
    for &idx in indices.iter().rev() {
        taken.push(pool.remove(idx));
    }
    taken.reverse();
    taken
}

/// A light DAG's dry-run outcome inside [`admit_batch`]'s simulation,
/// before any real core is claimed.
enum LightOutcome {
    /// Placed on passive-VPs (Algorithm 2 lines 11-17): indices into the
    /// (placeholder-addressed) pool at the moment this DAG was planned,
    /// exactly as [`plan_light`] returned them.
    Passive(Vec<usize>),
    /// Placed by `Partition(L, M_l)` (Algorithm 2 line 20): index of the
    /// processor among the `M_l` partitioned-EDF processors.
    Partitioned(usize),
}

/// The full decision for a batch, computed against an explicit `max_cores`
/// budget rather than the live `resource` ledger -- everything
/// [`admit_batch`] needs to commit, and everything [`is_batch_feasible`]
/// needs to answer yes/no, with no shared/global state touched to produce
/// it. Fields are placeholder-addressed (`bins`' indices, `pool` entries)
/// exactly as the dry run that built them, not real cpu ids.
struct BatchPlan {
    /// `configs`' heavy entries, `(original_index, config)`, sorted
    /// ascending by laxity `D - L` (Algorithm 2 line 1).
    heavy: Vec<(usize, DagMetrics)>,
    /// `configs`' light entries, `(original_index, config)`, sorted
    /// descending by density `C/D` (Algorithm 2 line 8).
    light: Vec<(usize, DagMetrics)>,
    mh: u16,
    heavy_plans: Vec<HeavyPlan>,
    /// One outcome per `light` entry, same order.
    outcomes: Vec<LightOutcome>,
    /// The `M_l` partitioned-EDF processors as `Partition` left them.
    bins: Vec<partition::Bin>,
}

/// Compute the full batch decision for `configs` against `max_cores` free
/// cores -- the paper's Algorithm 2 (`Allo_Both`):
///
/// 1. heavy DAGs (`C > D`) sorted ascending by `D - L` (line 1), then the
///    smallest `M_h` for which Algorithm 1 succeeds (lines 3-6,
///    [`search_min_heavy_cores`]);
/// 2. light DAGs sorted descending by `C/D` (line 8), each tried on the
///    passive-VPs left in `V` by Theorem 2 (lines 10-19);
/// 3. every light DAG still left, partitioned onto the remaining
///    `M_l = m - M_h` processors by best-fit partitioned EDF with the test
///    of the paper's reference \[15\] (line 20; see
///    [`crate::dag_sched::partition`]). `packing` selects the fit rule --
///    the paper specifies [`PackingStrategy::BestFit`].
///
/// Pure: touches only local state (`max_cores` is a parameter, not
/// [`resource::free_core_count`]), so a task set that doesn't fit leaves
/// nothing to undo -- the single source of truth for both [`admit_batch`]'s
/// commit and [`is_batch_feasible`]'s pure yes/no.
fn plan_batch(
    configs: &[DagMetrics],
    max_cores: u16,
    packing: PackingStrategy,
) -> Result<BatchPlan, VFedError> {
    let mut heavy: Vec<(usize, DagMetrics)> = Vec::new();
    let mut light: Vec<(usize, DagMetrics)> = Vec::new();
    for (i, &config) in configs.iter().enumerate() {
        match classify(&config)? {
            TaskClass::Heavy { .. } => heavy.push((i, config)),
            TaskClass::Light => light.push((i, config)),
        }
    }

    heavy.sort_by_key(|(_, c)| c.relative_deadline - c.critical_path);
    light.sort_by(|(_, a), (_, b)| {
        (b.volume * a.relative_deadline).cmp(&(a.volume * b.relative_deadline))
    });

    let heavy_configs: Vec<DagMetrics> = heavy.iter().map(|(_, c)| *c).collect();
    let (mh, heavy_plans, mut pool) =
        search_min_heavy_cores(&heavy_configs, max_cores).ok_or(VFedError::NoFeasibleAllocation)?;

    // Lines 10-19: passive-VPs first, for every light DAG in density order.
    let mut outcomes: Vec<Option<LightOutcome>> = Vec::with_capacity(light.len());
    let mut leftover: Vec<usize> = Vec::new(); // positions in `light`
    for (pos, (_, config)) in light.iter().enumerate() {
        match plan_light(config, &pool) {
            Ok(indices) => {
                take_indices(&mut pool, &indices);
                outcomes.push(Some(LightOutcome::Passive(indices)));
            }
            Err(_) => {
                outcomes.push(None);
                leftover.push(pos);
            }
        }
    }

    // Line 20: Partition(L, M_l).
    let ml = max_cores - mh;
    let seq: Vec<partition::SeqTask> = leftover
        .iter()
        .map(|&pos| {
            let c = &light[pos].1;
            partition::SeqTask {
                volume: c.volume,
                deadline: c.relative_deadline,
                period: c.period,
            }
        })
        .collect();
    let (placement, bins) = partition::partition(&seq, ml, partition::Condition::Demand, packing)
        .ok_or(VFedError::NoFeasibleAllocation)?;
    for (&pos, &bin) in leftover.iter().zip(placement.iter()) {
        outcomes[pos] = Some(LightOutcome::Partitioned(bin));
    }

    Ok(BatchPlan {
        heavy,
        light,
        mh,
        heavy_plans,
        // Every entry is `Some` by now: each light DAG was either placed on
        // passive-VPs above or is in `leftover`, which `partition` placed.
        outcomes: outcomes
            .into_iter()
            .collect::<Option<Vec<_>>>()
            .ok_or(VFedError::NoFeasibleAllocation)?,
        bins,
    })
}

/// Pure feasibility check: would [`admit_batch`] accept this whole `configs`
/// batch against `max_cores` free cores? No `resource`/`PASSIVE_POOL`/
/// `PARTITIONS` global state is touched either way, so this is safe to call
/// repeatedly against many candidate task sets (e.g. an offline
/// schedulability-ratio sweep) without any admission/rollback bookkeeping.
pub fn is_batch_feasible(configs: &[DagMetrics], max_cores: u16, packing: PackingStrategy) -> bool {
    plan_batch(configs, max_cores, packing).is_ok()
}

/// Admit every DAG in `configs` as one all-or-nothing batch (see
/// [`plan_batch`] for the decision itself). The whole simulation runs
/// against placeholder ids first — no real core is claimed and neither
/// [`PASSIVE_POOL`] nor [`PARTITIONS`] is touched — so a task set that
/// doesn't fit leaves every shared ledger exactly as it was.
///
/// Returns one [`VFedAssignment`] per input config, in `configs`' original
/// order.
pub fn admit_batch(
    configs: &[DagMetrics],
    packing: PackingStrategy,
) -> Result<Vec<VFedAssignment>, VFedError> {
    let max_cores = resource::free_core_count();
    let BatchPlan {
        heavy,
        light,
        mh,
        heavy_plans,
        outcomes,
        bins,
        ..
    } = plan_batch(configs, max_cores, packing)?;

    // Every config fits — commit for real. Claim exactly `mh` cores up
    // front and hand out slices of them to each heavy plan in order,
    // substituting real cpu ids for the placeholder slot ids the dry run
    // above used; passive-VP indices were already computed relative to
    // insertion order, which claiming cores in the same order preserves.
    let heavy_cores = if mh > 0 {
        resource::allocate_cluster(mh)?
    } else {
        CpuSet::empty()
    };
    let mut heavy_cores_iter = heavy_cores.iter();

    let mut real_pool: Vec<PassiveVp> = Vec::new();
    let mut results: Vec<(usize, VFedAssignment)> = Vec::with_capacity(configs.len());

    for (plan, (orig_idx, config)) in heavy_plans.iter().zip(heavy.iter()) {
        let cpus: Vec<usize> = (0..plan.cores_used)
            .filter_map(|_| heavy_cores_iter.next())
            .collect();
        let active: Vec<ActiveVp> = cpus
            .iter()
            .zip(plan.active_budgets.iter().copied())
            .map(|(&cpu, budget)| ActiveVp { cpu, budget })
            .collect();
        for a in &active {
            crate::scheduler::active_vp::set_initial_budget(a.cpu, a.budget);
        }

        // Same replay order as `try_alloc_heavy_within` planned against:
        // this DAG's own passive-VPs enter `V` first, then its `Π` leaves.
        real_pool.extend(generate_passive_vps(
            &plan.active_budgets,
            active.iter().map(|a| a.cpu),
            config.period,
            config.relative_deadline,
            config.max_parallelism,
        ));
        let consumed = take_indices(&mut real_pool, &plan.passive_indices);

        results.push((
            *orig_idx,
            VFedAssignment::Heavy {
                active,
                passive: consumed,
            },
        ));
    }

    let mut real_partition_cpus: Vec<usize> = Vec::new();
    for (outcome, (orig_idx, _config)) in outcomes.into_iter().zip(light.iter()) {
        match outcome {
            LightOutcome::Passive(indices) => {
                let consumed = take_indices(&mut real_pool, &indices);
                results.push((
                    *orig_idx,
                    VFedAssignment::LightPassive { passive: consumed },
                ));
            }
            LightOutcome::Partitioned(idx) => {
                while real_partition_cpus.len() <= idx {
                    let cores = resource::allocate_cluster(1)?;
                    let cpu = cores.iter().next().ok_or(VFedError::NoFeasibleAllocation)?;
                    real_partition_cpus.push(cpu);
                }
                results.push((
                    *orig_idx,
                    VFedAssignment::LightPartitioned {
                        cpu: real_partition_cpus[idx],
                    },
                ));
            }
        }
    }

    // Persist the leftover passive-VP pool and newly-opened partitions so
    // later `admit_one` calls can still draw on this batch's leftovers.
    {
        let mut node = MCSNode::new();
        let mut global_pool = PASSIVE_POOL.lock(&mut node);
        global_pool.extend(real_pool);
    }
    {
        let mut node = MCSNode::new();
        let mut global_partitions = PARTITIONS.lock(&mut node);
        for (i, &cpu) in real_partition_cpus.iter().enumerate() {
            // The incremental `admit_one` path tracks a partition by its
            // committed density only.
            let committed_density_scaled = bins[i]
                .tasks()
                .iter()
                .map(|t| resource::utilization_scaled(t.volume, t.deadline))
                .sum();
            global_partitions.push((
                cpu,
                Partition {
                    committed_density_scaled,
                },
            ));
        }
    }

    results.sort_by_key(|(i, _)| *i);
    Ok(results.into_iter().map(|(_, a)| a).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vp(cpu: usize, active_budget: u64, owner_period: u64, owner_deadline: u64) -> PassiveVp {
        PassiveVp {
            cpu,
            kind: PassiveVpKind::Independent {
                active_budget,
                owner_period,
                owner_deadline,
            },
        }
    }

    #[test]
    fn test_is_heavy_boundary() {
        // density = C/D == 1 is Light (density <= 1), not Heavy.
        assert!(!is_heavy(100, 100));
        assert!(is_heavy(101, 100));
    }

    #[test]
    fn test_classify_rejects_arbitrary_deadline() {
        let config = DagMetrics::from_static(10, 5, 100, 150); // D=150 > T=100
        assert_eq!(
            classify(&config),
            Err(VFedError::ArbitraryDeadlineNotSupported {
                relative_deadline: 150,
                period: 100,
            })
        );
    }

    #[test]
    fn test_classify_infeasible_when_strictly_past_deadline() {
        // relative_deadline(40) < critical_path(50): infeasible regardless
        // of heaviness, since even the critical path alone can't finish.
        let config = DagMetrics::from_static(10, 50, 1000, 40);
        assert_eq!(
            classify(&config),
            Err(VFedError::Infeasible {
                critical_path: 50,
                relative_deadline: 40,
            })
        );
    }

    #[test]
    fn test_classify_deadline_equals_critical_path_light_ok() {
        // relative_deadline == critical_path == volume: a purely sequential
        // DAG fits its own critical path in exactly D == L time on a single
        // core -- feasible, not the unconditional Infeasible case.
        let config = DagMetrics::from_static(50, 50, 1000, 50);
        assert_eq!(classify(&config), Ok(TaskClass::Light));
    }

    #[test]
    fn test_classify_deadline_equals_critical_path_heavy_infeasible() {
        // relative_deadline == critical_path but volume > critical_path:
        // there's parallel-only work left to fit in a now-zero slack
        // window, which no core count can do.
        let config = DagMetrics::from_static(100, 50, 1000, 50);
        assert_eq!(
            classify(&config),
            Err(VFedError::Infeasible {
                critical_path: 50,
                relative_deadline: 50,
            })
        );
    }

    #[test]
    fn test_classify_light_and_heavy() {
        assert_eq!(
            classify(&DagMetrics::from_static(80, 20, 100, 90)).unwrap(),
            TaskClass::Light
        );
        assert_eq!(
            classify(&DagMetrics::from_static(100, 20, 100, 50)).unwrap(),
            TaskClass::Heavy { required_cores: 3 } // ceil((100-20)/(50-20)) = 3
        );
    }

    // Table I / Fig. 1 in RTSS21 (and reused verbatim in TPDS23 Sec 3):
    // Ci=11, Li=7. sbf isn't exercised on that example directly, so this
    // spot-checks the closed-form formula against Fig. 5/7's own numbers
    // instead: Θ = {θ1=6, θ2=2} serving a task with T=10 (job in Fig. 5 has
    // no explicit D; Fig. 7 plots sbf for D implied by the figure's own
    // deadline marker at t=10, i.e. D=T=10 there).
    #[test]
    fn test_sbf_zero_below_budget() {
        assert_eq!(sbf(5, 6, 10, 10), 0); // delta < budget
    }

    #[test]
    fn test_sbf_matches_paper_worked_example_leading_vp() {
        // From TPDS23 Sec 6.2: p1 hosts tau1's leading active-VP,
        // theta1=8, T1=10, D1=8. sbf_p1(9) = 1 (used when planning tau3).
        let p1 = vp(1, 8, 10, 8);
        assert_eq!(p1.sbf(9), 1);
        // sbf_p1(D2=8) = 0, i.e. tau1's own leading complementary VP is
        // never useful back to a task whose deadline is no looser than 8.
        assert_eq!(p1.sbf(8), 0);
    }

    #[test]
    fn test_sbf_matches_paper_worked_example_non_leading_vp() {
        // p2..p5 host tau1's non-leading active-VPs: theta=1, T1=10, D1=8.
        let p = vp(2, 1, 10, 8);
        assert_eq!(p.sbf(8), 6); // used when planning tau2 (D2=8)
        assert_eq!(p.sbf(9), 7); // used when planning tau3 (D3=9)
    }

    // --- OURS2 (Pi', TPDS23 Sec 4.2): PassiveVpKind::AlwaysFree/Shared ---

    #[test]
    fn test_always_free_slot_sbf_is_full_delta() {
        let always_free = PassiveVp {
            cpu: 0,
            kind: PassiveVpKind::AlwaysFree,
        };
        for delta in [0, 1, 9, 1000] {
            assert_eq!(always_free.sbf(delta), delta);
        }
    }

    #[test]
    fn test_shared_slot_matches_hand_computed_example() {
        // mi=3 active-VPs, budgets {8,1,1} (tau1's own group from the paper's
        // worked example: theta1=8, theta2=theta3=1, T=10, D=8), li=2 shared
        // slots (so mi - li = 1 slot would instead be AlwaysFree, generated
        // separately -- this test only exercises the Shared formula itself).
        let shared = PassiveVp {
            cpu: 0,
            kind: PassiveVpKind::Shared {
                budgets: alloc::vec![8, 1, 1],
                owner_period: 10,
                owner_deadline: 8,
                li: 2,
            },
        };
        // At delta=9: sbf(9,8,10,8)=1, sbf(9,1,10,8)=7 each (from
        // test_sbf_matches_paper_worked_example_{leading,non_leading}_vp) ->
        // raw = 1+7+7 = 15. mi-li = 1, so subtract delta*1 = 9: 15-9 = 6,
        // divided by li=2 -> 3.
        assert_eq!(shared.sbf(9), 3);
    }

    #[test]
    fn test_shared_slot_li_zero_is_defensive_zero_not_panic() {
        let shared = PassiveVp {
            cpu: 0,
            kind: PassiveVpKind::Shared {
                budgets: alloc::vec![8, 1, 1],
                owner_period: 10,
                owner_deadline: 8,
                li: 0,
            },
        };
        assert_eq!(shared.sbf(9), 0);
    }

    #[test]
    fn test_generate_passive_vps_falls_back_to_independent_when_unknown() {
        let budgets = alloc::vec![8, 1, 1, 1, 1];
        let vps = generate_passive_vps(&budgets, 0..5, 10, 8, u16::MAX);
        assert_eq!(vps.len(), 5);
        for (v, &b) in vps.iter().zip(&budgets) {
            assert_eq!(
                v.kind,
                PassiveVpKind::Independent {
                    active_budget: b,
                    owner_period: 10,
                    owner_deadline: 8,
                }
            );
        }
    }

    #[test]
    fn test_generate_passive_vps_falls_back_to_independent_when_li_not_less_than_mi() {
        let budgets = alloc::vec![8, 1, 1];
        // max_parallelism == mi (3): precondition `Li < mi` violated.
        let vps = generate_passive_vps(&budgets, 0..3, 10, 8, 3);
        assert!(vps
            .iter()
            .all(|v| matches!(v.kind, PassiveVpKind::Independent { .. })));
    }

    #[test]
    fn test_generate_passive_vps_splits_always_free_then_shared() {
        let budgets = alloc::vec![8, 1, 1];
        let vps = generate_passive_vps(&budgets, 10..13, 10, 8, 2); // li=2, mi-li=1
        assert_eq!(vps.len(), 3);
        assert_eq!(vps[0].cpu, 10);
        assert!(matches!(vps[0].kind, PassiveVpKind::AlwaysFree));
        for v in &vps[1..] {
            assert!(matches!(v.kind, PassiveVpKind::Shared { li: 2, .. }));
        }
        assert_eq!(vps[1].cpu, 11);
        assert_eq!(vps[2].cpu, 12);
    }

    /// OURS2's split is *not* a pointwise improvement over OURS1 for every
    /// `(Δ, L)` — both are independently sound (each dominated by the real,
    /// unknown platform per Lemma 2/3), but folding `li` VPs' individual sbf
    /// values into one shared, averaged number trades away precision that
    /// can occasionally beat what the unconditional-availability slots gain
    /// back. Found empirically while writing this test suite: `budgets =
    /// [8,1,1,1,1]`, `li=3`, `delta=5` gives split=10 < independent=12
    /// (confirmed by hand: the two AlwaysFree slots (mi-li=2) only give
    /// `delta` each = 10 total, while the 3 Shared slots' averaged value
    /// floors to 0 at this delta, whereas the 4 independent budget=1 VPs
    /// each individually clear the L=0 bar with sbf=3). This matches the
    /// paper's own framing: OURS1 and OURS2 are reported as two *separate*
    /// evaluation curves (Fig. 9), not a provably-dominant pair — the
    /// paper's own empirical claim is aggregate ("OUR2 performs better...
    /// in all experiments" on randomly generated task sets), not a
    /// per-instance theorem. No regression test asserts the reverse either;
    /// this is documented behavior, not a bug.
    #[test]
    fn test_ours2_split_is_not_always_better_than_ours1_pointwise() {
        let budgets = alloc::vec![8u64, 1, 1, 1, 1];
        let independent = generate_passive_vps(&budgets, 0..5, 10, 8, u16::MAX);
        let split = generate_passive_vps(&budgets, 0..5, 10, 8, 3);
        let sum = |vps: &[PassiveVp]| -> u64 { vps.iter().map(|v| v.sbf(5)).sum() };
        assert_eq!(sum(&independent), 12);
        assert_eq!(sum(&split), 10);
    }

    /// Ports the TPDS23 Sec 6.2 / RTSS21 Sec VII worked example end to end:
    /// 5 tasks on 7 processors, tau1..tau3 heavy admitted against a
    /// (paper-given) budget of 6 cores for heavy tasks, tau4/tau5 light
    /// admitted from the leftover passive-VP pool. Every intermediate
    /// number below is quoted from the paper's own walkthrough.
    #[test]
    fn test_paper_worked_example() {
        let tau1 = DagMetrics::from_static(12, 7, 10, 8);
        let tau2 = DagMetrics::from_static(10, 4, 10, 8);
        let tau3 = DagMetrics::from_static(10, 2, 9, 9);
        let tau4 = DagMetrics::from_static(6, 2, 10, 10);
        let tau5 = DagMetrics::from_static(5, 3, 10, 10);

        assert_eq!(
            classify(&tau1).unwrap(),
            TaskClass::Heavy { required_cores: 5 }
        );
        assert_eq!(
            classify(&tau2).unwrap(),
            TaskClass::Heavy { required_cores: 2 }
        );
        assert_eq!(
            classify(&tau3).unwrap(),
            TaskClass::Heavy { required_cores: 2 }
        );
        assert_eq!(classify(&tau4).unwrap(), TaskClass::Light);
        assert_eq!(classify(&tau5).unwrap(), TaskClass::Light);

        let mut pool: Vec<PassiveVp> = Vec::new();

        // tau1: D1-L1=1 smallest, gets a full 5-core group; 5 of the 6
        // heavy-dedicated cores are now spoken for.
        let plan1 = plan_heavy(&tau1, 5, &pool).unwrap();
        assert_eq!(plan1.cores_used, 5);
        assert_eq!(plan1.active_budgets, alloc::vec![8, 1, 1, 1, 1]);
        assert!(plan1.passive_indices.is_empty());
        pool.extend(
            plan1
                .active_budgets
                .iter()
                .enumerate()
                .map(|(i, &b)| vp(i + 1, b, tau1.period, tau1.relative_deadline)),
        ); // p1..p5

        // tau2: D2-L2=4, only 1 core (p6) remains of the 6-core heavy
        // budget; leading-only budget of 8 falls short of C2=10, topped up
        // from p2 (the paper's own choice).
        let plan2 = plan_heavy(&tau2, 1, &pool).unwrap();
        assert_eq!(plan2.cores_used, 1);
        assert_eq!(plan2.active_budgets, alloc::vec![8]);
        assert_eq!(plan2.passive_indices, alloc::vec![1]); // p2, index 1
        for &idx in plan2.passive_indices.iter().rev() {
            pool.remove(idx);
        }
        pool.push(vp(6, 8, tau2.period, tau2.relative_deadline)); // p6

        // tau3: D3-L3=7 largest, no cores left at all (all 6 heavy-budget
        // cores claimed); schedulable purely from p3 and p4.
        let plan3 = plan_heavy(&tau3, 0, &pool).unwrap();
        assert_eq!(plan3.cores_used, 0);
        assert!(plan3.active_budgets.is_empty());
        // Remaining pool at this point (insertion order): p1, p3, p4, p5,
        // p6 (p2 was removed above). p3 and p4 are indices 1 and 2.
        assert_eq!(plan3.passive_indices, alloc::vec![1, 2]);
        for &idx in plan3.passive_indices.iter().rev() {
            pool.remove(idx);
        }

        // Light tasks: pool is now p1, p5, p6 (indices 0,1,2). Paper picks
        // p5 for tau4 (C4/D4 > C5/D5), leaving p1/p6 useless to either
        // (both fail condition (17) for D=10, as the sbf tests above imply
        // for p1; p6 has the same shape as p1 with theta=8).
        let plan4 = plan_light(&tau4, &pool).unwrap();
        assert_eq!(plan4, alloc::vec![1]); // p5
    }

    /// The paper's own worked example states "we omit the enumerating of
    /// Mh when Mh<=5, where AllocH(Mh) returns failure" and then walks
    /// through Mh=6 directly. This confirms the search itself (not just
    /// the per-Mh math already validated above) independently arrives at
    /// the same Mh=6 for tau1..tau3 out of up to 7 available cores.
    #[test]
    fn test_search_min_heavy_cores_matches_paper() {
        let tau1 = DagMetrics::from_static(12, 7, 10, 8);
        let tau2 = DagMetrics::from_static(10, 4, 10, 8);
        let tau3 = DagMetrics::from_static(10, 2, 9, 9);
        let heavy = [tau1, tau2, tau3]; // already ascending D-L: 1, 4, 7

        let (mh, plans, pool) = search_min_heavy_cores(&heavy, 7).unwrap();
        assert_eq!(mh, 6);
        assert_eq!(plans.len(), 3);
        assert_eq!(plans[0].cores_used, 5); // tau1
        assert_eq!(plans[1].cores_used, 1); // tau2
        assert_eq!(plans[2].cores_used, 0); // tau3
                                            // tau3 consumed 2 of the passive-VPs generated along the way (p3,
                                            // p4 in the paper's own labeling); 5+1 active slots minus 2
                                            // consumed by tau2/tau3 plus... simplest direct check: the leftover
                                            // pool has 3 entries left over for light tasks, as the paper's
                                            // walkthrough shows (p1, p5, p6).
        assert_eq!(pool.len(), 3);
    }

    #[test]
    fn test_search_min_heavy_cores_fails_when_no_mh_fits() {
        // A single heavy task needing far more cores than exist anywhere.
        let huge = DagMetrics::from_static(10_000, 10, 20, 15);
        assert!(search_min_heavy_cores(&[huge], 4).is_none());
    }

    #[test]
    fn test_search_min_heavy_cores_empty_input() {
        let (mh, plans, pool) = search_min_heavy_cores(&[], 7).unwrap();
        assert_eq!(mh, 0);
        assert!(plans.is_empty());
        assert!(pool.is_empty());
    }

    fn partition(committed_scaled: u64) -> Partition {
        Partition {
            committed_density_scaled: committed_scaled,
        }
    }

    #[test]
    fn test_choose_partition_first_fit_picks_earliest_that_fits() {
        let partitions = [partition(900_000), partition(100_000), partition(500_000)];
        // Needs 400_000: partition 0 has only 100_000 free (doesn't fit),
        // partition 1 has 900_000 free (fits) — first-fit stops there even
        // though partition 2 also fits.
        assert_eq!(
            choose_partition(PackingStrategy::FirstFit, &partitions, 400_000),
            Some(1)
        );
    }

    #[test]
    fn test_choose_partition_best_fit_picks_tightest_fit() {
        let partitions = [partition(900_000), partition(100_000), partition(500_000)];
        // Needs 400_000: partition 1 (900_000 free) and partition 2
        // (500_000 free) both fit; best-fit picks the tighter one (2).
        assert_eq!(
            choose_partition(PackingStrategy::BestFit, &partitions, 400_000),
            Some(2)
        );
    }

    #[test]
    fn test_choose_partition_worst_fit_picks_roomiest() {
        let partitions = [partition(900_000), partition(100_000), partition(500_000)];
        assert_eq!(
            choose_partition(PackingStrategy::WorstFit, &partitions, 400_000),
            Some(1)
        );
    }

    #[test]
    fn test_choose_partition_none_fit() {
        let partitions = [partition(700_000), partition(800_000)];
        assert_eq!(
            choose_partition(PackingStrategy::BestFit, &partitions, 400_000),
            None
        );
    }

    #[test]
    fn test_into_scheduler_type_pure_active_vp() {
        let assignment = VFedAssignment::Heavy {
            active: alloc::vec![
                ActiveVp { cpu: 3, budget: 8 },
                ActiveVp { cpu: 4, budget: 1 },
            ],
            passive: Vec::new(),
        };
        let sched = assignment.into_scheduler_type(50).unwrap();
        match sched {
            SchedulerType::ActiveVp(cpu_set, leading) => {
                assert_eq!(leading, 3); // first entry is always the leading VP
                assert!(cpu_set.contains(3) && cpu_set.contains(4));
            }
            other => panic!("expected ActiveVp, got {other:?}"),
        }
    }

    #[test]
    fn test_into_scheduler_type_mixed_heavy() {
        let assignment = VFedAssignment::Heavy {
            active: alloc::vec![ActiveVp { cpu: 3, budget: 8 }],
            passive: alloc::vec![vp(9, 1, 10, 8)],
        };
        let sched = assignment.into_scheduler_type(50).unwrap();
        match sched {
            SchedulerType::MixedVp(active_set, leading, passive_set) => {
                assert_eq!(leading, 3);
                assert!(active_set.contains(3) && !active_set.contains(9));
                assert!(passive_set.contains(9) && !passive_set.contains(3));
            }
            other => panic!("expected MixedVp, got {other:?}"),
        }
    }

    #[test]
    fn test_into_scheduler_type_mixed_heavy_fully_passive() {
        // `active` empty (cores_available == 0 in `plan_heavy`): the whole
        // heavy DAG is served from other DAGs' passive-VPs, no dedicated
        // core at all. `leading` is a meaningless placeholder here (0),
        // never read since `active_set` is empty too.
        let assignment = VFedAssignment::Heavy {
            active: alloc::vec![],
            passive: alloc::vec![vp(5, 1, 10, 8), vp(6, 1, 10, 8)],
        };
        let sched = assignment.into_scheduler_type(50).unwrap();
        match sched {
            SchedulerType::MixedVp(active_set, leading, passive_set) => {
                assert!(active_set.is_empty());
                assert_eq!(leading, 0);
                assert!(passive_set.contains(5) && passive_set.contains(6));
            }
            other => panic!("expected MixedVp, got {other:?}"),
        }
    }

    #[test]
    fn test_into_scheduler_type_pure_passive() {
        let assignment = VFedAssignment::LightPassive {
            passive: alloc::vec![vp(5, 1, 10, 8), vp(6, 1, 10, 8)],
        };
        let sched = assignment.into_scheduler_type(50).unwrap();
        match sched {
            SchedulerType::PassiveVp(cpu_set) => {
                assert!(cpu_set.contains(5) && cpu_set.contains(6));
            }
            other => panic!("expected PassiveVp, got {other:?}"),
        }
    }

    #[test]
    fn test_into_scheduler_type_partitioned() {
        let assignment = VFedAssignment::LightPartitioned { cpu: 7 };
        let sched = assignment.into_scheduler_type(50).unwrap();
        match sched {
            SchedulerType::ClusteredEDF(deadline, cpu_set) => {
                assert_eq!(deadline, 50);
                assert!(cpu_set.contains(7));
                assert_eq!(cpu_set.iter().count(), 1);
            }
            other => panic!("expected ClusteredEDF, got {other:?}"),
        }
    }

    #[test]
    fn test_is_batch_feasible_matches_paper_worked_example() {
        // Same 5-task, 7-core scenario `test_search_min_heavy_cores_matches_paper`
        // validates piecemeal; `is_batch_feasible` runs the whole `plan_batch`
        // decision (heavy search + light dry run) end to end and touches no
        // global state, unlike `admit_batch`/`admit_one`.
        let tau1 = DagMetrics::from_static(12, 7, 10, 8);
        let tau2 = DagMetrics::from_static(10, 4, 10, 8);
        let tau3 = DagMetrics::from_static(10, 2, 9, 9);
        let tau4 = DagMetrics::from_static(6, 2, 10, 10);
        let tau5 = DagMetrics::from_static(5, 3, 10, 10);
        assert!(is_batch_feasible(
            &[tau1, tau2, tau3, tau4, tau5],
            7,
            PackingStrategy::FirstFit
        ));
    }

    /// Theorem 2's condition (12) keeps the `L_i` term that Theorem 4's
    /// condition (13) replaces with `Σθ`: with one always-available
    /// passive-VP (`sbf(D) = D`), (12) reads `C <= L + (D - L) = D`, while
    /// (13) with an empty `Θ` reads `C <= D - L`.
    #[test]
    fn test_theorem2_keeps_critical_path_term() {
        let free = PassiveVp {
            cpu: 0,
            kind: PassiveVpKind::AlwaysFree,
        };
        let (c, l, d) = (8, 4, 10);
        assert!(PassiveTest::Theorem2.holds(c, l, d, &[&free]));
        assert!(!PassiveTest::Theorem4 {
            active_budget_sum: 0
        }
        .holds(c, l, d, &[&free]));
    }

    /// The paper's Sec. 6.2 walkthrough for tau3 (C=10, L=2, D=9) after the
    /// `goto line 19`: "Theorem 2 is not satisfied, i.e., 8 > 5" with p3
    /// alone (sbf(9) - 2 = 5), then satisfied with p3 and p4 (8 < 5 + 5) --
    /// i.e. the paper evaluates (12) as `C - L <= Σ(sbf - L)`.
    #[test]
    fn test_theorem2_matches_paper_walkthrough_for_tau3() {
        let tau1 = DagMetrics::from_static(12, 7, 10, 8);
        let p3 = vp(3, 1, tau1.period, tau1.relative_deadline);
        let p4 = vp(4, 1, tau1.period, tau1.relative_deadline);
        assert_eq!(p3.sbf(9), 7);
        assert!(!PassiveTest::Theorem2.holds(10, 2, 9, &[&p3]));
        assert!(PassiveTest::Theorem2.holds(10, 2, 9, &[&p3, &p4]));
    }

    #[test]
    fn test_is_batch_feasible_false_when_infeasible() {
        let huge = DagMetrics::from_static(10_000, 10, 20, 15);
        assert!(!is_batch_feasible(&[huge], 4, PackingStrategy::FirstFit));
    }

    #[test]
    fn test_is_batch_feasible_false_when_core_budget_too_small() {
        // Same worked example, but only 5 cores instead of 7 — not enough
        // even for the heavy DAGs' minimum-Mh search to succeed.
        let tau1 = DagMetrics::from_static(12, 7, 10, 8);
        let tau2 = DagMetrics::from_static(10, 4, 10, 8);
        let tau3 = DagMetrics::from_static(10, 2, 9, 9);
        assert!(!is_batch_feasible(
            &[tau1, tau2, tau3],
            5,
            PackingStrategy::FirstFit
        ));
    }
}
