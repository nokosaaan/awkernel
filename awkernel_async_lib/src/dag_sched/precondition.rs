//! Each admission policy's preconditions on a DAG's timing `(C, L, T, D)`,
//! kept as one table ([`requirements`]) and checked by one function
//! ([`check`]) -- so the kernel's DAG creation, the policies themselves and
//! the offline tools reject a DAG for the same reason with the same
//! [`DagError`], before any admission test runs.
//!
//! | check | Li | DATE | IPDPS | V-Fed | DAG-Fluid | SFS | Laxity |
//! |---|---|---|---|---|---|---|---|
//! | [`Check::ImplicitDeadline`] (`D = T`) | x | | | | | | |
//! | [`Check::ConstrainedDeadline`] (`D <= T`) | | x | | x | | x | |
//! | [`Check::CriticalPathWithinDeadline`] (`L <= D`) | x | x | x | x | x | x | |
//! | [`Check::SlackForParallelWork`] (`D = L` only if `C = L`) | x | | | x | | | |
//! | [`Check::PositiveSlack`] (`L < D`) | | | | | | | x |
//!
//! Li = Li et al. ECRTS 2014 (`n = ceil((C-L)/(D-L))`, implicit deadlines);
//! DATE/IPDPS = Baruah 2015, whose `MINPROCS` is list scheduling and so
//! stays defined at `D = L`; V-Fed = Jiang et al. (constrained deadlines,
//! `m = ceil((C-L)/(D-L))`); DAG-Fluid = Guan et al. TC 2022 (constrained
//! or arbitrary deadlines, Table 1); SFS = Lendve et al. JSA 2026
//! (constrained deadlines); Laxity = this project's static-laxity baseline
//! (`compute_node_laxity` needs `D > L`). A DAG missing its period or
//! deadline ([`DagError::MissingTiming`]) is rejected under every policy
//! before its metrics exist; structural errors (cycles, several sources,
//! ...) are [`crate::dag`]'s own checks.

use crate::{
    dag::DagError,
    dag_sched::{metrics::DagMetrics, policy::federated::FederatedVariant},
};

/// An admission policy, as the key of [`requirements`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Policy {
    /// Federated, per variant (the batch path picks the variant from the
    /// task set's deadline type, see [`FederatedVariant::for_task_set`]).
    Federated(FederatedVariant),
    VFed,
    DagFluid,
    Sfs,
    Laxity,
}

/// One precondition on a DAG's timing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Check {
    /// `D = T`.
    ImplicitDeadline,
    /// `D <= T`.
    ConstrainedDeadline,
    /// `L <= D`.
    CriticalPathWithinDeadline,
    /// `D = L` only for a purely sequential DAG (`C = L`): the processor
    /// count `ceil((C-L)/(D-L))` must be defined.
    SlackForParallelWork,
    /// `L < D`.
    PositiveSlack,
}

/// The preconditions `policy` puts on every DAG it admits, in the order
/// [`check`] tests them.
pub const fn requirements(policy: Policy) -> &'static [Check] {
    use Check::*;
    match policy {
        Policy::Federated(FederatedVariant::LiImplicit) => &[
            ImplicitDeadline,
            CriticalPathWithinDeadline,
            SlackForParallelWork,
        ],
        Policy::Federated(FederatedVariant::BaruahConstrained) => {
            &[ConstrainedDeadline, CriticalPathWithinDeadline]
        }
        Policy::Federated(FederatedVariant::BaruahArbitrary) => &[CriticalPathWithinDeadline],
        Policy::VFed => &[
            ConstrainedDeadline,
            CriticalPathWithinDeadline,
            SlackForParallelWork,
        ],
        Policy::DagFluid => &[CriticalPathWithinDeadline],
        Policy::Sfs => &[ConstrainedDeadline, CriticalPathWithinDeadline],
        Policy::Laxity => &[PositiveSlack],
    }
}

impl Check {
    /// `Ok` if `m` (the DAG reported as `dag_id`) satisfies this check.
    pub fn verify(self, dag_id: u32, m: &DagMetrics) -> Result<(), DagError> {
        let (c, l, t, d) = (m.volume, m.critical_path, m.period, m.relative_deadline);
        let ok = match self {
            Check::ImplicitDeadline => d == t,
            Check::ConstrainedDeadline => d <= t,
            Check::CriticalPathWithinDeadline => l <= d,
            Check::SlackForParallelWork => d != l || c <= l,
            Check::PositiveSlack => l < d,
        };
        if ok {
            return Ok(());
        }
        Err(match self {
            Check::ImplicitDeadline => DagError::ImplicitDeadlineRequired {
                dag_id,
                relative_deadline: d,
                period: t,
            },
            Check::ConstrainedDeadline => DagError::ConstrainedDeadlineRequired {
                dag_id,
                relative_deadline: d,
                period: t,
            },
            Check::CriticalPathWithinDeadline => DagError::CriticalPathExceedsDeadline {
                dag_id,
                critical_path: l,
                relative_deadline: d,
            },
            Check::SlackForParallelWork => DagError::NoSlackForParallelWork {
                dag_id,
                volume: c,
                critical_path: l,
            },
            Check::PositiveSlack => DagError::NonPositiveSlack {
                dag_id,
                critical_path: l,
                relative_deadline: d,
            },
        })
    }
}

/// Check one DAG against every precondition of `policy`; the first one it
/// fails is the error.
pub fn check(policy: Policy, dag_id: u32, m: &DagMetrics) -> Result<(), DagError> {
    requirements(policy)
        .iter()
        .try_for_each(|c| c.verify(dag_id, m))
}

/// [`check`] for a whole batch, reporting each DAG by its position in
/// `metrics`.
pub fn check_all<'a>(
    policy: Policy,
    metrics: impl IntoIterator<Item = &'a DagMetrics>,
) -> Result<(), DagError> {
    metrics
        .into_iter()
        .enumerate()
        .try_for_each(|(i, m)| check(policy, i as u32, m))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn m(c: u64, l: u64, t: u64, d: u64) -> DagMetrics {
        DagMetrics::from_static(c, l, t, d)
    }

    const ALL: [Policy; 7] = [
        Policy::Federated(FederatedVariant::LiImplicit),
        Policy::Federated(FederatedVariant::BaruahConstrained),
        Policy::Federated(FederatedVariant::BaruahArbitrary),
        Policy::VFed,
        Policy::DagFluid,
        Policy::Sfs,
        Policy::Laxity,
    ];

    /// Which policies accept a DAG with the given timing, in [`ALL`]'s order.
    fn accepted_by(metrics: DagMetrics) -> [bool; 7] {
        ALL.map(|p| check(p, 0, &metrics).is_ok())
    }

    #[test]
    fn deadline_model_rows() {
        // Implicit (D = T): every policy's deadline model admits it.
        assert_eq!(accepted_by(m(30, 10, 20, 20)), [true; 7]);
        // Constrained (D < T): all but Li.
        assert_eq!(
            accepted_by(m(30, 10, 40, 20)),
            [false, true, true, true, true, true, true]
        );
        // Arbitrary (D > T): IPDPS, DAG-Fluid and the laxity baseline only.
        assert_eq!(
            accepted_by(m(30, 10, 15, 20)),
            [false, false, true, false, true, false, true]
        );
    }

    #[test]
    fn critical_path_rows() {
        // L > D: nobody.
        assert_eq!(accepted_by(m(30, 25, 20, 20)), [false; 7]);
        // D = L, parallel work (C > L): only the policies whose processor
        // count stays defined (DATE/IPDPS list scheduling, DAG-Fluid, SFS).
        assert_eq!(
            accepted_by(m(30, 20, 20, 20)),
            [false, true, true, false, true, true, false]
        );
        // D = L, purely sequential (C = L): all but the laxity baseline.
        assert_eq!(
            accepted_by(m(20, 20, 20, 20)),
            [true, true, true, true, true, true, false]
        );
    }

    #[test]
    fn errors_name_the_failed_check_and_the_dag() {
        assert_eq!(
            check(Policy::VFed, 3, &m(10, 5, 100, 150)),
            Err(DagError::ConstrainedDeadlineRequired {
                dag_id: 3,
                relative_deadline: 150,
                period: 100,
            })
        );
        assert_eq!(
            check_all(Policy::VFed, &[m(10, 5, 100, 50), m(100, 50, 1000, 50)]),
            Err(DagError::NoSlackForParallelWork {
                dag_id: 1,
                volume: 100,
                critical_path: 50,
            })
        );
        assert_eq!(
            check(Policy::Laxity, 0, &m(50, 50, 100, 50)).map_err(|e| e.with_dag_id(7)),
            Err(DagError::NonPositiveSlack {
                dag_id: 7,
                critical_path: 50,
                relative_deadline: 50,
            })
        );
    }
}
