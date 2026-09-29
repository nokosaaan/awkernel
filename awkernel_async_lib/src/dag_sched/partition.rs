//! Partitioned-EDF assignment of sequential (low-density) tasks, shared by
//! every paper-faithful admission test in [`super::policy`] that partitions
//! its light tasks:
//!
//! - Federated, constrained deadlines (Baruah, DATE 2015, Fig. 4):
//!   first-fit, [`Condition::Demand`] only. (Li et al.'s implicit-deadline
//!   variant admits its low-utilization tasks by `m_low >= 2 Σ U` instead
//!   and only uses this for the run-time placement; with `D = T` the DBF*
//!   condition reduces to per-processor `Σ u <= 1`.)
//! - Federated, arbitrary deadlines (Baruah, IPDPS 2015, Fig. 4):
//!   first-fit, [`Condition::DemandAndUtilization`].
//! - V-Fed (Jiang et al., TPDS 2023, Algorithm 2 line 20): best-fit
//!   partitioned EDF with the test of its reference \[15\] (Baruah & Fisher,
//!   "The partitioned multiprocessor scheduling of deadline-constrained
//!   sporadic task systems", IEEE TC 2006) -- the same DBF* test as DATE
//!   2015's Fig. 4, which cites that same paper as its \[7\].
//!
//! The DBF* test at `D_i` is only valid when every task already on the
//! processor has `D_j <= D_i` (Baruah & Fisher's own "without loss of
//! generality, assume `D_i <= D_{i+1}`"), so [`partition`] sorts its input
//! by non-decreasing relative deadline before assigning anything.

use alloc::vec::Vec;

/// One sequential task as the partitioning step sees it: a DAG's internal
/// parallelism cannot be exploited on a single processor, so each
/// low-density DAG is treated as a three-parameter sporadic task
/// `(C = volume, D, T)` (DATE 2015 Sec. IV-B / IPDPS 2015 Sec. IV-B).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SeqTask {
    pub volume: u64,
    pub deadline: u64,
    pub period: u64,
}

impl SeqTask {
    fn utilization(&self) -> f64 {
        self.volume as f64 / self.period as f64
    }

    /// `DBF*(τ, t)` (DATE 2015 Eq. (1) / IPDPS 2015 Eq. (3)).
    fn dbf_star(&self, t: u64) -> f64 {
        if t < self.deadline {
            0.0
        } else {
            self.volume as f64 + self.utilization() * (t - self.deadline) as f64
        }
    }
}

/// Which per-processor admission condition to apply.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Condition {
    /// `C_i <= D_i - Σ_{j on k} DBF*(τ_j, D_i)` (DATE 2015 Fig. 4 line 3,
    /// Baruah & Fisher 2006).
    Demand,
    /// The condition above plus `u_i <= 1 - Σ_{j on k} u_j` (IPDPS 2015
    /// Fig. 4 line 3, for arbitrary deadlines).
    DemandAndUtilization,
}

/// How to pick among the processors that pass [`Condition`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PackingStrategy {
    /// Lowest-indexed processor that fits (Baruah's Fig. 4 loop over `k`).
    FirstFit,
    /// The fitting processor with the *largest* total utilization, i.e.
    /// the least remaining capacity (V-Fed's "best-fit packing strategy";
    /// the paper doesn't define the fullness measure, utilization is the
    /// standard one for EDF bins). Ties go to the lowest index.
    BestFit,
    /// The fitting processor with the smallest total utilization.
    WorstFit,
}

/// Tasks already assigned to one processor.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Bin {
    tasks: Vec<SeqTask>,
}

impl Bin {
    pub fn utilization(&self) -> f64 {
        self.tasks.iter().map(SeqTask::utilization).sum()
    }

    pub fn tasks(&self) -> &[SeqTask] {
        &self.tasks
    }

    /// Whether `task` may join this bin under `condition`. Assumes every
    /// task already here has a relative deadline no larger than `task`'s
    /// (see the module doc).
    pub fn fits(&self, task: &SeqTask, condition: Condition) -> bool {
        const EPS: f64 = 1e-9;
        let demand: f64 = self.tasks.iter().map(|t| t.dbf_star(task.deadline)).sum();
        if task.volume as f64 > task.deadline as f64 - demand + EPS {
            return false;
        }
        match condition {
            Condition::Demand => true,
            Condition::DemandAndUtilization => task.utilization() <= 1.0 - self.utilization() + EPS,
        }
    }
}

/// Assign `tasks` to at most `max_bins` processors. Returns, for each task
/// in `tasks`' *original* order, the index of the processor it was placed
/// on, plus the bins themselves; `None` if some task fits nowhere (the
/// papers' `return FAILURE`).
pub fn partition(
    tasks: &[SeqTask],
    max_bins: u16,
    condition: Condition,
    strategy: PackingStrategy,
) -> Option<(Vec<usize>, Vec<Bin>)> {
    let mut order: Vec<usize> = (0..tasks.len()).collect();
    order.sort_by_key(|&i| tasks[i].deadline);

    // All `max_bins` processors exist from the start (the papers partition
    // onto a fixed `m_r`/`M_l` processors); an unused one is simply empty.
    let mut bins: Vec<Bin> = alloc::vec![Bin::default(); max_bins as usize];
    let mut placement = alloc::vec![0usize; tasks.len()];

    for i in order {
        let task = &tasks[i];
        let mut fitting = bins
            .iter()
            .enumerate()
            .filter(|(_, b)| b.fits(task, condition));
        let chosen = match strategy {
            PackingStrategy::FirstFit => fitting.next().map(|(k, _)| k),
            PackingStrategy::BestFit => fitting
                .fold(None, |best: Option<(usize, f64)>, (k, b)| {
                    let u = b.utilization();
                    match best {
                        Some((_, bu)) if bu >= u => best,
                        _ => Some((k, u)),
                    }
                })
                .map(|(k, _)| k),
            PackingStrategy::WorstFit => fitting
                .fold(None, |best: Option<(usize, f64)>, (k, b)| {
                    let u = b.utilization();
                    match best {
                        Some((_, bu)) if bu <= u => best,
                        _ => Some((k, u)),
                    }
                })
                .map(|(k, _)| k),
        }?;
        bins[chosen].tasks.push(*task);
        placement[i] = chosen;
    }

    Some((placement, bins))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(volume: u64, deadline: u64, period: u64) -> SeqTask {
        SeqTask {
            volume,
            deadline,
            period,
        }
    }

    #[test]
    fn test_dbf_star_matches_definition() {
        let a = t(2, 5, 10);
        assert_eq!(a.dbf_star(4), 0.0);
        assert_eq!(a.dbf_star(5), 2.0);
        // 2 + 0.2 * (15 - 5) = 4
        assert!((a.dbf_star(15) - 4.0).abs() < 1e-12);
    }

    #[test]
    fn test_implicit_deadline_demand_reduces_to_utilization() {
        // D = T: DBF*(τ_j, D_i) = u_j * D_i, so the Demand condition is
        // exactly Σu <= 1.
        let bin = Bin {
            tasks: alloc::vec![t(3, 10, 10), t(4, 10, 10)],
        };
        assert!(bin.fits(&t(3, 10, 10), Condition::Demand)); // Σu = 1.0
        assert!(!bin.fits(&t(4, 10, 10), Condition::Demand)); // Σu = 1.1
    }

    #[test]
    fn test_demand_is_less_pessimistic_than_density() {
        // τ_a = (2, 4, 100): δ = 0.5; τ_b = (6, 10, 100): δ = 0.6, so a
        // density-sum test (0.5 + 0.6 = 1.1 > 1) would reject the pair, but
        // DBF*(τ_a, 10) = 2 + 0.02 * 6 = 2.12 and 6 <= 10 - 2.12 -> fits.
        let bin = Bin {
            tasks: alloc::vec![t(2, 4, 100)],
        };
        assert!(bin.fits(&t(6, 10, 100), Condition::Demand));
    }

    #[test]
    fn test_utilization_condition_rejects_overloaded_arbitrary_deadline() {
        // Arbitrary deadlines (D > T): demand alone can pass while the
        // long-run utilization already exceeds 1.
        let bin = Bin {
            tasks: alloc::vec![t(6, 20, 10)], // u = 0.6
        };
        let new = t(5, 100, 10); // u = 0.5; DBF*(old, 100) = 6 + 0.6*80 = 54 <= 95
        assert!(bin.fits(&new, Condition::Demand));
        assert!(!bin.fits(&new, Condition::DemandAndUtilization));
    }

    #[test]
    fn test_partition_sorts_by_deadline_and_reports_original_order() {
        let tasks = [t(5, 10, 10), t(5, 10, 10), t(1, 2, 10)];
        let (placement, bins) =
            partition(&tasks, 2, Condition::Demand, PackingStrategy::FirstFit).unwrap();
        // D=2 task placed first on bin 0; then the two D=10 tasks: bin 0
        // has DBF*(τ(1,2,10), 10) = 1 + 0.1*8 = 1.8, 5 <= 10 - 1.8 fits;
        // second: 1.8 + 5 = 6.8, 5 > 3.2 -> bin 1.
        assert_eq!(placement, alloc::vec![0, 1, 0]);
        assert_eq!(bins[0].tasks().len(), 2);
    }

    #[test]
    fn test_partition_fails_when_out_of_bins() {
        let tasks = [t(6, 10, 10), t(6, 10, 10)];
        assert!(partition(&tasks, 1, Condition::Demand, PackingStrategy::FirstFit).is_none());
        assert!(partition(&tasks, 2, Condition::Demand, PackingStrategy::FirstFit).is_some());
    }

    #[test]
    fn test_best_fit_picks_fullest_fitting_bin() {
        let tasks = [t(2, 10, 10), t(9, 10, 10), t(3, 10, 10)];
        let (placement, _) =
            partition(&tasks, 3, Condition::Demand, PackingStrategy::BestFit).unwrap();
        // t(2): all empty -> bin 0. t(9): bin 0 has 0.2, 0.9 fits? 0.2+0.9>1
        // no -> bin 1 (empty, u=0). t(3): bin 0 (0.2) fits, bin 1 (0.9) no,
        // bin 2 (0) fits -> fullest fitting is bin 0.
        assert_eq!(placement, alloc::vec![0, 1, 0]);
    }
}
