//! The common shape an admission policy's resource decision is expressed in,
//! before [`Provision::into_scheduler_type`] turns it into a concrete
//! [`crate::scheduler::SchedulerType`].
//!
//! Every admission policy implemented so far ends up deciding one of two
//! resource shapes: a DAG either gets an exclusive set of cores, or it
//! shares the rest of the pool and is tracked only by how much utilization
//! it consumes. Expressing that decision as `Provision` rather than directly
//! as a `SchedulerType` keeps the `SchedulerType` <-> resource-shape mapping
//! in one place instead of duplicated in every policy module.

use awkernel_lib::cpu::CpuSet;

use crate::scheduler::SchedulerType;

/// A DAG admission policy's resource decision, independent of which
/// `SchedulerType` ultimately implements it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Provision {
    /// An exclusive cluster of cores, e.g. a Federated heavy DAG.
    Dedicated { cores: CpuSet },
    /// A share of the remaining pool, tracked only by aggregate
    /// utilization (scaled by a fixed factor internal to
    /// [`super::resource`]), e.g. a Federated light DAG.
    Shared { utilization_scaled: u64 },
}

impl Provision {
    /// Turn this provision into the `SchedulerType` every node of the DAG it
    /// was decided for should be registered with.
    pub const fn into_scheduler_type(self, relative_deadline: u64) -> SchedulerType {
        match self {
            Provision::Dedicated { cores } => SchedulerType::ClusteredEDF(relative_deadline, cores),
            Provision::Shared { .. } => SchedulerType::GEDF(relative_deadline),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dedicated_becomes_clustered_edf() {
        let cores = CpuSet::empty().with(1).with(2);
        let provision = Provision::Dedicated { cores };
        assert!(matches!(
            provision.into_scheduler_type(50),
            SchedulerType::ClusteredEDF(50, set) if set == cores
        ));
    }

    #[test]
    fn test_shared_becomes_gedf() {
        let provision = Provision::Shared {
            utilization_scaled: 500_000,
        };
        assert!(matches!(
            provision.into_scheduler_type(50),
            SchedulerType::GEDF(50)
        ));
    }
}
