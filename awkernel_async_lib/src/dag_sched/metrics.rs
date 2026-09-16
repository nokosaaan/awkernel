//! DAG-level metrics shared by every admission policy.

/// Where a [`DagMetrics`]'s `volume`/`critical_path` came from.
///
/// Never affects any policy's admission math — every policy reads the same
/// four numbers either way — it only records provenance, e.g. so a caller
/// can avoid presenting a `Static` (`rd_gen`-sourced) `L` as if it still
/// meant something after admission, when in fact it never does again
/// ([`Static`]'s only job is being an input to admission).
///
/// [`Static`]: MetricsSource::Static
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetricsSource {
    /// Computed ahead of time from a fully-known static DAG structure, e.g.
    /// a topological-sort DP over per-node WCET declared in an `rd_gen`
    /// YAML file (see `rd_gen_to_dags::dag_stats::compute_dag_stats`).
    Static,
    /// A human's own estimate today; once a live-measurement source exists
    /// for DAGs whose structure isn't known ahead of admission, that would
    /// also produce this variant.
    Measured,
}

/// Everything a DAG admission policy needs to decide a DAG's nodes'
/// [`crate::scheduler::SchedulerType`], gathered into one config. Build it
/// with [`DagMetrics::from_static`] (a priori known structure) or
/// [`DagMetrics::from_measured`] (a human-supplied estimate); both produce
/// the identical type, so a policy never has to know which path a value took
/// to get here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DagMetrics {
    /// `C`: total WCET volume, i.e. the sum of every node's WCET.
    pub volume: u64,
    /// `L`: critical-path length, i.e. the WCET sum along the longest
    /// source-to-sink path. Only ever consulted during admission; it has no
    /// meaning afterwards.
    pub critical_path: u64,
    /// `T`: period. Drives Federated's heavy/light classification
    /// (`u = C/T`); other policies may read it differently.
    pub period: u64,
    /// `D`: relative deadline. In the same time unit as
    /// `volume`/`critical_path`/`period`, and as
    /// [`crate::scheduler::SchedulerType::GEDF`]/[`crate::scheduler::SchedulerType::ClusteredEDF`]'s
    /// own `relative_deadline` parameter (the scheduler layer treats it as
    /// an opaque `u64`; callers must stay consistent, exactly as those two
    /// variants already require).
    pub relative_deadline: u64,
    /// A task's own maximum degree of parallelism (how many nodes can ever
    /// be eligible at once) — *not* the same quantity as `critical_path`,
    /// despite both being called `L` in the literature. Only consulted by
    /// [`super::policy::vfed`]'s OURS2 (PSF/`Π'`) refinement, to prove some
    /// of a heavy DAG's own active-VPs are structurally always idle; every
    /// other policy, and `vfed`'s own OURS1 path, ignores this field
    /// entirely. `u16::MAX` (the value [`DagMetrics::from_static`]/
    /// [`DagMetrics::from_measured`] set) means "unknown" — `vfed` then
    /// falls back to its OURS1 behavior for this DAG, exactly as if this
    /// field didn't exist. Set it explicitly (struct-update syntax; every
    /// field here is `pub`) when it's known, e.g. an offline evaluation
    /// harness drawing it the same way the V-Fed paper's own evaluation
    /// does: `Uniform[ceil(m_i/2), m_i]` where `m_i` is this same DAG's
    /// [`DagMetrics::min_dedicated_cores`].
    pub max_parallelism: u16,
    pub source: MetricsSource,
}

impl DagMetrics {
    /// Build a config from a priori-known values, e.g. rd_gen_to_dags's
    /// `compute_dag_stats` over a YAML-declared DAG structure.
    pub const fn from_static(
        volume: u64,
        critical_path: u64,
        period: u64,
        relative_deadline: u64,
    ) -> Self {
        Self {
            volume,
            critical_path,
            period,
            relative_deadline,
            max_parallelism: u16::MAX,
            source: MetricsSource::Static,
        }
    }

    /// Build a config from a human-supplied estimate (today) or a future
    /// live measurement (once that infrastructure exists), for a DAG whose
    /// full structure isn't known ahead of admission.
    pub const fn from_measured(
        volume: u64,
        critical_path: u64,
        period: u64,
        relative_deadline: u64,
    ) -> Self {
        Self {
            volume,
            critical_path,
            period,
            relative_deadline,
            max_parallelism: u16::MAX,
            source: MetricsSource::Measured,
        }
    }

    pub const fn is_measured(&self) -> bool {
        matches!(self.source, MetricsSource::Measured)
    }

    /// `m = ceil((C - L) / (D - L))`, clamped to at least 1: a DAG whose
    /// volume equals its critical path has no exploitable parallelism, but
    /// still needs one dedicated core to run on. Returns `None` if
    /// `D <= L` (infeasible — no core count can help once the critical path
    /// alone already exceeds the deadline) or if the result does not fit a
    /// `u16` (unreachable in practice: bounded by `NUM_MAX_CPU`).
    ///
    /// Shared by every admission policy that dedicates a whole cluster to a
    /// heavy DAG ([`super::policy::federated`], [`super::policy::vfed`]):
    /// both derive the same minimum core count from the same three numbers,
    /// so it lives here once instead of drifting apart in two copies.
    pub fn min_dedicated_cores(&self) -> Option<u16> {
        let numerator = self.volume.checked_sub(self.critical_path)?;
        let denominator = self.relative_deadline.checked_sub(self.critical_path)?;
        if denominator == 0 {
            return None;
        }
        let cores = numerator.div_ceil(denominator).max(1);
        u16::try_from(cores).ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_measured_and_from_static_differ_only_in_source() {
        let static_config = DagMetrics::from_static(100, 20, 50, 50);
        let measured_config = DagMetrics::from_measured(100, 20, 50, 50);

        assert!(!static_config.is_measured());
        assert!(measured_config.is_measured());
        assert_eq!(static_config.volume, measured_config.volume);
        assert_eq!(static_config.critical_path, measured_config.critical_path);
        assert_eq!(static_config.period, measured_config.period);
        assert_eq!(
            static_config.relative_deadline,
            measured_config.relative_deadline
        );
    }
}
