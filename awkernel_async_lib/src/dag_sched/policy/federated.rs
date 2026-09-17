//! Federated Scheduling admission policy (Li et al., RTSS 2014, generalized
//! to constrained/arbitrary-deadline DAGs by Baruah) for DAG tasks.
//!
//! Federated Scheduling classifies a DAG by its *density* `C / min(D, T)`
//! (volume over the shorter of relative deadline and period — see
//! [`density_window`]):
//! - **Heavy** (`density > 1`): given an exclusive cluster of `m` cores.
//! - **Light** (`density <= 1`): shares the remaining cores with other light
//!   DAGs, admitted only while the sum of every admitted light DAG's density
//!   still fits the pool (see [`resource::reserve_light_utilization`]) — a
//!   heavy DAG's theorem-backed core count is worthless if the light side is
//!   silently oversubscribed instead.
//!
//! Li et al.'s original design only covers implicit-deadline DAGs (`D = T`),
//! where density and plain utilization `u = C/T` coincide. For a
//! constrained-deadline DAG (`D < T`), using `u = C/T` alone would
//! under-count a DAG whose deadline is tighter than its period: it could
//! read as comfortably light by utilization while still needing an entire
//! core continuously to finish within its shorter deadline window. Using
//! `min(D, T)` throughout (density) instead of `T` alone (utilization)
//! covers both the classical implicit case and Baruah's constrained/
//! arbitrary-deadline generalization with the same formula, since `D >= T`
//! reduces `min(D, T)` back to `T`.
//!
//! This module is admission-time only. It does not add a new run queue or a
//! new [`crate::scheduler::Scheduler`] impl: [`SchedulerType::ClusteredEDF`]
//! already implements "EDF restricted to a `CpuSet`" (cluster reservation,
//! preemption, the lot), so a heavy DAG's cluster is simply a `ClusteredEDF`
//! cpu_set; a light DAG uses plain [`SchedulerType::GEDF`] — both computed
//! by [`Provision::into_scheduler_type`]. Both already compute a
//! per-job-instance absolute deadline for every node of a DAG via
//! [`crate::dag::calculate_and_update_dag_deadline`], so this module does
//! not duplicate that logic — it only decides *which* scheduler and *which*
//! cores.
//!
//! [`admit_dag`] is the single entry point, taking one [`DagMetrics`]. That
//! config is deliberately the *only* thing admission looks at, so a caller
//! with a fully-known static DAG structure (e.g. `rd_gen_to_dags`, via
//! [`DagMetrics::from_static`]) and a caller who can only supply a human
//! estimate for a dynamically-built DAG (via [`DagMetrics::from_measured`])
//! go through the exact same admission logic — neither path is a special
//! case of the other.

use crate::{
    dag_sched::{
        metrics::{DagMetrics, MetricsSource},
        provision::Provision,
        resource::{self, ResourceError},
    },
    scheduler::SchedulerType,
};

/// Result of [`classify_dag`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskClass {
    Light,
    Heavy { required_cores: u16 },
}

/// The [`Provision`] (and, from it, the `SchedulerType`) that [`admit_dag`]
/// decided a DAG's nodes should use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FederatedAssignment {
    pub class: TaskClass,
    /// The resource shape this DAG was admitted with, independent of which
    /// `SchedulerType` implements it.
    pub provision: Provision,
    pub scheduler_type: SchedulerType,
    /// Carried over from the [`DagMetrics`] that produced this assignment,
    /// so a caller can log/display it without keeping the original config
    /// around.
    pub source: MetricsSource,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FederatedError {
    /// `relative_deadline < critical_path` (traversing the critical path
    /// alone already exceeds the deadline, so no core count can help), or
    /// `relative_deadline == critical_path` with exploitable parallelism
    /// (`volume > critical_path`, i.e. Heavy): the implicit-deadline
    /// boundary `D == L` is only feasible when `volume == critical_path`
    /// (a purely sequential DAG needs exactly one core to run its own
    /// critical path in `D == L` time; see [`classify_dag`]'s `is_heavy`
    /// check, which lets that case through as Light), never when there is
    /// other work to also finish in a now-zero slack window.
    Infeasible {
        critical_path: u64,
        relative_deadline: u64,
    },
    /// The shared core/utilization ledger could not satisfy this DAG's
    /// resource request; see [`ResourceError`].
    Resource(ResourceError),
}

impl From<ResourceError> for FederatedError {
    fn from(e: ResourceError) -> Self {
        FederatedError::Resource(e)
    }
}

impl core::fmt::Display for FederatedError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            FederatedError::Infeasible {
                critical_path,
                relative_deadline,
            } => write!(
                f,
                "relative_deadline({relative_deadline}) <= critical_path({critical_path}); no core count can meet this deadline"
            ),
            FederatedError::Resource(e) => write!(f, "{e}"),
        }
    }
}

/// `density = C / window`: a DAG is heavy iff its WCET volume exceeds
/// `window`, which the caller has already narrowed to `min(D, T)` (see
/// [`density_window`]) so this reduces to the classical `u = C/T` test for
/// an implicit/lenient (`D >= T`) DAG.
const fn is_heavy(volume: u64, window: u64) -> bool {
    volume > window
}

/// `min(D, T)`: the interval a DAG's volume must be spread over for the
/// heavy/light density test and the light-pool ledger to remain a valid
/// bound regardless of deadline model (see the module-level doc comment).
const fn density_window(config: &DagMetrics) -> u64 {
    if config.relative_deadline < config.period {
        config.relative_deadline
    } else {
        config.period
    }
}

/// Classify a DAG and, if heavy, compute its required core count. Does not
/// claim any cores; see [`resource::allocate_cluster`] / [`admit_dag`] for
/// that.
pub fn classify_dag(config: &DagMetrics) -> Result<TaskClass, FederatedError> {
    if config.relative_deadline < config.critical_path {
        return Err(FederatedError::Infeasible {
            critical_path: config.critical_path,
            relative_deadline: config.relative_deadline,
        });
    }

    if !is_heavy(config.volume, density_window(config)) {
        return Ok(TaskClass::Light);
    }

    let Some(required_cores) = config.min_dedicated_cores() else {
        return Err(FederatedError::Infeasible {
            critical_path: config.critical_path,
            relative_deadline: config.relative_deadline,
        });
    };

    Ok(TaskClass::Heavy { required_cores })
}

/// Admit a DAG: classify it and decide the `SchedulerType` every one of its
/// nodes should be registered with. Works identically whether `config` was
/// built via [`DagMetrics::from_static`] or [`DagMetrics::from_measured`].
///
/// - Heavy: claims its exclusive cluster from the shared ledger (release it
///   with [`resource::release_cluster`] once the DAG is torn down, if ever —
///   none of this test bed's DAGs currently are).
/// - Light: commits its utilization against the shared ledger (release it
///   with [`resource::release_light_utilization`] likewise).
pub fn admit_dag(config: DagMetrics) -> Result<FederatedAssignment, FederatedError> {
    match classify_dag(&config)? {
        TaskClass::Light => {
            let utilization_scaled =
                resource::reserve_light_utilization(config.volume, density_window(&config))?;
            let provision = Provision::Shared { utilization_scaled };
            Ok(FederatedAssignment {
                class: TaskClass::Light,
                scheduler_type: provision.into_scheduler_type(config.relative_deadline),
                provision,
                source: config.source,
            })
        }
        TaskClass::Heavy { required_cores } => {
            let cores = resource::allocate_cluster(required_cores)?;
            let provision = Provision::Dedicated { cores };
            Ok(FederatedAssignment {
                class: TaskClass::Heavy { required_cores },
                scheduler_type: provision.into_scheduler_type(config.relative_deadline),
                provision,
                source: config.source,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_heavy_boundary() {
        // u = C/T == 1 is Light (u <= 1), not Heavy.
        assert!(!is_heavy(100, 100));
        assert!(is_heavy(101, 100));
    }

    #[test]
    fn test_required_cores_formula() {
        // m = ceil((100 - 20) / (50 - 20)) = ceil(80 / 30) = 3
        let config = DagMetrics::from_static(100, 20, 1000, 50);
        assert_eq!(config.min_dedicated_cores(), Some(3));
    }

    #[test]
    fn test_required_cores_sequential_dag_clamped_to_one() {
        // A DAG with no exploitable parallelism (volume == critical_path)
        // still needs exactly one dedicated core.
        let config = DagMetrics::from_static(50, 50, 1000, 60);
        assert_eq!(config.min_dedicated_cores(), Some(1));
    }

    #[test]
    fn test_required_cores_infeasible_deadline() {
        assert_eq!(
            DagMetrics::from_static(100, 50, 1000, 50).min_dedicated_cores(),
            None
        ); // D <= L
        assert_eq!(
            DagMetrics::from_static(100, 50, 1000, 40).min_dedicated_cores(),
            None
        ); // D < L
    }

    #[test]
    fn test_classify_dag_infeasible_when_strictly_past_deadline() {
        // relative_deadline(40) < critical_path(50): infeasible regardless
        // of heaviness, since even the critical path alone can't finish.
        let config = DagMetrics::from_static(10, 50, 1000, 40);
        assert_eq!(
            classify_dag(&config),
            Err(FederatedError::Infeasible {
                critical_path: 50,
                relative_deadline: 40,
            })
        );
    }

    #[test]
    fn test_classify_dag_deadline_equals_critical_path_light_ok() {
        // relative_deadline == critical_path == volume: a purely sequential
        // DAG (no parallel-only work) fits its own critical path in exactly
        // D == L time on a single core -- feasible, not the unconditional
        // Infeasible case.
        let config = DagMetrics::from_static(50, 50, 1000, 50);
        assert_eq!(classify_dag(&config), Ok(TaskClass::Light));
    }

    #[test]
    fn test_classify_dag_deadline_equals_critical_path_heavy_infeasible() {
        // relative_deadline == critical_path but volume > critical_path:
        // there's parallel-only work left to fit in a now-zero slack
        // window, which no core count can do.
        let config = DagMetrics::from_static(100, 50, 1000, 50);
        assert_eq!(
            classify_dag(&config),
            Err(FederatedError::Infeasible {
                critical_path: 50,
                relative_deadline: 50,
            })
        );
    }

    #[test]
    fn test_classify_dag_light() {
        // density_window = min(D=90, T=100) = 90; volume(80) <= 90 => Light
        let config = DagMetrics::from_static(80, 20, 100, 90);
        assert_eq!(classify_dag(&config), Ok(TaskClass::Light));
    }

    #[test]
    fn test_classify_dag_heavy() {
        // D == T (implicit deadline): density_window = 50; volume(100) > 50 => Heavy
        let config = DagMetrics::from_static(100, 20, 50, 50);
        assert_eq!(
            classify_dag(&config),
            Ok(TaskClass::Heavy { required_cores: 3 }) // ceil((100-20)/(50-20)) = 3
        );
    }

    #[test]
    fn test_classify_dag_constrained_deadline_reclassifies_as_heavy() {
        // volume(60) <= period(100) => Light under plain utilization u=C/T
        // (0.6), but relative_deadline(50) < period(100) makes this a
        // constrained-deadline DAG whose density_window is 50, not 100:
        // density = 60/50 = 1.2 > 1 => Heavy. A DAG this tight cannot
        // actually be spread thin over the shared light pool just because
        // its long-run utilization looks low.
        let config = DagMetrics::from_static(60, 10, 100, 50);
        assert_eq!(
            classify_dag(&config),
            Ok(TaskClass::Heavy { required_cores: 2 }) // ceil((60-10)/(50-10)) = 2
        );
    }

    #[test]
    fn test_classify_dag_from_measured_matches_from_static() {
        // The two constructors differ only in `source`; admission math must
        // be identical either way, since a human-entered estimate and an
        // rd_gen-computed value are interchangeable inputs to the same
        // classification.
        let static_config = DagMetrics::from_static(100, 20, 50, 50);
        let measured_config = DagMetrics::from_measured(100, 20, 50, 50);

        assert!(!static_config.is_measured());
        assert!(measured_config.is_measured());
        assert_eq!(classify_dag(&static_config), classify_dag(&measured_config));
    }

    // Exercises resource::allocate_cluster/release_cluster/admit_dag
    // together in one test: they share the process-global ledger in
    // `dag_sched::resource`, and `cargo test` runs tests in parallel
    // threads, so splitting this across multiple #[test] fns would risk
    // cross-test interference.
    #[test]
    fn test_cluster_allocation_lifecycle() {
        unsafe {
            // workers 1..10 (9); core 9 is the regular pool, so the DAG pool
            // (1..9) has 8 cores available — matching the "8 available"
            // comments below.
            awkernel_lib::cpu::set_num_cpu(10);
        }

        let heavy_a = resource::allocate_cluster(3).unwrap();
        let heavy_b = resource::allocate_cluster(3).unwrap();
        assert!(!heavy_a.contains(9) && !heavy_b.contains(9)); // never claims the regular-pool core
        assert_eq!(heavy_a.union(heavy_b).iter().count(), 6); // disjoint: 3 + 3 distinct cores
        for cpu in heavy_a.iter() {
            assert!(!heavy_b.contains(cpu));
        }

        // Only 2 workers remain (8 - 3 - 3); a third 3-core cluster must fail.
        match resource::allocate_cluster(3) {
            Err(ResourceError::InsufficientCores {
                required: 3,
                available: 2,
            }) => {}
            other => {
                panic!("expected InsufficientCores{{required: 3, available: 2}}, got {other:?}")
            }
        }

        resource::release_cluster(heavy_a);
        let heavy_c = resource::allocate_cluster(3).unwrap();
        assert!(heavy_c.iter().all(|cpu| !heavy_b.contains(cpu)));

        // admit_dag end-to-end: Light gets GEDF, Heavy gets a fresh cluster.
        // density_window = min(D=50, T=100) = 50, so admit_dag reserves
        // utilization against 50, not 100 — release must match.
        let light = admit_dag(DagMetrics::from_static(10, 5, 100, 50)).unwrap();
        assert_eq!(light.class, TaskClass::Light);
        assert_eq!(light.source, MetricsSource::Static);
        assert!(matches!(light.provision, Provision::Shared { .. }));
        assert!(matches!(light.scheduler_type, SchedulerType::GEDF(50)));
        resource::release_light_utilization(10, 50); // undo admit_dag's reservation before the next scenario

        resource::release_cluster(heavy_b);
        resource::release_cluster(heavy_c);

        // Light-pool oversubscription: claim 7 of the 8 workers for a heavy
        // cluster, leaving exactly 1 core (100% = 1_000_000 scaled) for the
        // light pool.
        let heavy_big = resource::allocate_cluster(7).unwrap();

        resource::reserve_light_utilization(60, 100).unwrap(); // u = 0.6, committed = 600_000

        match resource::reserve_light_utilization(50, 100) {
            // u = 0.5 => additional 500_000; 600_000 + 500_000 > 1_000_000
            Err(ResourceError::LightPoolOversubscribed {
                additional_utilization_scaled: 500_000,
                available_capacity_scaled: 400_000,
            }) => {}
            other => panic!(
                "expected LightPoolOversubscribed{{additional: 500_000, available: 400_000}}, got {other:?}"
            ),
        }

        resource::release_light_utilization(60, 100); // frees the 600_000 back up

        // Now the same 0.5-utilization DAG fits, this time via a
        // human-supplied estimate rather than a static rd_gen value.
        let heavy_config = DagMetrics::from_measured(50, 10, 100, 90);
        assert!(matches!(
            classify_dag(&heavy_config).unwrap(),
            TaskClass::Light
        ));
        resource::reserve_light_utilization(50, 100).unwrap();
        resource::release_light_utilization(50, 100);

        resource::release_cluster(heavy_big);
    }
}
