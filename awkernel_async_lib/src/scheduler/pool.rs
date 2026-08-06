//! The DAG-pool / regular-pool worker CPU split.
//!
//! One worker core is carved out for regular (non-DAG) tasks — the shell,
//! driver services, and the like — so that a DAG-side scheduler (currently
//! GEDF) is never delayed by interference its admission math has no way to
//! account for, and conversely so that shell/service latency is never at the
//! mercy of DAG-to-DAG contention.
//!
//! This is pure CPU-id arithmetic: it holds no shared state and does not
//! depend on which admission policy (Federated or otherwise) is deciding
//! `SchedulerType`s. It used to live in `federated.rs`, but every DAG
//! scheduler needs it regardless of admission policy, and `scheduler.rs`
//! (dispatch) / `gedf.rs`, `prioritized_fifo.rs`, `prioritized_rr.rs`
//! (preemption target filtering) all need to call it without depending on a
//! specific admission policy module.

use awkernel_lib::cpu::num_cpu;

/// Whether the DAG pool / regular pool split (see [`is_dag_pool_core`] /
/// [`is_regular_pool_core`]) is in effect. Splitting off one core needs at
/// least 2 worker cores to leave anything for the DAG side, so systems with
/// only 1 worker (`num_cpu() < 3`) fall back to every worker being eligible
/// for both — i.e. today's shared-pool behavior, not a broken one.
pub(crate) fn dag_pool_split_active() -> bool {
    num_cpu() >= 3
}

/// True if `cpu_id` may run DAG-pool (GEDF) work.
///
/// The last worker core is carved out for regular (non-DAG) tasks — the
/// shell, driver services, and the like — so that a DAG's admission math
/// (e.g. Federated's `u = C/T`) is never invalidated by interference it has
/// no way to model, since none of that analysis accounts for shell/service
/// load. See [`super::get_next_task`] for where this gates dispatch, and
/// [`crate::dag::calculate_and_update_dag_deadline`]'s callers (`gedf.rs`'s
/// `invoke_preemption`) for where it gates preemption targets.
pub(crate) fn is_dag_pool_core(cpu_id: usize) -> bool {
    cpu_id != 0 && (!dag_pool_split_active() || cpu_id != num_cpu() - 1)
}

/// True if `cpu_id` may run regular (non-DAG) work: the complement of
/// [`is_dag_pool_core`] among worker cores while the split is active, and
/// (like it) true everywhere while the split is inactive.
pub(crate) fn is_regular_pool_core(cpu_id: usize) -> bool {
    cpu_id != 0 && (!dag_pool_split_active() || cpu_id == num_cpu() - 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dag_pool_split() {
        unsafe {
            awkernel_lib::cpu::set_num_cpu(10); // workers 1..10; last (9) is the regular-pool core
        }
        assert!(dag_pool_split_active());
        assert!(is_regular_pool_core(9));
        assert!(!is_dag_pool_core(9));
        for cpu in 1..9 {
            assert!(is_dag_pool_core(cpu), "cpu {cpu} should be in the DAG pool");
            assert!(
                !is_regular_pool_core(cpu),
                "cpu {cpu} should not be the regular-pool core"
            );
        }

        // Too few workers to split: every worker is eligible for both pools
        // (today's shared-pool behavior), not eligible for neither.
        unsafe {
            awkernel_lib::cpu::set_num_cpu(2); // 1 worker only
        }
        assert!(!dag_pool_split_active());
        assert!(is_dag_pool_core(1));
        assert!(is_regular_pool_core(1));
    }
}
