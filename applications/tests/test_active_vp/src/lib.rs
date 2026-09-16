#![no_std]

//! Smoke test for V-Fed's scheduling mechanism: admits heavy DAG-shaped
//! configs via `dag_sched::policy::vfed::admit_one`, converts the resulting
//! `VFedAssignment` into a `SchedulerType` via `into_scheduler_type`, and
//! spawns tasks with it directly (mirroring how `test_clustered_edf`
//! exercises `ClusteredEDF` without a full DAG/rd_gen pipeline) to observe
//! real dispatch behavior under QEMU.
//!
//! [`test_mixed_vp_group`] is the main scenario: a donor DAG gets a pure
//! active-VP group, then a consumer DAG that doesn't fit in the cores left
//! over gets a *mixed* group (Theorem 4) — its own partial active-VP core
//! plus the donor's leftover passive-VP — exercising `scheduler::mixed_vp`
//! end to end (does it actually dispatch on both halves, does the borrowed
//! core correctly track the donor's busy/idle state, does it complete
//! cleanly with no hang). The pure active-VP path (no passive-VP top-up)
//! was validated the same way earlier and is exercised again here
//! incidentally: the donor itself is a plain `SchedulerType::ActiveVp`
//! group.

extern crate alloc;

use awkernel_async_lib::{
    dag_sched::policy::vfed::{self, PackingStrategy, VFedAssignment},
    dag_sched::metrics::DagMetrics,
    scheduler::SchedulerType,
    spawn,
};
use awkernel_lib::{
    cpu::{cpu_id, num_cpu, CpuSet},
    delay::{uptime, wait_microsec},
};

pub async fn run() {
    wait_microsec(2_000_000);

    // DAG-pool worker cores (`resource::free_core_count()`'s domain) are
    // `num_cpu() - 2` (cpu 0 is the primary core; the *last* worker is
    // carved out for the regular pool — see `scheduler::pool`).
    // `test_mixed_vp_group` needs exactly 3 of them: 2 for a donor's full
    // active-VP group, 1 for the consumer's partial one topped up by the
    // donor's leftover passive-VP — so `num_cpu() >= 5`. Deliberately not
    // more: the consumer also needs 2 for a *full* group of its own, so if
    // more than 1 dag-pool core were left over after the donor, admission
    // would give it a full group instead of the partial-plus-passive-VP
    // (Theorem 4) case this test exists to exercise.
    //
    // Note: `userland`'s own `default = ["rd_gen_to_dags"]` runs a demo DAG
    // before this module starts, admitted through the same
    // `dag_sched::resource` ledger, that competes for the same dag-pool
    // cores (observed to scale its own claim with however many are free,
    // rather than a fixed count) — build with `default-features = false`
    // on the `userland` dependency (kernel/Cargo.toml) to get this test a
    // deterministic core budget instead of racing that demo for one.
    if num_cpu() != 5 {
        log::warn!(
            "test_active_vp: requires exactly 5 CPUs (smp=5) so the consumer's admission is forced \
             into the partial-plus-passive-VP case (see the comment above), skipping"
        );
        return;
    }

    test_control_clustered_edf_same_priority_churn().await;
    test_mixed_vp_group().await;

    wait_microsec(100_000_000);
}

/// Control experiment using the pre-existing, unmodified `ClusteredEDF`
/// scheduler (no V-Fed code involved at all): two plain tasks, no
/// `DagInfo`, sharing one 2-core cpu_set. With no `DagInfo` each wake
/// computes `absolute_deadline = wake_time` (see `ClusteredEDFScheduler::wake_task`'s
/// fallback), so whichever task woke most recently always looks most
/// urgent, causing the two tasks to repeatedly preempt each other for the
/// same core — isolates whether a stall under that churn is specific to
/// `scheduler::active_vp` or a property of the shared
/// `invoke_preemption`/`do_preemption`/`PREEMPTED_TASKS` machinery
/// regardless of which scheduler drives it.
async fn test_control_clustered_edf_same_priority_churn() {
    log::info!("=== test_control_clustered_edf_same_priority_churn start ===");
    // Single core, forcing the two tasks to actually contend for it (the
    // two-core version never needed to preempt each other at all — each
    // task just settled onto its own free core).
    let cluster = CpuSet::empty().with(1);
    for name in ["control_a", "control_b"] {
        spawn(
            name.into(),
            async move {
                for i in 0..30 {
                    log::info!(
                        "test_active_vp: CONTROL {name} iter={i} cpu={} t={}",
                        cpu_id(),
                        uptime()
                    );
                    wait_microsec(1_000);
                    awkernel_async_lib::r#yield().await;
                }
                log::info!("test_active_vp: CONTROL {name} finished [OK]");
            },
            SchedulerType::ClusteredEDF(1_000_000, cluster),
        )
        .await;
    }
    wait_microsec(10_000_000);
    log::info!("=== test_control_clustered_edf_same_priority_churn done ===");
}

/// Smoke test for V-Fed's *mixed* active+passive path (Theorem 4). Requires
/// exactly 4 dag-pool worker cores (`num_cpu() == 5`, see the cpu-count
/// guard in [`run`]): the donor's full 2-core group must leave *exactly* 1
/// dag-pool core free, so the consumer (which needs 2 for a full group of
/// its own) is forced into the partial-plus-passive-VP case this test
/// exists to exercise, rather than happening to still fit a full group.
/// - Donor: C=500000, L=100000, T=600000, D=300000 — admitted first here
///   so it claims a full 2-core active-VP group (leading budget 300000,
///   non-leading 200000), contributing both cores to the shared
///   passive-VP pool.
/// - Consumer: C=260000, L=10000, T=250000, D=250000 — heavy
///   (density ≈ 1.04), needs `ceil((C-L)/(D-L)) = 2` cores for a full
///   group, but only 1 dag-pool core is free after the donor claims 2, so
///   `plan_heavy` gives it a *partial* 1-core group
///   (leading only, budget = D = 250000) and tops up the `260000-250000 =
///   10000` shortfall from the donor's *non-leading* passive-VP (its
///   leading one's own `sbf` is 0 at any delta <= its own budget, so it is
///   never useful — see `sbf`'s doc/tests): `sbf(250000, 200000, 600000,
///   300000) = 50000`, minus the consumer's own critical path (10000) =
///   40000 of usable supply, comfortably covering the 10000 shortfall.
///   Verified numerically against this module's own `plan_heavy`/`sbf`
///   before picking these constants (see the `scheduler::mixed_vp` module
///   doc for the mechanism this exercises).
async fn test_mixed_vp_group() {
    log::info!("=== test_mixed_vp_group start ===");

    let donor_config = DagMetrics::from_static(500_000, 100_000, 600_000, 300_000);
    let donor_assignment = match vfed::admit_one(donor_config, PackingStrategy::FirstFit) {
        Ok(a) => a,
        Err(e) => {
            log::error!("test_active_vp: mixed: donor admit_one failed: {e:?} [FAIL]");
            return;
        }
    };
    let VFedAssignment::Heavy {
        active: donor_active,
        passive: donor_passive,
    } = &donor_assignment
    else {
        log::error!("test_active_vp: mixed: expected donor Heavy assignment, got {donor_assignment:?} [FAIL]");
        return;
    };
    if !donor_passive.is_empty() || donor_active.len() != 2 {
        log::error!(
            "test_active_vp: mixed: expected donor to get a pure 2-core group, got {donor_assignment:?} [FAIL]"
        );
        return;
    }
    let donor_leading_cpu = donor_active[0].cpu;
    let donor_non_leading_cpu = donor_active[1].cpu;
    log::info!(
        "test_active_vp: mixed: donor admitted as pure ActiveVp, leading={donor_leading_cpu} non_leading={donor_non_leading_cpu} [OK]"
    );
    let Some(donor_sched_type) = donor_assignment.into_scheduler_type(donor_config.relative_deadline)
    else {
        log::error!("test_active_vp: mixed: donor into_scheduler_type returned None [FAIL]");
        return;
    };

    let consumer_config = DagMetrics::from_static(260_000, 10_000, 250_000, 250_000);
    let consumer_assignment = match vfed::admit_one(consumer_config, PackingStrategy::FirstFit) {
        Ok(a) => a,
        Err(e) => {
            log::error!("test_active_vp: mixed: consumer admit_one failed: {e:?} [FAIL]");
            return;
        }
    };
    let VFedAssignment::Heavy {
        active: consumer_active,
        passive: consumer_passive,
    } = &consumer_assignment
    else {
        log::error!("test_active_vp: mixed: expected consumer Heavy assignment, got {consumer_assignment:?} [FAIL]");
        return;
    };
    if consumer_active.len() != 1 || consumer_passive.len() != 1 {
        log::error!(
            "test_active_vp: mixed: expected consumer to get a partial 1-core group topped up by 1 passive-VP, got {consumer_assignment:?} [FAIL]"
        );
        return;
    }
    if consumer_passive[0].cpu != donor_non_leading_cpu {
        log::error!(
            "test_active_vp: mixed: expected consumer's passive-VP to be the donor's non-leading core {donor_non_leading_cpu}, got {} [FAIL]",
            consumer_passive[0].cpu
        );
        return;
    }
    log::info!(
        "test_active_vp: mixed: consumer admitted as Heavy+passive-top-up: active={:?} passive={:?} [OK]",
        consumer_active.iter().map(|a| a.cpu).collect::<alloc::vec::Vec<_>>(),
        consumer_passive.iter().map(|p| p.cpu).collect::<alloc::vec::Vec<_>>(),
    );

    let Some(consumer_sched_type) =
        consumer_assignment.into_scheduler_type(consumer_config.relative_deadline)
    else {
        log::error!("test_active_vp: mixed: consumer into_scheduler_type returned None [FAIL]");
        return;
    };
    let SchedulerType::MixedVp(active_set, leading, passive_set) = consumer_sched_type else {
        log::error!(
            "test_active_vp: mixed: expected SchedulerType::MixedVp, got {consumer_sched_type:?} [FAIL]"
        );
        return;
    };
    log::info!(
        "test_active_vp: mixed: SchedulerType::MixedVp(active={active_set:?}, leading={leading}, passive={passive_set:?})"
    );

    // The donor keeps its own leading+non-leading cores busy for a while —
    // long enough that the consumer's early attempts to borrow the
    // non-leading core should observe it busy (see `passive_vp`'s
    // `active_vp_is_busy` gate) — then finishes, freeing it for the
    // consumer to actually use.
    spawn(
        "donor_vertex".into(),
        async move {
            for i in 0..10 {
                log::info!(
                    "test_active_vp: mixed: donor_vertex iter={i} cpu={} t={}",
                    cpu_id(),
                    uptime()
                );
                wait_microsec(1_000);
                awkernel_async_lib::r#yield().await;
            }
            log::info!("test_active_vp: mixed: donor_vertex finished [OK]");
        },
        donor_sched_type,
    )
    .await;

    // Two consumer vertices sharing the mixed group: reachable on both the
    // consumer's own (budget/leading-gated) active core and the donor's
    // borrowed (busy/idle-gated) non-leading core.
    for name in ["mixed_a", "mixed_b"] {
        spawn(
            name.into(),
            async move {
                for i in 0..15 {
                    let cpu = cpu_id();
                    let role = if active_set.contains(cpu) {
                        if cpu == leading {
                            "own-leading"
                        } else {
                            "own-non-leading"
                        }
                    } else if passive_set.contains(cpu) {
                        "borrowed-passive"
                    } else {
                        "unexpected"
                    };
                    log::info!(
                        "test_active_vp: mixed: {name} iter={i} cpu={cpu} ({role}) t={}",
                        uptime()
                    );
                    wait_microsec(1_000);
                    awkernel_async_lib::r#yield().await;
                }
                log::info!("test_active_vp: mixed: {name} finished all iterations [OK]");
            },
            consumer_sched_type,
        )
        .await;
    }

    wait_microsec(10_000_000);
    log::info!("=== test_mixed_vp_group done ===");
}
