//! Per-task CPU time, and a busy loop that runs for a given amount of it
//! (the `cpu_time_work` feature).
//!
//! `awkernel_lib::busy_work::spin_work` performs a fixed *amount of work*,
//! calibrated per CPU at boot, so the CPU time it takes follows the core's
//! speed at that moment: on a VM whose host moves vCPU threads between
//! P-cores, E-cores and SMT siblings, the same work took 0.37x to 2.9x its
//! nominal time (2026-11-03). [`spin_cpu_time`] instead runs until the
//! calling task has been *on a CPU* for the requested time, so every node
//! uses exactly its execution time, whatever the core's speed. Like
//! `spin_work`, time spent preempted does not count, so a fluid rate theta
//! still stretches a node's wall-clock time.
//!
//! # Accounting
//! A task runs in slices: its poll in the run loop, or the pieces between
//! preemptions (`task::preempt`). Each slice start records, per CPU, the
//! task's CPU time so far and the slice's start time; each slice end adds
//! the elapsed time and stores the total in the task. Interrupts handled
//! during a slice count as the task's time (no separate IRQ accounting).

use array_macro::array;
use awkernel_lib::{cpu::NUM_MAX_CPU, interrupt::InterruptGuard};
use core::{
    sync::atomic::{AtomicBool, AtomicU64, Ordering},
    time::Duration,
};

/// How much CPU time [`spin_cpu_time`] works for between checks: the most
/// it can overshoot the requested time by.
const STEP: Duration = Duration::from_micros(10);

static IN_SLICE: [AtomicBool; NUM_MAX_CPU] = array![_ => AtomicBool::new(false); NUM_MAX_CPU];
static SLICE_START_NS: [AtomicU64; NUM_MAX_CPU] = array![_ => AtomicU64::new(0); NUM_MAX_CPU];
/// The running task's CPU time before the current slice.
static SLICE_BASE_NS: [AtomicU64; NUM_MAX_CPU] = array![_ => AtomicU64::new(0); NUM_MAX_CPU];

fn now_ns() -> u64 {
    u64::try_from(awkernel_lib::delay::uptime_nano()).unwrap_or(u64::MAX)
}

/// A slice of a task that has used `base_ns` of CPU time so far starts on
/// this CPU.
pub(crate) fn slice_start(base_ns: u64) {
    let cpu_id = awkernel_lib::cpu::cpu_id();
    SLICE_BASE_NS[cpu_id].store(base_ns, Ordering::Relaxed);
    SLICE_START_NS[cpu_id].store(now_ns(), Ordering::Relaxed);
    IN_SLICE[cpu_id].store(true, Ordering::Release);
}

/// The current slice on this CPU ends; returns the task's CPU time so far.
pub(crate) fn slice_end() -> u64 {
    let cpu_id = awkernel_lib::cpu::cpu_id();
    IN_SLICE[cpu_id].store(false, Ordering::Release);
    let start = SLICE_START_NS[cpu_id].load(Ordering::Relaxed);
    SLICE_BASE_NS[cpu_id]
        .load(Ordering::Relaxed)
        .saturating_add(now_ns().saturating_sub(start))
}

/// The calling task's CPU time so far, or `None` outside a task slice.
fn current_ns() -> Option<u64> {
    // A preemption between reading the CPU id and the slice must not move
    // this task to another CPU halfway.
    let _guard = InterruptGuard::new();
    let cpu_id = awkernel_lib::cpu::cpu_id();
    if !IN_SLICE[cpu_id].load(Ordering::Acquire) {
        return None;
    }
    let start = SLICE_START_NS[cpu_id].load(Ordering::Relaxed);
    Some(
        SLICE_BASE_NS[cpu_id]
            .load(Ordering::Relaxed)
            .saturating_add(now_ns().saturating_sub(start)),
    )
}

/// Busy-run until the calling task has been on a CPU for `duration` more
/// (see the module doc). Outside a task it falls back to
/// `awkernel_lib::busy_work::spin_work`.
pub fn spin_cpu_time(duration: Duration) {
    let Some(start) = current_ns() else {
        awkernel_lib::busy_work::spin_work(duration);
        return;
    };
    let target = start.saturating_add(u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX));
    loop {
        let Some(now) = current_ns() else {
            return;
        };
        let Some(left) = target.checked_sub(now).filter(|left| *left > 0) else {
            return;
        };
        awkernel_lib::busy_work::spin_work(Duration::from_nanos(left).min(STEP));
    }
}
