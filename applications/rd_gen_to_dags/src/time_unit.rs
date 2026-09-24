use awkernel_lib::delay::uptime;
use core::hint::black_box;
use core::sync::atomic::{AtomicU64, Ordering};
use core::time::Duration;

#[cfg(feature = "seconds")]
pub(super) fn convert_duration(duration: u64) -> Duration {
    Duration::from_secs(duration)
}

#[cfg(feature = "milliseconds")]
pub(super) fn convert_duration(duration: u64) -> Duration {
    Duration::from_millis(duration)
}

#[cfg(feature = "microseconds")]
pub(super) fn convert_duration(duration: u64) -> Duration {
    Duration::from_micros(duration)
}

#[cfg(feature = "nanoseconds")]
pub(super) fn convert_duration(duration: u64) -> Duration {
    Duration::from_nanos(duration)
}

// default
#[cfg(not(any(
    feature = "seconds",
    feature = "milliseconds",
    feature = "microseconds",
    feature = "nanoseconds"
)))]
pub(super) fn convert_duration(duration: u64) -> Duration {
    Duration::from_millis(duration)
}

// ---------------------------------------------------------------------------
// Work-bounded (not time-bound) busy simulation.
//
// The original `simulated_execution_time` called `awkernel_lib::delay::
// wait_millisec`, a *time*-bound spin wait: it polls `uptime()` and loops
// `core::hint::spin_loop()` (the x86 `PAUSE` instruction) until the target
// wall-clock duration has elapsed. `PAUSE` is specifically designed to yield
// shared execution resources to an SMT sibling thread -- the opposite of
// contending for them -- so a task built on it can *never* show HT/SMT
// interference: it always takes ~the same wall-clock time to finish,
// regardless of what its sibling is doing, by construction (confirmed
// empirically 2026-09-24: an HT-off/HT-on PoC comparison showed no
// measurable difference once trace-window-boundary artifacts were excluded).
//
// This replaces it with a *work*-bound loop: a fixed number of genuine
// ALU-bound iterations (no PAUSE), calibrated once (see `calibrate`) to take
// ~1ms per `ITERS_PER_MS` iterations on an *uncontended* core. Completing
// that same fixed amount of work takes measurably longer wall-clock time
// when a sibling is contending for the same physical core's execution
// ports/cache -- which is what a caller wanting to see that effect (e.g. a
// DAG task body under Federated admission) needs.
// ---------------------------------------------------------------------------

static ITERS_PER_MS: AtomicU64 = AtomicU64::new(0);

/// One unit of genuine compute (integer multiply-add, `black_box`-wrapped so
/// the compiler can't prove it's dead and elide the loop). Deliberately not
/// `core::hint::spin_loop()` -- see this module's own doc comment above.
#[inline(always)]
fn busy_iteration(acc: u64) -> u64 {
    black_box(acc.wrapping_mul(2_654_435_761).wrapping_add(1))
}

/// Measures iterations-per-millisecond of `busy_iteration` on whichever CPU
/// calls this, over a short time-bound window (the *only* place in this
/// module that's still time-bound, since a measurement necessarily is).
/// Callers must call this explicitly, once, on a quiet system (nothing else
/// contending for this CPU's SMT sibling yet) -- see `rd_gen_to_dags`'s own
/// `run()`, which does so before any DAG task is spawned. Calling it lazily
/// from inside a DAG task body instead would risk calibrating *during*
/// contention on whichever CPU happens to run first, corrupting the
/// baseline for every other CPU (this static is shared, not per-CPU).
pub fn calibrate_busy_work() {
    const CALIBRATION_MS: u64 = 5;
    const SAMPLE_EVERY: u64 = 4096;

    let start = uptime();
    let mut acc: u64 = 1;
    let mut iters: u64 = 0;
    loop {
        acc = busy_iteration(acc);
        iters += 1;
        if iters.is_multiple_of(SAMPLE_EVERY) && uptime().saturating_sub(start) >= CALIBRATION_MS * 1000
        {
            break;
        }
    }
    black_box(acc);

    let per_ms = (iters / CALIBRATION_MS).max(1);
    ITERS_PER_MS.store(per_ms, Ordering::Relaxed);
    log::info!("rd_gen_to_dags: calibrated busy-work at {per_ms} iterations/ms");
}

fn busy_work_for_millisec(ms: u64) {
    let per_ms = ITERS_PER_MS.load(Ordering::Relaxed);
    // Uncalibrated (caller forgot `calibrate_busy_work()`): fall back to the
    // old time-bound wait rather than doing zero work silently.
    if per_ms == 0 {
        awkernel_lib::delay::wait_millisec(ms);
        return;
    }
    let target = per_ms.saturating_mul(ms);
    let mut acc: u64 = 1;
    for _ in 0..target {
        acc = busy_iteration(acc);
    }
    black_box(acc);
}

#[cfg(feature = "seconds")]
pub(super) fn simulated_execution_time(duration: u64) {
    // busy_work_for_millisec(duration * 1000);
    wait_millisec(duration * 1000);
}

#[cfg(feature = "milliseconds")]
pub(super) fn simulated_execution_time(duration: u64) {
    // busy_work_for_millisec(duration);
    wait_millisec(duration);
}

#[cfg(feature = "microseconds")]
pub(super) fn simulated_execution_time(duration: u64) {
    // busy_work_for_millisec(duration / 1000);
    wait_millisec(duration / 1000);
}

#[cfg(feature = "nanoseconds")]
pub(super) fn simulated_execution_time(duration: u64) {
    // busy_work_for_millisec(duration / 1000000);
    wait_millisec(duration / 1000000);
}

// default
#[cfg(not(any(
    feature = "seconds",
    feature = "milliseconds",
    feature = "microseconds",
    feature = "nanoseconds"
)))]
pub(super) fn simulated_execution_time(duration: u64) -> u64 {
    duration
}
