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

// Only DAG-Fluid admission (the `dagfluid` feature) converts fractional
// values; the `cfg_attr`s keep other builds free of dead-code warnings.

/// How many of this build's time units make one second.
#[cfg(feature = "seconds")]
#[cfg_attr(not(feature = "dagfluid"), allow(dead_code))]
const UNITS_PER_SEC: f64 = 1.0;
#[cfg(feature = "microseconds")]
#[cfg_attr(not(feature = "dagfluid"), allow(dead_code))]
const UNITS_PER_SEC: f64 = 1_000_000.0;
#[cfg(feature = "nanoseconds")]
#[cfg_attr(not(feature = "dagfluid"), allow(dead_code))]
const UNITS_PER_SEC: f64 = 1_000_000_000.0;
// milliseconds, and the default
#[cfg(not(any(feature = "seconds", feature = "microseconds", feature = "nanoseconds")))]
#[cfg_attr(not(feature = "dagfluid"), allow(dead_code))]
const UNITS_PER_SEC: f64 = 1_000.0;

/// Like [`convert_duration`], for a fractional value such as a DAG-Fluid
/// segment's offset or deadline. Keeps sub-unit precision (down to 1 ns)
/// instead of truncating to whole units. `None` for a negative, NaN or
/// too large value.
#[cfg_attr(not(feature = "dagfluid"), allow(dead_code))]
pub(super) fn convert_duration_f64(duration: f64) -> Option<Duration> {
    Duration::try_from_secs_f64(duration / UNITS_PER_SEC).ok()
}

/// Perform `duration` of work-bound computation (see `awkernel_lib::busy_work`):
/// time spent preempted does not count toward it, unlike a wall-clock wait.
#[cfg(any(
    feature = "seconds",
    feature = "milliseconds",
    feature = "microseconds",
    feature = "nanoseconds"
))]
pub(super) fn simulated_execution_time(duration: u64) {
    #[cfg(not(feature = "cpu_time_work"))]
    awkernel_lib::busy_work::spin_work(convert_duration(duration));
    // `cpu_time_work`: exactly `duration` of CPU time, whatever the core's
    // speed (see `awkernel_async_lib::task::cpu_time`).
    #[cfg(feature = "cpu_time_work")]
    awkernel_async_lib::task::cpu_time::spin_cpu_time(convert_duration(duration));
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fractional_values_keep_sub_unit_precision() {
        // 2.5 units, whatever the unit, is 2.5 / UNITS_PER_SEC seconds.
        let expected = Duration::from_secs_f64(2.5 / UNITS_PER_SEC);
        assert_eq!(convert_duration_f64(2.5), Some(expected));
        assert!(convert_duration_f64(2.5).unwrap() > convert_duration(2));
        assert!(convert_duration_f64(2.5).unwrap() < convert_duration(3));
    }

    #[test]
    fn whole_values_match_convert_duration() {
        assert_eq!(convert_duration_f64(7.0), Some(convert_duration(7)));
    }

    #[test]
    fn invalid_values_are_rejected() {
        assert_eq!(convert_duration_f64(-1.0), None);
        assert_eq!(convert_duration_f64(f64::NAN), None);
        assert_eq!(convert_duration_f64(f64::INFINITY), None);
    }
}
