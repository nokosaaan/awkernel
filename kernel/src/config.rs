#[cfg(not(feature = "std"))]
pub use crate::arch::config::*;

/// Backup Heap size is 64 MiB
#[allow(dead_code)]
pub const BACKUP_HEAP_SIZE: usize = 64 * 1024 * 1024;

/// Auto-trace: record a task execution trace automatically after boot and
/// dump it to the serial console in the `TRACE_*` format.
///
/// Intended for real hardware where no shell input is available; capture the
/// serial output on the host (e.g. `minicom -D /dev/ttyUSB0 -C trace.log`)
/// and plot it with `awkernel_script/trace_view/plot_trace.py`.
///
/// Set `AUTO_TRACE_ENABLED` to `false` to use only the shell commands
/// (`(trace_start)` / `(trace_stop)` / `(trace)`) instead.
#[allow(dead_code)]
pub const AUTO_TRACE_ENABLED: bool = true;

/// Seconds to wait after boot before starting the auto-trace.
///
/// Keep this shorter than the userland app's startup (test DAG apps wait
/// ~1 s and finish `finish_create_dags` at ~5 s after boot) so the trace
/// window covers the DAG from its very first release.
#[allow(dead_code)]
pub const AUTO_TRACE_START_DELAY_SECS: u64 = 2;

/// Length of the auto-trace recording in seconds.
#[allow(dead_code)]
pub const AUTO_TRACE_DURATION_SECS: u64 = 10;

/// Auto-reboot: reboot the machine (see
/// `awkernel_lib::arch::x86_64::power::reboot`) a fixed time after boot,
/// with no shell/network trigger needed.
///
/// Intended for the same no-shell-input real hardware as auto-trace, but
/// for a different problem: this crate's own one-shot PXE boot (Windows
/// `bcdedit /set {fwbootmgr} bootsequence`, or Linux `efibootmgr
/// --bootnext`) is consumed by the *next* boot regardless of which OS that
/// boot loads, so once the real machine finishes whatever this trial
/// needed (auto-trace's dump, DAG admission, etc.) there is nothing further
/// for it to do -- but it has no way to know that and would otherwise just
/// idle indefinitely. Rebooting here means that consumed one-shot entry
/// falls through to the normal boot order and the machine comes back up in
/// its regular OS on its own, reachable again over ssh with no external
/// power intervention (WoL, physical power button, etc.) needed -- an
/// external orchestrator (see `scripts/real_machine_trial.py`) can just
/// keep polling ssh. (An earlier revision used `power::shutdown` instead,
/// for a cleaner "done" signal at the cost of needing Wake-on-LAN to start
/// the next trial; switch back to that if the always-on-between-trials
/// power/heat cost of rebooting ever matters more than that simplicity.)
///
/// Set `AUTO_REBOOT_ENABLED` to `false` to disable (e.g. interactive
/// QEMU/shell-input development, where an unattended reboot would just get
/// in the way).
#[allow(dead_code)]
pub const AUTO_REBOOT_ENABLED: bool = true;

/// Upper-bound safety net, in seconds after boot, for auto-reboot.
///
/// `main.rs`'s `spawn_auto_reboot` no longer just sleeps this long and
/// reboots: it polls `AUTO_TRACE_DONE` (set once `trace::dump_to_console()`
/// actually returns) and reboots as soon as that's true, so a quick trace
/// doesn't sit idle for the rest of this window and a slow one doesn't get
/// cut off mid-dump. This constant only matters as the fallback cap for
/// when that signal never arrives -- auto-trace disabled, its feature
/// (`perf`) not compiled in, or its task panicking -- so the machine still
/// reboots eventually instead of idling forever.
///
/// Previously (before the poll-based wait existed) this *was* the sole
/// trigger, sleeping unconditionally: a 60s value only left ~28s of margin
/// after `AUTO_TRACE_START_DELAY_SECS + AUTO_TRACE_DURATION_SECS` (32s),
/// and a real-machine trial with a large enough DAG batch (more
/// tasks/events -> a bigger dump, sent out over a 115200-baud serial line
/// with no size cap) got its trace cut off mid-write when the fixed timer
/// fired before the dump finished. Kept generous here anyway since it's
/// now just a safety net, not the thing every trial waits out.
#[allow(dead_code)]
pub const AUTO_REBOOT_SECS: u64 = 300;

#[cfg(test)]
#[allow(dead_code)]
pub const HEAP_START: usize = 0;

/// Initialize the architecture specific configuration.
///
/// # Safety
///
/// This function must be called before at the beginning of the kernel.
pub unsafe fn init() {
    #[cfg(all(feature = "x86", not(feature = "linux")))]
    {
        awkernel_lib::config::set_stack_size(STACK_SIZE);
        awkernel_lib::config::set_stack_start(STACK_START);
    }
}
