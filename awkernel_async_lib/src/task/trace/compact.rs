//! Compact encoding of the bulk `TRACE_*` records for the serial dump.
//!
//! On real hardware the dump goes out over a 115200-baud serial line
//! (about 11 KB/s), so a 12 MB trace takes about 18 minutes to send. Most of
//! it is `TRACE_EV`, `TRACE_DPSW` and `TRACE_NDONE` lines whose timestamps
//! repeat the same leading digits, so these are sent as differences in hex
//! under a short per-CPU header instead:
//!
//! ```text
//! TRACE_EVZ,<cpu>
//! e<task_id><S|E|R><tsc delta>[,<period_index|->]
//! TRACE_DPSWZ,<cpu>
//! d<scheduled delta>,<handler-scheduled>,<decided>,<started>,<from|->,<to|->,
//!   <path code>,<skipped>,<entered>,<wake_lock>,<data_lock>,<pre_lock>
//! TRACE_DAGEVZ,<cpu>
//! r<dag_id>,<job>,<release delta>,<woke-release>
//! n<dag_id>,<node_id>,<job>,<done delta>
//! ```
//!
//! - A delta is from the previous record of the same kind on the same CPU
//!   (the first one is from 0, so it is the absolute value), in lowercase
//!   hex, with a `~` prefix when negative.
//! - In `d` lines, `decided` .. `pre_lock` are relative to `handler`, and
//!   `-` stands for 0 (did not happen), as in `TRACE_DPSW`. The path is the
//!   `DPSW_*` code number.
//! - `e` lines carry the period index only when it differs from the
//!   previous `e` line on the same CPU (and never without the
//!   `period-index-propagation` feature).
//! - Lines end with `\n` only (minicom drops the `\r` from its capture
//!   anyway).
//!
//! The host expands these back to the usual `TRACE_EV`, `TRACE_DPSW`,
//! `TRACE_REL` and `TRACE_NDONE` lines
//! (`awkernel_script/trace_view/trace_log.py`), so the analysis scripts read
//! the same records as before.

use alloc::string::String;
use core::fmt::Write;

/// Append `to - from` as signed hex (`~` prefix when negative).
fn push_delta(out: &mut String, from: u64, to: u64) {
    if to >= from {
        let _ = write!(out, "{:x}", to - from);
    } else {
        let _ = write!(out, "~{:x}", from - to);
    }
}

/// `-` for 0 (did not happen), otherwise `v - base` as in [`push_delta`].
fn push_rel_or_dash(out: &mut String, base: u64, v: u64) {
    if v == 0 {
        out.push('-');
    } else {
        push_delta(out, base, v);
    }
}

fn push_dag_or_dash(out: &mut String, v: u32) {
    if v == u32::MAX {
        out.push('-');
    } else {
        let _ = write!(out, "{v}");
    }
}

/// Encoder for one CPU's `TRACE_EV` records.
#[derive(Default)]
pub(super) struct EvEncoder {
    prev_tsc: u64,
    /// The last period field written; `None` before the first line.
    prev_period: Option<Option<u32>>,
}

impl EvEncoder {
    /// `period`: `None` when the build has no period field at all
    /// (`period-index-propagation` disabled), otherwise the event's period
    /// index (`Some(None)` is printed as `-`).
    pub(super) fn push(
        &mut self,
        out: &mut String,
        task_id: u32,
        kind: char,
        tsc: u64,
        period: Option<Option<u32>>,
    ) {
        let _ = write!(out, "e{task_id}{kind}");
        push_delta(out, self.prev_tsc, tsc);
        self.prev_tsc = tsc;
        if let Some(p) = period {
            if self.prev_period != Some(p) {
                match p {
                    Some(p) => {
                        let _ = write!(out, ",{p}");
                    }
                    None => out.push_str(",-"),
                }
                self.prev_period = Some(p);
            }
        }
        out.push('\n');
    }
}

/// Encoder for one CPU's `TRACE_DPSW` records.
#[derive(Default)]
pub(super) struct DpswEncoder {
    prev_scheduled: u64,
}

impl DpswEncoder {
    pub(super) fn push(&mut self, out: &mut String, e: &super::DpSwitchEvent) {
        out.push('d');
        push_delta(out, self.prev_scheduled, e.scheduled_ns);
        self.prev_scheduled = e.scheduled_ns;
        out.push(',');
        push_delta(out, e.scheduled_ns, e.handler_ns);
        for v in [e.decided_ns, e.started_ns] {
            out.push(',');
            push_rel_or_dash(out, e.handler_ns, v);
        }
        out.push(',');
        push_dag_or_dash(out, e.from_dag);
        out.push(',');
        push_dag_or_dash(out, e.to_dag);
        let _ = write!(out, ",{},{}", e.path, e.skipped);
        for v in [e.entered_ns, e.wake_lock_ns, e.data_lock_ns, e.pre_lock_ns] {
            out.push(',');
            push_rel_or_dash(out, e.handler_ns, v);
        }
        out.push('\n');
    }
}

/// Encoder for one CPU's `TRACE_REL` / `TRACE_NDONE` records.
#[derive(Default)]
pub(super) struct DagEvEncoder {
    prev_at: u64,
}

impl DagEvEncoder {
    pub(super) fn push(&mut self, out: &mut String, e: &super::DagJobEvent) {
        if e.kind == super::DAG_EV_RELEASE {
            let _ = write!(out, "r{},{},", e.dag_id, e.job);
            push_delta(out, self.prev_at, e.at_ns);
            out.push(',');
            push_delta(out, e.at_ns, e.observed_ns);
        } else {
            let _ = write!(out, "n{},{},{},", e.dag_id, e.node_id, e.job);
            push_delta(out, self.prev_at, e.at_ns);
        }
        self.prev_at = e.at_ns;
        out.push('\n');
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::task::trace::{DagJobEvent, DpSwitchEvent, DAG_EV_NODE_DONE, DAG_EV_RELEASE};

    // The same vectors are checked by the host decoder's self-test
    // (`awkernel_script/trace_view/trace_log.py --self-test`); keep the two
    // in sync when the format changes.

    #[test]
    fn ev_deltas_and_period() {
        let mut enc = EvEncoder::default();
        let mut out = String::new();
        enc.push(&mut out, 11, 'S', 50701689519, Some(Some(4)));
        enc.push(&mut out, 11, 'E', 50701802776, Some(Some(4)));
        enc.push(&mut out, 12, 'S', 50701802000, Some(None));
        enc.push(&mut out, 12, 'E', 50701803000, Some(None));
        assert_eq!(out, "e11Sbce0e62af,4\ne11E1ba69\ne12S~308,-\ne12E3e8\n");
    }

    #[test]
    fn ev_without_period_field() {
        let mut enc = EvEncoder::default();
        let mut out = String::new();
        enc.push(&mut out, 3, 'S', 16, None);
        enc.push(&mut out, 3, 'E', 32, None);
        assert_eq!(out, "e3S10\ne3E10\n");
    }

    #[test]
    fn dpsw_relative_to_handler() {
        let mut enc = DpswEncoder::default();
        let mut out = String::new();
        enc.push(
            &mut out,
            &DpSwitchEvent {
                scheduled_ns: 5464669017,
                handler_ns: 5464711379,
                decided_ns: 5464781906,
                started_ns: 5464787239,
                from_dag: u32::MAX,
                to_dag: 1,
                path: 1,
                skipped: 0,
                entered_ns: 5464764204,
                wake_lock_ns: 5464777186,
                data_lock_ns: 5464779764,
                pre_lock_ns: 5464764235,
            },
        );
        enc.push(
            &mut out,
            &DpSwitchEvent {
                scheduled_ns: 5515612041,
                handler_ns: 5515752850,
                from_dag: 1,
                to_dag: 1,
                path: 2,
                ..Default::default()
            },
        );
        assert_eq!(
            out,
            "d145b83b59,a57a,1137f,12854,-,1,1,0,ce59,1010f,10b21,ce78\n\
             d3095430,22609,-,-,1,1,2,0,-,-,-,-\n"
        );
    }

    #[test]
    fn dag_events() {
        let mut enc = DagEvEncoder::default();
        let mut out = String::new();
        enc.push(
            &mut out,
            &DagJobEvent {
                kind: DAG_EV_RELEASE,
                dag_id: 1,
                node_id: 0,
                job: 0,
                at_ns: 5400000000,
                observed_ns: 5400012345,
            },
        );
        enc.push(
            &mut out,
            &DagJobEvent {
                kind: DAG_EV_NODE_DONE,
                dag_id: 1,
                node_id: 2,
                job: 0,
                at_ns: 5618404086,
                observed_ns: 0,
            },
        );
        assert_eq!(out, "r1,0,141dd7600,3039\nn1,2,0,d0494f6\n");
    }
}
