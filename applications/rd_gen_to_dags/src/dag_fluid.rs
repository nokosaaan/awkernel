//! DAG-Fluid's *static* admission math only — segment decomposition and
//! Algorithm 2's per-task capacity computation, used purely as an offline
//! schedulability test (see this crate's `examples/acceptance_ratio.rs`).
//! Deliberately excludes the papers' dynamic runtime layer (DP-Fair/DP-Wrap
//! dispatch, Deadline Partition boundary detection) — that requires new,
//! system-wide kernel scheduling infrastructure that does not exist in this
//! codebase today.
//!
//! # Which paper, and why this module was rewritten
//!
//! There are two DAG-Fluid papers: Guan, Qiao, Han (IEEE TC 2020/2021,
//! "DAG-Fluid: A Real-Time Scheduling Algorithm for DAGs") covers only
//! *implicit*-deadline tasks (`D_i = T_i` unconditionally, Section 3: "we
//! only consider the condition that T_i = D_i"). Guan, Peng, Qiao (IEEE TC
//! 2022, "A Fluid Scheduling Algorithm for DAG Tasks With Constrained or
//! Arbitrary Deadlines") extends it to `D_i <= T_i` (constrained) and
//! unconstrained deadlines via a *different* per-task algorithm (a virtual
//! deadline `D*_i` plus an iterative segment-classification procedure,
//! Sections 4–6).
//!
//! This module previously implemented the 2020 paper's algorithm (a fixed,
//! `m`-dependent threshold `alpha_i = C_i/(T_i - (m/(m+1))*L_i)` for
//! light/heavy classification), but `examples/acceptance_ratio.rs` generates
//! *constrained*-deadline task sets (via the V-Fed paper's own `D ~
//! Uniform[L, L/alpha]` then `T = D/beta`, `beta <= 1`, so `T >= D`
//! unconditionally) — exactly the case the 2020 paper's model excludes.
//! Applying the 2020 formulas to that data was a category error: it
//! compiled, had passing unit tests, and produced monotonic-looking curves,
//! but the underlying math didn't apply to the deadlines actually being fed
//! into it. Confirmed by direct comparison against the 2022 paper's own
//! Table 1/Algorithm 1/Algorithm 2 text (reproduced below) after the user
//! supplied both primary sources; this rewrite replaces the 2020-paper
//! logic with the 2022 paper's constrained/arbitrary-deadline algorithm.
//!
//! # The algorithm (transcribed from the 2022 paper)
//!
//! ## Step 1 — virtual deadline `D*_i` ([`virtual_deadline`], Table 1)
//!
//! `D*_i` does not necessarily equal `D_i`, but always satisfies `L_i <=
//! D*_i <= D_i`. Selected by a 6-row table keyed on `T_i/D_i` and `L_i` vs
//! `T_i` (`rho` below is a placeholder for whichever capacity-augmentation
//! bound is being analyzed — irrelevant here, this module only needs the
//! table's row *selection*, not the bounds themselves):
//!
//! | row | condition                                              | `D*_i`  |
//! |-----|---------------------------------------------------------|---------|
//! | 1   | `T_i/D_i in [1, inf)`                                    | `D_i`   |
//! | 2   | `T_i/D_i in (0,1)` and `L_i > T_i`                       | `D_i`   |
//! | 3   | `T_i/D_i in (0, (sqrt(3)-1)/2]` and `L_i <= T_i`         | `D_i`   |
//! | 4   | `T_i/D_i in ((sqrt(3)-1)/2, 0.5]` and `L_i <= T_i`       | `2*T_i` |
//! | 5   | `T_i/D_i in (0.5, sqrt(2)/2]` and `L_i <= T_i`           | `D_i`   |
//! | 6   | `T_i/D_i in (sqrt(2)/2, 1)` and `L_i <= T_i`             | `T_i`   |
//!
//! Row 1 (`T_i >= D_i`, i.e. constrained deadline) is the *only* row this
//! crate's own experimental setup can ever hit: `acceptance_ratio.rs`
//! guarantees `T_i >= D_i` by construction (`T = D/beta`, `beta <= 1`), so
//! `T_i/D_i >= 1` unconditionally, and `D*_i = D_i` always in practice. Rows
//! 2–6 (arbitrary deadlines, `D_i > T_i`) are implemented anyway for
//! faithfulness and reusability, but are dead code against this crate's own
//! generated pools today.
//!
//! ## Step 2 — classify by `C_i` vs `D*_i` (not `T_i`)
//!
//! - `C_i <= D*_i` (`tau_seq`): stretched into a sequential task with a
//!   constant execution rate `C_i/D*_i` (2022 paper, Algorithm 2 line 6).
//! - `C_i > D*_i` (`tau_paral`): decomposed into segments (unchanged from
//!   before, see [`decompose_segments`] — this step doesn't depend on the
//!   deadline model), then run through Algorithm 1 ([`heavy_capacity_and_deadline`])
//!   to get `C^H_i`, `D^H_i`.
//!
//! ## Step 3 — Algorithm 1 (`heavy_capacity_and_deadline`): light/heavy split
//!
//! Unlike the 2020 paper's fixed `alpha_i` threshold, this is an **iterative
//! greedy peel**, and critically **does not depend on `m` at all**: sort
//! segments by ascending `m_i,j` (thread count); repeatedly test the
//! smallest remaining segment against the *running* remainders `C^r_i`
//! (initially `C_i`), `D^r_i` (initially `D*_i`) — if `C^r_i/(D^r_i * m_i,h)
//! > 1`, that segment is light (peel it off: `C^r_i -= l_i,h * m_i,h`,
//! `D^r_i -= l_i,h`, continue to the next-smallest); the first time the test
//! fails, stop — every remaining segment (including the one that just
//! failed) is heavy, and the final `C^r_i`, `D^r_i` are `C^H_i`, `D^H_i`.
//!
//! ## Step 4 — Algorithm 2: per-task capacity contribution ([`required_capacity`])
//!
//! `rate_i = C_i/D*_i` (tau_seq) or `C^H_i/D^H_i` (tau_paral) — this is a
//! processor-*count* contribution (can exceed 1; it is the per-task analogue
//! of Federated's `required_cores`), not a per-thread rate. The task's full
//! contribution to the system-wide capacity sum is `ceil(D*_i/T_i) *
//! rate_i` (`ceil(D*_i/T_i)` accounts for multiple concurrent job instances
//! when `D*_i > T_i`; always `1` under this crate's constrained-deadline
//! pools since `D*_i = D_i <= T_i` there).
//!
//! `m` **never appears** in Steps 2–4 above — every quantity is computed
//! from a single task's own `C`, `L`, `T`, `D` alone, exactly like
//! `federated::classify_dag`'s own `(volume, critical_path, period,
//! deadline) -> TaskClass` shape (no ledger, no shared state). This
//! corrects the previous implementation's design mistake of threading `m`
//! through the per-task computation (a property specific to the 2020
//! paper's *different*, implicit-deadline-only `alpha_i` formula) and the
//! `concurrency <= m` guard bolted on top of it — neither belongs here: in
//! the 2022 paper's model, `m` only ever appears in the final aggregate sum
//! (see [`is_batch_feasible`]).
//!
//! ## Step 5 — system-wide admission ([`is_batch_feasible`])
//!
//! A task set is feasible on `m` shared cores iff `sum(required_capacity_i)
//! <= m` (2022 paper, Algorithm 2 lines 1–17: accumulate `delta` across all
//! tasks, reject if `delta > m` at any point). This is now a direct
//! transcription of Algorithm 2 rather than an inferred composition.

use crate::parse_yaml::DagData;

use alloc::collections::{BTreeMap, VecDeque};
use alloc::vec::Vec;

/// One interval of [`decompose_segments`]'s output: `duration` = `l_i,j`
/// (the WCET of each thread in this segment, under the papers' "infinite
/// processors" idealization; also how long this interval lasts),
/// `concurrency` = `m_i,j` (how many threads/nodes run simultaneously
/// during it).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Segment {
    pub duration: u64,
    pub concurrency: u32,
}

/// Decompose `dag_data` into segments via the papers' shared construction
/// (2022 paper Section 4.1, identical to the 2020 paper's Section 4.2):
/// build the "infinite processors" timing diagram (every node starts the
/// instant its last predecessor finishes — `start[v] = max(finish[pred] for
/// pred in in_links)`, `finish[v] = start[v] + execution_time[v]`), collect
/// every distinct start/finish instant into a sorted timeline, and treat
/// each consecutive pair of instants as one segment, with `concurrency` =
/// how many nodes are active (`start <= t0 < finish`) during it. This step
/// is shared by both papers' algorithms and is unaffected by which deadline
/// model is in use.
///
/// Two structural invariants hold for any DAG (checked in this module's own
/// tests against [`crate::dag_stats::compute_dag_stats`]'s independently
/// computed values, since both are derived from the same node/edge data by
/// a different route):
/// - `Σ duration * concurrency == volume` (every unit of work is in exactly
///   one segment, running on exactly `concurrency` nodes for that
///   segment's `duration`).
/// - `Σ duration == critical_path` (segments partition the *entire*
///   infinite-processor timeline end to end, and that timeline's total span
///   is, by definition, the longest path's own length).
pub(crate) fn decompose_segments(dag_data: &DagData) -> Vec<Segment> {
    let nodes = dag_data.get_nodes();

    let node_by_id: BTreeMap<u32, &crate::parse_yaml::NodeData> =
        nodes.iter().map(|node| (node.get_id(), node)).collect();

    let mut in_degree: BTreeMap<u32, usize> = nodes
        .iter()
        .map(|node| (node.get_id(), node.get_in_links().len()))
        .collect();

    let mut queue: VecDeque<u32> = in_degree
        .iter()
        .filter(|&(_, &degree)| degree == 0)
        .map(|(&id, _)| id)
        .collect();

    let mut start: BTreeMap<u32, u64> = BTreeMap::new();
    let mut finish: BTreeMap<u32, u64> = BTreeMap::new();

    while let Some(id) = queue.pop_front() {
        let Some(node) = node_by_id.get(&id) else {
            continue;
        };

        let pred_max = node
            .get_in_links()
            .iter()
            .filter_map(|pred_id| finish.get(pred_id).copied())
            .max()
            .unwrap_or(0);
        let node_start = pred_max;
        let node_finish = node_start + node.get_execution_time();
        start.insert(id, node_start);
        finish.insert(id, node_finish);

        for out_id in node.get_out_links() {
            if let Some(degree) = in_degree.get_mut(out_id) {
                *degree = degree.saturating_sub(1);
                if *degree == 0 {
                    queue.push_back(*out_id);
                }
            }
        }
    }

    let mut events: Vec<u64> = start.values().chain(finish.values()).copied().collect();
    events.sort_unstable();
    events.dedup();

    let mut segments = Vec::new();
    for window in events.windows(2) {
        let (t0, t1) = (window[0], window[1]);
        let concurrency = start
            .iter()
            .filter(|&(id, &node_start)| {
                let node_finish = finish.get(id).copied().unwrap_or(node_start);
                node_start <= t0 && t0 < node_finish
            })
            .count();
        if concurrency == 0 {
            continue; // no node active in this interval (shouldn't happen for a connected DAG, but harmless if it does)
        }
        segments.push(Segment {
            duration: t1 - t0,
            concurrency: concurrency as u32,
        });
    }
    segments
}

/// Table 1 (2022 paper): pick the virtual deadline `D*_i`. See this
/// module's own doc for the full table; `period`/`deadline`/`critical_path`
/// are `T_i`/`D_i`/`L_i`. Always returns `deadline` (row 1) when `period >=
/// deadline` (constrained deadline, `T_i/D_i >= 1`) — the only case this
/// crate's own generated pools ever exercise.
pub(crate) fn virtual_deadline(period: u64, deadline: u64, critical_path: u64) -> u64 {
    let t = period as f64;
    let d = deadline as f64;
    let l = critical_path as f64;
    let ratio = t / d;

    // (sqrt(3)-1)/2 and sqrt(2)/2, the two row-3/4/5/6 boundary constants.
    let row3_upper = (libm_sqrt(3.0) - 1.0) / 2.0;
    let row5_upper = libm_sqrt(2.0) / 2.0;

    if ratio >= 1.0 {
        deadline // Row 1: constrained deadline (T_i/D_i in [1, inf)).
    } else if l > t {
        deadline // Row 2: arbitrary deadline, L_i > T_i.
    } else if ratio <= row3_upper {
        deadline // Row 3.
    } else if ratio <= 0.5 {
        2 * period // Row 4.
    } else if ratio <= row5_upper {
        deadline // Row 5.
    } else {
        period // Row 6.
    }
}

// `f64::sqrt` isn't available in `core` without `libm`/`std`; this crate
// builds with `std` for the `acceptance_ratio` example (see its own
// `--features std`), but stays `#![no_std]` at the crate root, so spell the
// two needed square roots out via Newton's method rather than add a new
// dependency for two constants.
fn libm_sqrt(x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    let mut guess = x;
    for _ in 0..64 {
        guess = 0.5 * (guess + x / guess);
    }
    guess
}

/// Algorithm 1 (2022 paper), lines 1–15 only: the iterative greedy
/// light/heavy split, returning `(C^H_i, D^H_i)`. Lines 16–21 (assigning
/// each segment's own relative deadline `d_i,j`) are handled separately by
/// [`assign_segment_deadlines`] (dynamic-dispatch support, added for the
/// real-machine DAG-Fluid work's Phase 1 -- DP boundary computation only,
/// no dispatch yet); only the aggregate `C^H_i`/`D^H_i` computed here feed
/// into [`required_capacity`].
///
/// Returns `None` if every segment ends up peeled as light (no heavy
/// segment remains) or either aggregate is non-positive — per Lemma 5.2 of
/// the 2022 paper, a task with `volume > virtual_deadline` (the only case
/// this is called for) always has at least one heavy segment, so this is a
/// defensive fallback for a degenerate/inconsistent input rather than an
/// expected outcome.
fn heavy_capacity_and_deadline(
    segments: &[Segment],
    volume: u64,
    virtual_deadline: u64,
) -> Option<(f64, f64)> {
    let mut sorted: Vec<Segment> = segments.to_vec();
    sorted.sort_by_key(|s| s.concurrency);

    let mut c_r = volume as f64;
    let mut d_r = virtual_deadline as f64;
    let mut heavy_start = 0;

    for seg in &sorted {
        let m = seg.concurrency as f64;
        if c_r / (d_r * m) > 1.0 {
            c_r -= seg.duration as f64 * m;
            d_r -= seg.duration as f64;
            heavy_start += 1;
        } else {
            break;
        }
    }

    if heavy_start == sorted.len() {
        return None;
    }
    (c_r > 0.0 && d_r > 0.0).then_some((c_r, d_r))
}

/// One entry of [`assign_segment_deadlines`]'s output, in `segments`' own
/// (chronological/timeline) order — NOT the `m_i,j`-sorted order Algorithm
/// 1 uses internally only for light/heavy classification (lines 1–15).
/// Section 8's `r_i,j` (segment release offset) needs the timeline order to
/// accumulate predecessor segments' deadlines; see
/// [`crate::dag_fluid`]'s own module doc and
/// [`segment_release_offsets`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SegmentSchedule {
    /// `l_i,j`: this segment's own thread WCET, copied from the input
    /// [`Segment`].
    pub duration: u64,
    /// `m_i,j`: this segment's thread count, copied from the input
    /// [`Segment`].
    pub concurrency: u32,
    /// `d_i,j` (Algorithm 1 lines 17/20): this segment's own relative
    /// deadline.
    pub relative_deadline: f64,
    /// `θ_i,j` (Definition 4.1, `l_i,j / d_i,j`): the execution rate every
    /// thread in this segment runs at once dispatched. `1.0` for light
    /// segments (by construction: `d_i,j == l_i,j` there); at most `1.0`
    /// for heavy segments too, per Lemma 5.4's own proof (not re-derived
    /// here — this function trusts the paper's proof rather than
    /// asserting it defensively, since `#[cfg(test)]` below checks it
    /// against the paper's own worked example instead).
    pub rate: f64,
}

/// Algorithm 1 (2022 paper) lines 16–21, in full — reproduced here
/// verbatim from the primary source (Guan, Peng, Qiao, IEEE TC 2022,
/// Section 4.1):
///
/// ```text
///  1: σ^H_i ← σ_i, σ^L_i ← ∅
///  2: sort σ^H_i in increasing order of m_i,j
///  3: C^r_i ← C_i, D^r_i ← D*_i
///  4: while σ^H_i ≠ ∅ do
///  5:   get the head element σ_i,h from σ^H_i
///  6:   if C^r_i / (D^r_i · m_i,h) > 1 then
///  7:     σ^L_i ← σ^L_i ∪ {σ_i,h}
///  8:     σ^H_i ← σ^H_i \ {σ_i,h}
///  9:     C^r_i ← C^r_i − l_i,h·m_i,h
/// 10:     D^r_i ← D^r_i − l_i,h
/// 11:   else
/// 12:     break
/// 13:   end if
/// 14: end while
/// 15: C^H_i ← C^r_i, D^H_i ← D^r_i
/// 16: for each element σ_i,j in σ^L_i do
/// 17:   d_i,j ← l_i,j
/// 18: end for
/// 19: for each element σ_i,j in σ^H_i do
/// 20:   d_i,j ← D^H_i · m_i,j · l_i,j / C^H_i
/// 21: end for
/// ```
///
/// Lines 1–15 are re-run here rather than reusing
/// [`heavy_capacity_and_deadline`] (which computes the exact same
/// `(C^H_i, D^H_i)`): that function only returns the aggregate, having
/// already discarded *which* original segments ended up in `σ^L_i` versus
/// `σ^H_i`, which lines 16–21 need per segment. Duplicating lines 1–15
/// here (instead of changing that function's return type) keeps
/// [`required_capacity`]'s already-tested static-admission path — Phase 0
/// of the real-machine DAG-Fluid work, already wired into
/// `crate::build_dag` and shipped — untouched by this Phase 1 addition.
///
/// Returns `None` under the same conditions as
/// [`heavy_capacity_and_deadline`] (no heavy segment, or a non-positive
/// aggregate) — both call sites should only reach this once
/// `required_capacity` has already confirmed `volume > virtual_deadline`
/// admits at least one heavy segment (Lemma 5.2).
pub fn assign_segment_deadlines(
    segments: &[Segment],
    volume: u64,
    virtual_deadline: u64,
) -> Option<Vec<SegmentSchedule>> {
    let mut order: Vec<usize> = (0..segments.len()).collect();
    order.sort_by_key(|&i| segments[i].concurrency);

    let mut c_r = volume as f64;
    let mut d_r = virtual_deadline as f64;
    let mut light: alloc::collections::BTreeSet<usize> = alloc::collections::BTreeSet::new();

    for &i in &order {
        let seg = &segments[i];
        let m = seg.concurrency as f64;
        if c_r / (d_r * m) > 1.0 {
            light.insert(i);
            c_r -= seg.duration as f64 * m;
            d_r -= seg.duration as f64;
        } else {
            break;
        }
    }

    if light.len() == segments.len() {
        return None;
    }
    if !(c_r > 0.0 && d_r > 0.0) {
        return None;
    }
    let c_heavy = c_r;
    let d_heavy = d_r;

    let mut out = Vec::with_capacity(segments.len());
    for (i, seg) in segments.iter().enumerate() {
        let relative_deadline = if light.contains(&i) {
            seg.duration as f64 // line 17
        } else {
            // line 20
            d_heavy * seg.concurrency as f64 * seg.duration as f64 / c_heavy
        };
        let rate = seg.duration as f64 / relative_deadline; // Definition 4.1
        out.push(SegmentSchedule {
            duration: seg.duration,
            concurrency: seg.concurrency,
            relative_deadline,
            rate,
        });
    }
    Some(out)
}

/// Section 4.2's `r_i,j` (each segment's own release offset, relative to
/// the task's own release/arrival time — not an absolute system time):
/// "we set `r_i,j` equal [to] the release time of `τ_i` plus the sum of
/// all its predecessor segments' relative deadlines." `schedule` must be
/// in the same chronological order [`assign_segment_deadlines`] returns
/// it in; `offsets[j]` is the sum of `schedule[..j]`'s own
/// `relative_deadline`s (`offsets[0] == 0.0`, the first segment starts at
/// the task's own release time).
pub fn segment_release_offsets(schedule: &[SegmentSchedule]) -> Vec<f64> {
    let mut offsets = Vec::with_capacity(schedule.len());
    let mut acc = 0.0;
    for s in schedule {
        offsets.push(acc);
        acc += s.relative_deadline;
    }
    offsets
}

/// Algorithm 2 (2022 paper)'s per-task capacity contribution: `None` if the
/// task is unconditionally infeasible (`critical_path > deadline`, i.e.
/// `L_i > D_i`, violating this crate's own base precondition — mirrors
/// `DagMetrics::min_dedicated_cores`'s own `None` convention for the same
/// condition), or degenerate segment decomposition (see
/// [`heavy_capacity_and_deadline`]'s own doc); otherwise
/// `ceil(D*_i/T_i) * rate_i`, this task's own contribution to
/// [`is_batch_feasible`]'s aggregate sum. `m` does not appear here — see
/// this module's own doc for why.
pub fn required_capacity(
    volume: u64,
    period: u64,
    critical_path: u64,
    deadline: u64,
    segments: &[Segment],
) -> Option<f64> {
    if critical_path > deadline {
        return None;
    }
    let d_star = virtual_deadline(period, deadline, critical_path);

    let rate = if volume <= d_star {
        volume as f64 / d_star as f64
    } else {
        let (c_heavy, d_heavy) = heavy_capacity_and_deadline(segments, volume, d_star)?;
        c_heavy / d_heavy
    };

    // Exact integer ceiling division (`f64::ceil` needs `std`/`libm`,
    // unavailable in this crate's actual `no_std` kernel build).
    let concurrent_jobs = d_star.div_ceil(period) as f64;
    Some(concurrent_jobs * rate)
}

/// A whole task set of `(volume, period, critical_path, deadline,
/// segments)` tuples is feasible on `m` shared cores iff
/// `sum(required_capacity_i) <= m` — a direct transcription of Algorithm
/// 2's own accumulation loop (2022 paper), not an inferred composition:
/// `m` is only ever compared against the aggregate, never threaded into any
/// individual task's own computation (see [`required_capacity`]'s own doc).
pub fn is_batch_feasible(entries: &[(u64, u64, u64, u64, &[Segment])], m: u16) -> bool {
    let mut delta = 0.0;
    for &(volume, period, critical_path, deadline, segments) in entries {
        match required_capacity(volume, period, critical_path, deadline, segments) {
            Some(c) => delta += c,
            None => return false,
        }
    }
    delta <= m as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dag_stats::compute_dag_stats;
    use crate::parse_yaml::parse_dags;

    /// Algorithm 1's own worked example (2022 paper, Fig. 1 and its
    /// caption): `C_i=120, L_i=60, T_i=90, D_i=80`, five segments
    /// `(l_i,j, m_i,j)` = `(10,1), (10,2), (20,3), (10,2), (10,1)` in
    /// timeline order. `T_i/D_i = 90/80 = 1.125 >= 1` selects Table 1's
    /// row 1, so `D*_i = D_i = 80` (confirmed by
    /// `virtual_deadline(90, 80, 60) == 80` below). The paper's own text
    /// states the resulting `D^H_i=60, C^H_i=100` directly (Fig. 1's
    /// caption), and Fig. 1d gives `d_i,j` = 10, 12, 36, 12, 10 and
    /// `θ_i,j` = 1, 5/6, 5/9, 5/6, 1 for the five segments in order —
    /// every value checked here is transcribed from the primary source,
    /// not derived.
    #[test]
    fn test_assign_segment_deadlines_matches_paper_worked_example() {
        let segments = [
            Segment { duration: 10, concurrency: 1 },
            Segment { duration: 10, concurrency: 2 },
            Segment { duration: 20, concurrency: 3 },
            Segment { duration: 10, concurrency: 2 },
            Segment { duration: 10, concurrency: 1 },
        ];
        let volume = 120;
        let d_star = virtual_deadline(90, 80, 60);
        assert_eq!(d_star, 80);

        let schedule = assign_segment_deadlines(&segments, volume, d_star).unwrap();
        let deadlines: Vec<f64> = schedule.iter().map(|s| s.relative_deadline).collect();
        assert_eq!(deadlines, alloc::vec![10.0, 12.0, 36.0, 12.0, 10.0]);

        let rates: Vec<f64> = schedule.iter().map(|s| s.rate).collect();
        assert_eq!(rates, alloc::vec![1.0, 10.0 / 12.0, 20.0 / 36.0, 10.0 / 12.0, 1.0]);
        // Paper's own Definition 4.1 states the light-segment rate as
        // exactly 5/6 and 5/9 -- confirm the fractions reduce to those.
        assert!((rates[1] - 5.0 / 6.0).abs() < 1e-9);
        assert!((rates[2] - 5.0 / 9.0).abs() < 1e-9);

        let offsets = segment_release_offsets(&schedule);
        assert_eq!(offsets, alloc::vec![0.0, 10.0, 22.0, 58.0, 70.0]);
        // Sum of every segment's own relative deadline equals D*_i (Eq. 3).
        let total: f64 = deadlines.iter().sum();
        assert!((total - d_star as f64).abs() < 1e-9);
    }

    #[test]
    fn test_decompose_segments_linear_chain() {
        // Same fixture as dag_stats::tests::test_compute_dag_stats_chain:
        // 0 --10--> 1 --20--> 2 (single path, never concurrent).
        let dag_file = "links:
  - source: 0
    target: 1
  - source: 1
    target: 2
nodes:
  - execution_time: 10
    id: 0
    period: 50
  - execution_time: 20
    id: 1
  - end_to_end_deadline: 40
    execution_time: 5
    id: 2
";
        let dags = parse_dags(&[dag_file]).unwrap();
        let segments = decompose_segments(&dags[0]);
        assert!(segments.iter().all(|s| s.concurrency == 1));
        let total_duration: u64 = segments.iter().map(|s| s.duration).sum();
        let total_work: u64 = segments.iter().map(|s| s.duration * s.concurrency as u64).sum();
        let stats = compute_dag_stats(&dags[0]);
        assert_eq!(total_duration, stats.critical_path);
        assert_eq!(total_work, stats.volume);
    }

    #[test]
    fn test_decompose_segments_diamond_matches_dag_stats() {
        // Same fixture as dag_stats::tests::test_compute_dag_stats_diamond.
        // Hand-derived under infinite processors: start/finish = 0(node0,
        // s=0,f=10), node1(s=10,f=40), node2(s=10,f=15), node3(s=40,f=45).
        // Segments: [0,10) concurrency=1 (node0); [10,15) concurrency=2
        // (node1,node2); [15,40) concurrency=1 (node1 only, node2 already
        // finished at 15); [40,45) concurrency=1 (node3).
        let dag_file = "links:
  - source: 0
    target: 1
  - source: 0
    target: 2
  - source: 1
    target: 3
  - source: 2
    target: 3
nodes:
  - execution_time: 10
    id: 0
    period: 50
  - execution_time: 30
    id: 1
  - execution_time: 5
    id: 2
  - end_to_end_deadline: 100
    execution_time: 5
    id: 3
";
        let dags = parse_dags(&[dag_file]).unwrap();
        let segments = decompose_segments(&dags[0]);
        assert_eq!(
            segments,
            alloc::vec![
                Segment { duration: 10, concurrency: 1 },
                Segment { duration: 5, concurrency: 2 },
                Segment { duration: 25, concurrency: 1 },
                Segment { duration: 5, concurrency: 1 },
            ]
        );

        let total_duration: u64 = segments.iter().map(|s| s.duration).sum();
        let total_work: u64 = segments.iter().map(|s| s.duration * s.concurrency as u64).sum();
        let stats = compute_dag_stats(&dags[0]);
        assert_eq!(total_duration, stats.critical_path); // 45
        assert_eq!(total_work, stats.volume); // 50
    }

    #[test]
    fn test_virtual_deadline_row1_constrained_deadline() {
        // T_i/D_i = 20/10 = 2 >= 1 -> row 1, D*_i = D_i. This is the only
        // row this crate's own pools ever exercise (T >= D unconditionally).
        assert_eq!(virtual_deadline(20, 10, 8), 10);
    }

    #[test]
    fn test_virtual_deadline_row2_arbitrary_deadline_long_critical_path() {
        // T_i/D_i = 50/100 = 0.5 < 1, and L_i(60) > T_i(50) -> row 2,
        // D*_i = D_i regardless of the ratio.
        assert_eq!(virtual_deadline(50, 100, 60), 100);
    }

    #[test]
    fn test_virtual_deadline_row3() {
        // T_i/D_i = 30/100 = 0.3 <= (sqrt(3)-1)/2 (~0.366), L_i(20) <=
        // T_i(30) -> row 3, D*_i = D_i.
        assert_eq!(virtual_deadline(30, 100, 20), 100);
    }

    #[test]
    fn test_virtual_deadline_row4() {
        // T_i/D_i = 45/100 = 0.45, in ((sqrt(3)-1)/2, 0.5], L_i(40) <=
        // T_i(45) -> row 4, D*_i = 2*T_i = 90.
        assert_eq!(virtual_deadline(45, 100, 40), 90);
    }

    #[test]
    fn test_virtual_deadline_row5() {
        // T_i/D_i = 60/100 = 0.6, in (0.5, sqrt(2)/2 (~0.707)], L_i(50) <=
        // T_i(60) -> row 5, D*_i = D_i.
        assert_eq!(virtual_deadline(60, 100, 50), 100);
    }

    #[test]
    fn test_virtual_deadline_row6() {
        // T_i/D_i = 90/100 = 0.9, in (sqrt(2)/2, 1), L_i(80) <= T_i(90) ->
        // row 6, D*_i = T_i = 90.
        assert_eq!(virtual_deadline(90, 100, 80), 90);
    }

    #[test]
    fn test_heavy_capacity_and_deadline_hand_computed() {
        // Segments (5,conc=1), (3,conc=2), (4,conc=1); volume=15,
        // virtual_deadline=14 (chosen >= critical_path=5+3+4=12, as Table 1
        // guarantees D*_i >= L_i in practice).
        //
        // Sorted by concurrency ascending, the two concurrency=1 segments
        // come first (stable sort keeps their original relative order:
        // (5,1) then (4,1)), then (3,2).
        //   c_r=15, d_r=14. seg(5,1): 15/(14*1)=1.071>1 -> peel:
        //     c_r=15-5=10, d_r=14-5=9.
        //   seg(4,1): 10/(9*1)=1.111>1 -> peel: c_r=10-4=6, d_r=9-4=5.
        //   seg(3,2): 6/(5*2)=0.6<=1 -> stop. Heavy = {(3,2)}.
        // C^H_i=6, D^H_i=5 (order-independent: both concurrency=1 segments
        // get peeled before the concurrency=2 one is ever tested).
        let segments = [
            Segment { duration: 5, concurrency: 1 },
            Segment { duration: 3, concurrency: 2 },
            Segment { duration: 4, concurrency: 1 },
        ];
        assert_eq!(heavy_capacity_and_deadline(&segments, 15, 14), Some((6.0, 5.0)));
    }

    #[test]
    fn test_heavy_capacity_and_deadline_no_peeling_when_first_segment_already_heavy() {
        // Single segment (10, concurrency=1), volume=10, virtual_deadline=20:
        // 10/(20*1)=0.5<=1 -> stop immediately, nothing peeled. Heavy = the
        // whole (single) segment set.
        let segments = [Segment { duration: 10, concurrency: 1 }];
        assert_eq!(heavy_capacity_and_deadline(&segments, 10, 20), Some((10.0, 20.0)));
    }

    #[test]
    fn test_heavy_capacity_and_deadline_none_when_fully_peeled() {
        // Single segment (10, concurrency=1), volume=10, virtual_deadline=5:
        // 10/(5*1)=2>1 -> peeled as light, and it was the only segment ->
        // no heavy segment remains (a degenerate input Lemma 5.2 says can't
        // arise from a real volume > virtual_deadline task, but the
        // function guards it defensively regardless).
        let segments = [Segment { duration: 10, concurrency: 1 }];
        assert_eq!(heavy_capacity_and_deadline(&segments, 10, 5), None);
    }

    #[test]
    fn test_required_capacity_tau_seq() {
        // period=20, deadline=10 -> ratio=2>=1 -> row 1, D*_i=10.
        // volume=8<=D*_i=10 -> tau_seq, rate=8/10=0.8.
        // concurrent_jobs=ceil(D*_i/T_i)=ceil(10/20)=1.
        let segments = [Segment { duration: 5, concurrency: 1 }];
        let got = required_capacity(8, 20, 5, 10, &segments).unwrap();
        assert!((got - 0.8).abs() < 1e-9);
    }

    #[test]
    fn test_required_capacity_tau_paral_uses_heavy_capacity() {
        // period=20, deadline=14 -> ratio=20/14>=1 -> row 1, D*_i=14.
        // Reuses test_heavy_capacity_and_deadline_hand_computed's segments
        // (critical_path=5+3+4=12<=deadline=14, volume=15>D*_i=14 ->
        // tau_paral). C^H_i/D^H_i=6/5=1.2.
        // concurrent_jobs=ceil(14/20)=1.
        let segments = [
            Segment { duration: 5, concurrency: 1 },
            Segment { duration: 3, concurrency: 2 },
            Segment { duration: 4, concurrency: 1 },
        ];
        let got = required_capacity(15, 20, 12, 14, &segments).unwrap();
        assert!((got - 1.2).abs() < 1e-9);
    }

    #[test]
    fn test_required_capacity_none_when_critical_path_exceeds_deadline() {
        let segments = [Segment { duration: 50, concurrency: 2 }];
        assert!(required_capacity(10, 100, 50, 40, &segments).is_none());
    }

    #[test]
    fn test_is_batch_feasible_sums_required_capacity() {
        // The tau_seq (0.8) and tau_paral (1.2) examples above sum to 2.0.
        let seq_segments = [Segment { duration: 5, concurrency: 1 }];
        let paral_segments = [
            Segment { duration: 5, concurrency: 1 },
            Segment { duration: 3, concurrency: 2 },
            Segment { duration: 4, concurrency: 1 },
        ];
        let entries: [(u64, u64, u64, u64, &[Segment]); 2] = [
            (8, 20, 5, 10, &seq_segments),
            (15, 20, 12, 14, &paral_segments),
        ];
        assert!(is_batch_feasible(&entries, 2));
        assert!(!is_batch_feasible(&entries, 1));
    }

    #[test]
    fn test_is_batch_feasible_rejects_when_any_entry_is_infeasible() {
        let ok_segments = [Segment { duration: 5, concurrency: 1 }];
        let bad_segments = [Segment { duration: 50, concurrency: 2 }];
        let entries: [(u64, u64, u64, u64, &[Segment]); 2] = [
            (8, 20, 5, 10, &ok_segments),
            (10, 100, 50, 40, &bad_segments), // critical_path > deadline
        ];
        assert!(!is_batch_feasible(&entries, 100));
    }
}
