//! DAG-Fluid's planner for the per-core dispatch (方式B, Step 4): a task that
//! wakes at the start of every span, lays out the DP-Wrap plan of the
//! *next* span, and publishes each DAG-pool CPU's share of it to that CPU's
//! switch table (`scheduler::dp_wrap::switch_table`).
//!
//! # Windows and Deadline Partitions
//! Releases are strictly periodic (`job_table`): job `k` of a DAG is
//! released at `r_0 + k * T`. A *window* runs from one release (of any DAG)
//! to the next. Inside a window, every segment start `r_k + offset_j` and
//! end `r_k + offset_j + d_j` of every alive job is a Deadline Partition
//! boundary (TC 2020 Sect. 8, TC 2022 Sect. 4.2: the papers replace the
//! dependencies with these times). Within each DP, McNaughton's wrap
//! (`dp_wrap::wrap_blocks`) lays out the active segments' rates (`m_i,j`
//! copies of `theta_i,j`) on the pool's CPUs, and multiplying by the DP's
//! length gives each CPU's switch points. A CPU's plan for a window is the
//! concatenation over its DPs; trailing idle time is a switch to no DAG.
//!
//! All DAGs are planned together: CPUs are shared by rate, and two DAGs
//! sharing a CPU must cut it at the same DP boundaries, or their slices on
//! that CPU would overlap. A release of one DAG therefore moves the other
//! DAGs' switch points too.
//!
//! # Per-node slots (the `dagfluid_node` feature)
//! By default a block of the wrap is one of a segment's `m_i,j` threads,
//! and its slot goes to the DAG as a whole. With `dagfluid_node`, the
//! threads are the segment's active nodes (`SegmentTemplate::active_nodes`,
//! the nodes running in it on the infinite-processor timeline): each gets
//! one block of length `theta_i,j`, and its slot names that node and the
//! job it belongs to (`switch_table::NodeSlot`). A node active in several
//! segments gets a block in each of them; DP-Wrap may place those blocks on
//! different CPUs.
//!
//! # Spans (one plane each)
//! Releases of different DAGs can fall close together, leaving windows of a
//! few milliseconds. Since every release is known in advance, consecutive
//! windows are joined into one *span* until it reaches [`min_span`] (half
//! the shortest period); a span stops early before any CPU would exceed
//! `switch_table::SWITCH_TABLE_CAPACITY` entries. Joining changes nothing
//! in the plan itself, only how often the planner wakes and how much slack
//! it has: a full span instead of the shortest window.
//!
//! # Waking
//! The planner sleeps until the span it published last starts (a
//! time-driven wake: it needs no notification from the source nodes). By
//! then each CPU has moved on to that span, so the planner may fill the
//! other plane with the span after it.
//!
//! # Event-driven variant (③, the `dagfluid_event` feature)
//! Releases stay time-driven, but segment `j + 1` of a job arrives when
//! segment `j`'s completion gate has finished (`dag_sched::event_driven`),
//! and must finish by its arrival plus `d_j+1`. Future boundaries are
//! therefore unknown: the planner instead wakes at every segment completion
//! (notified), every release and every segment deadline, lays out only the
//! current Deadline Partition (until the next known release or deadline),
//! publishes it to every CPU and kicks them to replace what is left of
//! their current plane (`dp_switch::kick_cpu_replace`). A segment still
//! running past its deadline keeps its rate until it completes (the
//! papers leave this open); such overruns are counted and logged.
//!
//! # Before the first release (bootstrap)
//! A source node is a DP-Wrap task too, so it cannot release its first job
//! before its DAG is entitled to some CPU. Until a DAG's first release shows
//! up in its ring, the planner assumes it was released when the planner
//! started, the same idea as the old path's `apply_initial_entitlement`.
//! While any DAG is still in that state, the planner also wakes every
//! [`BOOTSTRAP_POLL`] so it switches to the real release grid promptly.
//!
//! # Recovery
//! A CPU whose plan ran out (`switch_table::take_cpu_starved`, also true
//! before its first plan) gets the rest of the current span and the next
//! span in one plane, and a kick (`dp_switch::kick_cpu`) so it arms its
//! own timer again. If the CPU ran out just after the planner had already
//! published its next plane (the CPU never polled again, so that plane
//! stays unread), publishing fails with `ReaderBehind`; the kick alone then
//! makes the CPU move on to it. Any other failure (the plane overflowing)
//! is logged; the CPU then runs out of plan and is recovered the same way
//! at the next wake.

use super::job_table::{self, JobRing, SegmentTemplate};
use crate::scheduler::dp_wrap::{
    dp_switch,
    switch_table::{self, NodeSlot, SwitchEntry, SwitchTableError, SWITCH_TABLE_CAPACITY},
    wrap_blocks,
};
use alloc::{borrow::Cow, sync::Arc, vec, vec::Vec};
use awkernel_lib::time::Instant;
use core::time::Duration;

/// How often the planner looks for first releases while some DAG has not
/// released yet.
const BOOTSTRAP_POLL: Duration = Duration::from_millis(1);

/// Fractions this close to a chunk's end count as the end (`wrap_blocks`
/// works in `f64`).
const FRACTION_EPSILON: f64 = 1e-9;

/// Whether slots go to single nodes (the `dagfluid_node` feature) rather
/// than to whole DAGs. See the module doc.
const PER_NODE: bool = cfg!(feature = "dagfluid_node");

/// One active segment of one job inside a DP, as [`plan_dp`] lays it out.
#[derive(Debug, Clone, Copy)]
pub struct ActiveSegment<'a> {
    pub dag_id: u32,
    pub job: u64,
    /// `m_i,j`: the number of threads, used when `nodes` is empty.
    pub concurrency: u32,
    /// `theta_i,j`: each thread's rate.
    pub rate: f64,
    /// The threads as nodes, for per-node slots; empty for per-DAG slots.
    pub nodes: &'a [u32],
}

/// One DAG as the plan computation sees it.
#[derive(Debug, Clone, Copy)]
pub struct PlanDag<'a> {
    pub dag_id: u32,
    pub period: Duration,
    pub virtual_deadline: Duration,
    pub segments: &'a [SegmentTemplate],
    /// Release of job 0. Job `k` is released at `anchor + k * period`.
    pub anchor: Instant,
}

fn nanos_between(later: Instant, earlier: Instant) -> u128 {
    later.saturating_duration_since(earlier).as_nanos()
}

impl PlanDag<'_> {
    fn release(&self, job: u64) -> Option<Instant> {
        let job = u32::try_from(job).ok()?;
        Some(self.anchor + self.period.checked_mul(job)?)
    }

    /// The first job released strictly after `t`.
    fn first_job_after(&self, t: Instant) -> Option<u64> {
        if t < self.anchor {
            return Some(0);
        }
        let jobs = nanos_between(t, self.anchor).checked_div(self.period.as_nanos())?;
        u64::try_from(jobs).ok()?.checked_add(1)
    }

    /// Jobs alive at some point of `[start, end)`: `r_k < end` and
    /// `r_k + D* > start`.
    fn jobs_overlapping(&self, start: Instant, end: Instant) -> core::ops::Range<u64> {
        let period = self.period.as_nanos();
        if period == 0 || end <= self.anchor {
            return 0..0;
        }
        // Last job with r_k < end.
        let to_end = nanos_between(end, self.anchor);
        let last_plus_one = to_end.div_ceil(period);
        // First job with r_k + D* > start.
        let since = nanos_between(start, self.anchor);
        let d_star = self.virtual_deadline.as_nanos();
        let first = if since < d_star {
            0
        } else {
            (since - d_star) / period + 1
        };
        let first = u64::try_from(first).unwrap_or(u64::MAX);
        let last_plus_one = u64::try_from(last_plus_one).unwrap_or(u64::MAX);
        first..last_plus_one.max(first)
    }
}

/// The earliest release of any DAG strictly after `t`.
pub fn next_release_after(dags: &[PlanDag<'_>], t: Instant) -> Option<Instant> {
    dags.iter()
        .filter_map(|dag| dag.release(dag.first_job_after(t)?))
        .min()
}

/// Each pool CPU's switch points for `[start, end)` (index = position in
/// the pool, not the CPU id). See the module doc.
pub fn plan_window(
    dags: &[PlanDag<'_>],
    start: Instant,
    end: Instant,
    num_cpus: usize,
) -> Vec<Vec<SwitchEntry>> {
    plan_window_with(dags, start, end, num_cpus, PER_NODE)
}

/// [`plan_window`] with the slot granularity given explicitly (per node if
/// `per_node`, else per DAG).
fn plan_window_with(
    dags: &[PlanDag<'_>],
    start: Instant,
    end: Instant,
    num_cpus: usize,
    per_node: bool,
) -> Vec<Vec<SwitchEntry>> {
    let mut plans: Vec<Vec<SwitchEntry>> = vec![Vec::new(); num_cpus];
    if end <= start || num_cpus == 0 {
        return plans;
    }

    // DP boundaries inside the window.
    let mut cuts = vec![start, end];
    for dag in dags {
        for job in dag.jobs_overlapping(start, end) {
            let Some(release) = dag.release(job) else {
                continue;
            };
            for segment in dag.segments {
                let seg_start = release + segment.offset;
                let seg_end = seg_start + segment.relative_deadline;
                for cut in [seg_start, seg_end] {
                    if start < cut && cut < end {
                        cuts.push(cut);
                    }
                }
            }
        }
    }
    cuts.sort_unstable();
    cuts.dedup();

    let mut active: Vec<ActiveSegment<'_>> = Vec::new();
    for dp in cuts.windows(2) {
        let (Some(&dp_start), Some(&dp_end)) = (dp.first(), dp.get(1)) else {
            continue;
        };
        let dp_len = dp_end.saturating_duration_since(dp_start);
        if dp_len.is_zero() {
            continue;
        }

        active.clear();
        for dag in dags {
            for job in dag.jobs_overlapping(dp_start, dp_end) {
                let Some(release) = dag.release(job) else {
                    continue;
                };
                for segment in dag.segments {
                    let seg_start = release + segment.offset;
                    let seg_end = seg_start + segment.relative_deadline;
                    if seg_start <= dp_start && dp_start < seg_end {
                        active.push(ActiveSegment {
                            dag_id: dag.dag_id,
                            job,
                            concurrency: segment.concurrency,
                            rate: segment.rate,
                            nodes: if per_node { &segment.active_nodes } else { &[] },
                        });
                    }
                }
            }
        }
        plan_dp(&active, dp_start, dp_end, &mut plans);
    }
    plans
}

/// Append each CPU's switch points for one DP `[dp_start, dp_end)` running
/// the `active` segments to `plans` (one per pool CPU): McNaughton's wrap
/// of one block of length `rate` per thread (each of `nodes`, or
/// `concurrency` anonymous ones if `nodes` is empty), scaled by the DP's
/// length; trailing idle time is a switch to no DAG.
pub fn plan_dp(
    active: &[ActiveSegment<'_>],
    dp_start: Instant,
    dp_end: Instant,
    plans: &mut [Vec<SwitchEntry>],
) {
    let dp_len = dp_end.saturating_duration_since(dp_start);
    if dp_len.is_zero() {
        return;
    }
    // The owner of each block: its DAG, and its node slot if per node.
    let mut owners: Vec<(u32, Option<NodeSlot>)> = Vec::new();
    let mut densities: Vec<(usize, f64)> = Vec::new();
    for segment in active {
        if segment.nodes.is_empty() {
            for _ in 0..segment.concurrency {
                densities.push((owners.len(), segment.rate));
                owners.push((segment.dag_id, None));
            }
        } else {
            for &node_id in segment.nodes {
                densities.push((owners.len(), segment.rate));
                owners.push((
                    segment.dag_id,
                    Some(NodeSlot {
                        node_id,
                        job: segment.job,
                    }),
                ));
            }
        }
    }

    let blocks = wrap_blocks(&densities, plans.len());
    let at = |fraction: f64| {
        Duration::try_from_secs_f64(dp_len.as_secs_f64() * fraction)
            .ok()
            .map(|offset| dp_start + offset)
    };
    for (plan, chunk) in plans.iter_mut().zip(blocks.iter()) {
        let mut used = 0.0_f64;
        for &(owner, from, to) in chunk {
            if let (Some(time), Some(&(dag_id, node))) = (at(from), owners.get(owner)) {
                plan.push(SwitchEntry {
                    at: time,
                    dag_id: Some(dag_id),
                    node,
                });
            }
            used = to;
        }
        if used < 1.0 - FRACTION_EPSILON {
            if let Some(time) = at(used) {
                plan.push(SwitchEntry::dag(time, None));
            }
        }
    }
}

/// Half the shortest period: the length a span is joined up to.
pub fn min_span(dags: &[PlanDag<'_>]) -> Duration {
    dags.iter()
        .map(|dag| dag.period)
        .min()
        .unwrap_or_default()
        .checked_div(2)
        .unwrap_or_default()
        .max(BOOTSTRAP_POLL)
}

/// Join windows from `start` into one span (see the module doc) and return
/// its end with each CPU's plan for it. `None` if no release follows.
pub fn plan_span(
    dags: &[PlanDag<'_>],
    start: Instant,
    min_len: Duration,
    num_cpus: usize,
) -> Option<(Instant, Vec<Vec<SwitchEntry>>)> {
    let mut end = start;
    let mut plans: Vec<Vec<SwitchEntry>> = vec![Vec::new(); num_cpus];
    loop {
        let window_end = next_release_after(dags, end)?;
        let window = plan_window(dags, end, window_end, num_cpus);
        let fits = plans
            .iter()
            .zip(window.iter())
            .all(|(have, add)| have.len().saturating_add(add.len()) <= SWITCH_TABLE_CAPACITY);
        if !fits && end > start {
            return Some((end, plans));
        }
        for (have, add) in plans.iter_mut().zip(window) {
            have.extend(add);
        }
        end = window_end;
        if end.saturating_duration_since(start) >= min_len {
            return Some((end, plans));
        }
    }
}

/// Where a DAG's job 0 was released, from its newest recorded release.
fn anchor_of(ring: &JobRing) -> Option<Instant> {
    let latest = ring.latest().ok()??;
    let back = ring
        .template()
        .period()
        .checked_mul(u32::try_from(latest.job).ok()?)?;
    latest.release.checked_sub(back)
}

/// Start the planner task. Call once, after every DAG-Fluid DAG has been
/// admitted.
pub fn spawn_planner() {
    // Per-node slots stop a node through a placeholder task per CPU.
    crate::scheduler::dp_wrap::spawn_placeholders(super::resource::dagfluid_pool_cpu_set().iter());
    crate::task::spawn(
        "dag_fluid planner".into(),
        #[cfg(not(feature = "dagfluid_event"))]
        run_planner(),
        #[cfg(feature = "dagfluid_event")]
        run_planner_event(),
        crate::scheduler::SchedulerType::PrioritizedFIFO(31),
    );
}

#[cfg(not(feature = "dagfluid_event"))]
async fn run_planner() -> Result<(), Cow<'static, str>> {
    let rings: Vec<(u32, Arc<JobRing>)> = job_table::registered_rings();
    let pool: Vec<usize> = super::resource::dagfluid_pool_cpu_set().iter().collect();
    if rings.is_empty() || pool.is_empty() {
        log::warn!(
            "dag_fluid planner: nothing to plan ({} DAGs, {} pool CPUs)",
            rings.len(),
            pool.len()
        );
        return Ok(());
    }
    log::info!(
        "dag_fluid planner: {} DAGs on pool CPUs {pool:?}",
        rings.len()
    );

    let boot = Instant::now();
    let mut anchors: Vec<Option<Instant>> = vec![None; rings.len()];
    // Start and end of the newest span published to every CPU.
    let mut newest: Option<(Instant, Instant)> = None;

    loop {
        let now = Instant::now();
        for (anchor, (_, ring)) in anchors.iter_mut().zip(rings.iter()) {
            if let Some(real) = anchor_of(ring) {
                *anchor = Some(real);
            }
        }
        let all_released = anchors.iter().all(Option::is_some);
        let dags: Vec<PlanDag<'_>> = rings
            .iter()
            .zip(anchors.iter())
            .map(|((dag_id, ring), anchor)| PlanDag {
                dag_id: *dag_id,
                period: ring.template().period(),
                virtual_deadline: ring.template().virtual_deadline(),
                segments: ring.template().segments(),
                anchor: anchor.unwrap_or(boot),
            })
            .collect();
        let min_len = min_span(&dags);

        let starved: Vec<bool> = pool
            .iter()
            .map(|&cpu_id| switch_table::take_cpu_starved(cpu_id).unwrap_or(false))
            .collect();

        // Everything up to the end of the newest published span is already
        // in the CPUs' tables, unless that end has passed (or nothing was
        // published yet): new plans start there.
        let frontier = match newest {
            Some((_, end)) if end > now => end,
            _ => now,
        };
        // Publish the span after the newest one once the CPUs have reached
        // the newest one (see the module doc's "Waking"), or right away if
        // there is none ahead of them.
        let due = newest.is_none_or(|(start, end)| start <= now || end <= now);
        if !due && !starved.contains(&true) {
            let wake = newest.map_or(now, |(start, _)| start);
            crate::sleep_until(if all_released {
                wake
            } else {
                wake.min(now + BOOTSTRAP_POLL)
            })
            .await;
            continue;
        }

        // A new span goes to every CPU only when due; a starved CPU always
        // gets what is left up to the frontier (what the others already
        // hold), plus the new span when one is made.
        let span = if due {
            let Some(span) = plan_span(&dags, frontier, min_len, pool.len()) else {
                log::warn!("dag_fluid planner: no future release; stopping");
                return Ok(());
            };
            Some(span)
        } else {
            None
        };
        let rest = plan_window(&dags, now, frontier, pool.len());

        for (p, &cpu_id) in pool.iter().enumerate() {
            let plan = span.as_ref().and_then(|(_, plans)| plans.get(p));
            let (Some(rest), Some(&starved)) = (rest.get(p), starved.get(p)) else {
                continue;
            };
            if starved {
                // Nothing left to read: the rest of the current span (and
                // the next one) in one plane, then make the CPU arm its
                // timer.
                let mut entries = rest.clone();
                if let Some(plan) = plan {
                    entries.extend_from_slice(plan);
                }
                match switch_table::publish_cpu_plan(cpu_id, &entries) {
                    Ok(_) => dp_switch::kick_cpu(cpu_id),
                    // A plane was published after the CPU ran out but it
                    // never polled again (its timer is off): the kick makes
                    // it move on to that plane and arm its timer.
                    Err(e @ SwitchTableError::ReaderBehind { .. }) => {
                        log_publish_error(cpu_id, e);
                        dp_switch::kick_cpu(cpu_id);
                    }
                    Err(e) => log_publish_error(cpu_id, e),
                }
            } else if let Some(plan) = plan {
                if let Err(e) = switch_table::publish_cpu_plan(cpu_id, plan) {
                    log_publish_error(cpu_id, e);
                }
            }
        }
        if let Some((span_end, _)) = span {
            newest = Some((frontier, span_end));
        }

        // Wake when the CPUs reach the newest span published.
        let wake = newest.map_or(now, |(start, _)| start);
        crate::sleep_until(if all_released {
            wake
        } else {
            wake.min(now + BOOTSTRAP_POLL)
        })
        .await;
    }
}

/// Where one job stands in the event-driven variant (③).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EventJobState {
    /// `segment` arrived at `arrival` and has not completed; it is due by
    /// `deadline`.
    Active {
        segment: usize,
        arrival: Instant,
        deadline: Instant,
    },
    /// Every segment has completed.
    Done,
}

/// ③: the state of a job released at `release`, given when each of its
/// segments completed (`event_driven::completed_segments`). Segment 0
/// arrives at the release; segment `j + 1` when segment `j` completed (or
/// arrived, if it completed even earlier).
pub fn event_job_state(
    segments: &[SegmentTemplate],
    release: Instant,
    complete_at: &[Option<Instant>],
) -> EventJobState {
    let mut arrival = release;
    for (segment, template) in segments.iter().enumerate() {
        match complete_at.get(segment).copied().flatten() {
            Some(done) => arrival = arrival.max(done),
            None => {
                return EventJobState::Active {
                    segment,
                    arrival,
                    deadline: arrival + template.relative_deadline,
                };
            }
        }
    }
    EventJobState::Done
}

/// How long the event-driven planner waits for a CPU to take its previous
/// plane before re-planning (`ReaderBehind`).
#[cfg(feature = "dagfluid_event")]
const REPLACE_RETRY: Duration = Duration::from_micros(50);

#[cfg(feature = "dagfluid_event")]
async fn run_planner_event() -> Result<(), Cow<'static, str>> {
    use super::event_driven;
    use alloc::collections::BTreeSet;
    use core::pin::pin;

    let rings: Vec<(u32, Arc<JobRing>)> = job_table::registered_rings();
    let pool: Vec<usize> = super::resource::dagfluid_pool_cpu_set().iter().collect();
    if rings.is_empty() || pool.is_empty() {
        log::warn!(
            "dag_fluid planner (event): nothing to plan ({} DAGs, {} pool CPUs)",
            rings.len(),
            pool.len()
        );
        return Ok(());
    }
    log::info!(
        "dag_fluid planner (event): {} DAGs on pool CPUs {pool:?}",
        rings.len()
    );

    let boot = Instant::now();
    let mut anchors: Vec<Option<Instant>> = vec![None; rings.len()];
    // Per DAG: the oldest job not known to be done.
    let mut first_open: Vec<u64> = vec![0; rings.len()];
    // Segments seen running past their deadline: (dag_id, job, segment).
    let mut overruns: BTreeSet<(u32, u64, usize)> = BTreeSet::new();
    let mut overruns_logged = 0;
    let mut last_log = boot;

    loop {
        let now = Instant::now();
        for (anchor, (_, ring)) in anchors.iter_mut().zip(rings.iter()) {
            if let Some(real) = anchor_of(ring) {
                *anchor = Some(real);
            }
        }
        let all_released = anchors.iter().all(Option::is_some);
        let dags: Vec<PlanDag<'_>> = rings
            .iter()
            .zip(anchors.iter())
            .map(|((dag_id, ring), anchor)| PlanDag {
                dag_id: *dag_id,
                period: ring.template().period(),
                virtual_deadline: ring.template().virtual_deadline(),
                segments: ring.template().segments(),
                anchor: anchor.unwrap_or(boot),
            })
            .collect();

        // The active segment of every released, unfinished job.
        let mut active: Vec<ActiveSegment<'_>> = Vec::new();
        let mut next_event = next_release_after(&dags, now);
        for (dag, first) in dags.iter().zip(first_open.iter_mut()) {
            let Some(last) = dag.first_job_after(now).and_then(|k| k.checked_sub(1)) else {
                continue;
            };
            let mut job = *first;
            while job <= last {
                let Some(release) = dag.release(job) else {
                    break;
                };
                let done = event_driven::completed_segments(dag.dag_id, job);
                match event_job_state(dag.segments, release, &done) {
                    EventJobState::Done => {
                        if job == *first {
                            *first = job.saturating_add(1);
                            event_driven::forget_jobs_before(dag.dag_id, *first);
                        }
                    }
                    EventJobState::Active {
                        segment, deadline, ..
                    } => {
                        if let Some(template) = dag.segments.get(segment) {
                            active.push(ActiveSegment {
                                dag_id: dag.dag_id,
                                job,
                                concurrency: template.concurrency,
                                rate: template.rate,
                                nodes: &[],
                            });
                        }
                        if deadline > now {
                            next_event = Some(next_event.map_or(deadline, |t| t.min(deadline)));
                        } else {
                            overruns.insert((dag.dag_id, job, segment));
                        }
                    }
                }
                job = job.saturating_add(1);
            }
        }
        overruns.retain(|&(dag_id, job, _)| {
            rings
                .iter()
                .position(|(id, _)| *id == dag_id)
                .and_then(|i| first_open.get(i))
                .is_some_and(|&first| job >= first)
        });

        // Lay out the current DP and replace every CPU's plan with it.
        let end = next_event.unwrap_or(now + BOOTSTRAP_POLL);
        let mut plans: Vec<Vec<SwitchEntry>> = vec![Vec::new(); pool.len()];
        plan_dp(&active, now, end, &mut plans);
        let mut behind = false;
        for (&cpu_id, plan) in pool.iter().zip(plans.iter()) {
            // Taken only to keep the flag fresh: every plan here is followed
            // by a kick anyway.
            let _ = switch_table::take_cpu_starved(cpu_id);
            match switch_table::publish_cpu_plan(cpu_id, plan) {
                Ok(_) => dp_switch::kick_cpu_replace(cpu_id),
                // The CPU has not taken the previous plan yet: re-plan
                // shortly with whatever has changed meanwhile.
                Err(SwitchTableError::ReaderBehind { .. }) => behind = true,
                Err(e) => log_publish_error(cpu_id, e),
            }
        }

        if overruns.len() != overruns_logged
            && now.saturating_duration_since(last_log) >= Duration::from_secs(1)
        {
            log::warn!(
                "dag_fluid planner (event): {} segments running past their deadline",
                overruns.len()
            );
            overruns_logged = overruns.len();
            last_log = now;
        }

        let wake = if behind {
            now + REPLACE_RETRY
        } else if all_released {
            end
        } else {
            end.min(now + BOOTSTRAP_POLL)
        };
        let sleep = pin!(crate::sleep_until(wake));
        let notified = pin!(event_driven::notified());
        futures::future::select(sleep, notified).await;
    }
}

fn log_publish_error(cpu_id: usize, e: SwitchTableError) {
    log::warn!("dag_fluid planner: CPU#{cpu_id} plan not published: {e:?}");
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ms(v: u64) -> Duration {
        Duration::from_millis(v)
    }

    fn at_ms(v: u64) -> Instant {
        Instant::zero() + ms(v)
    }

    fn seg(offset: u64, deadline: u64, concurrency: u32, rate: f64) -> SegmentTemplate {
        SegmentTemplate {
            offset: ms(offset),
            relative_deadline: ms(deadline),
            concurrency,
            rate,
            gate_nodes: Vec::new(),
            active_nodes: Vec::new(),
        }
    }

    fn dag(dag_id: u32, period: u64, segments: &[SegmentTemplate], anchor: u64) -> PlanDag<'_> {
        let virtual_deadline = segments
            .iter()
            .map(|s| s.offset + s.relative_deadline)
            .max()
            .unwrap_or_default();
        PlanDag {
            dag_id,
            period: ms(period),
            virtual_deadline,
            segments,
            anchor: at_ms(anchor),
        }
    }

    fn e(at: u64, dag_id: Option<u32>) -> SwitchEntry {
        SwitchEntry::dag(at_ms(at), dag_id)
    }

    fn n(at: u64, dag_id: u32, node_id: u32, job: u64) -> SwitchEntry {
        SwitchEntry::node(at_ms(at), dag_id, node_id, job)
    }

    fn seg_nodes(offset: u64, deadline: u64, rate: f64, nodes: &[u32]) -> SegmentTemplate {
        SegmentTemplate {
            active_nodes: nodes.to_vec(),
            ..seg(offset, deadline, nodes.len() as u32, rate)
        }
    }

    fn active(dag_id: u32, concurrency: u32, rate: f64) -> ActiveSegment<'static> {
        ActiveSegment {
            dag_id,
            job: 0,
            concurrency,
            rate,
            nodes: &[],
        }
    }

    #[test]
    fn windows_run_between_releases_of_any_dag() {
        let a = [seg(0, 50, 1, 1.0)];
        let b = [seg(0, 50, 1, 1.0)];
        let dags = [dag(1, 100, &a, 0), dag(2, 100, &b, 30)];
        assert_eq!(next_release_after(&dags, at_ms(0)), Some(at_ms(30)));
        assert_eq!(next_release_after(&dags, at_ms(30)), Some(at_ms(100)));
        assert_eq!(next_release_after(&dags, at_ms(100)), Some(at_ms(130)));
    }

    #[test]
    fn single_segment_runs_then_idles() {
        let s = [seg(0, 40, 1, 1.0)];
        let dags = [dag(1, 100, &s, 0)];
        let plans = plan_window(&dags, at_ms(0), at_ms(100), 1);
        assert_eq!(plans, vec![vec![e(0, Some(1)), e(40, None)]]);
    }

    #[test]
    fn rates_share_one_cpu_by_time() {
        // Two DAGs at rate 0.5 each over the DP [0, 100): DAG 1 for the
        // first half, DAG 2 for the second (McNaughton's wrap).
        let a = [seg(0, 100, 1, 0.5)];
        let b = [seg(0, 100, 1, 0.5)];
        let dags = [dag(1, 100, &a, 0), dag(2, 100, &b, 0)];
        let plans = plan_window(&dags, at_ms(0), at_ms(100), 1);
        assert_eq!(plans, vec![vec![e(0, Some(1)), e(50, Some(2))]]);
    }

    #[test]
    fn segment_boundaries_split_the_window_into_dps() {
        // Segment 0: [0, 20) on one CPU at rate 1; segment 1: [20, 60) with
        // two threads at rate 0.5 each, wrapped onto CPU 0 only.
        let s = [seg(0, 20, 1, 1.0), seg(20, 40, 2, 0.5)];
        let dags = [dag(1, 100, &s, 0)];
        let plans = plan_window(&dags, at_ms(0), at_ms(100), 2);
        assert_eq!(
            plans,
            vec![
                vec![e(0, Some(1)), e(20, Some(1)), e(40, Some(1)), e(60, None)],
                vec![e(0, None), e(20, None), e(60, None)],
            ]
        );
    }

    #[test]
    fn migrating_thread_is_split_across_two_cpus() {
        // Rates 0.6 + 0.6 on 2 CPUs: the second thread runs at the end of
        // CPU 0's chunk and the start of CPU 1's.
        let a = [seg(0, 100, 1, 0.6)];
        let b = [seg(0, 100, 1, 0.6)];
        let dags = [dag(1, 100, &a, 0), dag(2, 100, &b, 0)];
        let plans = plan_window(&dags, at_ms(0), at_ms(100), 2);
        assert_eq!(plans[0], vec![e(0, Some(1)), e(60, Some(2))]);
        assert_eq!(plans[1], vec![e(0, Some(2)), e(20, None)]);
    }

    #[test]
    fn overlapping_jobs_of_an_arbitrary_deadline_dag_both_count() {
        // T = 100, D* = 150: in [100, 150) job 0 (segment 1) and job 1
        // (segment 0) are both alive.
        let s = [seg(0, 100, 1, 1.0), seg(100, 50, 1, 1.0)];
        let dags = [dag(1, 100, &s, 0)];
        let d = &dags[0];
        assert_eq!(d.jobs_overlapping(at_ms(100), at_ms(150)), 0..2);
        let plans = plan_window(&dags, at_ms(100), at_ms(200), 2);
        assert_eq!(plans[0], vec![e(100, Some(1)), e(150, Some(1))]);
        assert_eq!(plans[1], vec![e(100, Some(1)), e(150, None)]);
    }

    #[test]
    fn short_windows_are_joined_into_one_span() {
        // Releases at 0, 3 and 100 (period 100): windows [0,3) and [3,100)
        // are joined because the first is shorter than half a period.
        let a = [seg(0, 50, 1, 1.0)];
        let b = [seg(0, 50, 1, 1.0)];
        let dags = [dag(1, 100, &a, 0), dag(2, 100, &b, 3)];
        assert_eq!(min_span(&dags), ms(50));
        let (end, plans) = plan_span(&dags, at_ms(0), ms(50), 2).unwrap();
        assert_eq!(end, at_ms(100));
        // The joined span is exactly the two windows back to back.
        let mut expected = plan_window(&dags, at_ms(0), at_ms(3), 2);
        for (have, add) in expected
            .iter_mut()
            .zip(plan_window(&dags, at_ms(3), at_ms(100), 2))
        {
            have.extend(add);
        }
        assert_eq!(plans, expected);
    }

    #[test]
    fn span_stops_before_a_cpu_would_overflow_its_plane() {
        // 200 releases of a DAG with period 1, so windows are 1 ms long and
        // each adds a few entries; joining up to 1 s would need far more
        // than the plane capacity.
        let s = [seg(0, 1, 1, 0.5)];
        let dags = [dag(1, 1, &s, 0)];
        let (end, plans) = plan_span(&dags, at_ms(0), ms(1_000), 1).unwrap();
        assert!(end < at_ms(1_000));
        assert!(plans[0].len() <= SWITCH_TABLE_CAPACITY);
        assert!(!plans[0].is_empty());
    }

    #[test]
    fn event_job_moves_through_segments_as_they_complete() {
        let s = [seg(0, 20, 1, 1.0), seg(20, 30, 2, 0.5), seg(50, 10, 1, 1.0)];
        let r = at_ms(100);
        assert_eq!(
            event_job_state(&s, r, &[]),
            EventJobState::Active {
                segment: 0,
                arrival: at_ms(100),
                deadline: at_ms(120)
            }
        );
        // Segment 0 completed early at 112: segment 1 arrives then, due 30 ms
        // later (not at the fixed offset 120 / 150).
        assert_eq!(
            event_job_state(&s, r, &[Some(at_ms(112))]),
            EventJobState::Active {
                segment: 1,
                arrival: at_ms(112),
                deadline: at_ms(142)
            }
        );
        assert_eq!(
            event_job_state(
                &s,
                r,
                &[Some(at_ms(112)), Some(at_ms(160)), Some(at_ms(165))]
            ),
            EventJobState::Done
        );
    }

    #[test]
    fn event_arrival_never_goes_back_in_time() {
        // Segment 1 "completed" before segment 0 did (its gate ran early):
        // segment 2 still arrives when segment 0 completed.
        let s = [seg(0, 20, 1, 1.0), seg(20, 30, 1, 1.0), seg(50, 10, 1, 1.0)];
        assert_eq!(
            event_job_state(&s, at_ms(0), &[Some(at_ms(15)), Some(at_ms(10))]),
            EventJobState::Active {
                segment: 2,
                arrival: at_ms(15),
                deadline: at_ms(25)
            }
        );
    }

    #[test]
    fn plan_dp_lays_out_one_partition() {
        let mut plans = vec![Vec::new(), Vec::new()];
        plan_dp(
            &[active(1, 1, 0.6), active(2, 1, 0.6)],
            at_ms(0),
            at_ms(100),
            &mut plans,
        );
        assert_eq!(plans[0], vec![e(0, Some(1)), e(60, Some(2))]);
        assert_eq!(plans[1], vec![e(0, Some(2)), e(20, None)]);
    }

    /// Per-node slots: segment 0 `[0, 20)` has node 0 at rate 1; segment 1
    /// `[20, 60)` has nodes 1 and 3 at rate 0.5 each, wrapped onto CPU 0
    /// (node 1 for the first half of the DP, node 3 for the second).
    #[test]
    fn per_node_slots_name_each_thread() {
        let s = [seg_nodes(0, 20, 1.0, &[0]), seg_nodes(20, 40, 0.5, &[1, 3])];
        let dags = [dag(1, 100, &s, 0)];
        let plans = plan_window_with(&dags, at_ms(0), at_ms(100), 2, true);
        assert_eq!(
            plans,
            vec![
                vec![n(0, 1, 0, 0), n(20, 1, 1, 0), n(40, 1, 3, 0), e(60, None)],
                vec![e(0, None), e(20, None), e(60, None)],
            ]
        );
        // Per DAG, the same layout names only the DAG.
        let plans = plan_window_with(&dags, at_ms(0), at_ms(100), 2, false);
        assert_eq!(
            plans[0],
            vec![e(0, Some(1)), e(20, Some(1)), e(40, Some(1)), e(60, None)]
        );
    }

    /// Each node gets exactly `theta * DP length` of every DP its segment is
    /// active in, summed over the CPUs it lands on, and slots carry the job.
    #[test]
    fn per_node_time_matches_the_rate_and_carries_the_job() {
        // Fig. 1-like segment with three threads at rate 5/9 on 2 CPUs.
        let s = [seg_nodes(0, 36, 5.0 / 9.0, &[2, 3, 4])];
        let dags = [dag(1, 100, &s, 0)];
        // Window of job 1 ([100, 200)).
        let plans = plan_window_with(&dags, at_ms(100), at_ms(200), 2, true);
        let mut per_node = [Duration::ZERO; 5];
        for plan in &plans {
            for pair in plan.windows(2) {
                if let Some(slot) = pair[0].node {
                    assert_eq!(slot.job, 1);
                    per_node[slot.node_id as usize] +=
                        pair[1].at.saturating_duration_since(pair[0].at);
                }
            }
        }
        for node in [2, 3, 4] {
            let got = per_node[node].as_secs_f64();
            assert!((got - 0.020).abs() < 1e-6, "node {node}: {got}");
        }
    }

    #[test]
    fn nothing_alive_means_idle() {
        let s = [seg(0, 10, 1, 1.0)];
        let dags = [dag(1, 100, &s, 1_000)];
        let plans = plan_window(&dags, at_ms(0), at_ms(100), 2);
        assert_eq!(plans, vec![vec![e(0, None)], vec![e(0, None)]]);
    }

    #[test]
    fn jobs_overlapping_respects_the_virtual_deadline() {
        let s = [seg(0, 60, 1, 1.0)];
        let d = dag(1, 100, &s, 0);
        // Job 0 is alive in [0, 60), job 1 in [100, 160).
        assert_eq!(d.jobs_overlapping(at_ms(0), at_ms(50)), 0..1);
        assert_eq!(d.jobs_overlapping(at_ms(70), at_ms(90)).count(), 0);
        assert_eq!(d.jobs_overlapping(at_ms(50), at_ms(110)), 0..2);
    }
}
