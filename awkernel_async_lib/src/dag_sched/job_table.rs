//! Per-DAG periodic job releases for DAG-Fluid's per-core dispatch (方式B,
//! Step 1).
//!
//! The papers (Guan et al., IEEE TC 2020 Section 8 / TC 2022 Section 4.2)
//! anchor every segment of job `k` at that job's own release `r_k`:
//! segment `j` runs over `[r_k + offset_j, r_k + offset_j + d_j)`. This
//! module records those `r_k` as they actually happen, so the planner
//! (Step 4) can lay out the DP-Wrap plan for every job that is still alive,
//! instead of [`super::dp_partition`]'s one-shot registration of the first
//! job only.
//!
//! # What a DAG registers
//! At admission (boot, non-RT) the caller registers a [`DagTemplate`]: the
//! period `T`, and each segment's offset, deadline, `m_i,j`, `theta_i,j` and
//! completion gate. All of it is immutable afterwards, so readers need no
//! lock to look at it.
//!
//! # Release time
//! `r_k` is the tick target `Interval::tick` returns (`next_tick_target`),
//! not the time the source node actually woke up. Releases are therefore
//! strictly periodic, matching the papers' sporadic-with-minimum-period
//! model at its tightest.
//!
//! # Ring, not a growing list
//! Under an arbitrary deadline (`D* > T`, TC 2022 Table 1 rows 2-6) up to
//! `ceil(D* / T)` jobs of one DAG are alive at once. Each DAG gets a ring of
//! `ceil(D* / T) + 1` slots, sized at admission; a release overwrites the
//! oldest slot and never allocates. One extra slot keeps the job that
//! expires exactly at the newest release readable for that instant.
//!
//! # Concurrency: single writer, seqlock readers
//! Only the DAG's own source node publishes releases (one task per DAG, see
//! `dag::spawn_periodic_reactor`), so each ring has exactly one writer.
//! Every slot is a seqlock whose payload is atomics, so a reader never sees
//! a torn slot and never blocks the writer. A reader that keeps colliding
//! with the writer gives up after [`MAX_READ_RETRIES`] and reports
//! [`JobTableError::ReadContended`] rather than spinning without bound.
//!
//! The ring itself is looked up once per task through [`job_ring`] (a
//! `Mutex`-guarded map written only at admission); the per-release path
//! touches no lock.

use alloc::{boxed::Box, collections::BTreeMap, sync::Arc, vec::Vec};
use awkernel_lib::{
    sync::mutex::{MCSNode, Mutex},
    time::Instant,
};
use core::{
    sync::atomic::{fence, AtomicU64, Ordering},
    time::Duration,
};

/// How many times a reader retries a slot it caught mid-write before giving
/// up. The writer's critical section is three atomic stores, so a reader
/// that collides this often is being starved (e.g. it preempted the writer
/// on the same CPU), and should try again later rather than spin here.
pub const MAX_READ_RETRIES: u32 = 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JobTableError {
    /// A template was registered with a zero period.
    ZeroPeriod,
    /// A template was registered without any segment.
    NoSegment,
    /// `offset + deadline`, `ceil(D* / T) + 1`, or a release time does not
    /// fit the integer types used here.
    Overflow,
    /// A template for this DAG is already registered.
    AlreadyRegistered(u32),
    /// A reader kept catching a slot mid-write; see [`MAX_READ_RETRIES`].
    ReadContended,
}

/// One segment of a DAG-Fluid task, relative to its job's release.
#[derive(Debug, Clone, PartialEq)]
pub struct SegmentTemplate {
    /// Release offset from the job's release (`sum of d` of the earlier
    /// segments, `dag_fluid::segment_release_offsets`).
    pub offset: Duration,
    /// `d_i,j`: this segment's relative deadline.
    pub relative_deadline: Duration,
    /// `m_i,j`: thread count.
    pub concurrency: u32,
    /// `theta_i,j`: execution rate.
    pub rate: f64,
    /// Node ids expected to finish by this segment's end
    /// (`dag_fluid::segment_completion_gates`).
    pub gate_nodes: Vec<u32>,
}

/// Immutable per-DAG parameters registered at admission.
#[derive(Debug, Clone, PartialEq)]
pub struct DagTemplate {
    period: Duration,
    /// `D*`: the end of the last segment, relative to the release.
    virtual_deadline: Duration,
    segments: Box<[SegmentTemplate]>,
}

impl DagTemplate {
    pub fn new(period: Duration, segments: Vec<SegmentTemplate>) -> Result<Self, JobTableError> {
        if period.is_zero() {
            return Err(JobTableError::ZeroPeriod);
        }
        let mut virtual_deadline = Duration::ZERO;
        for segment in &segments {
            let end = segment
                .offset
                .checked_add(segment.relative_deadline)
                .ok_or(JobTableError::Overflow)?;
            virtual_deadline = virtual_deadline.max(end);
        }
        if segments.is_empty() {
            return Err(JobTableError::NoSegment);
        }
        Ok(Self {
            period,
            virtual_deadline,
            segments: segments.into_boxed_slice(),
        })
    }

    pub fn period(&self) -> Duration {
        self.period
    }

    pub fn virtual_deadline(&self) -> Duration {
        self.virtual_deadline
    }

    pub fn segments(&self) -> &[SegmentTemplate] {
        &self.segments
    }

    /// `ceil(D* / T) + 1`: the ring size (see the module doc).
    fn ring_capacity(&self) -> Result<usize, JobTableError> {
        let d_star = self.virtual_deadline.as_nanos();
        let period = self.period.as_nanos();
        let alive = d_star
            .checked_add(period.checked_sub(1).ok_or(JobTableError::ZeroPeriod)?)
            .and_then(|n| n.checked_div(period))
            .ok_or(JobTableError::Overflow)?;
        let capacity = alive.checked_add(1).ok_or(JobTableError::Overflow)?;
        usize::try_from(capacity).map_err(|_| JobTableError::Overflow)
    }
}

/// A released job that has not passed its virtual deadline yet.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActiveJob {
    /// 0 for the first release, counting up by one per release.
    pub job: u64,
    /// `r_k`.
    pub release: Instant,
}

/// One seqlock-protected ring slot. `seq` is odd while the writer is
/// filling it.
struct Slot {
    seq: AtomicU64,
    job: AtomicU64,
    release_ns: AtomicU64,
}

impl Slot {
    const fn new() -> Self {
        Self {
            seq: AtomicU64::new(0),
            job: AtomicU64::new(0),
            release_ns: AtomicU64::new(0),
        }
    }

    /// Read `(job, release_ns)`, or `None` if never written.
    fn read(&self) -> Result<Option<(u64, u64)>, JobTableError> {
        for _ in 0..MAX_READ_RETRIES {
            let before = self.seq.load(Ordering::Acquire);
            if before % 2 == 1 {
                core::hint::spin_loop();
                continue;
            }
            let job = self.job.load(Ordering::Relaxed);
            let release_ns = self.release_ns.load(Ordering::Relaxed);
            fence(Ordering::Acquire);
            if self.seq.load(Ordering::Relaxed) == before {
                return Ok((before != 0).then_some((job, release_ns)));
            }
        }
        Err(JobTableError::ReadContended)
    }
}

/// The release ring of one DAG. See the module doc.
pub struct JobRing {
    template: DagTemplate,
    slots: Box<[Slot]>,
    /// Number of releases published so far; also the next job number.
    next_job: AtomicU64,
}

impl JobRing {
    pub fn new(template: DagTemplate) -> Result<Self, JobTableError> {
        let capacity = template.ring_capacity()?;
        let slots = (0..capacity).map(|_| Slot::new()).collect();
        Ok(Self {
            template,
            slots,
            next_job: AtomicU64::new(0),
        })
    }

    pub fn template(&self) -> &DagTemplate {
        &self.template
    }

    pub fn capacity(&self) -> usize {
        self.slots.len()
    }

    fn slot_of(&self, job: u64) -> Result<&Slot, JobTableError> {
        let len = u64::try_from(self.slots.len()).map_err(|_| JobTableError::Overflow)?;
        let index = job.checked_rem(len).ok_or(JobTableError::Overflow)?;
        let index = usize::try_from(index).map_err(|_| JobTableError::Overflow)?;
        self.slots.get(index).ok_or(JobTableError::Overflow)
    }

    /// Record the next job's release and return its job number.
    ///
    /// Must only be called by the DAG's source node (the single writer);
    /// see the module doc. Never allocates or locks.
    pub fn publish_release(&self, release: Instant) -> Result<u64, JobTableError> {
        let release_ns =
            u64::try_from(release.uptime().as_nanos()).map_err(|_| JobTableError::Overflow)?;
        let job = self.next_job.load(Ordering::Relaxed);
        let next = job.checked_add(1).ok_or(JobTableError::Overflow)?;
        let slot = self.slot_of(job)?;

        // Seqlock write. `seq` is a sequence counter, so wrapping is part
        // of its contract; only its parity and equality are ever compared.
        let seq = slot.seq.load(Ordering::Relaxed);
        slot.seq.store(seq.wrapping_add(1), Ordering::Relaxed);
        fence(Ordering::Release);
        slot.job.store(job, Ordering::Relaxed);
        slot.release_ns.store(release_ns, Ordering::Relaxed);
        slot.seq.store(seq.wrapping_add(2), Ordering::Release);

        self.next_job.store(next, Ordering::Release);
        Ok(job)
    }

    /// Copy the released jobs whose virtual deadline `r_k + D*` is still
    /// after `now` into `out`, oldest first, and return how many were
    /// written. `out` should hold [`Self::capacity`] entries; jobs that do
    /// not fit are dropped from the newest end.
    pub fn active_jobs(&self, now: Instant, out: &mut [ActiveJob]) -> Result<usize, JobTableError> {
        let head = self.next_job.load(Ordering::Acquire);
        let len = u64::try_from(self.slots.len()).map_err(|_| JobTableError::Overflow)?;
        let oldest = head.saturating_sub(len);

        let mut written = 0;
        for job in oldest..head {
            let Some((stored_job, release_ns)) = self.slot_of(job)?.read()? else {
                continue;
            };
            // Overwritten by a newer release after `head` was read: the job
            // we wanted is gone, and the newer one is not ours to report.
            if stored_job != job {
                continue;
            }
            let release = Instant::zero() + Duration::from_nanos(release_ns);
            let deadline = release + self.template.virtual_deadline;
            if deadline <= now {
                continue;
            }
            let Some(entry) = out.get_mut(written) else {
                break;
            };
            *entry = ActiveJob { job, release };
            written += 1;
        }
        Ok(written)
    }

    /// The newest recorded release, or `None` before the first one.
    pub fn latest(&self) -> Result<Option<ActiveJob>, JobTableError> {
        let head = self.next_job.load(Ordering::Acquire);
        let Some(newest) = head.checked_sub(1) else {
            return Ok(None);
        };
        let Some((stored_job, release_ns)) = self.slot_of(newest)?.read()? else {
            return Ok(None);
        };
        if stored_job != newest {
            // A newer release landed after `head` was read; ask again.
            return Err(JobTableError::ReadContended);
        }
        Ok(Some(ActiveJob {
            job: newest,
            release: Instant::zero() + Duration::from_nanos(release_ns),
        }))
    }

    /// The release the next job will have if the source keeps its period:
    /// the newest release plus `T`. `None` before the first release.
    pub fn next_expected_release(&self) -> Result<Option<Instant>, JobTableError> {
        let head = self.next_job.load(Ordering::Acquire);
        let Some(newest) = head.checked_sub(1) else {
            return Ok(None);
        };
        let Some((stored_job, release_ns)) = self.slot_of(newest)?.read()? else {
            return Ok(None);
        };
        if stored_job != newest {
            // A newer release landed after `head` was read; ask again.
            return Err(JobTableError::ReadContended);
        }
        let release = Instant::zero() + Duration::from_nanos(release_ns);
        Ok(Some(release + self.template.period))
    }
}

static JOB_RINGS: Mutex<BTreeMap<u32, Arc<JobRing>>> = Mutex::new(BTreeMap::new());

/// Register `dag_id`'s template and allocate its ring. Called once per
/// DAG-Fluid DAG at admission (boot, non-RT), before its nodes are spawned.
pub fn register_dag_template(dag_id: u32, template: DagTemplate) -> Result<(), JobTableError> {
    let ring = Arc::new(JobRing::new(template)?);
    let mut node = MCSNode::new();
    let mut rings = JOB_RINGS.lock(&mut node);
    if rings.contains_key(&dag_id) {
        return Err(JobTableError::AlreadyRegistered(dag_id));
    }
    rings.insert(dag_id, ring);
    Ok(())
}

/// `dag_id`'s ring, or `None` if it is not a registered DAG-Fluid DAG.
/// Takes a lock: look it up once per task, not once per release.
pub fn job_ring(dag_id: u32) -> Option<Arc<JobRing>> {
    let mut node = MCSNode::new();
    let rings = JOB_RINGS.lock(&mut node);
    rings.get(&dag_id).cloned()
}

/// Every registered DAG and its ring, in `dag_id` order. Takes a lock:
/// for the planner's setup, not for a per-release path.
pub fn registered_rings() -> Vec<(u32, Arc<JobRing>)> {
    let mut node = MCSNode::new();
    let rings = JOB_RINGS.lock(&mut node);
    rings
        .iter()
        .map(|(dag_id, ring)| (*dag_id, Arc::clone(ring)))
        .collect()
}

/// Publish a release from a source node's loop. `ring` is `None` for DAGs
/// that are not DAG-Fluid, which makes this a no-op.
///
/// A failure is logged and the job is not recorded: the source node has no
/// caller to hand it to, and the job itself must still run.
pub fn notify_job_release(ring: Option<&JobRing>, dag_id: u32, release: Instant) {
    let Some(ring) = ring else {
        return;
    };
    if let Err(e) = ring.publish_release(release) {
        log::warn!("job_table: DAG#{dag_id} failed to record release at {release:?}: {e:?}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    fn segment(offset_us: u64, deadline_us: u64) -> SegmentTemplate {
        SegmentTemplate {
            offset: Duration::from_micros(offset_us),
            relative_deadline: Duration::from_micros(deadline_us),
            concurrency: 1,
            rate: 1.0,
            gate_nodes: Vec::new(),
        }
    }

    fn at_us(us: u64) -> Instant {
        Instant::zero() + Duration::from_micros(us)
    }

    fn ring(period_us: u64, segments: Vec<SegmentTemplate>) -> JobRing {
        let template = DagTemplate::new(Duration::from_micros(period_us), segments).unwrap();
        JobRing::new(template).unwrap()
    }

    #[test]
    fn virtual_deadline_is_the_end_of_the_last_segment() {
        let template = DagTemplate::new(
            Duration::from_micros(100),
            vec![segment(0, 30), segment(30, 50), segment(80, 45)],
        )
        .unwrap();
        assert_eq!(template.virtual_deadline(), Duration::from_micros(125));
    }

    #[test]
    fn rejects_zero_period_and_empty_segments() {
        assert_eq!(
            DagTemplate::new(Duration::ZERO, vec![segment(0, 10)]),
            Err(JobTableError::ZeroPeriod)
        );
        assert_eq!(
            DagTemplate::new(Duration::from_micros(10), Vec::new()),
            Err(JobTableError::NoSegment)
        );
    }

    #[test]
    fn capacity_is_ceil_d_star_over_t_plus_one() {
        // Constrained: D* <= T keeps one job alive.
        assert_eq!(ring(100, vec![segment(0, 100)]).capacity(), 2);
        assert_eq!(ring(100, vec![segment(0, 60)]).capacity(), 2);
        // Arbitrary: D* = 250, T = 100 keeps ceil(2.5) = 3 jobs alive.
        assert_eq!(ring(100, vec![segment(0, 250)]).capacity(), 4);
        // Exact multiple: D* = 300, T = 100.
        assert_eq!(ring(100, vec![segment(0, 300)]).capacity(), 4);
    }

    #[test]
    fn empty_ring_has_no_active_job() {
        let r = ring(100, vec![segment(0, 100)]);
        let mut out = [ActiveJob {
            job: 0,
            release: Instant::zero(),
        }; 2];
        assert_eq!(r.active_jobs(at_us(0), &mut out), Ok(0));
        assert_eq!(r.next_expected_release(), Ok(None));
    }

    #[test]
    fn job_numbers_count_releases() {
        let r = ring(100, vec![segment(0, 100)]);
        assert_eq!(r.publish_release(at_us(1_000)), Ok(0));
        assert_eq!(r.publish_release(at_us(1_100)), Ok(1));
        assert_eq!(r.publish_release(at_us(1_200)), Ok(2));
        assert_eq!(r.next_expected_release(), Ok(Some(at_us(1_300))));
    }

    #[test]
    fn latest_is_the_newest_release() {
        let r = ring(100, vec![segment(0, 100)]);
        assert_eq!(r.latest(), Ok(None));
        r.publish_release(at_us(1_000)).unwrap();
        r.publish_release(at_us(1_100)).unwrap();
        r.publish_release(at_us(1_200)).unwrap();
        assert_eq!(
            r.latest(),
            Ok(Some(ActiveJob {
                job: 2,
                release: at_us(1_200)
            }))
        );
    }

    #[test]
    fn arbitrary_deadline_keeps_every_overlapping_job() {
        // T = 100, D* = 250: three jobs overlap right after the third release.
        let r = ring(100, vec![segment(0, 120), segment(120, 130)]);
        let mut out = [ActiveJob {
            job: 0,
            release: Instant::zero(),
        }; 4];
        for k in 0..3 {
            r.publish_release(at_us(1_000 + 100 * k)).unwrap();
        }
        let n = r.active_jobs(at_us(1_210), &mut out).unwrap();
        assert_eq!(n, 3);
        assert_eq!(
            out[0],
            ActiveJob {
                job: 0,
                release: at_us(1_000)
            }
        );
        assert_eq!(
            out[1],
            ActiveJob {
                job: 1,
                release: at_us(1_100)
            }
        );
        assert_eq!(
            out[2],
            ActiveJob {
                job: 2,
                release: at_us(1_200)
            }
        );
    }

    #[test]
    fn expired_jobs_are_not_active() {
        let r = ring(100, vec![segment(0, 250)]);
        let mut out = [ActiveJob {
            job: 0,
            release: Instant::zero(),
        }; 4];
        for k in 0..4 {
            r.publish_release(at_us(1_000 + 100 * k)).unwrap();
        }
        // Job 0 ends at 1_250 (excluded at exactly 1_250), job 1 at 1_350.
        let n = r.active_jobs(at_us(1_250), &mut out).unwrap();
        assert_eq!(n, 3);
        assert_eq!(out[0].job, 1);
        assert_eq!(out[2].job, 3);
    }

    #[test]
    fn ring_overwrites_the_oldest_slot() {
        // Capacity 2. After 5 releases only jobs 3 and 4 remain stored.
        let r = ring(100, vec![segment(0, 100)]);
        for k in 0..5 {
            r.publish_release(at_us(1_000 + 100 * k)).unwrap();
        }
        let mut out = [ActiveJob {
            job: 0,
            release: Instant::zero(),
        }; 2];
        let n = r.active_jobs(at_us(1_000), &mut out).unwrap();
        assert_eq!(n, 2);
        assert_eq!(
            out[0],
            ActiveJob {
                job: 3,
                release: at_us(1_300)
            }
        );
        assert_eq!(
            out[1],
            ActiveJob {
                job: 4,
                release: at_us(1_400)
            }
        );
    }

    #[test]
    fn short_output_keeps_the_oldest_jobs() {
        let r = ring(100, vec![segment(0, 250)]);
        for k in 0..3 {
            r.publish_release(at_us(1_000 + 100 * k)).unwrap();
        }
        let mut out = [ActiveJob {
            job: 0,
            release: Instant::zero(),
        }; 2];
        assert_eq!(r.active_jobs(at_us(1_210), &mut out), Ok(2));
        assert_eq!(out[0].job, 0);
        assert_eq!(out[1].job, 1);
    }

    #[test]
    fn slot_caught_mid_write_reports_contention() {
        let r = ring(100, vec![segment(0, 100)]);
        r.publish_release(at_us(1_000)).unwrap();
        // Simulate the writer stuck between its two `seq` stores.
        let slot = r.slot_of(0).unwrap();
        slot.seq.fetch_add(1, Ordering::Relaxed);
        let mut out = [ActiveJob {
            job: 0,
            release: Instant::zero(),
        }; 2];
        assert_eq!(
            r.active_jobs(at_us(1_000), &mut out),
            Err(JobTableError::ReadContended)
        );
        slot.seq.fetch_add(1, Ordering::Relaxed);
        assert_eq!(r.active_jobs(at_us(1_000), &mut out), Ok(1));
    }
}
