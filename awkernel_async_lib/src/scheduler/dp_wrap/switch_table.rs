//! Per-CPU, double-buffered switch tables for DAG-Fluid's per-core dispatch
//! (方式B, Step 2).
//!
//! Each DAG-pool CPU gets its own table of `(time, owner)` switch points.
//! The planner (a task) writes the plan for the *next window* into the
//! table's back plane and publishes it by bumping a generation number; the
//! CPU itself reads only its own table, from its `DpSwitch` timer interrupt
//! (Step 3), and never takes a lock.
//!
//! # What one plane holds
//! One plane is one *window*: the interval from one job release (of any
//! DAG-Fluid DAG) to the next. Releases are strictly periodic
//! (`dag_sched::job_table`), so every Deadline Partition inside a window,
//! and therefore every DP-Wrap switch point in it, is known before the
//! window starts. A plane lists all of them in time order.
//!
//! "Two planes" is about windows, not jobs: the CPU reads the current
//! window's plane while the planner fills the other one with the next
//! window. Any number of overlapping jobs (arbitrary deadlines) fit in one
//! plane, because a plane is the merged schedule of the whole window.
//!
//! # What an entry names
//! By default an entry names a DAG, not a segment: a CPU entitled to a DAG
//! may run any of that DAG's ready nodes (see `ENTITLEMENT` in the parent
//! module for why per-segment tracking was ruled out). With the
//! `dagfluid_node` feature the planner also fills [`SwitchEntry::node`]:
//! the slot then belongs to one node's thread of one job (a [`NodeSlot`]).
//! Consecutive entries for the same owner (DAG, and node slot if any) would
//! not change anything, so [`PerCpuSwitchTable::publish`] folds them before
//! counting against [`SWITCH_TABLE_CAPACITY`].
//!
//! # Publication protocol (one writer, one reader per table)
//! - `published_gen` is the newest generation the planner has published;
//!   generation `g` lives in plane `g % 2`. Generation 0 means "nothing".
//! - `reader_gen` is the generation the CPU is currently reading (0 before
//!   its first plane). Only the CPU writes it.
//! - The planner may only write the plane the CPU is *not* on, so it can
//!   run at most one window ahead: publishing `g + 1` requires the CPU to
//!   have reached `g` already (before the CPU's first plane, two planes may
//!   be queued). Otherwise [`PerCpuSwitchTable::publish`] returns
//!   [`SwitchTableError::ReaderBehind`] and the planner falls back to the
//!   recovery path (Step 4).
//! - The CPU moves to the next generation as soon as it has consumed every
//!   entry of its current plane and that generation is published. If none
//!   is published yet, the plan has run out ([`PollResult::next_at`] is
//!   `None`), which is also the planner's problem to recover from.

use array_macro::array;
use awkernel_lib::{cpu::NUM_MAX_CPU, time::Instant};
use core::{
    sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering},
    time::Duration,
};

/// How many switch points one plane holds after folding.
pub const SWITCH_TABLE_CAPACITY: usize = 128;

/// Stored in place of a DAG id for "no DAG" (the CPU idles DP-Wrap-wise).
const NO_DAG: u32 = u32::MAX;

/// Stored in place of a node id for "no node slot" (a per-DAG entry).
const NO_NODE: u32 = u32::MAX;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SwitchTableError {
    /// More entries than [`SWITCH_TABLE_CAPACITY`] after folding.
    Overflow { needed: usize },
    /// Entries are not in non-decreasing time order.
    Unsorted { index: usize },
    /// A switch time does not fit in `u64` nanoseconds of uptime.
    TimeOverflow,
    /// The CPU is still on the plane the next generation would overwrite.
    ReaderBehind { published: u64, reader: u64 },
    /// `cpu_id` is out of range.
    NoSuchCpu(usize),
}

/// The node thread a per-node slot belongs to (the `dagfluid_node`
/// feature): node `node_id` of the DAG, working on job `job`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NodeSlot {
    pub node_id: u32,
    pub job: u64,
}

/// One switch point: from `at` on, the CPU is entitled to `dag_id` (`None`:
/// to no DAG), and with `node` set, to that node's slot only.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SwitchEntry {
    pub at: Instant,
    pub dag_id: Option<u32>,
    /// `None` for a per-DAG entry (and always for no DAG).
    pub node: Option<NodeSlot>,
}

impl SwitchEntry {
    /// A per-DAG entry (`None`: no DAG).
    pub const fn dag(at: Instant, dag_id: Option<u32>) -> Self {
        Self {
            at,
            dag_id,
            node: None,
        }
    }

    /// A per-node entry: `dag_id`'s node `node_id`, job `job`.
    pub const fn node(at: Instant, dag_id: u32, node_id: u32, job: u64) -> Self {
        Self {
            at,
            dag_id: Some(dag_id),
            node: Some(NodeSlot { node_id, job }),
        }
    }

    /// Whether `self` and `other` hand the CPU to the same owner.
    fn same_owner(&self, other: &Self) -> bool {
        self.dag_id == other.dag_id && self.node == other.node
    }
}

/// What [`PerCpuSwitchTable::poll`] found.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PollResult {
    /// The newest entry whose time has come, if any. Older due entries are
    /// superseded by it and only counted in `skipped`.
    pub due: Option<SwitchEntry>,
    /// Due entries passed over because a later one was also due (the CPU
    /// polled late).
    pub skipped: u32,
    /// When the next entry is due, i.e. when to arm the timer. `None`: the
    /// published plan has run out.
    pub next_at: Option<Instant>,
}

struct Slot {
    at_ns: AtomicU64,
    dag_id: AtomicU32,
    node_id: AtomicU32,
    job: AtomicU64,
}

impl Slot {
    const fn new() -> Self {
        Self {
            at_ns: AtomicU64::new(0),
            dag_id: AtomicU32::new(NO_DAG),
            node_id: AtomicU32::new(NO_NODE),
            job: AtomicU64::new(0),
        }
    }

    fn store(&self, at_ns: u64, entry: &SwitchEntry) {
        self.at_ns.store(at_ns, Ordering::Relaxed);
        self.dag_id
            .store(entry.dag_id.unwrap_or(NO_DAG), Ordering::Relaxed);
        let (node_id, job) = entry
            .node
            .map_or((NO_NODE, 0), |slot| (slot.node_id, slot.job));
        self.node_id.store(node_id, Ordering::Relaxed);
        self.job.store(job, Ordering::Relaxed);
    }

    fn load(&self) -> SwitchEntry {
        let at = Instant::zero() + Duration::from_nanos(self.at_ns.load(Ordering::Relaxed));
        let dag_id = match self.dag_id.load(Ordering::Relaxed) {
            NO_DAG => None,
            dag_id => Some(dag_id),
        };
        let node = match self.node_id.load(Ordering::Relaxed) {
            NO_NODE => None,
            node_id => Some(NodeSlot {
                node_id,
                job: self.job.load(Ordering::Relaxed),
            }),
        };
        SwitchEntry { at, dag_id, node }
    }
}

struct Plane {
    len: AtomicUsize,
    slots: [Slot; SWITCH_TABLE_CAPACITY],
}

impl Plane {
    const fn new() -> Self {
        Self {
            len: AtomicUsize::new(0),
            slots: array![_ => Slot::new(); SWITCH_TABLE_CAPACITY],
        }
    }

    fn entry(&self, index: usize) -> Option<SwitchEntry> {
        if index >= self.len.load(Ordering::Relaxed) {
            return None;
        }
        self.slots.get(index).map(Slot::load)
    }
}

/// One CPU's switch table. See the module doc for the protocol.
pub struct PerCpuSwitchTable {
    planes: [Plane; 2],
    published_gen: AtomicU64,
    reader_gen: AtomicU64,
    /// Next entry to look at in the reader's current plane. Only the
    /// reader touches it; atomic only so the table can be a `static`.
    reader_idx: AtomicUsize,
    /// Set by the CPU when its plan ran out (and initially: no plan yet);
    /// taken by the planner, which then republishes and kicks the CPU.
    starved: AtomicBool,
}

impl Default for PerCpuSwitchTable {
    fn default() -> Self {
        Self::new()
    }
}

impl PerCpuSwitchTable {
    pub const fn new() -> Self {
        Self {
            planes: [Plane::new(), Plane::new()],
            published_gen: AtomicU64::new(0),
            reader_gen: AtomicU64::new(0),
            reader_idx: AtomicUsize::new(0),
            starved: AtomicBool::new(true),
        }
    }

    fn plane(&self, generation: u64) -> &Plane {
        if generation.is_multiple_of(2) {
            &self.planes[0]
        } else {
            &self.planes[1]
        }
    }

    /// Planner side: publish `entries` (one window, in time order) as the
    /// next generation and return that generation. Folds consecutive
    /// entries for the same owner (DAG and node slot). Never allocates.
    pub fn publish(&self, entries: &[SwitchEntry]) -> Result<u64, SwitchTableError> {
        let published = self.published_gen.load(Ordering::Relaxed);
        let reader = self.reader_gen.load(Ordering::Acquire);
        let next = published.wrapping_add(1);
        // The CPU must not be on (or still headed for) the plane `next`
        // overwrites. Before its first plane it holds none, so two
        // generations may be queued.
        let allowed = if reader == 0 {
            next <= 2
        } else {
            published <= reader
        };
        if !allowed {
            return Err(SwitchTableError::ReaderBehind { published, reader });
        }

        let mut count = 0;
        let mut last: Option<SwitchEntry> = None;
        for (index, entry) in entries.iter().enumerate() {
            if let Some(prev) = last {
                if entry.at < prev.at {
                    return Err(SwitchTableError::Unsorted { index });
                }
                if entry.same_owner(&prev) {
                    continue;
                }
            }
            u64::try_from(entry.at.uptime().as_nanos())
                .map_err(|_| SwitchTableError::TimeOverflow)?;
            count += 1;
            last = Some(*entry);
        }
        if count > SWITCH_TABLE_CAPACITY {
            return Err(SwitchTableError::Overflow { needed: count });
        }

        let plane = self.plane(next);
        let mut written = 0;
        let mut last: Option<&SwitchEntry> = None;
        for entry in entries {
            if last.is_some_and(|prev| entry.same_owner(prev)) {
                continue;
            }
            let Some(slot) = plane.slots.get(written) else {
                break;
            };
            // Range-checked in the counting pass above.
            let at_ns = u64::try_from(entry.at.uptime().as_nanos()).unwrap_or(u64::MAX);
            slot.store(at_ns, entry);
            last = Some(entry);
            written += 1;
        }
        plane.len.store(written, Ordering::Relaxed);
        self.published_gen.store(next, Ordering::Release);
        Ok(next)
    }

    /// CPU side (its own table only): consume every entry due at `now` and
    /// report the newest one plus when the next one is due. Moves to the
    /// next published plane once the current one is used up. Never
    /// allocates, locks or spins.
    pub fn poll(&self, now: Instant) -> PollResult {
        let mut result = PollResult {
            due: None,
            skipped: 0,
            next_at: None,
        };
        loop {
            let reader = self.reader_gen.load(Ordering::Relaxed);
            let published = self.published_gen.load(Ordering::Acquire);
            let idx = self.reader_idx.load(Ordering::Relaxed);
            let current = (reader != 0)
                .then(|| self.plane(reader).entry(idx))
                .flatten();
            match current {
                Some(entry) if entry.at <= now => {
                    if result.due.is_some() {
                        result.skipped = result.skipped.saturating_add(1);
                    }
                    result.due = Some(entry);
                    self.reader_idx
                        .store(idx.saturating_add(1), Ordering::Relaxed);
                }
                Some(entry) => {
                    result.next_at = Some(entry.at);
                    return result;
                }
                None if published > reader => {
                    // Current plane used up (or none yet): move on. Every
                    // read of the old plane happens before this store, so
                    // the planner may reuse it once it sees the new value.
                    self.reader_idx.store(0, Ordering::Relaxed);
                    self.reader_gen
                        .store(reader.wrapping_add(1), Ordering::Release);
                }
                None => {
                    self.starved.store(true, Ordering::Release);
                    return result;
                }
            }
        }
    }

    /// CPU side: drop the rest of the current plane and move to the newest
    /// published one. For the event-driven planner (③), whose newest plan
    /// replaces the current one rather than following it. A no-op if
    /// nothing newer is published.
    pub fn jump_to_latest(&self) {
        let reader = self.reader_gen.load(Ordering::Relaxed);
        let published = self.published_gen.load(Ordering::Acquire);
        if published > reader {
            self.reader_idx.store(0, Ordering::Relaxed);
            self.reader_gen.store(published, Ordering::Release);
        }
    }

    /// Planner side: whether the CPU's plan ran out since the last call
    /// (true before its first plan too). Clears the flag.
    pub fn take_starved(&self) -> bool {
        self.starved.swap(false, Ordering::AcqRel)
    }

    /// The generation the CPU is reading (0: none yet).
    pub fn reader_generation(&self) -> u64 {
        self.reader_gen.load(Ordering::Acquire)
    }

    /// The newest generation published (0: none yet).
    pub fn published_generation(&self) -> u64 {
        self.published_gen.load(Ordering::Acquire)
    }
}

static SWITCH_TABLES: [PerCpuSwitchTable; NUM_MAX_CPU] =
    array![_ => PerCpuSwitchTable::new(); NUM_MAX_CPU];

/// Planner side: publish the next window's plan for `cpu_id`.
pub fn publish_cpu_plan(cpu_id: usize, entries: &[SwitchEntry]) -> Result<u64, SwitchTableError> {
    SWITCH_TABLES
        .get(cpu_id)
        .ok_or(SwitchTableError::NoSuchCpu(cpu_id))?
        .publish(entries)
}

/// Planner side: whether `cpu_id`'s plan ran out (see
/// [`PerCpuSwitchTable::take_starved`]).
pub fn take_cpu_starved(cpu_id: usize) -> Result<bool, SwitchTableError> {
    Ok(SWITCH_TABLES
        .get(cpu_id)
        .ok_or(SwitchTableError::NoSuchCpu(cpu_id))?
        .take_starved())
}

/// CPU side: see [`PerCpuSwitchTable::jump_to_latest`]. Must only be called
/// on `cpu_id`.
pub fn jump_cpu_to_latest(cpu_id: usize) -> Result<(), SwitchTableError> {
    SWITCH_TABLES
        .get(cpu_id)
        .ok_or(SwitchTableError::NoSuchCpu(cpu_id))?
        .jump_to_latest();
    Ok(())
}

/// CPU side: poll `cpu_id`'s own table. Must only be called on `cpu_id`.
pub fn poll_due_switch(cpu_id: usize, now: Instant) -> Result<PollResult, SwitchTableError> {
    Ok(SWITCH_TABLES
        .get(cpu_id)
        .ok_or(SwitchTableError::NoSuchCpu(cpu_id))?
        .poll(now))
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    fn at_us(us: u64) -> Instant {
        Instant::zero() + Duration::from_micros(us)
    }

    fn e(us: u64, dag_id: Option<u32>) -> SwitchEntry {
        SwitchEntry::dag(at_us(us), dag_id)
    }

    fn n(us: u64, dag_id: u32, node_id: u32, job: u64) -> SwitchEntry {
        SwitchEntry::node(at_us(us), dag_id, node_id, job)
    }

    #[test]
    fn nothing_published_means_no_plan() {
        let t = PerCpuSwitchTable::new();
        let r = t.poll(at_us(1_000));
        assert_eq!(
            r,
            PollResult {
                due: None,
                skipped: 0,
                next_at: None
            }
        );
        assert_eq!(t.reader_generation(), 0);
    }

    #[test]
    fn entries_come_out_in_time_order() {
        let t = PerCpuSwitchTable::new();
        assert_eq!(
            t.publish(&[e(100, Some(1)), e(200, Some(2)), e(300, None)]),
            Ok(1)
        );

        let r = t.poll(at_us(50));
        assert_eq!(r.due, None);
        assert_eq!(r.next_at, Some(at_us(100)));
        assert_eq!(t.reader_generation(), 1);

        let r = t.poll(at_us(100));
        assert_eq!(r.due, Some(e(100, Some(1))));
        assert_eq!(r.next_at, Some(at_us(200)));

        let r = t.poll(at_us(250));
        assert_eq!(r.due, Some(e(200, Some(2))));
        assert_eq!(r.next_at, Some(at_us(300)));

        let r = t.poll(at_us(300));
        assert_eq!(r.due, Some(e(300, None)));
        assert_eq!(r.next_at, None, "plan runs out after the last entry");
    }

    #[test]
    fn starved_flag_marks_a_cpu_whose_plan_ran_out() {
        let t = PerCpuSwitchTable::new();
        assert!(t.take_starved(), "no plan yet counts as starved");
        assert!(!t.take_starved(), "taking clears it");
        t.publish(&[e(100, Some(1))]).unwrap();
        t.poll(at_us(50));
        assert!(!t.take_starved(), "an entry is still pending");
        t.poll(at_us(100));
        assert!(t.take_starved(), "the last entry was consumed");
    }

    #[test]
    fn jump_replaces_the_rest_of_the_current_plane() {
        let t = PerCpuSwitchTable::new();
        t.publish(&[e(100, Some(1)), e(200, Some(2)), e(300, Some(3))])
            .unwrap();
        assert_eq!(t.poll(at_us(100)).due, Some(e(100, Some(1))));
        // A new plan made at 150 replaces 200/300.
        t.publish(&[e(150, Some(4)), e(400, None)]).unwrap();
        t.jump_to_latest();
        assert_eq!(t.reader_generation(), 2);
        let r = t.poll(at_us(250));
        assert_eq!(r.due, Some(e(150, Some(4))));
        assert_eq!(r.next_at, Some(at_us(400)));
        // Nothing newer: jumping again changes nothing.
        t.jump_to_latest();
        assert_eq!(t.reader_generation(), 2);
        assert_eq!(t.poll(at_us(400)).due, Some(e(400, None)));
    }

    #[test]
    fn late_poll_reports_only_the_newest_due_entry() {
        let t = PerCpuSwitchTable::new();
        t.publish(&[
            e(100, Some(1)),
            e(200, Some(2)),
            e(300, Some(3)),
            e(400, Some(4)),
        ])
        .unwrap();
        let r = t.poll(at_us(350));
        assert_eq!(r.due, Some(e(300, Some(3))));
        assert_eq!(r.skipped, 2);
        assert_eq!(r.next_at, Some(at_us(400)));
    }

    #[test]
    fn consecutive_entries_for_the_same_dag_are_folded() {
        let t = PerCpuSwitchTable::new();
        t.publish(&[
            e(100, Some(1)),
            e(150, Some(1)),
            e(200, Some(2)),
            e(250, Some(2)),
            e(300, Some(1)),
        ])
        .unwrap();
        let mut seen = Vec::new();
        let mut now = 0;
        while let Some(next) = t.poll(at_us(now)).next_at {
            now = next.uptime().as_micros() as u64;
            seen.push(t.poll(at_us(now)).due.unwrap());
            now += 1;
        }
        assert_eq!(seen, [e(100, Some(1)), e(200, Some(2)), e(300, Some(1))]);
    }

    #[test]
    fn node_slots_round_trip_and_fold_per_node_and_job() {
        let t = PerCpuSwitchTable::new();
        t.publish(&[
            n(100, 1, 3, 0),
            n(150, 1, 3, 0), // same node, same job: folded
            n(200, 1, 4, 0), // another node of the same DAG
            n(300, 1, 4, 1), // same node, next job
            e(400, Some(1)), // a per-DAG entry differs from any node slot
            e(500, None),
        ])
        .unwrap();
        let mut seen = Vec::new();
        let mut now = 0;
        while let Some(next) = t.poll(at_us(now)).next_at {
            now = next.uptime().as_micros() as u64;
            seen.push(t.poll(at_us(now)).due.unwrap());
            now += 1;
        }
        assert_eq!(
            seen,
            [
                n(100, 1, 3, 0),
                n(200, 1, 4, 0),
                n(300, 1, 4, 1),
                e(400, Some(1)),
                e(500, None)
            ]
        );
    }

    #[test]
    fn overwritten_slot_drops_a_stale_node() {
        // A plane reused for a per-DAG entry must not keep the node slot
        // written there two generations before.
        let t = PerCpuSwitchTable::new();
        t.publish(&[n(100, 1, 3, 7)]).unwrap();
        t.publish(&[e(1_000, Some(2))]).unwrap();
        t.poll(at_us(100));
        assert_eq!(t.poll(at_us(100)).next_at, Some(at_us(1_000)));
        t.publish(&[e(2_000, Some(1))]).unwrap();
        t.poll(at_us(1_000));
        assert_eq!(t.poll(at_us(2_000)).due, Some(e(2_000, Some(1))));
    }

    #[test]
    fn reader_moves_to_the_next_window_when_its_plane_is_used_up() {
        let t = PerCpuSwitchTable::new();
        t.publish(&[e(100, Some(1)), e(200, Some(2))]).unwrap();
        t.publish(&[e(1_000, Some(3)), e(1_200, Some(1))]).unwrap();

        assert_eq!(t.poll(at_us(100)).due, Some(e(100, Some(1))));
        // The last entry of window 1 is consumed and window 2 is waiting:
        // the reader moves on and reports window 2's first entry.
        let r = t.poll(at_us(200));
        assert_eq!(r.due, Some(e(200, Some(2))));
        assert_eq!(r.next_at, Some(at_us(1_000)));
        assert_eq!(t.reader_generation(), 2);

        // A poll after both windows' due times sees window 2's entries too.
        let r = t.poll(at_us(1_300));
        assert_eq!(r.due, Some(e(1_200, Some(1))));
        assert_eq!(r.skipped, 1);
    }

    #[test]
    fn planner_can_only_run_one_window_ahead() {
        let t = PerCpuSwitchTable::new();
        // Before the reader starts, two windows may be queued.
        assert_eq!(t.publish(&[e(100, Some(1))]), Ok(1));
        assert_eq!(t.publish(&[e(1_000, Some(2))]), Ok(2));
        assert_eq!(
            t.publish(&[e(2_000, Some(3))]),
            Err(SwitchTableError::ReaderBehind {
                published: 2,
                reader: 0
            })
        );

        // Reader on window 1, window 2 queued: window 3 would overwrite 1.
        t.poll(at_us(0));
        assert_eq!(t.reader_generation(), 1);
        assert_eq!(
            t.publish(&[e(2_000, Some(3))]),
            Err(SwitchTableError::ReaderBehind {
                published: 2,
                reader: 1
            })
        );

        // Once the reader is on window 2, window 3 may go into plane 1.
        t.poll(at_us(100));
        assert_eq!(t.poll(at_us(100)).next_at, Some(at_us(1_000)));
        assert_eq!(t.reader_generation(), 2);
        assert_eq!(t.publish(&[e(2_000, Some(3))]), Ok(3));
    }

    #[test]
    fn overwritten_plane_carries_the_new_window_only() {
        let t = PerCpuSwitchTable::new();
        t.publish(&[e(100, Some(1)), e(200, Some(2)), e(300, Some(3))])
            .unwrap();
        t.publish(&[e(1_000, Some(4))]).unwrap();
        t.poll(at_us(300));
        assert_eq!(t.reader_generation(), 2);
        // Window 3 reuses plane 1 with fewer entries than window 1 had.
        t.publish(&[e(2_000, Some(5))]).unwrap();
        t.poll(at_us(1_000));
        assert_eq!(t.reader_generation(), 3);
        let r = t.poll(at_us(5_000));
        assert_eq!(r.due, Some(e(2_000, Some(5))));
        assert_eq!(r.next_at, None, "no stale entries from window 1");
    }

    #[test]
    fn rejects_unsorted_and_oversized_plans() {
        let t = PerCpuSwitchTable::new();
        assert_eq!(
            t.publish(&[e(200, Some(1)), e(100, Some(2))]),
            Err(SwitchTableError::Unsorted { index: 1 })
        );

        let too_many: Vec<SwitchEntry> = (0..=SWITCH_TABLE_CAPACITY as u64)
            .map(|i| e(i, Some((i % 2) as u32)))
            .collect();
        assert_eq!(
            t.publish(&too_many),
            Err(SwitchTableError::Overflow {
                needed: SWITCH_TABLE_CAPACITY + 1
            })
        );

        // Folding happens before the capacity check.
        let folds_to_one: Vec<SwitchEntry> = (0..1_000).map(|i| e(i, Some(7))).collect();
        assert_eq!(t.publish(&folds_to_one), Ok(1));
        assert_eq!(t.published_generation(), 1);
    }

    #[test]
    fn exactly_full_plane_is_accepted() {
        let t = PerCpuSwitchTable::new();
        let full: Vec<SwitchEntry> = (0..SWITCH_TABLE_CAPACITY as u64)
            .map(|i| e(i, Some((i % 2) as u32)))
            .collect();
        assert_eq!(t.publish(&full), Ok(1));
        let r = t.poll(at_us(SWITCH_TABLE_CAPACITY as u64));
        assert_eq!(r.skipped, SWITCH_TABLE_CAPACITY as u32 - 1);
        assert_eq!(r.due, full.last().copied());
    }

    /// One planner thread and one reader thread running at once. Window
    /// `w` covers `[w * 1000, w * 1000 + 999]` µs and every entry in it names
    /// DAG `w`, so a reader that ever mixed two windows' data would see an
    /// entry whose time and DAG disagree, or time going backwards.
    #[cfg(feature = "std")]
    #[test]
    fn concurrent_planner_and_reader_never_mix_windows() {
        extern crate std;
        use std::{sync::Arc, thread};

        const WINDOWS: u64 = 2_000;
        let table = Arc::new(PerCpuSwitchTable::new());

        let writer = {
            let table = Arc::clone(&table);
            thread::spawn(move || {
                let mut w = 1;
                while w <= WINDOWS {
                    let entries: Vec<SwitchEntry> = (0..(w % 7 + 1))
                        .map(|k| {
                            e(
                                w * 1_000 + k * 100,
                                Some(((w + k) % 2 * 1_000_000 + w) as u32),
                            )
                        })
                        .collect();
                    match table.publish(&entries) {
                        Ok(_) => w += 1,
                        Err(SwitchTableError::ReaderBehind { .. }) => thread::yield_now(),
                        Err(other) => panic!("unexpected {other:?}"),
                    }
                }
            })
        };

        let mut now = 0;
        let mut last_at = 0;
        let mut seen = 0_u64;
        while now <= WINDOWS * 1_000 + 999 {
            let r = table.poll(at_us(now));
            if let Some(entry) = r.due {
                let at = entry.at.uptime().as_micros() as u64;
                let dag = u64::from(entry.dag_id.unwrap());
                assert_eq!(dag % 1_000_000, at / 1_000, "entry from another window");
                assert!(at >= last_at, "time went backwards");
                last_at = at;
                seen += 1 + u64::from(r.skipped);
            }
            match r.next_at {
                Some(next) => now = next.uptime().as_micros() as u64,
                None if table.published_generation() >= WINDOWS
                    && table.reader_generation() >= WINDOWS =>
                {
                    break
                }
                None => thread::yield_now(),
            }
        }
        writer.join().unwrap();
        let expected: u64 = (1..=WINDOWS).map(|w| w % 7 + 1).sum();
        assert_eq!(seen, expected, "every published entry is seen exactly once");
    }

    #[test]
    fn global_wrappers_check_the_cpu_id() {
        assert_eq!(
            publish_cpu_plan(NUM_MAX_CPU, &[]),
            Err(SwitchTableError::NoSuchCpu(NUM_MAX_CPU))
        );
        assert_eq!(
            poll_due_switch(NUM_MAX_CPU, at_us(0)),
            Err(SwitchTableError::NoSuchCpu(NUM_MAX_CPU))
        );
    }
}
