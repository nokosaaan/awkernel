//! State shared across DAGs (and across the EKF -> NDT feedback edge).
//!
//! Topics cannot cross DAG boundaries (topic names are unique system-wide and the validator
//! rejects inter-DAG edges), so data between DAGs of different periods goes through these:
//!
//! - [`Latest`]: a single overwritten slot, the `KeepLast(1)` behaviour of a ROS subscription
//!   read from a timer callback. For values where only the newest matters (kinematic state,
//!   acceleration, biased pose).
//! - [`Queue`]: a bounded FIFO that the consumer drains, so a measurement produced at 30 Hz
//!   and consumed at 50 Hz is fused exactly once. Full queues drop the oldest entry (the
//!   `KeepLast(N)` behaviour); drops are counted.

use awkernel_lib::sync::mutex::{MCSNode, Mutex};
use core::sync::atomic::{AtomicU32, Ordering};

pub struct Latest<T: Send> {
    slot: Mutex<Option<T>>,
}

impl<T: Clone + Send> Latest<T> {
    pub const fn new() -> Self {
        Self {
            slot: Mutex::new(None),
        }
    }

    pub fn set(&self, value: T) {
        let mut node = MCSNode::new();
        *self.slot.lock(&mut node) = Some(value);
    }

    pub fn get(&self) -> Option<T> {
        let mut node = MCSNode::new();
        let value = self.slot.lock(&mut node).clone();
        value
    }
}

pub struct Queue<T: Send, const N: usize> {
    items: Mutex<heapless::Deque<T, N>>,
    dropped: AtomicU32,
}

impl<T: Send, const N: usize> Queue<T, N> {
    pub const fn new() -> Self {
        Self {
            items: Mutex::new(heapless::Deque::new()),
            dropped: AtomicU32::new(0),
        }
    }

    /// Enqueue; when full, the oldest entry is discarded to make room.
    pub fn push(&self, value: T) {
        let mut node = MCSNode::new();
        let mut items = self.items.lock(&mut node);
        if items.is_full() {
            items.pop_front();
            self.dropped.fetch_add(1, Ordering::Relaxed);
        }
        // Cannot fail: a slot was freed above if the deque was full.
        if items.push_back(value).is_err() {
            self.dropped.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Take every queued entry, oldest first.
    pub fn drain(&self) -> heapless::Vec<T, N> {
        let mut node = MCSNode::new();
        let mut items = self.items.lock(&mut node);
        let mut out = heapless::Vec::new();
        while let Some(v) = items.pop_front() {
            // `out` has the same capacity as `items`, so this never overflows.
            if out.push(v).is_err() {
                self.dropped.fetch_add(1, Ordering::Relaxed);
            }
        }
        out
    }

    pub fn dropped(&self) -> u32 {
        self.dropped.load(Ordering::Relaxed)
    }
}
