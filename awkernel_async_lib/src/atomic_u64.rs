//! `AtomicU64` that also builds on targets without 64-bit atomics (riscv32).
//! There it is a lock-backed stand-in covering only the operations this crate
//! uses; elsewhere it is `core`'s own type.

#[cfg(target_has_atomic = "64")]
pub(crate) use core::sync::atomic::AtomicU64;

#[cfg(not(target_has_atomic = "64"))]
pub(crate) use fallback::AtomicU64;

#[cfg(not(target_has_atomic = "64"))]
mod fallback {
    use awkernel_lib::sync::mutex::{MCSNode, Mutex};
    use core::sync::atomic::Ordering;

    pub(crate) struct AtomicU64(Mutex<u64>);

    impl AtomicU64 {
        pub(crate) const fn new(value: u64) -> Self {
            Self(Mutex::new(value))
        }

        pub(crate) fn load(&self, _order: Ordering) -> u64 {
            let mut node = MCSNode::new();
            let value = *self.0.lock(&mut node);
            value
        }

        pub(crate) fn store(&self, value: u64, _order: Ordering) {
            let mut node = MCSNode::new();
            *self.0.lock(&mut node) = value;
        }

        pub(crate) fn swap(&self, value: u64, _order: Ordering) -> u64 {
            let mut node = MCSNode::new();
            let previous = core::mem::replace(&mut *self.0.lock(&mut node), value);
            previous
        }
    }
}
