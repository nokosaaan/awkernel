//! DAG scheduling *policy* layer: decides which [`crate::scheduler::SchedulerType`]
//! and which cores a DAG's nodes should use. A sibling of [`crate::dag`]
//! (graph structure) and [`crate::scheduler`] (run-queue mechanisms) rather
//! than a child of either, so that adding an admission algorithm never
//! requires touching the run queues, and adding a run queue never requires
//! knowing which admission algorithm feeds it.
//!
//! Split into layers so each concern is isolated from the others:
//!
//! - [`metrics`]: the DAG-level numbers every admission policy reads
//!   ([`metrics::DagMetrics`]) and how they were obtained
//!   ([`metrics::MetricsSource`]).
//! - [`resource`]: the shared, algorithm-agnostic ledger of which cores are
//!   claimed and how much of the shared pool's utilization is committed.
//! - [`provision`]: [`provision::Provision`], the common shape an admission
//!   policy's resource decision is expressed in, before it is turned into a
//!   concrete [`crate::scheduler::SchedulerType`].
//! - [`policy`]: one module per admission algorithm (currently just
//!   [`policy::federated`]), each combining `metrics`/`resource`/`provision`
//!   into that algorithm's own `admit_dag`.
pub mod metrics;
pub mod policy;
pub mod provision;
pub mod resource;
