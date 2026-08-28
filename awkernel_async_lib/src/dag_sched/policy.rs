//! One module per DAG admission policy: an algorithm that turns
//! [`super::metrics::DagMetrics`] into a [`super::provision::Provision`] and,
//! from it, a [`crate::scheduler::SchedulerType`] for a DAG's nodes.

pub mod federated;
