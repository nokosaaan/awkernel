//! [`AdmissionError`]: why an admission policy rejected a DAG or a task
//! set -- the one error type every policy in [`super::policy`] returns.

use crate::{dag::DagError, dag_sched::resource::ResourceError};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AdmissionError {
    /// A DAG violates one of the policy's preconditions (see
    /// [`super::precondition`]); no admission test was run.
    Precondition(DagError),
    /// The admission test itself failed: the task set does not fit the
    /// available processors under this policy (e.g. Federated's `MINPROCS`
    /// or `PARTITION` returned `FAILURE`, V-Fed's Algorithm 2 found no
    /// `M_h`).
    NoFeasibleAllocation,
    /// The shared core/utilization ledger could not satisfy an admitted
    /// DAG's resource request; see [`ResourceError`].
    Resource(ResourceError),
}

impl From<DagError> for AdmissionError {
    fn from(e: DagError) -> Self {
        AdmissionError::Precondition(e)
    }
}

impl From<ResourceError> for AdmissionError {
    fn from(e: ResourceError) -> Self {
        AdmissionError::Resource(e)
    }
}

impl AdmissionError {
    /// The same error with a precondition reported against `dag_id` (see
    /// [`DagError::with_dag_id`]).
    pub fn with_dag_id(self, dag_id: u32) -> Self {
        match self {
            AdmissionError::Precondition(e) => AdmissionError::Precondition(e.with_dag_id(dag_id)),
            other => other,
        }
    }
}

impl core::fmt::Display for AdmissionError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            AdmissionError::Precondition(e) => write!(f, "{e}"),
            AdmissionError::NoFeasibleAllocation => {
                write!(
                    f,
                    "admission test failed: the task set does not fit the processors"
                )
            }
            AdmissionError::Resource(e) => write!(f, "{e}"),
        }
    }
}
