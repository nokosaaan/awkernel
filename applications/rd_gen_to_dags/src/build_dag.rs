#[cfg(feature = "laxity")]
use crate::dag_stats::compute_node_laxity;
use crate::dag_stats::compute_dag_stats;
use crate::parse_yaml::{DagData, NodeData};
use crate::time_unit::{convert_duration, simulated_execution_time};

use alloc::{borrow::Cow, format, sync::Arc, vec::Vec};
use awkernel_async_lib::{
    dag::{Dag, create_dag},
    dag_sched::{metrics::DagMetrics, policy::federated::FederatedError},
    scheduler::SchedulerType,
};
#[cfg(not(any(feature = "vfed", feature = "laxity")))]
use awkernel_async_lib::dag_sched::policy::federated::admit_dag;
#[cfg(feature = "vfed")]
use awkernel_async_lib::dag_sched::policy::vfed::{self, PackingStrategy, VFedError};

#[cfg(all(feature = "vfed", feature = "laxity"))]
compile_error!("features \"vfed\" and \"laxity\" select mutually exclusive admission policies");

/// Represents errors related to the number of links for a node.
/// `(DAG ID, Node ID)` tuple to identify the specific DAG and node where the error occurred.
pub(crate) enum LinkNumError {
    Input(u32, u32),
    Output(u32, u32),
    InOut(u32, u32),
}

impl core::fmt::Display for LinkNumError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            LinkNumError::Input(dag_id, node_id) => {
                write!(f, "DAG#{dag_id} Node#{node_id} has no input links")
            }
            LinkNumError::Output(dag_id, node_id) => {
                write!(f, "DAG#{dag_id} Node#{node_id} has no output links")
            }
            LinkNumError::InOut(dag_id, node_id) => {
                write!(f, "DAG#{dag_id} Node#{node_id} has no links")
            }
        }
    }
}

/// Errors that can prevent a DAG from being built, from either link-arity
/// validation or admission (Federated Scheduling or, with the `vfed`
/// feature, V-Fed).
pub(crate) enum BuildDagError {
    LinkNum(LinkNumError),
    Federated(FederatedError),
    #[cfg(feature = "vfed")]
    VFed(VFedError),
    /// `vfed::admit_one` succeeded but `into_scheduler_type` returned
    /// `None` — should not happen for any `VFedAssignment` this crate's own
    /// admission call can produce (see that method's own doc), kept only so
    /// the conversion stays total rather than panicking on a should-never
    /// happen case.
    #[cfg(feature = "vfed")]
    VFedSchedulerTypeMissing(u32),
    /// The DAG's source node has no `period`, or its sink node has no
    /// `end_to_end_deadline` — both are required for admission.
    MissingDagTiming(u32),
    /// (`laxity` feature only) `compute_node_laxity` found
    /// `relative_deadline <= critical_path` — the DAG is unconditionally
    /// infeasible under any policy, not specific to this one.
    #[cfg(feature = "laxity")]
    LaxityInfeasible(u32),
}

impl core::fmt::Display for BuildDagError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            BuildDagError::LinkNum(e) => write!(f, "{e}"),
            BuildDagError::Federated(e) => write!(f, "{e}"),
            #[cfg(feature = "vfed")]
            BuildDagError::VFed(e) => write!(f, "{e:?}"),
            #[cfg(feature = "vfed")]
            BuildDagError::VFedSchedulerTypeMissing(dag_id) => write!(
                f,
                "DAG#{dag_id}: vfed admitted it but produced no SchedulerType"
            ),
            BuildDagError::MissingDagTiming(dag_id) => write!(
                f,
                "DAG#{dag_id} has no source period or no sink end-to-end deadline"
            ),
            #[cfg(feature = "laxity")]
            BuildDagError::LaxityInfeasible(dag_id) => write!(
                f,
                "DAG#{dag_id}: relative_deadline <= critical_path, unconditionally infeasible"
            ),
        }
    }
}

impl From<LinkNumError> for BuildDagError {
    fn from(e: LinkNumError) -> Self {
        BuildDagError::LinkNum(e)
    }
}

impl From<FederatedError> for BuildDagError {
    fn from(e: FederatedError) -> Self {
        BuildDagError::Federated(e)
    }
}

#[cfg(feature = "vfed")]
impl From<VFedError> for BuildDagError {
    fn from(e: VFedError) -> Self {
        BuildDagError::VFed(e)
    }
}

struct NodeRegistrationInfo {
    execution_time: u64,
    reactor_name: Cow<'static, str>,
    pub_topics: Vec<Cow<'static, str>>,
    sub_topics: Vec<Cow<'static, str>>,
}

fn create_reactor_name(dag_id: u32, node_id: u32) -> Cow<'static, str> {
    Cow::from(format!("dag{dag_id}_node{node_id}"))
}

fn create_sub_topics(dag_id: u32, node_id: u32, in_links: &[u32]) -> Vec<Cow<'static, str>> {
    let mut topics = Vec::with_capacity(in_links.len());
    in_links.iter().for_each(|link| {
        topics.push(Cow::from(format!("dag{dag_id}_path{link}-{node_id}")));
    });
    topics
}

fn create_pub_topics(dag_id: u32, node_id: u32, out_links: &[u32]) -> Vec<Cow<'static, str>> {
    let mut topics = Vec::with_capacity(out_links.len());
    out_links.iter().for_each(|link| {
        topics.push(Cow::from(format!("dag{dag_id}_path{node_id}-{link}")));
    });
    topics
}

fn setup_node_registration(dag_id: u32, node_data: &NodeData) -> NodeRegistrationInfo {
    let node_id = node_data.get_id();

    let execution_time = node_data.get_execution_time();
    let reactor_name = create_reactor_name(dag_id, node_id);
    let pub_topics = create_pub_topics(dag_id, node_id, node_data.get_out_links());
    let sub_topics = create_sub_topics(dag_id, node_id, node_data.get_in_links());

    NodeRegistrationInfo {
        execution_time,
        reactor_name,
        pub_topics,
        sub_topics,
    }
}

macro_rules! register_source {
    ($dag:expr, $node_data:expr, $sched_type:expr, $($T_out:ident),*) => {
        {
            let dag_id = $dag.get_id();
            let registration_info = setup_node_registration(dag_id, $node_data);

            let pub_topics = registration_info.pub_topics;

            assert!(
                [$(stringify!($T_out)),*].len() == pub_topics.len(),
                "LinkNumError::MisMatch: dag_id={:?}, node_id={:?}",
                dag_id,
                $node_data.get_id()
            );

            let reactor_name = registration_info.reactor_name;
            let execution_time = registration_info.execution_time;
            $dag.register_periodic_reactor::<_, ($($T_out,)*)>(
                reactor_name.clone(),
                move || -> ($($T_out,)*) {
                    simulated_execution_time(execution_time);

                    let outputs = ($(execution_time as $T_out,)*);
                    // log::debug!("name: {reactor_name}, outputs: {outputs:?}");
                    outputs
                },
                pub_topics,
                $sched_type,
                convert_duration($node_data.get_period().expect("Source node's period should always be Some.")),
            ).await;
            Ok(())
        }
    };
}

async fn register_source_node(
    dag: &Arc<Dag>,
    node_data: &NodeData,
    sched_type: SchedulerType,
) -> Result<(), LinkNumError> {
    let dag_id = dag.get_id();
    let node_id = node_data.get_id();

    let out_links_num = node_data.get_out_links().len();

    match out_links_num {
        1 => register_source!(dag, node_data, sched_type, u64),
        2 => register_source!(dag, node_data, sched_type, u64, u64),
        3 => register_source!(dag, node_data, sched_type, u64, u64, u64),
        4 => register_source!(dag, node_data, sched_type, u64, u64, u64, u64),
        5 => register_source!(dag, node_data, sched_type, u64, u64, u64, u64, u64),
        6 => register_source!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64),
        7 => register_source!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64),
        8 => register_source!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64, u64),
        9 => register_source!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64, u64, u64),
        10 => register_source!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64),
        11 => register_source!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64),
        12 => register_source!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64),
        13 => register_source!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64),
        14 => register_source!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64),
        15 => register_source!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64),
        16 => register_source!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64),
        _ => Err(LinkNumError::Output(dag_id, node_id)),
    }
}

macro_rules! register_sink {
    ($dag:expr, $node_data:expr, $sched_type:expr, $($T_in:ident),*) => {
        {
            let dag_id = $dag.get_id();
            let registration_info = setup_node_registration(dag_id, $node_data);

            let sub_topics = registration_info.sub_topics;

            assert!(
                [$(stringify!($T_in)),*].len() == sub_topics.len(),
                "LinkNumError::MisMatch: dag_id={:?}, node_id={:?}",
                dag_id,
                $node_data.get_id()
            );

            let reactor_name = registration_info.reactor_name;
            let execution_time = registration_info.execution_time;
            $dag.register_sink_reactor::<_, ($($T_in,)*)>(
                reactor_name.clone(),
                move |inputs: ($($T_in,)*)| {
                    simulated_execution_time(execution_time);
                    // log::debug!("name: {reactor_name}, inputs: {inputs:?}");
                },
                sub_topics,
                $sched_type,
                convert_duration($node_data.get_end_to_end_deadline().expect("Sink node's relative_deadline should always be Some.")),
            ).await;
            Ok(())
        }
    };
}

async fn register_sink_node(
    dag: &Arc<Dag>,
    node_data: &NodeData,
    sched_type: SchedulerType,
) -> Result<(), LinkNumError> {
    let dag_id = dag.get_id();
    let node_id = node_data.get_id();

    let in_links_num = node_data.get_in_links().len();

    match in_links_num {
        1 => register_sink!(dag, node_data, sched_type, u64),
        2 => register_sink!(dag, node_data, sched_type, u64, u64),
        3 => register_sink!(dag, node_data, sched_type, u64, u64, u64),
        4 => register_sink!(dag, node_data, sched_type, u64, u64, u64, u64),
        5 => register_sink!(dag, node_data, sched_type, u64, u64, u64, u64, u64),
        6 => register_sink!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64),
        7 => register_sink!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64),
        8 => register_sink!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64, u64),
        9 => register_sink!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64, u64, u64),
        10 => register_sink!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64),
        11 => register_sink!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64),
        12 => register_sink!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64),
        13 => register_sink!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64),
        14 => register_sink!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64),
        15 => register_sink!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64),
        16 => register_sink!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64),
        _ => Err(LinkNumError::Input(dag_id, node_id)),
    }
}

macro_rules! register_intermediate {
    ($dag:expr, $node_data:expr, $sched_type:expr, $($T_in:ident),*; $($T_out:ident),*) => {
        {
            let dag_id = $dag.get_id();
            let node_id = $node_data.get_id();

            let registration_info = setup_node_registration(dag_id, $node_data);

            let sub_topics = registration_info.sub_topics;
            let pub_topics = registration_info.pub_topics;

            assert!(
                [$(stringify!($T_in)),*].len() == sub_topics.len(),
                "LinkNumError::MisMatch (input topics): dag_id={:?}, node_id={:?}",
                dag_id,
                node_id
            );

            assert!(
                [$(stringify!($T_out)),*].len() == pub_topics.len(),
                "LinkNumError::MisMatch (output topics): dag_id={:?}, node_id={:?}",
                dag_id,
                node_id
            );

            let execution_time = registration_info.execution_time;
            let reactor_name = registration_info.reactor_name;
            $dag.register_reactor::<_, ($($T_in,)*), ($($T_out,)*)>(
                reactor_name.clone(),
                move |inputs: ($($T_in,)*)| -> ($($T_out,)*) {
                    simulated_execution_time(execution_time);
                    let outputs = ($(execution_time as $T_out,)*);
                    // log::debug!("name: {reactor_name}, inputs: {inputs:?}, outputs: {outputs:?}");
                    outputs
                },
                sub_topics,
                pub_topics,
                $sched_type,
            ).await;
            Ok(())
        }
    };
}

async fn register_intermediate_node(
    dag: &Arc<Dag>,
    node_data: &NodeData,
    sched_type: SchedulerType,
) -> Result<(), LinkNumError> {
    let dag_id = dag.get_id();
    let node_id = node_data.get_id();

    let in_links_num = node_data.get_in_links().len();
    let out_links_num = node_data.get_out_links().len();

    match (in_links_num, out_links_num) {
        (1, 1) => register_intermediate!(dag, node_data, sched_type, u64; u64),
        (1, 2) => register_intermediate!(dag, node_data, sched_type, u64; u64, u64),
        (1, 3) => register_intermediate!(dag, node_data, sched_type, u64; u64, u64, u64),
        (1, 4) => register_intermediate!(dag, node_data, sched_type, u64; u64, u64, u64, u64),
        (1, 5) => register_intermediate!(dag, node_data, sched_type, u64; u64, u64, u64, u64, u64),
        (1, 6) => register_intermediate!(dag, node_data, sched_type, u64; u64, u64, u64, u64, u64, u64),
        (1, 7) => register_intermediate!(dag, node_data, sched_type, u64; u64, u64, u64, u64, u64, u64, u64),
        (1, 8) => register_intermediate!(dag, node_data, sched_type, u64; u64, u64, u64, u64, u64, u64, u64, u64),
        (1, 9) => register_intermediate!(dag, node_data, sched_type, u64; u64, u64, u64, u64, u64, u64, u64, u64, u64),
        (1, 10) => register_intermediate!(dag, node_data, sched_type, u64; u64, u64, u64, u64, u64, u64, u64, u64, u64, u64),
        (1, 11) => register_intermediate!(dag, node_data, sched_type, u64; u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64),
        (1, 12) => register_intermediate!(dag, node_data, sched_type, u64; u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64),
        (1, 13) => register_intermediate!(dag, node_data, sched_type, u64; u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64),
        (1, 14) => register_intermediate!(dag, node_data, sched_type, u64; u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64),
        (1, 15) => register_intermediate!(dag, node_data, sched_type, u64; u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64),
        (1, 16) => register_intermediate!(dag, node_data, sched_type, u64; u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64),
        (2, 1) => register_intermediate!(dag, node_data, sched_type, u64, u64; u64),
        (2, 2) => register_intermediate!(dag, node_data, sched_type, u64, u64; u64, u64),
        (2, 3) => register_intermediate!(dag, node_data, sched_type, u64, u64; u64, u64, u64),
        (2, 4) => register_intermediate!(dag, node_data, sched_type, u64, u64; u64, u64, u64, u64),
        (2, 5) => register_intermediate!(dag, node_data, sched_type, u64, u64; u64, u64, u64, u64, u64),
        (2, 6) => register_intermediate!(dag, node_data, sched_type, u64, u64; u64, u64, u64, u64, u64, u64),
        (2, 7) => register_intermediate!(dag, node_data, sched_type, u64, u64; u64, u64, u64, u64, u64, u64, u64),
        (2, 8) => register_intermediate!(dag, node_data, sched_type, u64, u64; u64, u64, u64, u64, u64, u64, u64, u64),
        (2, 9) => register_intermediate!(dag, node_data, sched_type, u64, u64; u64, u64, u64, u64, u64, u64, u64, u64, u64),
        (2, 10) => register_intermediate!(dag, node_data, sched_type, u64, u64; u64, u64, u64, u64, u64, u64, u64, u64, u64, u64),
        (2, 11) => register_intermediate!(dag, node_data, sched_type, u64, u64; u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64),
        (2, 12) => register_intermediate!(dag, node_data, sched_type, u64, u64; u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64),
        (2, 13) => register_intermediate!(dag, node_data, sched_type, u64, u64; u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64),
        (2, 14) => register_intermediate!(dag, node_data, sched_type, u64, u64; u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64),
        (2, 15) => register_intermediate!(dag, node_data, sched_type, u64, u64; u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64),
        (2, 16) => register_intermediate!(dag, node_data, sched_type, u64, u64; u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64),
        (3, 1) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64; u64),
        (3, 2) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64; u64, u64),
        (3, 3) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64; u64, u64, u64),
        (3, 4) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64; u64, u64, u64, u64),
        (3, 5) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64; u64, u64, u64, u64, u64),
        (3, 6) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64; u64, u64, u64, u64, u64, u64),
        (3, 7) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64; u64, u64, u64, u64, u64, u64, u64),
        (3, 8) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64; u64, u64, u64, u64, u64, u64, u64, u64),
        (3, 9) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64; u64, u64, u64, u64, u64, u64, u64, u64, u64),
        (3, 10) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64; u64, u64, u64, u64, u64, u64, u64, u64, u64, u64),
        (3, 11) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64; u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64),
        (3, 12) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64; u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64),
        (3, 13) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64; u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64),
        (3, 14) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64; u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64),
        (3, 15) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64; u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64),
        (3, 16) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64; u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64),
        (4, 1) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64; u64),
        (4, 2) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64; u64, u64),
        (4, 3) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64; u64, u64, u64),
        (4, 4) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64; u64, u64, u64, u64),
        (4, 5) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64; u64, u64, u64, u64, u64),
        (4, 6) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64; u64, u64, u64, u64, u64, u64),
        (4, 7) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64; u64, u64, u64, u64, u64, u64, u64),
        (4, 8) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64; u64, u64, u64, u64, u64, u64, u64, u64),
        (4, 9) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64; u64, u64, u64, u64, u64, u64, u64, u64, u64),
        (4, 10) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64; u64, u64, u64, u64, u64, u64, u64, u64, u64, u64),
        (4, 11) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64; u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64),
        (4, 12) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64; u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64),
        (4, 13) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64; u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64),
        (4, 14) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64; u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64),
        (4, 15) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64; u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64),
        (4, 16) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64; u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64),
        (5, 1) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64; u64),
        (5, 2) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64; u64, u64),
        (5, 3) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64; u64, u64, u64),
        (5, 4) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64; u64, u64, u64, u64),
        (5, 5) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64; u64, u64, u64, u64, u64),
        (5, 6) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64; u64, u64, u64, u64, u64, u64),
        (5, 7) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64; u64, u64, u64, u64, u64, u64, u64),
        (5, 8) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64; u64, u64, u64, u64, u64, u64, u64, u64),
        (5, 9) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64; u64, u64, u64, u64, u64, u64, u64, u64, u64),
        (5, 10) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64; u64, u64, u64, u64, u64, u64, u64, u64, u64, u64),
        (5, 11) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64; u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64),
        (5, 12) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64; u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64),
        (5, 13) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64; u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64),
        (5, 14) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64; u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64),
        (5, 15) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64; u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64),
        (5, 16) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64; u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64),
        (6, 1) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64; u64),
        (6, 2) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64; u64, u64),
        (6, 3) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64; u64, u64, u64),
        (6, 4) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64; u64, u64, u64, u64),
        (6, 5) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64; u64, u64, u64, u64, u64),
        (6, 6) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64; u64, u64, u64, u64, u64, u64),
        (6, 7) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64; u64, u64, u64, u64, u64, u64, u64),
        (6, 8) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64; u64, u64, u64, u64, u64, u64, u64, u64),
        (6, 9) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64; u64, u64, u64, u64, u64, u64, u64, u64, u64),
        (6, 10) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64; u64, u64, u64, u64, u64, u64, u64, u64, u64, u64),
        (6, 11) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64; u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64),
        (6, 12) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64; u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64),
        (6, 13) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64; u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64),
        (6, 14) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64; u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64),
        (6, 15) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64; u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64),
        (6, 16) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64; u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64),
        (7, 1) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64; u64),
        (7, 2) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64; u64, u64),
        (7, 3) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64; u64, u64, u64),
        (7, 4) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64; u64, u64, u64, u64),
        (7, 5) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64; u64, u64, u64, u64, u64),
        (7, 6) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64; u64, u64, u64, u64, u64, u64),
        (7, 7) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64; u64, u64, u64, u64, u64, u64, u64),
        (7, 8) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64; u64, u64, u64, u64, u64, u64, u64, u64),
        (7, 9) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64; u64, u64, u64, u64, u64, u64, u64, u64, u64),
        (7, 10) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64; u64, u64, u64, u64, u64, u64, u64, u64, u64, u64),
        (7, 11) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64; u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64),
        (7, 12) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64; u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64),
        (7, 13) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64; u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64),
        (7, 14) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64; u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64),
        (7, 15) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64; u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64),
        (7, 16) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64; u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64),
        (8, 1) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64, u64; u64),
        (8, 2) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64, u64; u64, u64),
        (8, 3) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64, u64; u64, u64, u64),
        (8, 4) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64, u64; u64, u64, u64, u64),
        (8, 5) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64, u64; u64, u64, u64, u64, u64),
        (8, 6) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64, u64; u64, u64, u64, u64, u64, u64),
        (8, 7) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64, u64; u64, u64, u64, u64, u64, u64, u64),
        (8, 8) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64, u64; u64, u64, u64, u64, u64, u64, u64, u64),
        (8, 9) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64, u64; u64, u64, u64, u64, u64, u64, u64, u64, u64),
        (8, 10) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64, u64; u64, u64, u64, u64, u64, u64, u64, u64, u64, u64),
        (8, 11) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64, u64; u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64),
        (8, 12) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64, u64; u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64),
        (8, 13) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64, u64; u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64),
        (8, 14) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64, u64; u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64),
        (8, 15) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64, u64; u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64),
        (8, 16) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64, u64; u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64),
        (9, 1) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64, u64, u64; u64),
        (9, 2) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64, u64, u64; u64, u64),
        (9, 3) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64, u64, u64; u64, u64, u64),
        (9, 4) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64, u64, u64; u64, u64, u64, u64),
        (9, 5) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64, u64, u64; u64, u64, u64, u64, u64),
        (9, 6) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64, u64, u64; u64, u64, u64, u64, u64, u64),
        (9, 7) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64, u64, u64; u64, u64, u64, u64, u64, u64, u64),
        (9, 8) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64, u64, u64; u64, u64, u64, u64, u64, u64, u64, u64),
        (9, 9) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64, u64, u64; u64, u64, u64, u64, u64, u64, u64, u64, u64),
        (9, 10) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64, u64, u64; u64, u64, u64, u64, u64, u64, u64, u64, u64, u64),
        (9, 11) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64, u64, u64; u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64),
        (9, 12) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64, u64, u64; u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64),
        (9, 13) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64, u64, u64; u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64),
        (9, 14) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64, u64, u64; u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64),
        (9, 15) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64, u64, u64; u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64),
        (9, 16) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64, u64, u64; u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64),
        (10, 1) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64; u64),
        (10, 2) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64; u64, u64),
        (10, 3) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64; u64, u64, u64),
        (10, 4) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64; u64, u64, u64, u64),
        (10, 5) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64; u64, u64, u64, u64, u64),
        (10, 6) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64; u64, u64, u64, u64, u64, u64),
        (10, 7) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64; u64, u64, u64, u64, u64, u64, u64),
        (10, 8) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64; u64, u64, u64, u64, u64, u64, u64, u64),
        (10, 9) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64; u64, u64, u64, u64, u64, u64, u64, u64, u64),
        (10, 10) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64; u64, u64, u64, u64, u64, u64, u64, u64, u64, u64),
        (10, 11) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64; u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64),
        (10, 12) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64; u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64),
        (10, 13) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64; u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64),
        (10, 14) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64; u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64),
        (10, 15) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64; u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64),
        (10, 16) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64; u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64),
        (11, 1) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64; u64),
        (11, 2) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64; u64, u64),
        (11, 3) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64; u64, u64, u64),
        (11, 4) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64; u64, u64, u64, u64),
        (11, 5) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64; u64, u64, u64, u64, u64),
        (11, 6) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64; u64, u64, u64, u64, u64, u64),
        (11, 7) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64; u64, u64, u64, u64, u64, u64, u64),
        (11, 8) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64; u64, u64, u64, u64, u64, u64, u64, u64),
        (11, 9) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64; u64, u64, u64, u64, u64, u64, u64, u64, u64),
        (11, 10) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64; u64, u64, u64, u64, u64, u64, u64, u64, u64, u64),
        (11, 11) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64; u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64),
        (11, 12) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64; u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64),
        (11, 13) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64; u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64),
        (11, 14) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64; u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64),
        (11, 15) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64; u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64),
        (11, 16) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64; u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64),
        (12, 1) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64; u64),
        (12, 2) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64; u64, u64),
        (12, 3) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64; u64, u64, u64),
        (12, 4) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64; u64, u64, u64, u64),
        (12, 5) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64; u64, u64, u64, u64, u64),
        (12, 6) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64; u64, u64, u64, u64, u64, u64),
        (12, 7) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64; u64, u64, u64, u64, u64, u64, u64),
        (12, 8) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64; u64, u64, u64, u64, u64, u64, u64, u64),
        (12, 9) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64; u64, u64, u64, u64, u64, u64, u64, u64, u64),
        (12, 10) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64; u64, u64, u64, u64, u64, u64, u64, u64, u64, u64),
        (12, 11) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64; u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64),
        (12, 12) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64; u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64),
        (12, 13) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64; u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64),
        (12, 14) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64; u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64),
        (12, 15) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64; u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64),
        (12, 16) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64; u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64),
        (13, 1) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64; u64),
        (13, 2) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64; u64, u64),
        (13, 3) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64; u64, u64, u64),
        (13, 4) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64; u64, u64, u64, u64),
        (13, 5) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64; u64, u64, u64, u64, u64),
        (13, 6) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64; u64, u64, u64, u64, u64, u64),
        (13, 7) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64; u64, u64, u64, u64, u64, u64, u64),
        (13, 8) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64; u64, u64, u64, u64, u64, u64, u64, u64),
        (13, 9) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64; u64, u64, u64, u64, u64, u64, u64, u64, u64),
        (13, 10) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64; u64, u64, u64, u64, u64, u64, u64, u64, u64, u64),
        (13, 11) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64; u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64),
        (13, 12) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64; u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64),
        (13, 13) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64; u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64),
        (13, 14) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64; u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64),
        (13, 15) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64; u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64),
        (13, 16) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64; u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64),
        (14, 1) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64; u64),
        (14, 2) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64; u64, u64),
        (14, 3) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64; u64, u64, u64),
        (14, 4) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64; u64, u64, u64, u64),
        (14, 5) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64; u64, u64, u64, u64, u64),
        (14, 6) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64; u64, u64, u64, u64, u64, u64),
        (14, 7) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64; u64, u64, u64, u64, u64, u64, u64),
        (14, 8) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64; u64, u64, u64, u64, u64, u64, u64, u64),
        (14, 9) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64; u64, u64, u64, u64, u64, u64, u64, u64, u64),
        (14, 10) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64; u64, u64, u64, u64, u64, u64, u64, u64, u64, u64),
        (14, 11) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64; u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64),
        (14, 12) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64; u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64),
        (14, 13) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64; u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64),
        (14, 14) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64; u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64),
        (14, 15) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64; u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64),
        (14, 16) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64; u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64),
        (15, 1) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64; u64),
        (15, 2) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64; u64, u64),
        (15, 3) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64; u64, u64, u64),
        (15, 4) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64; u64, u64, u64, u64),
        (15, 5) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64; u64, u64, u64, u64, u64),
        (15, 6) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64; u64, u64, u64, u64, u64, u64),
        (15, 7) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64; u64, u64, u64, u64, u64, u64, u64),
        (15, 8) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64; u64, u64, u64, u64, u64, u64, u64, u64),
        (15, 9) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64; u64, u64, u64, u64, u64, u64, u64, u64, u64),
        (15, 10) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64; u64, u64, u64, u64, u64, u64, u64, u64, u64, u64),
        (15, 11) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64; u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64),
        (15, 12) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64; u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64),
        (15, 13) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64; u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64),
        (15, 14) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64; u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64),
        (15, 15) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64; u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64),
        (15, 16) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64; u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64),
        (16, 1) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64; u64),
        (16, 2) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64; u64, u64),
        (16, 3) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64; u64, u64, u64),
        (16, 4) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64; u64, u64, u64, u64),
        (16, 5) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64; u64, u64, u64, u64, u64),
        (16, 6) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64; u64, u64, u64, u64, u64, u64),
        (16, 7) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64; u64, u64, u64, u64, u64, u64, u64),
        (16, 8) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64; u64, u64, u64, u64, u64, u64, u64, u64),
        (16, 9) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64; u64, u64, u64, u64, u64, u64, u64, u64, u64),
        (16, 10) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64; u64, u64, u64, u64, u64, u64, u64, u64, u64, u64),
        (16, 11) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64; u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64),
        (16, 12) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64; u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64),
        (16, 13) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64; u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64),
        (16, 14) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64; u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64),
        (16, 15) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64; u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64),
        (16, 16) => register_intermediate!(dag, node_data, sched_type, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64; u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64),
        (i, o) if i > 16 && o > 16 => Err(LinkNumError::InOut(dag_id, node_id)),
        (i, _) if i > 16 => Err(LinkNumError::Input(dag_id, node_id)),
        (_, o) if o > 16 => Err(LinkNumError::Output(dag_id, node_id)),
        // Every (in, out) pair with both sides in 1..=16 is handled above; this
        // is a defensive fallback only, so a truly unexpected shape fails this
        // one DAG gracefully rather than panicking the whole kernel.
        _ => Err(LinkNumError::InOut(dag_id, node_id)),
    }
}

pub(super) async fn build_dag(dag_data: DagData) -> Result<Arc<Dag>, BuildDagError> {
    let dag = create_dag();
    let dag_id = dag.get_id();

    let stats = compute_dag_stats(&dag_data);
    log::debug!(
        "DAG#{dag_id}: volume(C)={}, critical_path(L)={}",
        stats.volume,
        stats.critical_path
    );

    let period = dag_data
        .get_nodes()
        .iter()
        .find(|node| node.is_source())
        .and_then(NodeData::get_period)
        .ok_or(BuildDagError::MissingDagTiming(dag_id))?;
    let relative_deadline = dag_data
        .get_nodes()
        .iter()
        .find(|node| node.is_sink())
        .and_then(NodeData::get_end_to_end_deadline)
        .ok_or(BuildDagError::MissingDagTiming(dag_id))?;

    let config = DagMetrics::from_static(stats.volume, stats.critical_path, period, relative_deadline);

    #[cfg(not(any(feature = "vfed", feature = "laxity")))]
    let sched_type = {
        let assignment = admit_dag(config)?;
        let sched_type = assignment.scheduler_type;
        if let SchedulerType::ClusteredEDF(deadline, cluster) = sched_type {
            let cores: Vec<usize> = cluster.iter().collect();
            log::info!(
                "DAG#{dag_id}: admitted (federated) as {:?} ({:?}) -> ClusteredEDF(relative_deadline={deadline}, cores={cores:?})",
                assignment.class,
                assignment.source
            );
        } else {
            log::info!(
                "DAG#{dag_id}: admitted (federated) as {:?} ({:?}) -> {sched_type:?}",
                assignment.class,
                assignment.source
            );
        }
        sched_type
    };

    #[cfg(feature = "vfed")]
    let sched_type = {
        let assignment = vfed::admit_one(config, PackingStrategy::FirstFit)?;
        let sched_type = assignment
            .into_scheduler_type(relative_deadline)
            .ok_or(BuildDagError::VFedSchedulerTypeMissing(dag_id))?;
        log::info!("DAG#{dag_id}: admitted (vfed) as {assignment:?} -> {sched_type:?}");
        sched_type
    };

    // "Static Laxity-Based" baseline: plain (non-clustered) GEDF, with a
    // per-node priority tie-break computed offline from `compute_node_laxity`
    // (shorter laxity = more urgent = higher priority, per `dag.rs`/
    // `gedf.rs`/`clustered_edf.rs`'s shared "higher `node_priority` picked
    // first" convention). This baseline has no published primary source of
    // its own -- see `compute_node_laxity`'s own doc for why -- it exists as
    // a self-derived structural counterpart to He et al. 2019 for
    // comparison purposes, not a reproduction of any specific paper.
    #[cfg(feature = "laxity")]
    let (sched_type, node_laxity) = {
        let laxity =
            compute_node_laxity(&dag_data, relative_deadline).ok_or(BuildDagError::LaxityInfeasible(dag_id))?;
        let sched_type = SchedulerType::GEDF(relative_deadline);
        log::info!(
            "DAG#{dag_id}: admitted (laxity) {config:?} -> {sched_type:?}, node_laxity={laxity:?}"
        );
        (sched_type, laxity)
    };

    for node in dag_data.get_nodes() {
        if node.is_source() {
            register_source_node(&dag, node, sched_type).await?;
        } else if node.is_sink() {
            register_sink_node(&dag, node, sched_type).await?;
        } else {
            register_intermediate_node(&dag, node, sched_type).await?;
        }

        // Shorter laxity = more urgent: invert so it maps to a *larger*
        // `node_priority`, matching `gedf.rs`'s ordering
        // (`GEDFTask`'s `Ord` picks the *largest* `node_priority` first
        // among equal-deadline tasks). Called here, before
        // `finish_create_dags`, per `set_node_priority`'s own doc
        // requirement. `node.get_id()` is safe to use directly as the
        // runtime `node_id`: `DagData::get_nodes()` is backed by a
        // `BTreeMap<u32, NodeData>` keyed by this same id
        // (parse_yaml.rs's `convert_to_dag`), so registration happens in
        // ascending-id order and lines up with the sequential id
        // `register_*_reactor` assigns internally.
        #[cfg(feature = "laxity")]
        if let Some(&laxity) = node_laxity.get(&node.get_id()) {
            dag.set_node_priority(node.get_id(), u64::MAX.saturating_sub(laxity));
        }
    }

    Ok(dag)
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn test_create_reactor_name() {
        let dag_id = 1;
        let node_id = 2;
        let expected_name = Cow::from("dag1_node2");
        let reactor_name = create_reactor_name(dag_id, node_id);
        assert_eq!(reactor_name, expected_name);
    }

    #[test]
    fn test_create_sub_topics() {
        let dag_id = 1;
        let node_id = 2;
        let in_links = vec![3, 4];
        let expected_topics = vec![Cow::from("dag1_path3-2"), Cow::from("dag1_path4-2")];
        let sub_topics = create_sub_topics(dag_id, node_id, &in_links);
        assert_eq!(sub_topics, expected_topics);
    }

    #[test]
    fn test_create_pub_topics() {
        let dag_id = 1;
        let node_id = 2;
        let out_links = vec![3, 4];
        let expected_topics = vec![Cow::from("dag1_path2-3"), Cow::from("dag1_path2-4")];
        let pub_topics = create_pub_topics(dag_id, node_id, &out_links);
        assert_eq!(pub_topics, expected_topics);
    }
}
