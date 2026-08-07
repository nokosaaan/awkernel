//! Provides functionality for building and executing Directed Acyclic Graphs (DAGs) of reactors.
//!
//! # Main Workflow
//!
//! 1. Create a new `Dag` instance by calling the `create_dag` function.
//! 2. Define the graph structure by calling methods on the `Dag` instance (e.g., `register_reactor`) to register reactors as nodes.
//! 3. Once all DAGs are defined, call the `finish_create_dags` function.
//!    This validates the DAGs and spawns all registered reactor tasks, starting their execution.
//!
//! # Key Public Components
//!
//! - `Dag`: The central struct representing an individual task graph.
//! - `create_dag()`: The entry point for creating a new DAG.
//! - `register_xx_reactor()`: Methods to register different types of reactors.
//! - `finish_create_dags()`: The function to validate the defined DAGs and start their execution.
//! - `DagError`: Defines potential errors that can occur during DAG validation.
//!
//! # DAG Constraints and Validation
//!
//! To ensure data flow integrity and predictable execution, the system enforces several constraints during the `finish_create_dags` call.
//!
//! ## 1. Structural Integrity
//! These rules concern the overall shape of the graph.
//!
//! - **Must be Acyclic**: The graph cannot contain any directed cycles.
//! - **Must be Connected**: All nodes must form a single, weakly connected component. Orphaned nodes or subgraphs are not allowed.
//!
//! ## 2. Register API Interface Correctness
//!
//! - **Arity Matching**: The number of topics a reactor subscribes to must match its function's argument count. The number of topics it publishes must match its return value count.
//! - **No Duplicate Subscriptions/Publications**: A single reactor cannot subscribe to or publish the same topic multiple times.
//!
//! ## 3. Data Flow Integrity
//!
//! - **No Dangling Edges**: Every published topic must be subscribed to by another node, and every subscribed topic must have a publisher.
//! - **Globally Unique Topics (Inter-DAG)**: A topic name must be unique across all active DAGs in the system. This prevents different data pipelines from unintentionally interfering with each other.
//!
//! # Example Usage
//!
//! See the `applications/tests/test_dag` example for practical usage.
//!
//! # Notes
//!
//! This constraint may be relaxed in the future to support more complex topologies.
//!
//! - **Single Publisher per Topic (Intra-DAG)**: Within a single DAG, a topic can only be published to by one node.
//! - **Source and Sink**: The current implementation assumes that a DAG has a single source and a single sink.
//!
mod graph;
mod unionfind;
mod visit;

#[cfg(feature = "perf")]
mod performance;

use crate::{
    dag::{
        graph::{
            algo::{connected_components, is_cyclic_directed},
            direction::Direction,
            NodeIndex,
        },
        visit::{EdgeRef, IntoNodeReferences, NodeRef},
    },
    scheduler::SchedulerType,
    task::DagInfo,
    time_interval::interval,
    Attribute, MultipleReceiver, MultipleSender, VectorToPublishers, VectorToSubscribers,
};

#[cfg(feature = "period-index-propagation")]
use crate::task::perf::{
    get_period_index, increment_period_index, record_subscribe_timestamp,
    update_cycle_end_timestamp, update_cycle_start_timestamp,
};

use alloc::{
    borrow::Cow,
    boxed::Box,
    collections::{btree_map, btree_set::BTreeSet, BTreeMap},
    sync::Arc,
    vec::Vec,
};
use awkernel_lib::sync::mutex::{MCSNode, Mutex};
use core::{future::Future, pin::Pin, sync::atomic::Ordering, time::Duration};

#[cfg(feature = "perf")]
use performance::ResponseInfo;

static DAGS: Mutex<Dags> = Mutex::new(Dags::new()); // Set of DAGs.

// The following BTreeMaps use dag_id (u32) as the key.
static PENDING_TASKS: Mutex<BTreeMap<u32, Vec<PendingTask>>> = Mutex::new(BTreeMap::new());
static SOURCE_PENDING_TASKS: Mutex<BTreeMap<u32, PendingTask>> = Mutex::new(BTreeMap::new());
static MISMATCH_SUBSCRIBE_NODES: Mutex<BTreeMap<u32, Vec<usize>>> = Mutex::new(BTreeMap::new());
static MISMATCH_PUBLISH_NODES: Mutex<BTreeMap<u32, Vec<usize>>> = Mutex::new(BTreeMap::new());
static DUPLICATE_SUBSCRIBE_NODES: Mutex<BTreeMap<u32, Vec<usize>>> = Mutex::new(BTreeMap::new());
static DUPLICATE_PUBLISH_NODES: Mutex<BTreeMap<u32, Vec<usize>>> = Mutex::new(BTreeMap::new());

type MeasureF = Arc<dyn Fn() + Send + Sync + 'static>;

pub trait TupleSize {
    const SIZE: usize;
}

macro_rules! impl_tuple_size {
    (@count) => { 0 };
    (@count $_head:ident $($tail:ident)*) => { 1 + impl_tuple_size!(@count $($tail)*) };

    ($($T:ident),*) => {
        impl<$($T),*> TupleSize for ($($T,)*) {
            const SIZE: usize = impl_tuple_size!(@count $($T)*);
        }
    };
}

impl_tuple_size!();
impl_tuple_size!(T1);
impl_tuple_size!(T1, T2);
impl_tuple_size!(T1, T2, T3);

#[derive(Clone)]
pub enum DagError {
    NotWeaklyConnected(u32),
    ContainsCycle(u32),
    MissingPendingTasks(u32),
    MultipleSourceNodes(u32),
    MultipleSinkNodes(u32),
    NoPublisherFound(u32, usize),
    NoSubscriberFound(u32, usize),
    SubscribeArityMismatch(u32, usize),
    PublishArityMismatch(u32, usize),
    DuplicateSubscribe(u32, usize),
    DuplicatePublish(u32, usize),
    TopicHasMultiplePublishers(u32, Cow<'static, str>),
    InterDagTopicConflict(Cow<'static, str>, Vec<u32>),
}

#[rustfmt::skip]
impl core::fmt::Display for DagError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            DagError::NotWeaklyConnected(dag_id) => write!(f, "DAG#{dag_id} is not weakly connected"),
            DagError::ContainsCycle(dag_id) => write!(f, "DAG#{dag_id} contains a cycle"),
            DagError::MissingPendingTasks(dag_id) => write!(f, "DAG#{dag_id} has missing pending tasks"),
            DagError::MultipleSourceNodes(dag_id) => write!(f, "DAG#{dag_id} has multiple source nodes"),
            DagError::MultipleSinkNodes(dag_id) => write!(f, "DAG#{dag_id} has multiple sink nodes"),
            DagError::NoPublisherFound(dag_id, node_id) => {
                write!(f, "DAG#{dag_id} Node#{node_id}: One or more subscribed topics have no corresponding publisher")
            }
            DagError::NoSubscriberFound(dag_id, node_id) => {
                write!(f, "DAG#{dag_id} Node#{node_id}: One or more published topics have no corresponding subscriber")
            }
            DagError::SubscribeArityMismatch(dag_id, node_id) => {
                write!(f, "DAG#{dag_id} Node#{node_id}: Mismatch in subscribed topics and arguments")
            }
            DagError::PublishArityMismatch(dag_id, node_id) => {
                write!(f, "DAG#{dag_id} Node#{node_id}: Mismatch in published topics and return values")
            }
            DagError::DuplicateSubscribe(dag_id, node_id) => {
                write!(f, "DAG#{dag_id} Node#{node_id}: found duplicate subscription,")
            }
            DagError::DuplicatePublish(dag_id, node_id) => {
                write!(f, "DAG#{dag_id} Node#{node_id}: found duplicate publication.")
            }
            DagError::TopicHasMultiplePublishers(dag_id, topic_name) => {
                write!(f, "DAG#{dag_id}: Topic '{topic_name}' has multiple publishers")
            }
            DagError::InterDagTopicConflict(topic_name, dag_ids) => {
                write!(f, "Topic '{topic_name}' is used in multiple DAGs. Conflicting DAG IDs: {dag_ids:?}")
            }
        }
    }
}

pub struct Dag {
    id: u32,
    graph: Mutex<graph::Graph<NodeInfo, EdgeInfo>>,
    job_table: Mutex<DagJobTable>,

    #[cfg(feature = "perf")]
    response_info: Mutex<ResponseInfo>,
}

/// Number of concurrent job (DAG instance) absolute deadlines a single
/// [`Dag`] can track at once, indexed by `job_index % DAG_JOB_TABLE_SIZE`.
///
/// A DAG whose relative deadline `D` exceeds its period `T` ("arbitrary
/// deadline") can have more than one instance in flight at the same time —
/// up to roughly `ceil(D / T)` of them, since the source releases a new
/// instance every `T` while an older instance's downstream nodes may still
/// be running. Sizing this per-DAG would need a dynamic allocation on the
/// scheduling-critical wake path, so it is instead one fixed, generous bound
/// shared by every DAG (bounded WCET, allocation-free lookups). If a DAG's
/// `D / T` ratio ever exceeds this, [`calculate_and_update_dag_deadline`]
/// logs a warning and falls back to an approximation rather than silently
/// reusing a stale entry.
const DAG_JOB_TABLE_SIZE: usize = 16;

/// Per-DAG-instance ("job") absolute deadlines, keyed by a monotonically
/// increasing job index. Replaces a single `Mutex<Option<u64>>` shared by
/// every node of a DAG, which corrupted the deadline of an older, still
/// in-flight instance whenever a newer instance's source node recomputed and
/// overwrote it (see [`DAG_JOB_TABLE_SIZE`]).
struct DagJobTable {
    deadlines: [Option<u64>; DAG_JOB_TABLE_SIZE],
}

impl DagJobTable {
    const fn new() -> Self {
        Self {
            deadlines: [None; DAG_JOB_TABLE_SIZE],
        }
    }

    fn slot(job_index: u64) -> usize {
        (job_index % DAG_JOB_TABLE_SIZE as u64) as usize
    }

    fn set(&mut self, job_index: u64, deadline: u64) {
        self.deadlines[Self::slot(job_index)] = Some(deadline);
    }

    fn get(&self, job_index: u64) -> Option<u64> {
        self.deadlines[Self::slot(job_index)]
    }
}

impl Dag {
    pub fn get_id(&self) -> u32 {
        self.id
    }

    fn get_source_nodes(&self) -> Vec<NodeIndex> {
        let mut node = MCSNode::new();
        let graph = self.graph.lock(&mut node);
        graph.externals(Direction::Incoming).collect()
    }

    fn get_sink_nodes(&self) -> Vec<NodeIndex> {
        let mut node = MCSNode::new();
        let graph = self.graph.lock(&mut node);
        graph.externals(Direction::Outgoing).collect()
    }

    pub fn is_source_node(&self, node_index: NodeIndex) -> bool {
        let source_nodes = self.get_source_nodes();
        source_nodes.contains(&node_index)
    }

    // Returns the relative deadline of the first sink node, if it exists.
    pub fn get_sink_relative_deadline(&self) -> Option<Duration> {
        let sink_nodes: Vec<NodeIndex> = self.get_sink_nodes();

        if let Some(sink_node_index) = sink_nodes.first() {
            self.graph
                .lock(&mut MCSNode::new())
                .node_weight(*sink_node_index)?
                .relative_deadline
        } else {
            None
        }
    }

    fn set_relative_deadline(&self, node_idx: NodeIndex, deadline: Duration) {
        let mut node = MCSNode::new();
        let mut graph = self.graph.lock(&mut node);
        graph.node_weight_mut(node_idx).unwrap().relative_deadline = Some(deadline);
    }

    fn add_node_with_topic_edges(
        &self,
        subscribe_topic_names: &[Cow<'static, str>],
        publish_topic_names: &[Cow<'static, str>],
    ) -> NodeIndex {
        let add_node_info = NodeInfo {
            task_id: 0, // Temporary task_id
            subscribe_topic_names: subscribe_topic_names.to_vec(),
            publish_topic_names: publish_topic_names.to_vec(),
            relative_deadline: None,
        };

        let mut node = MCSNode::new();
        let mut graph = self.graph.lock(&mut node);
        let add_node_idx = graph.add_node(add_node_info);

        let subscribe_topic_set: BTreeSet<_> = subscribe_topic_names.iter().collect();
        let publish_topic_set: BTreeSet<_> = publish_topic_names.iter().collect();

        if subscribe_topic_names.len() != subscribe_topic_set.len() {
            let mut node = MCSNode::new();
            let mut duplicate_subscribe_nodes = DUPLICATE_SUBSCRIBE_NODES.lock(&mut node);
            duplicate_subscribe_nodes
                .entry(self.id)
                .or_default()
                .push(add_node_idx.index());
        }
        if publish_topic_names.len() != publish_topic_set.len() {
            let mut node = MCSNode::new();
            let mut duplicate_publish_nodes = DUPLICATE_PUBLISH_NODES.lock(&mut node);
            duplicate_publish_nodes
                .entry(self.id)
                .or_default()
                .push(add_node_idx.index());
        }

        let edges_to_add: Vec<_> = graph
            .node_references()
            .flat_map(|node_ref| {
                let node_info = node_ref.weight();

                let edges_from = node_info
                    .subscribe_topic_names
                    .iter()
                    .filter(|topic| publish_topic_set.contains(*topic))
                    .map(move |topic| (add_node_idx, node_ref.id(), topic.clone()));

                let edges_to = node_info
                    .publish_topic_names
                    .iter()
                    .filter(|topic| subscribe_topic_set.contains(*topic))
                    .map(move |topic| (node_ref.id(), add_node_idx, topic.clone()));

                edges_to.chain(edges_from).collect::<Vec<_>>()
            })
            .collect();

        for (from, to, topic_name) in edges_to_add {
            graph.add_edge(from, to, EdgeInfo { topic_name });
        }

        add_node_idx
    }

    pub async fn register_reactor<F, Args, Ret>(
        &self,
        reactor_name: Cow<'static, str>,
        f: F,
        subscribe_topic_names: Vec<Cow<'static, str>>,
        publish_topic_names: Vec<Cow<'static, str>>,
        sched_type: SchedulerType,
    ) where
        F: Fn(
                <Args::Subscribers as MultipleReceiver>::Item,
            ) -> <Ret::Publishers as MultipleSender>::Item
            + Send
            + 'static,
        Args: VectorToSubscribers,
        Ret: VectorToPublishers,
        Ret::Publishers: Send,
        Args::Subscribers: Send,
        <Args::Subscribers as MultipleReceiver>::Item: TupleSize,
        <Ret::Publishers as MultipleSender>::Item: TupleSize,
    {
        let node_idx = self.add_node_with_topic_edges(&subscribe_topic_names, &publish_topic_names);
        self.check_subscribe_mismatch::<Args>(&subscribe_topic_names, node_idx);
        self.check_publish_mismatch::<Ret>(&publish_topic_names, node_idx);

        // To prevent errors caused by ownership moves
        let dag_id = self.id;
        let node_id = node_idx.index() as u32;

        let mut node = MCSNode::new();
        let mut pending_tasks = PENDING_TASKS.lock(&mut node);
        pending_tasks
            .entry(self.id)
            .or_default()
            .push(PendingTask::new(node_idx, move || {
                Box::pin(async move {
                    spawn_reactor::<F, Args, Ret>(
                        reactor_name,
                        f,
                        subscribe_topic_names,
                        publish_topic_names,
                        sched_type,
                        DagInfo::new(dag_id, node_id),
                    )
                    .await
                })
            }));
    }

    pub async fn register_periodic_reactor<F, Ret>(
        &self,
        reactor_name: Cow<'static, str>,
        f: F,
        publish_topic_names: Vec<Cow<'static, str>>,
        sched_type: SchedulerType,
        period: Duration,
    ) where
        F: Fn() -> <Ret::Publishers as MultipleSender>::Item + Send + 'static,
        Ret: VectorToPublishers,
        Ret::Publishers: Send,
        <Ret::Publishers as MultipleSender>::Item: TupleSize,
    {
        let node_idx = self.add_node_with_topic_edges(&Vec::new(), &publish_topic_names);
        self.check_publish_mismatch::<Ret>(&publish_topic_names, node_idx);

        let measure_f: Option<MeasureF> = {
            #[cfg(feature = "perf")]
            {
                Some(performance::create_release_recorder(self.id))
            }
            #[cfg(not(feature = "perf"))]
            {
                None
            }
        };

        // To prevent errors caused by ownership moves
        let dag_id = self.id;
        let node_id = node_idx.index() as u32;

        let mut node = MCSNode::new();
        let mut source_pending_tasks = SOURCE_PENDING_TASKS.lock(&mut node);
        source_pending_tasks.insert(
            self.id,
            PendingTask::new(node_idx, move || {
                Box::pin(async move {
                    spawn_periodic_reactor::<F, Ret>(
                        reactor_name,
                        f,
                        publish_topic_names,
                        sched_type,
                        period,
                        measure_f,
                        DagInfo::new(dag_id, node_id),
                    )
                    .await
                })
            }),
        );
    }

    pub async fn register_sink_reactor<F, Args>(
        &self,
        reactor_name: Cow<'static, str>,
        f: F,
        subscribe_topic_names: Vec<Cow<'static, str>>,
        sched_type: SchedulerType,
        relative_deadline: Duration,
    ) where
        F: Fn(<Args::Subscribers as MultipleReceiver>::Item) + Send + 'static,
        Args: VectorToSubscribers,
        Args::Subscribers: Send,
        <Args::Subscribers as MultipleReceiver>::Item: TupleSize,
    {
        let node_idx = self.add_node_with_topic_edges(&subscribe_topic_names, &Vec::new());
        self.set_relative_deadline(node_idx, relative_deadline);
        self.check_subscribe_mismatch::<Args>(&subscribe_topic_names, node_idx);

        let measure_f = {
            #[cfg(feature = "perf")]
            {
                let dag_id = self.id;
                move |arg: <Args::Subscribers as MultipleReceiver>::Item| {
                    f(arg);
                    performance::record_response_time(dag_id);
                }
            }
            #[cfg(not(feature = "perf"))]
            {
                move |arg: <Args::Subscribers as MultipleReceiver>::Item| {
                    f(arg);
                }
            }
        };

        // To prevent errors caused by ownership moves
        let dag_id = self.id;
        let node_id = node_idx.index() as u32;

        let mut node = MCSNode::new();
        let mut pending_tasks = PENDING_TASKS.lock(&mut node);
        pending_tasks
            .entry(self.id)
            .or_default()
            .push(PendingTask::new(node_idx, move || {
                Box::pin(async move {
                    spawn_sink_reactor::<_, Args>(
                        reactor_name,
                        measure_f,
                        subscribe_topic_names,
                        sched_type,
                        DagInfo::new(dag_id, node_id),
                    )
                    .await
                })
            }));
    }

    fn check_subscribe_mismatch<Args>(
        &self,
        subscribe_topic_names: &[Cow<'static, str>],
        node_idx: NodeIndex,
    ) where
        Args: VectorToSubscribers,
        <Args::Subscribers as MultipleReceiver>::Item: TupleSize,
    {
        if <Args::Subscribers as MultipleReceiver>::Item::SIZE != subscribe_topic_names.len() {
            let mut node = MCSNode::new();
            let mut mismatch_subscribe_nodes = MISMATCH_SUBSCRIBE_NODES.lock(&mut node);
            mismatch_subscribe_nodes
                .entry(self.id)
                .or_default()
                .push(node_idx.index());
        }
    }

    fn check_publish_mismatch<Ret>(
        &self,
        publish_topic_names: &[Cow<'static, str>],
        node_idx: NodeIndex,
    ) where
        Ret: VectorToPublishers,
        <Ret::Publishers as MultipleSender>::Item: TupleSize,
    {
        if <Ret::Publishers as MultipleSender>::Item::SIZE != publish_topic_names.len() {
            let mut node = MCSNode::new();
            let mut mismatch_publish_nodes = MISMATCH_PUBLISH_NODES.lock(&mut node);
            mismatch_publish_nodes
                .entry(self.id)
                .or_default()
                .push(node_idx.index());
        }
    }

    /// Record `deadline` as the absolute deadline of job `job_index` of this
    /// DAG. Called once per instance, by that instance's source node.
    #[inline(always)]
    fn set_job_absolute_deadline(&self, job_index: u64, deadline: u64) {
        let mut node = MCSNode::new();
        let mut job_table = self.job_table.lock(&mut node);
        job_table.set(job_index, deadline);
    }

    /// Look up the absolute deadline previously recorded for job `job_index`
    /// of this DAG. Returns `None` if the source has not recorded it yet, or
    /// if it was already overwritten by a later job wrapping around the
    /// fixed-size ring (see [`DAG_JOB_TABLE_SIZE`]).
    #[inline(always)]
    fn get_job_absolute_deadline(&self, job_index: u64) -> Option<u64> {
        let mut node = MCSNode::new();
        let job_table = self.job_table.lock(&mut node);
        job_table.get(job_index)
    }
}

struct PendingTask {
    node_idx: NodeIndex,
    spawn: Box<dyn FnOnce() -> Pin<Box<dyn Future<Output = u32> + Send>> + Send>,
}

impl PendingTask {
    fn new<F>(node_idx: NodeIndex, spawn_fn: F) -> Self
    where
        F: FnOnce() -> Pin<Box<dyn Future<Output = u32> + Send>> + Send + 'static,
    {
        Self {
            node_idx,
            spawn: Box::new(spawn_fn),
        }
    }
}

struct NodeInfo {
    task_id: u32,
    subscribe_topic_names: Vec<Cow<'static, str>>,
    publish_topic_names: Vec<Cow<'static, str>>,
    relative_deadline: Option<Duration>,
}

pub fn to_node_index(index: u32) -> NodeIndex {
    NodeIndex::new(index as usize)
}

struct EdgeInfo {
    topic_name: Cow<'static, str>,
}

/// DAGs.
struct Dags {
    candidate_id: u32, // Next candidate of Dag ID.
    id_to_dag: BTreeMap<u32, Arc<Dag>>,
}

impl Dags {
    const fn new() -> Self {
        Self {
            candidate_id: 1,
            id_to_dag: BTreeMap::new(),
        }
    }

    fn create(&mut self) -> Arc<Dag> {
        let mut id = self.candidate_id;
        loop {
            // Find an unused DAG ID.
            if let btree_map::Entry::Vacant(e) = self.id_to_dag.entry(id) {
                let dag = Arc::new(Dag {
                    id,
                    graph: Mutex::new(graph::Graph::new()),
                    job_table: Mutex::new(DagJobTable::new()),

                    #[cfg(feature = "perf")]
                    response_info: Mutex::new(ResponseInfo::new()),
                });

                e.insert(dag.clone());
                self.candidate_id = id;

                return dag;
            } else {
                // The candidate DAG ID is already used.
                // Check next candidate.
                // If it overflows, start from 1.
                id = id.wrapping_add(1).max(1);
            }
        }
    }

    #[inline(always)]
    fn remove(&mut self, id: u32) {
        self.id_to_dag.remove(&id);
    }
}

pub fn create_dag() -> Arc<Dag> {
    let mut node = MCSNode::new();
    let mut dags = DAGS.lock(&mut node);
    dags.create()
}

pub fn get_dag(id: u32) -> Option<Arc<Dag>> {
    let mut node = MCSNode::new();
    let dags = DAGS.lock(&mut node);
    dags.id_to_dag.get(&id).cloned()
}

/// `(dag_id, src_node_id, dst_node_id)` of every edge of every registered DAG.
///
/// Used by the trace dump (`task::trace::dump_to_console`) so the host can
/// reconstruct the DAG topology (e.g. critical-path analysis) from a serial
/// log alone.  Node ids match `DagInfo::node_id`.
pub fn get_all_dag_edges() -> Vec<(u32, u32, u32)> {
    let dags = {
        let mut node = MCSNode::new();
        let dags = DAGS.lock(&mut node);
        dags.id_to_dag
            .iter()
            .map(|(id, dag)| (*id, dag.clone()))
            .collect::<Vec<_>>()
    };

    let mut edges = Vec::new();
    for (id, dag) in dags {
        let mut node = MCSNode::new();
        let graph = dag.graph.lock(&mut node);
        for e in graph.edge_references() {
            edges.push((id, e.source().index() as u32, e.target().index() as u32));
        }
    }
    edges
}

fn get_dag_sink_relative_deadline_ms(dag_id: u32) -> u64 {
    let dag = get_dag(dag_id).unwrap_or_else(|| panic!("DAG {dag_id} not found"));
    dag.get_sink_relative_deadline()
        .map(|deadline| deadline.as_millis() as u64)
        .unwrap_or_else(|| panic!("DAG {dag_id} has no sink relative deadline set"))
}

/// This node's job index: how many jobs (DAG instances) it has already
/// consumed/released so far, i.e. the index of the job whose message this
/// wake is about to process.
///
/// Derived from [`DagInfo::period_index`], which every reactor body
/// (source/intermediate/sink, see `spawn_reactor` and friends) stores to
/// only once it has actually consumed (or, for the source, released) a
/// job's data. At `wake_task` time — before the future runs and reads the
/// new message — the counter therefore still holds the *previous* job's
/// index, so the job this wake belongs to is always `previous + 1`. This
/// holds uniformly for the source (whose "previous" is the job it released
/// last) and for intermediate/sink nodes (whose "previous" is the job whose
/// message they last consumed): every node sees every job exactly once, in
/// release order, because pubsub topics have a single publisher and an
/// in-order queue, and no node here filters, reorders, or drops jobs.
///
/// Without the `period-index-propagation` feature the counter never moves
/// off `0`, so every job maps to the same slot here — i.e. this degrades to
/// the pre-fix single-shared-deadline behavior (no worse than before)
/// rather than silently mis-tracking jobs it has no way to distinguish.
fn next_job_index(dag_info: &DagInfo) -> u64 {
    dag_info.period_index.load(Ordering::Relaxed) as u64 + 1
}

/// Compute (for a source node) or look up (for every other node) the
/// absolute deadline of the job this wake belongs to, disambiguated by
/// [`next_job_index`] so that two instances of the same DAG in flight at
/// once (an "arbitrary deadline" DAG, i.e. relative deadline `D` > period
/// `T`) do not clobber each other's deadline — see [`DAG_JOB_TABLE_SIZE`].
pub fn calculate_and_update_dag_deadline(dag_info: &DagInfo, wake_time: u64) -> u64 {
    let dag_id = dag_info.dag_id;
    let job_index = next_job_index(dag_info);

    let dag = get_dag(dag_id).unwrap_or_else(|| panic!("DAG {dag_id} not found"));
    let current_node_index = to_node_index(dag_info.node_id);

    if dag.is_source_node(current_node_index) {
        // Source: this wake always releases a brand-new job, so always
        // compute and record a fresh deadline for it.
        let relative_deadline_ms = get_dag_sink_relative_deadline_ms(dag_id);
        let absolute_deadline = wake_time + relative_deadline_ms;
        dag.set_job_absolute_deadline(job_index, absolute_deadline);
        absolute_deadline
    } else if let Some(absolute_deadline) = dag.get_job_absolute_deadline(job_index) {
        absolute_deadline
    } else {
        // A miss here has two distinct, identical-looking causes:
        //
        // 1. Startup transient (harmless): every task — including every
        //    non-source DAG node — gets one initial wake right after
        //    `spawn_dag` creates it, before any real message has ever
        //    arrived (see `task::inner_spawn`'s `task.wake()`), and
        //    non-source nodes are spawned *before* the source (`spawn_dag`
        //    spawns the source last, precisely so it never publishes before
        //    its subscribers exist). That poll immediately re-suspends on
        //    `recv` without consuming anything, so `next_job_index` never
        //    advances past this — it can only ever land within the ring's
        //    first lap (`job_index <= DAG_JOB_TABLE_SIZE`).
        // 2. Ring wraparound (actionable): this node is running further
        //    behind the source's releases than the fixed-size ring
        //    (`DAG_JOB_TABLE_SIZE`) can track — its D/T ratio is wider than
        //    provisioned — and the entry was overwritten. Only possible once
        //    `job_index` has lapped the ring at least once.
        //
        // Only the second case is worth an operator's attention.
        if job_index > DAG_JOB_TABLE_SIZE as u64 {
            log::warn!(
                "DAG {dag_id} job {job_index}: absolute deadline not found in the job table \
                 (DAG_JOB_TABLE_SIZE={DAG_JOB_TABLE_SIZE} may be too small for this DAG's D/T \
                 ratio); falling back to wake_time + relative_deadline"
            );
        }
        wake_time + get_dag_sink_relative_deadline_ms(dag_id)
    }
}

pub async fn finish_create_dags(dags: &[Arc<Dag>]) -> Result<(), Vec<DagError>> {
    match validate_all_rules(dags) {
        Ok(()) => {
            for dag in dags {
                spawn_dag(dag).await;
            }
            Ok(())
        }
        Err(errors) => {
            for dag in dags {
                remove_dag(dag.id);
            }
            Err(errors)
        }
    }
}

fn remove_dag(id: u32) {
    let mut dags_node = MCSNode::new();
    let mut dags = DAGS.lock(&mut dags_node);
    dags.remove(id);

    let mut pending_node = MCSNode::new();
    let mut pending_tasks = PENDING_TASKS.lock(&mut pending_node);
    pending_tasks.remove(&id);

    let mut source_pending_node = MCSNode::new();
    let mut source_pending_tasks = SOURCE_PENDING_TASKS.lock(&mut source_pending_node);
    source_pending_tasks.remove(&id);
}

fn validate_all_rules(dags: &[Arc<Dag>]) -> Result<(), Vec<DagError>> {
    let mut errors: Vec<DagError> = Vec::new();

    for dag in dags {
        // Skip DAG validation if an arity mismatch is found, as it's the root cause of potential subsequent errors.
        if let Err(arg_errors) = check_for_arity_mismatches(dag.id) {
            errors.extend(arg_errors);
        } else if let Err(dag_validation_error) = validate_dag(dag) {
            errors.push(dag_validation_error);
        } else if let Err(pubsub_duplicate_errors) = check_for_pubsub_duplicates(dag.id) {
            errors.extend(pubsub_duplicate_errors);
        } else if let Err(duplicate_error) = validate_single_publisher_per_topic(dag) {
            errors.extend(duplicate_error);
        }
    }

    if !errors.is_empty() {
        return Err(errors);
    }

    validate_dag_topic_conflicts()?;

    Ok(())
}

fn validate_dag_topic_conflicts() -> Result<(), Vec<DagError>> {
    let mut topic_to_dags: BTreeMap<Cow<'static, str>, Vec<u32>> = BTreeMap::new();

    {
        let mut node = MCSNode::new();
        let dags = DAGS.lock(&mut node);

        for dag in dags.id_to_dag.values() {
            let mut node = MCSNode::new();
            let graph = dag.graph.lock(&mut node);
            for edge_ref in graph.edge_references() {
                topic_to_dags
                    .entry(edge_ref.weight().topic_name.clone())
                    .or_default()
                    .push(dag.id);
            }
        }
    }

    let errors: Vec<_> = topic_to_dags
        .into_iter()
        .filter_map(|(topic, mut dag_ids)| {
            dag_ids.sort_unstable();
            dag_ids.dedup();

            if dag_ids.len() > 1 {
                Some(DagError::InterDagTopicConflict(topic, dag_ids))
            } else {
                None
            }
        })
        .collect();

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

fn validate_dag(dag: &Dag) -> Result<(), DagError> {
    let dag_id = dag.id;
    assert!(
        get_dag(dag_id).is_some(),
        "Invariant Violation: DAG with id {dag_id} must exist, but was not found.",
    );

    let mut pending_node = MCSNode::new();
    if PENDING_TASKS.lock(&mut pending_node).get(&dag_id).is_none() {
        return Err(DagError::MissingPendingTasks(dag_id));
    }

    {
        let mut graph_node = MCSNode::new();
        let graph = dag.graph.lock(&mut graph_node);
        if connected_components(&*graph) != 1 {
            return Err(DagError::NotWeaklyConnected(dag_id));
        }
        if is_cyclic_directed(&*graph) {
            return Err(DagError::ContainsCycle(dag_id));
        }
    }

    if dag.get_source_nodes().len() > 1 {
        return Err(DagError::MultipleSourceNodes(dag_id));
    }
    if dag.get_sink_nodes().len() > 1 {
        return Err(DagError::MultipleSinkNodes(dag_id));
    }

    validate_edge_connect(dag)?;
    Ok(())
}

/// NOTE: On the architecture for this arity validation.
///
/// Ideally, this validation would be performed at an earlier stage, such as inside
/// the `impl_tuple_to_pub_sub` macro in `pubsub.rs`.
///
/// However, that approach would perform the check after the reactor is already spawned.
/// This would limit error handling to just stopping the affected reactor, and implementing
/// a full cleanup of all related DAG data and other reactors would be overly complex.
///
/// Therefore, we adopted the current architecture: errors are first recorded
/// to a `static` variable, and then collected and reported in a batch by this function.
fn check_for_arity_mismatches(dag_id: u32) -> Result<(), Vec<DagError>> {
    let mut node = MCSNode::new();

    let errors: Vec<_> = {
        let subscribe_errors = MISMATCH_SUBSCRIBE_NODES
            .lock(&mut node)
            .remove(&dag_id)
            .unwrap_or_default()
            .into_iter()
            .map(|node_id| DagError::SubscribeArityMismatch(dag_id, node_id));

        let publish_errors = MISMATCH_PUBLISH_NODES
            .lock(&mut node)
            .remove(&dag_id)
            .unwrap_or_default()
            .into_iter()
            .map(|node_id| DagError::PublishArityMismatch(dag_id, node_id));

        subscribe_errors.chain(publish_errors).collect()
    };

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

/// Ideally, this validation would be performed at an earlier stage, such as inside
/// the `add_node_with_topic_edges()`.
///
/// However, that approach would force the caller to handle a Result on every
/// single node addition, making the API cumbersome, and implementing
/// a full cleanup of all related DAG data and other reactors would be overly complex.
///
/// Therefore, we adopted the current architecture: errors are first recorded
/// to a `static` variable, and then collected and reported in a batch by this function.
fn check_for_pubsub_duplicates(dag_id: u32) -> Result<(), Vec<DagError>> {
    let mut node = MCSNode::new();

    let errors: Vec<_> = {
        let subscribe_errors = DUPLICATE_SUBSCRIBE_NODES
            .lock(&mut node)
            .remove(&dag_id)
            .unwrap_or_default()
            .into_iter()
            .map(|node_id| DagError::DuplicateSubscribe(dag_id, node_id));

        let publish_errors = DUPLICATE_PUBLISH_NODES
            .lock(&mut node)
            .remove(&dag_id)
            .unwrap_or_default()
            .into_iter()
            .map(|node_id| DagError::DuplicatePublish(dag_id, node_id));

        subscribe_errors.chain(publish_errors).collect()
    };

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

/// This validation prevents issues caused by the following misconfigurations:
/// - Message Loss: A published topic is not subscribed to by any reactor.
/// - Indefinite Wait: A subscribed topic has no corresponding publisher.
fn validate_edge_connect(dag: &Dag) -> Result<(), DagError> {
    type DagErrorFn = fn(u32, usize) -> DagError;

    let mut node = MCSNode::new();
    let graph = dag.graph.lock(&mut node);
    for node_idx in graph.node_indices() {
        let node_info = graph.node_weight(node_idx).unwrap();
        for direction in [Direction::Incoming, Direction::Outgoing] {
            let (expect_count, error) = match direction {
                Direction::Incoming => (
                    node_info.subscribe_topic_names.len(),
                    DagError::NoPublisherFound as DagErrorFn,
                ),
                Direction::Outgoing => (
                    node_info.publish_topic_names.len(),
                    DagError::NoSubscriberFound as DagErrorFn,
                ),
            };

            let actual_count = graph.neighbors_directed(node_idx, direction).count();
            if actual_count < expect_count {
                return Err(error(dag.id, node_idx.index()));
            }
        }
    }

    Ok(())
}

/// To simplify the current system design and management,
/// we have adopted a rule that only one publisher is allowed per topic.
///
/// Note:
/// This restriction may need to be relaxed in the future to support use cases
/// where multiple publishers need to publish to a single, common topic.
fn validate_single_publisher_per_topic(dag: &Dag) -> Result<(), Vec<DagError>> {
    let mut node = MCSNode::new();
    let graph = dag.graph.lock(&mut node);

    let publisher_counts_by_topic = graph.edge_references().fold(
        BTreeMap::<Cow<'static, str>, usize>::new(),
        |mut acc, edge| {
            *acc.entry(edge.weight().topic_name.clone()).or_insert(0) += 1;
            acc
        },
    );

    let errors: Vec<_> = publisher_counts_by_topic
        .into_iter()
        .filter(|(_, count)| *count > 1)
        .map(|(topic, _)| DagError::TopicHasMultiplePublishers(dag.id, topic))
        .collect();

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

async fn spawn_dag(dag: &Arc<Dag>) {
    let dag_id = dag.id;
    let pending_tasks = {
        let mut node = MCSNode::new();
        let mut pending_tasks = PENDING_TASKS.lock(&mut node);
        pending_tasks.remove(&dag_id).unwrap()
    };

    for task in pending_tasks {
        let task_id = (task.spawn)().await;
        let mut node = MCSNode::new();
        let mut graph = dag.graph.lock(&mut node);
        if let Some(node_info) = graph.node_weight_mut(task.node_idx) {
            node_info.task_id = task_id;
        }
    }

    // To prevent message drops, source reactor is spawned after all subsequent reactors have been spawned.
    // If the source reactor is not spawned last, it may publish a message before subsequent reactors are ready to receive it.
    let source_pending_task = {
        let mut node = MCSNode::new();
        let mut source_pending_tasks = SOURCE_PENDING_TASKS.lock(&mut node);
        source_pending_tasks.remove(&dag_id).unwrap()
    };

    let task_id = (source_pending_task.spawn)().await;
    let mut node = MCSNode::new();
    let mut graph = dag.graph.lock(&mut node);
    if let Some(node_info) = graph.node_weight_mut(source_pending_task.node_idx) {
        node_info.task_id = task_id;
    }
}

async fn spawn_reactor<F, Args, Ret>(
    reactor_name: Cow<'static, str>,
    f: F,
    subscribe_topic_names: Vec<Cow<'static, str>>,
    publish_topic_names: Vec<Cow<'static, str>>,
    sched_type: SchedulerType,
    dag_info: DagInfo,
) -> u32
where
    F: Fn(
            <Args::Subscribers as MultipleReceiver>::Item,
        ) -> <Ret::Publishers as MultipleSender>::Item
        + Send
        + 'static,
    Args: VectorToSubscribers,
    Ret: VectorToPublishers,
    Ret::Publishers: Send,
    Args::Subscribers: Send,
{
    // A separate handle to the same counter as `dag_info.period_index`: the
    // closure below only touches this field, so capturing the Arc directly
    // (rather than the whole `dag_info`) leaves `dag_info` itself intact for
    // the `spawn_with_dag_info` call after the closure is built.
    #[cfg(feature = "period-index-propagation")]
    let period_index_cell = dag_info.period_index.clone();

    let future = async move {
        let publishers = <Ret as VectorToPublishers>::create_publishers(
            publish_topic_names,
            Attribute::default(),
        );

        let subscribers: <Args as VectorToSubscribers>::Subscribers =
            Args::create_subscribers(subscribe_topic_names, Attribute::default());

        loop {
            #[cfg(feature = "period-index-propagation")]
            {
                let (args, period_index): (
                    <<Args as VectorToSubscribers>::Subscribers as MultipleReceiver>::Item,
                    u32,
                ) = subscribers.recv_all_with_period_index().await;
                period_index_cell.store(period_index, core::sync::atomic::Ordering::Relaxed);

                // [end] pubsub communication latency
                let end = awkernel_lib::time::Time::now().uptime().as_nanos() as u64;
                record_subscribe_timestamp(
                    period_index as usize,
                    end,
                    1,
                    dag_info.dag_id,
                    dag_info.node_id,
                );

                let results = f(args);
                publishers
                    .send_all_with_period_index(
                        results,
                        1,
                        period_index as usize,
                        dag_info.dag_id,
                        dag_info.node_id,
                    )
                    .await;
            }

            #[cfg(not(feature = "period-index-propagation"))]
            {
                let args: <<Args as VectorToSubscribers>::Subscribers as MultipleReceiver>::Item =
                    subscribers.recv_all().await;

                let results = f(args);
                publishers.send_all(results).await;
            }
        }
    };

    crate::task::spawn_with_dag_info(reactor_name, future, sched_type, dag_info)
}

async fn spawn_periodic_reactor<F, Ret>(
    reactor_name: Cow<'static, str>,
    f: F,
    publish_topic_names: Vec<Cow<'static, str>>,
    sched_type: SchedulerType,
    period: Duration,
    _release_measure: Option<MeasureF>,
    dag_info: DagInfo,
) -> u32
where
    F: Fn() -> <Ret::Publishers as MultipleSender>::Item + Send + 'static,
    Ret: VectorToPublishers,
    Ret::Publishers: Send,
{
    #[cfg(feature = "perf")]
    let (release_measure, periodic_measure) = {
        if let Some(measure) = _release_measure {
            (measure.clone(), measure)
        } else {
            unreachable!("Measure function must be provided for performance tracking.");
        }
    };

    // See `spawn_reactor` for why this is cloned out separately instead of
    // referenced via `dag_info` from inside the closure.
    #[cfg(feature = "period-index-propagation")]
    let period_index_cell = dag_info.period_index.clone();

    // TODO(sykwer): Improve mechanisms to more closely align performance behavior with the DAG scheduling model.
    let future = async move {
        let publishers = <Ret as VectorToPublishers>::create_publishers(
            publish_topic_names,
            Attribute::default(),
        );

        let mut interval = interval(period, dag_info.dag_id);
        // Consume the first tick here to start the loop's main body without an initial delay.
        interval.tick().await;

        loop {
            #[cfg(feature = "period-index-propagation")]
            {
                let index = get_period_index(dag_info.dag_id) as usize;
                period_index_cell.store(index as u32, core::sync::atomic::Ordering::Relaxed);
                if index != 0 {
                    // [start] cycle deviation index >= 1
                    let release_time = awkernel_lib::time::Time::now().uptime().as_nanos() as u64;
                    update_cycle_start_timestamp(index, release_time, dag_info.dag_id);
                }
                let results = f();
                publishers
                    .send_all_with_period_index(
                        results,
                        0,
                        index,
                        dag_info.dag_id,
                        dag_info.node_id,
                    )
                    .await;
                increment_period_index(dag_info.dag_id);
            }

            #[cfg(not(feature = "period-index-propagation"))]
            {
                let results = f();
                publishers.send_all(results).await;
            }

            #[cfg(feature = "perf")]
            periodic_measure();

            interval.tick().await;
        }
    };

    let task_id = crate::task::spawn_with_dag_info(reactor_name, future, sched_type, dag_info);

    #[cfg(feature = "perf")]
    release_measure();

    task_id
}

async fn spawn_sink_reactor<F, Args>(
    reactor_name: Cow<'static, str>,
    f: F,
    subscribe_topic_names: Vec<Cow<'static, str>>,
    sched_type: SchedulerType,
    dag_info: DagInfo,
) -> u32
where
    F: Fn(<Args::Subscribers as MultipleReceiver>::Item) + Send + 'static,
    Args: VectorToSubscribers,
    Args::Subscribers: Send,
{
    // See `spawn_reactor` for why this is cloned out separately instead of
    // referenced via `dag_info` from inside the closure.
    #[cfg(feature = "period-index-propagation")]
    let period_index_cell = dag_info.period_index.clone();

    let future = async move {
        let subscribers: <Args as VectorToSubscribers>::Subscribers =
            Args::create_subscribers(subscribe_topic_names, Attribute::default());

        loop {
            #[cfg(feature = "period-index-propagation")]
            {
                let (args, period_index): (<Args::Subscribers as MultipleReceiver>::Item, u32) =
                    subscribers.recv_all_with_period_index().await;
                period_index_cell.store(period_index, core::sync::atomic::Ordering::Relaxed);

                // [end] pubsub communication latency
                let end = awkernel_lib::time::Time::now().uptime().as_nanos() as u64;
                record_subscribe_timestamp(
                    period_index as usize,
                    end,
                    2,
                    dag_info.dag_id,
                    dag_info.node_id,
                );

                let timenow = awkernel_lib::time::Time::now().uptime().as_nanos() as u64;
                if period_index != 0 {
                    update_cycle_end_timestamp(period_index as usize, timenow, dag_info.dag_id);
                }

                f(args);
            }

            #[cfg(not(feature = "period-index-propagation"))]
            {
                let args: <Args::Subscribers as MultipleReceiver>::Item =
                    subscribers.recv_all().await;
                f(args);
            }
        }
    };

    crate::task::spawn_with_dag_info(reactor_name, future, sched_type, dag_info)
}
