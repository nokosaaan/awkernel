//! DAG-Fluid admission and job templates, shared by the kernel-side build
//! (`build_dag.rs`) and the host-side offline switch table generation
//! (`build.rs`, which includes this file with `#[path]`), so both decide
//! exactly the same: which DAGs are admitted, and with which segments.
//!
//! This file must compile in both crates: it only uses `alloc`, the YAML
//! model (`parse_yaml`), `dag_stats`, and `awkernel_async_lib`'s pure
//! DAG-Fluid functions. Each crate uses only part of it (the kernel build
//! never calls [`admit_offline`], and non-DAG-Fluid builds only
//! [`dag_graph`]), hence the module-wide `dead_code` allowance.
#![allow(dead_code)]

use crate::{
    dag_stats::compute_dag_stats,
    parse_yaml::{DagData, NodeData},
};
use alloc::{string::String, vec, vec::Vec};
use awkernel_async_lib::dag_sched::{
    fluid_table::OfflineDag,
    graph::DagGraph,
    job_table::SegmentTemplate,
    metrics::DagMetrics,
    policy::dag_fluid,
    precondition::{self, Policy},
    resource::UTILIZATION_SCALE,
};
use core::time::Duration;

/// `dag_data`'s node ids in ascending order: [`DagGraph`] index `v` of
/// [`dag_graph`]'s output is node `sorted_node_ids(dag_data)[v]`.
pub(crate) fn sorted_node_ids(dag_data: &DagData) -> Vec<u32> {
    let mut ids: Vec<u32> = dag_data.get_nodes().iter().map(|n| n.get_id()).collect();
    ids.sort_unstable();
    ids
}

/// One DAG's [`DagGraph`]: nodes in ascending node-id order (see
/// [`sorted_node_ids`]; also the list-scheduling priority list), WCET =
/// `execution_time`, one edge per `out_links` entry.
pub(crate) fn dag_graph(dag_data: &DagData) -> Result<DagGraph, String> {
    let ids = sorted_node_ids(dag_data);
    let index_of = |id: u32| ids.binary_search(&id).ok();

    let mut wcet = vec![0u64; ids.len()];
    let mut edges = Vec::new();
    for node in dag_data.get_nodes() {
        let from = index_of(node.get_id()).ok_or_else(|| String::from("duplicate node id"))?;
        wcet[from] = node.get_execution_time();
        for &to_id in node.get_out_links() {
            let to =
                index_of(to_id).ok_or_else(|| alloc::format!("link to unknown node id {to_id}"))?;
            edges.push((from, to));
        }
    }
    DagGraph::new(wcet, &edges).ok_or_else(|| String::from("DAG edge out of range"))
}

/// Why a DAG cannot be turned into a DAG-Fluid job template.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum FluidDagError {
    /// No source period or no sink deadline.
    MissingTiming,
    Graph(String),
    /// The admission test has no answer (degenerate decomposition).
    Infeasible,
    /// A time does not convert to a `Duration`.
    InvalidDuration,
}

/// One DAG prepared for DAG-Fluid: its required capacity and its job
/// template (the segments the planner lays out).
#[derive(Debug, Clone)]
pub(crate) struct FluidDag {
    pub(crate) metrics: DagMetrics,
    pub(crate) required: f64,
    pub(crate) period: Duration,
    pub(crate) virtual_deadline: Duration,
    pub(crate) segments: Vec<SegmentTemplate>,
    /// `τ_seq` (`C <= D*`): the whole DAG as one sequential thread of rate
    /// `C/D*`, one segment without a node list. Otherwise `τ_paral`.
    pub(crate) is_seq: bool,
}

/// Prepare `dag_data` for DAG-Fluid. `unit` converts a value in the YAML's
/// time unit (possibly fractional) to a `Duration`.
///
/// - `τ_paral` (`C > D*`): the segments of Algorithm 1 (deadlines, rates,
///   release offsets), with each segment's completion gate and active nodes
///   as node ids.
/// - `τ_seq` (`C <= D*`): one segment `[0, D*)` of rate `C/D*` and one
///   thread, without a node list (any node of the job may run in its slot),
///   gated by the sink(s). TC 2020/2022 Sect. 8 put it into DP-Wrap like any
///   other thread.
pub(crate) fn fluid_dag(
    dag_data: &DagData,
    unit: impl Fn(f64) -> Option<Duration>,
) -> Result<FluidDag, FluidDagError> {
    let stats = compute_dag_stats(dag_data);
    let period = dag_data
        .get_nodes()
        .iter()
        .find(|node| node.is_source())
        .and_then(NodeData::get_period)
        .ok_or(FluidDagError::MissingTiming)?;
    let relative_deadline = dag_data
        .get_nodes()
        .iter()
        .find(|node| node.is_sink())
        .and_then(NodeData::get_end_to_end_deadline)
        .ok_or(FluidDagError::MissingTiming)?;
    let metrics =
        DagMetrics::from_static(stats.volume, stats.critical_path, period, relative_deadline);

    let graph = dag_graph(dag_data).map_err(FluidDagError::Graph)?;
    let segments = dag_fluid::decompose_segments(&graph);
    let required = dag_fluid::required_capacity(
        stats.volume,
        period,
        stats.critical_path,
        relative_deadline,
        &segments,
    )
    .ok_or(FluidDagError::Infeasible)?;
    let d_star = dag_fluid::virtual_deadline(period, relative_deadline, stats.critical_path);

    let to_duration = |v: f64| unit(v).ok_or(FluidDagError::InvalidDuration);
    let node_ids = sorted_node_ids(dag_data);

    let (templates, is_seq) = if stats.volume > d_star {
        let schedule = dag_fluid::assign_segment_deadlines(&segments, stats.volume, d_star)
            .ok_or(FluidDagError::Infeasible)?;
        let offsets = dag_fluid::segment_release_offsets(&schedule);
        // Gates and active nodes come back as graph indices; templates keep
        // node ids.
        let gates: Vec<Vec<u32>> = dag_fluid::segment_completion_gates(&graph)
            .into_iter()
            .map(|gate| {
                gate.iter()
                    .filter_map(|&v| node_ids.get(v).copied())
                    .collect()
            })
            .collect();
        let active: Vec<Vec<u32>> = dag_fluid::segment_active_nodes(&graph)
            .into_iter()
            .map(|nodes| {
                nodes
                    .iter()
                    .filter_map(|&v| node_ids.get(v).copied())
                    .collect()
            })
            .collect();
        let mut templates = Vec::with_capacity(schedule.len());
        for (j, (seg, offset)) in schedule.iter().zip(offsets.iter()).enumerate() {
            templates.push(SegmentTemplate {
                offset: to_duration(*offset)?,
                relative_deadline: to_duration(seg.relative_deadline)?,
                concurrency: seg.concurrency,
                rate: seg.rate,
                gate_nodes: gates.get(j).cloned().unwrap_or_default(),
                active_nodes: active.get(j).cloned().unwrap_or_default(),
            });
        }
        (templates, false)
    } else {
        let sinks: Vec<u32> = dag_data
            .get_nodes()
            .iter()
            .filter(|node| node.is_sink())
            .map(NodeData::get_id)
            .collect();
        let template = SegmentTemplate {
            offset: Duration::ZERO,
            relative_deadline: to_duration(d_star as f64)?,
            concurrency: 1,
            rate: stats.volume as f64 / d_star as f64,
            gate_nodes: sinks,
            active_nodes: Vec::new(),
        };
        (alloc::vec![template], true)
    };

    Ok(FluidDag {
        metrics,
        required,
        period: to_duration(period as f64)?,
        virtual_deadline: to_duration(d_star as f64)?,
        segments: templates,
        is_seq,
    })
}

/// [`UTILIZATION_SCALE`]-scaled capacity a DAG reserves, exactly as
/// `resource::reserve_dagfluid_capacity` computes it.
pub(crate) fn scaled_capacity(required: f64) -> u64 {
    (required.max(0.0) * UTILIZATION_SCALE as f64) as u64
}

/// The offline counterpart of `build_dag`'s DAG-Fluid admission: admit
/// `dags` in order against `pool_cpus` shared cores, with the same
/// precondition, capacity test and rounding as the kernel. Returns the
/// indices of the admitted DAGs and their job templates.
pub(crate) fn admit_offline(
    dags: &[DagData],
    pool_cpus: usize,
    unit: impl Fn(f64) -> Option<Duration> + Copy,
) -> (Vec<usize>, Vec<OfflineDag>) {
    let capacity = (pool_cpus as u64).saturating_mul(UTILIZATION_SCALE);
    let mut committed: u64 = 0;
    let mut admitted = Vec::new();
    let mut offline = Vec::new();
    for (index, dag_data) in dags.iter().enumerate() {
        let Ok(dag) = fluid_dag(dag_data, unit) else {
            continue;
        };
        if precondition::check(Policy::DagFluid, index as u32, &dag.metrics).is_err() {
            continue;
        }
        let additional = scaled_capacity(dag.required);
        if committed.saturating_add(additional) > capacity {
            continue;
        }
        committed += additional;
        admitted.push(index);
        offline.push(OfflineDag {
            period: dag.period,
            virtual_deadline: dag.virtual_deadline,
            segments: dag.segments,
        });
    }
    (admitted, offline)
}
