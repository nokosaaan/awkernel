//! Scheduler-agnostic DAG-level aggregation.
//!
//! Computes per-DAG statistics (total WCET volume and critical-path length)
//! from the fully-known node/edge topology that `parse_yaml::DagData` already
//! holds before any reactor is registered. This does not depend on which
//! `SchedulerType` the DAG will eventually use (ClusteredEDF, GEDF,
//! PrioritizedFIFO, or a future Federated scheduler); it is shared groundwork
//! for all of them.

use crate::parse_yaml::{DagData, NodeData};

use alloc::collections::{BTreeMap, VecDeque};

/// DAG-level aggregates derived from per-node WCET (`execution_time`).
pub(crate) struct DagAggregateStats {
    /// `C`: total WCET volume, i.e. the sum of every node's `execution_time`.
    pub(crate) volume: u64,
    /// `L`: critical-path length, i.e. the WCET sum along the longest
    /// source-to-sink path.
    pub(crate) critical_path: u64,
}

/// Compute [`DagAggregateStats`] for `dag_data` via a topological-order DP:
/// `dp[node] = execution_time[node] + max(dp[pred] for pred in in_links)`,
/// and `critical_path = max(dp[node])` over all nodes.
pub(crate) fn compute_dag_stats(dag_data: &DagData) -> DagAggregateStats {
    let nodes = dag_data.get_nodes();

    let volume: u64 = nodes.iter().map(NodeData::get_execution_time).sum();

    let node_by_id: BTreeMap<u32, &NodeData> =
        nodes.iter().map(|node| (node.get_id(), node)).collect();

    let mut in_degree: BTreeMap<u32, usize> = nodes
        .iter()
        .map(|node| (node.get_id(), node.get_in_links().len()))
        .collect();

    let mut queue: VecDeque<u32> = in_degree
        .iter()
        .filter(|&(_, &degree)| degree == 0)
        .map(|(&id, _)| id)
        .collect();

    let mut dp: BTreeMap<u32, u64> = BTreeMap::new();
    let mut critical_path = 0;

    while let Some(id) = queue.pop_front() {
        let Some(node) = node_by_id.get(&id) else {
            continue;
        };

        let pred_max = node
            .get_in_links()
            .iter()
            .filter_map(|pred_id| dp.get(pred_id).copied())
            .max()
            .unwrap_or(0);
        let finish = pred_max + node.get_execution_time();
        dp.insert(id, finish);
        critical_path = critical_path.max(finish);

        for out_id in node.get_out_links() {
            if let Some(degree) = in_degree.get_mut(out_id) {
                *degree = degree.saturating_sub(1);
                if *degree == 0 {
                    queue.push_back(*out_id);
                }
            }
        }
    }

    DagAggregateStats {
        volume,
        critical_path,
    }
}

/// Reverse (sink-to-source) topological DP computing every node's *laxity*
/// for the "static Laxity-Based" scheduling baseline: `laxity(sink) = D -
/// execution_time(sink)`; `laxity(v) = min(laxity(u) for u in out_links(v))
/// - execution_time(v)` for every other node. Mirrors [`compute_dag_stats`]
/// exactly (same topological-DP shape, `in_links`/`out_links` and `min`/`max`
/// swapped) -- deliberately omits communication delay between nodes, the
/// same "don't model communication cost" convention `compute_dag_stats`'s
/// own critical path already follows, and every admission policy in this
/// crate shares.
///
/// This baseline has no published primary source of its own -- literature
/// "Laxity-Based Scheduling" (e.g. Qamhieh et al. 2012, Suzuki et al.'s
/// HLBS) is a fundamentally *dynamic* quantity (laxity shrinks as a job's
/// *remaining* execution time is consumed at runtime), and neither matches
/// this formula. This is a self-derived, offline, WCET-only DP -- positioned
/// specifically as a structural counterpart to He et al. 2019 (IEEE TPDS)'s
/// own offline batch node-priority DP, for baseline comparison purposes
/// only, not as a reproduction of any specific paper's algorithm.
///
/// Returns `None` if any node's laxity would go negative -- this only
/// happens when `relative_deadline <= critical_path`, i.e. the DAG is
/// already unconditionally infeasible under *any* scheduling policy (not
/// specific to this baseline); mirrors
/// [`awkernel_async_lib::dag_sched::metrics::DagMetrics::min_dedicated_cores`]'s
/// own `None` case for the same condition.
pub(crate) fn compute_node_laxity(
    dag_data: &DagData,
    relative_deadline: u64,
) -> Option<BTreeMap<u32, u64>> {
    let nodes = dag_data.get_nodes();

    let node_by_id: BTreeMap<u32, &NodeData> =
        nodes.iter().map(|node| (node.get_id(), node)).collect();

    let mut out_degree: BTreeMap<u32, usize> = nodes
        .iter()
        .map(|node| (node.get_id(), node.get_out_links().len()))
        .collect();

    let mut queue: VecDeque<u32> = out_degree
        .iter()
        .filter(|&(_, &degree)| degree == 0)
        .map(|(&id, _)| id)
        .collect();

    let mut laxity: BTreeMap<u32, u64> = BTreeMap::new();

    while let Some(id) = queue.pop_front() {
        let Some(node) = node_by_id.get(&id) else {
            continue;
        };

        let succ_min = node
            .get_out_links()
            .iter()
            .filter_map(|succ_id| laxity.get(succ_id).copied())
            .min();
        // No successors (the sink) -> the full relative_deadline is
        // available; otherwise the tightest successor's own laxity.
        let slack = succ_min.unwrap_or(relative_deadline);
        let this_laxity = slack.checked_sub(node.get_execution_time())?;
        laxity.insert(id, this_laxity);

        for in_id in node.get_in_links() {
            if let Some(degree) = out_degree.get_mut(in_id) {
                *degree = degree.saturating_sub(1);
                if *degree == 0 {
                    queue.push_back(*in_id);
                }
            }
        }
    }

    Some(laxity)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse_yaml::parse_dags;

    #[test]
    fn test_compute_dag_stats_chain() {
        // 0 --10--> 1 --20--> 2 (linear chain, single path)
        let dag_file = "links:
  - source: 0
    target: 1
  - source: 1
    target: 2
nodes:
  - execution_time: 10
    id: 0
    period: 50
  - execution_time: 20
    id: 1
  - end_to_end_deadline: 40
    execution_time: 5
    id: 2
";
        let dags = parse_dags(&[dag_file]).unwrap();
        let stats = compute_dag_stats(&dags[0]);
        assert_eq!(stats.volume, 35);
        assert_eq!(stats.critical_path, 35);
    }

    #[test]
    fn test_compute_dag_stats_diamond() {
        // 0 fans out to 1 (heavy, 30) and 2 (light, 5); both join at 3.
        // Critical path must follow the heavier branch (0 -> 1 -> 3), not
        // simply sum every node.
        let dag_file = "links:
  - source: 0
    target: 1
  - source: 0
    target: 2
  - source: 1
    target: 3
  - source: 2
    target: 3
nodes:
  - execution_time: 10
    id: 0
    period: 50
  - execution_time: 30
    id: 1
  - execution_time: 5
    id: 2
  - end_to_end_deadline: 100
    execution_time: 5
    id: 3
";
        let dags = parse_dags(&[dag_file]).unwrap();
        let stats = compute_dag_stats(&dags[0]);
        assert_eq!(stats.volume, 50); // 10 + 30 + 5 + 5
        assert_eq!(stats.critical_path, 45); // 10 + 30 + 5, not the 2-fanned sum
    }

    #[test]
    fn test_compute_node_laxity_matches_worked_example_chain() {
        // The A -> B -> C portion of the project's own worked example
        // (Notion "2026-08-28-DAGスケジューリング実装とV-Fed調査": original
        // slide DAG was A->B->C, D->E->C, D->F -- two sinks, C and F, each
        // with its own deadline; adapted here to the single-sink chain
        // Awkernel's own DAG model requires, per the user's own confirmed
        // scope in that same page ("現状は単一source/sinkで想定していると
        // の回答")). w_C=10, D=80 -> laxity(C)=70; w_B=20 ->
        // laxity(B)=50; w_A=20 -> laxity(A)=30 -- exactly the numbers
        // recorded as hand-verified in that page.
        let dag_file = "links:
  - source: 0
    target: 1
  - source: 1
    target: 2
nodes:
  - execution_time: 20
    id: 0
    period: 1000
  - execution_time: 20
    id: 1
  - end_to_end_deadline: 80
    execution_time: 10
    id: 2
";
        let dags = parse_dags(&[dag_file]).unwrap();
        let laxity = compute_node_laxity(&dags[0], 80).unwrap();
        assert_eq!(laxity[&2], 70); // C (sink)
        assert_eq!(laxity[&1], 50); // B
        assert_eq!(laxity[&0], 30); // A
    }

    #[test]
    fn test_compute_node_laxity_diamond_takes_min_successor() {
        // Same diamond as test_compute_dag_stats_diamond: 0 fans out to 1
        // (heavy, 30) and 2 (light, 5); both join at sink 3 (deadline 100,
        // execution_time 5). laxity(3) = 100 - 5 = 95. laxity(1) = 95 - 30
        // = 65. laxity(2) = 95 - 5 = 90. laxity(0) must take the *tighter*
        // (min) of its two successors' laxity, not their max: min(65, 90)
        // - 10 = 55.
        let dag_file = "links:
  - source: 0
    target: 1
  - source: 0
    target: 2
  - source: 1
    target: 3
  - source: 2
    target: 3
nodes:
  - execution_time: 10
    id: 0
    period: 50
  - execution_time: 30
    id: 1
  - execution_time: 5
    id: 2
  - end_to_end_deadline: 100
    execution_time: 5
    id: 3
";
        let dags = parse_dags(&[dag_file]).unwrap();
        let laxity = compute_node_laxity(&dags[0], 100).unwrap();
        assert_eq!(laxity[&3], 95);
        assert_eq!(laxity[&1], 65);
        assert_eq!(laxity[&2], 90);
        assert_eq!(laxity[&0], 55);
    }

    #[test]
    fn test_compute_node_laxity_none_when_infeasible() {
        // relative_deadline (5) <= critical_path (35, from the chain test
        // above): unconditionally infeasible under any policy.
        let dag_file = "links:
  - source: 0
    target: 1
  - source: 1
    target: 2
nodes:
  - execution_time: 10
    id: 0
    period: 50
  - execution_time: 20
    id: 1
  - end_to_end_deadline: 40
    execution_time: 5
    id: 2
";
        let dags = parse_dags(&[dag_file]).unwrap();
        assert!(compute_node_laxity(&dags[0], 5).is_none());
    }
}
