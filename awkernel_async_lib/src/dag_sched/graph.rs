//! A DAG's precedence structure as the admission tests in
//! [`super::policy`] consume it (Federated's list scheduling, SFS's
//! segmentation, DAG-Fluid's segment decomposition): per-node WCETs and
//! edges only, independent of how an application declares its DAGs.

use alloc::vec::Vec;

/// A DAG's precedence structure: node WCETs and edges, nodes addressed
/// `0..len()`. Callers map their own node ids to these indices (e.g.
/// `rd_gen_to_dags` uses ascending node-id order); the index order is also
/// the priority list of Federated's Graham list scheduling (the papers
/// leave the list order unspecified -- any fixed order gives a valid LS
/// schedule).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DagGraph {
    wcet: Vec<u64>,
    successors: Vec<Vec<usize>>,
    in_degree: Vec<usize>,
}

impl DagGraph {
    /// `wcet[v]` is node `v`'s WCET; each `(u, v)` in `edges` means `u`
    /// must finish before `v` starts. `None` if an edge names a node
    /// outside `0..wcet.len()`.
    pub fn new(wcet: Vec<u64>, edges: &[(usize, usize)]) -> Option<Self> {
        let n = wcet.len();
        let mut successors = alloc::vec![Vec::new(); n];
        let mut in_degree = alloc::vec![0usize; n];
        for &(u, v) in edges {
            if u >= n || v >= n {
                return None;
            }
            successors[u].push(v);
            in_degree[v] += 1;
        }
        Some(Self {
            wcet,
            successors,
            in_degree,
        })
    }

    pub fn len(&self) -> usize {
        self.wcet.len()
    }

    pub fn is_empty(&self) -> bool {
        self.wcet.is_empty()
    }

    /// Node `v`'s WCET.
    pub fn wcet(&self, v: usize) -> u64 {
        self.wcet[v]
    }

    /// Node `v`'s immediate successors, in edge-insertion order.
    pub fn successors(&self, v: usize) -> &[usize] {
        &self.successors[v]
    }

    /// Number of immediate predecessors of node `v`.
    pub fn in_degree(&self, v: usize) -> usize {
        self.in_degree[v]
    }
}
