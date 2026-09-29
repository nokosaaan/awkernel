//! Segmented-Flattened-and-Split (SFS) scheduling -- S. Lendve, K. Bletsas,
//! P. F. Souto, "Resource-efficient scheduling of parallel DAG tasks on
//! identical multiprocessors", Journal of Systems Architecture 176 (2026)
//! 103775. Offline admission (the SFS-G variant: Graham's bound as the
//! work-conserving fall-back) only; the runtime (gang EDF per cluster,
//! table-driven threads, migration between clusters) is not implemented.
//!
//! Correspondence with the paper:
//! - [`segments`] = Eq. (1): `S_k` holds the nodes whose longest hop
//!   distance from the dummy source is `k`.
//! - `flatten_segment` = Algorithm 1 (McNaughton's wrap-around on `m'`
//!   processors, length `max(ceil(W/m'), C_max)`); `flatten` concatenates
//!   the segments.
//! - [`feasibly_max_flatten`] = Algorithm 2 (smallest `m'` from
//!   `ceil(W / min(D, T))` upwards; FAILURE if `sum_k C_max(S_k) > D`).
//! - `cluster_size_requirements` = Sec. 4.2 / 5.1: the smaller cluster of
//!   flattening and the Graham fall-back (Eqs. (2), (3)); on a tie
//!   flattening, unless its length exceeds the Graham makespan bound.
//! - Every cluster is a set of gang tasks with parallelism equal to the
//!   cluster size, hence (Lemma 3) exactly a uniprocessor EDF system:
//!   `edf_schedulable` is the exact processor-demand test (Eq. (5)),
//!   evaluated with QPA (Zhang & Burns). `sensitivity` = Algorithm 6.
//! - [`plan`] = Algorithms 3-5: first pass (non-increasing `D`; heavy
//!   `W > D` tasks to a fresh cluster, light ones First-Fit to bins) and
//!   second pass (split the skipped tasks with C=D pieces), with the
//!   target-selection heuristic of Sec. 5.1.
//!
//! Deviations from the printed pseudocode (each an evident artifact that
//! would otherwise loop forever or stop early; see the code comments):
//! 1. Algorithm 4, line 20 `break` would end the whole first pass after a
//!    light task is placed by First-Fit; it is read as "go to the next
//!    DAG task".
//! 2. Algorithm 6 with C integer division never terminates when
//!    `C_max = C_min + 1` and `C_min` is schedulable; the same search is
//!    done with a terminating binary search (identical result: the largest
//!    schedulable `C` in `[0, C_max]`, the test being monotone in `C`).
//! 3. Algorithm 5 with `piece_C == 0` assigns nothing and would pick the
//!    same target again forever; such a target is excluded until the next
//!    piece is placed.
//!
//! Interpretations where the paper leaves a detail open:
//! - Second-pass targets are the existing clusters, the existing bins and
//!   every processor left empty after the first pass, the latter each as a
//!   single-processor target ("an existing cluster or an individual
//!   processor that is not part of any cluster", Sec. 5.1). No new
//!   multi-processor cluster is formed in the second pass.
//! - Nodes inside a segment are flattened in node-index order (the order
//!   of Fig. 2); zero-WCET nodes are skipped.
//! - Ties in the first-pass order are broken by task index; ties in target
//!   selection (size, then gross normalised density) by target creation
//!   order.

use super::federated::DagGraph;
use alloc::{vec, vec::Vec};

/// One sporadic DAG task with a constrained deadline (`D <= T`).
#[derive(Debug, Clone, Copy)]
pub struct SfsTask<'a> {
    pub graph: &'a DagGraph,
    pub period: u64,
    pub deadline: u64,
}

/// A task (or task piece) as the uniprocessor EDF view of a cluster sees it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Piece {
    /// Index of the DAG task in the input.
    pub dag: usize,
    /// Gang WCET (schedule length) of the piece.
    pub wcet: u64,
    pub deadline: u64,
    pub period: u64,
    /// A non-final piece of a split task (`C = D`).
    pub zero_laxity: bool,
}

/// An assignment target: a cluster (`processors > 1`) or a single processor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Target {
    pub processors: usize,
    pub pieces: Vec<Piece>,
}

impl Target {
    /// Definition 1 (gross normalised density), as a fraction `num / den`
    /// is not needed for comparison; f64 suffices for tie-breaking only.
    fn gross_normalised_density(&self) -> f64 {
        self.pieces
            .iter()
            .map(|p| p.wcet as f64 / p.deadline as f64)
            .sum::<f64>()
            / self.processors as f64
    }

    fn edf_ok_with(&self, extra: (u64, u64, u64)) -> bool {
        let mut tasks: Vec<(u64, u64, u64)> = self
            .pieces
            .iter()
            .map(|p| (p.wcet, p.deadline, p.period))
            .collect();
        tasks.push(extra);
        edf_schedulable(&tasks)
    }
}

/// A successful SFS assignment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SfsPlan {
    pub targets: Vec<Target>,
    /// Number of DAG tasks split in the second pass.
    pub split_dags: usize,
}

/// Whether `tasks` is schedulable under SFS-G on `m` processors.
pub fn is_schedulable(tasks: &[SfsTask<'_>], m: u16) -> bool {
    plan(tasks, m).is_some()
}

// ---------------------------------------------------------------------------
// Segmentation and flattening (Sec. 4.1, 4.2)
// ---------------------------------------------------------------------------

/// Eq. (1): the nodes of each segment `S_1, S_2, ...`, each in index order.
/// `None` if the graph has a cycle.
pub fn segments(graph: &DagGraph) -> Option<Vec<Vec<usize>>> {
    let n = graph.len();
    let mut remaining: Vec<usize> = (0..n).map(|v| graph.in_degree(v)).collect();
    let mut hop = vec![1usize; n]; // sources are 1 hop from the dummy source
    let mut stack: Vec<usize> = (0..n).filter(|&v| remaining[v] == 0).collect();
    let mut seen = 0;
    while let Some(v) = stack.pop() {
        seen += 1;
        for &s in graph.successors(v) {
            hop[s] = hop[s].max(hop[v] + 1);
            remaining[s] -= 1;
            if remaining[s] == 0 {
                stack.push(s);
            }
        }
    }
    if seen != n {
        return None;
    }
    let depth = hop.iter().copied().max().unwrap_or(0);
    let mut segs = vec![Vec::new(); depth];
    for (v, &h) in hop.iter().enumerate() {
        segs[h - 1].push(v);
    }
    Some(segs)
}

/// One node's execution interval in a flattened schedule (absolute time
/// from the start of the schedule; the processor is irrelevant here).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Interval {
    node: usize,
    start: u64,
    end: u64,
}

/// Algorithm 1 for one segment starting at `base`: returns its length and
/// appends its intervals. `rem[v]` is node `v`'s (remaining) WCET.
fn flatten_segment(
    nodes: &[usize],
    rem: &[u64],
    m: usize,
    base: u64,
    out: &mut Vec<Interval>,
) -> u64 {
    let work: u64 = nodes.iter().map(|&v| rem[v]).sum();
    if work == 0 {
        return 0;
    }
    let cmax = nodes.iter().map(|&v| rem[v]).max().unwrap_or(0);
    let len = work.div_ceil(m as u64).max(cmax); // lines 8-11
    let mut offset = 0u64;
    for &v in nodes {
        let c = rem[v];
        if c == 0 {
            continue;
        }
        let start = offset;
        let end = (offset + c - 1) % len + 1; // line 18
        if end > start {
            out.push(Interval {
                node: v,
                start: base + start,
                end: base + end,
            });
            offset = if end == len { 0 } else { end }; // lines 22-26
        } else {
            // Lines 29-32: the tail on the next processor runs first in time.
            out.push(Interval {
                node: v,
                start: base + start,
                end: base + len,
            });
            out.push(Interval {
                node: v,
                start: base,
                end: base + end,
            });
            offset = end;
        }
    }
    len
}

/// The concatenated flattened schedule of all segments on `m` processors.
fn flatten(segs: &[Vec<usize>], rem: &[u64], m: usize) -> (u64, Vec<Interval>) {
    let mut out = Vec::new();
    let mut length = 0u64;
    for seg in segs {
        length += flatten_segment(seg, rem, m, length, &mut out);
    }
    (length, out)
}

/// Length of the flattened schedule on `m` processors, without intervals.
fn flattened_length(segs: &[Vec<usize>], rem: &[u64], m: usize) -> u64 {
    segs.iter()
        .map(|seg| {
            let work: u64 = seg.iter().map(|&v| rem[v]).sum();
            let cmax = seg.iter().map(|&v| rem[v]).max().unwrap_or(0);
            work.div_ceil(m as u64).max(cmax)
        })
        .sum()
}

/// Algorithm 2: the smallest `m'` whose flattened schedule fits in
/// `deadline` (with `min(D, T) = deadline` under constrained deadlines),
/// and that schedule's length. `None` = FAILURE (lines 2-6).
pub fn feasibly_max_flatten(
    segs: &[Vec<usize>],
    rem: &[u64],
    deadline: u64,
) -> Option<(usize, u64)> {
    let lb: u64 = segs
        .iter()
        .map(|seg| seg.iter().map(|&v| rem[v]).max().unwrap_or(0))
        .sum();
    if lb > deadline {
        return None;
    }
    let work: u64 = rem.iter().sum();
    let widest = segs.iter().map(Vec::len).max().unwrap_or(1).max(1);
    let mut m = (work.div_ceil(deadline.max(1)) as usize).max(1); // line 8
    loop {
        let len = flattened_length(segs, rem, m);
        // With `m >= widest` every segment is as short as its longest node,
        // i.e. the length is `lb <= deadline`, so the loop terminates.
        if len <= deadline || m >= widest {
            return Some((m, len));
        }
        m += 1;
    }
}

fn critical_path(graph: &DagGraph) -> u64 {
    let n = graph.len();
    let mut remaining: Vec<usize> = (0..n).map(|v| graph.in_degree(v)).collect();
    let mut start = vec![0u64; n];
    let mut stack: Vec<usize> = (0..n).filter(|&v| remaining[v] == 0).collect();
    let mut best = 0;
    while let Some(v) = stack.pop() {
        let finish = start[v] + graph.wcet(v);
        best = best.max(finish);
        for &s in graph.successors(v) {
            start[s] = start[s].max(finish);
            remaining[s] -= 1;
            if remaining[s] == 0 {
                stack.push(s);
            }
        }
    }
    best
}

/// Eq. (3) and Eq. (2): the Graham cluster size and makespan bound on it,
/// or `None` if no cluster size suffices (`D <= L < W`, or `L > D`).
fn graham_cluster(work: u64, length: u64, deadline: u64) -> Option<(usize, u64)> {
    if length > deadline {
        return None;
    }
    if work == length {
        return Some((1, length));
    }
    if deadline == length {
        return None;
    }
    let m = (work - length).div_ceil(deadline - length).max(1);
    Some((m as usize, length + (work - length).div_ceil(m)))
}

/// `cluster_size_requirements` (Algorithm 4, line 8): `(m', gang WCET)`.
fn cluster_size_requirements(prep: &Prepared<'_>) -> Option<(usize, u64)> {
    let rem: Vec<u64> = (0..prep.graph.len()).map(|v| prep.graph.wcet(v)).collect();
    let flat = feasibly_max_flatten(&prep.segs, &rem, prep.deadline);
    let graham = graham_cluster(prep.work, prep.length, prep.deadline);
    match (flat, graham) {
        (None, None) => None,
        (Some(f), None) => Some(f),
        (None, Some(g)) => Some(g),
        (Some(f), Some(g)) => {
            // Fewer processors wins; on a tie, flattening unless it is
            // longer than the fall-back's makespan bound.
            if f.0 < g.0 || (f.0 == g.0 && f.1 <= g.1) {
                Some(f)
            } else {
                Some(g)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Uniprocessor EDF (Lemma 3, Eq. (5)) and C=D sensitivity (Algorithm 6)
// ---------------------------------------------------------------------------

/// Busy-period iterations allowed before falling back to the `L_a` bound.
const MAX_BUSY_ITERATIONS: usize = 100_000;

/// Processor demand `h(t) = sum_i dbf(tau_i, t)` (Eq. (5)).
fn demand(tasks: &[(u64, u64, u64)], t: u64) -> u128 {
    tasks
        .iter()
        .filter(|&&(_, d, _)| d <= t)
        .map(|&(c, d, p)| (((t - d) / p) as u128 + 1) * c as u128)
        .sum()
}

/// The largest absolute deadline strictly below `x`, if any.
fn max_deadline_below(tasks: &[(u64, u64, u64)], x: u64) -> Option<u64> {
    tasks
        .iter()
        .filter(|&&(_, d, _)| d < x)
        .map(|&(_, d, p)| d + (x - d - 1) / p * p)
        .max()
}

/// Exact uniprocessor EDF test for constrained-deadline sporadic tasks
/// `(C, D, T)`, via QPA. Utilization is compared in f64; a set whose
/// utilization is within 1e-9 of 1 is only accepted if its synchronous
/// busy period converges within [`MAX_BUSY_ITERATIONS`] (otherwise it is
/// conservatively rejected).
pub fn edf_schedulable(tasks: &[(u64, u64, u64)]) -> bool {
    let tasks: Vec<(u64, u64, u64)> = tasks.iter().copied().filter(|&(c, _, _)| c > 0).collect();
    if tasks.is_empty() {
        return true;
    }
    if tasks.iter().any(|&(c, d, p)| c > d || d > p || d == 0) {
        return false;
    }
    let u: f64 = tasks.iter().map(|&(c, _, p)| c as f64 / p as f64).sum();
    if u > 1.0 + 1e-9 {
        return false;
    }
    // Synchronous busy period.
    let mut w: u128 = tasks.iter().map(|&(c, _, _)| c as u128).sum();
    let mut busy = None;
    for _ in 0..MAX_BUSY_ITERATIONS {
        let next: u128 = tasks
            .iter()
            .map(|&(c, _, p)| w.div_ceil(p as u128) * c as u128)
            .sum();
        if next == w {
            busy = Some(w);
            break;
        }
        w = next;
    }
    let bound = match busy {
        Some(b) => b.min(u64::MAX as u128) as u64,
        None if u < 1.0 - 1e-9 => {
            let dmax = tasks.iter().map(|&(_, d, _)| d).max().unwrap_or(0);
            let la = tasks
                .iter()
                .map(|&(c, d, p)| (p - d) as f64 * c as f64 / p as f64)
                .sum::<f64>()
                / (1.0 - u);
            // Manual ceil: `f64::ceil` needs `std`/`libm`. `la >= 0` since `D <= T`.
            let truncated = la as u64;
            let la_ceil = if (truncated as f64) < la {
                truncated + 1
            } else {
                truncated
            };
            dmax.max(la_ceil + 1)
        }
        None => return false,
    };
    let dmin = tasks.iter().map(|&(_, d, _)| d).min().unwrap_or(0);
    // QPA (Zhang & Burns 2009).
    let Some(mut t) = max_deadline_below(&tasks, bound) else {
        return true;
    };
    loop {
        let h = demand(&tasks, t);
        if h > t as u128 {
            return false;
        }
        if h <= dmin as u128 {
            return true;
        }
        if h < t as u128 {
            t = h as u64;
        } else {
            match max_deadline_below(&tasks, t) {
                Some(next) => t = next,
                None => return true,
            }
        }
    }
}

/// Algorithm 6: the largest `C` in `[0, c_max]` such that `target` plus a
/// zero-laxity task `(C, C, period)` is EDF-schedulable (deviation 2 in the
/// module doc).
fn sensitivity(target: &Target, c_max: u64, period: u64) -> u64 {
    let (mut lo, mut hi) = (0u64, c_max);
    while lo < hi {
        let mid = lo + (hi - lo).div_ceil(2);
        if target.edf_ok_with((mid, mid, period)) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    lo
}

// ---------------------------------------------------------------------------
// The complete algorithm (Algorithms 3-5)
// ---------------------------------------------------------------------------

struct Prepared<'a> {
    graph: &'a DagGraph,
    period: u64,
    deadline: u64,
    work: u64,
    length: u64,
    segs: Vec<Vec<usize>>,
}

/// Runs SFS-G on `m` processors. `None` = FAILURE (or a cyclic graph, or a
/// task with `D > T`).
pub fn plan(tasks: &[SfsTask<'_>], m: u16) -> Option<SfsPlan> {
    let preps: Vec<Prepared<'_>> = tasks
        .iter()
        .map(|t| {
            (t.deadline <= t.period && t.deadline > 0).then_some(())?;
            Some(Prepared {
                graph: t.graph,
                period: t.period,
                deadline: t.deadline,
                work: (0..t.graph.len()).map(|v| t.graph.wcet(v)).sum(),
                length: critical_path(t.graph),
                segs: segments(t.graph)?,
            })
        })
        .collect::<Option<_>>()?;

    // Algorithm 3, lines 2 and 17: non-increasing D (ties by index).
    let mut order: Vec<usize> = (0..tasks.len()).collect();
    order.sort_by(|&a, &b| preps[b].deadline.cmp(&preps[a].deadline).then(a.cmp(&b)));

    // First pass (Algorithm 4).
    let mut targets: Vec<Target> = Vec::new();
    let mut m_empty = m as usize;
    let mut unassigned = Vec::new();
    for &i in &order {
        let p = &preps[i];
        if p.work > p.deadline {
            // Heavy: a fresh cluster, if its size still fits.
            match cluster_size_requirements(p) {
                Some((size, len)) if size <= m_empty => {
                    targets.push(Target {
                        processors: size,
                        pieces: vec![Piece {
                            dag: i,
                            wcet: len,
                            deadline: p.deadline,
                            period: p.period,
                            zero_laxity: false,
                        }],
                    });
                    m_empty -= size;
                }
                _ => unassigned.push(i),
            }
        } else {
            // Light: First-Fit over the bins with the exact EDF test.
            let task = (p.work, p.deadline, p.period);
            let piece = Piece {
                dag: i,
                wcet: p.work,
                deadline: p.deadline,
                period: p.period,
                zero_laxity: false,
            };
            if let Some(bin) = targets
                .iter_mut()
                .find(|t| t.processors == 1 && t.edf_ok_with(task))
            {
                bin.pieces.push(piece);
                // Deviation 1: Algorithm 4 line 20 `break` -> next DAG task.
            } else if m_empty > 0 {
                targets.push(Target {
                    processors: 1,
                    pieces: vec![piece],
                });
                m_empty -= 1;
            } else {
                unassigned.push(i);
            }
        }
    }

    // Second pass (Algorithm 5). Processors left empty are single-processor
    // targets (see the module doc).
    for _ in 0..m_empty {
        targets.push(Target {
            processors: 1,
            pieces: Vec::new(),
        });
    }
    let split_dags = unassigned.len();
    for &i in &unassigned {
        let p = &preps[i];
        let mut rem: Vec<u64> = (0..p.graph.len()).map(|v| p.graph.wcet(v)).collect();
        let mut d = p.deadline;
        let mut split_len = 0u64;
        let mut used = vec![false; targets.len()];
        let mut excluded = vec![false; targets.len()];
        loop {
            // pick_assignment_target_for_piece_of (Sec. 5.1).
            let (need, _) = feasibly_max_flatten(&p.segs, &rem, d)?;
            let a = (0..targets.len())
                .filter(|&k| {
                    !used[k]
                        && !excluded[k]
                        && targets[k].processors >= need
                        && !targets[k]
                            .pieces
                            .iter()
                            .any(|x| x.zero_laxity && x.dag != i)
                })
                .min_by(|&x, &y| {
                    targets[x]
                        .processors
                        .cmp(&targets[y].processors)
                        .then(
                            targets[y]
                                .gross_normalised_density()
                                .total_cmp(&targets[x].gross_normalised_density()),
                        )
                        .then(x.cmp(&y))
                })?; // lines 37-38
            let (fs_len, fs) = flatten(&p.segs, &rem, targets[a].processors); // lines 39-42
            if targets[a].edf_ok_with((fs_len, d, p.period)) {
                // Lines 44-46.
                targets[a].pieces.push(Piece {
                    dag: i,
                    wcet: fs_len,
                    deadline: d,
                    period: p.period,
                    zero_laxity: false,
                });
                break;
            }
            let piece_c = sensitivity(&targets[a], fs_len, p.period); // line 49
            if piece_c == 0 {
                excluded[a] = true; // deviation 3
                continue;
            }
            split_len += piece_c;
            if split_len >= p.deadline {
                return None; // lines 52-53
            }
            targets[a].pieces.push(Piece {
                dag: i,
                wcet: piece_c,
                deadline: piece_c,
                period: p.period,
                zero_laxity: true,
            });
            used[a] = true;
            excluded.iter_mut().for_each(|e| *e = false);
            // Line 56: remove what the first `piece_c` time units executed.
            for iv in &fs {
                let done = iv.end.min(piece_c).saturating_sub(iv.start);
                rem[iv.node] -= done.min(rem[iv.node]);
            }
            d -= piece_c; // line 57
        }
    }

    Some(SfsPlan {
        targets,
        split_dags,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng};

    /// Fig. 1 of the paper (tau_1..tau_10 as nodes 0..9, D = 130).
    fn fig1() -> DagGraph {
        DagGraph::new(
            vec![10, 20, 30, 20, 20, 10, 20, 20, 20, 10],
            &[
                (0, 1),
                (0, 2),
                (0, 3),
                (1, 4),
                (1, 6),
                (2, 4),
                (2, 5),
                (3, 5),
                (4, 6),
                (4, 7),
                (5, 7),
                (5, 8),
                (6, 9),
                (7, 9),
                (8, 9),
            ],
        )
        .unwrap()
    }

    fn wcets(g: &DagGraph) -> Vec<u64> {
        (0..g.len()).map(|v| g.wcet(v)).collect()
    }

    #[test]
    fn test_segments_match_fig1() {
        let segs = segments(&fig1()).unwrap();
        assert_eq!(
            segs,
            vec![vec![0], vec![1, 2, 3], vec![4, 5], vec![6, 7, 8], vec![9]]
        );
    }

    #[test]
    fn test_flatten_fig2_makespan_105() {
        let g = fig1();
        let (len, ivs) = flatten(&segments(&g).unwrap(), &wcets(&g), 2);
        assert_eq!(len, 105);
        // Every node receives exactly its WCET.
        let mut got = vec![0u64; g.len()];
        for iv in &ivs {
            got[iv.node] += iv.end - iv.start;
        }
        assert_eq!(got, wcets(&g));
        // At most 2 nodes run at any time and no node runs twice at once.
        for t in 0..len {
            let running: Vec<usize> = ivs
                .iter()
                .filter(|iv| iv.start <= t && t < iv.end)
                .map(|iv| iv.node)
                .collect();
            assert!(running.len() <= 2, "t={t}: {running:?}");
            let mut dedup = running.clone();
            dedup.dedup();
            assert_eq!(dedup.len(), running.len());
        }
    }

    #[test]
    fn test_fig5_rump_after_60_flattens_to_35_on_3() {
        let g = fig1();
        let segs = segments(&g).unwrap();
        let mut rem = wcets(&g);
        let (_, ivs) = flatten(&segs, &rem, 2);
        for iv in &ivs {
            let done = iv.end.min(60).saturating_sub(iv.start);
            rem[iv.node] -= done.min(rem[iv.node]);
        }
        assert_eq!(rem, vec![0, 0, 0, 0, 5, 0, 20, 20, 20, 10]); // Fig. 5(c)
        assert_eq!(flatten(&segs, &rem, 3).0, 35); // Fig. 5(d)
    }

    #[test]
    fn test_fig3_prefers_graham_fallback() {
        // Two chains 1 -> 49 and 49 -> 1: flattening needs 49 + 49 = 98.
        let g = DagGraph::new(vec![1, 49, 49, 1], &[(0, 2), (1, 3)]).unwrap();
        let task = SfsTask {
            graph: &g,
            period: 80,
            deadline: 80,
        };
        let prep = Prepared {
            graph: &g,
            period: 80,
            deadline: 80,
            work: 100,
            length: critical_path(&g),
            segs: segments(&g).unwrap(),
        };
        assert_eq!(prep.length, 50);
        assert!(feasibly_max_flatten(&prep.segs, &wcets(&g), 80).is_none());
        assert_eq!(cluster_size_requirements(&prep), Some((2, 75))); // Eq. (2)
        assert!(is_schedulable(&[task], 2));
        assert!(!is_schedulable(&[task], 1));
    }

    #[test]
    fn test_edf_exact_test() {
        assert!(edf_schedulable(&[(1, 2, 2), (1, 2, 2)])); // U = 1
        assert!(!edf_schedulable(&[(1, 2, 2), (1, 2, 2), (1, 10, 10)]));
        // Constrained deadlines: U < 1 but dbf(2) = 3 > 2.
        assert!(!edf_schedulable(&[(2, 2, 10), (1, 2, 10)]));
        assert!(edf_schedulable(&[(2, 2, 10), (1, 3, 10)]));
        assert!(edf_schedulable(&[(3, 7, 20), (2, 4, 5), (2, 8, 10)]));
    }

    #[test]
    fn test_sensitivity_is_largest_feasible_budget() {
        let t = Target {
            processors: 1,
            pieces: vec![Piece {
                dag: 0,
                wcet: 4,
                deadline: 10,
                period: 10,
                zero_laxity: false,
            }],
        };
        let c = sensitivity(&t, 100, 10);
        assert!(t.edf_ok_with((c, c, 10)));
        assert!(!t.edf_ok_with((c + 1, c + 1, 10)));
        assert_eq!(c, 6);
    }

    /// FS-G (Definition 2's corresponding federated variant, for SFS-G):
    /// the same first pass with Graham cluster sizing only, and no second
    /// pass.
    fn fs_g(tasks: &[SfsTask<'_>], m: u16) -> bool {
        let mut order: Vec<usize> = (0..tasks.len()).collect();
        order.sort_by(|&a, &b| tasks[b].deadline.cmp(&tasks[a].deadline).then(a.cmp(&b)));
        let mut m_empty = m as usize;
        let mut bins: Vec<Target> = Vec::new();
        for &i in &order {
            let t = &tasks[i];
            let work: u64 = wcets(t.graph).iter().sum();
            if work > t.deadline {
                match graham_cluster(work, critical_path(t.graph), t.deadline) {
                    Some((size, _)) if size <= m_empty => m_empty -= size,
                    _ => return false,
                }
            } else {
                let task = (work, t.deadline, t.period);
                let piece = Piece {
                    dag: i,
                    wcet: work,
                    deadline: t.deadline,
                    period: t.period,
                    zero_laxity: false,
                };
                if let Some(b) = bins.iter_mut().find(|b| b.edf_ok_with(task)) {
                    b.pieces.push(piece);
                } else if m_empty > 0 {
                    bins.push(Target {
                        processors: 1,
                        pieces: vec![piece],
                    });
                    m_empty -= 1;
                } else {
                    return false;
                }
            }
        }
        true
    }

    fn random_dag(rng: &mut impl Rng) -> DagGraph {
        // Layered: 2..6 layers of 1..5 nodes, edges between consecutive
        // layers with probability 0.5 (every node keeps a predecessor).
        let layers: Vec<usize> = (0..rng.random_range(2..=6))
            .map(|_| rng.random_range(1..=5))
            .collect();
        let mut wcet = Vec::new();
        let mut edges = Vec::new();
        let mut prev: Vec<usize> = Vec::new();
        for &k in &layers {
            let cur: Vec<usize> = (0..k).map(|j| wcet.len() + j).collect();
            wcet.extend((0..k).map(|_| rng.random_range(1..=50u64)));
            for &v in &cur {
                if prev.is_empty() {
                    continue;
                }
                let mut any = false;
                for &u in &prev {
                    if rng.random_bool(0.5) {
                        edges.push((u, v));
                        any = true;
                    }
                }
                if !any {
                    edges.push((prev[rng.random_range(0..prev.len())], v));
                }
            }
            prev = cur;
        }
        DagGraph::new(wcet, &edges).unwrap()
    }

    #[test]
    fn test_sfs_g_dominates_fs_g_lemma4() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let (mut fs_ok, mut sfs_only) = (0, 0);
        for _ in 0..3000 {
            let graphs: Vec<DagGraph> = (0..rng.random_range(2..=8))
                .map(|_| random_dag(&mut rng))
                .collect();
            let tasks: Vec<SfsTask<'_>> = graphs
                .iter()
                .map(|g| {
                    let l = critical_path(g);
                    let w: u64 = wcets(g).iter().sum();
                    let d = rng.random_range(l..=w.max(l) * 2);
                    let t = rng.random_range(d..=d * 2);
                    SfsTask {
                        graph: g,
                        period: t,
                        deadline: d,
                    }
                })
                .collect();
            let m = rng.random_range(2..=8u16);
            let sfs = is_schedulable(&tasks, m);
            if fs_g(&tasks, m) {
                fs_ok += 1;
                assert!(sfs, "Lemma 4 violated");
            } else if sfs {
                sfs_only += 1;
            }
        }
        assert!(
            fs_ok > 100 && sfs_only > 0,
            "fs_ok={fs_ok} sfs_only={sfs_only}"
        );
    }

    #[test]
    fn test_split_pieces_respect_deadline_and_edf() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(11);
        let mut splits = 0;
        for _ in 0..2000 {
            let graphs: Vec<DagGraph> = (0..rng.random_range(3..=8))
                .map(|_| random_dag(&mut rng))
                .collect();
            let tasks: Vec<SfsTask<'_>> = graphs
                .iter()
                .map(|g| {
                    let l = critical_path(g);
                    let w: u64 = wcets(g).iter().sum();
                    let d = rng.random_range(l..=w.max(l) * 2);
                    SfsTask {
                        graph: g,
                        period: d,
                        deadline: d,
                    }
                })
                .collect();
            let Some(plan) = plan(&tasks, rng.random_range(2..=8)) else {
                continue;
            };
            splits += plan.split_dags;
            for t in &plan.targets {
                let v: Vec<(u64, u64, u64)> = t
                    .pieces
                    .iter()
                    .map(|p| (p.wcet, p.deadline, p.period))
                    .collect();
                assert!(edf_schedulable(&v));
            }
            // The pieces of each split DAG add up to within its deadline.
            for i in 0..tasks.len() {
                let pieces: Vec<&Piece> = plan
                    .targets
                    .iter()
                    .flat_map(|t| &t.pieces)
                    .filter(|p| p.dag == i)
                    .collect();
                assert!(!pieces.is_empty());
                if pieces.len() > 1 {
                    let zl: u64 = pieces
                        .iter()
                        .filter(|p| p.zero_laxity)
                        .map(|p| p.wcet)
                        .sum();
                    let last = pieces.iter().find(|p| !p.zero_laxity).unwrap();
                    assert_eq!(zl + last.deadline, tasks[i].deadline);
                }
            }
        }
        assert!(splits > 0);
    }
}
