//! DP-WRAP's core computation (Funk, Levin, Sadowski, Pye, Brandt, "DP-FAIR:
//! a unifying theory for optimal hard real-time multiprocessor scheduling",
//! Real-Time Systems 2011, Section 4.2): McNaughton's wrap-around algorithm,
//! used here as DAG-Fluid's real dispatch mechanism within one Deadline
//! Partition (DP) -- the piece the Phase 0/1/2 placeholder dispatch (plain
//! `GEDF` over `dag_sched::resource`'s shared pool) stands in for. See
//! `dag_sched::dp_partition`'s own module doc for how DP boundaries are
//! detected; this module is what should run once a boundary fires, not yet
//! wired into live dispatch (see [`wrap_blocks`]'s own doc for what is and
//! isn't implemented so far).
//!
//! # The algorithm, transcribed from the primary source (Section 4.2)
//!
//! "To schedule jobs in σj, make a 'block' of length δi for each Ti, and
//! line these blocks up along a number line (in any order), starting at
//! zero. Their total length will be no more than m. Split this stack of
//! blocks into length 1 chunks at 1, 2, ..., m − 1, and assign each chunk
//! to its own processor. Each length 1 chunk of tasks represents the
//! scheduling of tasks on the respective processor; tasks which are sliced
//! in two migrate between their two processors [...] To find the actual
//! timing points of context switches [...] within any σj, multiply each
//! length 1 segment by Lj."
//!
//! For DAG-Fluid, each "task" `Ti` in the quote above is one currently-
//! active segment of one admitted DAG-Fluid DAG, and `δi` is that
//! segment's own execution rate `θ_i,j` (`dag_fluid::SegmentSchedule::rate`,
//! computed in Phase 1) rather than a whole periodic task's density -- the
//! DP-FAIR paper's `δi = ei/min(pi,Di)` and DAG-Fluid's `θ_i,j` play the
//! same structural role (a fraction of one processor a unit of work is
//! entitled to for the duration of one scheduling interval), which is why
//! the same wrap-around construction applies directly.
//!
//! # WCET note
//! By explicit user direction (this is a measurement probe for the paper's
//! theory-vs-reality discussion, not a WCET-proven dispatch mechanism --
//! see `dag_sched::dp_partition`'s own module doc for the same scope
//! decision), this module does not carry a proven WCET bound. It stays
//! allocation-bounded and panic-free regardless: [`wrap_blocks`] allocates
//! exactly `m` `Vec`s up front (`(0..m).map(...)`, no further reallocation
//! beyond ordinary `Vec::push` growth bounded by `densities.len()`), and
//! every arithmetic path is on plain `f64` (no indexing that can panic: the
//! `proc >= m` case is checked and simply stops emitting further blocks
//! rather than indexing out of bounds -- reachable only if the *caller*
//! violates the `Σdensities <= m` precondition, at which point silently
//! dropping the remainder here is the same "defensive, not expected"
//! posture `dag_fluid::heavy_capacity_and_deadline` documents for its own
//! analogous precondition).

use alloc::vec::Vec;

/// McNaughton's wrap-around algorithm. `densities` are `(BlockId, density)`
/// pairs in the order to lay them along the number line (Funk et al.'s "in
/// any order" -- see this module's own doc for what `density` means for a
/// DAG-Fluid segment); `Σdensity <= m` is the caller's precondition (the
/// same shared-pool admission check `dag_sched::resource::reserve_dagfluid_capacity`
/// already enforces).
///
/// Returns one `Vec` per processor (`result.len() == m`, `result[p]` is
/// processor `p`'s own dispatch order), each an ordered list of `(index
/// into densities, start, end)`. `start`/`end` are offsets *within that
/// processor's own unit-length chunk* (`0.0..=1.0`), not yet multiplied by
/// the DP's own length `Lj` -- the paper's own "except for this last
/// multiplication, all calculations can be done once as a preprocessing
/// step" (Sect. 4.2); the caller scales by `Lj` (see this module's own
/// tests for a worked example of both steps together).
pub fn wrap_blocks(densities: &[(usize, f64)], m: usize) -> Vec<Vec<(usize, f64, f64)>> {
    let mut result: Vec<Vec<(usize, f64, f64)>> = (0..m).map(|_| Vec::new()).collect();
    let mut pos = 0.0_f64;

    for &(id, density) in densities {
        let mut remaining = density;
        while remaining > 0.0 {
            let proc = pos as usize;
            if proc >= m {
                // Caller violated the `Σdensity <= m` precondition -- see
                // this module's own doc for why this is a defensive stop,
                // not a panic.
                break;
            }
            let chunk_offset = pos - proc as f64;
            let space_in_chunk = 1.0 - chunk_offset;
            let take = remaining.min(space_in_chunk);
            result[proc].push((id, chunk_offset, chunk_offset + take));
            remaining -= take;
            pos += take;
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    /// Funk et al.'s own worked example (Fig. 7): seven tasks with
    /// densities `[0.3, 0.5, 0.5, 0.6, 0.5, 0.4, 0.2]` (summing to
    /// exactly `3 = m`) on 3 processors, lined up in that order. Hand
    /// re-derived from the paper's own algorithm description (Sect. 4.2)
    /// independently of the figure's pixels, then cross-checked against
    /// the figure's own textual claims: "Task 3 migrates from Processor 2
    /// to Processor 1" and "Task 5 migrates from Processor 3 to Processor
    /// 2" (1-indexed in the paper; tasks 3 and 5 are indices 2 and 4
    /// here) -- i.e. each migrating task runs at the *end* of its first
    /// processor's chunk and the *start* of its second, matching "tasks
    /// which migrate are run at the beginning of the slice on one
    /// processor, and at the end on the other" (Sect. 4.2).
    #[test]
    fn test_wrap_blocks_matches_paper_worked_example() {
        let densities: Vec<(usize, f64)> = vec![
            (0, 0.3),
            (1, 0.5),
            (2, 0.5),
            (3, 0.6),
            (4, 0.5),
            (5, 0.4),
            (6, 0.2),
        ];
        let result = wrap_blocks(&densities, 3);
        assert_eq!(result.len(), 3);

        // `f64` accumulation (`pos += take`) leaves the usual last-bit
        // rounding noise (e.g. `0.30000000000000004`), so compare with a
        // tolerance rather than `assert_eq!` on the tuples directly.
        fn assert_close(actual: &[(usize, f64, f64)], expected: &[(usize, f64, f64)]) {
            assert_eq!(actual.len(), expected.len());
            for (&(aid, astart, aend), &(eid, estart, eend)) in actual.iter().zip(expected) {
                assert_eq!(aid, eid);
                assert!((astart - estart).abs() < 1e-9, "{astart} vs {estart}");
                assert!((aend - eend).abs() < 1e-9, "{aend} vs {eend}");
            }
        }

        assert_close(&result[0], &[(0, 0.0, 0.3), (1, 0.3, 0.8), (2, 0.8, 1.0)]);
        assert_close(&result[1], &[(2, 0.0, 0.3), (3, 0.3, 0.9), (4, 0.9, 1.0)]);
        assert_close(&result[2], &[(4, 0.0, 0.4), (5, 0.4, 0.8), (6, 0.8, 1.0)]);

        // Every block's total assigned time (summed across whichever
        // processor(s) it landed on) equals its own density -- Sect. 4.2's
        // own invariant ("their total length will be no more than m").
        let mut totals = [0.0_f64; 7];
        for proc in &result {
            for &(id, start, end) in proc {
                totals[id] += end - start;
            }
        }
        for (id, &(_, expected_density)) in densities.iter().enumerate() {
            assert!((totals[id] - expected_density).abs() < 1e-9, "block {id}");
        }
    }

    /// A block that fits entirely within one processor's chunk (no
    /// migration) is a single `(id, 0.0, density)` entry there and
    /// nowhere else.
    #[test]
    fn test_wrap_blocks_single_task_no_migration() {
        let densities = vec![(0, 0.5)];
        let result = wrap_blocks(&densities, 2);
        assert_eq!(result[0], vec![(0, 0.0, 0.5)]);
        assert_eq!(result[1], vec![]);
    }

    /// Scaling step (the paper's own "multiply each length 1 segment by
    /// Lj"): with a DP of length `Lj = 10`, processor 0's chunk from the
    /// worked example above becomes absolute-within-DP offsets
    /// `[0, 3), [3, 8), [8, 10)`.
    #[test]
    fn test_scaling_by_slice_length() {
        let densities: Vec<(usize, f64)> = vec![(0, 0.3), (1, 0.5), (2, 0.5)];
        let result = wrap_blocks(&densities, 1);
        let l_j = 10.0_f64;
        let scaled: Vec<(usize, f64, f64)> = result[0]
            .iter()
            .map(|&(id, s, e)| (id, s * l_j, e * l_j))
            .collect();
        assert_eq!(scaled, vec![(0, 0.0, 3.0), (1, 3.0, 8.0), (2, 8.0, 10.0)]);
    }
}
