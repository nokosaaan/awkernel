#!/usr/bin/env python3
"""DAG-structure generator matching the TPDS 2023 V-Fed paper's evaluation
(Jiang et al., "Scheduling Parallel Real-Time Tasks on Virtual Processors",
Sec. 7): a Python port of Melani et al.'s generator, the one the paper cites
as [21] (https://github.com/mive93/dag-scheduling,
src/DAGTask/DAGTask_Melani.cpp: expandTaskSeriesParallel, assignWCET,
makeItDag, then DAGTask::transitiveReduction).

The paper's own settings are the defaults: p_par = 0.8 (so p_term = 0.2, no
conditional branches), n_par = 8, maximal recursion depth 3, p_add = 0.1,
vertex WCET uniform in [1, 100].

One deliberate difference from the C++ code: its intRandMaxMin(a, b) is
`rand() % (b - a) + a`, i.e. uniform in [a, b-1], so the code actually draws
branches in [2, n_par - 1] and WCETs in [1, 99]. The paper text states
[2, n_par] and [1, 100]; this port follows the paper text.

Only the *structure* and vertex WCETs are generated here. Each DAG is written
in rd_gen_to_dags' YAML format with a placeholder period/deadline
(ceil(L / 0.3)) on its source/sink; the evaluation harness
(applications/rd_gen_to_dags/examples/paper_setting_comparison.rs) redraws
D and T per trial exactly as the paper does.

Each DAG is also drawn to dag_<i>.pdf with graphviz `dot` (same labelling
as RD-Gen's own exporter: "[i]" + "C: <WCET>", the source node as a box, the
sink node in bold), so the generated structures can be inspected by eye;
--no-figures skips this.

Usage: melani_dag_generator.py <out_dir> <num_dags> [--seed S] [--no-figures]
"""

import argparse
import math
import random
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


class Gen:
    def __init__(self, rng, p_par, n_par, depth, p_add, c_min, c_max):
        self.rng = rng
        self.p_par = p_par
        self.n_par = n_par
        self.depth = depth
        self.p_add = p_add
        self.c_min = c_min
        self.c_max = c_max

    def generate(self):
        # Vertex i: depth[i], succ[i] (list, creation order as in the C++ V).
        self.node_depth = []
        self.succ = []

        # expandTaskSeriesParallel(nullptr, nullptr, recDepth, ...): source
        # (V[0], depth = recDepth) and sink (V[1], depth = -recDepth), then
        # the root is always a parallel expansion (probSCond = 0 for DAGs).
        so = self._new(self.depth)
        si = self._new(-self.depth)
        self._expand(so, si, self.depth - 1, self._branches())

        n = len(self.succ)
        wcet = [self.rng.randint(self.c_min, self.c_max) for _ in range(n)]  # assignWCET
        self._make_it_dag()
        self._transitive_reduction()
        return wcet, self.succ

    def _new(self, depth):
        self.node_depth.append(depth)
        self.succ.append([])
        return len(self.succ) - 1

    def _branches(self):
        return self.rng.randint(2, self.n_par)

    def _expand(self, source, sink, depth, num_branches):
        for _ in range(num_branches):
            # state = TERMINAL unless depth != 0, then drawn from
            # {COND: 0, PAR: p_par, TERM: 1 - p_par}.
            parallel = depth != 0 and self.rng.random() < self.p_par
            if not parallel:
                v = self._new(depth)
                self.succ[source].append(v)
                self.succ[v].append(sink)
            else:
                v1 = self._new(depth)
                self.succ[source].append(v1)
                v2 = self._new(-depth)
                self.succ[v2].append(sink)
                self._expand(v1, v2, depth - 1, self._branches())

    def _make_it_dag(self):
        # For every ordered pair (v, w) in creation order: add v -> w with
        # probability p_add if depth(v) > depth(w) and w is not already a
        # descendant of v. depth strictly decreases along every edge, so the
        # graph stays acyclic. Descendant sets are kept as bitmasks and
        # updated on every insertion, so each check sees the current graph
        # exactly as the C++ recursive isSuccessor() does.
        n = len(self.succ)
        desc = [0] * n
        order = self._topo_order()
        for v in reversed(order):
            m = 0
            for s in self.succ[v]:
                m |= (1 << s) | desc[s]
            desc[v] = m
        anc = [0] * n
        for v in range(n):
            for u in range(n):
                if desc[u] >> v & 1:
                    anc[v] |= 1 << u

        for v in range(n):
            for w in range(n):
                if (self.node_depth[v] > self.node_depth[w]
                        and not (desc[v] >> w & 1)
                        and self.rng.random() < self.p_add):
                    self.succ[v].append(w)
                    add = (1 << w) | desc[w]
                    targets = anc[v] | (1 << v)
                    x = targets
                    while x:
                        low = x & -x
                        u = low.bit_length() - 1
                        desc[u] |= add
                        x ^= low
                    y = add
                    while y:
                        low = y & -y
                        d = low.bit_length() - 1
                        anc[d] |= targets
                        y ^= low

    def _topo_order(self):
        n = len(self.succ)
        indeg = [0] * n
        for v in range(n):
            for s in self.succ[v]:
                indeg[s] += 1
        order, stack = [], [v for v in range(n) if indeg[v] == 0]
        while stack:
            v = stack.pop()
            order.append(v)
            for s in self.succ[v]:
                indeg[s] -= 1
                if indeg[s] == 0:
                    stack.append(s)
        return order

    def _transitive_reduction(self):
        # Drop v -> s when s is also a descendant of another successor of v.
        n = len(self.succ)
        desc = [0] * n
        for v in reversed(self._topo_order()):
            m = 0
            for s in self.succ[v]:
                m |= (1 << s) | desc[s]
            desc[v] = m
        for v in range(n):
            succs = list(dict.fromkeys(self.succ[v]))
            via_others = 0
            for s in succs:
                via_others |= desc[s]
            self.succ[v] = [s for s in succs if not (via_others >> s & 1)]


def critical_path(wcet, succ):
    n = len(wcet)
    indeg = [0] * n
    for v in range(n):
        for s in succ[v]:
            indeg[s] += 1
    finish = [0] * n
    stack = [v for v in range(n) if indeg[v] == 0]
    start = [0] * n
    while stack:
        v = stack.pop()
        finish[v] = start[v] + wcet[v]
        for s in succ[v]:
            start[s] = max(start[s], finish[v])
            indeg[s] -= 1
            if indeg[s] == 0:
                stack.append(s)
    return max(finish)


def to_yaml(wcet, succ, placeholder):
    lines = ["links:"]
    for v, ss in enumerate(succ):
        for s in ss:
            lines += [f"  - source: {v}", f"    target: {s}"]
    lines.append("nodes:")
    for v, c in enumerate(wcet):
        if v == 0:  # source (V[0])
            lines += [f"  - execution_time: {c}", f"    id: {v}", f"    period: {placeholder}"]
        elif v == 1:  # sink (V[1])
            lines += [f"  - end_to_end_deadline: {placeholder}", f"    execution_time: {c}", f"    id: {v}"]
        else:
            lines += [f"  - execution_time: {c}", f"    id: {v}"]
    return "\n".join(lines) + "\n"


def to_dot(wcet, succ):
    lines = [
        "digraph G {",
        '  graph [label="source period / sink deadline: assigned per trial by the evaluation harness", '
        'labelloc=b, fontsize=10];',
        "  node [fontsize=10];",
    ]
    for v, c in enumerate(wcet):
        attrs = [f'label="[{v}]\\nC: {c}"']
        if v == 0:
            attrs.append("shape=box")
        elif v == 1:
            attrs.append("style=bold")
        lines.append(f"  {v} [{', '.join(attrs)}];")
    for v, ss in enumerate(succ):
        for s in ss:
            lines.append(f"  {v} -> {s};")
    lines.append("}")
    return "\n".join(lines) + "\n"


def render_pdf(dot_source, pdf_path):
    subprocess.run(["dot", "-Tpdf", "-o", str(pdf_path)], input=dot_source, text=True, check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir")
    ap.add_argument("num_dags", type=int)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--p-par", type=float, default=0.8)
    ap.add_argument("--n-par", type=int, default=8)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--p-add", type=float, default=0.1)
    ap.add_argument("--no-figures", action="store_true", help="skip the per-DAG dag_<i>.pdf drawings")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    gen = Gen(rng, args.p_par, args.n_par, args.depth, args.p_add, 1, 100)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    drawings = []
    for i in range(args.num_dags):
        wcet, succ = gen.generate()
        length = critical_path(wcet, succ)
        placeholder = math.ceil(length / 0.3)
        (out / f"dag_{i}.yaml").write_text(to_yaml(wcet, succ, placeholder))
        if not args.no_figures:
            drawings.append((to_dot(wcet, succ), out / f"dag_{i}.pdf"))

    # Drawing is independent of generation (the RNG is only used above), so
    # it runs in parallel without affecting the generated DAGs.
    with ThreadPoolExecutor() as ex:
        list(ex.map(lambda d: render_pdf(*d), drawings))


if __name__ == "__main__":
    main()
