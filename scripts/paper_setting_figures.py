#!/usr/bin/env python3
"""One-shot acceptance-ratio figures comparing Li et al. federated, Baruah
DATE 2015 federated, V-Fed (OURS1/OURS2) and DAG-Fluid on DAGs from the
V-Fed paper's generator (Melani et al.; scripts/melani_dag_generator.py).

Default: a FIXED core count m (--cores, default 14 = the real machine's
DAG pool), with each task set's utilization U_sigma = U_norm * m split over
its DAGs (Guan et al., TC 2022, Sec. 7.1; see paper_setting_comparison.rs).
Six single-figure PDFs -- three views x {constrained, implicit}:

  1. acceptance ratio vs U_norm
  2. acceptance ratio vs mean parallelism C/L at a fixed U_norm
     (pools generated with n_par = 2, 3, 4, 6, 8)
  3. acceptance ratio vs core count m at a fixed U_norm

--per-trial-m instead reproduces the V-Fed paper's own method
(m = ceil(U_sigma / U_norm) per task set, D ~ U[L, L/alpha]); page 3 then
sweeps alpha (the paper's Fig. 8(b)/9(b) axis) instead of m.

Files go to --out-dir, named after what they show, e.g.
  acceptance_vs_unorm_constrained_m14_N8_t1000.pdf
  acceptance_vs_parallelism_implicit_m14_unorm0.70_N8_t1000.pdf
  acceptance_vs_cores_constrained_unorm0.30_N8_t1000.pdf
  acceptance_vs_alpha_implicit_mPerTrial_unorm0.50_N8_t1000.pdf  (--per-trial-m)
plus data_<method>_N8_t<trials>.csv with every plotted number (the table
view of the figures).

Everything is driven from here: missing DAG pools are generated with
scripts/melani_dag_generator.py, the comparison harness
(applications/rd_gen_to_dags/examples/paper_setting_comparison.rs) is built,
and every (figure, mode, x) point runs as its own process in parallel.
Li et al. is implicit-deadline only, so it appears in the implicit panels only.

Usage:
  python3 scripts/paper_setting_figures.py [--cores M | --per-trial-m]
      [--out-dir DIR] [--trials N] [--pool-root DIR]
      [--fixed-unorm-constrained U] [--fixed-unorm-implicit U] [--jobs N]
"""

import argparse
import csv
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

AWKERNEL = Path(__file__).resolve().parent.parent
GENERATOR = AWKERNEL / "scripts" / "melani_dag_generator.py"
HARNESS = AWKERNEL / "target" / "release" / "examples" / "paper_setting_comparison"

MAIN_POOL = ("melani_vfed_pool", 2000, 8)           # (dir name, DAGs, n_par)
NPAR_POOLS = [(f"melani_npar{n}", 800, n) for n in (2, 3, 4, 6, 8)]
POOL_SEEDS = {"melani_vfed_pool": 0}
DEFAULT_SEED = 7

MODES = ("constrained", "implicit")
UNORM_RANGE = {"constrained": (0.05, 0.60, 0.05), "implicit": (0.05, 1.00, 0.05)}
ALPHAS = [round(0.1 * k, 1) for k in range(1, 10)]
CORE_SWEEP = [4, 8, 14, 16, 24, 32]
DAGS_PER_SET = 8  # N; fixed by the harness

# (column, legend label, color, marker). Fixed per entity, never by rank;
# palette validated with the dataviz skill's validator (light surface).
SERIES = [
    ("li", "Federated (Li et al. 2014)", "#2a78d6", "o"),
    ("date", "Federated (Baruah DATE 2015)", "#eb6834", "s"),
    ("ours1", "V-Fed OURS1", "#1baf7a", "^"),
    ("ours2", "V-Fed OURS2", "#eda100", "D"),
    ("fluid", "DAG-Fluid (Guan et al. 2022)", "#e87ba4", "v"),
    ("sfs", "SFS-G (Lendve et al. 2026)", "#008300", "P"),
]


def ensure_pool(pool_root, name, count, n_par):
    path = pool_root / name
    have = len(list(path.glob("dag_*.yaml"))) if path.is_dir() else 0
    if have >= count:
        return path
    seed = POOL_SEEDS.get(name, DEFAULT_SEED)
    print(f"generating {name} ({count} DAGs, n_par={n_par})", file=sys.stderr)
    subprocess.run(
        [sys.executable, str(GENERATOR), str(path), str(count), "--seed", str(seed), "--n-par", str(n_par)],
        check=True,
    )
    return path


def build_harness(any_pool):
    # build.rs embeds some DAG directory into the library for kernel builds;
    # the harness itself reads its pool at run time, so any valid pool works.
    env = {**os.environ, "RD_GEN_DAGS_DIR": str(any_pool)}
    subprocess.run(
        ["cargo", "build", "--release", "--example", "paper_setting_comparison",
         "-p", "rd_gen_to_dags", "--no-default-features", "--features", "std"],
        cwd=AWKERNEL, env=env, check=True,
    )


def run_point(pool, mode, u_norm, alpha, cores, trials):
    """One harness run at a single U_norm; returns (mean C/L, {col: pct or None}).
    `cores=None` means the per-trial-m method. A point the harness cannot
    generate (the pool's C/L caps can't reach U_norm * m) comes back empty."""
    cmd = [str(HARNESS), str(pool), mode, f"{u_norm}", f"{u_norm}", "1", str(trials), f"{alpha}"]
    if cores is not None:
        cmd += ["--cores", str(cores)]
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode != 0:
        print(f"skipped {pool.name} {mode} u_norm={u_norm} cores={cores}: {out.stderr.strip()[-200:]}",
              file=sys.stderr)
        return float("nan"), {}
    m = re.search(r"mean C/L = ([0-9.]+)", out.stderr)
    mean_cl = float(m.group(1)) if m else float("nan")
    lines = [l for l in out.stdout.strip().splitlines() if l]
    header, row = lines[0].split(","), lines[1].split(",")
    values = {h: (None if v == "-" else float(v)) for h, v in zip(header, row)}
    return mean_cl, values


def frange(lo, hi, step):
    n = int(round((hi - lo) / step))
    return [round(lo + i * step, 4) for i in range(n + 1)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=str(AWKERNEL / "log" / "paper_setting_figures"))
    ap.add_argument("--trials", type=int, default=1000)
    ap.add_argument("--pool-root", default=str(AWKERNEL.parent / "RD-Gen" / "test"))
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--cores", type=int, default=14,
                       help="fixed core count m (default 14, the real machine's DAG pool)")
    group.add_argument("--per-trial-m", action="store_true",
                       help="the V-Fed paper's m = ceil(U_sigma/U_norm) per task set instead")
    ap.add_argument("--fixed-unorm-constrained", type=float, default=None,
                    help="U_norm for pages 2-3 (default 0.3)")
    ap.add_argument("--fixed-unorm-implicit", type=float, default=None,
                    help="U_norm for pages 2-3 (default 0.7 with fixed m, 0.5 with --per-trial-m)")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 4)
    args = ap.parse_args()

    pool_root = Path(args.pool_root)
    main_pool = ensure_pool(pool_root, *MAIN_POOL)
    npar_pools = [ensure_pool(pool_root, *p) for p in NPAR_POOLS]
    build_harness(npar_pools[0])

    cores = None if args.per_trial_m else args.cores
    fixed = {
        "constrained": args.fixed_unorm_constrained if args.fixed_unorm_constrained is not None else 0.3,
        "implicit": args.fixed_unorm_implicit if args.fixed_unorm_implicit is not None
        else (0.5 if cores is None else 0.7),
    }
    unorm_range = dict(UNORM_RANGE)
    if cores is not None:
        unorm_range["constrained"] = (0.05, 1.00, 0.05)
    jobs = []  # (figure, mode, x_key, pool, u_norm, alpha, cores)
    for mode in MODES:
        for u in frange(*unorm_range[mode]):
            jobs.append(("unorm", mode, u, main_pool, u, 0.3, cores))
        for pool in npar_pools:
            jobs.append(("parallelism", mode, pool.name, pool, fixed[mode], 0.3, cores))
        if cores is None:
            for a in ALPHAS:
                jobs.append(("alpha", mode, a, main_pool, fixed[mode], a, None))
        else:
            for m in CORE_SWEEP:
                jobs.append(("cores", mode, m, main_pool, fixed[mode], 0.3, m))

    print(f"running {len(jobs)} points x {args.trials} trials on {args.jobs} workers", file=sys.stderr)
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        results = list(ex.map(lambda j: run_point(j[3], j[1], j[4], j[5], j[6], args.trials), jobs))

    rows = []
    for (fig, mode, key, pool, u, a, m), (mean_cl, vals) in zip(jobs, results):
        x = mean_cl if fig == "parallelism" else key
        rows.append({"figure": fig, "mode": mode, "x": x, "pool": pool.name, "u_norm": u,
                     "alpha": a, "cores": "per-trial" if m is None else m,
                     "mean_C_over_L": mean_cl, **{c: vals.get(c) for c, *_ in SERIES}})

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"data_{method_tag(cores)}_N{DAGS_PER_SET}_t{args.trials}.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    for path in plot(rows, out_dir, args.trials, fixed, cores):
        print(f"wrote {path}", file=sys.stderr)
    print(f"wrote {out_csv}", file=sys.stderr)


def method_tag(cores):
    return "mPerTrial" if cores is None else f"m{cores}"


def plot(rows, out_dir, trials, fixed, cores):
    """One PDF per (view, deadline mode); returns the written paths."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, grid = "#1f1f1e", "#6b6a63", "#e4e3dc"
    plt.rcParams.update({
        "font.size": 10, "axes.edgecolor": muted, "axes.labelcolor": ink,
        "xtick.color": muted, "ytick.color": muted, "text.color": ink,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    views = [
        ("unorm", "Acceptance ratio vs normalized utilization", "U_norm", False),
        ("parallelism", "Acceptance ratio vs DAG parallelism",
         "mean parallelism C/L of the pool (n_par = 2, 3, 4, 6, 8)", True),
    ]
    if cores is None:
        views.append(("alpha", "Acceptance ratio vs deadline slack",
                      "alpha  (D ~ U[L, L/alpha]; larger = tighter)", True))
    else:
        views.append(("cores", "Acceptance ratio vs core count", "cores m", True))

    def method_text(view):
        if cores is None:
            return "m = ceil(U_sigma / U_norm) per task set (V-Fed paper's method)"
        m_text = "m on the x-axis" if view == "cores" else f"m = {cores}"
        return f"fixed {m_text}, U_sigma = U_norm * m (UUniFast-Discard)"

    def file_name(view, mode):
        parts = [f"acceptance_vs_{view}", mode]
        if view != "cores":
            parts.append(method_tag(cores))
        if view != "unorm":
            parts.append(f"unorm{fixed[mode]:.2f}")
        parts += [f"N{DAGS_PER_SET}", f"t{trials}"]
        return "_".join(parts) + ".pdf"

    written = []
    for view, title, xlabel, at_fixed_unorm in views:
        for mode in MODES:
            fig, ax = plt.subplots(figsize=(6.4, 4.6))
            pts = sorted((r for r in rows if r["figure"] == view and r["mode"] == mode
                          and r["x"] == r["x"]), key=lambda r: r["x"])
            for col, label, color, marker in SERIES:
                xy = [(r["x"], r[col]) for r in pts if r[col] is not None]
                if not xy:
                    continue
                ax.plot([p[0] for p in xy], [p[1] for p in xy], color=color, marker=marker,
                        markersize=6, linewidth=1.6, label=label, markeredgecolor="white", markeredgewidth=0.8)
            panel = f"{title} -- {mode} deadlines"
            if at_fixed_unorm:
                panel += f", U_norm = {fixed[mode]}"
            ax.set_title(f"{panel}\nMelani DAGs (p_par 0.8, depth 3, p_add 0.1), N = {DAGS_PER_SET}, "
                         f"{trials} task sets/point\n{method_text(view)}",
                         fontsize=9, color=ink, loc="left")
            ax.set_xlabel(xlabel)
            ax.set_ylabel("acceptance ratio (%)")
            ax.set_ylim(-3, 103)
            ax.grid(True, color=grid, linewidth=0.8)
            ax.set_axisbelow(True)
            # Legend in the fixed SERIES order; the constrained figures have
            # no Li series, which simply drops out.
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, frameon=False, fontsize=8)
            fig.tight_layout()
            path = out_dir / file_name(view, mode)
            fig.savefig(path)
            plt.close(fig)
            written.append(path)
    return written


if __name__ == "__main__":
    main()
