#!/usr/bin/env python3
"""Regenerates an RD-Gen DAG pool from a config file and immediately reports
DAG-Fluid's own per-DAG segment-concurrency stats against it -- the same
numbers shown when RD-Gen/sample_config/branching/
my_branching_theory_pool_dagfluid.yaml was tuned against this project's real
target's DAG-pool core count (see that file's own doc comments for the
methodology). Run this after every RD-Gen generation-parameter change
instead of eyeballing pool output by hand, so a regression (or an
improvement) in max segment concurrency is never missed.

Wraps two steps that would otherwise need to be run and remembered
separately:
  1. RD-Gen's own `venv/bin/python3 run_generator.py -c <config> -d <dest>`.
  2. rd_gen_to_dags's `segment_concurrency_stats` example (built once, on
     first use, if the release binary isn't there yet) against `<dest>/DAGs`.

Usage:
  scripts/generate_and_check_pool.py \\
      --config /home/nokosan/ws/RD-Gen/sample_config/branching/my_branching_theory_pool_dagfluid.yaml \\
      --dest-dir /home/nokosan/ws/RD-Gen/test/awkernel_theory_pool_branching_dagfluid \\
      --target 4
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

DEFAULT_RD_GEN_DIR = Path("/home/nokosan/ws/RD-Gen")
DEFAULT_AWKERNEL_DIR = Path("/home/nokosan/ws/awkernel")


def generate_pool(rd_gen_dir: Path, config: Path, dest_dir: Path):
    """Regenerate `dest_dir` from `config` via RD-Gen's own generator.
    `run_generator.py` refuses to overwrite an existing directory without an
    interactive y/n prompt (see its own `input("[Y]es / [N]o?:")`) -- removed
    up front instead so this stays fully unattended and reproducible, the
    same way `real_machine_trial.py`'s own `stage_dag`/`stage_group` already
    do for their own staging directories."""
    import shutil

    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    print(f"[generate] {rd_gen_dir}/venv/bin/python3 run_generator.py -c {config} -d {dest_dir}")
    subprocess.run(
        ["venv/bin/python3", "run_generator.py", "-c", str(config), "-d", str(dest_dir)],
        cwd=rd_gen_dir,
        check=True,
    )


def ensure_stats_tool_built(awkernel_dir: Path, dags_dir_for_embed: Path) -> Path:
    """Build rd_gen_to_dags's `segment_concurrency_stats` example (host-side,
    `--features std`, same as `acceptance_ratio`/`predict_admission`) if the
    release binary isn't already there. Cargo itself is a no-op when nothing
    changed, so this is cheap to call every time rather than caching a
    "did I already build this" flag.

    `rd_gen_to_dags`'s own `build.rs` unconditionally requires `RD_GEN_DAGS_DIR`
    to point at a directory with at least one `dag_<N>.yaml` (it panics
    otherwise, even though this particular example never touches the files it
    embeds from there -- it reads its own pool directory at runtime instead).
    `dags_dir_for_embed` -- the pool this call is about to check anyway -- is
    used to satisfy that requirement rather than depending on some unrelated,
    possibly-stale directory a previous build happened to leave behind."""
    binary = awkernel_dir / "target" / "release" / "examples" / "segment_concurrency_stats"
    print("[build] segment_concurrency_stats (skipped if already up to date)")
    subprocess.run(
        ["cargo", "build", "--release", "--example", "segment_concurrency_stats", "--features", "std"],
        cwd=awkernel_dir / "applications" / "rd_gen_to_dags",
        env={**os.environ, "RD_GEN_DAGS_DIR": str(dags_dir_for_embed)},
        check=True,
    )
    if not binary.exists():
        raise RuntimeError(f"build succeeded but {binary} still doesn't exist -- unexpected cargo layout?")
    return binary


def run_stats(binary: Path, dags_dir: Path, target: int):
    print("[stats]")
    subprocess.run([str(binary), str(dags_dir), str(target)], check=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, type=Path, help="RD-Gen config YAML to generate from")
    p.add_argument("--dest-dir", required=True, type=Path,
                   help="destination directory (an existing one is removed and recreated -- "
                        "matches run_generator.py's own -d semantics)")
    p.add_argument("--target", type=int, default=4,
                   help="target DAG-pool core count to report the over-target percentage "
                        "against (default: 4, this project's real target's own DAG-pool "
                        "core count -- see scheduler::pool's own doc)")
    p.add_argument("--rd-gen-dir", type=Path, default=DEFAULT_RD_GEN_DIR)
    p.add_argument("--awkernel-dir", type=Path, default=DEFAULT_AWKERNEL_DIR)
    p.add_argument("--skip-generate", action="store_true",
                   help="only run the stats step, against an already-generated --dest-dir "
                        "(e.g. to re-check a pool without regenerating it)")
    args = p.parse_args()

    try:
        if not args.skip_generate:
            generate_pool(args.rd_gen_dir, args.config, args.dest_dir)
        binary = ensure_stats_tool_built(args.awkernel_dir, args.dest_dir / "DAGs")
        run_stats(binary, args.dest_dir / "DAGs", args.target)
    except subprocess.CalledProcessError as e:
        print(f"[abort] {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
