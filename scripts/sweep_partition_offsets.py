#!/usr/bin/env python3
"""Sweep `VMECPP_AXIS_SURFACES_OFF` x `VMECPP_LCFS_SURFACES_OFF` and report
wall-time medians for each (n_axis, n_lcfs) combination at a given thread
count and input file. Pick the combination that minimises wall time.

Defaults are tuned for examples/data/input.w7x at NS=300 (heavy bench
workload).
"""

from __future__ import annotations

import argparse
import os
import re
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def run_one(vmecpp: Path, input_file: Path, threads: int,
            axis_off: int, lcfs_off: int, repeats: int,
            policy: str) -> list[float]:
    env = os.environ.copy()
    env["OMP_WAIT_POLICY"] = policy
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["VMECPP_AXIS_SURFACES_OFF"] = str(axis_off)
    env["VMECPP_LCFS_SURFACES_OFF"] = str(lcfs_off)
    cmd = [str(vmecpp), "-t", str(threads), "-q", str(input_file)]
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        proc = subprocess.run(cmd, env=env, capture_output=True)
        elapsed = time.perf_counter() - t0
        if proc.returncode != 0:
            print(proc.stderr[-500:].decode(errors="replace"), file=sys.stderr)
            sys.exit(f"vmecpp failed for axis={axis_off} lcfs={lcfs_off}")
        times.append(elapsed)
    return times


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--vmecpp", type=Path,
                   default=Path("/home/jurasic/vmecpp/.venv-partition/bin/vmecpp"))
    p.add_argument("--input", type=Path,
                   default=Path("/home/jurasic/vmecpp/examples/data/input.w7x"))
    p.add_argument("--threads", type=int, default=10)
    p.add_argument("--axis-range", type=str, default="0,1,2,3,4,5,6")
    p.add_argument("--lcfs-range", type=str, default="0,1,2,3,4,5,6")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--policy", choices=["active", "passive"], default="active",
                   help="OMP_WAIT_POLICY (default: active, the real-world setting)")
    p.add_argument("--ns", type=int, default=None,
                   help="If set, override NS_ARRAY in a copy of the input file "
                        "(does not mutate the original). Useful for sweeping "
                        "across radial-grid sizes.")
    args = p.parse_args()

    input_path = args.input
    tmp_dir = None
    if args.ns is not None:
        tmp_dir = tempfile.mkdtemp(prefix="vmecpp_sweep_")
        text = args.input.read_text()
        text = re.sub(r" NS_ARRAY\s*=\s*\d+", f" NS_ARRAY    = {args.ns}", text)
        input_path = Path(tmp_dir) / "input.bench"
        input_path.write_text(text)
        print(f"Using temporary input with NS={args.ns} at {input_path}")

    axis_values = [int(s) for s in args.axis_range.split(",")]
    lcfs_values = [int(s) for s in args.lcfs_range.split(",")]

    print(f"Workload: {input_path}, -t {args.threads}, {args.policy} policy, "
          f"{args.repeats} repeats per cell")
    print(f"Sweeping axis_off in {axis_values}, lcfs_off in {lcfs_values}")
    print()

    # Header row
    header = ["a\\l"] + [f"{l}" for l in lcfs_values]
    print(" ".join(f"{h:>8}" for h in header))

    results: dict[tuple[int, int], float] = {}
    raw: dict[tuple[int, int], list[float]] = {}

    for a in axis_values:
        row = [f"{a}"]
        for l in lcfs_values:
            times = run_one(args.vmecpp, input_path, args.threads,
                            a, l, args.repeats, args.policy)
            med = statistics.median(times)
            results[(a, l)] = med
            raw[(a, l)] = times
            row.append(f"{med:.2f}")
        print(" ".join(f"{c:>8}" for c in row), flush=True)

    print()
    # Find best (lowest median wall time)
    best = min(results.items(), key=lambda kv: kv[1])
    baseline = results[(0, 0)]
    speedup = baseline / best[1]
    print(f"Baseline (axis=0, lcfs=0): {baseline:.3f} s")
    print(f"Best (axis={best[0][0]}, lcfs={best[0][1]}): {best[1]:.3f} s "
          f"({speedup:.3f}x speedup, +{100*(speedup-1):.2f}%)")

    # Spread of medians vs raw runs
    print()
    print("Per-cell raw timings (in case of ties):")
    sorted_keys = sorted(results.keys(), key=lambda k: results[k])[:5]
    for k in sorted_keys:
        ts = raw[k]
        print(f"  axis={k[0]} lcfs={k[1]}: median={results[k]:.3f}s, "
              f"raws={[f'{t:.2f}' for t in ts]}")


if __name__ == "__main__":
    main()
