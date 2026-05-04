#!/usr/bin/env python3
"""Benchmark OMP barrier overhead across thread counts and wait policies.

Usage:
    python scripts/bench_omp_barriers.py [input_file] [--venv PATH] [--venv2 PATH]

Runs vmecpp -t <N> <input> under OMP_WAIT_POLICY=active and =passive for
thread counts 1, 2, 4, 8.  Reports wall time (median of 3 runs), parallel
scaling efficiency, and the active/passive ratio.

When two venvs are given (--venv / --venv2), runs the full table for each
and prints a comparison so you can see the before/after impact of a refactor.

Defaults:
    input_file  examples/data/w7x.json
    --venv      .venv   (baseline)
    --venv2     (not set -- single-venv mode)
"""

import argparse
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

THREAD_COUNTS = [1, 2, 4, 8]
POLICIES = ["active", "passive"]
REPEATS = 3


def vmecpp_bin(venv: Path) -> Path:
    return venv / "bin" / "vmecpp"


def run_once(vmecpp: Path, input_file: Path, threads: int, policy: str) -> float:
    env = os.environ.copy()
    env["OMP_WAIT_POLICY"] = policy
    env["OMP_NUM_THREADS"] = str(threads)
    # numpy/scipy each ship their own OpenBLAS, which by default spawns a
    # thread pool sized to nproc. With OMP-based vmecpp on top, that becomes
    # severe oversubscription (~27% CPU went to blas_thread_server in
    # measurement). Pin BLAS to 1 thread for clean OMP timings.
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    cmd = [str(vmecpp), "-t", str(threads), "-q", str(input_file)]
    t0 = time.perf_counter()
    result = subprocess.run(cmd, env=env, capture_output=True)
    elapsed = time.perf_counter() - t0
    if result.returncode != 0:
        print(f"  ERROR (exit {result.returncode}):", file=sys.stderr)
        print(result.stderr.decode(errors="replace")[-500:], file=sys.stderr)
        return float("nan")
    return elapsed


def bench(vmecpp: Path, input_file: Path) -> dict:
    results = {}
    total = len(THREAD_COUNTS) * len(POLICIES) * REPEATS
    done = 0
    for threads in THREAD_COUNTS:
        for policy in POLICIES:
            times = []
            for rep in range(REPEATS):
                done += 1
                print(
                    f"  [{done:2d}/{total}] t={threads} policy={policy} rep={rep+1} ...",
                    end="",
                    flush=True,
                )
                t = run_once(vmecpp, input_file, threads, policy)
                times.append(t)
                print(f" {t:.2f}s")
            results[(threads, policy)] = statistics.median(times)
    return results


def print_table(label: str, results: dict) -> None:
    t1_active = results.get((1, "active"), float("nan"))
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    header = f"  {'threads':>7}  {'active (s)':>10}  {'passive (s)':>11}  {'act/pas':>7}  {'scaling (act)':>13}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for threads in THREAD_COUNTS:
        active = results.get((threads, "active"), float("nan"))
        passive = results.get((threads, "passive"), float("nan"))
        ratio = active / passive if passive else float("nan")
        efficiency = (t1_active / active / threads) if active else float("nan")
        print(
            f"  {threads:>7}  {active:>10.3f}  {passive:>11.3f}  {ratio:>7.3f}  {efficiency:>13.1%}"
        )


def print_comparison(label1: str, r1: dict, label2: str, r2: dict) -> None:
    print(f"\n{'='*70}")
    print(f"  Comparison: {label1}  vs  {label2}")
    print(f"{'='*70}")
    header = (
        f"  {'threads':>7}  {'pol':>7}"
        f"  {label1[:10]:>10}  {label2[:10]:>10}  {'speedup':>7}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for threads in THREAD_COUNTS:
        for policy in POLICIES:
            t1 = r1.get((threads, policy), float("nan"))
            t2 = r2.get((threads, policy), float("nan"))
            speedup = t1 / t2 if t2 else float("nan")
            print(
                f"  {threads:>7}  {policy:>7}  {t1:>10.3f}  {t2:>10.3f}  {speedup:>7.3f}"
            )


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "input_file",
        nargs="?",
        default=str(repo_root / "examples" / "data" / "w7x.json"),
        help="VMEC++ input file (default: examples/data/w7x.json)",
    )
    parser.add_argument(
        "--venv",
        default=str(repo_root / ".venv"),
        help="Path to baseline venv (default: .venv)",
    )
    parser.add_argument(
        "--venv2",
        default=None,
        help="Path to second (under-test) venv for comparison",
    )
    args = parser.parse_args()

    input_file = Path(args.input_file).resolve()
    if not input_file.exists():
        sys.exit(f"Input file not found: {input_file}")

    venv1 = Path(args.venv).resolve()
    bin1 = vmecpp_bin(venv1)
    if not bin1.exists():
        sys.exit(f"vmecpp not found in venv: {bin1}")

    print(f"Input:   {input_file}")
    print(f"Repeats: {REPEATS} per (threads, policy) combination")

    print(f"\nBenchmarking venv1: {venv1}")
    r1 = bench(bin1, input_file)
    print_table(str(venv1.name), r1)

    if args.venv2:
        venv2 = Path(args.venv2).resolve()
        bin2 = vmecpp_bin(venv2)
        if not bin2.exists():
            sys.exit(f"vmecpp not found in venv2: {bin2}")
        print(f"\nBenchmarking venv2: {venv2}")
        r2 = bench(bin2, input_file)
        print_table(str(venv2.name), r2)
        print_comparison(str(venv1.name), r1, str(venv2.name), r2)


if __name__ == "__main__":
    main()
