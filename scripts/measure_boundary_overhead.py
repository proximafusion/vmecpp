#!/usr/bin/env python3
"""Measure per-thread `computeJacobian.body` time as a function of NS.

The hypothesis (from inspection of the data at NS=300) is that the axis
and LCFS owners pay a *constant* per-call overhead (axis-degenerate-case
init, LCFS handover) on top of a per-surface cost. Equivalently, for
thread t with slice size S_t:

    time_t  ~  a  +  b * S_t  +  delta_t

where `delta_axis` and `delta_lcfs` are small constants (zero for
interior threads). If this model holds, the right partitioning is to
shift `delta_axis / b` surfaces off the axis owner and `delta_lcfs / b`
off the LCFS owner -- an *additive* compensation, not a *multiplicative*
weight.

This script varies NS at fixed thread count (-t 10 by default) and reads
the per-thread cumulative `computeJacobian.body` time from the
instrumented binary's stderr dump, then fits the model and reports
delta_axis / delta_lcfs in microseconds and in equivalent-surface units.
"""

from __future__ import annotations

import argparse
import os
import re
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path


def edit_input_ns(template_path: Path, ns: int, dst_path: Path) -> None:
    text = template_path.read_text()
    text = re.sub(r" NS_ARRAY\s*=\s*\d+", f" NS_ARRAY    = {ns}", text)
    dst_path.write_text(text)


def run_one(vmecpp: Path, input_file: Path, threads: int) -> str:
    env = os.environ.copy()
    env["OMP_WAIT_POLICY"] = "passive"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["VMECPP_DUMP_BARRIER_TIMINGS"] = "1"
    cmd = [str(vmecpp), "-t", str(threads), "-q", str(input_file)]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stderr[-1000:], file=sys.stderr)
        sys.exit(f"vmecpp failed for {input_file}")
    return proc.stderr


def parse_function_timer(stderr: str, site: str, num_threads: int) -> tuple[list[float], int]:
    """Return (per_thread_total_s, count) for the given FunctionTimer site.

    Dump format (from omp-instrument-v2's instrumentor):

        site: computeJacobian.body
           tid       count       total_s        avg_us
             0         341      0.245909        721.14
             1         341      ...

    We slice out the table after `site: <site>` and parse rows until we
    hit a blank line or another `site:` header.
    """
    out = [float("nan")] * num_threads
    n_calls = 0
    header = re.search(rf"site:\s*{re.escape(site)}\s*$", stderr, re.MULTILINE)
    if not header:
        return out, n_calls
    block = stderr[header.end():]
    # Stop at the next "site:" or blank-line-after-data
    end_match = re.search(r"^\s*site:\s", block, re.MULTILINE)
    if end_match:
        block = block[: end_match.start()]
    # Skip the "tid count total_s avg_us" header line
    row_pattern = re.compile(
        r"^\s*(\d+)\s+(\d+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s*$",
        re.MULTILINE,
    )
    for m in row_pattern.finditer(block):
        tid = int(m.group(1))
        cnt = int(m.group(2))
        total_s = float(m.group(3))
        if 0 <= tid < num_threads:
            out[tid] = total_s
            n_calls = max(n_calls, cnt)
    return out, n_calls


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--threads", type=int, default=10)
    p.add_argument("--ns-values", type=str, default="50,100,150,200,300,400,500")
    p.add_argument(
        "--vmecpp",
        type=Path,
        default=Path("/home/jurasic/vmecpp/.venv-instr2/bin/vmecpp"),
    )
    p.add_argument(
        "--input-template",
        type=Path,
        default=Path("/home/jurasic/vmecpp/examples/data/input.w7x"),
    )
    p.add_argument(
        "--site", type=str, default="computeJacobian.body",
        help="FunctionTimer site to measure",
    )
    args = p.parse_args()

    ns_values = [int(s) for s in args.ns_values.split(",")]

    print(f"Workload: {args.input_template} (NS varied), -t {args.threads}, passive policy")
    print(f"Measuring: per-thread cumulative time at site `{args.site}`\n")

    # Header
    header = ["NS", "tid0(us/call)", "tid_lcfs(us/call)", "tid_inner_min", "tid_inner_max"]
    print(" ".join(f"{h:>16}" for h in header))
    print("-" * (17 * len(header)))

    rows = []
    for ns in ns_values:
        with tempfile.TemporaryDirectory() as td:
            tmp_input = Path(td) / "input.bench"
            edit_input_ns(args.input_template, ns, tmp_input)
            stderr = run_one(args.vmecpp, tmp_input, args.threads)

        # Parse per-thread total seconds
        per_thread_total, n_calls = parse_function_timer(stderr, args.site, args.threads)
        if n_calls == 0:
            print(f"  (could not parse {args.site} for NS={ns}; dump excerpt:)")
            print("  ", stderr[-500:])
            continue

        # Convert to us/call
        per_thread_us = [t * 1e6 / n_calls for t in per_thread_total]
        tid_axis = per_thread_us[0]
        tid_lcfs = per_thread_us[args.threads - 1]
        interior = per_thread_us[1:args.threads - 1]
        inner_min = min(interior) if interior else float("nan")
        inner_max = max(interior) if interior else float("nan")
        rows.append((ns, tid_axis, tid_lcfs, inner_min, inner_max, n_calls))

        print(
            f"{ns:>16d} {tid_axis:>16.1f} {tid_lcfs:>16.1f} "
            f"{inner_min:>16.1f} {inner_max:>16.1f}"
        )

    # Fit additive model: time = a + b * slice + delta_t
    # We use slice_size for each thread (NS-1)/T plus remainder.
    print()
    print("Fit (additive model): time = a + b*slice_size + delta")
    print(f"  {args.threads} threads, NS varied")

    # For each NS, slice_size for thread 0 is ceil((NS-1)/T) when 0 < remainder
    # else (NS-1)/T. Use the actual partitioning logic.
    def slice_size(ns: int, t: int, tid: int) -> int:
        n = ns - 1  # fixed-bdy
        base = n // t
        rem = n % t
        return base + 1 if tid < rem else base

    # Use middle-interior threads (tid t/2) as the "interior baseline"
    interior_tid = args.threads // 2
    interior_data = []  # (slice_size, time_us)
    axis_data = []
    lcfs_data = []
    for ns, tid_axis, tid_lcfs, inner_min, inner_max, n_calls in rows:
        slice_axis = slice_size(ns, args.threads, 0)
        slice_lcfs = slice_size(ns, args.threads, args.threads - 1)
        slice_inner = slice_size(ns, args.threads, interior_tid)
        # Use the median interior thread's time (assumes inner_min <=
        # tid_inner_median <= inner_max, but we don't have it directly --
        # use mean of inner_min and inner_max as a proxy)
        time_inner = 0.5 * (inner_min + inner_max)
        interior_data.append((slice_inner, time_inner))
        axis_data.append((slice_axis, tid_axis))
        lcfs_data.append((slice_lcfs, tid_lcfs))

    # Linear fit: time = a + b * slice
    def fit_line(xs, ys):
        n = len(xs)
        sx = sum(xs); sy = sum(ys); sxx = sum(x * x for x in xs); sxy = sum(x * y for x, y in zip(xs, ys))
        denom = n * sxx - sx * sx
        if abs(denom) < 1e-9:
            return float("nan"), float("nan")
        b = (n * sxy - sx * sy) / denom
        a = (sy - b * sx) / n
        return a, b

    a_int, b_int = fit_line([d[0] for d in interior_data], [d[1] for d in interior_data])
    print(f"  interior (median): a = {a_int:.1f} us, b = {b_int:.2f} us/surface")

    # Compute delta_axis and delta_lcfs as residuals from the interior fit
    deltas_axis = [t - (a_int + b_int * s) for s, t in axis_data]
    deltas_lcfs = [t - (a_int + b_int * s) for s, t in lcfs_data]

    print(f"\n  delta_axis (per NS): {[f'{d:.0f}' for d in deltas_axis]} us")
    print(f"  median delta_axis = {statistics.median(deltas_axis):.0f} us = "
          f"{statistics.median(deltas_axis) / b_int:.2f} surfaces")
    print(f"\n  delta_lcfs (per NS): {[f'{d:.0f}' for d in deltas_lcfs]} us")
    print(f"  median delta_lcfs = {statistics.median(deltas_lcfs):.0f} us = "
          f"{statistics.median(deltas_lcfs) / b_int:.2f} surfaces")


if __name__ == "__main__":
    main()
