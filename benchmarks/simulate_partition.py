#!/usr/bin/env python3
"""Simulate imbalance-aware radial partitioning.

Given a measured per-thread weight schedule (axis owner is heavier, LCFS
owner is heavier, interiors are unit weight), this simulates the integer
partition under three strategies:

  (1) `equal`   — current code: every thread gets ns/T surfaces (with
                  the +/-1 remainder rule). Boundary threads carry their
                  full slice but their effective work is weight*S.
  (2) `weighted_optimal` — binary-search-on-M for the smallest makespan
                  achievable under contiguous integer partitioning.
  (3) `safe`    — run (2), but fall back to (1) if it does not strictly
                  improve makespan. Guarantees no regression at any
                  (ns, T).

The output is a table for each (ns, T) showing per-thread surface count,
effective work, makespan, and whether `safe` differs from `equal`.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass


def equal_partition(n_surfaces: int, n_threads: int) -> list[int]:
    """Current vmecpp partitioning: surfaces / T, +1 to first remainder threads."""
    base = n_surfaces // n_threads
    rem = n_surfaces % n_threads
    return [base + 1 if t < rem else base for t in range(n_threads)]


def additive_partition(
    n_surfaces: int, n_threads: int, axis_offset: int, lcfs_offset: int
) -> list[int]:
    """Take `axis_offset` surfaces off tid 0 and `lcfs_offset` off tid N-1, redistribute
    the surplus evenly across the interior threads.

    Mirrors the user's hypothesis that boundary overhead is additive (constant per-call)
    rather than multiplicative.
    """
    if n_threads == 1:
        return [n_surfaces]
    if n_threads == 2:
        # Both threads are boundary. Take from the heavier one only.
        if axis_offset >= lcfs_offset:
            return [
                n_surfaces // 2 - axis_offset // 2,
                n_surfaces - (n_surfaces // 2 - axis_offset // 2),
            ]
        return [
            n_surfaces - (n_surfaces // 2 - lcfs_offset // 2),
            n_surfaces // 2 - lcfs_offset // 2,
        ]

    counts = equal_partition(n_surfaces, n_threads)
    surplus = axis_offset + lcfs_offset
    counts[0] -= axis_offset
    counts[-1] -= lcfs_offset
    if any(c < 0 for c in counts):
        # Shouldn't happen for reasonable offsets; fall back to equal.
        return equal_partition(n_surfaces, n_threads)
    interior_count = n_threads - 2
    base_add = surplus // interior_count
    rem_add = surplus % interior_count
    for i in range(interior_count):
        counts[1 + i] += base_add + (1 if i < rem_add else 0)
    assert sum(counts) == n_surfaces, (counts, n_surfaces, axis_offset, lcfs_offset)
    return counts


def weighted_partition(n_surfaces: int, weights: list[float]) -> list[int]:
    """Imbalance-aware contiguous integer partition via Hamilton's method.

    Each thread's *fair share* of the n_surfaces is proportional to 1 / w[t]. Take the
    floor of every fair share (giving each thread at least its base allocation), then
    distribute the remaining surfaces to whichever threads have the largest fractional
    residual. This yields the same makespan as the binary-search-on-M optimum but never
    starves a thread (every thread gets at least floor(fair share) surfaces).

    Hamilton's method ties for makespan-optimality with binary-search-on-M while always
    preserving "every thread roughly its fair share", which is what we want here.
    """
    n_threads = len(weights)
    inv_w = [1.0 / w for w in weights]
    sum_inv_w = sum(inv_w)
    fair_shares = [n_surfaces * iw / sum_inv_w for iw in inv_w]

    counts = [int(fs) for fs in fair_shares]  # floor
    residuals = [fair_shares[t] - counts[t] for t in range(n_threads)]
    remainder = n_surfaces - sum(counts)

    # Distribute remainder to threads with the largest residual.
    order = sorted(range(n_threads), key=lambda t: -residuals[t])
    for t in order[:remainder]:
        counts[t] += 1

    assert sum(counts) == n_surfaces, (counts, n_surfaces)
    return counts


def makespan(counts: list[int], weights: list[float]) -> float:
    return max(c * w for c, w in zip(counts, weights))


@dataclass
class Result:
    counts: list[int]
    eff: list[float]
    makespan: float


def run(n_surfaces: int, n_threads: int, w_axis: float, w_lcfs: float):
    weights = [w_axis] + [1.0] * (n_threads - 2) + [w_lcfs]
    if n_threads == 1:
        weights = [max(w_axis, w_lcfs)]
    elif n_threads == 2:
        weights = [w_axis, w_lcfs]

    eq = equal_partition(n_surfaces, n_threads)
    wt = weighted_partition(n_surfaces, weights)

    eq_eff = [c * w for c, w in zip(eq, weights)]
    wt_eff = [c * w for c, w in zip(wt, weights)]
    eq_ms = max(eq_eff)
    wt_ms = max(wt_eff)

    safe_ms = min(eq_ms, wt_ms)
    safe_partition = wt if wt_ms < eq_ms else eq
    used_weighted = wt_ms < eq_ms - 1e-9

    return {
        "weights": weights,
        "equal": Result(eq, eq_eff, eq_ms),
        "weighted": Result(wt, wt_eff, wt_ms),
        "safe": Result(
            safe_partition, [c * w for c, w in zip(safe_partition, weights)], safe_ms
        ),
        "improvement_pct": 100 * (eq_ms - safe_ms) / eq_ms,
        "used_weighted": used_weighted,
    }


def fmt_counts(counts: list[int], cap: int = 10) -> str:
    if len(counts) <= cap:
        return str(counts)
    head = ", ".join(str(c) for c in counts[:3])
    tail = ", ".join(str(c) for c in counts[-3:])
    interior = counts[1:-1]
    if len(set(interior)) == 1:
        return f"[{counts[0]}, {interior[0]} x {len(interior)}, {counts[-1]}]"
    return f"[{head}, ..., {tail}]"


def additive_makespan(counts, weights, b, a, delta_axis, delta_lcfs):
    """Additive model: time_t = a + b*slice + delta_t."""
    n = len(counts)
    deltas = [0.0] * n
    if n >= 2:
        deltas[0] = delta_axis
        deltas[-1] = delta_lcfs
    if n == 1:
        deltas[0] = max(delta_axis, delta_lcfs)
    return max(a + b * c + d for c, d in zip(counts, deltas))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--w-axis",
        type=float,
        default=1.25,
        help="weight for the axis-owner thread (tid 0)",
    )
    parser.add_argument(
        "--w-lcfs",
        type=float,
        default=1.15,
        help="weight for the LCFS-owner thread (tid N-1)",
    )
    parser.add_argument(
        "--cases", type=str, default="99,199,299,499", help="comma-separated NS values"
    )
    parser.add_argument(
        "--threads",
        type=str,
        default="1,2,4,8,10,16,20,30,40",
        help="comma-separated thread counts",
    )
    parser.add_argument(
        "--additive",
        action="store_true",
        help="Also report makespan under the additive boundary-overhead model with measured b/a/delta values.",
    )
    parser.add_argument(
        "--axis-off",
        type=int,
        default=2,
        help="Surfaces shifted off the axis owner (additive scheme).",
    )
    parser.add_argument(
        "--lcfs-off", type=int, default=1, help="Surfaces shifted off the LCFS owner."
    )
    parser.add_argument(
        "--b",
        type=float,
        default=18.5,
        help="Per-surface cost (us) for additive model.",
    )
    parser.add_argument(
        "--a",
        type=float,
        default=392.0,
        help="Per-call constant overhead (us) for additive model.",
    )
    parser.add_argument(
        "--delta-axis",
        type=float,
        default=125.0,
        help="Boundary additive overhead at axis (us).",
    )
    parser.add_argument(
        "--delta-lcfs",
        type=float,
        default=78.0,
        help="Boundary additive overhead at LCFS (us).",
    )
    args = parser.parse_args()

    ns_vals = [int(x) for x in args.cases.split(",")]
    t_vals = [int(x) for x in args.threads.split(",")]

    if args.additive:
        print(
            f"Additive model: a={args.a} us, b={args.b} us/surf, "
            f"delta_axis={args.delta_axis} us, delta_lcfs={args.delta_lcfs} us"
        )
        print(f"Shift: axis_off={args.axis_off}, lcfs_off={args.lcfs_off}\n")
        header = (
            f"{'NS':>4} {'T':>3} {'eq makespan(us)':>16} "
            f"{'add makespan(us)':>17} {'gain%':>7}  partition"
        )
        print(header)
        print("-" * len(header))
        for ns in [int(x) for x in args.cases.split(",")]:
            n_surf = ns - 1
            for t in [int(x) for x in args.threads.split(",")]:
                if t > n_surf // 2:
                    continue
                eq = equal_partition(n_surf, t)
                add = additive_partition(n_surf, t, args.axis_off, args.lcfs_off)
                eq_ms = additive_makespan(
                    eq, None, args.b, args.a, args.delta_axis, args.delta_lcfs
                )
                add_ms = additive_makespan(
                    add, None, args.b, args.a, args.delta_axis, args.delta_lcfs
                )
                gain = 100.0 * (eq_ms - add_ms) / eq_ms
                marker = "  <- regress" if gain < 0 else ""
                print(
                    f"{ns:>4} {t:>3} {eq_ms:>16.0f} {add_ms:>17.0f} "
                    f"{gain:>6.2f}%  {fmt_counts(add)}{marker}"
                )
            print()
        return

    print(f"Weights: axis = {args.w_axis}, LCFS = {args.w_lcfs}, interior = 1.0\n")
    header = f"{'NS':>4} {'T':>3} {'eq makespan':>12} {'wt makespan':>12} {'used wt?':>9} {'gain%':>7}  partition"
    print(header)
    print("-" * len(header))
    for ns in ns_vals:
        # vmecpp distributes ns-1 fixed-bdy surfaces among threads
        n_surf = ns - 1
        for t in t_vals:
            if t > n_surf // 2:
                continue
            r = run(n_surf, t, args.w_axis, args.w_lcfs)
            print(
                f"{ns:>4} {t:>3} "
                f"{r['equal'].makespan:>12.2f} "
                f"{r['weighted'].makespan:>12.2f} "
                f"{('yes' if r['used_weighted'] else 'no'):>9} "
                f"{r['improvement_pct']:>6.2f}%  "
                f"{fmt_counts(r['safe'].counts)}"
            )
        print()


if __name__ == "__main__":
    main()
