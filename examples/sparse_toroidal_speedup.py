# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Demonstrate the sparse-toroidal-mode speedup.

VMEC++ stores its internal geometry coefficients only for the toroidal mode
numbers ``n`` that actually appear in the boundary/axis (the "active" set),
instead of densely for every ``n`` in ``[0, ntor]``. When a configuration uses
a large ``ntor`` to admit a few high-``n`` modes -- but most toroidal modes are
zero -- the toroidal Fourier transform (a hot inner loop) only does work
proportional to the number of active modes, not to ``ntor + 1``.

This is a behaviour-preserving optimization: the converged equilibrium is
identical to a dense run, only faster. The input here uses ``ntor = 36`` with
nfp = 5 but only ``|n| <= 4`` populated, so the active set is ``{0, 1, 2, 3, 4}``
(5 modes) out of a dense 37.

The published PyPI build of VMEC++ predates this feature and always runs dense,
so running the *same* input through both builds shows the speedup directly. The
single-thread speedup grows with ``ntor`` (the wider the dense grid, the more
work the reference build wastes on zero modes); when both builds are allowed
multiple threads the dense build also burns proportionally more CPU for the same
result.

Usage (A/B across two installs)::

    # .pypi_venv holds the reference (published) build; .venv holds this build.
    uv run --python .venv/bin/python examples/sparse_toroidal_speedup.py \
        --reference-python .pypi_venv/bin/python

Run with a single interpreter (just times this build) by omitting
``--reference-python``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import vmecpp

# A child process runs exactly one equilibrium and prints a one-line JSON result
# to stdout. We use this so each VMEC++ build executes in its own interpreter.
_CHILD = """
import json, sys, time
import vmecpp

inp = vmecpp.VmecInput.from_file(sys.argv[1])
max_threads = int(sys.argv[2])

t = time.perf_counter()
out = vmecpp.run(inp, max_threads=max_threads, verbose=False)
elapsed = time.perf_counter() - t

print("VMECPP_RESULT " + json.dumps({
    "seconds": elapsed,
    "ntor": int(inp.ntor),
    "volume_p": float(out.wout.volume_p),
    "betatotal": float(out.wout.betatotal),
    "version": getattr(vmecpp, "__version__", "unknown"),
}))
"""


def run_in(python: str, input_path: Path, max_threads: int) -> dict:
    """Run one equilibrium in the given Python interpreter, return its result."""
    proc = subprocess.run(
        [python, "-c", _CHILD, str(input_path), str(max_threads)],
        capture_output=True,
        text=True,
        check=True,
    )
    for line in proc.stdout.splitlines():
        if line.startswith("VMECPP_RESULT "):
            return json.loads(line[len("VMECPP_RESULT ") :])
    message = f"{python} did not produce a result. stderr:\n{proc.stderr[-2000:]}"
    raise RuntimeError(message)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference-python",
        default=None,
        help="Path to the reference VMEC++ interpreter (e.g. .pypi_venv/bin/python). "
        "If omitted, only this build is timed.",
    )
    parser.add_argument(
        "--max-threads",
        type=int,
        default=1,
        help="Threads per run; use 1 for a clean apples-to-apples comparison.",
    )
    args = parser.parse_args()

    input_path = Path("examples") / "data" / "cth_like_high_ntor.json"
    inp = vmecpp.VmecInput.from_file(input_path)
    print(
        f"Input: {input_path.name}  (nfp={inp.nfp}, mpol={inp.mpol}, ntor={inp.ntor})\n"
        "Only |n| <= 4 are populated, so the active toroidal set is "
        f"{{0, 1, 2, 3, 4}} out of a dense {inp.ntor + 1}.\n"
    )

    # This build (sparse).
    this = run_in(sys.executable, input_path, args.max_threads)
    print(f"this build  ({sys.executable}):")
    print(f"    time = {this['seconds']:.2f} s   volume = {this['volume_p']:.6e}")

    if args.reference_python is None:
        print(
            "\n(no --reference-python given; pass .pypi_venv/bin/python to compare "
            "against the published build)"
        )
        return

    ref = run_in(args.reference_python, input_path, args.max_threads)
    print(f"reference   ({args.reference_python}):")
    print(f"    time = {ref['seconds']:.2f} s   volume = {ref['volume_p']:.6e}")

    speedup = ref["seconds"] / this["seconds"]
    # Identical physics is the whole point: this is an optimization, not an
    # approximation. Volumes should agree to many digits.
    rel_diff = abs(ref["volume_p"] - this["volume_p"]) / abs(ref["volume_p"])
    print(
        f"\nspeedup: {speedup:.2f}x faster than the reference build\n"
        f"plasma volume relative difference: {rel_diff:.2e} (same equilibrium)"
    )


if __name__ == "__main__":
    main()
