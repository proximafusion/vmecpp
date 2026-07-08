# SPDX-FileCopyrightText: 2025-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Continuation in Fourier resolution.

VMEC++ converges a hard equilibrium more reliably, and often in fewer total
iterations, when the Fourier resolution is increased gradually rather than solved
at the target resolution from scratch. Each step solves a single resolution and
hot-restarts from the previous step, whose solution is interpolated to the new
resolution by :func:`vmecpp.interpolate_solution` (radial interpolation in
``sqrt(s)`` plus Fourier zero-padding).

``vmecpp.run_continuation`` performs the whole schedule in one call:

    output = vmecpp.run_continuation(
        vmec_input,
        ns_array=[15, 31, 31],
        mpol_array=[5, 9, 13],
        ntor_array=[4, 4, 4],
    )

This example runs the schedule step by step instead, so it can report the
per-step iteration counts and compare the total against solving at the target
Fourier resolution directly (the radial multi-grid is used in both cases, so the
only difference is whether ``mpol`` is ramped up or held fixed).
"""

from pathlib import Path

import numpy as np

import vmecpp

vmec_input = vmecpp.VmecInput.from_file(
    Path("examples") / "data" / "cth_like_fixed_bdy.json"
)

# The continuation schedule: ramp the poloidal resolution mpol = 5 -> 9 -> 13
# while refining the radial grid, at fixed toroidal resolution ntor = 4.
ns_array = [15, 31, 31]
mpol_array = [5, 9, 13]
ntor_array = [4, 4, 4]

ftol = float(np.asarray(vmec_input.ftol_array)[-1])
niter = int(np.asarray(vmec_input.niter_array)[-1])


def n_iterations(output: vmecpp.VmecOutput) -> int:
    """Number of solver iterations, i.e. the length of the force-residual history."""
    return int(np.asarray(output.wout.fsqt).shape[0])


def step_input(ns: int, mpol: int, ntor: int) -> vmecpp.VmecInput:
    """A single-resolution input for one continuation step."""
    step = vmec_input.model_copy(deep=True)
    step.mpol = mpol
    step.ntor = ntor
    step.ns_array = np.array([ns])
    step.ftol_array = np.array([ftol])
    step.niter_array = np.array([niter])
    step.rbc = vmecpp.VmecInput.resize_2d_coeff(step.rbc, mpol_new=mpol, ntor_new=ntor)
    step.zbs = vmecpp.VmecInput.resize_2d_coeff(step.zbs, mpol_new=mpol, ntor_new=ntor)
    return step


# --- Fourier continuation ---------------------------------------------------
# vmecpp.run_continuation(vmec_input, ns_array=ns_array, mpol_array=mpol_array,
# ntor_array=ntor_array) runs exactly this schedule in a single call; it is
# unrolled here to report the per-step iteration counts.
print("Fourier continuation:")
schedule = list(zip(ns_array, mpol_array, ntor_array, strict=True))

# First step: solve from scratch at the lowest resolution.
ns, mpol, ntor = schedule[0]
continued = vmecpp.run(step_input(ns, mpol, ntor), verbose=False)
continuation_iterations = n_iterations(continued)
print(f"  mpol={mpol:2d}, ns={ns:2d}: {n_iterations(continued):5d} iterations")

# Each later step hot-restarts from the previous solution, interpolated up to the
# new (ns, mpol, ntor) by interpolate_solution().
for ns, mpol, ntor in schedule[1:]:
    current_input = step_input(ns, mpol, ntor)
    guess = vmecpp.interpolate_solution(continued, current_input)
    continued = vmecpp.run(current_input, restart_from=guess, verbose=False)
    continuation_iterations += n_iterations(continued)
    print(f"  mpol={mpol:2d}, ns={ns:2d}: {n_iterations(continued):5d} iterations")

# --- Fixed Fourier resolution -----------------------------------------------
fixed_input = step_input(ns_array[-1], mpol_array[-1], ntor_array[-1])
fixed_input.ns_array = np.array([ns_array[0], ns_array[-1]])
fixed_input.ftol_array = np.array([ftol, ftol])
fixed_input.niter_array = np.array([niter, niter])
fixed_output = vmecpp.run(fixed_input, verbose=False)
fixed_iterations = n_iterations(fixed_output)

# --- comparison -------------------------------------------------------------
print(f"\nfixed resolution (mpol={mpol_array[-1]}): {fixed_iterations:5d} iterations")
print(
    f"Fourier continuation:        {continuation_iterations:5d} iterations "
    f"({continuation_iterations / fixed_iterations:.0%} of fixed)"
)

max_geometry_difference = np.max(
    np.abs(np.asarray(fixed_output.wout.rmnc) - np.asarray(continued.wout.rmnc))
)
print(
    "both approaches reach the same equilibrium: "
    f"max|delta rmnc| = {max_geometry_difference:.1e}"
)
