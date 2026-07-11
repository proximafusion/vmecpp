# Free-boundary solvers: NESTOR, BIEST, and Vac2

VMEC++ supports three user-selectable solvers for the vacuum-field
contribution to the free-boundary force (plus `only_coils` for
verification):

```json
{
  "free_boundary_method": "nestor",  // or "biest" or "vac2"
  "biest_accuracy_digits": 8         // only used by biest, in [1, 14]
}
```

or from Python:

```python
import vmecpp

inp = vmecpp.VmecInput.from_file("input.json")
inp.free_boundary_method = "biest"
inp.biest_accuracy_digits = 8
out = vmecpp.run(inp)
```

## NESTOR (default)

The classic Merkel solver: exterior Neumann problem for a scalar magnetic
potential, discretized with an analytic singularity subtraction and a dense
LU solve in a truncated Fourier basis. Accuracy is limited by the low-order
singular quadrature and, at high poloidal resolution, by the factorial
growth of the analytic kernel coefficients (`c_mn ~ binom(2m, m)`), which
caps the usable poloidal mode number at mpol <= ~12-16 on 3D geometries.
Supports axisymmetric (`nzeta == 1`) configurations.

## BIEST

Solves the same exterior Neumann problem (vacuum field with `B . n = 0` on
the LCFS and prescribed net toroidal current) with the BIEST boundary
integral equation solver (Malhotra et al.,
https://github.com/dmalhotra/BIEST): high-order partition-of-unity singular
quadrature and a GMRES solve. `biest_accuracy_digits` sets the requested
number of decimal digits of accuracy of both the quadrature and the GMRES
tolerance.

Compared to NESTOR at typical resolutions, the BIEST vacuum pressure is the
more accurate one: NESTOR-vs-BIEST differences equal NESTOR's discretization
error (consistent with independent comparisons against the Strumberger
vac2 implementation). In-loop, BIEST reaches a lower converged
DELBSQ (vacuum/plasma pressure mismatch at the LCFS) than NESTOR.

Requirements and limitations:

- Requires `nzeta >= 4` (a true 3D toroidal discretization); use NESTOR for
  axisymmetric runs.
- Stellarator-symmetric (`lasym = false`) and non-symmetric boundaries are
  supported.
- System dependencies: Intel MKL's FFTW3 interface and LAPACKE
  (Ubuntu: `libmkl-dev`, `liblapacke-dev`). FFTW itself is GPL and cannot
  be linked into the MIT-licensed VMEC++.

### Update strategy (important for convergence)

VMEC++ calls the vacuum solver every iteration once the vacuum pressure is
switched on. BIEST solves on the fresh boundary at every call, with a GMRES
warm start from the previous solution (cheap once the boundary settles).
`nvacskip` is effectively ignored by BIEST.

This per-iteration tracking is essential: an experiment that instead kept
the vacuum pressure frozen between full updates (every `nvacskip`-th
iteration) stalled VMEC's descent at `fsq ~ 1e-3` on the CTH-like test
case, while per-iteration updates converge to `ftol = 1e-10` in fewer
iterations than NESTOR (421 vs 489) with a ~25% lower converged DELBSQ.

NESTOR-style cheap partial updates (frozen singular-quadrature setup and
net-current field, fresh coil-field right-hand side) are implemented but
disabled by default: the frozen-setup inconsistency was observed to slow
the descent (~1.7-2x iterations, occasional restarts) even at tight
thresholds. For experiments they can be enabled by setting
`VMECPP_BIEST_PARTIAL_DRIFT_TOL` (maximum boundary-coefficient drift since
the last full solve, relative to the major radius, below which a partial
update is used; e.g. `1e-5`).

## Vac2

The Strumberger/Tichmann reformulation of the Neumann solver, a C++ port
of the Fortran vac2 reference. It solves the same
problem as NESTOR but avoids the factorial growth of NESTOR's analytic
kernel coefficients: its accuracy improves monotonically with resolution
(O(h^3) observed against BIEST) where NESTOR saturates at its formulation
floor.

On the CTH-like test case at production resolution, Vac2 agrees with BIEST
to 2e-3 relative RMS in the vacuum pressure (NESTOR: 8e-3) at a cost
comparable to NESTOR (~0.1 s per solve), and in-loop converges in the same
number of iterations as BIEST with the same lower DELBSQ floor. This makes
it a strong default candidate for accurate free-boundary runs.

Performance: the kernel assembly exploits stellarator symmetry (mirrored
columns are exact sign-flipped copies, verified to 5e-16 against the
unfolded path; disable with `VAC2_NO_SYMM_FOLD=1`), and the solve uses a
bounded nested OpenMP team inside VMEC's parallel region
(`VMECPP_VAC2_SOLVE_THREADS` to override). On the CTH-like case the full
free-boundary run costs ~2.4 s vs 0.33 s for NESTOR (which amortizes via
frozen-matrix partial updates) and 0.58 s for NESTOR with full updates
every iteration.

NESTOR-style partial updates (frozen kernel matrices + Cholesky factor,
fresh right-hand side) are implemented but opt-in
(`VMECPP_VAC2_PARTIAL_UPDATES=1`): they preserve the converged answer and
cut wall time, but slow the descent (~2.6x iterations on CTH-like), the
same frozen-operator pathology observed with BIEST partial updates.

Current limitations: `lasym = true` is not supported yet by the wrapper;
axisymmetric (`nzeta == 1`) configurations are unverified (use NESTOR).

## Dual-run diagnostics

For solver comparisons, both solvers can be run side by side on the same
iteration; the configured `free_boundary_method` drives the run and the
shadow solver's output is only recorded:

```sh
VMECPP_FB_SHADOW=nestor \
VMECPP_FB_DUAL_DUMP=dual_dump.jsonl \
  ./vmec_standalone input_with_biest.json
```

Each vacuum update appends one JSON line with both |B|^2/2 fields, both
net-current integrals, and the boundary Fourier coefficients. Visualize
with:

```sh
python tools/plot_fb_dual_run.py dual_dump.jsonl
```

which produces side-by-side imshow panels (primary | shadow | difference),
the poloidal-spectrum evolution of both fields across the iteration, and
scalar traces.

`VMECPP_BIEST_TIMING=1` prints per-update setup/solve timings.

A standalone A/B harness (no VMEC iteration; evaluates both solvers on the
boundary given by `rbc`/`zbs` of an input file) is available as:

```sh
bazel run //vmecpp_tools/free_boundary_standalone -- input.json nestor
bazel run //vmecpp_tools/free_boundary_standalone -- input.json biest
```
