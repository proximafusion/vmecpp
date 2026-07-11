# Free-boundary solvers: NESTOR and BIEST

VMEC++ supports two user-selectable solvers for the vacuum-field
contribution to the free-boundary force (plus `only_coils` for
verification):

```json
{
  "free_boundary_method": "nestor",  // or "biest"
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
error. In-loop, BIEST reaches a lower converged
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
