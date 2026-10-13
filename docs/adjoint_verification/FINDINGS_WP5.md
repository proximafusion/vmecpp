# WP5: transpose solve and Newton prototype (partial)

## Implemented

`_BlockTridiagonal.solve_transpose` reuses the existing H^T block LU to solve
with H. Dense nonsymmetric tests cover uneven blocks, non-surface ordering,
row pivoting, and an empty intermediate surface.

`autodiff_newton.newton_polish` takes exact Newton steps on the current
probe-deflated interior, excluding prescribed boundary and gauge entries. It
uses WP1's raw force and a backtracking line search, restores the last accepted
state on failure, and rebuilds factors at the returned state. It requires exact
derivatives and the symmetric fixed-boundary, pinned-gauge model.

This is a standalone experimental implementation. The WP0 stub and production
JAX solve remain unchanged; no reference-equilibrium cache is published yet.

## N1 and dense-solve measurements

On the shared small CTH prescribed-current case, starting from the
VMEC-converged state:

| Accepted step | Active raw-force norm |
| --- | --- |
| 0 | 2.3227988e-8 |
| 1 | 9.4859488e-11 |
| 2 | 2.5262039e-15 |

The requested tolerance was `1e-14`; two accepted steps reached it. The
successive values of `F_next / F_current^2` are approximately `1.76e5` and
`2.81e5`, consistent with quadratic convergence over these two steps.
This is evidence on one small configuration, not N1 acceptance on all fixtures.

The final active set has 272 entries. For a seeded random right-hand side,
the H solve from the transposed block factors differs from the exact dense
H solve by `4.42255e-14` relative. Relative equation residuals are
`5.76195e-13` (block) and `5.25861e-13` (dense).

The native force-evaluation counters before polishing, after polishing, and
after dense assembly/restoration are `[416, 420, 423]`. They do not count every
exact local-kernel HVP as a full forward evaluation, and are not wall-time cost
measurements.

Reproduction: `scripts/plot_newton.py --output <directory>`.
`results/plot_01_newton_small.npz` includes the states, active H, right-hand side,
solutions, residual history, input JSON, and counters; the PNG plots the history.

## Validation and remaining acceptance

- Three parent-owned block-algebra tests pass against dense solves.
- All 25 owned Newton tests pass on the Enzyme build, including the native
  final-factorization solve check. Owned pre-commit hooks pass.
- N1 across the remaining configurations is not measured. No mismatch was
  observed in the small case; no force-component bisection was needed there.
- Plot 10 / the lagged-`tcon` fixed-point comparison is not measured.
- `reference_equilibrium` and its cache remain deferred until reference validity
  is established across the intended configurations.
- The active set still depends on current probe thresholds. WP3's analytic
  classes, including their scaled-state/gauge initialization checks, are not
  assumed to have passed.
