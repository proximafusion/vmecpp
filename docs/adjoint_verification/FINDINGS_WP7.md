# WP7: Option B and value correction

## API and objective contract

`vmecpp.autodiff_corrections.corrected_vjp` and the shared contract entry point
accept `(model, x_bar, "B", objective_state_gradient=callback,
directional_step=1e-3)`. They return `(boundary_gradient, value_correction,
ImplicitVjpDiagnostics)`. Add `value_correction` to the objective value.

The callback takes a flat scaled state and returns its full objective gradient,
including cotangents on prescribed boundary and pinned entries. The base
gradient must agree with `x_bar`. A fixed `x_bar` cannot determine a nonlinear
objective Hessian: Option B requires the callback and does not silently set
`J_xx` to zero. A constant callback is valid only for an actually linear
objective. All boundary dependence must enter through the state; separate
explicit objective dependence on boundary parameters is unsupported. Profiles
and other parameters remain fixed. Callbacks must not change model parameters.
Option A remains explicitly unsupported.

### Nonlinear Geometry and Mercier objectives

Use `geometry_objective_state_gradient(model, objective)` from
`vmecpp.autodiff_corrections` to construct the callback. `objective` must be a
JAX-differentiable scalar function of the supplied `geometry.Geometry`.
JAX x64 must be enabled when constructing and invoking the callback. The
helper raises a clear `ValueError` if `jax_enable_x64` is false, preventing
silent truncation of native doubles without changing global configuration.
The factory reevaluates `jax.value_and_grad`, packs the existing twelve
coefficient cotangents in their native ordering, and applies
`model.geometry_state_vjp`, including the poloidal-flux cotangent. Users do not
need to know or reproduce the scaled-state layout. It refreshes the geometry's
state-dependent profiles and restores caller state and raw force afterward.

For a manual callback, flatten each Geometry coefficient array in C-order
`(surface, m, n)` and concatenate in this exact order:
`r_cc, r_ss, r_sc, r_cs, z_sc, z_cs, z_cc, z_ss, lambda_sc, lambda_cs,
lambda_cc, lambda_ss`. Pass that contiguous vector as the first argument of
`model.geometry_state_vjp`; pass the full-grid `poloidal_flux` cotangent as its
second argument. Both cotangents come from the reevaluated JAX Geometry
gradient. Do not append fluxes to the coefficient vector or drop the
poloidal-flux cotangent. The toroidal-flux profile is prescribed and has no
state derivative.

For a nonlinear Mercier objective, close over the fixed `VmecInput` with
`lbsubs=True`; evaluate `autodiff_wout.wout_quantities(supplied_geometry, input)`
and form a scalar such as the weighted mean square of its interior `DMerc`
entries. Exclude the axis and boundary diagnostic entries. Construct the
callback with that objective, obtain `x_bar` by applying it to
`model.get_state()`, and pass the same callback as `objective_state_gradient`
to Option B. Evaluate the base scalar objective separately and add the returned
value correction. Do not reuse a cached wout or a frozen Mercier cotangent.

Toroidal flux, profiles, and input parameters are fixed. Nonzero poloidal-flux
cotangents for `ncurr=1` require Enzyme; they are not discarded. The helper does
not add derivatives with respect to input profiles or independent explicit
boundary parameters and does not make the correction callable JAX-traceable.
The verification-only `objective_state_cotangent` helper uses the same
JAX/native pullback recipe and additionally returns the scalar value.

## Implemented algebra

With `H = F_x` restricted to the structurally active entries:

```text
delta = -H^-1 F
lambda = H^-T J_x
L_x = J_x - F_x^T lambda
L_xx delta = [L_x(x + h delta) - L_x(x - h delta)] / (2 h)
lambda_dot = H^-T (L_xx delta)
g = boundary_pullback(L_x)
g_corrected = g + D_x g delta - boundary_pullback(F_x^T lambda_dot)
value_correction = -lambda^T F
```

The directional boundary derivative uses the same central differences with
`lambda` held fixed. Each pullback includes the boundary-dependent pinned m=1
gauge. Only active entries move along `delta`. The block factorization of
`H^T` is reused for both solve orientations and the second adjoint. Exact
products check all three solve residuals. The reported block condition estimate
is the minimum reciprocal 1-norm estimate of the factored Schur complements.

`directional_step` requests `h`. The actual displacement is
`max(h ||delta||_inf, eps^(1/3) max(1, ||x_active||_inf))`, along the normalized
Newton direction. Differences are scaled back by
`||delta||_inf / (2 displacement)`. This avoids rounded-to-zero perturbations
without imposing a correction floor as `delta` tends to zero. Nonzero tiny
directions still reevaluate the nonlinear objective; exactly zero directions
skip the two displaced objective evaluations. Increasing `directional_step`
can increase the displacement; reducing it cannot cross the numerical floor.

The saved raw residual is copied before probes overwrite force buffers. State
and raw force are restored on success and after derivative/callback failures.
Inputs, callback gradients, products, candidates, and outputs are checked for
finiteness. Symmetric fixed-boundary, pinned-gauge, exact-derivative models with
`lforbal=false` are required.

## Scope and verification

This is a local first-order residual correction with central-difference
truncation error, not a converged gradient or an N1 acceptance result.
Structural classes are random-probe estimates using the existing active-set
logic, not WP3 analytic classifications. Unrecognized pinned entries and
zero-column equations are rejected. Relabeling overlap is not measured.

The dense oracle has nonlinear forces, a nonsymmetric Jacobian, a nonlinear
objective, and a boundary-dependent pinned entry. It independently checks the
objective Hessian, both solve orientations, second-adjoint sign, gauge chain,
and value correction. Both value and boundary-gradient errors decline
quadratically near the manufactured root. Measured absolute errors for
displacements `e` from the root `(x, y) = (1, 1)`, with boundary and pinned
entries zero, are:

| e | Corrected value error | Corrected boundary-gradient error |
| --- | --- | --- |
| 0.02 | 4.612244e-5 | 1.975549e-4 |
| 0.01 | 1.139200e-5 | 5.052017e-5 |
| 0.005 | 2.830358e-6 | 1.277596e-5 |

The value test uses `(x, y) = (1 + e, 1 + 1.5 e)`; its measured log-log slopes
are `2.01745` and `2.00896`. The gradient test uses
`(x, y) = (1 + e, 1 + 2 e)`; its slopes are `1.96732` and `1.98343`.
These are algebraic nonlinear-oracle measurements, not native Mercier slopes.

A separate objective adds `exp(v^T x)` with
`v = (1.1, -0.5, 0.8, 0.3, 0)`, so its cotangent is not polynomial and the
central-difference truncation error is nonzero. Against its analytic Hessian
oracle, requested directional steps `h = (0.5, 0.25, 0.125)` produce corrected
gradient errors `(9.092994e-6, 2.273226e-6, 5.683050e-7)` and measured slopes
`2.000014` and `2.000004`. These steps are above the adaptive displacement
floor and demonstrate the central difference's quadratic step convergence.

Tests cover objective reevaluation errors, failed
products/assembly/solves, invalid candidates and inputs, zero residuals, and
state/raw-buffer restoration.

The nonlinear Geometry callback is tested against an analytic coefficient and
poloidal-flux oracle and a native coefficient-objective directional difference.
The full native small-model correction test is conditional on Enzyme availability. This
runtime reports `vmecpp.has_exact_force_jacobian() == False`; its Enzyme test
is skipped rather than represented as a native validation result. The editable
rebuild is disabled with `SKBUILD_EDITABLE_SKIP` set to the absolute build
directory to avoid concurrent rebuilds.

Validation: the corrections, shared-contract, and Newton regression tests
report 70 passed and 2 Enzyme-only skips, including the existing Option A
`NotImplementedError` regression. Empty active operators and empty Schur-block
condition estimates fail explicitly rather than calling `min([])`.
Existing pre-commit hooks pass
except the isolated Pyright hook, which cannot resolve dependencies because
the configured `.venv` is absent. Direct Pyright using the runtime interpreter
reports zero errors.
