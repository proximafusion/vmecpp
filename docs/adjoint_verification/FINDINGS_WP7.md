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

## Mercier-specific verification

The new shared `mercier` objective uses the full `DMerc` discriminant at fixed
normalized flux and requires `lbsubs=True`; the existing `well` objective
remains `DWell`. The targeted tests cover `DMerc`, `DCurr`, `DGeod`, and `DWell`
on two three-dimensional, prescribed-current configurations, including squared
interior objectives rather than volume or boundary-only quantities.

Fixed-state checks isolate output-stage differentiation from equilibrium
reference noise. The first completed run reported eight passes and five
Enzyme-only skips. Native values agree term-by-term to at most `3.56e-10` on
each term's own scale. Resolved directional central differences agree to
`2.25e-7`; nonlinear reverse derivatives agree with Richardson-extrapolated
differences to `1.31e-7`. Reverse and forward AD agree at a `1e-10` tolerance.
These checks validate the tested successful `lbsubs` solves, not all-case
equilibrium sensitivities.

### Rejected singular solve

A separate mixed-surface regression reproduces a finite metric fallback whose
reverse derivative was NaN. Masking the singular solve's result with
`jnp.where` did not prevent its transpose solve from entering reverse AD.
The output stage now determines acceptance without differentiating that probe,
then differentiates the solve only on accepted surfaces. Rejected surfaces
retain the metric field and its derivatives. The successful-solve definition,
residual threshold, and native fallback values are unchanged.

This is a JAX fallback-derivative bug, not a demonstrated failure of every
converged Mercier gradient. A fallback still means the force-balance field was
not obtained on that surface; finite derivatives do not establish Mercier
reliability there. The second-order correction is local to a smooth branch and
is not certified across a change of solve-acceptance branch.

After the fix, the mixed-surface regression retains forward value `62` and
returns gradient `148` instead of NaN, matching analytic, central-difference,
and forward-AD checks. The expanded fixed-state/native-parity suite reports
11 passes and five Enzyme-only skips, including the existing Solovev and CTH
`lbsubs` output comparisons. Direct Pyright reports zero errors; other
pre-commit hooks pass. Successful surfaces use a detached acceptance probe
followed by a differentiable solve, adding a solve to that path; no runtime
cost claim is made.

## Remaining native acceptance

The end-to-end tests independently polish perturbed equilibria and compare
nonlinear Mercier gradients at multiple FD steps. A separate Option B test
compares full boundary gradients and corrected values against a polished
reference, recording raw residual progression. These tests are Enzyme-gated;
adding them does not establish that their acceptance criteria have passed.

The runtime restoration did not produce a verified Enzyme-enabled editable
extension. The loaded extension still reported both native capability flags
as false. Overlapping restoration jobs invalidated temporary build artifacts;
restoration was stopped rather than treating configuration flags or partial
libraries as a working exact-derivative runtime. No build configuration or
dependency-version change is included in the final source changes.

Consequently, native Mercier implicit gradients, Option B improvement on
Mercier, N8 slopes on CTH/li383, and a default residual/cost recommendation
remain unverified. The measured quadratic correction slopes above are
nonlinear algebraic-oracle results only. This module is not integrated into
the public JAX solver by default.

## Objective-aware default solver integration

The solver returned by `autodiff.make_solver(input)` now exposes
`corrected_value_and_grad(objective, boundary, directional_step=1e-3)`,
returning the corrected scalar value, dense boundary gradient, and
`ImplicitVjpDiagnostics`. This native objective-level method uses Option B,
including reevaluation of the objective state gradient at displaced states.
It performs a fresh solve without consuming the ordinary JAX forward cache.

The scalar objective must depend on the boundary through the supplied
Geometry only, with profiles and objective parameters held fixed. For Mercier,
use the same fixed `VmecInput` with `lbsubs=True` when evaluating
`autodiff_wout.wout_quantities`. The method requires JAX x64 and Enzyme.
Profile overrides, JIT tracing, and higher-order differentiation are rejected.

The ordinary `solver(boundary, profiles)` and its generic JAX VJP are unchanged.
A Geometry cotangent cannot recover the Hessian of an arbitrary nonlinear
downstream objective, so enabling Option B in that generic VJP would silently
drop objective curvature. Correction therefore belongs at the objective-level
entry point, not in a frozen-cotangent Geometry pullback.

This integration retains the final native convergence requirement and is
experimental. It does not automatically loosen input tolerances or claim that
the correction improves every objective. The earlier native-validation limits
above describe the previous runtime; new Enzyme measurements are reported
separately below when available.

## Reproducible native experiment

`scripts/plot_mercier_corrections.py` measures all four normalized, nonlinear
interior objectives (`DMerc`, `DCurr`, `DGeod`, `DWell`) with fixed reference
scales. It records independent plus/minus equilibrium differences at
`h = (1e-3, 3e-4, 1e-4)`, exact implicit gradients, raw residual histories,
and Option B errors at two loose states separated by a Newton step.
Each completed stage is saved before the next measurement.

The `plot_08_mercier_corrections_CASE` PNG/NPZ pairs retain input, binary,
script, and runtime provenance, timings, solve outcomes, and derivative
acceptance statuses. A failure or unresolved plateau is recorded as such;
the experiment does not replace the existing acceptance criteria with a
favorable FD step. Gradient/value improvement is checked against the same
polished reference and fixed objective definition.

The continuation restored one Enzyme v0.0.264 plugin with Clang/LLVM 18 and
rebuilt the editable extension once. Both `_vmecpp.VMECPP_ENABLE_ENZYME` and
`vmecpp.has_exact_force_jacobian()` now report `True`. Existing fetched
dependency sources were retained; no dependency versions or tracked build
configuration were changed. Unlike the earlier failed restoration, these
flags were checked on the extension actually imported by Python.

### Small CTH end-to-end Mercier derivatives

The initial Enzyme run reported 9 passes, 4 unresolved FD plateaus, and 2
failures. One failure was a state-direction FD truncation error: the normalized
squared `DMerc` objective has a narrow linear neighborhood. At a state step
of `1e-6`, the FD derivative was `2.080859e6` versus AD `6.105289e4`.
Reducing the step gave relative errors `3.285e-3` at `1e-8`, `3.286e-5` at
`1e-9`, `3.204e-7` at `1e-10`, and `7.322e-8` at `3e-11`. The derivative
test now requires two adjacent resolved steps at the original `1e-6`
relative target; no tolerance was enlarged.

The four initial independent-equilibrium plateaus were also truncation-limited
at boundary steps `1e-3`, `3e-4`, and `1e-4`. Using independent polished solves
at `1e-5`, `3e-6`, and `1e-6` resolves all four nonlinear objectives:

| Term | Relative error at 3e-6 | Relative error at 1e-6 |
| --- | --- | --- |
| DMerc | 6.848e-8 | 5.626e-8 |
| DCurr | 1.363e-8 | 4.459e-8 |
| DGeod | 6.681e-7 | 2.126e-7 |
| DWell | 7.555e-9 | 4.226e-9 |

All seven raw residuals were below `5.699e-15`. The five refined state/implicit
tests passed, with no plateau skips. This establishes the tested small CTH
direction, not all-case or all-mode Mercier gradient acceptance.

### Native corrections are not uniformly improving

At native `ftol=1e-8`, small CTH's active raw residual was `7.102150e-6`.
The following errors are measured against the same polished reference and
fixed normalized objective:

| Term | Naive relative gradient error | Option B error | Naive absolute value error | Corrected value error |
| --- | --- | --- | --- | --- |
| DMerc | 6.59159 | 20.65823 | 9.800409e-2 | 3.192870e-1 |
| DCurr | 1.665170e-1 | 2.267638e-2 | 2.277580e-4 | 1.574573e-3 |
| DGeod | 608.53345 | 1894.46663 | 80.33339 | 245.78438 |
| DWell | 5.491075e-2 | 1.192675e-3 | 9.600035e-3 | 3.229019e-4 |

After one Newton step, the residual was `6.970675e-6`. `DWell` still improved
both quantities, and `DCurr` now improved both; `DMerc` and `DGeod` did not.
Residual norm alone therefore does not certify a local correction domain.
These are real correction-improvement failures, not failed Enzyme builds or
an excuse to enable the correction unconditionally.

The first original-resolution li383 reference reached `2.027140e-12` after
five Newton steps, above the `1e-12` requirement. Its checkpoint records this
failure and leaves derivative/correction entries pending. No li383 Mercier
acceptance follows from those missing measurements.

### Measured local improvement at tighter native tolerance

A second experiment preserves the original failure data and writes to
`results/mercier_tight`. Reproduce it with
`scripts/plot_mercier_corrections.py --cases small li383_low_res --steps 1e-5 3e-6
1e-6 --loose-ftol 1e-12 --max-steps 10 --output-dir results/mercier_tight`.
The FD and improvement criteria are unchanged.

Small CTH passed all four FD plateaus, gradient/value improvements at both
states, and the fixed-scale public `DMerc` API parity check:

| Term | Gradient error at F=6.801e-8, naive -> B | Gradient error at F=2.047e-10, naive -> B |
| --- | --- | --- |
| DMerc | 2.351881e-3 -> 4.149224e-4 | 6.352752e-7 -> 2.647604e-10 |
| DCurr | 2.279858e-3 -> 1.555997e-5 | 4.245480e-6 -> 2.418700e-11 |
| DGeod | 1.741149e-1 -> 1.028488e-2 | 2.167902e-5 -> 2.131566e-8 |
| DWell | 2.278713e-4 -> 6.690874e-8 | 1.253529e-7 -> 4.317835e-13 |

`DMerc` absolute value errors changed from `1.063907e-4` to `4.781394e-6`
at the first state, and from `3.551264e-7` to `5.428991e-14` at the second.
All other terms' value errors also decreased. A final Newton step reduced
the raw force to about `1.6e-15`. Two states establish improvement here, not
the requested all-configuration N8 slopes or a universal residual threshold.

The small experiment took 28.85 seconds. Warm naive VJPs took 0.094--0.188
seconds, and warm Option B VJPs took 0.282--0.697 seconds per objective/state.
The first corrected `DMerc` call took 8.32 seconds, including JAX compilation.
These are local wall times, not a solver-wide performance claim.

At original li383 resolution, ten steps still left `1.767805e-12` above the
same `1e-12` reference requirement. That case remains blocked; increasing the
iteration budget did not establish the required reference.

The full Enzyme Mercier suite now passes all 15 tests. FD steps were refined
to resolve truncation error, and the improvement regression starts at the
measured local domain `ftol=1e-12`; the failing `ftol=1e-8` experiment remains
saved and documented above. Separate public API regressions report 39 passes,
one existing non-Enzyme-only skip, and six deselections, including native
interior-R and nonlinear `DWell` parity. Six plot/precision/CLI tests passed.

## Default integration conclusion

Ship only the explicitly invoked, experimental objective-aware method on the
default solver object. Keep the generic JAX Geometry solve and its tolerance
schedule unchanged. The loose-state counterexamples and blocked li383
reference do not justify making residual corrections or relaxed inner solves
automatic. Users must validate their objective and convergence domain; raw
residual diagnostics do not supply an all-case error guarantee.

Formatting hooks and explicit-interpreter Pyright checks passed. Secret scans
reported no secrets. Read-only final review found no significant issues after
fixing precision and source-provenance checks. The automated review executable
was unavailable, and the final CodeQL scan timed out; it is not reported as a
completed security scan.
