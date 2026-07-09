# VMEC Solver Control Flow (`vmec/vmec/`)

This directory contains the top-level solver driver (`vmec.cc` / `vmec.h`) and the flow
control state (`flow_control.h`). It orchestrates the iterative equilibrium solve; the actual
physics (forces, geometry, transforms) lives in `../ideal_mhd_model/`.

## Control flow

The solver is driven by `Vmec::run()` (`vmec.cc`), structured as nested loops around a
fixed-point force-balance solve:

- **Multigrid (outer loop)**: VMEC++ solves on a sequence of progressively finer radial grids
  from `indata_.ns_array` (with matching `ftol_array`, `niter_array`). For each grid step
  `igrid`, `run()` sets the resolution `fc_.nsval`, interpolates the converged coarse-grid
  solution onto the finer grid via `InterpolateToNextMultigridStep()` (linear radial
  interpolation of the spectral R/Z/lambda coefficients; odd-`m` modes extrapolated to the
  axis), then calls `SolveEquilibrium()`. An outer retry over `jacob_off_` reruns the whole
  sequence if the initial Jacobian is bad. This is multigrid only in the sense of
  grid-sequencing (a good coarse-grid guess seeds the fine grid) -- it is **not** a V-cycle
  multigrid solver.

- **Inner solve**: `SolveEquilibrium()` -> `SolveEquilibriumLoop()` repeatedly calls
  `Vmec::Evolve()` until convergence or `niterv` is exceeded.

## Descent algorithm

The inner solve is **not** Newton -- it is an accelerated (damped) first-order pseudo-time
descent: conjugate-gradient-without-line-search / second-order Richardson scheme (Garabedian).
Each step (`Evolve()` -> `PerformTimeStep()`/`performTimeStep()`) integrates a damped equation
of motion for the spectral coefficients `x` with velocity `v`:

- `v_new = fac * (b1 * v_old + delt * f)`, then `x += delt * v_new`
- `f` is the (preconditioned) MHD force, `delt` is the pseudo-time step (`delt0r`).
- Damping is adapted each step from the residual history: `b1 = 1 - dtau`,
  `fac = 1 / (1 + dtau)`, with `dtau` from a running average (`kNDamp = 10`) of `invTau_`.
- The `b1`-weighted momentum term is the acceleration; with no damping this reduces to plain
  steepest descent.

## Convergence

Reached when all three force residuals fall below the current stage tolerance:
`fsqr <= ftolv && fsqz <= ftolv && fsql <= ftolv` (radial, vertical, lambda forces), or when
the iteration count exceeds `niterv`. Residuals live in `FlowControl` (`flow_control.h`);
`fsq*1` are the preconditioned variants used for the damping average.

## Restart logic

`Vmec::RestartIteration()` (enum `RestartReason` in `flow_control.h`). When the iteration
misbehaves, the velocity `decomposed_v_` is zeroed and the state is rolled back to
`physical_x_backup_`, with the pseudo-time step reduced:

- `BAD_JACOBIAN` (overlapping flux surfaces): `delt0r *= 0.9`, increment `fc_.ijacob`.
- `BAD_PROGRESS` (residuals not decaying): `delt0r /= 1.03`.
- `NO_RESTART` (good path): back up the current state into `physical_x_backup_`.

Separately, **hot restart** seeds `run()` from a previously converged `HotRestartState`
(`wout` + `indata`) via `FourierGeometry::InitFromState()`, used for parameter scans.
The first element of `ns_array` must match the last `ns` of the restart state; subsequent
multigrid steps proceed normally via `InterpolateToNextMultigridStep()`.

## State variables (`Vmec`)

- `decomposed_x_` / `decomposed_v_` / `decomposed_f_`: spectral position / velocity /
  (preconditioned) force.
- `physical_x_backup_`: rollback snapshot for restarts.
- `iter1_` / `iter2_`: branch-point and total-evaluation iteration markers.

Output quantities are computed **after** convergence as a post-processing step, not during the
loop -- see `../output_quantities/` and `../ideal_mhd_model/AGENTS.md`.
