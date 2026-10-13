# Troubleshooting and Debugging Common Failures

VMEC++ solves the ideal-MHD force balance equations using a steepest-descent iterative method across multiple radial grid levels (multigrid). While robust for smooth configurations, non-linear force balance can occasionally fail to converge or stop due to numerical errors.

This guide explains the most common failure modes and gives practical steps to diagnose and fix them.

## Summary Checklist

| Symptom | Primary Cause | Recommended Fix |
|---|---|---|
| Force residual stalls or oscillates | Step size `delt` too large | Reduce `delt` (e.g. from 1.0 to 0.5 or 0.2) and increase iterations `niter_array` |
| Free boundary fails on shaped boundary | Ill-posedness near islands or separatrix | Reduce `phiedge` and truncate profiles to solve for an inner flux surface |
| Non-positive Jacobian ($J \le 0$) | Boundary ripples or bad axis initial guess | Spectrally condense boundary; check `raxis_c`/`zaxis_s`; reduce `delt` |
| Inaccurate Mercier criterion (`DMerc`) | Default full-grid $B_s$ interpolation | Enable `lbsubs = True` to re-derive $B_s$ from radial force balance |
| Slow convergence on weakly shaped boundary | Overly conservative damping | Set `lambda_preconditioner_scale = 1.0` and use `delt = 1.0` |

---

## 1. Solver Did Not Converge (Force Residual Stalled)

### Symptom
VMEC++ reaches the maximum number of iterations (`niter_array`) on a multigrid level without reaching the target force residual tolerance (`ftol_array`), raising an exception (e.g. `RuntimeError: Equilibrium solver did not reach target force tolerance`).

### Diagnosis
In steepest descent, the spatial coordinates are updated at each iteration using an artificial time step:

$$
\mathbf{x}^{k+1} = \mathbf{x}^k + \Delta t \, \mathbf{F}^k
$$

If $\Delta t$ (`delt`) is too large, the solver can overshoot the energy minimum, leading to numerical oscillations where the force residual oscillates between iterations instead of decaying.

### Remedy: Reduce `delt` to Trade Speed for Robustness
Reduce `delt` to take smaller, more stable steps:

```python
import vmecpp

vmec_input = vmecpp.VmecInput.from_file("input.example")

# Default delt is usually 1.0; reduce it to damp oscillations
vmec_input.delt = 0.5  # or 0.2, 0.1 for very stiff cases

# Because smaller steps take longer to reach the minimum,
# increase the maximum iteration limit
vmec_input.niter_array = vmec_input.niter_array * 2

output = vmecpp.run(vmec_input)
```

Lowering `delt` trades slower convergence for greater stability.

### Inspecting Non-Converged States
To inspect the state of the equilibrium when it fails to converge, set `return_outputs_even_if_not_converged=True`:

```python
output = vmecpp.run(vmec_input, return_outputs_even_if_not_converged=True)

print(f"Force residual reached: {output.wout.fsqr}")
print(f"Iterations completed: {output.wout.itfsq}")

# You can plot flux surfaces or inspect magnetic axis positions to see where the solve stalled
```

---

## 2. Free-Boundary Divergence on Shaped Boundaries

### Symptom
A free-boundary calculation (`lfreeb = True`) diverges, oscillates wildy at the edge, or produces a non-positive Jacobian at the outermost surfaces.

### Physics Cause: Ill-Posedness Near Islands and Separatrices
VMEC assumes that magnetic flux surfaces are everywhere smooth, topologically nested tori ($s \in [0, 1]$) with no magnetic islands or stochastic field lines.

In free-boundary mode, the vacuum magnetic field outside the plasma is computed using Green's function surface integrals (NESTOR or BIEST) applied directly on the outermost plasma surface ($s = 1$). If the plasma boundary extends close to:
- A magnetic separatrix or divertor X-point (where poloidal field vanishes and angles become singular), or
- Vacuum magnetic island chains or chaotic field regions,

the free-boundary force balance equation becomes mathematically ill-posed. Fourier series struggle to capture sharp features (Gibbs phenomenon), and coordinate surfaces can overlap.

### Remedy: Truncate Flux and Profiles to an Inner Surface
Instead of trying to solve all the way to a sharp, problematic boundary, solve for a slightly smaller, smoother inner surface:

1. **Reduce `phiedge`**: Lower the total enclosed toroidal magnetic flux by 10% to 20% (e.g. `vmec_input.phiedge *= 0.85`).
2. **Truncate profiles**: Adjust the pressure profile (`am`) and rotational transform or current profiles so that $s = 1$ in the new input corresponds to the truncated surface of the original configuration.
3. **Re-run**: Because inner surfaces are smoother, better nested, and isolated from separatrix or island distortions, the free-boundary solver converges reliably.

```python
# Shrink boundary to 85% flux surface
flux_fraction = 0.85
vmec_input.phiedge *= flux_fraction

# Scale pressure accordingly
vmec_input.pres_scale *= (1.0 - flux_fraction)

output = vmecpp.run(vmec_input)
```

In addition, if vacuum boundary updates cause abrupt force shocks, reduce `nvacskip` (e.g. `vmec_input.nvacskip = 1` or `2`) so the vacuum field adjusts smoothly every iteration.

---

## 3. Non-Positive Jacobian ($J \le 0$)

### Symptom
VMEC++ aborts with an error indicating a non-positive Jacobian on one of the radial surfaces.

### Diagnosis
The Jacobian $J = \sqrt{g} = (\nabla s \cdot \nabla \theta \times \nabla \zeta)^{-1}$ measures the coordinate volume element. A non-positive Jacobian ($J \le 0$) means that coordinate lines are crossing, self-intersecting, or folded over, which makes the coordinate system invalid.

Common causes and fixes:

1. **Boundary is not spectrally condensed**:
   High-order poloidal harmonics ($m \gg 1$) on the boundary that do not correspond to true physical shape create artificial wrinkles in $\theta$.
   - **Fix**: Run spectral condensation on the boundary using SIMSOPT (`simsopt.geo.surfacerzfourier.condense_spectrum`) before running VMEC++.
2. **Initial magnetic axis guess is poor**:
   The initial axis guess (`raxis_c`, `zaxis_s`) must lie well inside the boundary at all toroidal angles. If the axis starts near or outside the boundary, grid surfaces immediately fold over on step 1.
   - **Fix**: Check that `raxis_c[0]` matches the average major radius of the boundary cross-section and that `zaxis_s[0]` matches the vertical center.
3. **Step size too large in early iterations**:
   A large `delt` can launch the grid coordinates across each other during the first few steps before the metric tensor stabilizes.
   - **Fix**: Lower `delt` to `0.2` or `0.5`.

---

## 4. Current-Related Properties and Mercier Stability (`DMerc`)

### Symptom
The calculated Mercier stability criterion `DMerc`, magnetic shear, or current density profile shows unphysical spikes or does not match analytical expectations.

### Fix: Set `lbsubs = True`
By default (`lbsubs = False`), the full-grid covariant radial magnetic field component $B_s$ is calculated by averaging values from adjacent half-grid points. This simple interpolation neglects radial equilibrium constraints and can introduce errors in radial derivatives.

Setting `lbsubs = True` instructs VMEC++ to compute the full-grid $B_s$ by inverting the radial force balance equation in Fourier space:

$$
vmec\_input.lbsubs = True
$$

Always enable `lbsubs = True` when analyzing Mercier stability or current profiles.

---

## 5. Slow Convergence on Weakly Shaped Configurations (QA)

### Symptom
A quasi-axisymmetric (QA) stellarator or tokamak equilibrium converges very slowly, taking thousands of iterations even on coarse grids.

### Fix: Relax Damping and Preconditioning
VMEC includes conservative damping intended for strongly shaped, non-axisymmetric 3D stellarators (such as W7-X). For weakly shaped or QA boundaries, the mode coupling is weak and the problem is much less stiff.

You can relax the damping to speed up convergence:
- Set `lambda_preconditioner_scale = 1.0` (default is `0.5`).
- Keep `delt = 1.0` (or slightly higher).

```python
# Accelerated convergence settings for weakly shaped / QA boundaries
vmec_input.lambda_preconditioner_scale = 1.0
vmec_input.delt = 1.0
```
