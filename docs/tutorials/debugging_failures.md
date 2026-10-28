# Troubleshooting and Debugging Common Failures

VMEC++ solves the ideal-MHD force balance equations using a steepest-descent iterative method across multiple radial grid levels (multigrid). While robust for smooth configurations, non-linear force balance can occasionally fail to converge or stop due to numerical errors.

This guide explains the most common failure modes and gives practical steps to diagnose and fix them.

## Summary Checklist

We recommend setting `lbsubs=True` by default, especially for plasma currents
and quantities derived from them. The input field still defaults to `False` for
backward compatibility, so set it explicitly. See the [lbsubs guidance](fixed_boundary.md#enable-lbsubs-for-current-related-properties-and-mercier-stability-dmerc)
for the tradeoff and derivative requirements. For current calculations, use a
final force tolerance below `1e-16`; values as low as `1e-19` can be beneficial.

| Symptom | Possible Cause | Recommended Fix |
|---|---|---|
| Force residual stalls or oscillates | Step size `delt` too large | Reduce `delt` (e.g. from 1.0 to 0.5 or 0.2) and increase iterations `niter_array` |
| Free boundary fails on shaped boundary | Ill-posedness near islands or separatrix | Reduce `phiedge` and truncate profiles to solve for an inner flux surface |
| Non-positive Jacobian ($J \le 0$) | Boundary ripples or bad axis initial guess | Spectrally condense boundary; check `raxis_c`/`zaxis_s`; reduce `delt` |
| Large or growing current-density ring near the magnetic axis | Inconsistent $B_s$ reconstruction or plain odd-m radial differences | Enable `lbsubs=True`; use exported currents or the matching sqrt(s)-regularized derivative |
| Unphysical spikes in reconstructed current-based diagnostics | Downstream reconstruction uses different grids, derivatives, or units | Enable `lbsubs=True`; compare with `currumnc`/`currvmnc` and their sine partners; check Jacobian interpolation and the included $1/\mu_0$ factor |
| Sharp spikes or banding in parallel or Pfirsch-Schlueter currents | Inaccurate radial force balance or poorly resolved structure near resonant surfaces | Enable `lbsubs=True`; inspect signed angular profiles and check radial/angular resolution and force tolerance |
| Alternating bands or checkerboard patterns in current cross-sections | Insufficient force convergence or radial/angular resolution | Reduce the final force tolerance below `1e-16`, trying `1e-19`; inspect signed distributions under refinement |
| Unphysical spikes or strong grid dependence in the Mercier criterion (`DMerc`) | Errors in currents and radial derivatives, including near-axis cancellation | Enable `lbsubs=True` with matching derivatives; check convergence of currents and individual Mercier terms |

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

Plot the force-residual components and restart locations with [`examples/force_residual_convergence.py`](../../examples/force_residual_convergence.py). For per-iteration diagnostics, including the time-step and restart behavior, see [`examples/iteration_dynamics.py`](../../examples/iteration_dynamics.py). Adapt these examples to inspect the run that failed.

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
Current distributions show unphysical spikes, a current-density ring that grows
near the magnetic axis as `ns` increases, or alternating bands across flux
surfaces. These features can appear in parallel and Pfirsch-Schlueter currents,
current-based diagnostic reconstruction, and the Mercier criterion `DMerc`.

### Check the current reconstruction and convergence

Set `lbsubs=True` as the recommended default. It re-solves the full-grid
covariant component $B_s$ from radial force balance. This can reduce current errors caused by interpolation, but does not
improve the equilibrium iteration or guarantee accurate `DMerc`. It trades
agreement with the geometry-based magnetic field for agreement with the discrete
radial force-balance equation. See the [lbsubs discussion](fixed_boundary.md#enable-lbsubs-for-current-related-properties-and-mercier-stability-dmerc)
for the equations and usage.

- Use a final force tolerance below `1e-16` for current calculations; `1e-19`
  can be beneficial. Radial differentiation amplifies residual errors in the
  magnetic field. Check that the run reaches the requested tolerance and that
  the current distribution stabilizes as the tolerance is tightened.
- Compare `lbsubs=False` and `True` using the same equilibrium inputs.
- Use the exported current fields `currumnc`, `currvmnc`, and their sine partners
  when present. If rebuilding currents from $B_s$, $B_u$, and $B_v$, use the
  sqrt(s)-regularized odd-m radial derivative described in the `currumnc`
  docstring. Plain radial differences can create a false axis current whose
  amplitude grows as $h^{-1/2}$, where $h=1/(ns-1)$.
- Check signed current profiles and cross-sections as well as maxima. Increase
  radial and angular resolution and tighten the force tolerance to see whether
  the features converge. A smaller axis spike alone does not establish accuracy.
- Check C++ warnings: a failed lbsubs surface solve keeps the geometry-based
  $B_s$ on that surface. The JAX output stage currently uses this fallback
  silently.

The geometry and rotational transform are unchanged by `lbsubs`. Diagnose an
incorrect magnetic shear through the equilibrium and its radial resolution,
rather than expecting the $B_s$ re-solve to change the rotational transform.
