# Fixed-Boundary Equilibrium Tutorial

In a fixed-boundary equilibrium calculation, the outer plasma boundary shape is known and held fixed. You prescribe the outermost shape of the plasma using Fourier coefficients, along with radial profiles for pressure and rotational transform or toroidal current. VMEC++ then computes the nested flux surfaces inside that boundary that satisfy ideal magnetohydrodynamic (ideal-MHD) force balance:

$$
\mathbf{J} \times \mathbf{B} = \nabla p
$$

This tutorial walks through setting up and running a fixed-boundary case, followed by recommended settings for accuracy and convergence.

## Related Examples

- [Run VMEC++ from an input file](https://github.com/proximafusion/vmecpp/blob/main/examples/python_api.py)
- [Build a `VmecInput` in Python](https://github.com/proximafusion/vmecpp/blob/main/examples/python_api_input.py)
- [Compare current-density results with `lbsubs`](https://github.com/proximafusion/vmecpp/blob/main/examples/w7x_lbsubs_current_density.py)

## Basic Workflow

### 1. Running from an Input File

The most common way to run a fixed-boundary equilibrium is from a classic Fortran `input.*` file or a VMEC++ JSON file:

```python
import vmecpp

# Load input configuration from file
vmec_input = vmecpp.VmecInput.from_file("examples/data/w7x.json")

# Ensure fixed-boundary mode is active (default is False)
vmec_input.lfreeb = False

# Run the solver
output = vmecpp.run(vmec_input)

# Check convergence and inspect results
print(f"Force residual reached: {output.wout.fsqr}")
print(f"Plasma volume: {output.wout.volume:.4f} m^3")

# Save the standard NetCDF output file
output.wout.save("wout_w7x.nc")
```

### 2. Building an Input Object Programmatically

You can also construct or modify the `VmecInput` object directly in Python:

```python
import numpy as np
import vmecpp

# Create input with basic grid sizes and profiles
vmec_input = vmecpp.VmecInput(
    nfp=1,  # Number of field periods (e.g. 1 for tokamak)
    mpol=6,  # Number of poloidal Fourier modes
    ntor=0,  # Number of toroidal Fourier modes (0 for axisymmetric)
    ns_array=np.array([17, 33, 65]),  # Multigrid radial resolutions
    niter_array=np.array([500, 1000, 2000]),  # Max iterations per grid
    ftol_array=np.array([1e-6, 1e-8, 1e-10]),  # Target force tolerance
    am=np.array([1.0, -1.0]),  # Pressure profile: p(s) = p0 * (1 - s)
    pres_scale=1.0e4,  # Pressure scaling factor in Pa
    ai=np.array([1.0, -0.5]),  # Iota profile: iota(s) = 1.0 - 0.5 * s
    lfreeb=False,  # Fixed boundary
)

# Set the boundary Fourier coefficients (R ~ cos, Z ~ sin)
vmec_input.rbc = vmecpp.VmecInput.resize_2d_coeff(
    np.array([[3.0], [1.0]]),  # Major radius R0=3.0, minor radius r=1.0
    vmec_input.mpol,
    vmec_input.ntor,
)
vmec_input.zbs = vmecpp.VmecInput.resize_2d_coeff(
    np.array([[0.0], [1.0]]),  # Circular cross-section
    vmec_input.mpol,
    vmec_input.ntor,
)

# Set the initial guess for the magnetic axis
vmec_input.raxis_c = np.array([3.0])
vmec_input.zaxis_s = np.array([0.0])

# Run VMEC++
output = vmecpp.run(vmec_input)
print(f"Solved successfully in {output.wout.itfsq} iterations.")
```

## Recommended Settings for Fixed-Boundary Runs

### Enable `lbsubs` for Current-Related Properties and Mercier Stability (`DMerc`)

We recommend setting `lbsubs=True` by default. It is relevant to current density,
parallel and Pfirsch-Schlueter currents, current-based diagnostic reconstruction,
and Mercier terms. The input field still defaults to `False` for backward
compatibility, so set it explicitly. Check that the quantities you use converge
with radial and angular resolution. Current calculations need a final force
tolerance below `1e-16`; `1e-19` can be beneficial because radial differentiation
amplifies residual errors in the magnetic field:

```python
vmec_input.lbsubs = True
vmec_input.ftol_array[-1] = min(vmec_input.ftol_array[-1], 1e-19)
output = vmecpp.run(vmec_input)
dmerc = output.mercier.DMerc
print("Mercier criterion on interior surfaces:", dmerc[1:-1])
```

Inspect signed current profiles and cross-sections for unphysical spikes,
near-axis divergence, and banding across flux surfaces. Tighten the force
tolerance and refine the radial and angular grids to check whether these
features persist. Check that the run actually reaches the requested tolerance.

#### What changes, and what is the tradeoff?

With `lbsubs=False`, VMEC++ computes the covariant radial component $B_s$ from
the geometry and magnetic field on the half grid, then averages it onto the full
grid. That interpolation can leave large errors when currents involve the
cancellation of radial derivatives of $B_u$ or $B_v$ against angular derivatives
of $B_s$. Ampere's law uses these angular derivatives of $B_s$, rather than its
radial derivative.

With `lbsubs=True`, VMEC++ instead solves for $B_s$ on each interior full-grid
surface using radial force balance, with the surface-averaged part removed.
The geometry, pressure, rotational transform, and tangential magnetic field
are unchanged. Both symmetric and asymmetric configurations use this solve.

The tradeoff is **force-balance consistency versus geometric consistency**.
For one magnetic field tangent to the flux surfaces, the geometry requires

$$
B_s = \mathbf{B}\cdot\frac{\partial\mathbf{x}}{\partial s}
    = g_{s\theta} B^\theta + g_{s\zeta} B^\zeta.
$$

The independently re-solved $B_s$ need not satisfy this identity with the
unchanged geometry and contravariant field at finite resolution. Thus the
post-processed components need not describe exactly the same magnetic field.
Currents still come from Ampere's law applied to the chosen covariant components;
this does not make them the exact curl of the unchanged geometry-based field.
The re-solve improves one numerical condition, rather than making all equilibrium
conditions exact. Agreement between these constructions must be checked under
refinement.

#### Use matching derivatives in downstream calculations

Prefer the exported current coefficients `currumnc`, `currvmnc`, and their sine
partners for asymmetric equilibria. They already include $1/\mu_0$; see the
`currumnc` docstring for reconstruction and Jacobian interpolation.

If rebuilding currents or other quantities from $B_s$, $B_u$, and $B_v$, use
VMEC++'s sqrt(s)-regularized odd-m radial derivative. For odd m, write
$Q(s)=\sqrt{s}\,q(s)$ and difference the smooth remainder $q$; even m uses the
ordinary radial difference. The `currumnc` docstring gives the exact formula and
half-grid indices. Pairing the re-solved $B_s$ with plain odd-m differences breaks
the near-axis cancellation and can create a false current growing as $h^{-1/2}$
with radial refinement. Matching the derivative is separate from the geometric
tradeoff above.

The $B_s$ re-solve can amplify divergence near rational surfaces, where
$m\iota-nN_{\rm fp}$ is small, but is typically the preferred treatment for
quantities related to current density.

### Use a Spectrally Condensed Boundary

The boundary of a plasma is defined by Fourier coefficients $R_{m,n}$ and $Z_{m,n}$. A given physical 3D shape can be parameterized by infinitely many poloidal angle coordinates $\theta$. If the angle coordinate is poorly chosen, the boundary representation becomes **spectrally dense**: it requires many high poloidal modes $m$ to represent simple geometry, creating artificial ripples in the coordinate lines.

A spectrally dense boundary can cause:
- Slower convergence or failure to converge.
- Non-positive Jacobian errors (coordinate surfaces crossing each other).
- Unphysical force residuals.

Always ensure the input boundary is **spectrally condensed** (its spectral width $\langle M \rangle$ is minimized). You can condense a boundary using SIMSOPT before running VMEC++:

```python
# Example: condensing boundary Fourier harmonics using SIMSOPT
import simsopt.geo.surfacerzfourier as srf

# Load surface and perform spectral condensation
surface = srf.SurfaceRZFourier.from_vmec_input("input.w7x")
condensed_surface = srf.condense_spectrum(surface)

# Use the condensed boundary coefficients in vmec_input
vmec_input.rbc = condensed_surface.rc
vmec_input.zbs = condensed_surface.zs
```

VMEC++ will print a warning if a fixed-boundary input has high spectral content ($\langle M \rangle > 3.0$).

### Reduce Iterations for Easy Equilibria

VMEC++ uses an artificial time-stepping scheme (steepest descent) to minimize energy and drive force residuals to zero. For strongly shaped stellarators, conservative damping is needed to prevent numerical instability.

Weakly shaped configurations, such as quasi-axisymmetric (QA) stellarators or near-circular tokamaks, generally converge reliably. For cases that can benefit from fewer iterations, try increasing `lambda_preconditioner_scale` from its default value of `0.5` to `1.0`:

```python
vmec_input.lambda_preconditioner_scale = 1.0
```

This scales the preconditioned step for the magnetic stream function $\lambda$. Keep the default `delt` unless the residual history supports changing it, and check that the run reaches the same force tolerance.
