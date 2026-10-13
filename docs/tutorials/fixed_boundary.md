# Fixed-Boundary Equilibrium Tutorial

In a fixed-boundary equilibrium calculation, the outer plasma boundary shape is known and held fixed. You prescribe the outermost shape of the plasma using Fourier coefficients, along with radial profiles for pressure and rotational transform or toroidal current. VMEC++ then computes the nested flux surfaces inside that boundary that satisfy ideal magnetohydrodynamic (ideal-MHD) force balance:

$$
\mathbf{J} \times \mathbf{B} = \nabla p
$$

This tutorial walks through setting up and running a fixed-boundary case, followed by recommended settings for accuracy and convergence.

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

If you plan to inspect current-related quantities, such as the Mercier stability criterion (`DMerc`), the magnetic shear term, or the current term (`Dcurr`), always set:

```python
vmec_input.lbsubs = True
```

**Why this matters:**
VMEC uses a staggered radial grid with half-grid and full-grid quantities. By default (`lbsubs = False`), the covariant radial magnetic field component $B_s$ on the full grid is calculated by averaging values from adjacent half-grid points. While this is sufficient for basic geometric quantities, simple interpolation introduces numerical inconsistencies when calculating radial derivatives of $B_s$, which directly enter the current density $\mathbf{J}$ and the Mercier criterion.

When `lbsubs = True`, VMEC++ re-derives the full-grid $B_s$ by inverting the radial force balance equation in Fourier space. This ensures that the magnetic field components and the radial current balance are consistent, yielding accurate and reliable `DMerc` values:

```python
vmec_input.lbsubs = True
output = vmecpp.run(vmec_input)

# Inspect Mercier stability results on the interior flux surfaces
dmerc = output.mercier.DMerc
print("Mercier criterion DMerc (positive means stable):", dmerc[1:-1])
```

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

### Relax Damping and Preconditioning for Weakly Shaped Boundaries (e.g. QA)

VMEC++ uses an artificial time-stepping scheme (steepest descent) to minimize energy and drive force residuals to zero. For strongly shaped stellarators, conservative damping is needed to prevent numerical instability.

However, for **weakly shaped boundaries**, such as quasi-axisymmetric (QA) stellarators or near-circular tokamaks:
- Poloidal and toroidal mode couplings are relatively weak.
- The numerical system is less stiff.

In these cases, you can relax the damping and preconditioning factors to achieve significantly faster convergence:

1. **`lambda_preconditioner_scale`**: This factor scales the step size for the magnetic stream function $\lambda$. The default value of `0.5` provides conservative damping inherited from VMEC 8.52. For weakly shaped or QA equilibria, you can relax damping by setting:
   ```python
   vmec_input.lambda_preconditioner_scale = 1.0  # Undamped step, faster convergence
   ```
2. **`delt`**: This is the artificial time step. For weakly shaped boundaries, the default `delt = 1.0` works well and can sometimes be kept near `1.0` or increased slightly without triggering bad Jacobian steps.
