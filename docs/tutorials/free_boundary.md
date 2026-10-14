# Free-Boundary Equilibrium Tutorial

In a free-boundary equilibrium calculation, the outer shape of the plasma is not fixed in advance. Instead, it is determined self-consistently by the balance between the internal plasma pressure and currents, and the external magnetic field created by magnetic coils.

Ideal-MHD force balance requires continuity of the total (magnetic plus kinetic) pressure across the plasma-vacuum boundary:

$$
p + \frac{B_{\text{plasma}}^2}{2 \mu_0} = \frac{B_{\text{vacuum}}^2}{2 \mu_0}
$$

The external magnetic field can come from an MGRID file or an in-memory magnetic field response table. VMEC++ solves the vacuum region using a Green's function surface integral method (such as NESTOR or BIEST).

## Related Examples

- [Run and hot-restart a free-boundary equilibrium](https://github.com/proximafusion/vmecpp/blob/main/examples/freeboundary_run_and_hot_restart.py)

## Basic Setup

To run in free-boundary mode, set `lfreeb = True` and provide the external magnetic field specification.

### Typical Command-Line Workflow

Compute the MGRID file from a coils file, then run VMEC++ on an input file that references it:

```console
# writes mgrid_<name>.nc for the coils file coils.<name>
vmecpp mgrid makegrid_parameters.json coils.<name>

# the input's "mgrid_file" field must point to mgrid_<name>.nc
vmecpp input.<name>.json
```

`mgrid` is an alias of `makegrid`. `makegrid_parameters.json` defines the grid (see `src/vmecpp/cpp/makegrid/example_inputs/makegrid_parameters.json`).

### 1. Free-Boundary with an MGRID File

The standard method uses a precomputed MGRID file storing the vacuum magnetic field on a 3D cylindrical grid $(R, \phi, Z)$:

```python
import numpy as np
import vmecpp

# Load base configuration
vmec_input = vmecpp.VmecInput.from_file("examples/data/solovev.json")

# Enable free-boundary mode
vmec_input.lfreeb = True

# Path to the MGRID file containing the coil field grid
vmec_input.mgrid_file = "examples/data/mgrid_solovev.nc"

# Vector of currents flowing through each coil group in the MGRID file (in Amperes)
# The length of extcur must match the number of coil groups defined in the MGRID file
vmec_input.extcur = np.array([1.0e4])

# Number of iterations between vacuum boundary updates (default: 6)
# Updating the vacuum field every nvacskip iterations saves compute time
vmec_input.nvacskip = 6

# Choose the vacuum solver method ("NESTOR" or "BIEST")
vmec_input.free_boundary_method = "NESTOR"

# Run the free-boundary solver
output = vmecpp.run(vmec_input)

print(f"Solved in {output.wout.itfsq} iterations.")
print(f"Final major radius: {output.wout.rmnc[0, output.wout.ns - 1]:.3f} m")
```

### 2. Free-Boundary with In-Memory Response Tables

VMEC++ also supports running free-boundary calculations using an in-memory `MagneticFieldResponseTable` computed directly from coil geometry (such as Makegrid-format coils files):

```python
import numpy as np
import vmecpp
from vmecpp import MagneticFieldResponseTable

# Load input
vmec_input = vmecpp.VmecInput.from_file("examples/data/solovev.json")
vmec_input.lfreeb = True

# Generate response table directly from a coils file
response_table = MagneticFieldResponseTable.from_coils_file(
    coils_file="examples/data/coils.solovev",
    r_min=0.5,
    r_max=1.5,
    z_min=-0.5,
    z_max=0.5,
    num_r=33,
    num_z=33,
    num_phi=18,
    num_field_periods=vmec_input.nfp,
    is_stellarator_symmetric=True,
)

# Pass the response table directly to run()
output = vmecpp.run(
    vmec_input,
    magnetic_field_response_table=response_table,
)
```

## Physics Limitations and Practical Guidelines

### Why Free-Boundary Becomes Ill-Posed Near Sharp Features and Islands

VMEC makes a fundamental mathematical assumption: **magnetic flux surfaces are smooth, nested tori without magnetic islands or stochastic field lines**, represented by a single-valued radial coordinate $s \in [0, 1]$.

At the plasma boundary:
1. **Sharp features and X-points:** Near a magnetic separatrix or divertor X-point, the poloidal magnetic field vanishes ($B_p \to 0$) and the field lines develop sharp corners. Fourier series converge very slowly at sharp corners (Gibbs phenomenon), and coordinate lines can intersect, causing non-positive Jacobian errors.
2. **Magnetic islands and stochasticity:** Real stellarator edge vacuum fields often contain magnetic island chains or ergodic zones. The vacuum surface integral solvers (NESTOR and BIEST) assume that the boundary surface is a clean topological boundary between plasma and vacuum. When island structures distort the field lines near the boundary, the free-boundary force balance equation can become ill-posed, resulting in severe convergence oscillations.

### Remedy: Truncating Flux and Profiles to an Inner Surface

When a free-boundary calculation fails to converge because of strong boundary shaping, islands, or proximity to a separatrix, a reliable strategy is to **truncate to an inner flux surface**:

1. **Reduce the total toroidal magnetic flux (`phiedge`)**: Set `phiedge` to a fraction (e.g. 80% to 90%) of the original boundary flux.
2. **Truncate the plasma profiles**: Rescale or truncate the pressure profile (`am`) and rotational transform or current profiles so they correspond to the inner surface.
3. **Re-run the free-boundary solver**: The inner surface is smoother, well-nested, and located away from the problematic edge features, allowing the free-boundary iteration to converge smoothly.

```python
# Example: Truncating to an inner flux surface for robustness
fraction = 0.85

# Reduce the enclosed toroidal flux (in Weber)
vmec_input.phiedge *= fraction

# Truncate / scale the pressure profile to match the inner surface
# Here, if pressure was p(s) = p0 * (1 - s), adjust am coefficients
# so the new boundary (s=1) carries the pressure of the original s = fraction surface
vmec_input.pres_scale *= 1.0 - fraction  # or evaluate original p(s) on new s-grid

# Run the truncated free-boundary problem
output = vmecpp.run(vmec_input)
```

This technique allows you to capture the plasma equilibrium geometry reliably without numerical breakdown at the edge.
