# Free-Boundary Equilibrium Tutorial

In a free-boundary equilibrium calculation, the outer shape of the plasma is not fixed in advance. Instead, it is determined self-consistently by the balance between the internal plasma pressure and currents, and the external magnetic field created by magnetic coils.

Ideal-MHD force balance requires continuity of the total (magnetic plus kinetic) pressure across the plasma-vacuum boundary:

$$
p + \frac{B_{\text{plasma}}^2}{2 \mu_0} = \frac{B_{\text{vacuum}}^2}{2 \mu_0}
$$

The external magnetic field can come from an MGRID file or an in-memory magnetic field response table. VMEC++ solves the vacuum region using a Green's function surface integral method (such as NESTOR or BIEST).

## Related Examples

- [Run and hot-restart a free-boundary equilibrium](https://github.com/proximafusion/vmecpp/blob/main/examples/freeboundary_run_and_hot_restart.py)
- [Run a free-boundary equilibrium in an analytical magnetic field](https://github.com/proximafusion/vmecpp/blob/main/examples/free_boundary_analytical_field.py)

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
import vmecpp

test_data = "src/vmecpp/cpp/vmecpp/test_data"

# Load a free-boundary input; its mgrid_file is not used when a response table is passed
vmec_input = vmecpp.VmecInput.from_file(f"{test_data}/solovev_free_bdy.json")

# The grid of the response table: R, Z and phi extents and resolution, number of field periods
makegrid_parameters = vmecpp.MakegridParameters.from_file(
    f"{test_data}/makegrid_parameters_solovev.json"
)

# Generate the response table directly from a coils file
response_table = vmecpp.MagneticFieldResponseTable.from_coils_file(
    f"{test_data}/coils.solovev", makegrid_parameters
)

# Pass the response table directly to run(); it is weighted by vmec_input.extcur
output = vmecpp.run(vmec_input, magnetic_field=response_table)
```

### 3. Free-Boundary with a Magnetic Field Function

`magnetic_field` also accepts a function of position, `b_r, b_phi, b_z = f(r, phi, z)`, which VMEC++ evaluates at the points of the plasma boundary in every vacuum update, with no grid and no interpolation. `r`, `phi` and `z` are 1D arrays of cylindrical coordinates, `phi` the geometric toroidal angle, and the function returns the cylindrical components of the field in Tesla. `mgrid_file` and `extcur` are not used by VMEC++. Each vacuum thread calls the function on its own subset of the boundary points.

The external field of the Solov'ev free-boundary case is that of a straight wire along the $Z$ axis and of circular coils, which has a closed form:

```python
import numpy as np
from scipy.special import ellipe, ellipk
import vmecpp

MU0 = 4.0e-7 * np.pi
vmec_input = vmecpp.VmecInput.from_file(
    "src/vmecpp/cpp/vmecpp/test_data/solovev_free_bdy.json"
)
# (radius, height) of the circular coils, which carry extcur[1:]
loops = np.array([[0.867845, 1.86006], [4.49628, 3.69583], ...])


def magnetic_field(r, phi, z):
    a = loops[:, 0, None]
    dz = z - loops[:, 1, None]
    alpha2 = a**2 + r**2 + dz**2 - 2 * a * r
    beta2 = a**2 + r**2 + dz**2 + 2 * a * r
    m = 1.0 - alpha2 / beta2
    scale = MU0 * vmec_input.extcur[1:, None] / (2 * np.pi * alpha2 * np.sqrt(beta2))
    b_r = scale * dz / r * ((a**2 + r**2 + dz**2) * ellipe(m) - alpha2 * ellipk(m))
    b_z = scale * ((a**2 - r**2 - dz**2) * ellipe(m) + alpha2 * ellipk(m))
    # the wire along the Z axis carries extcur[0]
    b_phi = MU0 * vmec_input.extcur[0] / (2 * np.pi * r)
    return b_r.sum(axis=0), b_phi, b_z.sum(axis=0)


output = vmecpp.run(vmec_input, magnetic_field=magnetic_field)
```

The complete script, which also compares the result with the mgrid file of the same coils, is [free_boundary_analytical_field.py](https://github.com/proximafusion/vmecpp/blob/main/examples/free_boundary_analytical_field.py).

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
