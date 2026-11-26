# Post-Processing `wout` Quantities

A VMEC++ run returns a `wout` object (`output.wout`, or `vmecpp.VmecWOut.from_wout_file("wout_x.nc")`) holding Fourier coefficients on a radial grid. This tutorial shows how to read the arrays, convert between Fourier and real space, and take radial derivatives, including the regularized ones.

The code was run on `examples/data/cth_like_fixed_bdy.json` (`nfp = 5`, `ns = 25`, `signgs = -1`).

For a larger example, see SIMSOPT's [`vmec_compute_geometry`](https://github.com/hiddenSymmetries/simsopt/blob/master/src/simsopt/mhd/vmec_diagnostics.py). It computes R, Z, metric terms, B components, gradients and drifts on any $(s, \theta, \zeta)$ grid. It fits cubic splines in `s` to the coefficients and does not treat odd-`m` modes specially (see {ref}`regularized-derivatives`).

Related examples in this repository:
[plot the plasma boundary](https://github.com/proximafusion/vmecpp/blob/main/examples/plot_plasma_boundary.py),
[reconstruct B and J, and view them with `pyvista`](https://github.com/proximafusion/vmecpp/blob/main/examples/visualize_magnetic_field.py),
[current density with `lbsubs`](https://github.com/proximafusion/vmecpp/blob/main/examples/w7x_lbsubs_current_density.py),
[force-residual spectrum](https://github.com/proximafusion/vmecpp/blob/main/examples/force_balance_spectrum.py).

## 1. Layout of the arrays

### Radial grids

The `ns` flux surfaces are labeled by the normalized toroidal flux $s \in [0, 1]$. With `h = 1 / (ns - 1)`:

- **Full grid:** `s_full[j] = j * h`, `j = 0 ... ns-1`. From the magnetic axis to the boundary.
- **Half grid:** `s_half[j] = (j - 0.5) * h`, `j = 1 ... ns-1`. Between full-grid surfaces. **Column 0 of every half-grid array is unused and holds zero.**

```python
import numpy as np
import vmecpp

vmec_input = vmecpp.VmecInput.from_file("examples/data/cth_like_fixed_bdy.json")
vmec_input.ns_array = np.array([25])
vmec_input.niter_array = np.array([3000])
vmec_input.ftol_array = np.array([1e-12])
wout = vmecpp.run(vmec_input).wout

ns = wout.ns
h = 1.0 / (ns - 1)
s_full = np.arange(ns) * h
s_half = (np.arange(ns) - 0.5) * h  # entry 0 is meaningless
```

The docstring of each `VmecWOut` field names its grid. For example:

| Grid | Fields |
|---|---|
| Full | `rmnc`, `zmns`, `lmns_full`, `bsubsmns`, `currumnc`, `currvmnc`, `iotaf`, `phipf`, `presf`, `jdotb` |
| Half | `lmns`, `gmnc`, `bmnc`, `bsupumnc`, `bsupvmnc`, `bsubumnc`, `bsubvmnc`, `iotas`, `phips`, `pres`, `buco`, `bvco` |

### Shapes and modes

Shapes are annotated with `jaxtyping` on each field, for example `rmnc: jt.Float[NpOrJax, "mn_mode n_surfaces"]`. The annotation does not distinguish full and half grid: `n_surfaces` is `ns` for both.

- Coefficient arrays are `(n_modes, ns)`, the transpose of the NetCDF layout that SIMSOPT reads.
- `mn_mode` (`mnmax`, modes `xm`, `xn`): `rmnc`, `zmns`, `lmns`, `lmns_full`.
- `mn_mode_nyq` (`mnmax_nyq`, modes `xm_nyq`, `xn_nyq`): all other coefficient arrays (`gmnc`, `bmnc`, `bsup*`, `bsub*`, `curr*`). The larger mode set keeps products such as $\sqrt{g}B^\theta$ free of aliasing.
- `xn` and `xn_nyq` **already contain `nfp`**: mode `(xm, xn)` is $\cos(m\theta - n\zeta)$ with `n` a multiple of `nfp`.
- Suffix `c` is a cosine series, `s` a sine series. A stellarator-symmetric run fills `rmnc`, `zmns`, `lmns`, `gmnc`, `bmnc`, ... With `lasym = True` the partners (`rmns`, `zmnc`, `lmnc`, `gmns`, ...) are filled too. Otherwise they are `None`.

### Angles and signs

- $\theta$ is the VMEC poloidal angle, not a straight-field-line angle. $\zeta$ is the geometric toroidal angle; one field period spans $[0, 2\pi/\text{nfp})$.
- `signgs` is the sign of the Jacobian $\sqrt{g}$ (`-1` by default).
- `phipf` is $\phi'$ on the full grid. `phips` is $\phi'/2\pi$ on the half grid; in the example `phipf == -2 * pi * phips`.

## 2. Fourier space to real space

$$
R = \sum_{m,n} \texttt{rmnc}_{mn}(s)\,\cos(m\theta - n\zeta), \qquad
Z = \sum_{m,n} \texttt{zmns}_{mn}(s)\,\sin(m\theta - n\zeta).
$$

Angle derivatives are exact: each derivative shifts the phase by $\pi/2$ and brings down a factor $m$ (for $\theta$) or $-n$ (for $\zeta$).

```python
def evaluate(coefficients, xm, xn, theta, zeta, kind, dtheta=0, dzeta=0):
    """Evaluate a "cos" or "sin" series on 2D arrays theta, zeta.

    Returns shape (n_surfaces, *theta.shape).
    """
    angle = xm[:, None, None] * theta - xn[:, None, None] * zeta
    angle = angle + 0.5 * np.pi * (dtheta + dzeta)
    factor = xm[:, None, None] ** dtheta * (-xn[:, None, None]) ** dzeta
    trig = np.cos if kind == "cos" else np.sin
    return np.einsum("ms,mtz->stz", coefficients, factor * trig(angle))


theta, zeta = np.meshgrid(
    np.linspace(0, 2 * np.pi, 32, endpoint=False),
    np.linspace(0, 2 * np.pi / wout.nfp, 24, endpoint=False),
    indexing="ij",
)
R = evaluate(wout.rmnc, wout.xm, wout.xn, theta, zeta, "cos")
Z = evaluate(wout.zmns, wout.xm, wout.xn, theta, zeta, "sin")
R_theta = evaluate(wout.rmnc, wout.xm, wout.xn, theta, zeta, "cos", dtheta=1)
Z_theta = evaluate(wout.zmns, wout.xm, wout.xn, theta, zeta, "sin", dtheta=1)

# Nyquist fields use the Nyquist mode lists; drop the unused half-grid column 0.
B = evaluate(wout.bmnc, wout.xm_nyq, wout.xn_nyq, theta, zeta, "cos")[1:]
```

With `lasym = True`, add the partner series (for example `rmns` with `"sin"` for `R`). Cartesian coordinates at $\phi = \zeta$ are $x = R\cos\phi$, $y = R\sin\phi$, $z = Z$.

## 3. Real space to Fourier space

On a uniform grid, project with discrete orthogonality of the same basis:

$$
c_{mn} = \frac{2 - \delta_{m0}\delta_{n0}}{N_\theta N_\zeta} \sum_{i,k} f(\theta_i, \zeta_k)\,\cos(m\theta_i - n\zeta_k)
$$

(use $\sin$ for a sine series; its $m = n = 0$ mode does not exist). The grid covers $\theta \in [0, 2\pi)$ and $\zeta \in [0, 2\pi/\text{nfp})$ without repeating the end point.

```python
def project(f, xm, xn, theta, zeta, kind):
    """f: (n_surfaces, ntheta, nzeta) on 2D arrays theta, zeta. Returns (n_modes, n_surfaces)."""
    angle = xm[:, None, None] * theta - xn[:, None, None] * zeta
    basis = np.cos(angle) if kind == "cos" else np.sin(angle)
    weight = np.where((xm == 0) & (xn == 0), 1.0, 2.0) / theta.size
    return weight[:, None] * np.einsum("stz,mtz->ms", f, basis)


nth, nze = 2 * (wout.mpol + 1) + 4, 2 * (wout.ntor + 1) + 4
th, ze = np.meshgrid(
    np.arange(nth) * 2 * np.pi / nth,
    np.arange(nze) * 2 * np.pi / wout.nfp / nze,
    indexing="ij",
)
R_grid = evaluate(wout.rmnc, wout.xm, wout.xn, th, ze, "cos")
rmnc_back = project(R_grid, wout.xm, wout.xn, th, ze, "cos")
print(np.abs(rmnc_back - wout.rmnc).max())  # 8e-16 in the example run
```

Resolution: `ntheta > 2 * max(xm)` and `nzeta > 2 * max(|xn|) / nfp`, using the Nyquist lists for Nyquist fields. A product of two fields has modes up to the sum of both, so project it on a grid fine enough for the sum. Otherwise high modes alias into low ones.

Do not apply `numpy.fft` to the coefficients directly: they are in the combined basis with mixed-sign `n`. See {doc}`../fourier_basis_implementation` for the basis the solver uses internally.

## 4. Reading the field quantities

All of the following live on the half grid (columns `1 ... ns-1`).

**Jacobian.** `gmnc` is the series of $\sqrt{g}$, with sign `signgs`. In the example, $\sqrt{g} = -R\,(R_s Z_\theta - R_\theta Z_s)$. Half-grid differences of R and Z (section 5) reproduce it to $4\cdot 10^{-4}$, the radial discretization error.

**Stream function.** `lmns` is $\lambda$ on the half grid (`lmns_full` on the full grid). The straight-field-line angle is $\theta^* = \theta + \lambda$.

**Contravariant B.** `bsupumnc`, `bsupvmnc` follow from $\lambda$, `iotas` and `phips`:

$$
B^\theta = \frac{\texttt{phips}\,(\iota - \partial_\zeta \lambda)}{\sqrt{g}}, \qquad
B^\zeta = \frac{\texttt{phips}\,(1 + \partial_\theta \lambda)}{\sqrt{g}} .
$$

```python
lam_theta = evaluate(wout.lmns, wout.xm, wout.xn, theta, zeta, "sin", dtheta=1)[1:]
sqrtg = evaluate(wout.gmnc, wout.xm_nyq, wout.xn_nyq, theta, zeta, "cos")[1:]
phips = wout.phips[1:, None, None]
bsupv = evaluate(wout.bsupvmnc, wout.xm_nyq, wout.xn_nyq, theta, zeta, "cos")[1:]
print(np.abs(phips * (1 + lam_theta) / sqrtg - bsupv).max() / np.abs(bsupv).max())
# 2e-4 on this 32 x 24 grid, 1e-14 on a 16 x 12 grid
```

The Nyquist series are projections of the solver's grid values, truncated at `max(xm_nyq)`. Identities like this one hold to round-off only on the solver's own poloidal grid (here `ntheta = 16`), and to about $10^{-4}$ on a finer grid.

**Covariant B and currents.** `bsubumnc`, `bsubvmnc` ($B_\theta$, $B_\zeta$) are on the half grid; `bsubsmns` ($B_s$) is on the full grid. `currumnc`, `currvmnc` are $\sqrt{g}J^\theta$, $\sqrt{g}J^\zeta$ on the full grid and **already include $1/\mu_0$**. Divide by the full-grid Jacobian (the average of the two neighboring half-grid values). See the `VmecWOut.currumnc` docstring for the exact recipe.

**Flux-surface average** on a uniform $(\theta, \zeta)$ grid: `np.sum(f * sqrtg) / np.sum(sqrtg)` per surface.

(regularized-derivatives)=
## 5. Radial derivatives

### Why plain differences fail near the axis

An odd-`m` coefficient behaves like $\sqrt{s}\,q(s)$ with $q$ smooth, so its derivative grows like $1/\sqrt{s}$ toward the axis. Finite differences or splines of the raw coefficient are inaccurate in the first cells. Even-`m` coefficients are smooth in `s` and need no care.

In the example, the `m = 1` coefficient of `R` divided by $\sqrt{s}$ is `[-3.79e-6, -4.05e-6, -4.13e-6, -4.16e-6]` at `j = 1 ... 4`: smooth, while `R` is not. The plain difference over the first half-grid cell is 37% off the regularized one, and 1% off at the next.

### Full grid to half grid

For a full-grid coefficient `c[mode, j]` (for example `rmnc`), with odd `m`: set `q = c / sqrt(s_full)`, extrapolate `q[0] = 2 q[1] - q[2]` (since `c = 0` on the axis), and use $dc/ds = \sqrt{s}\,dq/ds + q/(2\sqrt{s})$ with `q` averaged over the two neighbors. Even `m` uses the plain difference.

```python
def d_ds_full_to_half(c, xm, s_full, s_half):
    """Derivative of full-grid coefficients (n_modes, ns) on half-grid columns 1 ... ns-1."""
    h = s_full[1] - s_full[0]
    odd = (xm % 2 == 1)[:, None]
    q = c.copy()
    q[:, 1:] = np.where(odd, c[:, 1:] / np.sqrt(s_full[1:]), c[:, 1:])
    q[:, 0] = np.where(odd[:, 0], 2 * q[:, 1] - q[:, 2], c[:, 0])
    dq = (q[:, 1:] - q[:, :-1]) / h
    q_half = 0.5 * (q[:, 1:] + q[:, :-1])
    root = np.sqrt(s_half[1:])[None, :]
    return np.where(odd, root * dq + q_half / (2 * root), dq)


dR_ds = d_ds_full_to_half(wout.rmnc, wout.xm, s_full, s_half)  # cosine coefficients of R_s
dZ_ds = d_ds_full_to_half(wout.zmns, wout.xm, s_full, s_half)  # sine coefficients of Z_s
```

### Half grid to full grid

For half-grid fields (`bsubumnc`, `bsubvmnc`, ...) the derivative lands on interior full-grid surfaces `j = 1 ... ns-2`. This is the derivative VMEC++ uses with `lbsubs=True`, and the one to use when rebuilding currents from $B_u$, $B_v$, $B_s$. With $Q = \sqrt{s}\,q$ for odd `m`:

```python
def d_ds_half_to_full(c, xm, s_full, s_half):
    """Derivative of half-grid coefficients (n_modes, ns) on full-grid columns 1 ... ns-2."""
    h = s_full[1] - s_full[0]
    odd = (xm % 2 == 1)[:, None]
    inner, outer = c[:, 1:-1], c[:, 2:]  # half-grid columns j and j + 1
    q_inner = inner / np.sqrt(s_half[1:-1])
    q_outer = outer / np.sqrt(s_half[2:])
    root = np.sqrt(s_full[1:-1])[None, :]
    regular = root * (q_outer - q_inner) / h + (q_inner + q_outer) / (4 * root)
    return np.where(odd, regular, (outer - inner) / h)


dBu_ds = d_ds_half_to_full(wout.bsubumnc, wout.xm_nyq, s_full, s_half)
```

Apply it to both $B_u$ and $B_v$, sine and cosine partners alike. A plain odd-`m` difference breaks a cancellation at the axis and gives a spurious current that grows like $h^{-1/2}$ under radial refinement.

### Other points

- **Averaging between grids** (no derivative): average the two neighbors, and for odd `m` do it on `c / sqrt(s)`, then multiply back by `sqrt(s)` on the target grid.
- **Axis values** and the first half-grid cell are extrapolations. Do not trust local quantities there; compare two radial resolutions.
- **Other radial labels:** use the chain rule, for example $d/d\rho = 2\rho\,d/ds$ for $\rho = \sqrt{s}$, and $d/d\phi = (d/ds)/\phi'$ with `phipf`.

## 6. Checklist

1. Full or half grid? Is column 0 meaningful?
2. `xm`/`xn` or `xm_nyq`/`xn_nyq`? Is `xn` used as given, without another `nfp`?
3. Sine or cosine? Is `lasym` on, so that both series are needed?
4. Is the angle grid fine enough for the product you form?
5. Are odd-`m` radial derivatives regularized?
6. Are the signs of `signgs` and `phipf` accounted for?
7. Does a known identity hold (round trip, `bsupv` from $\lambda$, $\sqrt{g}$ from R and Z)?
