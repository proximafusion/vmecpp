# Ideal MHD Model (`vmec/ideal_mhd_model/`)

This is the physics core called once per iteration via `IdealMhdModel::update()`: it maps the
current spectral geometry to real space, computes the Jacobian, metric, fields, pressure and
MHD forces, then maps the forces back to Fourier space. The solver driver that calls it lives
in `../vmec/` (see `../vmec/AGENTS.md`).

## Hot kernels

The forward/inverse spectral transforms dominate runtime; they run every iteration inside
`update()`:

- **Inverse transform** (Fourier -> real-space geometry): `geometryFromFourier()` ->
  `dft_FourierToReal_3d_symm()`, dispatching to the FFT path
  `FourierToReal3DSymmFastPoloidalFft()` (`fft_toroidal.*`, FFTX c2r) when FFT plans are
  available, else the DFT fallback `FourierToReal3DSymmFastPoloidal()` (`dft_toroidal.*`).
- **Forward transform** (real-space forces -> Fourier): `forcesToFourier()` ->
  `dft_ForcesToFourier_3d_symm()` -> `ForcesToFourier3DSymmFastPoloidalFft()` (FFTX r2c) or
  DFT fallback `ForcesToFourier3DSymmFastPoloidal()`.
- **Constraint-force de-aliasing**: `deAliasConstraintForce()` (`ideal_mhd_model.cc`) --
  bandpass-filters the spectral-condensation constraint force to suppress aliasing of high
  modes; itself a forward+inverse poloidal/toroidal transform pair.

2D axisymmetric (`*_2d_symm`) variants exist for tokamak cases. See
`docs/fourier_basis_implementation.md` for the DFT math and basis conventions.

## Staggered radial grid (full vs half)

The iteration uses a **staggered radial discretization**. Profile naming
(`../radial_profiles/`): trailing `F` = full grid (flux surfaces `s_j`), trailing `H`/`s` =
half grid (between surfaces, `s_{j-1/2}`).

- **Full grid (dynamical variables)**: the geometry spectral coefficients and their real-space
  values `R, Z, lambda` and poloidal/toroidal derivatives (`r1_e/o`, `z1_e/o`, `ru/rv`,
  `lu/lv`), plus full-grid profiles `iotaF`, `phipF`, `chipF`, `sqrtSF`. The force-balance
  variables themselves live on the full grid.
- **Half grid (derived each step)**: everything obtained by radial interpolation/differencing
  between surfaces -- the Jacobian `gsqrt` and metric elements `guu`, `guv`, `gvv`
  (`computeMetricElements()`); the half-grid geometry interpolants `r12`, `ru12`, `zu12`,
  `rs`, `zs`, `tau` (`computeJacobian()`); the contravariant field `bsupu`, `bsupv`
  (`computeBContra()`); and the profiles `iotaH`, `presH`, `massH`, `phipH`, `chipH`,
  `currH`, `dVdsH`. Kinetic pressure `presH` is computed from the (fixed) `massH` profile.

## Post-processing boundary

`../output_quantities/ComputeOutputQuantities()` runs **only after convergence** (or when
outputs are requested for a non-converged run). It produces the `wout` contents, `bsubs` on
both half and full grids, `jxbout` (J x B), Mercier stability, and other diagnostics. None of
these feed back into the force-balance loop -- they are derived once from the converged state,
so do not assume any output quantity is available mid-iteration.
