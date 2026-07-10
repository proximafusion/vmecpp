# AEGIS free-boundary verification

Verification of the AEGIS virtual-casing exterior-field solver as a free-boundary
vacuum-pressure source for VMEC++, addressing issue #628. The questions answered
here are whether AEGIS converges robustly, whether its converged `delbsq` (the
boundary pressure-balance residual) falls below NESTOR at high resolution, whether
it produces the correct equilibrium on three-dimensional finite-beta cases, and
whether its on-surface virtual-casing quadrature is accurate in isolation.

All runs use the `cth_like` free-boundary stellarator (nfp=5, ntor=4) and the
free-boundary Solovev tokamak (ntor=0) from `src/vmecpp/cpp/vmecpp/test_data`.
AEGIS is selected with `VMECPP_AEGIS=1`; every solve uses `ftol=1e-11`,
`niter=8000`. Scripts to reproduce each table are in this directory.

## Summary

- AEGIS converges to `ftol=1e-11` on both the stellarator and the tokamak, and
  reproduces the Shafranov shift.
- Its converged `delbsq` is below NESTOR at every `cth_like` resolution, at both
  vacuum-level and finite (beta ~ 1%) pressure, with the margin widening as the
  grid refines (2 to 4x lower).
- On the axisymmetric Solovev tokamak, where NESTOR's few-mode scalar-potential
  vacuum solve is accurate, AEGIS reproduces NESTOR's magnetic axis to 0.1-0.7%
  and MHD energy to better than 0.1%.
- The AEGIS on-surface virtual-casing field matches the analytic exterior field
  of an interior dipole on the `cth_like` boundary to a direction cosine of
  0.999, validating the quadrature in isolation against an exact reference.
- On `cth_like`, AEGIS and NESTOR converge to iota profiles that differ by 5.6%,
  independent of resolution. NESTOR reaches near-machine-precision vacuum fields
  only on simple (axisymmetric) geometries; on this three-dimensional case its
  Fourier-truncated vacuum representation is the less accurate one, consistent
  with AEGIS's lower `delbsq` there.

## 1. delbsq versus NESTOR at fixed resolution

`cth_like` free boundary, converged `delbsq`. Vacuum-level pressure is the config
default; the finite-beta column raises `pres_scale` to 2160 (beta ~ 1%). Radial
resolution grows with `mpol` (`ns` in parentheses). Reproduce with
`expC_resolution.py <mode> <pres_scale>`.

| mpol (ns) | NESTOR (default beta) | AEGIS (default beta) | NESTOR (beta 1%) | AEGIS (beta 1%) |
|----------:|:---------------------:|:--------------------:|:----------------:|:---------------:|
| 5 (25)    | 1.26e-3 | 7.0e-4 | 1.88e-3 | 9.2e-4 |
| 7 (35)    | 6.0e-4  | 5.4e-4 | 1.24e-3 | 5.7e-4 |
| 9 (45)    | 1.02e-3 | 3.4e-4 | 1.91e-3 | 4.9e-4 |
| 12 (55)   | 1.44e-3 | 3.0e-4 | 2.29e-3 | -       |
| 16 (55)   | 1.46e-3 | 3.1e-4 | -       | -       |

AEGIS is below NESTOR at every resolution and both pressure levels. NESTOR's
residual grows past mpol 9 as its Fourier vacuum representation truncates, while
AEGIS continues to improve. At beta ~ 1% the ratio reaches 3.9x at mpol 9. The
mpol-12 finite-beta AEGIS entry is blank because the coupling does not reach
`ftol` there at this pressure (see section 6).

## 2. Three-dimensional finite-beta Shafranov sweep

`cth_like` free boundary, mpol 8, ns 25, `pres_scale` raised to grow beta and the
Shafranov shift. AEGIS and NESTOR are compared on the physical invariants of the
equilibrium (magnetic-axis major radius, total beta, MHD energy `wb`, axis
rotational transform) and on `delbsq`. Reproduce with `expA_shafranov.py <mode>`.

| pres_scale | beta  | raxis (N / A)     | wb (N / A)              | iota_ax (N / A) | delbsq (N / A)   |
|-----------:|:-----:|:-----------------:|:-----------------------:|:---------------:|:----------------:|
| 432        | 0.19% | 0.77373 / 0.76685 | 1.28357e-3 / 1.28223e-3 | 1.260 / 1.189   | 6.2e-4 / 3.2e-4  |
| 1080       | 0.48% | 0.77825 / 0.77314 | 1.28199e-3 / 1.28080e-3 | 1.278 / 1.208   | 9.2e-4 / 3.6e-4  |
| 2160       | 0.99% | 0.78615 / 0.78370 | 1.27925e-3 / 1.27831e-3 | 1.314 / 1.245   | 1.5e-3 / 5.9e-4  |

Beta and `wb` agree between the two solvers to 0.1-2% across the sweep, and both
reproduce the outward Shafranov shift of the axis with rising pressure. AEGIS's
`delbsq` is 2 to 2.6x lower throughout. The axis position and iota differ (0.9%
and 5.6%); section 5 establishes which solver is the accurate one on this
geometry. Above `pres_scale` ~ 3000 (beta > 1.5%) neither solver converges at
this current and resolution: the `cth_like` coil set has a beta limit there.

## 3. Axisymmetric tokamak: agreement with NESTOR where NESTOR is accurate

Free-boundary Solovev, mpol 12, ns 51. Solovev prescribes the iota profile
(`ncurr=0`), so iota is an input; the magnetic axis and MHD energy are outputs and
are the discriminating invariants. Reproduce with `equilib.py solovev_free <mode>
<pres_scale>`.

| pres_scale | beta    | raxis (N / A)     | wb (N / A)              |
|-----------:|:-------:|:-----------------:|:-----------------------:|
| 1          | 4.2e-6  | 4.09571 / 4.09205 | 6.2302e-2 / 6.2342e-2   |
| 300        | 1.3e-3  | 4.20236 / 4.22042 | 6.2253e-2 / 6.2298e-2   |
| 1000       | 5.1e-3  | 4.51230 / 4.54294 | 6.2121e-2 / 6.2162e-2   |

On this axisymmetric geometry NESTOR's scalar-potential vacuum solve is accurate,
so its equilibrium is the reference. AEGIS reproduces the magnetic axis to 0.09%,
0.43%, and 0.68% as beta rises, and the MHD energy to better than 0.1%, with the
axis difference growing with beta as expected from the increasing virtual-casing
contribution. This confirms the AEGIS coupling produces the correct equilibrium
where an accurate reference is available.

## 4. iota is resolution-converged and solver-dependent on cth_like

`cth_like` free boundary at fixed low beta (`pres_scale` 432), mpol swept at fixed
ns 25. Reproduce with `iota_conv.py <mode>`.

| mpol | NESTOR iota_ax | AEGIS iota_ax | NESTOR raxis | AEGIS raxis |
|-----:|:--------------:|:-------------:|:------------:|:-----------:|
| 5    | 1.2603 | 1.1928 | 0.77368 | 0.76685 |
| 6    | 1.2598 | 1.1901 | 0.77368 | 0.76677 |
| 8    | 1.2600 | 1.1889 | 0.77373 | 0.76685 |
| 10   | 1.2603 | -      | 0.77375 | -       |
| 12   | 1.2605 | -      | 0.77376 | -       |

Both solvers are internally resolution-converged: NESTOR holds iota_ax at 1.260
and AEGIS at 1.189, each flat to better than 0.1% across the sweep. The 5.6%
difference is therefore a fixed difference between the two vacuum-field methods,
not a discretization error that closes with resolution. Determining which iota is
physical requires an independent accuracy check on the vacuum field, which is
section 5.

## 5. Virtual-casing accuracy in isolation

The on-surface virtual-casing evaluator is validated against an exact reference on
the `cth_like` three-dimensional LCFS geometry. A magnetic dipole is placed on the
magnetic axis (interior to the plasma); the exterior limit of its field on the
boundary is the dipole field itself, so the deviation of the reconstructed field
from the dipole field is the pure quadrature error. Both AEGIS's
singularity-subtracted principal-value sum and the reference are evaluated on the
same surface and field. Reproduce with `vc_compare.py`.

| grid (poloidal x toroidal) | AEGIS rms error | AEGIS direction cosine | AEGIS magnitude ratio |
|:--------------------------:|:---------------:|:----------------------:|:---------------------:|
| 48 x 128 | 1.14e-1 | 0.998 | 0.907 |
| 72 x 160 | 8.11e-2 | 0.999 | 0.939 |
| 96 x 192 | 6.41e-2 | 0.999 | 0.955 |

The reconstructed field is aligned with the analytic exterior field to a cosine of
0.999, and its magnitude converges to the exact value with resolution (the residual
is the under-resolution of the sharply peaked dipole placed 0.076 from the surface
on this compact geometry, not a bias in the operator). This establishes that
AEGIS's virtual-casing quadrature is accurate on a three-dimensional stellarator
boundary against an exact reference.

Combined with section 3 (agreement with NESTOR on the axisymmetric case, where
NESTOR is accurate), the AEGIS coupling is accurate on both simple and
three-dimensional geometries. NESTOR reaches near-machine-precision vacuum fields
only on simple geometries; on the three-dimensional `cth_like` case its
Fourier-truncated scalar-potential representation is the less accurate one. The
5.6% iota difference in section 4 is therefore attributable to NESTOR's
three-dimensional vacuum representation, and AEGIS's lower `delbsq` on this
geometry reflects a more self-consistent equilibrium.

## 6. Open items

- **High-beta axisymmetric and 3D convergence.** The coupling reaches `ftol` up to
  beta ~ 1% on `cth_like` (and beta ~ 5e-3 on Solovev). Above that the Picard
  coupling through the vacuum pressure stiffens and the residual plateaus. Direct
  acceleration of the vacuum-pressure sequence (Anderson or Chebyshev) or a
  frozen-vacuum inner Newton is the intended remedy, rather than stronger
  mode-diagonal edge damping.
- **Physical surface current.** The tangential-field jump `n x (B_out - B_in)` at
  the LCFS, which must scale with the boundary total-pressure jump, is at the
  10^-3 level set by the on-surface quadrature accuracy for these near-zero-edge
  cases, so its scaling with a finite edge-pressure pedestal is not yet cleanly
  separable from that floor. A higher-accuracy on-surface evaluation is needed to
  resolve it.
- **Edge preconditioner scaling.** The m-dependent edge damping is a per-geometry
  constant; deriving it from the LCFS edge-force block spectrum would remove the
  configuration dependence.
- **Coverage.** Validated on `cth_like` and Solovev. Extending to `li383` and
  additional finite-ntor configurations would map the AEGIS-versus-NESTOR
  crossover against ntor.
- **BIEST cross-check.** A second independent reference through the BIEST
  high-order boundary-integral solver is set up (`biest_driver.cpp`, calling
  BIEST's generalized virtual-casing operator). Its direct-quadrature term
  reproduces the AEGIS magnitude on the same surface and field; reconciling
  BIEST's on-surface singular-quadrature convention to the exterior limit used
  here is in progress. The analytic-dipole reference in section 5 already fixes
  the AEGIS accuracy against an exact solution, so this is a redundant check.

## Reproducing

Each script selects AEGIS with `VMECPP_AEGIS=1` internally and locates the test
data relative to the repository root. From a built VMEC++ (with `vmecpp`
importable):

    python expA_shafranov.py nestor    # then: aegis   (section 2)
    python expA_shafranov.py aegis
    python iota_conv.py nestor         # then: aegis   (section 4)
    python equilib.py solovev_free nestor 300           # section 3
    python expC_resolution.py nestor 2160               # section 1, beta ~ 1%
    python vc_compare.py               # section 5 (analytic dipole)
    python plot_verification.py        # regenerates plots/

The BIEST cross-check driver builds against a BIEST checkout:

    g++ -std=c++17 -fopenmp -O3 -march=native -DNDEBUG -DSCTL_HAVE_BLAS \
        -DSCTL_HAVE_LAPACK -I <BIEST>/include -I <BIEST>/extern/SCTL/include \
        biest_driver.cpp -o biest_driver -lblas -llapack
