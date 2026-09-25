"""JAX port of the VMEC++ output stage: ``wout`` quantities from a converged geometry.

The solver returns the equilibrium as a :class:`vmecpp.geometry.Geometry` in the
internal product basis. This module maps it to the physics quantities of a ``wout``
file with the same discretization as the C++ output stage (output_quantities.cc):
the inverse DFT with the odd-m ``sqrt(s)`` scaling, the half-grid Jacobian, metric
and magnetic field, the low-pass filter of the covariant field, the jxbforce,
Mercier and threed1 stages, and the Nyquist-band forward DFT.

Only the geometry leaves are traced. Radial and angular grids, mode tables and the
flux and mass profiles are concrete NumPy arrays taken from the input, so
:func:`wout_quantities` works under ``jax.jit`` and ``jax.grad``.
"""

from __future__ import annotations

import dataclasses
import functools
import math
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from vmecpp import geometry as vmec_geometry

MU_0 = 4.0e-7 * math.pi

# Blending weight of the two full-grid estimates of B_v in the hybrid lambda
# force, 2 * kPDamp * (1 - s) in the C++ radial profiles.
_P_DAMP = 0.05

# Weight of the odd-m correction term in the half-grid Jacobian.
_D_S_HALF_D_S_INTERP = 0.25

# Exponents of the spectral width <M> = sum m^(p+q) |X_m|^2 / sum m^p |X_m|^2.
_SPECTRAL_WIDTH_P = 4
_SPECTRAL_WIDTH_Q = 1

# Larmor radius of a 1 keV proton at 1 T, in m (eqfor.f90).
_ION_LARMOR_AT_1T = 3.2e-3

# Fortran VMEC's padded lengths of the profile coefficient and spline arrays.
_PRESET = 21
_NDFMAX = 101

WOUT_QUANTITIES = (
    "rmnc",
    "zmns",
    "lmns",
    "lmns_full",
    "gmnc",
    "bmnc",
    "bsubumnc",
    "bsubvmnc",
    "bsubsmns",
    "bsupumnc",
    "bsupvmnc",
    "currumnc",
    "currvmnc",
    "rmns",
    "zmnc",
    "lmnc",
    "lmnc_full",
    "gmns",
    "bmns",
    "bsubumns",
    "bsubvmns",
    "bsubsmnc",
    "bsupumns",
    "bsupvmns",
    "currumns",
    "currvmns",
    "raxis_cc",
    "zaxis_cs",
    "raxis_cs",
    "zaxis_cc",
    "iotas",
    "iotaf",
    "q_factor",
    "phi",
    "phipf",
    "phips",
    "chi",
    "chipf",
    "presf",
    "pres",
    "mass",
    "vp",
    "buco",
    "bvco",
    "jcuru",
    "jcurv",
    "jdotb",
    "bdotb",
    "bdotgradv",
    "specw",
    "over_r",
    "beta_vol",
    "equif",
    "DMerc",
    "DShear",
    "DWell",
    "DCurr",
    "DGeod",
    "wb",
    "wp",
    "rmax_surf",
    "rmin_surf",
    "zmax_surf",
    "aspect",
    "betatotal",
    "betapol",
    "betator",
    "betaxis",
    "b0",
    "rbtor0",
    "rbtor",
    "IonLarmor",
    "volavgB",
    "ctor",
    "Aminor_p",
    "Rmajor_p",
    "volume",
)
"""The ``VmecWOut`` fields computed by :func:`wout_quantities`.

The non-stellarator-symmetric fields are ``None`` for ``lasym=False``.
"""

_ASYMMETRIC_QUANTITIES = frozenset(
    {
        "rmns",
        "zmnc",
        "lmnc",
        "lmnc_full",
        "gmns",
        "bmns",
        "bsubumns",
        "bsubvmns",
        "bsubsmnc",
        "bsupumns",
        "bsupvmns",
        "currumns",
        "currvmns",
        "raxis_cs",
        "zaxis_cc",
    }
)


def wout_quantities(
    geometry: vmec_geometry.Geometry,
    vmec_input: Any,
    *,
    iota_half=None,
    mass_half=None,
) -> dict[str, jax.Array | None]:
    """Map a converged geometry to the physics quantities of its ``wout`` file.

    Args:
        geometry: The solved equilibrium in the internal product basis.
        vmec_input: The :class:`vmecpp.VmecInput` the geometry was solved for;
            supplies the grid sizes, mode tables and the flux and mass profiles.
        iota_half: Rotational transform on the half grid. Defaults to the ratio
            of the poloidal and toroidal flux increments of ``geometry``.
        mass_half: The half-grid mass profile ``mu_0 p dV/ds^gamma``. Defaults to
            :func:`mass_profile` of the input, which is not differentiated.

    Returns:
        The fields listed in :data:`WOUT_QUANTITIES`, in the layout of
        :class:`vmecpp.VmecWOut`.
    """
    sizes = _sizes(vmec_input)
    profiles = _profiles(vmec_input, sizes, mass_half)
    if iota_half is not None:
        iota_half = jnp.asarray(iota_half)
    return _wout_quantities(sizes, profiles, _kernels(sizes), geometry, iota_half)


@functools.partial(jax.jit, static_argnums=0)
def _wout_quantities(sizes, profiles, kernels, geometry, iota_half):
    setup = _make_setup(sizes, profiles, kernels)
    ns = setup.ns
    delta_s = 1.0 / (ns - 1)
    signgs = setup.signgs
    w_int = jnp.asarray(setup.w_int)[None, None, :]

    toroidal_flux = jnp.asarray(geometry.toroidal_flux)
    poloidal_flux = jnp.asarray(geometry.poloidal_flux)
    if iota_half is None:
        iota_h = jnp.diff(poloidal_flux) / jnp.diff(toroidal_flux)
    else:
        iota_h = iota_half.astype(toroidal_flux.dtype)
    phip_f = jnp.asarray(setup.phip_full)
    phip_h = jnp.asarray(setup.phip_half)
    chip_h = iota_h * phip_h
    chip_f = _half_to_full(chip_h)

    # Inverse DFT to the full-grid real-space geometry, split by m parity.
    r = _real_space(setup, _r_terms(geometry, setup))
    z = _real_space(setup, _z_terms(geometry, setup))
    # The solver normalizes lambda by lamscale / phi'; undo it so the theta
    # derivative combines with phi' as in the C++ B^v.
    lam = _real_space(
        setup,
        _lambda_terms(geometry, setup, phip_f),
        extrapolate_m0=True,
    )

    # Half-grid Jacobian and metric (jacobian_kernel.h, metric_kernel.h).
    sqrt_s_h = jnp.asarray(setup.sqrt_s_half)[:, None, None]
    s_f = jnp.asarray(setup.sqrt_s_full**2)[:, None, None]
    s_i, s_o = s_f[:-1], s_f[1:]

    def inner(x):
        return x[:-1]

    def outer(x):
        return x[1:]

    def half(even, odd):
        return 0.5 * (
            (inner(even) + outer(even)) + sqrt_s_h * (inner(odd) + outer(odd))
        )

    r12 = half(r.value_e, r.value_o)
    ru12 = half(r.dtheta_e, r.dtheta_o)
    zu12 = half(z.dtheta_e, z.dtheta_o)
    rs = (
        (outer(r.value_e) - inner(r.value_e))
        + sqrt_s_h * (outer(r.value_o) - inner(r.value_o))
    ) / delta_s
    zs = (
        (outer(z.value_e) - inner(z.value_e))
        + sqrt_s_h * (outer(z.value_o) - inner(z.value_o))
    ) / delta_s
    tau1 = ru12 * zs - rs * zu12
    tau2 = (
        outer(r.dtheta_o) * outer(z.value_o)
        + inner(r.dtheta_o) * inner(z.value_o)
        - outer(z.dtheta_o) * outer(r.value_o)
        - inner(z.dtheta_o) * inner(r.value_o)
        + (
            outer(r.dtheta_e) * outer(z.value_o)
            + inner(r.dtheta_e) * inner(z.value_o)
            - outer(z.dtheta_e) * outer(r.value_o)
            - inner(z.dtheta_e) * inner(r.value_o)
        )
        / sqrt_s_h
    )
    tau = tau1 + _D_S_HALF_D_S_INTERP * tau2
    gsqrt = tau * r12

    def metric(a_e, a_o, b_e, b_o):
        return 0.5 * (
            inner(a_e) * inner(b_e)
            + outer(a_e) * outer(b_e)
            + s_i * inner(a_o) * inner(b_o)
            + s_o * outer(a_o) * outer(b_o)
        ) + 0.5 * sqrt_s_h * (
            inner(a_e) * inner(b_o)
            + outer(a_e) * outer(b_o)
            + inner(b_e) * inner(a_o)
            + outer(b_e) * outer(a_o)
        )

    guu = metric(r.dtheta_e, r.dtheta_o, r.dtheta_e, r.dtheta_o) + metric(
        z.dtheta_e, z.dtheta_o, z.dtheta_e, z.dtheta_o
    )
    guv = metric(r.dtheta_e, r.dtheta_o, r.dzeta_e, r.dzeta_o) + metric(
        z.dtheta_e, z.dtheta_o, z.dzeta_e, z.dzeta_o
    )
    gvv = (
        metric(r.value_e, r.value_o, r.value_e, r.value_o)
        + metric(r.dzeta_e, r.dzeta_o, r.dzeta_e, r.dzeta_o)
        + metric(z.dzeta_e, z.dzeta_o, z.dzeta_e, z.dzeta_o)
    )

    # Contravariant field (bcontra_kernel.h): lambda enters as phi' (1 +
    # d lambda / d theta) and -phi' d lambda / d zeta.
    lu_e = lam.dtheta_e + phip_f[:, None, None]
    lu_o = lam.dtheta_o
    lv_e = -lam.dzeta_e
    lv_o = -lam.dzeta_o
    bsupv = half(lu_e, lu_o) / gsqrt
    bsupu = half(lv_e, lv_o) / gsqrt + chip_h[:, None, None] / gsqrt
    bsubu = guu * bsupu + guv * bsupv
    bsubv = guv * bsupu + gvv * bsupv
    magnetic_pressure = 0.5 * (bsupu * bsubu + bsupv * bsubv)
    mod_b = jnp.sqrt(2.0 * jnp.abs(magnetic_pressure))

    dvds_h = signgs * jnp.sum(gsqrt * w_int, axis=(1, 2))
    mass_h = jnp.asarray(setup.mass_half)
    pres_h = mass_h / dvds_h**setup.gamma if setup.gamma != 0.0 else mass_h
    total_pressure = magnetic_pressure + pres_h[:, None, None]
    buco_solver = jnp.sum(bsubu * w_int, axis=(1, 2))
    bvco_solver = jnp.sum(bsubv * w_int, axis=(1, 2))

    # Energies and enclosed currents of the final solver evaluation
    # (IdealMhdModel::pressureAndEnergies, the rBtor/cTor block of update).
    wb = jnp.abs(jnp.sum(gsqrt * magnetic_pressure * w_int)) * delta_s
    wp = jnp.sum(pres_h * dvds_h) * delta_s
    rbtor0 = 1.5 * bvco_solver[0] - 0.5 * bvco_solver[1]
    rbtor = 1.5 * bvco_solver[-1] - 0.5 * bvco_solver[-2]
    ctor = (
        (1.5 * buco_solver[-1] - 0.5 * buco_solver[-2]) * signgs * 2.0 * math.pi / MU_0
    )

    # The output stage rebuilds B_v from the hybrid lambda force, restores the
    # enclosed poloidal current, and low-pass filters both covariant
    # components and B_s to the solver's mode range (LowPassFilterCovariantB).
    bsubv_fixed = _blend_bsubv(
        setup, bsubv, bvco_solver, gvv, gsqrt, guv, bsupu, lu_e, lu_o
    )
    bsubu_out, _, bsubuv = _filter_cosine_type(setup, bsubu)
    bsubv_out, bsubvu, _ = _filter_cosine_type(setup, bsubv_fixed)
    del bsubuv, bsubvu  # only the jsups diagnostic of jxbout uses them

    # Covariant B_s on the half grid (ComputeRemainingMetric, ComputeBSubSOnHalfGrid).
    rv12 = half(r.dzeta_e, r.dzeta_o)
    zv12 = half(z.dzeta_e, z.dzeta_o)
    rs12 = rs + 0.5 * (inner(r.value_o) + outer(r.value_o)) / (2.0 * sqrt_s_h)
    zs12 = zs + 0.5 * (inner(z.value_o) + outer(z.value_o)) / (2.0 * sqrt_s_h)
    gsu = rs12 * ru12 + zs12 * zu12
    gsv = rs12 * rv12 + zs12 * zv12
    bsubs = bsupu * gsu + bsupv * gsv

    # B_s on the full grid; axis and boundary stay zero through the filter
    # (PutBSubSOnFullGrid, then ExtrapolateBSubS after the filter).
    zero_surface = jnp.zeros_like(bsubs[:1])
    bsubs_full = jnp.concatenate(
        [zero_surface, 0.5 * (bsubs[1:] + bsubs[:-1]), zero_surface]
    )
    _, bsubsu, bsubsv = _filter_sine_type(setup, bsubs_full)

    # jxbforce: current density and flux-surface averages on the interior
    # full grid (ComputeJxBOutputFileContents).
    dnorm1 = 4.0 * math.pi**2
    ovp = 2.0 / (dvds_h[1:] + dvds_h[:-1]) / dnorm1
    tjnorm = ovp * signgs
    w_int_2d = jnp.asarray(setup.w_int)[None, :]

    def interior_full(values_h):
        return 0.5 * (values_h[1:] + values_h[:-1])

    bsq = total_pressure - pres_h[:, None, None]
    sqgb2 = gsqrt[1:] * bsq[1:] + gsqrt[:-1] * bsq[:-1]
    bsubus = (bsubu_out[1:] - bsubu_out[:-1]) / delta_s
    bsubvs = (bsubv_out[1:] - bsubv_out[:-1]) / delta_s
    itheta = (bsubsv[1:-1] - bsubvs) / MU_0
    izeta = (-bsubsu[1:-1] + bsubus) / MU_0
    bsubu1 = interior_full(bsubu_out)
    bsubv1 = interior_full(bsubv_out)
    bdotk = itheta * bsubu1 + izeta * bsubv1

    def surface_average(values):
        return jnp.sum(values * w_int_2d[None], axis=(1, 2))

    jdotb = _extrapolate_both(dnorm1 * tjnorm * surface_average(bdotk))
    bdotb = _extrapolate_both(dnorm1 * tjnorm * surface_average(sqgb2))
    bdotgradv = _extrapolate_both(dnorm1 * tjnorm * interior_full(phip_h))

    # Mercier stability (ComputeIntermediateMercierQuantities,
    # ComputeMercierStability).
    phip_real_h = 2.0 * math.pi * phip_h * signgs
    vp_real = signgs * dnorm1 * dvds_h / phip_real_h
    torcur = signgs * 2.0 * math.pi * jnp.sum(bsubu_out * w_int, axis=(1, 2))
    phip_real_f = interior_full(phip_real_h)
    denom = phip_real_f * delta_s
    shear = jnp.diff(iota_h) / denom
    vpp = jnp.diff(vp_real) / denom
    presp = jnp.diff(pres_h) / denom
    ip = jnp.diff(torcur) / denom
    gsqrt_full = interior_full(gsqrt)
    bdotj = bdotk * MU_0 / gsqrt_full
    gsqrt_full = gsqrt_full / phip_real_f[:, None, None]
    sqrt_s_interior = jnp.asarray(np.sqrt(np.arange(1, ns - 1) * delta_s))[
        :, None, None
    ]

    def full_value(x, j0, j1):
        return x.value_e[j0:j1] + sqrt_s_interior * x.value_o[j0:j1]

    def full_dtheta(x):
        return x.dtheta_e[1:-1] + sqrt_s_interior * x.dtheta_o[1:-1]

    def full_dzeta(x):
        return x.dzeta_e[1:-1] + sqrt_s_interior * x.dzeta_o[1:-1]

    rtf = full_dtheta(r)
    ztf = full_dtheta(z)
    rzf = full_dzeta(r)
    zzf = full_dzeta(z)
    r1f = full_value(r, 1, ns - 1)
    gtt = rtf * rtf + ztf * ztf
    gpp = gsqrt_full**2 / (gtt * r1f**2 + (rtf * zzf - rzf * ztf) ** 2)
    b2 = 2.0 * bsq
    b2i = interior_full(b2)
    tpp = dnorm1 * surface_average(gsqrt_full / b2i)
    tbb = dnorm1 * surface_average(b2i * gsqrt_full * gpp)
    jdotb_mercier = bdotj * gpp * gsqrt_full
    tjb = dnorm1 * surface_average(jdotb_mercier)
    tjj = dnorm1 * surface_average(jdotb_mercier * bdotj / b2i)
    dshear = _pad_both(shear * shear / 4.0)
    dcurr = _pad_both(-shear * (tjb - ip * tbb))
    dwell = _pad_both(presp * (vpp - presp * tpp) * tbb)
    dgeod = _pad_both(tjb * tjb - tbb * tjj)
    dmerc = dshear + dcurr + dwell + dgeod

    # threed1 first table (ComputeIntermediateThreed1FirstTableQuantities).
    tau_w = signgs * w_int * gsqrt
    s2 = jnp.sum(total_pressure * tau_w, axis=(1, 2)) / dvds_h - pres_h
    beta_vol = pres_h / s2
    over_r = jnp.sum(tau_w / r12, axis=(1, 2)) / dvds_h
    buco_h = jnp.sum(bsubu_out * w_int, axis=(1, 2))
    bvco_h = jnp.sum(bsubv_out * w_int, axis=(1, 2))
    chi = (
        2.0
        * math.pi
        * delta_s
        * jnp.concatenate([jnp.zeros(1), jnp.cumsum(phip_h * iota_h)])
    )
    sign_by_delta_s = signgs / delta_s
    jcurv = sign_by_delta_s * jnp.diff(buco_h)
    jcuru = -sign_by_delta_s * jnp.diff(bvco_h)
    presgrad = jnp.diff(pres_h) / delta_s
    vpphi = interior_full(dvds_h)
    chip_f_interior = chip_f[1:-1]
    phip_f_interior = phip_f[1:-1]
    equif = (chip_f_interior * jcurv - phip_f_interior * jcuru) / vpphi + presgrad
    equif = (
        equif
        * vpphi
        / (
            jnp.abs(jcurv * chip_f_interior)
            + jnp.abs(jcuru * phip_f_interior)
            + jnp.abs(presgrad * vpphi)
        )
    )

    # Geometric and magnetic quantities
    # (ComputeIntermediateThreed1GeometricMagneticQuantities,
    # ComputeThreed1GeometricMagneticQuantities, ComputeThreed1Betas).
    anorm = 2.0 * math.pi * delta_s
    vnorm = 2.0 * math.pi * anorm
    sump = vnorm * jnp.sum(dvds_h * pres_h)
    sumbtot = 2.0 * (vnorm * jnp.sum(total_pressure * tau_w) - sump)
    sumbtor = vnorm * jnp.sum(tau_w * (r12 * bsupv) ** 2)
    sumbpol = sumbtot - sumbtor
    r_lcfs = r.value_e[-1] + r.value_o[-1]
    z_lcfs = z.value_e[-1] + z.value_o[-1]
    zu_lcfs = z.dtheta_e[-1] + z.dtheta_o[-1]
    w_lcfs = jnp.asarray(setup.w_int)[None, :]
    cross_area_p = 2.0 * math.pi * jnp.abs(jnp.sum(r_lcfs * zu_lcfs * w_lcfs))
    volume_p = 2.0 * math.pi**2 * jnp.abs(jnp.sum(r_lcfs**2 * zu_lcfs * w_lcfs))
    rmajor_p = volume_p / (2.0 * math.pi * cross_area_p)
    aminor_p = jnp.sqrt(cross_area_p / math.pi)
    volavgb = jnp.sqrt(jnp.abs(sumbtot / volume_p))
    fpsi0 = 1.5 * bvco_h[0] - 0.5 * bvco_h[1]
    b0 = fpsi0 / r.value_e[0, 0, 0]

    # Nyquist-band forward DFT of the half-grid fields (ComputeWOutFileContents).
    gmnc, gmns = _nyquist_half(setup, gsqrt, cosine=True)
    bmnc, bmns = _nyquist_half(setup, mod_b, cosine=True)
    bsubumnc, bsubumns = _nyquist_half(setup, bsubu_out, cosine=True)
    bsubvmnc, bsubvmns = _nyquist_half(setup, bsubv_out, cosine=True)
    bsupumnc, bsupumns = _nyquist_half(setup, bsupu, cosine=True)
    bsupvmnc, bsupvmns = _nyquist_half(setup, bsupv, cosine=True)
    bsubsmns, bsubsmnc = _nyquist_half(setup, bsubs, cosine=False)
    # The classic output extrapolates the half-grid B_s one full step beyond
    # the innermost half-grid point.
    bsubsmns = _extrapolate_axis_column(bsubsmns)
    currumnc, currvmnc = _currents(setup, bsubsmns, bsubumnc, bsubvmnc, sign=1.0)

    # Combined-basis coefficients of R, Z and lambda on the full grid.
    rmnc = _to_combined(setup, geometry.r_cc, geometry.r_ss, cosine=True)
    zmns = _to_combined(setup, geometry.z_sc, geometry.z_cs, cosine=False)
    lmns_full = _to_combined(
        setup, geometry.lambda_sc, geometry.lambda_cs, cosine=False
    )
    if setup.lthreed:
        # The m = 0 lambda modes are not evolved at the axis; the output
        # extrapolates them from the first two surfaces (in solver scaling).
        m0 = setup.xm == 0
        lambda_cs = jnp.asarray(geometry.lambda_cs)
        axis = (
            -(2.0 * lambda_cs[1, 0, :] * phip_f[1] - lambda_cs[2, 0, :] * phip_f[2])
            / phip_f[0]
        )
        lmns_full = lmns_full.at[np.flatnonzero(m0), 0].set(axis)
    lmns = _lambda_to_half_grid(setup, lmns_full)
    raxis_cc = jnp.asarray(geometry.r_cc)[0, 0, :]
    zaxis_cs = (
        -jnp.asarray(geometry.z_cs)[0, 0, :]
        if setup.lthreed
        else jnp.zeros(setup.ntor + 1)
    )

    quantities: dict[str, jax.Array | None] = dict.fromkeys(_ASYMMETRIC_QUANTITIES)
    if setup.lasym:
        assert bsubsmnc is not None
        assert bsubumns is not None
        assert bsubvmns is not None
        bsubsmnc = _extrapolate_axis_column(bsubsmnc)
        currumns, currvmns = _currents(setup, bsubsmnc, bsubumns, bsubvmns, sign=-1.0)
        lmnc_full = _to_combined(
            setup, geometry.lambda_cc, geometry.lambda_ss, cosine=True
        )
        quantities.update(
            rmns=_to_combined(setup, geometry.r_sc, geometry.r_cs, cosine=False),
            zmnc=_to_combined(setup, geometry.z_cc, geometry.z_ss, cosine=True),
            lmnc_full=lmnc_full,
            lmnc=_lambda_to_half_grid(setup, lmnc_full),
            gmns=gmns,
            bmns=bmns,
            bsubumns=bsubumns,
            bsubvmns=bsubvmns,
            bsubsmnc=bsubsmnc,
            bsupumns=bsupumns,
            bsupvmns=bsupvmns,
            currumns=currumns,
            currvmns=currvmns,
            raxis_cs=(
                -jnp.asarray(geometry.r_cs)[0, 0, :]
                if setup.lthreed
                else jnp.zeros(setup.ntor + 1)
            ),
            zaxis_cc=jnp.asarray(geometry.z_cc)[0, 0, :],
        )

    iota_f = _half_to_full(iota_h)
    betatotal = 2.0 * sump / sumbtot
    quantities.update(
        rmnc=rmnc,
        zmns=zmns,
        lmns=lmns,
        lmns_full=lmns_full,
        gmnc=gmnc,
        bmnc=bmnc,
        bsubumnc=bsubumnc,
        bsubvmnc=bsubvmnc,
        bsubsmns=bsubsmns,
        bsupumnc=bsupumnc,
        bsupvmnc=bsupvmnc,
        currumnc=currumnc,
        currvmnc=currvmnc,
        raxis_cc=raxis_cc,
        zaxis_cs=zaxis_cs,
        iotas=_pad_axis(iota_h),
        iotaf=iota_f,
        q_factor=jnp.where(
            iota_f != 0.0,
            1.0 / jnp.where(iota_f != 0.0, iota_f, 1.0),
            np.finfo(float).max,
        ),
        phi=toroidal_flux,
        phipf=signgs * 2.0 * math.pi * phip_f,
        phips=_pad_axis(phip_h),
        chi=chi,
        chipf=signgs * 2.0 * math.pi * chip_f,
        presf=_half_to_full(pres_h) / MU_0,
        pres=_pad_axis(pres_h) / MU_0,
        mass=_pad_axis(mass_h) / MU_0,
        vp=_pad_axis(dvds_h),
        buco=_pad_axis(buco_h),
        bvco=_pad_axis(bvco_h),
        jcuru=_extrapolate_both(jcuru) / MU_0,
        jcurv=_extrapolate_both(jcurv) / MU_0,
        jdotb=jdotb,
        bdotb=bdotb,
        bdotgradv=bdotgradv,
        specw=_spectral_width(setup, geometry),
        over_r=_pad_axis(over_r),
        beta_vol=_pad_axis(beta_vol),
        equif=_extrapolate_both(equif),
        DMerc=dmerc,
        DShear=dshear,
        DWell=dwell,
        DCurr=dcurr,
        DGeod=dgeod,
        wb=wb,
        wp=wp,
        rmax_surf=jnp.max(r_lcfs),
        rmin_surf=jnp.min(r_lcfs),
        zmax_surf=jnp.max(jnp.abs(z_lcfs)),
        aspect=rmajor_p / aminor_p,
        betatotal=betatotal,
        betapol=2.0 * sump / sumbpol,
        betator=2.0 * sump / sumbtor,
        betaxis=1.5 * beta_vol[0] - 0.5 * beta_vol[1],
        b0=b0,
        rbtor0=rbtor0,
        rbtor=rbtor,
        IonLarmor=_ION_LARMOR_AT_1T / volavgb,
        volavgB=volavgb,
        ctor=ctor,
        Aminor_p=aminor_p,
        Rmajor_p=rmajor_p,
        volume=volume_p,
    )
    return quantities


def static_fields(vmec_input: Any) -> dict[str, Any]:
    """The ``VmecWOut`` fields fixed by the input: echoes, sizes and mode tables.

    Free-boundary fields are those of a fixed-boundary run.
    """
    sizes = _sizes(vmec_input)
    xm, xn = _mode_table(sizes.mpol, sizes.ntor, sizes.nfp)
    xm_nyq, xn_nyq = _mode_table(sizes.mnyq + 1, sizes.nnyq, sizes.nfp)

    def padded(values, size: int, fill: float) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64).ravel()
        if values.size == 0:
            values = np.asarray([fill])
        return np.pad(values, (0, max(0, size - values.size)), constant_values=fill)

    extcur = np.asarray(vmec_input.extcur, dtype=np.float64).ravel()
    return {
        "version_": 8.52,
        "input_extension": "",
        "signgs": sizes.signgs,
        "gamma": sizes.gamma,
        "pcurr_type": vmec_input.pcurr_type,
        "pmass_type": vmec_input.pmass_type,
        "piota_type": vmec_input.piota_type,
        "am": padded(vmec_input.am, _PRESET, 0.0),
        "ac": padded(vmec_input.ac, _PRESET, 0.0),
        "ai": padded(vmec_input.ai, _PRESET, 0.0),
        "am_aux_s": padded(vmec_input.am_aux_s, _NDFMAX, -1.0),
        "am_aux_f": padded(vmec_input.am_aux_f, _NDFMAX, 0.0),
        "ac_aux_s": padded(vmec_input.ac_aux_s, _NDFMAX, -1.0),
        "ac_aux_f": padded(vmec_input.ac_aux_f, _NDFMAX, 0.0),
        "ai_aux_s": padded(vmec_input.ai_aux_s, _NDFMAX, -1.0),
        "ai_aux_f": padded(vmec_input.ai_aux_f, _NDFMAX, 0.0),
        "nfp": sizes.nfp,
        "mpol": sizes.mpol,
        "ntor": sizes.ntor,
        "lasym": sizes.lasym,
        "ns": sizes.ns,
        "ftolv": float(np.asarray(vmec_input.ftol_array)[-1]),
        "lfreeb": bool(vmec_input.lfreeb),
        "lrfp": False,
        "mgrid_file": vmec_input.mgrid_file,
        "nextcur": int(extcur.size),
        "extcur": extcur,
        "mgrid_mode": "",
        "mnmax": int(xm.size),
        "mnmax_nyq": int(xm_nyq.size),
        "xm": xm,
        "xn": xn,
        "xm_nyq": xm_nyq,
        "xn_nyq": xn_nyq,
        "potvac": None,
        "xmpot": None,
        "xnpot": None,
    }


UNKNOWN_DIAGNOSTICS: dict[str, Any] = {
    "ier_flag": -1,
    "niter": 0,
    "itfsq": 0,
    "fsqr": math.nan,
    "fsqz": math.nan,
    "fsql": math.nan,
    "fsqt": np.zeros(0),
    "force_residual_r": np.zeros(0),
    "force_residual_z": np.zeros(0),
    "force_residual_lambda": np.zeros(0),
    "delbsq": np.zeros(0),
    "restart_reason_timetrace": np.zeros(0, dtype=np.int64),
    "wdot": np.zeros(0),
}
"""Solver diagnostics of a ``VmecWOut`` whose solve is not observable, e.g. under
``jax.jit``: ``ier_flag = -1``, NaN residuals and empty traces."""


def toroidal_flux_derivative(vmec_input: Any, s: np.ndarray) -> np.ndarray:
    """``d phi / d s`` of the input flux profile, before the ``2 pi`` and sign
    factors."""
    aphi = np.asarray(vmec_input.aphi, dtype=np.float64)
    if aphi.size == 0:
        aphi = np.asarray([1.0])
    powers = np.arange(1, aphi.size + 1)
    derivative = np.polyval((powers * aphi)[::-1], s)
    edge_flux = np.polyval(np.concatenate([aphi[::-1], [0.0]]), 1.0)
    scale = vmec_input.signgs * vmec_input.phiedge * vmec_input.bloat / (2.0 * math.pi)
    if edge_flux != 0.0:
        scale /= edge_flux
    return scale * derivative


def mass_profile(vmec_input: Any, s_half: np.ndarray) -> np.ndarray:
    """The half-grid mass profile ``mu_0 p`` of the C++ radial profiles."""
    aphi = np.asarray(vmec_input.aphi, dtype=np.float64)
    if aphi.size == 0:
        aphi = np.asarray([1.0])
    evaluation_position = np.minimum(s_half, vmec_input.spres_ped)
    toroidal_flux = np.polyval(np.concatenate([aphi[::-1], [0.0]]), evaluation_position)
    normalized = np.minimum(
        np.abs(np.minimum(toroidal_flux, 1.0) * vmec_input.bloat), 1.0
    )
    pressure = _evaluate_profile(
        vmec_input.pmass_type,
        np.asarray(vmec_input.am, dtype=np.float64),
        np.asarray(vmec_input.am_aux_s, dtype=np.float64),
        np.asarray(vmec_input.am_aux_f, dtype=np.float64),
        normalized,
    )
    mass = MU_0 * vmec_input.pres_scale * pressure
    if vmec_input.gamma != 0.0:
        r00 = float(np.asarray(vmec_input.rbc)[0, vmec_input.ntor])
        phip_half = toroidal_flux_derivative(vmec_input, s_half)
        mass = mass * (np.abs(phip_half) * r00) ** vmec_input.gamma
    return mass


@dataclasses.dataclass(frozen=True)
class _RealSpace:
    """Full-grid real-space values of one Fourier series, split by m parity.

    Arrays have shape ``(ns, nzeta, ntheta_eff)``; the odd-m part carries the
    solver's ``1 / sqrt(s)`` scaling.
    """

    value_e: jax.Array
    value_o: jax.Array
    dtheta_e: jax.Array
    dtheta_o: jax.Array
    dzeta_e: jax.Array
    dzeta_o: jax.Array


@dataclasses.dataclass(frozen=True)
class _Setup:
    """Grids and mode tables (NumPy) and the profiles and kernels (traced)."""

    ns: int
    mpol: int
    ntor: int
    nfp: int
    lthreed: bool
    lasym: bool
    signgs: int
    gamma: float
    ntheta_even: int
    ntheta_reduced: int
    ntheta_eff: int
    nzeta: int
    theta: np.ndarray
    zeta: np.ndarray
    w_int: np.ndarray
    zeta_reversed: np.ndarray
    theta_reversed: np.ndarray
    xm: np.ndarray
    xn: np.ndarray
    xm_nyq: np.ndarray
    xn_nyq: np.ndarray
    sqrt_s_full: np.ndarray
    sqrt_s_half: np.ndarray
    odd_scale: np.ndarray
    phip_full: jax.Array
    phip_half: jax.Array
    mass_half: jax.Array
    radial_blending: np.ndarray
    cos_kernel: jax.Array
    sin_kernel: jax.Array
    low_pass: dict[str, tuple[jax.Array, ...]]


@dataclasses.dataclass(frozen=True)
class _Sizes:
    """The static sizes of one input; ``jax.jit`` compiles once per value."""

    ns: int
    mpol: int
    ntor: int
    nfp: int
    lasym: bool
    signgs: int
    gamma: float
    ntheta_even: int
    nzeta: int

    @property
    def ntheta_reduced(self) -> int:
        return self.ntheta_even // 2 + 1

    @property
    def ntheta_eff(self) -> int:
        return self.ntheta_even if self.lasym else self.ntheta_reduced

    @property
    def mnyq(self) -> int:
        return max(0, self.ntheta_even // 2, self.mpol - 1)

    @property
    def nnyq(self) -> int:
        return max(0, self.nzeta // 2, self.ntor)


def _sizes(vmec_input: Any) -> _Sizes:
    if not isinstance(vmec_input.mpol, int) or not isinstance(vmec_input.ntor, int):
        error_message = "wout_quantities requires scalar mpol and ntor"
        raise ValueError(error_message)
    mpol = vmec_input.mpol
    ntor = vmec_input.ntor
    ns = int(np.asarray(vmec_input.ns_array)[-1])
    if ns < 3:
        error_message = "wout_quantities requires ns >= 3"
        raise ValueError(error_message)
    # Sizes::computeDerivedSizes
    ntheta = max(vmec_input.ntheta, 2 * mpol + 6)
    nzeta = (
        max(vmec_input.nzeta, 1) if ntor == 0 else max(vmec_input.nzeta, 2 * ntor + 4)
    )
    return _Sizes(
        ns=ns,
        mpol=mpol,
        ntor=ntor,
        nfp=int(vmec_input.nfp),
        lasym=bool(vmec_input.lasym),
        signgs=int(vmec_input.signgs),
        gamma=float(vmec_input.gamma),
        ntheta_even=2 * (ntheta // 2),
        nzeta=nzeta,
    )


def _profiles(vmec_input: Any, sizes: _Sizes, mass_half) -> dict[str, np.ndarray]:
    """The input-dependent radial profiles, passed to the compiled stage as data."""
    s_full = np.arange(sizes.ns) / (sizes.ns - 1.0)
    s_half = (np.arange(sizes.ns - 1) + 0.5) / (sizes.ns - 1.0)
    if mass_half is None:
        mass_half = mass_profile(vmec_input, s_half)
    return {
        "phip_full": toroidal_flux_derivative(vmec_input, s_full),
        "phip_half": toroidal_flux_derivative(vmec_input, s_half),
        "mass_half": np.asarray(mass_half, dtype=np.float64),
    }


def _grids(sizes: _Sizes) -> tuple[np.ndarray, np.ndarray]:
    theta = 2.0 * math.pi * np.arange(sizes.ntheta_eff) / sizes.ntheta_even
    zeta = 2.0 * math.pi * np.arange(sizes.nzeta) / sizes.nzeta
    return theta, zeta


@functools.lru_cache(maxsize=16)
def _kernels(sizes: _Sizes) -> dict[str, Any]:
    """The Nyquist and low-pass DFT kernels, which depend on the sizes only."""
    theta, zeta = _grids(sizes)
    theta_reduced = theta[: sizes.ntheta_reduced]
    xm_nyq, xn_nyq = _mode_table(sizes.mnyq + 1, sizes.nnyq, sizes.nfp)
    cos_kernel, sin_kernel = _nyquist_kernels(
        theta_reduced,
        zeta,
        sizes.ntheta_reduced,
        sizes.nzeta,
        sizes.mnyq,
        sizes.nnyq,
        xm_nyq,
        xn_nyq // sizes.nfp,
    )
    low_pass = _low_pass_kernels(
        theta_reduced,
        zeta,
        sizes.ntheta_reduced,
        sizes.nzeta,
        sizes.mpol,
        sizes.ntor,
        sizes.nfp,
        sizes.mnyq,
        sizes.nnyq,
    )
    return {"cos_kernel": cos_kernel, "sin_kernel": sin_kernel, "low_pass": low_pass}


def _make_setup(sizes: _Sizes, profiles, kernels) -> _Setup:
    ns = sizes.ns
    lasym = sizes.lasym
    ntheta_even = sizes.ntheta_even
    ntheta_reduced = sizes.ntheta_reduced
    ntheta_eff = sizes.ntheta_eff
    nzeta = sizes.nzeta

    theta, zeta = _grids(sizes)
    if lasym:
        w_int = np.full(ntheta_eff, 1.0 / (nzeta * ntheta_even))
    else:
        w_int = np.full(ntheta_eff, 1.0 / (nzeta * (ntheta_reduced - 1)))
        w_int[0] /= 2.0
        w_int[-1] /= 2.0
    # (theta, zeta) -> (-theta, -zeta) on the full grid, for the reduced range.
    zeta_reversed = (nzeta - np.arange(nzeta)) % nzeta
    theta_reversed = (ntheta_even - np.arange(ntheta_reduced)) % ntheta_even

    xm, xn = _mode_table(sizes.mpol, sizes.ntor, sizes.nfp)
    xm_nyq, xn_nyq = _mode_table(sizes.mnyq + 1, sizes.nnyq, sizes.nfp)

    s_full = np.arange(ns) / (ns - 1.0)
    sqrt_s_full = np.sqrt(s_full)
    sqrt_s_full[-1] = 1.0
    sqrt_s_half = np.sqrt((np.arange(ns - 1) + 0.5) / (ns - 1.0))
    # Odd-m coefficients are divided by sqrt(s), the axis taking the value of
    # the first surface (RadialProfiles::scalxc).
    odd_scale = 1.0 / np.maximum(sqrt_s_full, math.sqrt(1.0 / (ns - 1)))

    return _Setup(
        ns=ns,
        mpol=sizes.mpol,
        ntor=sizes.ntor,
        nfp=sizes.nfp,
        lthreed=sizes.ntor > 0,
        lasym=lasym,
        signgs=sizes.signgs,
        gamma=sizes.gamma,
        ntheta_even=ntheta_even,
        ntheta_reduced=ntheta_reduced,
        ntheta_eff=ntheta_eff,
        nzeta=nzeta,
        theta=theta,
        zeta=zeta,
        w_int=w_int,
        zeta_reversed=zeta_reversed,
        theta_reversed=theta_reversed,
        xm=xm,
        xn=xn,
        xm_nyq=xm_nyq,
        xn_nyq=xn_nyq,
        sqrt_s_full=sqrt_s_full,
        sqrt_s_half=sqrt_s_half,
        odd_scale=odd_scale,
        phip_full=profiles["phip_full"],
        phip_half=profiles["phip_half"],
        mass_half=profiles["mass_half"],
        radial_blending=2.0 * _P_DAMP * (1.0 - s_full),
        cos_kernel=kernels["cos_kernel"],
        sin_kernel=kernels["sin_kernel"],
        low_pass=kernels["low_pass"],
    )


def _mode_table(m_size: int, n_size: int, nfp: int) -> tuple[np.ndarray, np.ndarray]:
    xm = [0] * (n_size + 1)
    xn = list(range(n_size + 1))
    for m in range(1, m_size):
        xm.extend([m] * (2 * n_size + 1))
        xn.extend(range(-n_size, n_size + 1))
    return np.asarray(xm, dtype=np.int64), nfp * np.asarray(xn, dtype=np.int64)


def _nyquist_kernels(
    theta, zeta, ntheta_reduced, nzeta, mnyq, nnyq, xm_nyq, n_nyq
) -> tuple[np.ndarray, np.ndarray]:
    """Dense forward-DFT kernels ``(mn, k, l)`` for cos(mu - nv) and sin(mu - nv)."""
    m = np.arange(mnyq + 1)
    mscale = np.where(m == 0, 1.0, math.sqrt(2.0))
    int_norm = 1.0 / (nzeta * (ntheta_reduced - 1))
    cosmui = np.cos(np.outer(m, theta)) * mscale[:, None] * int_norm
    sinmui = np.sin(np.outer(m, theta)) * mscale[:, None] * int_norm
    cosmui[:, 0] /= 2.0
    cosmui[:, -1] /= 2.0
    if mnyq != 0:
        cosmui[mnyq] /= 2.0

    n = np.arange(nnyq + 1)
    nscale = np.where(n == 0, 1.0, math.sqrt(2.0))
    cosnv = np.cos(np.outer(zeta, n)) * nscale[None, :]
    sinnv = np.sin(np.outer(zeta, n)) * nscale[None, :]
    if nnyq != 0:
        cosnv[:, nnyq] /= 2.0

    abs_n = np.abs(n_nyq)
    sign_n = np.sign(n_nyq)
    dmult = mscale[xm_nyq] * nscale[abs_n] * 0.5
    dmult = np.where((xm_nyq == 0) | (n_nyq == 0), 2.0 * dmult, dmult)
    cos_kernel = dmult[:, None, None] * (
        cosnv[:, abs_n].T[:, :, None] * cosmui[xm_nyq][:, None, :]
        + sign_n[:, None, None]
        * sinnv[:, abs_n].T[:, :, None]
        * sinmui[xm_nyq][:, None, :]
    )
    sin_kernel = dmult[:, None, None] * (
        cosnv[:, abs_n].T[:, :, None] * sinmui[xm_nyq][:, None, :]
        - sign_n[:, None, None]
        * sinnv[:, abs_n].T[:, :, None]
        * cosmui[xm_nyq][:, None, :]
    )
    return cos_kernel, sin_kernel


def _low_pass_kernels(
    theta, zeta, ntheta_reduced, nzeta, mpol, ntor, nfp, mnyq, nnyq
) -> dict[str, tuple[np.ndarray, ...]]:
    """Projection onto the ``m < mpol, n <= ntor`` modes and its inverse.

    For each product basis ``"cc"``, ``"ss"``, ``"sc"``, ``"cs"`` (poloidal,
    toroidal) this returns the analysis kernel ``(mn, k, l)`` and the synthesis
    kernels of the value and of its theta and zeta derivatives
    (LowPassFilterCovariantB).
    """
    m = np.arange(mpol)
    n = np.arange(ntor + 1)
    mscale = np.where(m == 0, 1.0, math.sqrt(2.0))
    nscale = np.where(n == 0, 1.0, math.sqrt(2.0))
    int_norm = 1.0 / (nzeta * (ntheta_reduced - 1))
    poloidal = {
        "c": np.cos(np.outer(m, theta)) * mscale[:, None],
        "s": np.sin(np.outer(m, theta)) * mscale[:, None],
    }
    d_poloidal = {"c": -m[:, None] * poloidal["s"], "s": m[:, None] * poloidal["c"]}
    poloidal_analysis = {"c": poloidal["c"] * int_norm, "s": poloidal["s"] * int_norm}
    poloidal_analysis["c"][:, 0] /= 2.0
    poloidal_analysis["c"][:, -1] /= 2.0
    toroidal = {
        "c": np.cos(np.outer(zeta, n)) * nscale[None, :],
        "s": np.sin(np.outer(zeta, n)) * nscale[None, :],
    }
    d_toroidal = {
        "c": -nfp * n[None, :] * toroidal["s"],
        "s": nfp * n[None, :] * toroidal["c"],
    }
    dnorm = np.ones((mpol, ntor + 1))
    dnorm[m == mnyq, :] /= 2.0
    dnorm[:, (n == nnyq) & (n != 0)] /= 2.0
    ones = np.ones_like(dnorm)

    def kernel(pol, tor, weight):
        return (
            weight[:, :, None, None] * pol[:, None, None, :] * tor.T[None, :, :, None]
        ).reshape(mpol * (ntor + 1), nzeta, ntheta_reduced)

    kernels = {}
    for p, t in ("cc", "ss", "sc", "cs"):
        kernels[p + t] = (
            kernel(poloidal_analysis[p], toroidal[t], dnorm),
            kernel(poloidal[p], toroidal[t], ones),
            kernel(d_poloidal[p], toroidal[t], ones),
            kernel(poloidal[p], d_toroidal[t], ones),
        )
    return kernels


def _low_pass_series(setup: _Setup, field, bases: tuple[str, str]):
    """Low-pass filter a reduced-grid field onto two product bases.

    Returns the filtered field and its theta and zeta derivatives.
    """
    parts = []
    for basis in bases:
        analysis, *synthesis = (jnp.asarray(k) for k in setup.low_pass[basis])
        coefficients = jnp.einsum("jkl,mkl->jm", field, analysis)
        parts.append(
            [jnp.einsum("jm,mkl->jkl", coefficients, kernel) for kernel in synthesis]
        )
    value, dtheta, dzeta = (
        functools.reduce(jnp.add, terms) for terms in zip(*parts, strict=True)
    )
    return value, dtheta, dzeta


def _reflect(setup: _Setup, field):
    """The field at (-theta, -zeta) for theta in the reduced range [0, pi]."""
    return field[:, setup.zeta_reversed][:, :, setup.theta_reversed]


def _parity_split(setup: _Setup, field):
    """Reflection-even and reflection-odd parts on the reduced theta range."""
    reduced = field[:, :, : setup.ntheta_reduced]
    reflected = _reflect(setup, field)
    return 0.5 * (reduced + reflected), 0.5 * (reduced - reflected)


def _extend(setup: _Setup, even, odd):
    """Assemble a full-theta-range field from its reduced-range parity parts."""
    upper = np.arange(setup.ntheta_reduced, setup.ntheta_even)
    source = setup.ntheta_even - upper
    even_upper = even[:, setup.zeta_reversed][:, :, source]
    odd_upper = odd[:, setup.zeta_reversed][:, :, source]
    return jnp.concatenate([even + odd, even_upper - odd_upper], axis=2)


def _filter_cosine_type(setup: _Setup, field):
    """Low-pass filter of a field with cos(mu - nv) symmetric part (B_u, B_v)."""
    if not setup.lasym:
        return _low_pass_series(setup, field, ("cc", "ss"))
    even, odd = _parity_split(setup, field)
    s_value, s_dtheta, s_dzeta = _low_pass_series(setup, even, ("cc", "ss"))
    a_value, a_dtheta, a_dzeta = _low_pass_series(setup, odd, ("sc", "cs"))
    return (
        _extend(setup, s_value, a_value),
        _extend(setup, a_dtheta, s_dtheta),
        _extend(setup, a_dzeta, s_dzeta),
    )


def _filter_sine_type(setup: _Setup, field):
    """Low-pass filter of a field with sin(mu - nv) symmetric part (B_s)."""
    if not setup.lasym:
        return _low_pass_series(setup, field, ("sc", "cs"))
    even, odd = _parity_split(setup, field)
    s_value, s_dtheta, s_dzeta = _low_pass_series(setup, odd, ("sc", "cs"))
    a_value, a_dtheta, a_dzeta = _low_pass_series(setup, even, ("cc", "ss"))
    return (
        _extend(setup, a_value, s_value),
        _extend(setup, s_dtheta, a_dtheta),
        _extend(setup, s_dzeta, a_dzeta),
    )


def _nyquist_half(setup: _Setup, field, *, cosine: bool):
    """Nyquist-band coefficients of a half-grid field in the ``(mn, ns)`` layout.

    Returns the stellarator-symmetric coefficients (cos for ``cosine=True``,
    sin otherwise) and the non-symmetric ones, ``None`` for ``lasym=False``.
    """
    cos_kernel = jnp.asarray(setup.cos_kernel)
    sin_kernel = jnp.asarray(setup.sin_kernel)

    def transform(values, kernel):
        coefficients = jnp.einsum("jkl,mkl->mj", values, kernel)
        return jnp.pad(coefficients, ((0, 0), (1, 0)))

    if not setup.lasym:
        return transform(field, cos_kernel if cosine else sin_kernel), None
    even, odd = _parity_split(setup, field)
    if cosine:
        return transform(even, cos_kernel), transform(odd, sin_kernel)
    return transform(odd, sin_kernel), transform(even, cos_kernel)


def _currents(setup: _Setup, bsubs_mn, bsubu_mn, bsubv_mn, *, sign: float):
    """``sqrt(g) J^u`` and ``sqrt(g) J^v`` Fourier coefficients on the full grid.

    ``sign=1`` gives the cos(mu - nv) coefficients from the symmetric B_s, B_u,
    B_v coefficients, ``sign=-1`` the sin(mu - nv) ones (Compute_Currents).
    """
    ns = setup.ns
    delta_s = 1.0 / (ns - 1)
    m = setup.xm_nyq[:, None].astype(np.float64)
    n_nfp = setup.xn_nyq[:, None].astype(np.float64)
    odd = (setup.xm_nyq % 2 == 1)[:, None]
    sqrt_h_inner = setup.sqrt_s_half[:-1][None, :]
    sqrt_h_outer = setup.sqrt_s_half[1:][None, :]
    sqrt_f = setup.sqrt_s_full[1:-1][None, :]

    inner = slice(1, ns - 1)
    outer = slice(2, ns)
    t1_odd = (
        0.5
        * (sqrt_h_outer * bsubs_mn[:, outer] + sqrt_h_inner * bsubs_mn[:, inner])
        / sqrt_f
    )
    t1_even = 0.5 * (bsubs_mn[:, outer] + bsubs_mn[:, inner])

    def radial_derivative(values):
        v0 = values[:, inner] / sqrt_h_inner
        v1 = values[:, outer] / sqrt_h_outer
        odd_part = (v1 - v0) / delta_s * sqrt_f + 0.25 * (v0 + v1) / sqrt_f
        even_part = (values[:, outer] - values[:, inner]) / delta_s
        return jnp.where(odd, odd_part, even_part)

    t1 = jnp.where(odd, t1_odd, t1_even)
    t2 = radial_derivative(bsubu_mn)
    t3 = radial_derivative(bsubv_mn)
    curru = -sign * n_nfp * t1 - t3
    currv = -sign * m * t1 + t2

    def extrapolate(values):
        axis = jnp.where((setup.xm_nyq <= 1), 2.0 * values[:, 0] - values[:, 1], 0.0)
        with_axis = jnp.concatenate([axis[:, None], values], axis=1)
        edge = 2.0 * with_axis[:, -1] - with_axis[:, -2]
        return jnp.concatenate([with_axis, edge[:, None]], axis=1) / MU_0

    return extrapolate(curru), extrapolate(currv)


def _spectral_width(setup: _Setup, geometry: vmec_geometry.Geometry) -> jax.Array:
    """Spectral width ``<M>`` of the R, Z coefficients on the full grid.

    The axis value is 1 (FourierGeometry::ComputeSpectralWidth). The C++ solver
    evaluates it on the last accepted state before the final time step, so the
    two agree to the force tolerance.
    """
    m = np.arange(setup.mpol)
    names = ["r_cc", "z_sc"]
    if setup.lthreed:
        names += ["r_ss", "z_cs"]
    if setup.lasym:
        names += ["r_sc", "z_cc"]
        if setup.lthreed:
            names += ["r_cs", "z_ss"]
    norm = functools.reduce(
        jnp.add,
        [jnp.sum(jnp.asarray(getattr(geometry, name)) ** 2, axis=2) for name in names],
    )[:, 1:]
    m = m[1:].astype(np.float64)
    numerator = jnp.sum(norm * m ** (_SPECTRAL_WIDTH_P + _SPECTRAL_WIDTH_Q), axis=1)
    denominator = jnp.sum(norm * m**_SPECTRAL_WIDTH_P, axis=1)
    return jnp.concatenate([jnp.ones(1), (numerator / denominator)[1:]])


def _evaluate_profile(
    kind: str, coefficients, knots, values, x: np.ndarray
) -> np.ndarray:
    if kind == "power_series":
        return (
            np.polyval(coefficients[::-1], x) if coefficients.size else np.zeros_like(x)
        )
    if kind == "two_power":
        if coefficients.size < 3:
            return np.zeros_like(x)
        return coefficients[0] * (1.0 - x ** coefficients[1]) ** coefficients[2]
    if kind == "line_segment":
        n = min(knots.size, values.size)
        if n < 2:
            return np.zeros_like(x)
        # Linear on each knot interval and along the end segments outside.
        low = np.clip(np.searchsorted(knots[:n], x, side="right") - 1, 0, n - 2)
        t = (x - knots[low]) / (knots[low + 1] - knots[low])
        return (1.0 - t) * values[low] + t * values[low + 1]
    error_message = f"mass_profile does not evaluate the '{kind}' mass profile"
    raise NotImplementedError(error_message)


def _prepare_coefficients(
    setup: _Setup, coefficients, *, extrapolate_m0: bool
) -> jax.Array:
    """Apply the axis rules and the odd-m scaling of the solver's inverse DFT."""
    c = jnp.asarray(coefficients)
    m = np.arange(setup.mpol)
    # At the axis only m = 0 and m = 1 contribute, and the m = 1 modes (plus
    # the m = 0 lambda modes) take the value of the first surface.
    axis = jnp.where((m <= 1)[:, None], c[0], 0.0)
    if setup.mpol > 1:
        axis = axis.at[1].set(c[1, 1])
    if extrapolate_m0:
        axis = axis.at[0].set(c[1, 0])
    c = c.at[0].set(axis)
    scale = np.where((m % 2 == 1)[None, :], setup.odd_scale[:, None], 1.0)
    return c * scale[:, :, None]


def _r_terms(geometry, setup):
    """(coefficients, poloidal, toroidal) terms of R = sum r cos/sin products."""
    terms = [(geometry.r_cc, "c", "c"), (geometry.r_ss, "s", "s")]
    if setup.lasym:
        terms += [(geometry.r_sc, "s", "c"), (geometry.r_cs, "c", "s")]
    return terms


def _z_terms(geometry, setup):
    terms = [(geometry.z_sc, "s", "c"), (geometry.z_cs, "c", "s")]
    if setup.lasym:
        terms += [(geometry.z_cc, "c", "c"), (geometry.z_ss, "s", "s")]
    return terms


def _lambda_terms(geometry, setup, phip_f):
    scale = phip_f[:, None, None]
    terms = [
        (jnp.asarray(geometry.lambda_sc) * scale, "s", "c"),
        (jnp.asarray(geometry.lambda_cs) * scale, "c", "s"),
    ]
    if setup.lasym:
        terms += [
            (jnp.asarray(geometry.lambda_cc) * scale, "c", "c"),
            (jnp.asarray(geometry.lambda_ss) * scale, "s", "s"),
        ]
    return terms


def _real_space(setup: _Setup, terms, *, extrapolate_m0: bool = False) -> _RealSpace:
    """Inverse DFT of product-basis series onto the ``(zeta, theta)`` grid.

    Each term is ``(coefficients, poloidal, toroidal)`` with ``"c"`` or ``"s"``
    naming the cos or sin factor of each angle.
    """
    m = np.arange(setup.mpol, dtype=np.float64)
    n = setup.nfp * np.arange(setup.ntor + 1, dtype=np.float64)
    poloidal = {
        "c": np.cos(np.outer(m, setup.theta)),
        "s": np.sin(np.outer(m, setup.theta)),
    }
    d_poloidal = {"c": -m[:, None] * poloidal["s"], "s": m[:, None] * poloidal["c"]}
    toroidal = {
        "c": np.cos(np.outer(n / setup.nfp, setup.zeta)),
        "s": np.sin(np.outer(n / setup.nfp, setup.zeta)),
    }
    d_toroidal = {"c": -n[:, None] * toroidal["s"], "s": n[:, None] * toroidal["c"]}
    even = (np.arange(setup.mpol) % 2 == 0).astype(np.float64)
    odd = 1.0 - even
    prepared = [
        (_prepare_coefficients(setup, c, extrapolate_m0=extrapolate_m0), p, t)
        for c, p, t in terms
    ]

    def series(parity, derivative):
        parts = []
        for coefficient, p, t in prepared:
            masked = coefficient * parity[None, :, None]
            pol = d_poloidal[p] if derivative == "theta" else poloidal[p]
            tor = d_toroidal[t] if derivative == "zeta" else toroidal[t]
            parts.append(jnp.einsum("jmn,ml,nk->jkl", masked, pol, tor))
        return functools.reduce(jnp.add, parts)

    return _RealSpace(
        value_e=series(even, None),
        value_o=series(odd, None),
        dtheta_e=series(even, "theta"),
        dtheta_o=series(odd, "theta"),
        dzeta_e=series(even, "zeta"),
        dzeta_o=series(odd, "zeta"),
    )


def _blend_bsubv(setup, bsubv, bvco_h, gvv, gsqrt, guv, bsupu, lu_e, lu_o) -> jax.Array:
    """The half-grid B_v of the output stage.

    The output stage rebuilds the half-grid B_v from the full-grid estimate of the
    hybrid lambda force (MeshBledingBSubZeta) and restores the enclosed poloidal current
    on every surface (FixupPoloidalCurrent).
    """
    ns = setup.ns
    zero = jnp.zeros_like(bsubv[:1])
    bsubv_i = jnp.concatenate([zero, bsubv])
    bsubv_o = jnp.concatenate([bsubv, zero])
    gvv_gsqrt = gvv / gsqrt
    gg_i = jnp.concatenate([zero, gvv_gsqrt])
    gg_o = jnp.concatenate([gvv_gsqrt, zero])
    guv_bsupu = guv * bsupu
    gb_i = jnp.concatenate([zero, guv_bsupu])
    gb_o = jnp.concatenate([guv_bsupu, zero])
    sqrt_h = jnp.asarray(setup.sqrt_s_half)
    s_i = jnp.concatenate([jnp.zeros(1), sqrt_h])[:, None, None]
    s_o = jnp.concatenate([sqrt_h, jnp.zeros(1)])[:, None, None]

    alternative = 0.5 * (gg_i + gg_o) * lu_e + 0.5 * (gg_i * s_i + gg_o * s_o) * lu_o
    if setup.lthreed:
        alternative = alternative + 0.5 * (gb_i + gb_o)
    average = 0.5 * (bsubv_o + bsubv_i)
    blending = jnp.asarray(setup.radial_blending)[:, None, None]
    bsubv_full = average * (1.0 - blending) + alternative * blending

    # bsubv[jH] = 2 bsubv_full[jH + 1] - bsubv[jH + 1], swept inward from the
    # unchanged outermost half-grid point.
    j = np.arange(1, ns - 1)
    signed = ((-1.0) ** j)[:, None, None] * bsubv_full[1 : ns - 1]
    tail_sum = jnp.cumsum(signed[::-1], axis=0)[::-1]
    j_h = np.arange(ns - 2)
    blended = ((-1.0) ** (j_h + 1))[:, None, None] * 2.0 * tail_sum + (
        (-1.0) ** (ns - 2 - j_h)
    )[:, None, None] * bsubv[ns - 2]
    bsubv_blended = jnp.concatenate([blended, bsubv[ns - 2 :]])

    w_int = jnp.asarray(setup.w_int)[None, None, :]
    deviation = jnp.sum(bsubv_blended * w_int, axis=(1, 2)) - bvco_h
    return bsubv_blended - deviation[:, None, None]


def _to_combined(setup: _Setup, first, second, *, cosine: bool) -> jax.Array:
    """Product to combined basis in the ``(mn, ns)`` layout of the ``wout`` file.

    ``cosine=True`` reads ``first, second`` as the ``cc, ss`` coefficients of a
    cos(mu - nv) series, ``cosine=False`` as the ``sc, cs`` coefficients of a
    sin(mu - nv) series.
    """
    a = jnp.asarray(first)
    b = jnp.asarray(second)
    m = setup.xm
    n = setup.xn // setup.nfp
    abs_n = np.abs(n)
    sign_n = np.sign(n).astype(np.float64)
    a_mn = a[:, m, abs_n].T
    b_mn = b[:, m, abs_n].T
    sign_b = 1.0 if cosine else -1.0
    m0_value = a_mn if cosine else -b_mn
    combined = jnp.where(
        (m == 0)[:, None],
        m0_value,
        jnp.where(
            (n == 0)[:, None], a_mn, 0.5 * (a_mn + sign_b * sign_n[:, None] * b_mn)
        ),
    )
    # m > 0, n != 0 modes are not stored at the axis.
    axis_mask = ((m > 0) & (n != 0))[:, None] & (np.arange(setup.ns) == 0)[None, :]
    return jnp.where(axis_mask, 0.0, combined)


def _lambda_to_half_grid(setup: _Setup, lambda_full: jax.Array) -> jax.Array:
    """Radial interpolation of lambda onto the half grid (classic ``lmns``)."""
    ns = setup.ns
    sm = setup.sqrt_s_half / setup.sqrt_s_full[1:]
    sp = np.empty_like(sm)
    sp[1:] = setup.sqrt_s_half[1:] / setup.sqrt_s_full[1:-1]
    sp[0] = sm[0]
    outside = lambda_full[:, 1:]
    inside = lambda_full[:, :-1]
    low_m = (setup.xm <= 1)[:, None] & (np.arange(ns - 1) == 0)[None, :]
    inside = jnp.where(low_m, outside, inside)
    odd = (setup.xm % 2 == 1)[:, None]
    half = jnp.where(
        odd,
        0.5 * (sm[None, :] * outside + sp[None, :] * inside),
        0.5 * (outside + inside),
    )
    return jnp.pad(half, ((0, 0), (1, 0)))


def _half_to_full(values_half: jax.Array) -> jax.Array:
    """Average onto the interior full grid, extrapolate linearly to axis and edge."""
    axis = 1.5 * values_half[0] - 0.5 * values_half[1]
    edge = 1.5 * values_half[-1] - 0.5 * values_half[-2]
    interior = 0.5 * (values_half[1:] + values_half[:-1])
    return jnp.concatenate([axis[None], interior, edge[None]])


def _pad_axis(values_half: jax.Array) -> jax.Array:
    """Half-grid profile in the ``wout`` layout: a zero in the axis slot."""
    return jnp.pad(values_half, (1, 0))


def _pad_both(values_interior: jax.Array) -> jax.Array:
    """Interior full-grid profile with zeros at axis and edge."""
    return jnp.pad(values_interior, (1, 1))


def _extrapolate_both(values_interior: jax.Array) -> jax.Array:
    """Interior full-grid profile extrapolated linearly to axis and edge."""
    axis = 2.0 * values_interior[0] - values_interior[1]
    edge = 2.0 * values_interior[-1] - values_interior[-2]
    return jnp.concatenate([axis[None], values_interior, edge[None]])


def _extrapolate_axis_column(coefficients: jax.Array) -> jax.Array:
    return coefficients.at[:, 0].set(2.0 * coefficients[:, 1] - coefficients[:, 2])


__all__ = [
    "MU_0",
    "UNKNOWN_DIAGNOSTICS",
    "WOUT_QUANTITIES",
    "mass_profile",
    "static_fields",
    "toroidal_flux_derivative",
    "wout_quantities",
]
