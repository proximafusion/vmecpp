"""JAX port of the VMEC++ output stage: ``wout`` arrays from a converged geometry.

The solver returns the equilibrium as a :class:`vmecpp.geometry.Geometry` in the
internal product basis. Objectives are usually written in the quantities of a
``wout`` file. This module maps the geometry to those quantities with the same
discretization as the C++ output stage: the full-grid inverse DFT with the odd-m
``sqrt(s)`` scaling, the half-grid Jacobian, metric and magnetic field, the
Nyquist-band forward DFT, and the radial interpolation rules of the profiles.

Only the geometry leaves are traced. Radial grids, angular grids, mode tables,
the toroidal-flux profile and the mass profile are concrete NumPy arrays taken
from the input, so :func:`wout_arrays` works under ``jax.jit`` and ``jax.grad``.
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

_WOUT_ARRAY_FIELDS = (
    "rmnc",
    "zmns",
    "lmns",
    "lmns_full",
    "gmnc",
    "bmnc",
    "bsubumnc",
    "bsubvmnc",
    "bsupumnc",
    "bsupvmnc",
    "bsubsmns",
    "iotas",
    "iotaf",
    "phi",
    "chi",
    "phipf",
    "chipf",
    "presf",
    "pres",
    "aspect",
    "volume_p",
    "volavgB",
    "betatotal",
    "b0",
    "xm",
    "xn",
    "xm_nyq",
    "xn_nyq",
)


@dataclasses.dataclass(frozen=True)
class WoutArrays:
    """The ``wout`` quantities of an equilibrium as JAX arrays.

    Two-dimensional Fourier arrays use the ``(mn, ns)`` layout of
    :class:`vmecpp.VmecWOut`. Half-grid arrays (``lmns``, ``gmnc``, ``bmnc``,
    ``bsub*``, ``bsup*``, ``iotas``, ``pres``) carry a zero in column 0;
    ``bsubsmns`` carries the linear extrapolation of the classic output there.
    """

    rmnc: jax.Array
    zmns: jax.Array
    lmns: jax.Array
    lmns_full: jax.Array
    gmnc: jax.Array
    bmnc: jax.Array
    bsubumnc: jax.Array
    bsubvmnc: jax.Array
    bsupumnc: jax.Array
    bsupvmnc: jax.Array
    bsubsmns: jax.Array
    iotas: jax.Array
    iotaf: jax.Array
    phi: jax.Array
    chi: jax.Array
    phipf: jax.Array
    chipf: jax.Array
    presf: jax.Array
    pres: jax.Array
    aspect: jax.Array
    volume_p: jax.Array
    volavgB: jax.Array
    betatotal: jax.Array
    b0: jax.Array
    xm: jax.Array
    xn: jax.Array
    xm_nyq: jax.Array
    xn_nyq: jax.Array


jax.tree_util.register_dataclass(
    WoutArrays, data_fields=list(_WOUT_ARRAY_FIELDS), meta_fields=[]
)


def wout_arrays(
    geometry: vmec_geometry.Geometry, vmec_input: Any, *, iota_half=None
) -> WoutArrays:
    """Map a converged geometry to the ``wout`` arrays of the C++ output stage.

    Args:
        geometry: The solved equilibrium in the internal product basis.
        vmec_input: The :class:`vmecpp.VmecInput` the geometry was solved for;
            supplies the grid sizes, mode tables and the flux and mass profiles.
        iota_half: Rotational transform on the half grid. Defaults to the ratio
            of the poloidal and toroidal flux increments of ``geometry``.

    The mass profile is evaluated in NumPy from the input and is not
    differentiated. Stellarator-symmetric geometries only.
    """
    setup = _make_setup(vmec_input)
    ns = setup.ns
    delta_s = 1.0 / (ns - 1)

    toroidal_flux = jnp.asarray(geometry.toroidal_flux)
    poloidal_flux = jnp.asarray(geometry.poloidal_flux)
    if iota_half is None:
        iota_h = jnp.diff(poloidal_flux) / jnp.diff(toroidal_flux)
    else:
        iota_h = jnp.asarray(iota_half, dtype=toroidal_flux.dtype)
    phip_f = jnp.asarray(setup.phip_full)
    phip_h = jnp.asarray(setup.phip_half)
    chip_h = iota_h * phip_h

    # Inverse DFT to the full-grid real-space geometry, split by m parity.
    r = _real_space(setup, geometry.r_cc, geometry.r_ss, cosine=True)
    z = _real_space(setup, geometry.z_sc, geometry.z_cs, cosine=False)
    # The solver normalizes lambda by lamscale / phi'; undo it so the theta
    # derivative combines with phi' as in the C++ B^v.
    scaled_lambda_sc = jnp.asarray(geometry.lambda_sc) * phip_f[:, None, None]
    scaled_lambda_cs = jnp.asarray(geometry.lambda_cs) * phip_f[:, None, None]
    lam = _real_space(
        setup, scaled_lambda_sc, scaled_lambda_cs, cosine=False, extrapolate_m0=True
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

    w_int = jnp.asarray(setup.w_int)[None, None, :]
    signgs = setup.signgs
    dvds_h = signgs * jnp.sum(gsqrt * w_int, axis=(1, 2))
    mass_h = jnp.asarray(setup.mass_half)
    pres_h = mass_h / dvds_h**setup.gamma if setup.gamma != 0.0 else mass_h
    total_pressure = magnetic_pressure + pres_h[:, None, None]
    bvco_h = jnp.sum(bsubv * w_int, axis=(1, 2))

    # The output stage rebuilds B_v from the hybrid lambda force and then
    # low-pass filters both covariant components to the solver's mode range.
    bsubv_out = _low_pass(
        setup, _blend_bsubv(setup, bsubv, bvco_h, gvv, gsqrt, guv, bsupu, lu_e, lu_o)
    )
    bsubu_out = _low_pass(setup, bsubu)

    # Covariant B_s on the half grid (ComputeRemainingMetric, ComputeBSubSOnHalfGrid).
    rv12 = half(r.dzeta_e, r.dzeta_o)
    zv12 = half(z.dzeta_e, z.dzeta_o)
    rs12 = rs + 0.5 * (inner(r.value_o) + outer(r.value_o)) / (2.0 * sqrt_s_h)
    zs12 = zs + 0.5 * (inner(z.value_o) + outer(z.value_o)) / (2.0 * sqrt_s_h)
    gsu = rs12 * ru12 + zs12 * zu12
    gsv = rs12 * rv12 + zs12 * zv12
    bsubs = bsupu * gsu + bsupv * gsv

    # Nyquist-band forward DFT of the half-grid fields.
    cos_kernel = jnp.asarray(setup.cos_kernel)
    sin_kernel = jnp.asarray(setup.sin_kernel)

    def half_grid_cos(field):
        coefficients = jnp.einsum("jkl,mkl->mj", field, cos_kernel)
        return jnp.pad(coefficients, ((0, 0), (1, 0)))

    gmnc = half_grid_cos(gsqrt)
    bmnc = half_grid_cos(mod_b)
    bsubumnc = half_grid_cos(bsubu_out)
    bsubvmnc = half_grid_cos(bsubv_out)
    bsupumnc = half_grid_cos(bsupu)
    bsupvmnc = half_grid_cos(bsupv)
    bsubsmns = jnp.pad(jnp.einsum("jkl,mkl->mj", bsubs, sin_kernel), ((0, 0), (1, 0)))
    # The classic output extrapolates the half-grid B_s one full step beyond
    # the innermost half-grid point.
    bsubsmns = bsubsmns.at[:, 0].set(2.0 * bsubsmns[:, 1] - bsubsmns[:, 2])

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

    # Radial profiles.
    iota_f = _half_to_full(iota_h)
    chip_f = _half_to_full(chip_h)
    iotas = jnp.pad(iota_h, (1, 0))
    chi = signgs * poloidal_flux
    phipf = signgs * 2.0 * math.pi * phip_f
    chipf = signgs * 2.0 * math.pi * chip_f
    presf = _half_to_full(pres_h) / MU_0
    pres = jnp.pad(pres_h, (1, 0)) / MU_0

    # Scalars (ComputeThreed1GeometricMagneticQuantities).
    w_lcfs = jnp.asarray(setup.w_int)[None, :]
    r_lcfs = r.value_e[-1] + r.value_o[-1]
    zu_lcfs = z.dtheta_e[-1] + z.dtheta_o[-1]
    cross_area_p = 2.0 * math.pi * jnp.abs(jnp.sum(r_lcfs * zu_lcfs * w_lcfs))
    volume_p = 2.0 * math.pi**2 * jnp.abs(jnp.sum(r_lcfs**2 * zu_lcfs * w_lcfs))
    rmajor_p = volume_p / (2.0 * math.pi * cross_area_p)
    aminor_p = jnp.sqrt(cross_area_p / math.pi)
    aspect = rmajor_p / aminor_p

    anorm = 2.0 * math.pi * delta_s
    vnorm = 2.0 * math.pi * anorm
    sump = vnorm * jnp.sum(dvds_h * pres_h)
    tau_w = signgs * w_int * gsqrt
    sumbtot = 2.0 * (vnorm * jnp.sum(total_pressure * tau_w) - sump)
    betatotal = 2.0 * sump / sumbtot
    volavgb = jnp.sqrt(jnp.abs(sumbtot / volume_p))
    bvco_out = jnp.sum(bsubv_out * w_int, axis=(1, 2))
    b0 = (1.5 * bvco_out[0] - 0.5 * bvco_out[1]) / r.value_e[0, 0, 0]

    return WoutArrays(
        rmnc=rmnc,
        zmns=zmns,
        lmns=lmns,
        lmns_full=lmns_full,
        gmnc=gmnc,
        bmnc=bmnc,
        bsubumnc=bsubumnc,
        bsubvmnc=bsubvmnc,
        bsupumnc=bsupumnc,
        bsupvmnc=bsupvmnc,
        bsubsmns=bsubsmns,
        iotas=iotas,
        iotaf=iota_f,
        phi=toroidal_flux,
        chi=chi,
        phipf=phipf,
        chipf=chipf,
        presf=presf,
        pres=pres,
        aspect=aspect,
        volume_p=volume_p,
        volavgB=volavgb,
        betatotal=betatotal,
        b0=b0,
        xm=jnp.asarray(setup.xm),
        xn=jnp.asarray(setup.xn),
        xm_nyq=jnp.asarray(setup.xm_nyq),
        xn_nyq=jnp.asarray(setup.xn_nyq),
    )


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

    Arrays have shape ``(ns, nzeta, ntheta_reduced)``; the odd-m part carries
    the solver's ``1 / sqrt(s)`` scaling.
    """

    value_e: jax.Array
    value_o: jax.Array
    dtheta_e: jax.Array
    dtheta_o: jax.Array
    dzeta_e: jax.Array
    dzeta_o: jax.Array


@dataclasses.dataclass(frozen=True)
class _Setup:
    """Static grids, mode tables and profiles of one input (all NumPy)."""

    ns: int
    mpol: int
    ntor: int
    nfp: int
    lthreed: bool
    signgs: int
    gamma: float
    ntheta_reduced: int
    nzeta: int
    theta: np.ndarray
    zeta: np.ndarray
    w_int: np.ndarray
    xm: np.ndarray
    xn: np.ndarray
    xm_nyq: np.ndarray
    xn_nyq: np.ndarray
    sqrt_s_full: np.ndarray
    sqrt_s_half: np.ndarray
    odd_scale: np.ndarray
    phip_full: np.ndarray
    phip_half: np.ndarray
    mass_half: np.ndarray
    radial_blending: np.ndarray
    cos_kernel: np.ndarray
    sin_kernel: np.ndarray
    low_pass_analysis: tuple[np.ndarray, np.ndarray]
    low_pass_synthesis: tuple[np.ndarray, np.ndarray]


def _make_setup(vmec_input: Any) -> _Setup:
    if vmec_input.lasym:
        error_message = (
            "wout_arrays requires a stellarator-symmetric input (lasym=false)"
        )
        raise ValueError(error_message)
    if not isinstance(vmec_input.mpol, int) or not isinstance(vmec_input.ntor, int):
        error_message = "wout_arrays requires scalar mpol and ntor"
        raise ValueError(error_message)
    mpol = vmec_input.mpol
    ntor = vmec_input.ntor
    nfp = vmec_input.nfp
    ns = int(np.asarray(vmec_input.ns_array)[-1])
    if ns < 3:
        error_message = "wout_arrays requires ns >= 3"
        raise ValueError(error_message)

    # Sizes::computeDerivedSizes
    ntheta = max(vmec_input.ntheta, 2 * mpol + 6)
    ntheta_even = 2 * (ntheta // 2)
    ntheta_reduced = ntheta_even // 2 + 1
    nzeta = (
        max(vmec_input.nzeta, 1) if ntor == 0 else max(vmec_input.nzeta, 2 * ntor + 4)
    )
    mnyq = max(0, ntheta_even // 2, mpol - 1)
    nnyq = max(0, nzeta // 2, ntor)

    theta = 2.0 * math.pi * np.arange(ntheta_reduced) / ntheta_even
    zeta = 2.0 * math.pi * np.arange(nzeta) / nzeta
    w_int = np.full(ntheta_reduced, 1.0 / (nzeta * (ntheta_reduced - 1)))
    w_int[0] /= 2.0
    w_int[-1] /= 2.0

    xm, xn = _mode_table(mpol, ntor, nfp)
    xm_nyq, xn_nyq = _mode_table(mnyq + 1, nnyq, nfp)

    s_full = np.arange(ns) / (ns - 1.0)
    sqrt_s_full = np.sqrt(s_full)
    sqrt_s_full[-1] = 1.0
    sqrt_s_half = np.sqrt((np.arange(ns - 1) + 0.5) / (ns - 1.0))
    # Odd-m coefficients are divided by sqrt(s), the axis taking the value of
    # the first surface (RadialProfiles::scalxc).
    odd_scale = 1.0 / np.maximum(sqrt_s_full, math.sqrt(1.0 / (ns - 1)))

    phip_full = toroidal_flux_derivative(vmec_input, s_full)
    phip_half = toroidal_flux_derivative(vmec_input, sqrt_s_half**2)
    mass_half = mass_profile(vmec_input, sqrt_s_half**2)
    radial_blending = 2.0 * _P_DAMP * (1.0 - s_full)

    cos_kernel, sin_kernel = _nyquist_kernels(
        theta, zeta, ntheta_reduced, nzeta, mnyq, nnyq, xm_nyq, xn_nyq // nfp
    )
    low_pass_analysis, low_pass_synthesis = _low_pass_kernels(
        theta, zeta, ntheta_reduced, nzeta, mpol, ntor, mnyq, nnyq
    )

    return _Setup(
        ns=ns,
        mpol=mpol,
        ntor=ntor,
        nfp=nfp,
        lthreed=ntor > 0,
        signgs=int(vmec_input.signgs),
        gamma=float(vmec_input.gamma),
        ntheta_reduced=ntheta_reduced,
        nzeta=nzeta,
        theta=theta,
        zeta=zeta,
        w_int=w_int,
        xm=xm,
        xn=xn,
        xm_nyq=xm_nyq,
        xn_nyq=xn_nyq,
        sqrt_s_full=sqrt_s_full,
        sqrt_s_half=sqrt_s_half,
        odd_scale=odd_scale,
        phip_full=phip_full,
        phip_half=phip_half,
        mass_half=mass_half,
        radial_blending=radial_blending,
        cos_kernel=cos_kernel,
        sin_kernel=sin_kernel,
        low_pass_analysis=low_pass_analysis,
        low_pass_synthesis=low_pass_synthesis,
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
    theta, zeta, ntheta_reduced, nzeta, mpol, ntor, mnyq, nnyq
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Projection onto the ``m < mpol, n <= ntor`` cos-cos and sin-sin modes.

    Returns the analysis kernels ``(mn, k, l)`` and the synthesis kernels of
    the two parities (LowPassFilterCovariantB).
    """
    m = np.arange(mpol)
    n = np.arange(ntor + 1)
    mscale = np.where(m == 0, 1.0, math.sqrt(2.0))
    nscale = np.where(n == 0, 1.0, math.sqrt(2.0))
    int_norm = 1.0 / (nzeta * (ntheta_reduced - 1))
    cosmu = np.cos(np.outer(m, theta)) * mscale[:, None]
    sinmu = np.sin(np.outer(m, theta)) * mscale[:, None]
    cosmui = cosmu * int_norm
    sinmui = sinmu * int_norm
    cosmui[:, 0] /= 2.0
    cosmui[:, -1] /= 2.0
    cosnv = np.cos(np.outer(zeta, n)) * nscale[None, :]
    sinnv = np.sin(np.outer(zeta, n)) * nscale[None, :]
    dnorm = np.ones((mpol, ntor + 1))
    dnorm[m == mnyq, :] /= 2.0
    dnorm[:, (n == nnyq) & (n != 0)] /= 2.0

    def kernel(poloidal, toroidal, weight):
        return (
            weight[:, :, None, None]
            * poloidal[:, None, None, :]
            * toroidal.T[None, :, :, None]
        ).reshape(mpol * (ntor + 1), nzeta, ntheta_reduced)

    analysis = (kernel(cosmui, cosnv, dnorm), kernel(sinmui, sinnv, dnorm))
    synthesis = (
        kernel(cosmu, cosnv, np.ones_like(dnorm)),
        kernel(sinmu, sinnv, np.ones_like(dnorm)),
    )
    return analysis, synthesis


def _low_pass(setup: _Setup, field: jax.Array) -> jax.Array:
    """Fourier low-pass filter of a half-grid field to the solver's mode range."""
    parts = [
        jnp.einsum(
            "jm,mkl->jkl",
            jnp.einsum("jkl,mkl->jm", field, jnp.asarray(analysis)),
            jnp.asarray(synthesis),
        )
        for analysis, synthesis in zip(
            setup.low_pass_analysis, setup.low_pass_synthesis, strict=True
        )
    ]
    return functools.reduce(jnp.add, parts)


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
    error_message = f"wout_arrays does not evaluate the '{kind}' mass profile"
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


def _real_space(
    setup: _Setup, first, second, *, cosine: bool, extrapolate_m0: bool = False
) -> _RealSpace:
    """Inverse DFT of one product-basis series onto the reduced angular grid.

    ``cosine=True`` reads ``first, second`` as the ``cc, ss`` coefficients of a
    cosine-type quantity; ``cosine=False`` as the ``sc, cs`` coefficients of a
    sine-type quantity.
    """
    a = _prepare_coefficients(setup, first, extrapolate_m0=extrapolate_m0)
    b = _prepare_coefficients(setup, second, extrapolate_m0=extrapolate_m0)
    m = np.arange(setup.mpol, dtype=np.float64)
    n = setup.nfp * np.arange(setup.ntor + 1, dtype=np.float64)
    cos_m = np.cos(np.outer(m, setup.theta))
    sin_m = np.sin(np.outer(m, setup.theta))
    cos_n = np.cos(np.outer(n / setup.nfp, setup.zeta))
    sin_n = np.sin(np.outer(n / setup.nfp, setup.zeta))
    d_cos_m = -m[:, None] * sin_m
    d_sin_m = m[:, None] * cos_m
    d_cos_n = -n[:, None] * sin_n
    d_sin_n = n[:, None] * cos_n
    even = (np.arange(setup.mpol) % 2 == 0).astype(np.float64)
    odd = 1.0 - even

    if cosine:
        terms = (
            (a, cos_m, d_cos_m, cos_n, d_cos_n),
            (b, sin_m, d_sin_m, sin_n, d_sin_n),
        )
    else:
        terms = (
            (a, sin_m, d_sin_m, cos_n, d_cos_n),
            (b, cos_m, d_cos_m, sin_n, d_sin_n),
        )

    def series(parity, poloidal, toroidal):
        parts = []
        for coefficient, p, dp, q, dq in terms:
            masked = coefficient * parity[None, :, None]
            pol = {"value": p, "dtheta": dp}[poloidal]
            tor = {"value": q, "dzeta": dq}[toroidal]
            parts.append(jnp.einsum("jmn,ml,nk->jkl", masked, pol, tor))
        return functools.reduce(jnp.add, parts)

    return _RealSpace(
        value_e=series(even, "value", "value"),
        value_o=series(odd, "value", "value"),
        dtheta_e=series(even, "dtheta", "value"),
        dtheta_o=series(odd, "dtheta", "value"),
        dzeta_e=series(even, "value", "dzeta"),
        dzeta_o=series(odd, "value", "dzeta"),
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
    """Product to combined basis in the ``(mn, ns)`` layout of the ``wout`` file."""
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


def _lambda_to_half_grid(setup: _Setup, lmns_full: jax.Array) -> jax.Array:
    """Radial interpolation of lambda onto the half grid (classic ``lmns``)."""
    ns = setup.ns
    sm = setup.sqrt_s_half / setup.sqrt_s_full[1:]
    sp = np.empty_like(sm)
    sp[1:] = setup.sqrt_s_half[1:] / setup.sqrt_s_full[1:-1]
    sp[0] = sm[0]
    outside = lmns_full[:, 1:]
    inside = lmns_full[:, :-1]
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


__all__ = [
    "MU_0",
    "WoutArrays",
    "mass_profile",
    "toroidal_flux_derivative",
    "wout_arrays",
]
