# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Stellarator-optimization objectives on a converged VMEC++ equilibrium.

Each function takes a converged equilibrium (a :class:`vmecpp.VmecOutput` or its
:class:`vmecpp.VmecWOut`) and returns a scalar objective (or a profile). The
aspect ratio, iota profile, magnetic well, and mirror ratio use the same
definitions as SIMSOPT, so the two agree to numerical precision. The Boozer-
coordinate transform (``boozer_spectrum``) and the quasi-symmetry residual built
on it are validated by self-consistency (``boozer_roundtrip_residual``): the
transform reproduces ``|B|`` from its own Boozer spectrum.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

if TYPE_CHECKING:
    from vmecpp import VmecOutput, VmecWOut


def _wout(equilibrium: VmecOutput | VmecWOut) -> VmecWOut:
    """Return the ``VmecWOut`` from a ``VmecOutput`` or a ``VmecWOut``."""
    return cast("VmecWOut", getattr(equilibrium, "wout", equilibrium))


def _mode_index(xm: np.ndarray, xn: np.ndarray, m: int, n: int) -> int:
    """Index of the ``(m, n)`` Fourier mode in the ``(xm, xn)`` mode set."""
    return int(np.argmin(np.abs(xm - m) + np.abs(xn - n)))


def aspect_ratio(equilibrium: VmecOutput | VmecWOut) -> float:
    """Plasma aspect ratio R0 / a, as reported by VMEC (``wout.aspect``).

    Matches ``simsopt.mhd.Vmec.aspect``.
    """
    return float(_wout(equilibrium).aspect)


def iota_profile(equilibrium: VmecOutput | VmecWOut) -> tuple[np.ndarray, np.ndarray]:
    """Rotational transform on the full-grid surfaces, returned as ``(s, iota)``."""
    iotaf = np.asarray(_wout(equilibrium).iotaf, dtype=float)
    s = np.linspace(0.0, 1.0, iotaf.size)
    return s, iotaf


def iota_profile_residual(equilibrium: VmecOutput | VmecWOut, target) -> float:
    """Mean-squared residual of the iota profile against ``target``.

    ``target`` is either a callable ``target(s)`` or an array on the full grid.
    """
    s, iota = iota_profile(equilibrium)
    reference = np.asarray(target(s) if callable(target) else target, dtype=float)
    delta = iota - reference
    return float(np.mean(delta * delta))


def magnetic_well(equilibrium: VmecOutput | VmecWOut) -> float:
    r"""Vacuum magnetic well :math:`W = (V'(0) - V'(1)) / V'(0)`.

    :math:`V'(s) = 4\pi^2 |\sqrt{g}_{0,0}|` is built from the m=n=0 Jacobian
    coefficient (``gmnc`` on the half grid) and extrapolated half a grid point to
    s=0 and s=1. Positive :math:`W` is favorable for interchange stability. Matches
    ``simsopt.mhd.Vmec.vacuum_well``.
    """
    gmnc = np.asarray(_wout(equilibrium).gmnc, dtype=float)
    dV_ds = 4.0 * np.pi * np.pi * np.abs(gmnc[0, 1:])
    dV_ds_axis = 1.5 * dV_ds[0] - 0.5 * dV_ds[1]
    dV_ds_edge = 1.5 * dV_ds[-1] - 0.5 * dV_ds[-2]
    return float((dV_ds_axis - dV_ds_edge) / dV_ds_axis)


def field_strength_on_surface(
    equilibrium: VmecOutput | VmecWOut,
    surface: int = -1,
    n_theta: int = 64,
    n_zeta: int = 64,
) -> np.ndarray:
    """``|B|`` on a half-grid flux surface over a (theta, zeta) grid of the full torus.

    Reconstructed from the Nyquist ``|B|`` spectrum (``bmnc``) at radial index
    ``surface`` (default -1, the half-grid surface nearest the boundary). ``|B|`` is
    a scalar field, so its surface extrema are independent of the angle coordinates.
    """
    wout = _wout(equilibrium)
    bmnc = np.asarray(wout.bmnc, dtype=float)[:, surface]
    xm = np.asarray(wout.xm_nyq, dtype=float)
    xn = np.asarray(wout.xn_nyq, dtype=float)
    theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    zeta = np.linspace(0.0, 2.0 * np.pi, n_zeta, endpoint=False)
    angle = (
        xm[:, None, None] * theta[None, :, None]
        - xn[:, None, None] * zeta[None, None, :]
    )
    return np.einsum("m,mij->ij", bmnc, np.cos(angle))


def mirror_ratio(
    equilibrium: VmecOutput | VmecWOut,
    surface: int = -1,
    n_theta: int = 64,
    n_zeta: int = 64,
) -> float:
    """Mirror ratio ``(Bmax - Bmin) / (Bmax + Bmin)`` of ``|B|`` on a flux surface."""
    mod_b = field_strength_on_surface(equilibrium, surface, n_theta, n_zeta)
    b_max = float(mod_b.max())
    b_min = float(mod_b.min())
    return (b_max - b_min) / (b_max + b_min)


def _boozer_transform(
    wout: VmecWOut,
    surface: int,
    m_boozer: int | None,
    n_boozer: int | None,
    n_theta: int | None,
    n_zeta: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Core Hirshman VMEC-to-Boozer transform on one half-grid flux surface.

    Returns ``(xm_b, xn_b, bmnc_b, theta_b, zeta_b, mod_b)``: the Boozer mode
    numbers and ``|B|`` cosine coefficients, plus the Boozer angles and ``|B|`` on
    the real-space evaluation grid (the last three support the self-consistency
    check).
    """
    nfp = int(wout.nfp)
    xm = np.asarray(wout.xm, dtype=float)
    xn = np.asarray(wout.xn, dtype=float)
    xm_nyq = np.asarray(wout.xm_nyq, dtype=float)
    xn_nyq = np.asarray(wout.xn_nyq, dtype=float)
    lmns = np.asarray(wout.lmns, dtype=float)[:, surface]
    bmnc = np.asarray(wout.bmnc, dtype=float)[:, surface]
    bsubumnc = np.asarray(wout.bsubumnc, dtype=float)[:, surface]
    bsubvmnc = np.asarray(wout.bsubvmnc, dtype=float)[:, surface]
    iota = float(np.asarray(wout.iotas, dtype=float)[surface])

    zero = _mode_index(xm_nyq, xn_nyq, 0, 0)
    big_i = bsubumnc[zero]
    big_g = bsubvmnc[zero]
    denom = big_g + iota * big_i

    # Periodic potential w (Hirshman Eq. 18, stellarator-symmetric: sine series).
    w_sin = np.zeros_like(bsubumnc)
    has_m = xm_nyq != 0.0
    w_sin[has_m] = bsubumnc[has_m] / xm_nyq[has_m]
    only_n = (~has_m) & (xn_nyq != 0.0)
    w_sin[only_n] = -bsubvmnc[only_n] / xn_nyq[only_n]

    mpol = round(float(xm.max()))
    ntor = round(float(xn.max()) / nfp) if nfp else 0
    if m_boozer is None:
        m_boozer = 6 * mpol
    if n_boozer is None:
        n_boozer = 6 * ntor
    if n_theta is None:
        n_theta = max(4 * (m_boozer + 1), 16)
    if n_zeta is None:
        n_zeta = max(4 * (n_boozer + 1), 16)

    # Integrate over one field period: the toroidal frequency per period is n
    # (the actual mode number is n * nfp), so n_zeta ~ 4 * n_boozer resolves it
    # without aliasing, and |B| is nfp-periodic so one period suffices.
    theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    zeta = np.linspace(0.0, 2.0 * np.pi / nfp, n_zeta, endpoint=False)
    th, ze = np.meshgrid(theta, zeta, indexing="ij")

    angle = xm[:, None, None] * th[None] - xn[:, None, None] * ze[None]
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    lam = np.einsum("m,mij->ij", lmns, sin_a)
    lam_t = np.einsum("m,mij->ij", lmns * xm, cos_a)
    lam_z = np.einsum("m,mij->ij", -lmns * xn, cos_a)

    angle_nyq = xm_nyq[:, None, None] * th[None] - xn_nyq[:, None, None] * ze[None]
    cos_n, sin_n = np.cos(angle_nyq), np.sin(angle_nyq)
    mod_b = np.einsum("m,mij->ij", bmnc, cos_n)
    w = np.einsum("m,mij->ij", w_sin, sin_n)
    w_t = np.einsum("m,mij->ij", w_sin * xm_nyq, cos_n)
    w_z = np.einsum("m,mij->ij", -w_sin * xn_nyq, cos_n)

    nu = (w - big_i * lam) / denom
    nu_t = (w_t - big_i * lam_t) / denom
    nu_z = (w_z - big_i * lam_z) / denom

    theta_b = th + lam + iota * nu
    zeta_b = ze + nu
    jacobian = (1.0 + lam_t) * (1.0 + nu_z) + (iota - lam_z) * nu_t

    modes = [(0, n * nfp) for n in range(n_boozer + 1)]
    modes += [
        (m, n * nfp)
        for m in range(1, m_boozer + 1)
        for n in range(-n_boozer, n_boozer + 1)
    ]
    xm_b = np.array([m for m, _ in modes], dtype=float)
    xn_b = np.array([n for _, n in modes], dtype=float)

    weight = jacobian * mod_b
    bmnc_b = np.empty(len(modes), dtype=float)
    for k, (m, n) in enumerate(modes):
        norm = 1.0 if (m == 0 and n == 0) else 2.0
        bmnc_b[k] = norm * float(np.mean(weight * np.cos(m * theta_b - n * zeta_b)))
    return xm_b, xn_b, bmnc_b, theta_b, zeta_b, mod_b


def boozer_spectrum(
    equilibrium: VmecOutput | VmecWOut,
    surface: int = -1,
    m_boozer: int | None = None,
    n_boozer: int | None = None,
    n_theta: int | None = None,
    n_zeta: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""``|B|`` in Boozer coordinates on a half-grid flux surface.

    Hirshman's VMEC-to-Boozer transform. With the covariant-field surface constants
    :math:`I = \widehat{B}_{\theta,00}` and :math:`G = \widehat{B}_{\zeta,00}`, the
    toroidal-angle shift is :math:`\nu = (w - I\lambda)/(G + \iota I)`, where the
    periodic potential :math:`w` has sine harmonics
    :math:`w_{mn} = \widehat{B}_{\theta,mn}/m = -\widehat{B}_{\zeta,mn}/n`. The
    Boozer angles are :math:`\theta_B = \theta + \lambda + \iota\nu` and
    :math:`\zeta_B = \zeta + \nu`, and ``|B|`` is resampled to Boozer-angle Fourier
    coefficients weighted by the coordinate Jacobian. Returns
    ``(xm_b, xn_b, bmnc_b)`` (stellarator-symmetric).
    """
    xm_b, xn_b, bmnc_b, *_ = _boozer_transform(
        _wout(equilibrium), surface, m_boozer, n_boozer, n_theta, n_zeta
    )
    return xm_b, xn_b, bmnc_b


def boozer_roundtrip_residual(
    equilibrium: VmecOutput | VmecWOut,
    surface: int = -1,
    m_boozer: int | None = None,
    n_boozer: int | None = None,
    n_theta: int | None = None,
    n_zeta: int | None = None,
) -> float:
    """Relative max error of reconstructing ``|B|`` from its Boozer spectrum.

    The VMEC angle grid is forward-mapped to Boozer angles, the Boozer ``|B|``
    spectrum is evaluated there, and the result is compared to the VMEC ``|B|`` on
    the same grid. A small residual confirms the transform is self-consistent: the
    ``|B|`` scalar field is faithfully represented in Boozer coordinates.
    """
    xm_b, xn_b, bmnc_b, theta_b, zeta_b, mod_b = _boozer_transform(
        _wout(equilibrium), surface, m_boozer, n_boozer, n_theta, n_zeta
    )
    basis = np.cos(
        xm_b[:, None, None] * theta_b[None] - xn_b[:, None, None] * zeta_b[None]
    )
    reconstructed = np.einsum("k,kij->ij", bmnc_b, basis)
    return float(np.max(np.abs(reconstructed - mod_b)) / np.max(np.abs(mod_b)))


def quasisymmetry_residual(
    equilibrium: VmecOutput | VmecWOut,
    helicity_m: int = 1,
    helicity_n: int = 0,
    surface: int = -1,
    **boozer_kwargs,
) -> float:
    r"""Boozer quasi-symmetry residual on a flux surface.

    With ``|B|`` in Boozer coordinates, quasi-symmetry of helicity ``(M, N)`` means
    ``|B|`` depends on the angles only through :math:`M\theta_B - N N_{fp}\zeta_B`.
    The residual is the fraction of spectral energy in the symmetry-breaking modes,
    :math:`\sum_{\text{break}} b_{mn}^2 / b_{00}^2`.
    """
    nfp = int(_wout(equilibrium).nfp)
    xm_b, xn_b, bmnc_b = boozer_spectrum(equilibrium, surface=surface, **boozer_kwargs)
    symmetric = xn_b * helicity_m == xm_b * helicity_n * nfp
    is_zero = (xm_b == 0) & (xn_b == 0)
    b00 = float(bmnc_b[is_zero][0])
    breaking = bmnc_b[~symmetric & ~is_zero]
    return float(np.sum(breaking * breaking) / (b00 * b00))
