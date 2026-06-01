# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Alternative initial guesses for the interior flux-surface geometry.

VMEC++ seeds the interior surfaces by interpolating the boundary toward the axis
(``FourierGeometry::interpFromBoundaryAndAxis``: ``m=0`` linear in ``s``, ``m>0``
like ``s^(m/2)``). For strongly shaped boundaries that interpolation can produce
overlapping surfaces (a negative Jacobian). This module provides two alternative
constructions that build the full interior coordinate map up front and return it
as a :class:`vmecpp.VmecOutput` suitable for ``vmecpp.run(..., restart_from=...)``:

- :func:`zeno_guess` -- the variational method of Tecchiolli et al.
  (arXiv:2405.08173): minimize a geometric action over a Fourier-Zernike basis.
  This empirically drives the Jacobian toward a single sign (correcting overlap)
  but does not guarantee it. Self-contained (numpy, optional SciPy minimizer).
- :func:`map2disc_guess` -- a wrapper over the ``map2disc`` package (Babin et al.,
  Plasma Phys. Control. Fusion 2025): conformally map each cross-section to the
  unit disc (a diffeomorphism, so the interior Jacobian is positive by
  construction). Requires the optional ``map2disc`` package
  (https://gitlab.mpcdf.mpg.de/gvec-group/map2disc).

The disc/Zernike radius ``rho`` maps to the VMEC flux label as ``s = rho^2``.

Note: VMEC++'s radial multi-grid already supplies a valid coarse-grid geometry for
most boundaries, so these guesses are primarily useful for single-resolution
workflows and for boundaries where the linear interpolation overlaps.
"""

from __future__ import annotations

import typing
from math import comb

import numpy as np

if typing.TYPE_CHECKING:
    from vmecpp import VmecInput, VmecOutput

# ---------------------------------------------------------------------------
# Fourier <-> real-space helpers (field-period angle convention: the boundary is
# R = sum rbc[m, n] cos(m*theta - n*zeta), zeta in [0, 2*pi) over one field period)
# ---------------------------------------------------------------------------


def _boundary_curve(rbc: np.ndarray, zbs: np.ndarray, ntor: int, zeta: float):
    """Return ``theta -> (R, Z)`` for the cross-section at field-period angle zeta."""
    mpol = rbc.shape[0]

    def curve(theta):
        theta = np.asarray(theta, dtype=float)
        big_r = np.zeros_like(theta)
        big_z = np.zeros_like(theta)
        for m in range(mpol):
            for n in range(-ntor, ntor + 1):
                if m == 0 and n < 0:
                    continue
                arg = m * theta - n * zeta
                big_r += rbc[m, n + ntor] * np.cos(arg)
                big_z += zbs[m, n + ntor] * np.sin(arg)
        return big_r, big_z

    return curve


def _grid_to_modes(grid: np.ndarray, mpol: int, ntor: int, parity: str) -> np.ndarray:
    """Project a real-space ``[n_theta, n_zeta]`` field onto VMEC ``[mpol, 2*ntor+1]``
    Fourier modes (``cos`` for R, ``sin`` for Z) on the equidistant angle grid."""
    n_theta, n_zeta = grid.shape
    theta = 2.0 * np.pi * np.arange(n_theta) / n_theta
    zeta = 2.0 * np.pi * np.arange(n_zeta) / n_zeta
    fun = np.cos if parity == "cos" else np.sin
    out = np.zeros((mpol, 2 * ntor + 1))
    for m in range(mpol):
        for n in range(-ntor, ntor + 1):
            if m == 0 and n < 0:
                continue
            basis = fun(m * theta[:, None] - n * zeta[None, :])
            norm = float(np.sum(basis * basis))
            if norm < 1e-12:
                continue
            out[m, n + ntor] = float(np.sum(grid * basis)) / norm
    return out


# ---------------------------------------------------------------------------
# Zeno / Tecchiolli variational guess
# ---------------------------------------------------------------------------


def _zernike_radial(ell: int, m: int, rho: np.ndarray) -> np.ndarray:
    """Zernike radial polynomial ``R_ell^m(rho)`` (``ell>=m>=0``, ``ell-m`` even).

    ``R_ell^m(1) = 1`` and ``~rho^m`` near ``rho=0``.
    """
    rho = np.asarray(rho, dtype=float)
    out = np.zeros_like(rho)
    for k in range((ell - m) // 2 + 1):
        coeff = ((-1.0) ** k) * comb(ell - k, k) * comb(ell - 2 * k, (ell - m) // 2 - k)
        out = out + coeff * rho ** (ell - 2 * k)
    return out


def _zeno_mode_table(mpol: int, ntor: int, lmax: int):
    """Fourier-Zernike modes ``(n, m, ell)`` for the zeno basis.

    ``ell`` runs ``m, m+2, ...`` up to ``lmax``.
    """
    modes = []
    for m in range(mpol):
        n_values = range(ntor + 1) if m == 0 else range(-ntor, ntor + 1)
        for n in n_values:
            ell = m
            while ell <= lmax:
                modes.append((n, m, ell))
                ell += 2
    return modes


def _zeno_interior(
    rbc: np.ndarray,
    zbs: np.ndarray,
    ns: int,
    lmax: int,
    omega: float,
    n_theta: int,
    n_zeta: int,
):
    """Minimize the geometric action over a Fourier-Zernike basis.

    Returns ``(rbc_out, zbs_out)`` each ``[ns, mpol, 2*ntor+1]``.
    """
    mpol = rbc.shape[0]
    ntor = (rbc.shape[1] - 1) // 2
    modes = _zeno_mode_table(mpol, ntor, lmax)
    n_modes = len(modes)

    s = np.linspace(0.0, 1.0, ns)
    rho = np.sqrt(s)
    theta = 2.0 * np.pi * np.arange(n_theta) / n_theta
    zeta = 2.0 * np.pi * np.arange(n_zeta) / n_zeta

    # Precompute, per mode, the radial profile, its s-derivative, and the angular
    # factor and its theta-derivative on the grid. R and Z are linear in the coeffs.
    eps = 1e-9
    rad = np.zeros((n_modes, ns))
    rad_s = np.zeros((n_modes, ns))
    cos_t = np.zeros((n_modes, n_theta, n_zeta))
    sin_t = np.zeros((n_modes, n_theta, n_zeta))
    dcos_t = np.zeros((n_modes, n_theta, n_zeta))  # d/dtheta cos = -m sin
    dsin_t = np.zeros((n_modes, n_theta, n_zeta))  # d/dtheta sin =  m cos
    th_g, ze_g = np.meshgrid(theta, zeta, indexing="ij")
    for i, (n, m, ell) in enumerate(modes):
        zr = _zernike_radial(ell, m, rho)
        rad[i] = zr
        # d/ds Z(sqrt(s)) = Z'(rho) / (2 rho); finite-difference Z' for simplicity.
        drho = 1e-4
        zr_prime = (
            _zernike_radial(ell, m, rho + drho) - _zernike_radial(ell, m, rho - drho)
        ) / (2 * drho)
        rad_s[i] = zr_prime / (2.0 * np.maximum(rho, eps))
        ang = m * th_g - n * ze_g
        cos_t[i] = np.cos(ang)
        sin_t[i] = np.sin(ang)
        dcos_t[i] = -m * np.sin(ang)
        dsin_t[i] = m * np.cos(ang)

    boundary_r = np.array([rbc[m, n + ntor] for (n, m, _ell) in modes])
    boundary_z = np.array([zbs[m, n + ntor] for (n, m, _ell) in modes])
    # Group mode indices by (n, m) for the boundary constraint sum_l coeff = boundary.
    nm_groups: dict = {}
    for i, (n, m, _ell) in enumerate(modes):
        nm_groups.setdefault((n, m), []).append(i)

    s_w = np.maximum(s, 1.0 / (2 * ns))  # avoid 1/s blow-up at the axis

    def geometry(cr, cz):
        # R, Z and their s/theta derivatives on the grid: [ns, n_theta, n_zeta]
        big_r = np.einsum("i,is,itz->stz", cr, rad, cos_t)
        big_z = np.einsum("i,is,itz->stz", cz, rad, sin_t)
        r_s = np.einsum("i,is,itz->stz", cr, rad_s, cos_t)
        z_s = np.einsum("i,is,itz->stz", cz, rad_s, sin_t)
        r_t = np.einsum("i,is,itz->stz", cr, rad, dcos_t)
        z_t = np.einsum("i,is,itz->stz", cz, rad, dsin_t)
        return big_r, big_z, r_s, z_s, r_t, z_t

    lam = 1.0e3 * (np.abs(boundary_r).max() + np.abs(boundary_z).max() + 1.0)

    def action(x):
        cr = x[:n_modes]
        cz = x[n_modes:]
        _, _, r_s, z_s, r_t, z_t = geometry(cr, cz)
        jac = r_s * z_t - r_t * z_s  # 2D poloidal Jacobian
        length = np.sqrt(r_s * r_s + z_s * z_s + 1e-30)
        integrand = jac * jac / (2.0 * s_w[:, None, None]) + omega * length
        s_action = float(np.mean(integrand))
        # boundary penalty: sum over l of coeff equals the boundary mode
        pen = 0.0
        for (_n, _m), idx in nm_groups.items():
            pen += (cr[idx].sum() - boundary_r[idx[0]]) ** 2
            pen += (cz[idx].sum() - boundary_z[idx[0]]) ** 2
        return s_action + lam * pen

    # Initial guess: all of each boundary mode in its l=m coefficient -> R ~ s^(m/2),
    # i.e. VMEC's own linear interpolation.
    x0 = np.zeros(2 * n_modes)
    for (_n, _m), idx in nm_groups.items():
        x0[idx[0]] = boundary_r[idx[0]]
        x0[n_modes + idx[0]] = boundary_z[idx[0]]

    try:
        from scipy.optimize import minimize  # noqa: PLC0415

        res = minimize(action, x0, method="L-BFGS-B", options={"maxiter": 200})
        x = res.x
    except ImportError:
        x = x0  # SciPy unavailable: fall back to the linear guess

    cr = x[:n_modes]
    cz = x[n_modes:]
    big_r, big_z, *_ = geometry(cr, cz)
    rbc_out = np.array([_grid_to_modes(big_r[j], mpol, ntor, "cos") for j in range(ns)])
    zbs_out = np.array([_grid_to_modes(big_z[j], mpol, ntor, "sin") for j in range(ns)])
    return rbc_out, zbs_out


# ---------------------------------------------------------------------------
# map2disc guess (wrapper over the external map2disc package)
# ---------------------------------------------------------------------------


def _map2disc_interior(
    rbc: np.ndarray, zbs: np.ndarray, ns: int, m_zernike: int | None, n_cuts: int | None
):
    """Conformal-map interior geometry via map2disc; returns ``(rbc_out, zbs_out)``."""
    try:
        from map2disc import BCM  # type: ignore  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - optional dependency
        msg = (
            "map2disc_guess requires the optional 'map2disc' package: "
            "pip install git+https://gitlab.mpcdf.mpg.de/gvec-group/map2disc.git"
        )
        raise ImportError(msg) from exc

    mpol = rbc.shape[0]
    ntor = (rbc.shape[1] - 1) // 2
    if m_zernike is None:
        m_zernike = max(mpol - 1, 2)
    n_zeta = max(2 * ntor + 1, 1) if n_cuts is None else n_cuts
    n_theta = 2 * mpol + 1
    theta_out = 2.0 * np.pi * np.arange(n_theta) / n_theta
    zeta_out = 2.0 * np.pi * np.arange(n_zeta) / n_zeta
    rho = np.sqrt(np.linspace(0.0, 1.0, ns))

    big_r = np.zeros((ns, n_theta, n_zeta))
    big_z = np.zeros((ns, n_theta, n_zeta))
    for k, zeta in enumerate(zeta_out):
        bcm = BCM(_boundary_curve(rbc, zbs, ntor, zeta), m_zernike)
        bcm.solve("interpolate")
        rz = bcm.eval_rt_1d(rho, theta_out)
        big_r[:, :, k] = rz[0]
        big_z[:, :, k] = rz[1]

    rbc_out = np.array([_grid_to_modes(big_r[j], mpol, ntor, "cos") for j in range(ns)])
    zbs_out = np.array([_grid_to_modes(big_z[j], mpol, ntor, "sin") for j in range(ns)])
    return rbc_out, zbs_out


# ---------------------------------------------------------------------------
# Injection: build a VmecOutput usable as restart_from
# ---------------------------------------------------------------------------


def _restart_output(
    vmec_input: VmecInput, rbc_out: np.ndarray, zbs_out: np.ndarray
) -> VmecOutput:
    """Assemble a :class:`VmecOutput` whose geometry holds the interior guess.

    A short low-resolution solve of the boundary truncated to ``m<=1`` (an ellipse,
    which always nests) provides the wout layout; its geometry is then overwritten
    with the guess, so the result can be passed as ``restart_from``.
    """
    import vmecpp  # noqa: PLC0415  (lazy import avoids a circular import)

    ns = rbc_out.shape[0]
    nfp = int(vmec_input.nfp)
    ntor = int(vmec_input.ntor)

    template_input = vmec_input.model_copy(deep=True)
    rbc = np.asarray(template_input.rbc, dtype=float).copy()
    zbs = np.asarray(template_input.zbs, dtype=float).copy()
    rbc[2:, :] = 0.0
    zbs[2:, :] = 0.0
    template_input.rbc = rbc
    template_input.zbs = zbs
    template_input.ns_array = np.array([ns], dtype=np.int64)
    template_input.ftol_array = np.array([1e-8])
    template_input.niter_array = np.array([1000], dtype=np.int64)
    template_input.return_outputs_even_if_not_converged = True
    template = vmecpp.run(template_input, verbose=False, max_threads=1)

    guess = template.model_copy(deep=True)
    xm = np.asarray(template.wout.xm).astype(int)
    xn = np.asarray(template.wout.xn).astype(int)
    rmnc = np.zeros_like(np.asarray(template.wout.rmnc, dtype=float))
    zmns = np.zeros_like(np.asarray(template.wout.zmns, dtype=float))
    for mn, (m, xnn) in enumerate(zip(xm, xn, strict=False)):
        n = round(xnn / nfp) if nfp else 0
        rmnc[mn, :] = rbc_out[:, m, n + ntor]
        zmns[mn, :] = zbs_out[:, m, n + ntor]
    guess.wout.rmnc = rmnc
    guess.wout.zmns = zmns
    guess.wout.lmns_full = np.zeros_like(
        np.asarray(template.wout.lmns_full, dtype=float)
    )
    guess.input = vmec_input
    return guess


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def zeno_guess(
    vmec_input: VmecInput,
    *,
    ns: int | None = None,
    lmax: int | None = None,
    omega: float = 0.1,
    n_theta: int | None = None,
    n_zeta: int | None = None,
) -> VmecOutput:
    """Variational (Tecchiolli et al.) initial guess for the interior geometry.

    Minimizes a geometric action (squared Jacobian plus a radial-straightness term
    weighted by ``omega``) over a Fourier-Zernike basis, with the boundary held
    fixed. Returns a :class:`VmecOutput` to pass as ``restart_from``.

    Args:
        vmec_input: the configuration whose boundary defines the guess.
        ns: number of flux surfaces (defaults to the last entry of ``ns_array``).
        lmax: maximum Zernike radial degree (defaults to ``mpol + 1``).
        omega: weight of the radial-straightness term in the action.
        n_theta, n_zeta: action quadrature resolution.
    """
    mpol = int(vmec_input.mpol)
    ntor = int(vmec_input.ntor)
    if ns is None:
        ns = int(np.asarray(vmec_input.ns_array)[-1])
    if lmax is None:
        lmax = mpol + 1
    if n_theta is None:
        n_theta = 2 * mpol + 1
    if n_zeta is None:
        n_zeta = max(2 * ntor + 1, 1)
    rbc = np.asarray(vmec_input.rbc, dtype=float)
    zbs = np.asarray(vmec_input.zbs, dtype=float)
    rbc_out, zbs_out = _zeno_interior(rbc, zbs, ns, lmax, omega, n_theta, n_zeta)
    return _restart_output(vmec_input, rbc_out, zbs_out)


def map2disc_guess(
    vmec_input: VmecInput,
    *,
    ns: int | None = None,
    m_zernike: int | None = None,
    n_cuts: int | None = None,
) -> VmecOutput:
    """Conformal-map (map2disc) initial guess for the interior geometry.

    Maps each cross-section to the unit disc (a proven diffeomorphism, so the
    interior Jacobian is positive) and projects back to VMEC modes. Returns a
    :class:`VmecOutput` to pass as ``restart_from``. Requires the optional
    ``map2disc`` package (https://gitlab.mpcdf.mpg.de/gvec-group/map2disc).

    Args:
        vmec_input: the configuration whose boundary defines the guess.
        ns: number of flux surfaces (defaults to the last entry of ``ns_array``).
        m_zernike: Zernike resolution of the disc map (defaults to ``mpol - 1``).
        n_cuts: number of toroidal cuts to solve the map on.
    """
    if ns is None:
        ns = int(np.asarray(vmec_input.ns_array)[-1])
    rbc = np.asarray(vmec_input.rbc, dtype=float)
    zbs = np.asarray(vmec_input.zbs, dtype=float)
    rbc_out, zbs_out = _map2disc_interior(rbc, zbs, ns, m_zernike, n_cuts)
    return _restart_output(vmec_input, rbc_out, zbs_out)
