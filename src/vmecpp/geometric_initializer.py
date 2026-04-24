# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Pre-VMEC geometric initializer using a Fourier-Zernike volume representation.

Given fixed-boundary Fourier coefficients (rbc, zbs), this module constructs
nested trial flux surfaces and an axis candidate by minimizing a purely
geometric action before solving MHD force balance.

The action minimized is:

    S = sum_{i,j,k} (0.5 * s_i * A_{ijk}^2 + omega * L_{ijk}^2)

where:
    A_{ijk} = dR/ds * dZ/dtheta - dR/dtheta * dZ/ds  (2D Jacobian at grid point)
    L_{ijk}^2 = (dR/ds)^2 + (dZ/ds)^2               (squared radial line length)

Minimizing S with the boundary fixed distributes flux surfaces as uniformly as
possible (penalizing Jacobian concentration and sign changes) while keeping
radial coordinate lines short (smooth mapping from axis to boundary).

The Fourier-Zernike volume representation is:

    R(s, theta, zeta) = sum_{n,m,k} r_{nmk} * f_{m,k}(s) * cos(m*theta - n*nfp*zeta)
    Z(s, theta, zeta) = sum_{n,m,k} z_{nmk} * f_{m,k}(s) * sin(m*theta - n*nfp*zeta)

with radial basis functions:

    f_{m,k}(s) = s^m * P_k(2s - 1)

where P_k is the Legendre polynomial of degree k. Note that f_{m,k}(1) = 1 for
all (m, k), so the boundary constraint reduces to:

    r_{nm0} = rbc[m, n] - sum_{k>=1} r_{nmk}

This eliminates the k=0 Zernike coefficient analytically, leaving the k>=1
coefficients as free optimization variables.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

logger = logging.getLogger(__name__)

# Enable 64-bit precision for JAX to match VMEC's double precision
jax.config.update("jax_enable_x64", True)


@dataclass
class GeometricInitializerConfig:
    """Configuration for the geometric initializer."""

    L_max: int = 4
    """Maximum Zernike radial order (must be >= 2 for any free variables).

    The free radial coefficients per mode (m, n) are at orders k = 1, ..., L_max//2.
    Higher values give more radial flexibility but increase cost.
    """

    n_s: int = 24
    """Number of radial grid points for optimization (includes s=0 and s=1)."""

    n_theta: int = 48
    """Number of poloidal grid points for optimization."""

    n_zeta: int = 16
    """Number of toroidal grid points per field period for optimization."""

    omega: float = 0.01
    """Weight for the radial line length (curvature) penalty term."""

    max_iter: int = 300
    """Maximum L-BFGS-B optimizer iterations."""

    ftol: float = 1e-10
    """Function value change tolerance for optimizer convergence."""

    gtol: float = 1e-6
    """Gradient infinity-norm tolerance for optimizer convergence."""


@dataclass
class GeometricInitializerDiagnostics:
    """Diagnostic quantities from the geometric initializer."""

    min_jacobian: float
    """Minimum value of A = dR/ds*dZ/dtheta - dR/dtheta*dZ/ds over the grid."""

    frac_negative_jacobian: float
    """Fraction of grid points with negative Jacobian A < 0."""

    final_objective: float
    """Final value of the geometric action S after optimization."""

    n_iterations: int
    """Number of optimizer iterations taken."""

    converged: bool
    """Whether the optimizer reported convergence."""


@dataclass
class GeometricInitializerResult:
    """Result of the geometric initializer."""

    rmnc: np.ndarray
    """Fourier coefficients (cos) for R; shape (mn_mode, ns)."""

    zmns: np.ndarray
    """Fourier coefficients (sin) for Z; shape (mn_mode, ns)."""

    raxis_c: np.ndarray
    """Axis Fourier coefficients for R (cos modes); shape (ntor+1,)."""

    zaxis_s: np.ndarray
    """Axis Fourier coefficients for Z (sin modes); shape (ntor+1,)."""

    diagnostics: GeometricInitializerDiagnostics
    """Diagnostic information about the optimization."""


def _legendre_values_and_derivatives(
    x: jax.Array, K_max: int
) -> tuple[jax.Array, jax.Array]:
    """Evaluate Legendre polynomials P_0..P_{K_max} and their x-derivatives at x.

    Uses the three-term recurrence for P_k and the associated recurrence for P_k'.

    Args:
        x: Points, shape (...,).
        K_max: Maximum degree.

    Returns:
        P: shape (..., K_max+1) with P[..., k] = P_k(x).
        dP: shape (..., K_max+1) with dP[..., k] = P_k'(x).
    """
    shape = x.shape
    # Initialize storage; use list-of-slices trick to allow JAX tracing
    P_list = [jnp.ones(shape), x]
    for k in range(2, K_max + 1):
        pk = ((2 * k - 1) * x * P_list[k - 1] - (k - 1) * P_list[k - 2]) / k
        P_list.append(pk)
    P = jnp.stack(P_list[: K_max + 1], axis=-1)  # (..., K_max+1)

    # Derivatives via recurrence (k+1)*P_{k+1}' = (2k+1)*P_k + (2k+1)*x*P_k' - k*P_{k-1}'
    dP_list = [jnp.zeros(shape), jnp.ones(shape)]
    for k in range(1, K_max):
        dpk1 = (
            (2 * k + 1) * P_list[k] + (2 * k + 1) * x * dP_list[k] - k * dP_list[k - 1]
        ) / (k + 1)
        dP_list.append(dpk1)
    dP = jnp.stack(dP_list[: K_max + 1], axis=-1)  # (..., K_max+1)

    return P, dP


def _compute_radial_basis(
    s: jax.Array, mpol: int, K_max: int
) -> tuple[jax.Array, jax.Array]:
    """Compute f_{m,k}(s) = s^m * P_k(2s-1) and its s-derivative.

    Args:
        s: Radial grid, shape (n_s,).
        mpol: Number of poloidal modes.
        K_max: Maximum radial order.

    Returns:
        basis: shape (n_s, mpol, K_max+1), where basis[i, m, k] = f_{m,k}(s_i).
        d_basis: shape (n_s, mpol, K_max+1), d/ds of basis.
    """
    x = 2.0 * s - 1.0  # map s in [0, 1] to x in [-1, 1]
    P, dP_dx = _legendre_values_and_derivatives(x, K_max)
    # d/ds [P_k(2s-1)] = 2 * P_k'(2s-1)
    dP_ds = 2.0 * dP_dx  # (n_s, K_max+1)

    # s^m for m=0,...,mpol-1: shape (n_s, mpol)
    m_vals = jnp.arange(mpol, dtype=jnp.float64)
    s_powers = s[:, None] ** m_vals[None, :]  # (n_s, mpol)

    # d/ds [s^m] = m * s^{m-1}; handle m=0 separately (derivative = 0)
    ds_powers_raw = m_vals[None, :] * s[:, None] ** jnp.where(
        m_vals > 0, m_vals - 1.0, 0.0
    )
    ds_powers = jnp.where(m_vals[None, :] > 0, ds_powers_raw, 0.0)  # (n_s, mpol)

    # basis[i, m, k] = s_powers[i, m] * P[i, k]
    basis = s_powers[:, :, None] * P[:, None, :]  # (n_s, mpol, K_max+1)
    # d_basis[i, m, k] = ds_powers[i, m]*P[i,k] + s_powers[i, m]*dP_ds[i, k]
    d_basis = (
        ds_powers[:, :, None] * P[:, None, :] + s_powers[:, :, None] * dP_ds[:, None, :]
    )  # (n_s, mpol, K_max+1)

    return basis, d_basis


def _compute_fourier_factors(
    theta: jax.Array, zeta: jax.Array, mpol: int, ntor: int, nfp: int
) -> tuple[jax.Array, jax.Array]:
    """Compute cos(m*theta - n*nfp*zeta) and sin(...) on the (theta, zeta) grid.

    Args:
        theta: Poloidal grid, shape (n_theta,).
        zeta: Toroidal grid (one field period), shape (n_zeta,).
        mpol: Number of poloidal modes.
        ntor: Number of toroidal modes.
        nfp: Number of field periods.

    Returns:
        cos_fac: shape (n_theta, n_zeta, mpol, 2*ntor+1).
        sin_fac: shape (n_theta, n_zeta, mpol, 2*ntor+1).
    """
    n_vals = jnp.arange(-ntor, ntor + 1, dtype=jnp.float64)  # (2*ntor+1,)
    m_vals = jnp.arange(mpol, dtype=jnp.float64)  # (mpol,)

    # angle[j, k, m, n_idx] = m*theta_j - n*nfp*zeta_k
    angles = (
        m_vals[None, None, :, None] * theta[:, None, None, None]
        - n_vals[None, None, None, :] * float(nfp) * zeta[None, :, None, None]
    )  # (n_theta, n_zeta, mpol, 2*ntor+1)

    return jnp.cos(angles), jnp.sin(angles)


def _build_coefficients(
    r_free: jax.Array,
    z_free: jax.Array,
    rbc: jax.Array,
    zbs: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Build full Zernike coefficient arrays from free variables + boundary constraint.

    The k=0 coefficient is constrained:
        coeff_r[n_idx, m, 0] = rbc[m, n_idx] - sum_{k>=1} r_free[n_idx, m, k-1]

    Args:
        r_free: Free R coefficients, shape (2*ntor+1, mpol, K_max).
        z_free: Free Z coefficients, shape (2*ntor+1, mpol, K_max).
        rbc: Boundary R Fourier coefficients, shape (mpol, 2*ntor+1).
        zbs: Boundary Z Fourier coefficients, shape (mpol, 2*ntor+1).

    Returns:
        coeff_r, coeff_z: shape (2*ntor+1, mpol, K_max+1).
    """
    # Constrained k=0 terms: rbc.T - sum_of_free_vars
    r_constrained = rbc.T - jnp.sum(r_free, axis=2)  # (2*ntor+1, mpol)
    z_constrained = zbs.T - jnp.sum(z_free, axis=2)

    # Concatenate to form full coefficient array
    coeff_r = jnp.concatenate(
        [r_constrained[:, :, None], r_free], axis=2
    )  # (2*ntor+1, mpol, K_max+1)
    coeff_z = jnp.concatenate([z_constrained[:, :, None], z_free], axis=2)

    return coeff_r, coeff_z


def _evaluate_geometry(
    coeff_r: jax.Array,
    coeff_z: jax.Array,
    basis: jax.Array,
    d_basis: jax.Array,
    cos_fac: jax.Array,
    sin_fac: jax.Array,
    mpol: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Evaluate R, Z and their partial derivatives on the grid.

    Args:
        coeff_r: R Zernike coefficients, shape (2*ntor+1, mpol, K_max+1).
        coeff_z: Z Zernike coefficients, shape (2*ntor+1, mpol, K_max+1).
        basis: Radial basis, shape (n_s, mpol, K_max+1).
        d_basis: Radial basis ds-derivatives, shape (n_s, mpol, K_max+1).
        cos_fac: cos(m*theta - n*nfp*zeta), shape (n_theta, n_zeta, mpol, 2*ntor+1).
        sin_fac: sin(m*theta - n*nfp*zeta), shape (n_theta, n_zeta, mpol, 2*ntor+1).
        mpol: Number of poloidal modes.

    Returns:
        R, Z, dR_ds, dZ_ds, dR_dtheta, dZ_dtheta: each shape (n_s, n_theta, n_zeta).
    """
    # R[i,j,k] = sum_{n,m,l} coeff_r[n,m,l] * basis[i,m,l] * cos_fac[j,k,m,n]
    R = jnp.einsum("nml,iml,jkmn->ijk", coeff_r, basis, cos_fac)
    Z = jnp.einsum("nml,iml,jkmn->ijk", coeff_z, basis, sin_fac)

    dR_ds = jnp.einsum("nml,iml,jkmn->ijk", coeff_r, d_basis, cos_fac)
    dZ_ds = jnp.einsum("nml,iml,jkmn->ijk", coeff_z, d_basis, sin_fac)

    # d/dtheta [cos(m*theta - ...)] = -m * sin(m*theta - ...)
    m_vals = jnp.arange(mpol, dtype=jnp.float64)
    dcosf_dt = (
        -m_vals[None, None, :, None] * sin_fac
    )  # (n_theta, n_zeta, mpol, 2*ntor+1)
    dsinf_dt = m_vals[None, None, :, None] * cos_fac

    dR_dtheta = jnp.einsum("nml,iml,jkmn->ijk", coeff_r, basis, dcosf_dt)
    dZ_dtheta = jnp.einsum("nml,iml,jkmn->ijk", coeff_z, basis, dsinf_dt)

    return R, Z, dR_ds, dZ_ds, dR_dtheta, dZ_dtheta


def _make_objective(
    rbc_jax: jax.Array,
    zbs_jax: jax.Array,
    basis: jax.Array,
    d_basis: jax.Array,
    cos_fac: jax.Array,
    sin_fac: jax.Array,
    s_grid: jax.Array,
    ntor: int,
    mpol: int,
    K_max: int,
    omega: float,
):
    """Return a JIT-compiled (objective, gradient) function of the free variables.

    Args:
        rbc_jax: Boundary R coefficients, shape (mpol, 2*ntor+1).
        zbs_jax: Boundary Z coefficients, shape (mpol, 2*ntor+1).
        basis: Precomputed radial basis.
        d_basis: Precomputed radial basis derivatives.
        cos_fac: Precomputed cosine Fourier factors.
        sin_fac: Precomputed sine Fourier factors.
        s_grid: Radial grid, shape (n_s,).
        ntor: Number of toroidal modes.
        mpol: Number of poloidal modes.
        K_max: Maximum radial order (number of free variables per mode = K_max).
        omega: Weight for curvature penalty.

    Returns:
        A JIT-compiled function that takes a flat free-variable array and returns
        (objective_value, gradient).
    """
    n_free_per = (2 * ntor + 1) * mpol * K_max

    @jax.jit
    def objective(free_vars_flat: jax.Array) -> jax.Array:
        r_free = free_vars_flat[:n_free_per].reshape(2 * ntor + 1, mpol, K_max)
        z_free = free_vars_flat[n_free_per:].reshape(2 * ntor + 1, mpol, K_max)

        coeff_r, coeff_z = _build_coefficients(r_free, z_free, rbc_jax, zbs_jax)

        _, _, dR_ds, dZ_ds, dR_dtheta, dZ_dtheta = _evaluate_geometry(
            coeff_r, coeff_z, basis, d_basis, cos_fac, sin_fac, mpol
        )

        # 2D Jacobian: A = dR/ds * dZ/dtheta - dR/dtheta * dZ/ds
        A = dR_ds * dZ_dtheta - dR_dtheta * dZ_ds  # (n_s, n_theta, n_zeta)

        # Squared radial line length: L^2 = (dR/ds)^2 + (dZ/ds)^2
        L2 = dR_ds**2 + dZ_ds**2  # (n_s, n_theta, n_zeta)

        # Geometric action: S = sum_ijk (0.5 * s_i * A_ijk^2 + omega * L_ijk^2)
        S = jnp.sum(0.5 * s_grid[:, None, None] * A**2 + omega * L2)
        return S

    value_and_grad = jax.value_and_grad(objective)

    @jax.jit
    def objective_and_grad(free_vars_flat: jax.Array) -> tuple[jax.Array, jax.Array]:
        return value_and_grad(free_vars_flat)

    return objective_and_grad


def _project_to_vmec_fourier(
    R_surf: np.ndarray,
    Z_surf: np.ndarray,
    mpol: int,
    ntor: int,
    nfp: int,
    n_theta: int,
    n_zeta: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Project R and Z on a 2D grid to VMEC Fourier coefficients.

    Uses the standard DFT projection:
        rmnc[mn] = (2/(n_theta*n_zeta)) * sum_{p,q} R[p,q] * cos(m*theta_p - n*nfp*zeta_q)
    with factor 1 instead of 2 for the (m=0, n=0) mode.

    Args:
        R_surf: R values on (theta, zeta) grid, shape (n_theta, n_zeta).
        Z_surf: Z values on (theta, zeta) grid, shape (n_theta, n_zeta).
        mpol: Number of poloidal modes.
        ntor: Number of toroidal modes.
        nfp: Number of field periods.
        n_theta: Grid resolution in theta.
        n_zeta: Grid resolution in zeta (per field period).

    Returns:
        rmnc_surf: 1D array of length mnmax.
        zmns_surf: 1D array of length mnmax.
    """
    mnmax = (ntor + 1) + (mpol - 1) * (2 * ntor + 1)
    rmnc_surf = np.zeros(mnmax)
    zmns_surf = np.zeros(mnmax)

    theta_grid = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    zeta_grid = np.linspace(0, 2 * np.pi / nfp, n_zeta, endpoint=False)
    norm = n_theta * n_zeta

    # m=0 modes: n from 0 to ntor
    for n in range(ntor + 1):
        cos_fac = np.cos(-n * nfp * zeta_grid[None, :])  # (1, n_zeta)
        mn = n  # mn_index(m=0, n)
        if n == 0:
            rmnc_surf[mn] = np.sum(R_surf) / norm
        else:
            rmnc_surf[mn] = 2.0 * np.sum(R_surf * cos_fac) / norm
        # Z has no (m=0) sin contribution: sin(0*theta - n*phi) = -sin(n*phi),
        # so zmns[0, n] would be -2/(n_theta*n_zeta)*sum(Z*sin(n*phi*nfp))
        # but by convention VMEC does not store m=0 zmns modes (they are zero
        # for a smooth surface). We keep zmns_surf[mn] = 0.

    # m>0 modes: n from -ntor to ntor
    for m in range(1, mpol):
        for n in range(-ntor, ntor + 1):
            mn = (ntor + 1) + (m - 1) * (2 * ntor + 1) + (n + ntor)
            angles = (
                m * theta_grid[:, None] - n * nfp * zeta_grid[None, :]
            )  # (n_theta, n_zeta)
            cos_fac = np.cos(angles)
            sin_fac = np.sin(angles)
            rmnc_surf[mn] = 2.0 * np.sum(R_surf * cos_fac) / norm
            zmns_surf[mn] = 2.0 * np.sum(Z_surf * sin_fac) / norm

    return rmnc_surf, zmns_surf


def compute_geometric_initialization(
    rbc: np.ndarray,
    zbs: np.ndarray,
    mpol: int,
    ntor: int,
    nfp: int,
    ns: int,
    config: GeometricInitializerConfig | None = None,
) -> GeometricInitializerResult:
    """Compute a geometrically optimized initial guess for VMEC.

    Minimizes the geometric action S = sum(0.5*s*A^2 + omega*L^2) over the
    Fourier-Zernike free coefficients, subject to the fixed-boundary constraint.

    Args:
        rbc: Boundary R Fourier coefficients, shape (mpol, 2*ntor+1).
             rbc[m, n+ntor] is the coefficient for cos(m*theta - n*nfp*zeta).
        zbs: Boundary Z Fourier coefficients, shape (mpol, 2*ntor+1).
             zbs[m, n+ntor] is the coefficient for sin(m*theta - n*nfp*zeta).
        mpol: Number of poloidal modes (boundary and output).
        ntor: Number of toroidal modes (boundary and output).
        nfp: Number of field periods.
        ns: Number of radial surfaces for output rmnc/zmns (full grid, s=0..1).
        config: Optimizer and grid configuration. Uses defaults if None.

    Returns:
        GeometricInitializerResult with rmnc, zmns, raxis_c, zaxis_s, diagnostics.
    """
    if config is None:
        config = GeometricInitializerConfig()

    K_max = config.L_max // 2
    if K_max == 0:
        logger.info(
            "L_max=%d gives K_max=0: no free variables, returning straight-line "
            "interpolation (rbc[m]*s^m).",
            config.L_max,
        )

    n_s = config.n_s
    n_theta = config.n_theta
    n_zeta = config.n_zeta

    # Build optimization grids
    s_grid = jnp.linspace(0.0, 1.0, n_s)
    theta_grid = jnp.linspace(0.0, 2.0 * jnp.pi, n_theta, endpoint=False)
    zeta_grid = jnp.linspace(0.0, 2.0 * jnp.pi / nfp, n_zeta, endpoint=False)

    # Precompute basis and Fourier factors (fixed throughout optimization)
    basis, d_basis = _compute_radial_basis(s_grid, mpol, K_max)
    cos_fac, sin_fac = _compute_fourier_factors(theta_grid, zeta_grid, mpol, ntor, nfp)

    rbc_jax = jnp.array(rbc, dtype=jnp.float64)
    zbs_jax = jnp.array(zbs, dtype=jnp.float64)

    if K_max == 0:
        # Trivial case: no free variables, geometry is rbc[m]*s^m * cos(...)
        free_vars = np.zeros(0)
        final_obj = 0.0
        n_iters = 0
        converged = True
    else:
        # Build JIT-compiled objective + gradient
        obj_and_grad = _make_objective(
            rbc_jax,
            zbs_jax,
            basis,
            d_basis,
            cos_fac,
            sin_fac,
            s_grid,
            ntor,
            mpol,
            K_max,
            config.omega,
        )

        n_free = 2 * (2 * ntor + 1) * mpol * K_max
        free_vars_init = np.zeros(n_free)

        # Warm-up JIT compilation before optimization
        logger.debug("Warming up JAX JIT for geometric initializer...")
        _ = obj_and_grad(jnp.array(free_vars_init))

        def scipy_obj(x: np.ndarray) -> tuple[float, np.ndarray]:
            val, grad = obj_and_grad(jnp.array(x))
            return float(val), np.array(grad, dtype=np.float64)

        logger.info(
            "Starting geometric action minimization: %d free vars, "
            "L_max=%d, n_s=%d, n_theta=%d, n_zeta=%d, omega=%.3g",
            n_free,
            config.L_max,
            n_s,
            n_theta,
            n_zeta,
            config.omega,
        )

        result = minimize(
            scipy_obj,
            free_vars_init,
            method="L-BFGS-B",
            jac=True,
            options={
                "maxiter": config.max_iter,
                "ftol": config.ftol,
                "gtol": config.gtol,
            },
        )

        free_vars = result.x
        final_obj = float(result.fun)
        n_iters = result.nit
        converged = bool(result.success)

        if not converged:
            logger.warning(
                "Geometric initializer did not converge after %d iterations "
                "(message: %s). Using best-found solution.",
                n_iters,
                result.message,
            )
        else:
            logger.info(
                "Geometric initializer converged in %d iterations, S=%.6g.",
                n_iters,
                final_obj,
            )

    # Reconstruct Zernike coefficients from optimized free variables
    if K_max == 0:
        coeff_r_np = np.zeros((2 * ntor + 1, mpol, 1))
        coeff_z_np = np.zeros((2 * ntor + 1, mpol, 1))
        coeff_r_np[:, :, 0] = rbc.T  # type: ignore[misc]
        coeff_z_np[:, :, 0] = zbs.T  # type: ignore[misc]
    else:
        n_free_per = (2 * ntor + 1) * mpol * K_max
        r_free = free_vars[:n_free_per].reshape(2 * ntor + 1, mpol, K_max)
        z_free = free_vars[n_free_per:].reshape(2 * ntor + 1, mpol, K_max)

        coeff_r_np = np.array(
            _build_coefficients(jnp.array(r_free), jnp.array(z_free), rbc_jax, zbs_jax)[
                0
            ]
        )
        coeff_z_np = np.array(
            _build_coefficients(jnp.array(r_free), jnp.array(z_free), rbc_jax, zbs_jax)[
                1
            ]
        )

    # Evaluate diagnostics on the optimization grid
    if K_max > 0:
        n_free_per = (2 * ntor + 1) * mpol * K_max
        r_free_diag = jnp.array(
            free_vars[:n_free_per].reshape(2 * ntor + 1, mpol, K_max)
        )
        z_free_diag = jnp.array(
            free_vars[n_free_per:].reshape(2 * ntor + 1, mpol, K_max)
        )
        coeff_r_diag, coeff_z_diag = _build_coefficients(
            r_free_diag, z_free_diag, rbc_jax, zbs_jax
        )
    else:
        coeff_r_diag = jnp.array(coeff_r_np)
        coeff_z_diag = jnp.array(coeff_z_np)

    _, _, dR_ds_d, dZ_ds_d, dR_dt_d, dZ_dt_d = _evaluate_geometry(
        coeff_r_diag, coeff_z_diag, basis, d_basis, cos_fac, sin_fac, mpol
    )
    A_grid = np.array(dR_ds_d * dZ_dt_d - dR_dt_d * dZ_ds_d)
    min_jac = float(np.min(A_grid))
    frac_neg = float(np.mean(A_grid < 0.0))

    diagnostics = GeometricInitializerDiagnostics(
        min_jacobian=min_jac,
        frac_negative_jacobian=frac_neg,
        final_objective=final_obj,
        n_iterations=n_iters if K_max > 0 else 0,
        converged=converged,
    )

    # Export to VMEC full-grid surfaces (s = 0, 1/(ns-1), ..., 1)
    s_vmec = np.linspace(0.0, 1.0, ns)
    basis_vmec, _ = _compute_radial_basis(jnp.array(s_vmec), mpol, K_max)
    basis_vmec_np = np.array(basis_vmec)  # (ns, mpol, K_max+1)
    coeff_r_np2 = np.array(coeff_r_np)
    coeff_z_np2 = np.array(coeff_z_np)

    # High-resolution Fourier grid for projection
    n_theta_proj = max(n_theta, 2 * mpol + 4)
    n_zeta_proj = max(n_zeta, max(1, 2 * ntor) + 4) if ntor > 0 else n_zeta
    theta_proj = np.linspace(0.0, 2.0 * np.pi, n_theta_proj, endpoint=False)
    zeta_proj = np.linspace(0.0, 2.0 * np.pi / nfp, n_zeta_proj, endpoint=False)

    n_vals = np.arange(-ntor, ntor + 1)  # (2*ntor+1,)
    m_vals_np = np.arange(mpol)  # (mpol,)

    # Fourier factors for projection grid: (n_theta_proj, n_zeta_proj, mpol, 2*ntor+1)
    angles_proj = (
        m_vals_np[None, None, :, None] * theta_proj[:, None, None, None]
        - n_vals[None, None, None, :] * nfp * zeta_proj[None, :, None, None]
    )
    cos_proj = np.cos(angles_proj)
    sin_proj = np.sin(angles_proj)

    mnmax = (ntor + 1) + (mpol - 1) * (2 * ntor + 1)
    rmnc = np.zeros((mnmax, ns))
    zmns = np.zeros((mnmax, ns))

    for j, _sj in enumerate(s_vmec):
        # R[p, q] = sum_{n_idx, m, k} coeff_r[n_idx, m, k] * basis_j[m, k] * cos_proj[p,q,m,n_idx]
        bj = basis_vmec_np[j]  # (mpol, K_max+1)
        # Weighted coefficients: w[m, k, n_idx] = coeff_r[n_idx, m, k] * bj[m, k]
        # R[p, q] = sum_{m, k, n_idx} w_r[m, k, n_idx] * cos_proj[p, q, m, n_idx]
        #         = einsum('mkn,pqmn->pq', w_r, cos_proj)
        w_r = np.einsum("nml,ml->mln", coeff_r_np2, bj)  # (mpol, K_max+1, 2*ntor+1)
        w_r_collapsed = np.sum(w_r, axis=1)  # (mpol, 2*ntor+1)
        # w_r_collapsed[m, n_idx] = sum_k coeff_r[n_idx, m, k] * bj[m, k]

        w_z = np.einsum("nml,ml->mln", coeff_z_np2, bj)
        w_z_collapsed = np.sum(w_z, axis=1)  # (mpol, 2*ntor+1)

        # R_surf[p, q] = einsum('mn,pqmn->pq', w_r_collapsed, cos_proj)
        R_surf = np.einsum("mn,pqmn->pq", w_r_collapsed, cos_proj)
        Z_surf = np.einsum("mn,pqmn->pq", w_z_collapsed, sin_proj)

        rmnc_j, zmns_j = _project_to_vmec_fourier(
            R_surf, Z_surf, mpol, ntor, nfp, n_theta_proj, n_zeta_proj
        )
        rmnc[:, j] = rmnc_j
        zmns[:, j] = zmns_j

    # Extract axis (s=0) Fourier coefficients
    # At s=0 only m=0 terms survive; R_axis = sum_{n_idx, k} coeff_r[n_idx, 0, k]*P_k(-1)*cos(n*phi)
    # Evaluate via projection at the axis surface
    raxis_c = rmnc[: ntor + 1, 0].copy()  # m=0 modes from the s=0 surface
    # Axis Z: from zmns at s=0 for m=0 (which are zero by stellarator symmetry for n=0)
    # and for the general case use the m=1 info; but the safest is the DFT result
    # from evaluating Z at the axis.
    zaxis_s = np.zeros(ntor + 1)
    if ntor > 0:
        # Evaluate Z at axis using Zernike expansion directly
        basis_axis_np = np.array(
            _compute_radial_basis(jnp.array([0.0]), mpol, K_max)[0]
        )
        baxis = basis_axis_np[0]  # (mpol, K_max+1) at s=0
        # Only m=0 survives at s=0 (0^m=0 for m>0)
        w_z_axis = np.einsum("nml,ml->mln", coeff_z_np2, baxis)
        w_z_axis_m0 = np.sum(
            w_z_axis[0], axis=0
        )  # (2*ntor+1,) = sum_k coeff_z[n,0,k]*baxis[0,k]

        # Z_axis(zeta) = sum_{n_idx} w_z_axis_m0[n_idx] * sin(-(n_idx-ntor)*nfp*zeta)
        # = -sum_{n>=1} (w_z_axis_m0[n+ntor] - w_z_axis_m0[-n+ntor]) * sin(n*nfp*zeta)
        for n in range(1, ntor + 1):
            zaxis_s[n] = -(w_z_axis_m0[n + ntor] - w_z_axis_m0[ntor - n])

    logger.info(
        "Geometric initializer: min_jacobian=%.4g, frac_negative=%.2f%%, converged=%s",
        min_jac,
        100.0 * frac_neg,
        converged,
    )

    return GeometricInitializerResult(
        rmnc=rmnc,
        zmns=zmns,
        raxis_c=raxis_c,
        zaxis_s=zaxis_s,
        diagnostics=diagnostics,
    )
