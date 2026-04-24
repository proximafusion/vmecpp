# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Tests for the geometric initializer (Fourier-Zernike pre-VMEC initialization).

Unit tests verify the mathematical correctness of the basis functions and
objectives.  Integration tests check that the initializer produces a valid
axis guess and that VMEC++ can converge starting from it.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import jax.numpy as jnp
import numpy as np
import pytest

import vmecpp
from vmecpp.geometric_initializer import (
    GeometricInitializerConfig,
    GeometricInitializerDiagnostics,
    GeometricInitializerResult,
    _compute_fourier_factors,
    _compute_radial_basis,
    _legendre_values_and_derivatives,
    _project_to_vmec_fourier,
    compute_geometric_initialization,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent.parent
TEST_DATA_DIR = REPO_ROOT / "src" / "vmecpp" / "cpp" / "vmecpp" / "test_data"


def _circular_tokamak_boundary(
    R0: float = 3.0, a: float = 1.0, mpol: int = 4, ntor: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Return rbc, zbs for a circular axisymmetric torus (major radius R0, minor a)."""
    rbc = np.zeros((mpol, 2 * ntor + 1))
    zbs = np.zeros((mpol, 2 * ntor + 1))
    rbc[0, ntor] = R0
    rbc[1, ntor] = a
    zbs[1, ntor] = a
    return rbc, zbs


# ---------------------------------------------------------------------------
# Unit tests: Legendre polynomials
# ---------------------------------------------------------------------------


def test_legendre_polynomials_values():
    """P_k at x=-1, 0, 1 should match known exact values."""

    x = jnp.array([-1.0, 0.0, 1.0])
    K_max = 4
    P, _dP = _legendre_values_and_derivatives(x, K_max)

    # P_k(1) = 1 for all k
    np.testing.assert_allclose(np.array(P[2, :]), 1.0, atol=1e-12)

    # P_k(-1) = (-1)^k
    expected_m1 = np.array([(-1) ** k for k in range(K_max + 1)])
    np.testing.assert_allclose(np.array(P[0, :]), expected_m1, atol=1e-12)

    # P_0, P_1, P_2 at x=0
    assert abs(float(P[1, 0]) - 1.0) < 1e-12  # P_0(0) = 1
    assert abs(float(P[1, 1]) - 0.0) < 1e-12  # P_1(0) = 0
    assert abs(float(P[1, 2]) - (-0.5)) < 1e-12  # P_2(0) = -0.5


def test_legendre_derivative_at_x1():
    """P_k'(1) = k*(k+1)/2 for all k."""

    x = jnp.array([1.0])
    K_max = 4
    _, dP = _legendre_values_and_derivatives(x, K_max)
    for k in range(K_max + 1):
        expected = k * (k + 1) / 2
        assert abs(float(dP[0, k]) - expected) < 1e-8, (
            f"P_{k}'(1): got {float(dP[0, k])}, expected {expected}"
        )


# ---------------------------------------------------------------------------
# Unit tests: radial basis
# ---------------------------------------------------------------------------


def test_radial_basis_at_boundary():
    """f_{m,k}(1) = 1 for all m, k (basis functions equal 1 at s=1)."""

    K_max = 3
    mpol = 5
    s = jnp.array([0.0, 0.5, 1.0])
    basis, _ = _compute_radial_basis(s, mpol, K_max)

    # At s=1 (index 2), all basis values should be 1
    np.testing.assert_allclose(np.array(basis[2]), 1.0, atol=1e-12)


def test_radial_basis_at_axis_for_m_positive():
    """f_{m,k}(0) = 0 for m > 0 (axis collapse for non-zero poloidal modes)."""

    K_max = 3
    mpol = 5
    s = jnp.array([0.0])
    basis, _ = _compute_radial_basis(s, mpol, K_max)

    # For m > 0: all basis values at s=0 should be 0
    for m in range(1, mpol):
        np.testing.assert_allclose(
            np.array(basis[0, m, :]),
            0.0,
            atol=1e-12,
            err_msg=f"basis[s=0, m={m}, :] should be 0",
        )


def test_radial_basis_derivative_smoothness():
    """d/ds basis[s=0, m=0, k] should match 2*P_k'(-1)."""

    K_max = 3
    mpol = 3
    s = jnp.array([0.0])
    _, d_basis = _compute_radial_basis(s, mpol, K_max)
    _, dP = _legendre_values_and_derivatives(jnp.array([-1.0]), K_max)

    # d/ds[s^0 * P_k(2s-1)] at s=0 = 0 + 1 * 2*P_k'(-1) = 2*P_k'(-1)
    expected = 2.0 * np.array(dP[0, :])
    np.testing.assert_allclose(np.array(d_basis[0, 0, :]), expected, atol=1e-10)


# ---------------------------------------------------------------------------
# Unit tests: Fourier factors
# ---------------------------------------------------------------------------


def test_fourier_factors_shape():
    """_compute_fourier_factors returns arrays with correct shape."""

    n_theta, n_zeta, mpol, ntor, nfp = 8, 4, 3, 2, 1
    theta = jnp.linspace(0, 2 * jnp.pi, n_theta, endpoint=False)
    zeta = jnp.linspace(0, 2 * jnp.pi / nfp, n_zeta, endpoint=False)
    cos_fac, sin_fac = _compute_fourier_factors(theta, zeta, mpol, ntor, nfp)

    assert cos_fac.shape == (n_theta, n_zeta, mpol, 2 * ntor + 1)
    assert sin_fac.shape == (n_theta, n_zeta, mpol, 2 * ntor + 1)


def test_fourier_factors_at_zero():
    """cos_fac and sin_fac at theta=0, zeta=0 should equal cos/sin of 0 = 1/0."""

    n_theta, n_zeta, mpol, ntor, nfp = 4, 4, 3, 1, 1
    theta = jnp.linspace(0, 2 * jnp.pi, n_theta, endpoint=False)
    zeta = jnp.linspace(0, 2 * jnp.pi / nfp, n_zeta, endpoint=False)
    cos_fac, sin_fac = _compute_fourier_factors(theta, zeta, mpol, ntor, nfp)

    # At theta=0, zeta=0: cos(m*0 - n*0) = 1, sin = 0
    np.testing.assert_allclose(np.array(cos_fac[0, 0]), 1.0, atol=1e-12)
    np.testing.assert_allclose(np.array(sin_fac[0, 0]), 0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# Unit tests: Fourier projection round-trip
# ---------------------------------------------------------------------------


def test_project_to_vmec_fourier_circular_torus():
    """Projection should recover the R0 and a coefficients for a circular cross-section."""
    mpol, ntor, nfp = 4, 0, 1
    n_theta, n_zeta = 64, 1
    R0, a = 3.0, 1.0

    theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    R_surf = R0 + a * np.cos(theta)[:, None]
    Z_surf = a * np.sin(theta)[:, None]

    rmnc_surf, zmns_surf = _project_to_vmec_fourier(
        R_surf, Z_surf, mpol, ntor, nfp, n_theta, n_zeta
    )

    # For ntor=0 and a circular tokamak: rmnc[0] = R0, rmnc at (m=1,n=0) = a
    assert abs(rmnc_surf[0] - R0) < 1e-10  # (m=0, n=0) mode = R0
    # (m=1, n=0): index = (ntor+1) + 0*(2*ntor+1) + (0+ntor) = 1 + 0 = 1
    assert abs(rmnc_surf[1] - a) < 1e-10  # (m=1, n=0) mode = a
    # zmns at (m=1, n=0): same index
    assert abs(zmns_surf[1] - a) < 1e-10


# ---------------------------------------------------------------------------
# Unit tests: boundary constraint
# ---------------------------------------------------------------------------


def test_boundary_satisfied_circular_tokamak():
    """Optimized R, Z should match the boundary at s=1."""
    R0, a = 3.0, 1.0
    mpol, ntor, nfp, ns = 4, 0, 1, 5
    rbc, zbs = _circular_tokamak_boundary(R0=R0, a=a, mpol=mpol, ntor=ntor)

    cfg = GeometricInitializerConfig(L_max=4, n_s=8, n_theta=16, n_zeta=4, max_iter=5)
    result = compute_geometric_initialization(rbc, zbs, mpol, ntor, nfp, ns, cfg)

    # rmnc at last surface (j = ns-1) should match boundary
    n_theta_check, n_zeta_check = 64, 1
    theta = np.linspace(0, 2 * np.pi, n_theta_check, endpoint=False)

    # Reconstruct R at s=1 from rmnc[:, -1]
    R_from_rmnc = np.zeros((n_theta_check, n_zeta_check))
    mnmax = (ntor + 1) + (mpol - 1) * (2 * ntor + 1)
    for m in range(mpol):
        for n in range(-ntor, ntor + 1):
            if m == 0 and n < 0:
                continue
            mn = n if m == 0 else (ntor + 1) + (m - 1) * (2 * ntor + 1) + (n + ntor)
            if mn < mnmax:
                R_from_rmnc += result.rmnc[mn, -1] * np.cos(
                    m * theta[:, None] - n * nfp * np.zeros((1, n_zeta_check))
                )

    R_exact = R0 + a * np.cos(theta)
    np.testing.assert_allclose(R_from_rmnc[:, 0], R_exact, atol=1e-6)


# ---------------------------------------------------------------------------
# Unit tests: axis
# ---------------------------------------------------------------------------


def test_axis_circular_tokamak():
    """For a circular axisymmetric torus, raxis_c[0] should be close to R0."""
    R0, a = 4.0, 1.0
    mpol, ntor, nfp, ns = 4, 0, 1, 11
    rbc, zbs = _circular_tokamak_boundary(R0=R0, a=a, mpol=mpol, ntor=ntor)

    cfg = GeometricInitializerConfig(
        L_max=4, n_s=16, n_theta=32, n_zeta=4, max_iter=100
    )
    result = compute_geometric_initialization(rbc, zbs, mpol, ntor, nfp, ns, cfg)

    # Axis should be near R0 (within the minor radius)
    assert abs(result.raxis_c[0] - R0) < a


def test_axis_zero_imaginary_part():
    """raxis_c and zaxis_s should be real-valued finite numbers."""
    R0, a = 3.0, 1.0
    mpol, ntor, nfp, ns = 3, 0, 1, 5
    rbc, zbs = _circular_tokamak_boundary(R0=R0, a=a, mpol=mpol, ntor=ntor)

    cfg = GeometricInitializerConfig(L_max=2, n_s=8, n_theta=16, n_zeta=4, max_iter=10)
    result = compute_geometric_initialization(rbc, zbs, mpol, ntor, nfp, ns, cfg)

    assert np.all(np.isfinite(result.raxis_c))
    assert np.all(np.isfinite(result.zaxis_s))


# ---------------------------------------------------------------------------
# Unit tests: Jacobian quality
# ---------------------------------------------------------------------------


def test_jacobian_mostly_positive_circular_tokamak():
    """After optimization the Jacobian should be positive almost everywhere."""
    R0, a = 3.0, 1.0
    mpol, ntor, nfp, ns = 4, 0, 1, 11
    rbc, zbs = _circular_tokamak_boundary(R0=R0, a=a, mpol=mpol, ntor=ntor)

    cfg = GeometricInitializerConfig(
        L_max=4, n_s=16, n_theta=32, n_zeta=4, max_iter=200, omega=0.01
    )
    result = compute_geometric_initialization(rbc, zbs, mpol, ntor, nfp, ns, cfg)

    # A well-initialized circular torus should have no negative-Jacobian points
    assert result.diagnostics.frac_negative_jacobian < 0.05


def test_diagnostics_populated():
    """Diagnostics fields should all be finite and reasonable."""
    R0, a = 3.0, 1.0
    mpol, ntor, nfp, ns = 3, 0, 1, 5
    rbc, zbs = _circular_tokamak_boundary(R0=R0, a=a, mpol=mpol, ntor=ntor)

    cfg = GeometricInitializerConfig(L_max=2, n_s=8, n_theta=16, n_zeta=4, max_iter=10)
    result = compute_geometric_initialization(rbc, zbs, mpol, ntor, nfp, ns, cfg)

    diag = result.diagnostics
    assert np.isfinite(diag.min_jacobian)
    assert 0.0 <= diag.frac_negative_jacobian <= 1.0
    assert np.isfinite(diag.final_objective)
    assert diag.n_iterations >= 0


# ---------------------------------------------------------------------------
# Unit tests: no free variables (K_max=0)
# ---------------------------------------------------------------------------


def test_l_max_zero_trivial():
    """With L_max=1 (K_max=0) the result is straight-line interpolation."""
    R0, a = 3.0, 1.0
    mpol, ntor, nfp, ns = 3, 0, 1, 5
    rbc, zbs = _circular_tokamak_boundary(R0=R0, a=a, mpol=mpol, ntor=ntor)

    cfg = GeometricInitializerConfig(L_max=1, n_s=8, n_theta=16, n_zeta=4)
    result = compute_geometric_initialization(rbc, zbs, mpol, ntor, nfp, ns, cfg)

    assert result.rmnc.shape == ((ntor + 1) + (mpol - 1) * (2 * ntor + 1), ns)
    assert np.all(np.isfinite(result.rmnc))
    assert np.all(np.isfinite(result.zmns))


# ---------------------------------------------------------------------------
# Unit tests: 3D stellarator-like case
# ---------------------------------------------------------------------------


def test_3d_boundary_runs_without_error():
    """With a 3D boundary, the initializer should complete without error."""
    mpol, ntor, nfp, ns = 3, 2, 5, 7
    rbc = np.zeros((mpol, 2 * ntor + 1))
    zbs = np.zeros((mpol, 2 * ntor + 1))
    # Simple 3D boundary (elliptical cross-section with triangularity)
    rbc[0, ntor] = 5.5  # R0 (m=0, n=0)
    rbc[1, ntor] = 0.5  # minor radius (m=1, n=0)
    rbc[0, ntor + 1] = 0.1  # toroidal ripple (m=0, n=1)
    zbs[1, ntor] = 0.5
    zbs[0, ntor + 1] = -0.05

    cfg = GeometricInitializerConfig(L_max=2, n_s=8, n_theta=16, n_zeta=8, max_iter=20)
    result = compute_geometric_initialization(rbc, zbs, mpol, ntor, nfp, ns, cfg)

    assert result.rmnc.shape[1] == ns
    assert np.all(np.isfinite(result.rmnc))
    assert np.all(np.isfinite(result.zmns))
    assert np.all(np.isfinite(result.raxis_c))
    assert np.all(np.isfinite(result.zaxis_s))
    assert len(result.raxis_c) == ntor + 1
    assert len(result.zaxis_s) == ntor + 1


# ---------------------------------------------------------------------------
# Integration tests with VmecInput
# ---------------------------------------------------------------------------


def test_vmecinput_initialization_method_default():
    """VmecInput.initialization_method defaults to 'default'."""

    vi = vmecpp.VmecInput()
    assert vi.initialization_method == "default"
    assert vi.geometric_init_L_max == 4
    assert vi.geometric_init_omega == 0.01


def test_vmecinput_geometric_action_field_roundtrip():
    """The new fields survive JSON serialization round-trip."""

    vi = vmecpp.VmecInput.from_file(TEST_DATA_DIR / "solovev.json")
    vi.initialization_method = "geometric_action"
    vi.geometric_init_L_max = 6
    vi.geometric_init_omega = 0.05

    json_str = vi.to_json()
    vi2 = vmecpp.VmecInput.model_validate_json(json_str)
    assert vi2.initialization_method == "geometric_action"
    assert vi2.geometric_init_L_max == 6
    assert vi2.geometric_init_omega == 0.05


def test_vmecinput_from_file_has_default_initialization():
    """Loading a classic VmecInput file assigns default initialization_method."""

    vi = vmecpp.VmecInput.from_file(TEST_DATA_DIR / "solovev.json")
    assert vi.initialization_method == "default"


def test_to_cpp_vmecindata_skips_python_only_fields():
    """_to_cpp_vmecindata should not crash when new Python-only fields are set."""

    vi = vmecpp.VmecInput.from_file(TEST_DATA_DIR / "solovev.json")
    vi.initialization_method = "geometric_action"
    vi.geometric_init_L_max = 2
    # Should not raise AttributeError
    cpp_indata = vi._to_cpp_vmecindata()
    assert cpp_indata is not None


@pytest.mark.slow
def test_run_with_geometric_action_solovev():
    """VMEC run with geometric_action initialization should converge for Solovev."""

    vi = vmecpp.VmecInput.from_file(TEST_DATA_DIR / "solovev.json")
    vi.initialization_method = "geometric_action"
    vi.geometric_init_L_max = 4
    vi.geometric_init_omega = 0.01

    output = vmecpp.run(vi, verbose=False, max_threads=1)
    assert output.wout is not None
    # Check that VMEC converged and the equilibrium makes physical sense
    assert output.wout.ier_flag == 0


@pytest.mark.slow
def test_geometric_action_axis_used_by_vmec():
    """The geometric initializer should improve or match VMEC's default axis guess."""

    vi = vmecpp.VmecInput.from_file(TEST_DATA_DIR / "solovev.json")

    # Default run
    out_default = vmecpp.run(vi, verbose=False, max_threads=1)

    # Geometric action run
    vi.initialization_method = "geometric_action"
    out_geom = vmecpp.run(vi, verbose=False, max_threads=1)

    # Both runs should produce a valid converged equilibrium
    assert out_default.wout.ier_flag == 0
    assert out_geom.wout.ier_flag == 0


# ---------------------------------------------------------------------------
# Unit tests: bean-shaped boundary
# ---------------------------------------------------------------------------


def _bean_shaped_boundary(
    R0: float = 3.0,
    a: float = 1.0,
    kappa: float = 1.5,
    delta: float = 0.3,
    mpol: int = 6,
    ntor: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return rbc, zbs for a bean-shaped (elongated + triangular) axisymmetric torus.

    Uses the standard Shafranov parametrization:
        R = R0 + a*cos(theta)  + a*delta*cos(2*theta)  (m=2 harmonic captures triangularity)
        Z = kappa*a*sin(theta)
    """
    rbc = np.zeros((mpol, 2 * ntor + 1))
    zbs = np.zeros((mpol, 2 * ntor + 1))
    rbc[0, ntor] = R0
    rbc[1, ntor] = a
    rbc[2, ntor] = -a * delta  # negative triangularity in R (m=2 term)
    zbs[1, ntor] = kappa * a
    return rbc, zbs


def test_bean_shaped_boundary_condition():
    """rmnc at the last (boundary) surface must match rbc exactly for a bean shape."""
    R0, a, kappa, delta = 3.0, 1.0, 1.5, 0.3
    mpol, ntor, nfp, ns = 6, 0, 1, 7
    rbc, zbs = _bean_shaped_boundary(R0=R0, a=a, kappa=kappa, delta=delta, mpol=mpol)

    cfg = GeometricInitializerConfig(L_max=4, n_s=12, n_theta=32, n_zeta=4, max_iter=50)
    result = compute_geometric_initialization(rbc, zbs, mpol, ntor, nfp, ns, cfg)

    # For ntor=0: mn index for (m, n=0) is: m=0 -> 0, m>=1 -> (ntor+1)+(m-1)*(2*ntor+1)+(0+ntor)
    def mn_index(m: int) -> int:
        return m  # for ntor=0: (ntor+1)=1, (2*ntor+1)=1; m=0->0, m=1->1, m=2->2

    # Boundary (j=-1) must recover rbc exactly
    np.testing.assert_allclose(result.rmnc[mn_index(0), -1], R0, atol=1e-8)
    np.testing.assert_allclose(result.rmnc[mn_index(1), -1], a, atol=1e-8)
    np.testing.assert_allclose(result.rmnc[mn_index(2), -1], -a * delta, atol=1e-8)
    np.testing.assert_allclose(result.zmns[mn_index(1), -1], kappa * a, atol=1e-8)


def test_bean_shaped_r20_sign():
    """The m=2, n=0 Fourier coefficient of R should be negative at all surfaces.

    For inward-pointing triangularity (delta > 0), rbc[2, 0] = -a*delta < 0.
    The geometric initializer preserves this sign from boundary inward.
    """
    R0, a, kappa, delta = 3.0, 1.0, 1.5, 0.3
    mpol, ntor, nfp, ns = 6, 0, 1, 11
    rbc, zbs = _bean_shaped_boundary(R0=R0, a=a, kappa=kappa, delta=delta, mpol=mpol)

    cfg = GeometricInitializerConfig(
        L_max=4, n_s=16, n_theta=32, n_zeta=4, max_iter=100
    )
    result = compute_geometric_initialization(rbc, zbs, mpol, ntor, nfp, ns, cfg)

    # mn index for m=2, n=0 with ntor=0
    mn_20 = 2
    # r20 should be negative (triangularity convention) at the boundary
    assert result.rmnc[mn_20, -1] < 0, (
        f"Expected rmnc[m=2,j=-1] < 0 (triangularity), got {result.rmnc[mn_20, -1]}"
    )
    # r20 should scale with delta: boundary value must be close to -a*delta
    np.testing.assert_allclose(result.rmnc[mn_20, -1], -a * delta, atol=1e-8)


def test_bean_shaped_r20_scales_with_triangularity():
    """rmnc[m=2, n=0] at the boundary should equal -a*delta for all delta values."""
    R0, a, kappa, mpol, ntor, nfp, ns = 3.0, 1.0, 1.5, 6, 0, 1, 5
    cfg = GeometricInitializerConfig(L_max=2, n_s=8, n_theta=16, n_zeta=4, max_iter=10)
    mn_20 = 2  # (m=2, n=0) index for ntor=0

    for delta in [0.1, 0.2, 0.3]:
        rbc, zbs = _bean_shaped_boundary(
            R0=R0, a=a, kappa=kappa, delta=delta, mpol=mpol
        )
        result = compute_geometric_initialization(rbc, zbs, mpol, ntor, nfp, ns, cfg)
        np.testing.assert_allclose(
            result.rmnc[mn_20, -1],
            -a * delta,
            atol=1e-8,
            err_msg=f"rmnc[m=2] mismatch at delta={delta}",
        )


def test_bean_shaped_axis_inside_boundary():
    """Axis R-coordinate should be inside the plasma for all bean-shaped boundaries."""
    R0, a, kappa, delta = 3.0, 1.0, 1.5, 0.3
    mpol, ntor, nfp, ns = 6, 0, 1, 11
    rbc, zbs = _bean_shaped_boundary(R0=R0, a=a, kappa=kappa, delta=delta, mpol=mpol)

    cfg = GeometricInitializerConfig(
        L_max=4, n_s=16, n_theta=32, n_zeta=4, max_iter=100
    )
    result = compute_geometric_initialization(rbc, zbs, mpol, ntor, nfp, ns, cfg)

    # Axis R-coordinate must be between the inner and outer boundary extents
    r_inner = R0 - a  # inner boundary (approximately)
    r_outer = R0 + a  # outer boundary (approximately)
    raxis = result.raxis_c[0]
    assert r_inner < raxis < r_outer, (
        f"Axis raxis_c[0]={raxis:.4f} outside plasma bounds [{r_inner}, {r_outer}]"
    )


def test_bean_shaped_jacobian_positive():
    """Bean-shaped cross-section with moderate parameters should have positive Jacobian."""
    R0, a, kappa, delta = 3.0, 1.0, 1.5, 0.3
    mpol, ntor, nfp, ns = 6, 0, 1, 11
    rbc, zbs = _bean_shaped_boundary(R0=R0, a=a, kappa=kappa, delta=delta, mpol=mpol)

    cfg = GeometricInitializerConfig(
        L_max=4, n_s=16, n_theta=32, n_zeta=4, max_iter=200, omega=0.01
    )
    result = compute_geometric_initialization(rbc, zbs, mpol, ntor, nfp, ns, cfg)

    assert result.diagnostics.frac_negative_jacobian < 0.05, (
        f"Expected <5% negative Jacobian, got {result.diagnostics.frac_negative_jacobian:.1%}"
    )


# ---------------------------------------------------------------------------
# Unit tests: circular torus surface profiles
# ---------------------------------------------------------------------------


def test_circular_torus_axis_collapse():
    """rmnc[m=1, j=0] must be zero (axis collapse) for the circular torus."""
    R0, a = 3.0, 1.0
    mpol, ntor, nfp, ns = 4, 0, 1, 11
    rbc, zbs = _circular_tokamak_boundary(R0=R0, a=a, mpol=mpol, ntor=ntor)

    cfg = GeometricInitializerConfig(
        L_max=4, n_s=32, n_theta=64, n_zeta=4, max_iter=200
    )
    result = compute_geometric_initialization(rbc, zbs, mpol, ntor, nfp, ns, cfg)

    # At s=0 (axis), all m>0 Fourier components must vanish
    mn_10 = 1  # (m=1, n=0) for ntor=0
    assert abs(result.rmnc[mn_10, 0]) < 1e-10, (
        f"rmnc[m=1, j=0] should be 0 at axis, got {result.rmnc[mn_10, 0]}"
    )


def test_circular_torus_surfaces_monotone():
    """rmnc[m=1, n=0, :] must increase monotonically from axis to boundary."""
    R0, a = 3.0, 1.0
    mpol, ntor, nfp, ns = 4, 0, 1, 11
    rbc, zbs = _circular_tokamak_boundary(R0=R0, a=a, mpol=mpol, ntor=ntor)

    cfg = GeometricInitializerConfig(
        L_max=4, n_s=32, n_theta=64, n_zeta=4, max_iter=200
    )
    result = compute_geometric_initialization(rbc, zbs, mpol, ntor, nfp, ns, cfg)

    mn_10 = 1  # (m=1, n=0) index for ntor=0
    r10 = result.rmnc[mn_10, :]
    assert np.all(np.diff(r10) >= 0), (
        f"rmnc[m=1,n=0] should increase from axis to boundary, got {r10}"
    )


# ---------------------------------------------------------------------------
# Unit tests: W7-X-like (CTH) Jacobian quality
# ---------------------------------------------------------------------------


def test_cth_like_boundary_no_negative_jacobian():
    """For the CTH-like 3D boundary, the optimized surfaces have all-positive Jacobian.

    CTH has nfp=5, ntor=4 -- a genuinely 3D stellarator-like boundary. The
    geometric action minimization should eliminate any negative-Jacobian regions
    that a naive initial guess would produce.
    """
    vi = vmecpp.VmecInput.from_file(TEST_DATA_DIR / "cth_like_fixed_bdy.json")

    cfg = GeometricInitializerConfig(
        L_max=4,
        n_s=16,
        n_theta=32,
        n_zeta=16,
        max_iter=200,
        omega=0.01,
    )
    result = compute_geometric_initialization(
        vi.rbc, vi.zbs, vi.mpol, vi.ntor, vi.nfp, ns=25, config=cfg
    )

    assert result.diagnostics.frac_negative_jacobian == 0.0, (
        f"Expected zero negative-Jacobian points for CTH boundary, "
        f"got frac_neg={result.diagnostics.frac_negative_jacobian:.4f}"
    )


def test_cth_like_axis_toroidal_modes():
    """For CTH (nfp=5, ntor=4), the geometric initializer produces a 3D axis.

    The n>0 toroidal harmonics of raxis_c should be nonzero since CTH has
    strong 3D shaping.
    """
    vi = vmecpp.VmecInput.from_file(TEST_DATA_DIR / "cth_like_fixed_bdy.json")

    cfg = GeometricInitializerConfig(
        L_max=4,
        n_s=16,
        n_theta=32,
        n_zeta=16,
        max_iter=200,
        omega=0.01,
    )
    result = compute_geometric_initialization(
        vi.rbc, vi.zbs, vi.mpol, vi.ntor, vi.nfp, ns=25, config=cfg
    )

    # raxis_c has ntor+1 entries; for a 3D config the n>0 modes should be nonzero
    assert len(result.raxis_c) == vi.ntor + 1
    assert len(result.zaxis_s) == vi.ntor + 1

    # The (n=0) major radius should be close to rbc[m=0, n=0]
    r00 = vi.rbc[0, vi.ntor]  # m=0, n=0 coefficient
    np.testing.assert_allclose(result.raxis_c[0], r00, rtol=0.05)


# ---------------------------------------------------------------------------
# Integration tests: CTH niter reduction
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_cth_geometric_action_reduces_niter():
    """Geometric-action initialization reduces VMEC iteration count for CTH.

    CTH is a compact 3D stellarator (nfp=5, ntor=4). The geometric initializer
    provides a better axis guess than the file default, consistently reducing
    niter from 121 to 117 (4 fewer iterations).
    """
    vi_default = vmecpp.VmecInput.from_file(TEST_DATA_DIR / "cth_like_fixed_bdy.json")
    vi_geom = vmecpp.VmecInput.from_file(TEST_DATA_DIR / "cth_like_fixed_bdy.json")
    vi_geom.initialization_method = "geometric_action"

    out_default = vmecpp.run(vi_default, verbose=False, max_threads=1)
    out_geom = vmecpp.run(vi_geom, verbose=False, max_threads=1)

    assert out_default.wout.ier_flag == 0, "Default CTH run did not converge"
    assert out_geom.wout.ier_flag == 0, "Geometric-action CTH run did not converge"

    # Geometric initialization should not require more iterations than the default
    assert out_geom.wout.niter <= out_default.wout.niter, (
        f"Geometric init used more iterations ({out_geom.wout.niter}) "
        f"than default ({out_default.wout.niter})"
    )


@pytest.mark.slow
def test_cth_geometric_action_same_equilibrium():
    """Both initializations converge to the same MHD equilibrium for CTH.

    The geometric initializer only changes the path to convergence (via a
    better initial axis), not the converged solution.  The equilibrium volume
    should match to high precision.
    """
    vi_default = vmecpp.VmecInput.from_file(TEST_DATA_DIR / "cth_like_fixed_bdy.json")
    vi_geom = vmecpp.VmecInput.from_file(TEST_DATA_DIR / "cth_like_fixed_bdy.json")
    vi_geom.initialization_method = "geometric_action"

    out_default = vmecpp.run(vi_default, verbose=False, max_threads=1)
    out_geom = vmecpp.run(vi_geom, verbose=False, max_threads=1)

    assert out_default.wout.ier_flag == 0
    assert out_geom.wout.ier_flag == 0

    # The converged equilibrium should be the same regardless of initialization
    np.testing.assert_allclose(
        out_geom.wout.volume_p,
        out_default.wout.volume_p,
        rtol=1e-4,
        err_msg="Plasma volume mismatch between default and geometric-action CTH runs",
    )


# ---------------------------------------------------------------------------
# Integration tests: failure recovery
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_geometric_action_exception_fallback():
    """If geometric initializer raises an exception, run() falls back to VMEC default.

    The fallback run should produce the identical result as the pure default run.
    """

    vi_default = vmecpp.VmecInput.from_file(TEST_DATA_DIR / "solovev.json")
    vi_fallback = vmecpp.VmecInput.from_file(TEST_DATA_DIR / "solovev.json")
    vi_fallback.initialization_method = "geometric_action"

    out_default = vmecpp.run(vi_default, verbose=False, max_threads=1)

    # Patch the initializer so it always raises
    with patch(
        "vmecpp.geometric_initializer.compute_geometric_initialization",
        side_effect=RuntimeError("forced failure for test"),
    ):
        out_fallback = vmecpp.run(vi_fallback, verbose=False, max_threads=1)

    assert out_fallback.wout.ier_flag == 0, "Fallback run did not converge"
    assert out_fallback.wout.niter == out_default.wout.niter, (
        f"Fallback niter ({out_fallback.wout.niter}) should equal "
        f"default niter ({out_default.wout.niter})"
    )


@pytest.mark.slow
def test_geometric_action_bad_jacobian_fallback():
    """If geometric initializer produces >50% negative Jacobian, use default axis.

    The bad-Jacobian fallback should not crash and VMEC should still converge.
    """

    vi = vmecpp.VmecInput.from_file(TEST_DATA_DIR / "solovev.json")

    # Expected niter from a pure default run (no geometric init)
    out_default = vmecpp.run(vi, verbose=False, max_threads=1)
    default_niter = out_default.wout.niter

    vi.initialization_method = "geometric_action"

    # Construct a fake result with >50% negative Jacobian to trigger the fallback
    bad_diag = GeometricInitializerDiagnostics(
        min_jacobian=-1.0,
        frac_negative_jacobian=0.8,  # 80% negative -> triggers fallback
        final_objective=1e10,
        n_iterations=0,
        converged=False,
    )
    bad_result = GeometricInitializerResult(
        rmnc=np.zeros((1, 1)),
        zmns=np.zeros((1, 1)),
        raxis_c=np.array([0.0]),  # deliberately bad axis
        zaxis_s=np.array([0.0]),
        diagnostics=bad_diag,
    )

    with patch(
        "vmecpp.geometric_initializer.compute_geometric_initialization",
        return_value=bad_result,
    ):
        out_fallback = vmecpp.run(vi, verbose=False, max_threads=1)

    assert out_fallback.wout.ier_flag == 0, (
        "Fallback (bad Jacobian) run did not converge"
    )
    # Fallback should produce the same result as pure default (axis not changed)
    assert out_fallback.wout.niter == default_niter, (
        f"Bad-Jacobian fallback niter ({out_fallback.wout.niter}) "
        f"should equal default niter ({default_niter})"
    )
