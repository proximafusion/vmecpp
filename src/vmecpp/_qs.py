"""Quasisymmetry objectives built only from the public geometry API.

Re-exported from vmecpp.simsopt_compat, next to the SIMSOPT wrapper this
mirrors; not part of the solver.

The metric is the one SIMSOPT's ``QuasisymmetryRatioResidual`` implements: for
quasisymmetry the ratio ``(B x grad B . grad psi) / (B . grad B)`` is constant
on a flux surface, so

    f = sum_j w_j < [ (1/B^3) ( (N - iota M) B x grad B . grad psi
                                - (M G + N I) B . grad B ) ]^2 >

with ``< . >`` the flux-surface average, ``G`` and ``I`` the poloidal and
toroidal current profiles, and ``(M, N)`` the desired helicity. Discretized on
a uniform ``(theta, phi)`` grid over one field period this is a sum of squares,

    R = sqrt( w_j nfp dtheta dphi / V' * sqrt(g) ) / B^3
        * ( (N - iota M) B x grad B . grad psi - (M G + N I) B . grad B ),

which is what ``quasisymmetry_residuals`` returns.

Everything is computed from the product-basis geometry contract in JAX. The
flux-surface measure ``sqrt(g)`` matters: dropping it changes the objective,
and its derivative is one of the two errors that only show up end to end.
"""

from __future__ import annotations

from collections.abc import Sequence

import jax
import jax.numpy as jnp

from vmecpp.geometry import Geometry, _radial_jet, evaluate

# VMEC++ fixes the sign of the Jacobian to -1, and stores the field components
# with the opposite sign to the one the raw flux derivatives give here. The
# magnitudes agree; only the convention differs. G and I are averages of the
# same covariant components, so the convention has to be applied consistently
# or the two terms of the residual pick up different signs.
_JACOBIAN_MAGNITUDE = jnp.abs


def _frame(geometry: Geometry, coordinates: jax.Array):
    """Covariant basis vectors, Jacobian and field components at one point."""
    r, z, lambda_, toroidal_flux, poloidal_flux = evaluate(geometry, coordinates)
    zeta = coordinates[2]
    cos_zeta = jnp.cos(zeta)
    sin_zeta = jnp.sin(zeta)

    e_s = jnp.asarray([r[1] * cos_zeta, r[1] * sin_zeta, z[1]])
    e_theta = jnp.asarray([r[2] * cos_zeta, r[2] * sin_zeta, z[2]])
    e_zeta = jnp.asarray(
        [
            r[3] * cos_zeta - r[0] * sin_zeta,
            r[3] * sin_zeta + r[0] * cos_zeta,
            z[3],
        ]
    )
    sqrt_g = jnp.dot(e_s, jnp.cross(e_theta, e_zeta))
    scale = 2.0 * jnp.pi * _JACOBIAN_MAGNITUDE(sqrt_g)
    b_sup_theta = (poloidal_flux[1] - toroidal_flux[1] * lambda_[3]) / scale
    b_sup_zeta = toroidal_flux[1] * (1.0 + lambda_[2]) / scale
    magnetic_field = b_sup_theta * e_theta + b_sup_zeta * e_zeta
    return {
        "sqrt_g": sqrt_g,
        "b_sup_theta": b_sup_theta,
        "b_sup_zeta": b_sup_zeta,
        "b_sub_theta": jnp.dot(magnetic_field, e_theta),
        "b_sub_zeta": jnp.dot(magnetic_field, e_zeta),
        "mod_b": jnp.linalg.norm(magnetic_field),
        "iota": poloidal_flux[1] / toroidal_flux[1],
        "toroidal_flux_derivative": toroidal_flux[1],
    }


def magnetic_field_strength(geometry: Geometry, coordinates: jax.Array) -> jax.Array:
    """Return ``|B|`` reconstructed from ``R``, ``Z``, ``lambda``, and fluxes."""
    return _frame(geometry, coordinates)["mod_b"]


def _surface_fields(geometry: Geometry, s, theta, zeta):
    """The frame quantities and the angular derivatives of |B| on a product grid."""
    m = jnp.arange(geometry.r_cc.shape[1], dtype=jnp.float64)
    n = jnp.arange(geometry.r_cc.shape[2], dtype=jnp.float64) * geometry.nfp

    def radial(name):
        return _radial_jet(getattr(geometry, name), s)

    def series(names):
        return tuple(zip(*(radial(name)[:2] for name in names), strict=True))

    r_series = series(("r_cc", "r_ss", "r_sc", "r_cs"))
    z_series = series(("z_cc", "z_ss", "z_sc", "z_cs"))
    lambda_value = tuple(
        radial(name)[0] for name in ("lambda_cc", "lambda_ss", "lambda_sc", "lambda_cs")
    )
    toroidal = _radial_jet(geometry.toroidal_flux, s)[1]
    poloidal = _radial_jet(geometry.poloidal_flux, s)[1]

    def frame(shift_theta, shift_zeta):
        m_theta = jnp.outer(theta + shift_theta, m)
        n_zeta = jnp.outer(zeta + shift_zeta, n)
        cos_m, sin_m = jnp.cos(m_theta), jnp.sin(m_theta)
        cos_n, sin_n = jnp.cos(n_zeta), jnp.sin(n_zeta)
        d_cos_m, d_sin_m = -m * sin_m, m * cos_m
        d_cos_n, d_sin_n = -n * sin_n, n * cos_n

        def grid(values, pol_c, pol_s, tor_c, tor_s):
            cc, ss, sc, cs = values
            return (
                pol_c @ cc @ tor_c.T
                + pol_s @ ss @ tor_s.T
                + pol_s @ sc @ tor_c.T
                + pol_c @ cs @ tor_s.T
            )

        plain = (cos_m, sin_m, cos_n, sin_n)
        along_theta = (d_cos_m, d_sin_m, cos_n, sin_n)
        along_zeta = (cos_m, sin_m, d_cos_n, d_sin_n)
        r = grid(r_series[0], *plain)
        r_s = grid(r_series[1], *plain)
        r_theta = grid(r_series[0], *along_theta)
        r_zeta = grid(r_series[0], *along_zeta)
        z_s = grid(z_series[1], *plain)
        z_theta = grid(z_series[0], *along_theta)
        z_zeta = grid(z_series[0], *along_zeta)
        lambda_theta = grid(lambda_value, *along_theta)
        lambda_zeta = grid(lambda_value, *along_zeta)

        sqrt_g = r * (r_theta * z_s - r_s * z_theta)
        scale = 2.0 * jnp.pi * _JACOBIAN_MAGNITUDE(sqrt_g)
        b_sup_theta = (poloidal - toroidal * lambda_zeta) / scale
        b_sup_zeta = toroidal * (1.0 + lambda_theta) / scale
        g_theta_theta = r_theta**2 + z_theta**2
        g_theta_zeta = r_theta * r_zeta + z_theta * z_zeta
        g_zeta_zeta = r_zeta**2 + r**2 + z_zeta**2
        b_sub_theta = b_sup_theta * g_theta_theta + b_sup_zeta * g_theta_zeta
        b_sub_zeta = b_sup_theta * g_theta_zeta + b_sup_zeta * g_zeta_zeta
        mod_b = jnp.sqrt(b_sup_theta * b_sub_theta + b_sup_zeta * b_sub_zeta)
        return mod_b, {
            "sqrt_g": sqrt_g,
            "b_sup_theta": b_sup_theta,
            "b_sup_zeta": b_sup_zeta,
            "b_sub_theta": b_sub_theta,
            "b_sub_zeta": b_sub_zeta,
            "mod_b": mod_b,
            "iota": poloidal / toroidal,
        }

    fields = frame(0.0, 0.0)[1]
    fields["d_mod_b_d_theta"] = jax.jvp(
        lambda shift: frame(shift, 0.0)[0], (0.0,), (1.0,)
    )[1]
    fields["d_mod_b_d_zeta"] = jax.jvp(
        lambda shift: frame(0.0, shift)[0], (0.0,), (1.0,)
    )[1]
    return fields


def quasisymmetry_residuals(
    geometry: Geometry,
    surfaces: Sequence[float] = (0.5,),
    *,
    helicity_m: int = 1,
    helicity_n: int = 0,
    weights: Sequence[float] | None = None,
    ntheta: int = 63,
    nphi: int = 64,
) -> jax.Array:
    """Return the flat vector of quasisymmetry residuals ``R``.

    ``helicity_n = 0`` is quasi-axisymmetry. The residuals are normalized so
    that the sum of their squares is the objective ``f``; this matches
    SIMSOPT's ``QuasisymmetryRatioResidual.residuals()`` term by term.
    """
    if geometry.toroidal_transits != 1:
        msg = "quasisymmetry_residuals requires toroidal_transits = 1"
        raise NotImplementedError(msg)
    surface_values = jnp.asarray(surfaces, dtype=jnp.float64)
    surface_weights = (
        jnp.ones_like(surface_values)
        if weights is None
        else jnp.asarray(weights, dtype=jnp.float64)
    )
    nfp = geometry.nfp

    theta_1d = 2.0 * jnp.pi * jnp.arange(ntheta) / ntheta
    phi_1d = 2.0 * jnp.pi * jnp.arange(nphi) / (nfp * nphi)
    d_theta = theta_1d[1] - theta_1d[0]
    d_phi = phi_1d[1] - phi_1d[0]

    # 2 pi psi is the toroidal flux, and VMEC++'s negative Jacobian carries the
    # sign SIMSOPT writes explicitly as -phi_edge / (2 pi).
    edge_flux = evaluate(geometry, jnp.asarray([1.0, 0.0, 0.0]))[3][0]
    d_psi_d_s = -edge_flux / (2.0 * jnp.pi)

    toroidal_mode = helicity_n * nfp

    def one_surface(s, weight):
        fields = _surface_fields(geometry, s, theta_1d, phi_1d)
        sqrt_g = fields["sqrt_g"]
        mod_b = fields["mod_b"]
        d_theta_b = fields["d_mod_b_d_theta"]
        d_zeta_b = fields["d_mod_b_d_zeta"]

        b_dot_grad_b = (
            fields["b_sup_theta"] * d_theta_b + fields["b_sup_zeta"] * d_zeta_b
        )
        b_cross_grad_b_dot_grad_psi = (
            d_psi_d_s
            * (fields["b_sub_theta"] * d_zeta_b - fields["b_sub_zeta"] * d_theta_b)
            / sqrt_g
        )
        # G and I are the (0, 0) harmonics of the covariant components, i.e.
        # their plain angular averages on the uniform grid.
        current_g = jnp.mean(fields["b_sub_zeta"])
        current_i = jnp.mean(fields["b_sub_theta"])
        iota = jnp.mean(fields["iota"])

        volume_derivative = nfp * d_theta * d_phi * jnp.sum(sqrt_g)
        measure = jnp.sqrt(weight * nfp * d_theta * d_phi / volume_derivative * sqrt_g)
        return (
            measure
            * (
                b_cross_grad_b_dot_grad_psi * (toroidal_mode - iota * helicity_m)
                - b_dot_grad_b * (helicity_m * current_g + toroidal_mode * current_i)
            )
            / mod_b**3
        ).ravel()

    return jax.vmap(one_surface)(surface_values, surface_weights).ravel()


def quasisymmetry_total(geometry: Geometry, *args, **kwargs) -> jax.Array:
    """Return the scalar quasisymmetry error ``f``, the sum of squared residuals."""
    residuals = quasisymmetry_residuals(geometry, *args, **kwargs)
    return jnp.sum(residuals**2)
