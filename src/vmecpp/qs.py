"""Quasisymmetry objectives built only from the public geometry API."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from vmecpp.geometry import Geometry, evaluate


def magnetic_field_strength(geometry: Geometry, coordinates: jax.Array) -> jax.Array:
    """Return ``|B|`` reconstructed from ``R``, ``Z``, ``lambda``, and fluxes."""
    values = evaluate(geometry, coordinates)
    r, z, lambda_, toroidal_flux, poloidal_flux = values
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
    b_sup_theta = (poloidal_flux[1] - toroidal_flux[1] * lambda_[3]) / (
        2.0 * jnp.pi * sqrt_g
    )
    b_sup_zeta = toroidal_flux[1] * (1.0 + lambda_[2]) / (2.0 * jnp.pi * sqrt_g)
    magnetic_field = b_sup_theta * e_theta + b_sup_zeta * e_zeta
    return jnp.linalg.norm(magnetic_field)


def quasisymmetry_residual(
    geometry: Geometry,
    s: float,
    *,
    helicity: int = 0,
    ntheta: int = 32,
    nzeta: int = 32,
) -> jax.Array:
    """Return normalized non-QS Fourier power on one flux surface.

    ``helicity=0`` is quasi-axisymmetry. Other integer helicities retain modes
    satisfying ``n = helicity * m`` on the field-period toroidal domain.
    """
    theta = 2.0 * jnp.pi * jnp.arange(ntheta) / ntheta
    zeta = 2.0 * jnp.pi * jnp.arange(nzeta) / (geometry.nfp * nzeta)
    theta_grid, zeta_grid = jnp.meshgrid(theta, zeta, indexing="ij")
    coordinates = jnp.stack(
        (
            jnp.full(theta_grid.size, s),
            theta_grid.ravel(),
            zeta_grid.ravel(),
        ),
        axis=1,
    )
    b = jax.vmap(magnetic_field_strength, in_axes=(None, 0))(
        geometry, coordinates
    ).reshape(ntheta, nzeta)
    harmonics = jnp.fft.fft2(b) / (ntheta * nzeta)
    m = jnp.fft.fftfreq(ntheta, 1.0 / ntheta).astype(int)[:, None]
    n = jnp.fft.fftfreq(nzeta, 1.0 / nzeta).astype(int)[None, :]
    non_qs = n != helicity * m
    return (
        jnp.sum(jnp.where(non_qs, jnp.abs(harmonics) ** 2, 0.0))
        / jnp.abs(harmonics[0, 0]) ** 2
    )
