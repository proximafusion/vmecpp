"""Output-independent VMEC geometry for C++, NumPy, and JAX clients."""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

from vmecpp.cpp import _vmecpp  # type: ignore


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass(frozen=True)
class Geometry:
    """The minimal VMEC equilibrium geometry in the internal product basis."""

    toroidal_flux: jax.Array
    poloidal_flux: jax.Array
    r_cc: jax.Array
    r_ss: jax.Array
    r_sc: jax.Array
    r_cs: jax.Array
    z_sc: jax.Array
    z_cs: jax.Array
    z_cc: jax.Array
    z_ss: jax.Array
    lambda_sc: jax.Array
    lambda_cs: jax.Array
    lambda_cc: jax.Array
    lambda_ss: jax.Array
    nfp: int

    def tree_flatten(self):
        children = dataclasses.astuple(self)[:-1]
        return children, self.nfp

    @classmethod
    def tree_unflatten(cls, nfp, children):
        return cls(*children, nfp)  # pyright: ignore[reportCallIssue]


def _array(values, shape):
    array = np.asarray(values)
    if array.size == 0:
        array = np.zeros(shape)
    return jnp.asarray(array.reshape(shape))


def from_cpp(geometry) -> Geometry:
    """Copy a C++ geometry snapshot into a JAX pytree."""
    dimensions = geometry.dimensions
    shape = (dimensions.ns, dimensions.mpol, dimensions.ntor + 1)
    coefficients = geometry.coefficients
    return Geometry(
        jnp.asarray(geometry.toroidal_flux),
        jnp.asarray(geometry.poloidal_flux),
        _array(coefficients.r_cc, shape),
        _array(coefficients.r_ss, shape),
        _array(coefficients.r_sc, shape),
        _array(coefficients.r_cs, shape),
        _array(coefficients.z_sc, shape),
        _array(coefficients.z_cs, shape),
        _array(coefficients.z_cc, shape),
        _array(coefficients.z_ss, shape),
        _array(coefficients.lambda_sc, shape),
        _array(coefficients.lambda_cs, shape),
        _array(coefficients.lambda_cc, shape),
        _array(coefficients.lambda_ss, shape),
        dimensions.nfp,
    )


def make(output) -> Geometry:
    """Construct geometry from the result of the low-level C++ ``run`` call."""
    return from_cpp(_vmecpp.make_geometry(output))


def _radial_jet(values, s):
    """Return value, first, and second radial derivatives of an interpolant."""
    ns = values.shape[0]
    scaled = s * (ns - 1)
    if ns >= 4:
        start = jnp.clip(jnp.floor(scaled).astype(int) - 1, 0, ns - 4)
        x = scaled - start
        value_weights = jnp.asarray(
            [
                -(x - 1) * (x - 2) * (x - 3) / 6,
                x * (x - 2) * (x - 3) / 2,
                -x * (x - 1) * (x - 3) / 2,
                x * (x - 1) * (x - 2) / 6,
            ]
        )
        first_weights = jnp.asarray(
            [
                -(3 * x**2 - 12 * x + 11) / 6,
                (3 * x**2 - 10 * x + 6) / 2,
                -(3 * x**2 - 8 * x + 3) / 2,
                (3 * x**2 - 6 * x + 2) / 6,
            ]
        )
        second_weights = jnp.asarray(
            [
                -(6 * x - 12) / 6,
                (6 * x - 10) / 2,
                -(6 * x - 8) / 2,
                (6 * x - 6) / 6,
            ]
        )
        sample = values[start + jnp.arange(4)]
        scale = ns - 1
        return (
            jnp.tensordot(value_weights, sample, axes=1),
            scale * jnp.tensordot(first_weights, sample, axes=1),
            scale**2 * jnp.tensordot(second_weights, sample, axes=1),
        )
    inner = jnp.clip(jnp.floor(scaled).astype(int), 0, ns - 2)
    weight = scaled - inner
    return (
        (1.0 - weight) * values[inner] + weight * values[inner + 1],
        (ns - 1) * (values[inner + 1] - values[inner]),
        jnp.zeros_like(values[inner]),
    )


def _series_jet(geometry: Geometry, coefficients, s, theta, zeta):
    """Evaluate one product-basis Fourier series and its spatial jet."""
    mpol = coefficients[0].shape[1]
    ntor = coefficients[0].shape[2] - 1
    m = jnp.arange(mpol)[:, None]
    n = jnp.arange(ntor + 1)[None, :] * geometry.nfp
    mtheta = m * theta
    nzeta = n * zeta
    poloidal_cosine = (
        jnp.cos(mtheta),
        -m * jnp.sin(mtheta),
        -(m**2) * jnp.cos(mtheta),
    )
    poloidal_sine = (
        jnp.sin(mtheta),
        m * jnp.cos(mtheta),
        -(m**2) * jnp.sin(mtheta),
    )
    toroidal_cosine = (
        jnp.cos(nzeta),
        -n * jnp.sin(nzeta),
        -(n**2) * jnp.cos(nzeta),
    )
    toroidal_sine = (
        jnp.sin(nzeta),
        n * jnp.cos(nzeta),
        -(n**2) * jnp.sin(nzeta),
    )
    jets = [_radial_jet(coefficient, s) for coefficient in coefficients]
    value = 0.0
    ds = 0.0
    dss = 0.0
    dtheta = 0.0
    dzeta = 0.0
    ds_dtheta = 0.0
    ds_dzeta = 0.0
    dtheta2 = 0.0
    dtheta_dzeta = 0.0
    dzeta2 = 0.0
    for jet, p, q in (
        (jets[0], poloidal_cosine, toroidal_cosine),
        (jets[1], poloidal_sine, toroidal_sine),
        (jets[2], poloidal_sine, toroidal_cosine),
        (jets[3], poloidal_cosine, toroidal_sine),
    ):
        radial_value, radial_first, radial_second = jet
        term = radial_value * p[0] * q[0]
        value = value + jnp.sum(term)
        ds = ds + jnp.sum(radial_first * p[0] * q[0])
        dss = dss + jnp.sum(radial_second * p[0] * q[0])
        dtheta = dtheta + jnp.sum(radial_value * p[1] * q[0])
        dzeta = dzeta + jnp.sum(radial_value * p[0] * q[1])
        ds_dtheta = ds_dtheta + jnp.sum(radial_first * p[1] * q[0])
        ds_dzeta = ds_dzeta + jnp.sum(radial_first * p[0] * q[1])
        dtheta2 = dtheta2 + jnp.sum(radial_value * p[2] * q[0])
        dtheta_dzeta = dtheta_dzeta + jnp.sum(radial_value * p[1] * q[1])
        dzeta2 = dzeta2 + jnp.sum(radial_value * p[0] * q[2])
    return jnp.asarray(
        [
            value,
            ds,
            dtheta,
            dzeta,
            dss,
            ds_dtheta,
            ds_dzeta,
            dtheta2,
            dtheta_dzeta,
            dzeta2,
        ]
    )


def _values(geometry: Geometry, coordinates: jax.Array) -> jax.Array:
    s, theta, zeta = coordinates
    series_coefficients = (
        (geometry.r_cc, geometry.r_ss, geometry.r_sc, geometry.r_cs),
        (geometry.z_cc, geometry.z_ss, geometry.z_sc, geometry.z_cs),
        (
            geometry.lambda_cc,
            geometry.lambda_ss,
            geometry.lambda_sc,
            geometry.lambda_cs,
        ),
    )
    series = [
        _series_jet(geometry, coefficients, s, theta, zeta)
        for coefficients in series_coefficients
    ]
    toroidal = _radial_jet(geometry.toroidal_flux, s)
    poloidal = _radial_jet(geometry.poloidal_flux, s)
    return jnp.stack(
        [
            *series,
            jnp.asarray(
                [
                    toroidal[0],
                    toroidal[1],
                    0.0,
                    0.0,
                    toroidal[2],
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ]
            ),
            jnp.asarray(
                [
                    poloidal[0],
                    poloidal[1],
                    0.0,
                    0.0,
                    poloidal[2],
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ]
            ),
        ]
    )


def evaluate(geometry: Geometry, coordinates: jax.Array) -> jax.Array:
    """Return shape ``(5, 10)``: values and first/second derivatives.

    Rows are ``R``, ``Z``, ``lambda``, toroidal flux, and poloidal flux. The
    columns follow :class:`vmecpp.GeometryJet`; all formulas are explicit so
    spatial derivatives do not invoke nested automatic differentiation.
    """
    return _values(geometry, coordinates)
