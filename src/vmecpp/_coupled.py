# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Several free-boundary plasmas in one vacuum field.

Each plasma is a free-boundary run in its own cylindrical frame, which a rotation and
a translation place in a common Cartesian frame. The plasmas share the external field
and the field of each other's currents. Outside a plasma, its currents produce the
field of the surface current ``n x B / mu_0`` on its boundary, with ``B`` the field
just outside and ``n`` the outward normal (the virtual casing principle). Each run
takes the virtual casing fields of the other plasmas as part of its external field,
and the runs alternate, each hot-restarted from its last solution, until the
boundaries stop changing. One vacuum field, the external field plus the field of
every plasma, is then tangent to every boundary and in pressure balance with every
plasma.
"""

from __future__ import annotations

import collections.abc
import dataclasses
import logging
import typing

import jaxtyping as jt
import numpy as np
import pydantic
from scipy.special import ellipe, ellipk

from vmecpp._pydantic_numpy import BaseModelWithNumpy

if typing.TYPE_CHECKING:
    from vmecpp import VmecInput, VmecOutput
    from vmecpp._free_boundary import MagneticFieldFunction

_logger = logging.getLogger(__name__)


class Placement(BaseModelWithNumpy):
    """Position of a plasma's frame in the common frame.

    A point with Cartesian coordinates ``x`` in the frame of the plasma, whose
    cylindrical coordinates are those of its run, sits at ``rotation @ x +
    translation`` in the common frame.
    """

    model_config = pydantic.ConfigDict(extra="forbid")

    rotation: jt.Float[np.ndarray, "3 3"] = pydantic.Field(
        default_factory=lambda: np.eye(3)
    )
    """A proper rotation."""

    translation: jt.Float[np.ndarray, "3"] = pydantic.Field(
        default_factory=lambda: np.zeros(3)
    )

    @pydantic.model_validator(mode="after")
    def _check_rotation(self) -> Placement:
        rotation = np.asarray(self.rotation, dtype=float)
        if rotation.shape != (3, 3) or np.shape(self.translation) != (3,):
            msg = "rotation has to be 3 x 3 and translation a 3-vector"
            raise ValueError(msg)
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-12) or not (
            np.linalg.det(rotation) > 0.0
        ):
            msg = "rotation has to be a proper rotation matrix"
            raise ValueError(msg)
        return self

    def is_coaxial(self) -> bool:
        """Whether the plasma's axis of symmetry is the z axis of the common frame."""
        rotation = np.asarray(self.rotation, dtype=float)
        translation = np.asarray(self.translation, dtype=float)
        return bool(
            np.allclose(np.abs(rotation[:, 2]), [0.0, 0.0, 1.0], atol=1e-12)
            and np.allclose(translation[:2], 0.0, atol=1e-12)
        )


def _cylindrical_to_cartesian(r, phi, z) -> np.ndarray:
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=-1)


def _vectors_to_cartesian(b_r, b_phi, b_z, phi) -> np.ndarray:
    cos, sin = np.cos(phi), np.sin(phi)
    return np.stack([b_r * cos - b_phi * sin, b_r * sin + b_phi * cos, b_z], axis=-1)


def _vectors_to_cylindrical(b, phi) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cos, sin = np.cos(phi), np.sin(phi)
    return (
        b[..., 0] * cos + b[..., 1] * sin,
        -b[..., 0] * sin + b[..., 1] * cos,
        b[..., 2],
    )


def _spectral_derivative(f: np.ndarray, axis: int) -> np.ndarray:
    """Derivative with respect to the angle sampled evenly over 2 pi along axis."""
    n = f.shape[axis]
    wavenumbers = np.fft.fftfreq(n, d=1.0 / n)
    if n % 2 == 0:
        wavenumbers[n // 2] = 0.0
    shape = [1] * f.ndim
    shape[axis] = n
    spectrum = np.fft.fft(f, axis=axis) * (1j * wavenumbers.reshape(shape))
    return np.real(np.fft.ifft(spectrum, axis=axis))


def _upsample(f: np.ndarray, axis: int, factor: int) -> np.ndarray:
    """Trigonometric interpolation of samples evenly spaced over a period onto a grid
    factor times finer."""
    if factor == 1:
        return f
    n = f.shape[axis]
    spectrum = np.fft.fft(f, axis=axis)
    m = n * factor
    padded_shape = list(f.shape)
    padded_shape[axis] = m
    padded = np.zeros(padded_shape, dtype=complex)
    half = (n + 1) // 2
    low = [slice(None)] * f.ndim
    low[axis] = slice(0, half)
    padded[tuple(low)] = spectrum[tuple(low)]
    high_src = [slice(None)] * f.ndim
    high_src[axis] = slice(n - (n - half), n)
    high_dst = [slice(None)] * f.ndim
    high_dst[axis] = slice(m - (n - half), m)
    padded[tuple(high_dst)] = spectrum[tuple(high_src)]
    if n % 2 == 0:
        # split the Nyquist mode between the two signs of the wavenumber
        nyquist = [slice(None)] * f.ndim
        nyquist[axis] = slice(n // 2, n // 2 + 1)
        mirror = [slice(None)] * f.ndim
        mirror[axis] = slice(m - n // 2, m - n // 2 + 1)
        padded[tuple(nyquist)] = 0.5 * spectrum[tuple(nyquist)]
        padded[tuple(mirror)] = 0.5 * spectrum[tuple(nyquist)]
    return np.real(np.fft.ifft(padded, axis=axis)) * factor


@dataclasses.dataclass(frozen=True)
class _BoundaryGrid:
    """The boundary and the vacuum field on it, in the frame of the plasma, on a grid
    over the whole torus: [toroidal, poloidal], both evenly spaced over 2 pi."""

    r: np.ndarray
    z: np.ndarray
    phi: np.ndarray
    b_r: np.ndarray
    b_phi: np.ndarray
    b_z: np.ndarray


def _boundary_grid(
    output: VmecOutput, poloidal_upsampling: int, toroidal_points: int
) -> _BoundaryGrid:
    """The boundary and its vacuum field from a free-boundary run, extended to the whole
    poloidal range by stellarator symmetry if the run assumes it, to the whole torus
    over the field periods, and upsampled by trigonometric interpolation.

    An axisymmetric run is repeated over toroidal_points angles; a three-dimensional run
    is upsampled to at least that many toroidal points.
    """
    fb = output.threed1_free_boundary
    nfp = int(output.wout.nfp)
    values = {
        name: np.asarray(getattr(fb, name), dtype=float)
        for name in ("rb", "zb", "brv", "bphiv", "bzv")
    }
    num_zeta, num_theta = values["rb"].shape
    if not output.wout.lasym:
        # theta -> 2 pi - theta, zeta -> -zeta: R, B^phi and B^Z are even, Z and B^R
        # odd
        full = 2 * (num_theta - 1)
        theta_index = full - np.arange(num_theta, full)
        zeta_index = (-np.arange(num_zeta)) % num_zeta
        for name, sign in (
            ("rb", 1.0),
            ("zb", -1.0),
            ("brv", -1.0),
            ("bphiv", 1.0),
            ("bzv", 1.0),
        ):
            mirrored = sign * values[name][np.ix_(zeta_index, theta_index)]
            values[name] = np.concatenate([values[name], mirrored], axis=1)
    if num_zeta == 1:
        values = {
            name: np.repeat(value, toroidal_points, axis=0)
            for name, value in values.items()
        }
    else:
        values = {name: np.tile(value, (nfp, 1)) for name, value in values.items()}
        factor = max(1, int(np.ceil(toroidal_points / (num_zeta * nfp))))
        values = {name: _upsample(value, 0, factor) for name, value in values.items()}
    values = {
        name: _upsample(value, 1, poloidal_upsampling) for name, value in values.items()
    }
    num_phi = values["rb"].shape[0]
    phi = np.broadcast_to(
        (2.0 * np.pi * np.arange(num_phi) / num_phi)[:, None], values["rb"].shape
    )
    return _BoundaryGrid(
        r=values["rb"],
        z=values["zb"],
        phi=np.array(phi),
        b_r=values["brv"],
        b_phi=values["bphiv"],
        b_z=values["bzv"],
    )


@dataclasses.dataclass(frozen=True)
class _SurfaceCurrent:
    """Quadrature points of the virtual casing integral in the common frame, and the
    surface current n x B dS / (4 pi) at each."""

    points: np.ndarray
    currents: np.ndarray

    def field(self, targets: np.ndarray) -> np.ndarray:
        """Cartesian field at the points targets [n, 3] of the common frame."""
        field = np.zeros_like(targets)
        chunk = max(1, 2_000_000 // len(self.points))
        for start in range(0, len(targets), chunk):
            d = targets[start : start + chunk, None, :] - self.points[None, :, :]
            inverse_cube = np.einsum("tsk,tsk->ts", d, d) ** -1.5
            k = self.currents[None, :, :]
            cross = np.stack(
                [
                    k[..., 1] * d[..., 2] - k[..., 2] * d[..., 1],
                    k[..., 2] * d[..., 0] - k[..., 0] * d[..., 2],
                    k[..., 0] * d[..., 1] - k[..., 1] * d[..., 0],
                ],
                axis=-1,
            )
            field[start : start + chunk] = np.einsum("tsk,ts->tk", cross, inverse_cube)
        return field


def _surface_current(grid: _BoundaryGrid, placement: Placement) -> _SurfaceCurrent:
    """The virtual casing surface current of a boundary grid, by the trapezoidal rule
    over the whole torus."""
    x = _cylindrical_to_cartesian(grid.r, grid.phi, grid.z)
    x_theta = _spectral_derivative(x, axis=1)
    x_phi = _spectral_derivative(x, axis=0)
    normal = np.cross(x_theta, x_phi)
    # outward: the volume (1/3) int x . n dS is positive
    if np.sum(x * normal) < 0.0:
        normal = -normal
    b = _vectors_to_cartesian(grid.b_r, grid.b_phi, grid.b_z, grid.phi)
    num_phi, num_theta = grid.r.shape
    weight = (2.0 * np.pi / num_theta) * (2.0 * np.pi / num_phi) / (4.0 * np.pi)
    currents = np.cross(normal, b) * weight
    rotation = np.asarray(placement.rotation, dtype=float)
    translation = np.asarray(placement.translation, dtype=float)
    return _SurfaceCurrent(
        points=x.reshape(-1, 3) @ rotation.T + translation,
        currents=currents.reshape(-1, 3) @ rotation.T,
    )


@dataclasses.dataclass(frozen=True)
class _RingCurrents:
    """An axisymmetric surface current on the z axis of the common frame as coaxial
    rings: radii, heights, and currents times mu_0."""

    radius: np.ndarray
    height: np.ndarray
    current: np.ndarray

    def field(self, targets: np.ndarray) -> np.ndarray:
        """Cartesian field at the points targets [n, 3] of the common frame, from the
        field of a circular loop in complete elliptic integrals."""
        x, y, z = targets[:, 0], targets[:, 1], targets[:, 2]
        r = np.hypot(x, y)
        a = self.radius[None, :]
        dz = z[:, None] - self.height[None, :]
        rr = r[:, None]
        alpha2 = (a - rr) ** 2 + dz**2
        beta2 = (a + rr) ** 2 + dz**2
        m = 1.0 - alpha2 / beta2
        scale = self.current[None, :] / (2.0 * np.pi * alpha2 * np.sqrt(beta2))
        e, k = ellipe(m), ellipk(m)
        b_r = np.sum(
            scale * dz / rr * ((a**2 + rr**2 + dz**2) * e - alpha2 * k), axis=1
        )
        b_z = np.sum(scale * ((a**2 - rr**2 - dz**2) * e + alpha2 * k), axis=1)
        phi = np.arctan2(y, x)
        return _vectors_to_cartesian(b_r, np.zeros_like(b_r), b_z, phi)


def _ring_currents(grid: _BoundaryGrid, placement: Placement) -> _RingCurrents:
    """The toroidal part of the virtual casing surface current of an axisymmetric
    boundary, which alone produces the field outside the torus, as rings.

    The poloidal part, from the toroidal field on the boundary, makes a field only
    inside the torus.
    """
    r, z = grid.r[0], grid.z[0]
    b_r, b_z = grid.b_r[0], grid.b_z[0]
    r_theta = _spectral_derivative(r, axis=0)
    z_theta = _spectral_derivative(z, axis=0)
    # the outward normal times |dx/dtheta| is (z_theta, -r_theta) for a boundary that
    # runs counterclockwise in (R, Z)
    orientation = np.sign(np.sum(r * z_theta - z * r_theta))
    # (n x B) . e_phi |dx/dtheta| dtheta = (n_Z B_R - n_R B_Z) |dx/dtheta| dtheta
    current = -orientation * (r_theta * b_r + z_theta * b_z) * (2.0 * np.pi / len(r))
    # a rotation that turns the z axis over also turns the toroidal direction over
    flip = float(np.sign(np.asarray(placement.rotation, dtype=float)[2, 2]))
    return _RingCurrents(
        radius=r,
        height=flip * z + float(placement.translation[2]),
        current=flip * current,
    )


_Source = _SurfaceCurrent | _RingCurrents

# The quadrature points of a plasma's field are spaced by at most this fraction of the
# distance to the nearest point where the field is wanted. The trapezoidal rule then
# makes a relative error of about exp(-2 pi / fraction).
_SPACING_PER_DISTANCE = 0.25
_MAX_UPSAMPLING = 32


def _plasma_field(
    output: VmecOutput,
    placement: Placement,
    *,
    axisymmetric_rings: bool,
    distance: float,
) -> _Source:
    """The field of a plasma's currents in the common frame, for points at least
    ``distance`` away from its boundary."""
    grid = _boundary_grid(output, 1, 1)
    spacing = _SPACING_PER_DISTANCE * distance
    x = _cylindrical_to_cartesian(grid.r, grid.phi, grid.z)
    num_phi, num_theta = grid.r.shape
    poloidal_step = np.linalg.norm(_spectral_derivative(x, axis=1), axis=-1).max()
    poloidal = int(np.ceil(poloidal_step * 2.0 * np.pi / num_theta / spacing))
    if axisymmetric_rings:
        grid = _boundary_grid(output, min(max(poloidal, 1), _MAX_UPSAMPLING), 1)
        return _ring_currents(grid, placement)
    toroidal_points = int(np.ceil(2.0 * np.pi * grid.r.max() / spacing))
    if num_phi > 1:
        toroidal_points = min(toroidal_points, num_phi * _MAX_UPSAMPLING)
    grid = _boundary_grid(
        output, min(max(poloidal, 1), _MAX_UPSAMPLING), max(toroidal_points, num_phi)
    )
    return _surface_current(grid, placement)


def _distance(
    output_a: VmecOutput,
    placement_a: Placement,
    output_b: VmecOutput,
    placement_b: Placement,
    *,
    axisymmetric: bool,
) -> float:
    """The smallest distance between the boundary points of two plasmas; for coaxial
    axisymmetric plasmas, between their boundaries in the (R, Z) plane."""
    a = _boundary_points(output_a, placement_a)
    b = _boundary_points(output_b, placement_b)
    if axisymmetric:
        a = np.stack([np.hypot(a[:, 0], a[:, 1]), a[:, 2]], axis=1)
        b = np.stack([np.hypot(b[:, 0], b[:, 1]), b[:, 2]], axis=1)
    differences = a[:, None, :] - b[None, :, :]
    return float(np.sqrt(np.einsum("abk,abk->ab", differences, differences).min()))


def _external_field_of(
    magnetic_field: MagneticFieldFunction,
    placement: Placement,
    sources: collections.abc.Sequence[_Source],
) -> MagneticFieldFunction:
    """The external field of one plasma in its own frame: the common external field
    plus the fields of the other plasmas."""
    rotation = np.asarray(placement.rotation, dtype=float)
    translation = np.asarray(placement.translation, dtype=float)
    # the frame of the plasma is the common frame
    identity = np.array_equal(rotation, np.eye(3)) and not np.any(translation)

    def field(r, phi, z):
        r, phi, z = (np.asarray(v, dtype=float) for v in (r, phi, z))
        points = _cylindrical_to_cartesian(r, phi, z) @ rotation.T + translation
        if identity:
            b_r, b_phi, b_z = (
                np.asarray(v, dtype=float) for v in magnetic_field(r, phi, z)
            )
        else:
            r_c = np.hypot(points[:, 0], points[:, 1])
            phi_c = np.arctan2(points[:, 1], points[:, 0])
            b_r_c, b_phi_c, b_z_c = (
                np.asarray(v, dtype=float)
                for v in magnetic_field(r_c, phi_c, points[:, 2])
            )
            b_c = _vectors_to_cartesian(b_r_c, b_phi_c, b_z_c, phi_c)
            b_r, b_phi, b_z = _vectors_to_cylindrical(b_c @ rotation, phi)
        if sources:
            b_sources = np.zeros_like(points)
            for source in sources:
                b_sources += source.field(points)
            s_r, s_phi, s_z = _vectors_to_cylindrical(b_sources @ rotation, phi)
            b_r, b_phi, b_z = b_r + s_r, b_phi + s_phi, b_z + s_z
        return b_r, b_phi, b_z

    return field


def _boundary_points(output: VmecOutput, placement: Placement) -> np.ndarray:
    """The boundary points of the vacuum solve [n, 3] in the common frame."""
    fb = output.threed1_free_boundary
    r, phi, z = (np.ravel(getattr(fb, name)) for name in ("rb", "phib", "zb"))
    rotation = np.asarray(placement.rotation, dtype=float)
    return _cylindrical_to_cartesian(r, phi, z) @ rotation.T + np.asarray(
        placement.translation, dtype=float
    )


def _vacuum_field_scale(output: VmecOutput) -> float:
    """The largest vacuum field on the boundary."""
    fb = output.threed1_free_boundary
    return float(
        np.sqrt(
            np.asarray(fb.brv) ** 2
            + np.asarray(fb.bphiv) ** 2
            + np.asarray(fb.bzv) ** 2
        ).max()
    )


def _restart_input(vmec_input: VmecInput) -> VmecInput:
    """The input with only the last step of its multigrid schedule, as a hot restart
    from a solution at that resolution needs."""
    return vmec_input.model_copy(
        update={
            "ns_array": np.asarray(vmec_input.ns_array)[-1:],
            "ftol_array": np.asarray(vmec_input.ftol_array)[-1:],
            "niter_array": np.asarray(vmec_input.niter_array)[-1:],
        }
    )


def run_coupled(
    inputs: collections.abc.Sequence[VmecInput],
    placements: collections.abc.Sequence[Placement],
    magnetic_field: MagneticFieldFunction,
    *,
    tolerance: float = 1.0e-8,
    max_iterations: int = 30,
    max_threads: int | None = None,
) -> list[VmecOutput]:
    """Free-boundary equilibria of several plasmas in one vacuum field.

    Each input is a free-boundary run, which ``placements`` puts into the common
    frame. ``magnetic_field`` is the external field, ``b_r, b_phi, b_z = f(r, phi,
    z)`` in cylindrical coordinates about the z axis of the common frame. Every
    plasma sees the external field and the fields of the other plasmas' currents,
    which the virtual casing principle gives from their boundaries and the vacuum
    field on them. The first round solves every plasma alone in the external field.
    Each later round solves the runs in turn, each hot-restarted from its last
    solution and converged to its own ``ftol``, in the fields of the others' latest
    solutions. The rounds end when the field that the other plasmas make on each
    boundary differs from the one its run was solved in by at most ``tolerance``
    times the largest vacuum field on that boundary.

    The field of another plasma breaks the stellarator symmetry, so every input
    needs ``lasym=True``. Plasmas without toroidal resolution (``ntor = 0``) have to
    share the z axis of the common frame with each other; otherwise every plasma
    needs ``nfp = 1`` and ``ntor > 0``, and the external field has to be given
    everywhere around the plasmas. The plasmas must not overlap.

    Returns:
        The output of every plasma, in the order of the inputs.

    Raises:
        RuntimeError: if the fields still disagree after ``max_iterations`` rounds.
    """
    import vmecpp  # noqa: PLC0415  (lazy import avoids a circular import)

    inputs = [vmecpp.VmecInput.model_validate(i) for i in inputs]
    placements = [Placement.model_validate(p) for p in placements]
    if len(inputs) != len(placements) or not inputs:
        msg = "run_coupled needs one placement per input, and at least one input"
        raise ValueError(msg)
    for index, vmec_input in enumerate(inputs):
        if not vmec_input.lfreeb:
            msg = f"input {index} is not a free-boundary run (lfreeb=False)"
            raise ValueError(msg)
        if not vmec_input.lasym:
            msg = (
                f"input {index} assumes stellarator symmetry, which the field of "
                "the other plasmas breaks: set lasym=True"
            )
            raise ValueError(msg)
    axisymmetric = [int(i.ntor) == 0 for i in inputs]
    if any(axisymmetric):
        if not all(axisymmetric) or not all(p.is_coaxial() for p in placements):
            msg = (
                "plasmas with ntor = 0 can only be coupled to each other on the z "
                "axis of the common frame"
            )
            raise ValueError(msg)
    elif any(int(i.nfp) != 1 for i in inputs):
        msg = (
            "three-dimensional plasmas need nfp = 1, since the field of the others "
            "breaks their periodicity"
        )
        raise ValueError(msg)
    rings = all(axisymmetric)

    outputs: list[VmecOutput | None] = [None] * len(inputs)
    # the field of each plasma, from its latest solution
    fields: list[_Source | None] = [None] * len(inputs)
    # the fields of the others that each run was last solved in
    solved_in: list[dict[int, _Source]] = [{} for _ in inputs]

    def update_fields() -> None:
        """The field of every solved plasma, resolved for the nearest other one."""
        for j, output in enumerate(outputs):
            if output is None:
                continue
            distances = [
                _distance(
                    output,
                    placements[j],
                    other,
                    placements[k],
                    axisymmetric=rings,
                )
                for k, other in enumerate(outputs)
                if k != j and other is not None
            ]
            if distances:
                fields[j] = _plasma_field(
                    output,
                    placements[j],
                    axisymmetric_rings=rings,
                    distance=min(distances),
                )

    for iteration in range(max_iterations):
        for i, vmec_input in enumerate(inputs):
            solved_in[i] = {
                j: field
                for j, field in enumerate(fields)
                if j != i and field is not None
            }
            last = outputs[i]
            outputs[i] = vmecpp.run(
                vmec_input if last is None else _restart_input(vmec_input),
                _external_field_of(
                    magnetic_field, placements[i], list(solved_in[i].values())
                ),
                max_threads=max_threads,
                verbose=False,
                restart_from=last,
            )
            update_fields()
        mismatch = 0.0
        for i, output in enumerate(outputs):
            assert output is not None
            points = _boundary_points(output, placements[i])
            difference = np.zeros_like(points)
            for j, field in enumerate(fields):
                if j == i or field is None:
                    continue
                difference += field.field(points)
                if j in solved_in[i]:
                    difference -= solved_in[i][j].field(points)
            mismatch = max(
                mismatch,
                float(np.linalg.norm(difference, axis=1).max())
                / _vacuum_field_scale(output),
            )
        _logger.info("round %d: largest field mismatch %.3e", iteration, mismatch)
        if mismatch <= tolerance:
            return typing.cast(list["VmecOutput"], outputs)
    msg = (
        f"the fields of the coupled plasmas still disagree after {max_iterations} "
        "rounds; raise max_iterations or tolerance"
    )
    raise RuntimeError(msg)


__all__ = ["Placement", "run_coupled"]
