# SPDX-FileCopyrightText: 2026-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Logic gates of kink parametrons that act on each other only through the vacuum field.

A kink parametron is a free-boundary equilibrium of a spherical tokamak that holds one
of two helical cores. A target crosses its kink threshold in the Biot-Savart field of
the currents of its held neighbours, and its core forms in the state that field selects.
"""

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import NamedTuple

import numpy as np
from scipy.ndimage import map_coordinates, spline_filter
from scipy.special import ellipe, ellipk

import vmecpp

MU0 = 4.0e-7 * np.pi
SPHERICAL_TOKAMAK = vmecpp.VmecInput.from_file(
    Path(__file__).parent / "data" / "spherical_tokamak.json"
)
# pres_scale of a held core, above the kink threshold of a plasma alone
HELD = 3.0e4
# pres_scale of a target below and above its kink threshold of 2.87e4 between neighbours
OFF, ON = 2.8e4, 3.3e4
# the height of the neighbours above and below a target
SEPARATION = 4.6
# the tilt about the x axis of the target's outer coil pair that biases it, in degrees
BIAS = -0.07
# the ellipticity of that pair that pins the core at 0 and 180 degrees
PIN = 0.05
# the force residual a target above threshold reaches without stellarator symmetry
ASYMMETRIC_FTOL = 3e-12
# the grid of the field tables: radius, toroidal planes, height
R_GRID = np.arange(0.2, 3.0 + 1e-9, 0.15)
NPHI = 16
Z_GRID = np.arange(-2.1, 2.1 + 1e-9, 0.15)


class Coils(NamedTuple):
    """The circular coils, in pairs above and below the midplane, and the axis wire."""

    radii: np.ndarray
    heights: np.ndarray
    currents: np.ndarray
    wire: float


def fixed_input(pres_scale):
    """The fixed-boundary spherical tokamak at pres_scale."""
    vmec_input = SPHERICAL_TOKAMAK.model_copy(deep=True)
    vmec_input.pres_scale = pres_scale
    vmec_input.ftol_array = np.array([1e-14])
    vmec_input.niter_array = np.array([100000])
    return vmec_input


def free_input(pres_scale, seed=0.0, lasym=False, ftol=1e-12):
    """The free-boundary spherical tokamak at pres_scale, with its axis guess shifted
    helically by ``seed``."""
    vmec_input = fixed_input(pres_scale)
    vmec_input.lfreeb = True
    vmec_input.mgrid_file = "NONE"
    vmec_input.nvacskip = 3
    vmec_input.ftol_array = np.array([ftol])
    vmec_input.niter_array = np.array([400000])
    vmec_input.raxis_c[1] = seed
    vmec_input.zaxis_s[1] = -seed
    if lasym:
        zeros, axis = np.zeros_like(vmec_input.rbc), np.zeros_like(vmec_input.raxis_c)
        vmec_input.lasym = True
        vmec_input.rbs, vmec_input.zbc = zeros, zeros.copy()
        vmec_input.raxis_s, vmec_input.zaxis_c = axis, axis.copy()
    return vmec_input


def solve(vmec_input, field, restart_from=None):
    return vmecpp.run(
        vmec_input, magnetic_field=field, restart_from=restart_from, verbose=False
    )


def loop_flux(radius, height, r, z):
    """R A_phi at (r, z) of a unit current in a circular loop."""
    m = np.minimum(
        4.0 * radius * r / ((radius + r) ** 2 + (z - height) ** 2), 1 - 1e-15
    )
    scale = r * MU0 / (np.pi * np.sqrt(m)) * np.sqrt(radius / r)
    return scale * ((1.0 - m / 2.0) * ellipk(m) - ellipe(m))


def loop_field(radii, heights, currents, r, z):
    """B_R and B_Z at (r, z) of circular loops."""
    radii, heights, currents = (
        np.asarray(x, float)[:, None] for x in (radii, heights, currents)
    )
    dz = z - heights
    alpha2 = radii**2 + r**2 + dz**2 - 2 * radii * r
    beta2 = radii**2 + r**2 + dz**2 + 2 * radii * r
    m = 1.0 - alpha2 / beta2
    scale = MU0 * currents / (2 * np.pi * alpha2 * np.sqrt(beta2))
    b_r = scale * dz / r * ((radii**2 + r**2 + dz**2) * ellipe(m) - alpha2 * ellipk(m))
    b_z = scale * ((radii**2 - r**2 - dz**2) * ellipe(m) + alpha2 * ellipk(m))
    return b_r.sum(0), b_z.sum(0)


def polygon_field(radius, height, tilt, ellipticity, current, r, phi, z, sides=256):
    """B_R, B_phi and B_Z of a loop of radius ``radius`` (1 + ellipticity cos 2 t)
    tilted by ``tilt`` about the x axis, as a polygon."""
    t = 2.0 * np.pi * np.arange(sides + 1) / sides
    a = radius * (1.0 + ellipticity * np.cos(2.0 * t))
    ring = np.stack(
        [
            a * np.cos(t),
            a * np.sin(t) * np.cos(tilt),
            height + a * np.sin(t) * np.sin(tilt),
        ],
        -1,
    )
    points = np.stack([r * np.cos(phi), r * np.sin(phi), z], -1)
    b = biot_savart(
        points, 0.5 * (ring[1:] + ring[:-1]), current * np.diff(ring, axis=0)
    )
    cp, sp = np.cos(phi), np.sin(phi)
    return b[:, 0] * cp + b[:, 1] * sp, -b[:, 0] * sp + b[:, 1] * cp, b[:, 2]


def biot_savart(points, positions, elements, chunk=48):
    """B at Cartesian points of the current elements I dl at positions."""
    field = np.empty_like(points)
    for start in range(0, len(points), chunk):
        d = points[start : start + chunk, None, :] - positions[None]
        field[start : start + chunk] = (
            MU0
            / (4 * np.pi)
            * np.einsum(
                "ij,ijk->ik", np.sum(d * d, -1) ** -1.5, np.cross(elements[None], d)
            )
        )
    return field


def current_elements(wout, nu=32, nv=16):
    """Cartesian positions and current elements J dV of an equilibrium, from
    sqrt(g) J^u = (dB_s/dv - dB_v/ds) / mu0 and sqrt(g) J^v = (dB_u/ds - dB_s/du) / mu0
    at the interior full-grid surfaces, scaled to ctor."""
    ds = 1.0 / (int(wout.ns) - 1)
    u = 2.0 * np.pi * (np.arange(nu) + 0.5) / nu
    v = 2.0 * np.pi * np.arange(nv) / nv
    uu, vv = (x.ravel() for x in np.meshgrid(u, v, indexing="ij"))
    xm, xn = np.asarray(wout.xm, float), np.asarray(wout.xn, float)
    xm_nyq, xn_nyq = np.asarray(wout.xm_nyq, float), np.asarray(wout.xn_nyq, float)
    angle = np.outer(uu, xm) - np.outer(vv, xn)
    c, s = np.cos(angle), np.sin(angle)
    c_nyq = np.cos(np.outer(uu, xm_nyq) - np.outer(vv, xn_nyq))
    rmnc, zmns = np.asarray(wout.rmnc)[:, 1:-1], np.asarray(wout.zmns)[:, 1:-1]
    b_s = np.asarray(wout.bsubsmns)[:, 1:-1]
    b_u, b_v = np.asarray(wout.bsubumnc), np.asarray(wout.bsubvmnc)
    # d/du and d/dv of b sin(m u - n v) are m b cos and -n b cos
    jv = c_nyq @ ((b_u[:, 2:] - b_u[:, 1:-1]) / ds - xm_nyq[:, None] * b_s) / MU0
    ju = c_nyq @ (-xn_nyq[:, None] * b_s - (b_v[:, 2:] - b_v[:, 1:-1]) / ds) / MU0
    r, z = c @ rmnc, s @ zmns
    r_u, z_u = -s @ (xm[:, None] * rmnc), c @ (xm[:, None] * zmns)
    r_v, z_v = s @ (xn[:, None] * rmnc), -c @ (xn[:, None] * zmns)
    cv, sv = np.cos(vv)[:, None], np.sin(vv)[:, None]
    e_u = np.stack([r_u * cv, r_u * sv, z_u], -1)
    e_v = np.stack([r_v * cv - r * sv, r_v * sv + r * cv, z_v], -1)
    weight = int(wout.signgs) * ds * (2 * np.pi / nu) * (2 * np.pi / nv)
    elements = weight * (ju[..., None] * e_u + jv[..., None] * e_v)
    # the toroidal current through the plane v = 0
    plane = vv == 0.0
    elements *= float(wout.ctor) / np.sum(
        int(wout.signgs) * jv[plane] * ds * 2 * np.pi / nu
    )
    return np.stack([r * cv, r * sv, z], -1).reshape(-1, 3), elements.reshape(-1, 3)


def coils(pres_scale=2.0e4):
    """The coils and the axis wire that make the boundary of the axisymmetric case at
    pres_scale a flux surface of coils and plasma."""
    vmec_input = fixed_input(pres_scale)
    wout = vmecpp.run(vmec_input, verbose=False).wout
    positions, elements = current_elements(wout, nu=128, nv=1)
    r_src, z_src = np.hypot(positions[:, 0], positions[:, 1]), positions[:, 2]
    # the toroidal current of each element of the plane v = 0
    d_i = elements[:, 1] / (2.0 * np.pi * r_src)
    u = np.linspace(0.0, np.pi, 181)
    m = np.arange(vmec_input.mpol_max)
    ntor = vmec_input.ntor_max
    r_b = np.cos(np.outer(u, m)) @ np.asarray(vmec_input.rbc)[:, ntor]
    z_b = np.sin(np.outer(u, m)) @ np.asarray(vmec_input.zbs)[:, ntor]
    psi = np.array(
        [d_i @ loop_flux(r_src, z_src, rb, zb) for rb, zb in zip(r_b, z_b, strict=True)]
    )
    # six pairs on a contour 0.6 m outside the boundary
    u_c = np.linspace(0.15, np.pi - 0.15, 8)
    r_c, z_c = 1.5 + 1.6 * np.cos(u_c), 2.2 * np.sin(u_c)
    r_c, z_c = r_c[r_c > 0.15], z_c[r_c > 0.15]
    columns = [
        loop_flux(rc, zc, r_b, z_b) + loop_flux(rc, -zc, r_b, z_b)
        for rc, zc in zip(r_c, z_c, strict=True)
    ]
    matrix = np.column_stack([*columns, -np.ones_like(r_b)])
    damping = 1e-6 * np.abs(matrix[:, :-1]).max() * np.eye(matrix.shape[1])[:-1]
    pair = np.linalg.lstsq(
        np.vstack([matrix, damping]),
        np.concatenate([-psi, np.zeros(len(damping))]),
        rcond=None,
    )[0][:-1]
    bvco = np.asarray(wout.bvco)
    wire = 2.0 * np.pi * (1.5 * bvco[-1] - 0.5 * bvco[-2]) / MU0
    return Coils(np.tile(r_c, 2), np.concatenate([z_c, -z_c]), np.tile(pair, 2), wire)


class Table:
    """B_R, B_phi and B_Z on a cylindrical grid as cubic splines, periodic in phi."""

    pad = 4

    def __init__(self, z_grid, values):
        self.z0, self.dz = z_grid[0], z_grid[1] - z_grid[0]
        padded = np.concatenate(
            [values[:, :, -self.pad :], values, values[:, :, : self.pad]], axis=2
        )
        self.coefficients = [spline_filter(v, order=3, mode="nearest") for v in padded]

    def __call__(self, r, phi, z):
        index = [
            (r - R_GRID[0]) / (R_GRID[1] - R_GRID[0]),
            np.mod(phi, 2.0 * np.pi) / (2.0 * np.pi / NPHI) + self.pad,
            (z - self.z0) / self.dz,
        ]
        return tuple(
            map_coordinates(c, index, order=3, prefilter=False, mode="nearest")
            for c in self.coefficients
        )


def grid(z_grid):
    """The points of the table grid at the heights ``z_grid`` and its shape."""
    phi_grid = 2.0 * np.pi * np.arange(NPHI) / NPHI
    r, phi, z = np.meshgrid(R_GRID, phi_grid, z_grid, indexing="ij")
    return r.ravel(), phi.ravel(), z.ravel(), r.shape


def field_of(coil_set, far=None):
    """The field function of the coils, the wire and a table of the far field."""

    def field(r, phi, z):
        b_r, b_z = loop_field(coil_set.radii, coil_set.heights, coil_set.currents, r, z)
        b_phi = MU0 * coil_set.wire / (2.0 * np.pi * r)
        if far is None:
            return b_r, b_phi, b_z
        f_r, f_phi, f_z = far(r, phi, z)
        return b_r + f_r, b_phi + f_phi, b_z + f_z

    return field


def neighbour_table(coil_set, pres_scale, seed):
    """The field of the currents of a neighbour at pres_scale, centred on the origin, at
    the heights of a target below it."""
    wout = solve(free_input(pres_scale, seed), field_of(coil_set)).wout
    positions, elements = current_elements(wout)
    z_grid = np.arange(-SEPARATION - 2.1, -SEPARATION + 2.1 + 1e-9, 0.15)
    r, phi, z, shape = grid(z_grid)
    b = biot_savart(
        np.stack([r * np.cos(phi), r * np.sin(phi), z], -1), positions, elements
    )
    cp, sp = np.cos(phi), np.sin(phi)
    values = np.stack(
        [b[:, 0] * cp + b[:, 1] * sp, -b[:, 0] * sp + b[:, 1] * cp, b[:, 2]]
    )
    return Table(z_grid, values.reshape(3, *shape))


class Write(NamedTuple):
    """The states of the neighbours above and below a target, 0 for a neighbour below
    its threshold, the tilt in degrees and the ellipticity of its outer coil pair."""

    above: int
    below: int
    tilt: float = 0.0
    pin: float = 0.0
    lasym: bool = False


def environment(coil_set, held, idle, write):
    """The field at a target of the coils and the wire, the currents of its neighbours,
    and its deformed outer coil pair."""
    r, phi, z, shape = grid(Z_GRID)
    b = np.zeros((3, r.size))
    for bit, side in ((write.above, 1.0), (write.below, -1.0)):
        table = held if bit else idle
        turn = np.pi if bit < 0 else 0.0
        if side > 0:
            b += np.stack(table(r, phi + turn, z - SEPARATION))
        else:
            # the mirror image under (R, phi, Z) -> (R, -phi, -Z) of a neighbour above
            b_r, b_phi, b_z = table(r, -phi + turn, -z - SEPARATION)
            b += np.stack([-b_r, b_phi, b_z])
    outer = int(np.argmax(coil_set.radii[: len(coil_set.radii) // 2]))
    if write.tilt or write.pin:
        for height in (coil_set.heights[outer], -coil_set.heights[outer]):
            args = (coil_set.radii[outer], height)
            current = coil_set.currents[outer]
            b += np.stack(
                polygon_field(
                    *args, np.radians(write.tilt), write.pin, current, r, phi, z
                )
            )
            b -= np.stack(polygon_field(*args, 0.0, 0.0, current, r, phi, z))
    local = Coils(
        np.tile(coil_set.radii, 3),
        np.concatenate([coil_set.heights + h for h in (0.0, SEPARATION, -SEPARATION)]),
        np.tile(coil_set.currents, 3),
        coil_set.wire,
    )
    return field_of(local, Table(Z_GRID, b.reshape(3, *shape)))


def core(wout):
    """The axis shift of the helical core of an equilibrium and its orientation in
    degrees."""
    xm, xn = np.asarray(wout.xm), np.asarray(wout.xn)
    mode = (xm == 0) & (xn == 1)
    c = float(np.asarray(wout.rmnc)[mode, 0][0])
    s = float(np.asarray(wout.rmns)[mode, 0][0]) if wout.lasym else 0.0
    return np.hypot(c, s), np.degrees(np.arctan2(-s, c)) % 360.0


def state(wout):
    """+1 or -1 for a core at 0 or 180 degrees, 0 for none."""
    amplitude, orientation = core(wout)
    return int(np.sign(np.cos(np.radians(orientation)))) if amplitude > 0.05 else 0


def anneal(context, write):
    """The equilibrium of a target that crosses its kink threshold in the field of
    ``write``."""
    field = environment(*context, write)
    low = solve(free_input(OFF, lasym=write.lasym), field)
    return solve(on_input(write), field, restart_from=low)


def on_input(write):
    """The input of a target above its threshold."""
    ftol = ASYMMETRIC_FTOL if write.lasym else 1e-12
    return free_input(ON, lasym=write.lasym, ftol=ftol)


def hold(context, write, written):
    """The equilibrium of a written target above its threshold once its neighbours are
    below theirs."""
    field = environment(*context, write._replace(above=0, below=0))
    return solve(on_input(write), field, restart_from=written)


def nand(context, x, y):
    """The state of a target written by the neighbours above and below in x and y."""
    return state(anneal(context, Write(x, y, BIAS, PIN, lasym=True)).wout)


def context_of():
    """The coils, the field of a held neighbour and of a neighbour below threshold."""
    coil_set = coils()
    return (
        coil_set,
        neighbour_table(coil_set, HELD, 0.3),
        neighbour_table(coil_set, 2.0e4, 0.0),
    )


def inverter_and_latch(context, x):
    """The state a pair of neighbours in x writes and the state the target keeps once
    they are below threshold."""
    write = Write(x, x)
    written = anneal(context, write)
    return state(written.wout), state(hold(context, write, written).wout)


def main():
    context = context_of()
    with ProcessPoolExecutor() as pool:
        results = list(pool.map(inverter_and_latch, [context] * 2, (1, -1)))
    wrong = 0
    for x, (written, kept) in zip((1, -1), results, strict=True):
        wrong += written != -x or kept != -x
        print(
            f"neighbours in {x:+d}: the target forms {written:+d} and keeps {kept:+d}"
        )
    if wrong:
        message = "a written target disagrees with its inverted neighbours"
        raise RuntimeError(message)


if __name__ == "__main__":
    main()
