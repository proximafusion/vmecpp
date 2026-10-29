# SPDX-FileCopyrightText: 2026-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Stellarators whose magnetic axis is a torus knot, the trefoil knotatron of Hudson,
Startsev and Feibush (2014) among them, as VMEC++ domains over several transits."""

import argparse
import dataclasses
import time

import numpy as np
from island_interior import axis_curve, fourier
from scipy.optimize import brentq, least_squares, minimize_scalar
from simsopt.field import BiotSavart, Coil, Current, InterpolatedField
from simsopt.geo import CurveXYZFourier

import vmecpp

NS = [13, 25]
PRESSURES = (1000.0, 2000.0, 4000.0)  # p0 in Pa; the coils carry 1 MA each


@dataclasses.dataclass(frozen=True)
class Knot:
    """The (p, q) torus knot R = r0 + r_knot cos q z, Z = -r_knot sin q z, phi = p z in
    m, its coils, the grid of their field and the resolution of the interior."""

    p: int  # toroidal transits, after which the domain closes
    q: int  # poloidal turns, the field periods of the domain
    r0: float
    r_knot: float
    coils: int
    radius: float  # geometric mean of the semi-axes of a coil
    kappa: float = 1.0  # ratio of the semi-axes
    twist: float = 0.0  # turns of the semi-axes about the knot along its length
    box: tuple[float, float, float] = (
        0.2,
        3.8,
        1.8,
    )  # R range and largest Z of the grid
    cells: tuple[int, int, int] = (48, 48, 24)
    axis: tuple[float, float] = (2.9, 3.05)  # bracket of R of the axis at phi = 0
    planes: int = 72  # planes per period at which the surface is traced
    offset: float = 0.15  # start of the traced surface outboard of the axis
    mpol: int = 8
    ntor: int = 24
    pressures: tuple[float, ...] = ()  # p0 in Pa of the runs with pressure

    @property
    def period(self):
        """Toroidal angle of a field period of the domain."""
        return 2.0 * np.pi * self.p / self.q

    @property
    def steps(self):
        """Multiple of the trefoil's RK4 steps, one per 12 coils of a period."""
        return -(-self.coils // (12 * self.q))

    def points(self, z):
        """Cartesian points of the knot at the knot parameter z."""
        r = self.r0 + self.r_knot * np.cos(self.q * z)
        x, y = r * np.cos(self.p * z), r * np.sin(self.p * z)
        return np.stack([x, y, -self.r_knot * np.sin(self.q * z)], -1)


KNOTS = {
    # Hudson et al.: 36 circular coils of radius 2/3 m
    "trefoil": Knot(
        p=2, q=3, r0=2.0, r_knot=1.0, coils=36, radius=2.0 / 3.0, pressures=PRESSURES
    ),
    # elliptical coils of axis ratio 2, turned one and a half times along the knot
    "shaped_trefoil": Knot(
        p=2,
        q=3,
        r0=2.0,
        r_knot=1.0,
        coils=36,
        radius=0.55,
        kappa=2.0,
        twist=1.5,
        box=(0.05, 4.75, 2.55),
        cells=(64, 48, 32),
        axis=(2.9, 2.95),
        planes=96,
        offset=0.1,
        mpol=10,
        ntor=36,
    ),
    # the trefoil as the (3, 2) torus knot over three transits, with 240 coils of axis
    # ratio 3 turned six times along it
    "trefoil_3_2": Knot(
        p=3,
        q=2,
        r0=3.0,
        r_knot=1.2,
        coils=240,
        radius=0.5,
        kappa=3.0,
        twist=6.0,
        box=(0.3, 5.9, 2.7),
        cells=(64, 48, 32),
        axis=(4.2, 4.25),
        planes=112,
        offset=0.06,
        mpol=8,
        ntor=24,
    ),
    # the cinquefoil, the (2, 5) torus knot, with 50 circular coils of radius 0.6 m
    "cinquefoil": Knot(
        p=2,
        q=5,
        r0=2.0,
        r_knot=1.0,
        coils=50,
        radius=0.6,
        box=(0.05, 4.8, 2.6),
        cells=(64, 48, 32),
        axis=(2.95, 3.0),
        planes=72,
        offset=0.1,
        mpol=8,
        ntor=20,
        pressures=(1000.0, 2000.0),
    ),
}


def frenet(knot, z, h=1e-4):
    """Tangent, principal normal and binormal of the knot at z."""
    d1 = (knot.points(z + h) - knot.points(z - h)) / (2.0 * h)
    d2 = (knot.points(z + h) - 2.0 * knot.points(z) + knot.points(z - h)) / h**2
    t = d1 / np.linalg.norm(d1, axis=-1, keepdims=True)
    n = d2 - np.sum(d2 * t, axis=-1, keepdims=True) * t
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    return t, n, np.cross(t, n)


def coils(knot, tilts, current=1.0e6):
    """Elliptical coils on the knot, their axes the tangent tilted by tilts toward the
    normal and binormal, their semi-axes turned from the normal by twist z."""
    z = 2.0 * np.pi * np.arange(knot.coils) / knot.coils
    centres = knot.points(z)
    tangents, normals, binormals = frenet(knot, z)
    semi_u = knot.radius * np.sqrt(knot.kappa)
    semi_v = knot.radius / np.sqrt(knot.kappa)
    out = []
    for i in range(knot.coils):
        axis = tangents[i] + tilts[2 * i] * normals[i] + tilts[2 * i + 1] * binormals[i]
        axis /= np.linalg.norm(axis)
        e1 = normals[i] - np.dot(normals[i], axis) * axis
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(axis, e1)
        chi = knot.twist * z[i]
        u = np.cos(chi) * e1 + np.sin(chi) * e2
        v = np.cos(chi) * e2 - np.sin(chi) * e1
        curve = CurveXYZFourier(64, 1)
        # c0 + s1 sin(2 pi t) + c1 cos(2 pi t) in each coordinate
        curve.x = np.concatenate(
            [[centres[i, k], semi_v * v[k], semi_u * u[k]] for k in range(3)]
        )
        out.append(Coil(curve, Current(current)))
    return out


def misalignment(knot, tilts):
    """B / |B| along the normal and the binormal at the coil centres."""
    z = 2.0 * np.pi * np.arange(knot.coils) / knot.coils
    _, normals, binormals = frenet(knot, z)
    field = BiotSavart(coils(knot, tilts))
    field.set_points(knot.points(z))
    b = field.B()
    b /= np.linalg.norm(b, axis=1, keepdims=True)
    return np.concatenate([np.sum(b * normals, axis=1), np.sum(b * binormals, axis=1)])


def coil_field(knot):
    """The coils turned to meet B x t = 0 at the centres, and their field interpolated
    over a field period with Z >= 0."""
    tilts = least_squares(
        lambda x: misalignment(knot, x),
        np.zeros(2 * knot.coils),
        xtol=1e-14,
        ftol=1e-14,
        gtol=1e-14,
    ).x
    turned = coils(knot, tilts)
    r_min, r_max, z_max = knot.box
    nr, nphi, nz = knot.cells
    field = InterpolatedField(
        BiotSavart(turned),
        4,
        (r_min, r_max, nr),
        (0.0, 2.0 * np.pi / knot.q, nphi),
        (0.0, z_max, nz),
        True,
        nfp=knot.q,
        stellsym=True,
    )
    return turned, field


class FieldLines:
    """Field lines of the coil field, by RK4 in the toroidal angle."""

    def __init__(self, field, period):
        self.field, self.period = field, period

    def b(self, r, phi, z):
        """Cylindrical components of B at the points (r, phi, z)."""
        cos, sin = np.cos(phi), np.sin(phi)
        self.field.set_points(np.ascontiguousarray(np.stack([r * cos, r * sin, z], -1)))
        b = np.asarray(self.field.B())
        return b[:, 0] * cos + b[:, 1] * sin, -b[:, 0] * sin + b[:, 1] * cos, b[:, 2]

    def trace(self, r0, z0, planes, periods, steps):
        """R and Z of the line through (r0, 0, z0) at phi = k period / planes, k = 0,
        ..., periods * planes."""
        r, z, phi = np.atleast_1d(float(r0)), np.atleast_1d(float(z0)), 0.0
        out = [(r[0], z[0])]
        h = self.period / planes / steps

        def slope(r, phi, z):
            b_r, b_phi, b_z = self.b(r, np.full_like(r, phi), z)
            return r * b_r / b_phi, r * b_z / b_phi

        for _ in range(periods * planes):
            for _ in range(steps):
                k1 = slope(r, phi, z)
                k2 = slope(r + 0.5 * h * k1[0], phi + 0.5 * h, z + 0.5 * h * k1[1])
                k3 = slope(r + 0.5 * h * k2[0], phi + 0.5 * h, z + 0.5 * h * k2[1])
                k4 = slope(r + h * k3[0], phi + h, z + h * k3[1])
                r = r + h / 6.0 * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0])
                z = z + h / 6.0 * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1])
                phi += h
            out.append((r[0], z[0]))
        out = np.asarray(out)
        return out[:, 0], out[:, 1]


def magnetic_axis(knot, lines):
    """R of the axis on the outboard midplane of phi = 0, its crossings of the planes of
    a period, and the rotational transform on it per toroidal turn."""
    steps = knot.steps
    r_axis = brentq(
        lambda r: lines.trace(r, 0.0, 2, 1, 80 * steps)[1][1], *knot.axis, xtol=1e-12
    )
    r, z = lines.trace(r_axis, 0.0, knot.planes, 1, 8 * steps)
    axis = np.stack([r[: knot.planes], z[: knot.planes]], -1)
    periods = 150
    r, z = lines.trace(r_axis + 1e-3, 0.0, 1, periods, 240 * steps)
    angle = np.unwrap(np.arctan2(z, r - r_axis))
    iota = (angle[-1] - angle[0]) / (periods * knot.period)
    return r_axis, axis, iota


def flux_surface(knot, lines, r_axis, periods=120, harmonics=10):
    """R and Z of the surface traced from offset outboard of the axis on a uniform grid
    of its straight-field-line angle, its iota and its misfit."""
    planes = knot.planes
    r, z = lines.trace(r_axis + knot.offset, 0.0, planes, periods, 4 * knot.steps)
    plane = np.arange(len(r)) % planes
    phi = np.arange(len(r)) * knot.period / planes

    def fit(iota, harmonics):
        theta = iota * phi
        coef = np.empty((2, planes, 2 * harmonics - 1))
        worst = 0.0
        for k in range(planes):
            sel = plane == k
            basis = np.column_stack(
                [np.ones(sel.sum())]
                + [
                    f(m * theta[sel])
                    for m in range(1, harmonics)
                    for f in (np.cos, np.sin)
                ]
            )
            for i, target in enumerate((r[sel], z[sel])):
                coef[i, k], *_ = np.linalg.lstsq(basis, target, rcond=None)
                worst = max(worst, np.abs(basis @ coef[i, k] - target).max())
        return coef, worst

    # first estimate from the angle about the axis in the planes a period apart, then
    # the best of a scan around it, refined
    first = plane == 0
    angle = np.unwrap(np.arctan2(z[first], r[first] - r_axis))
    iota0 = (angle[-1] - angle[0]) / phi[first][-1]
    scan = iota0 + np.linspace(-0.02, 0.02, 41)
    best = int(np.argmin([fit(x, 4)[1] for x in scan]))
    step = scan[1] - scan[0]
    iota = minimize_scalar(
        lambda x: fit(x, 4)[1],
        bounds=(scan[best] - step, scan[best] + step),
        method="bounded",
        options={"xatol": 1e-10},
    ).x
    coef, worst = fit(iota, harmonics)
    theta = 2.0 * np.pi * np.arange(96) / 96
    basis = np.column_stack(
        [np.ones_like(theta)]
        + [f(m * theta) for m in range(1, harmonics) for f in (np.cos, np.sin)]
    )
    return coef[0] @ basis.T, coef[1] @ basis.T, iota, worst


def toroidal_flux(lines, rb, zb, axis, planes):
    """The toroidal flux through the cross-section of the surface in each plane, from
    B_phi on a fan of triangles from the axis."""
    flux = np.empty(len(planes))
    for k, phi in enumerate(planes):
        p1 = np.stack([rb[k], zb[k]], -1)
        p2 = np.roll(p1, -1, axis=0)
        d1, d2 = p1 - axis[k], p2 - axis[k]
        area = 0.5 * (d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0])
        area *= np.sign(area.sum())
        mid = np.concatenate([axis[k] + 0.5 * d1, axis[k] + 0.5 * d2, 0.5 * (p1 + p2)])
        b_phi = lines.b(mid[:, 0], np.full(len(mid), phi), mid[:, 1])[1]
        flux[k] = np.sum(area * b_phi.reshape(3, -1).mean(axis=0))
    return flux


def interior_input(knot, rb, zb, axis, flux, planes):
    """The fixed-boundary input of the interior, without pressure or toroidal
    current."""
    mpol, ntor = knot.mpol, knot.ntor
    angle = planes * knot.q / knot.p
    theta = 2.0 * np.pi * np.arange(rb.shape[1]) / rb.shape[1]
    return vmecpp.VmecInput(
        nfp=knot.q,
        toroidal_transits=knot.p,
        lasym=False,
        mpol=mpol,
        ntor=ntor,
        nzeta=2 * ntor + 8,
        ns_array=np.array(NS),
        ftol_array=np.array([1e-10, 1e-12]),
        niter_array=np.array([30000, 60000]),
        phiedge=float(abs(flux.mean())),
        ncurr=1,
        pcurr_type="power_series",
        ac=np.array([0.0]),
        curtor=0.0,
        pmass_type="power_series",
        am=np.array([1.0, -1.0]),
        pres_scale=0.0,
        gamma=0.0,
        rbc=fourier(rb, angle, theta, mpol, ntor, 0),
        zbs=fourier(zb, angle, theta, mpol, ntor, 1),
        # VMEC++ takes the axis as R = sum raxis_c cos(n phi), Z = -sum zaxis_s sin(n phi)
        raxis_c=fourier(axis[:, :1], angle, np.zeros(1), 1, ntor, 0)[0, ntor:],
        zaxis_s=fourier(axis[:, 1:], angle, np.zeros(1), 1, ntor, 1)[0, ntor:],
    )


def knotatron(name="trefoil"):
    """The smallest coil distance, the traced axis and its iota, the traced surface and
    the fixed-boundary input of the interior of the knot name."""
    knot = KNOTS[name]
    turned, field = coil_field(knot)
    points = [c.curve.gamma() for c in turned]
    spacing = min(
        np.linalg.norm(points[i][:, None] - points[j][None], axis=-1).min()
        for i in range(knot.coils)
        for j in range(i + 1, knot.coils)
    )
    lines = FieldLines(field, knot.period)
    r_axis, axis, iota_axis = magnetic_axis(knot, lines)
    planes = np.arange(knot.planes) * knot.period / knot.planes
    rb, zb, iota_surface, misfit = flux_surface(knot, lines, r_axis)
    flux = toroidal_flux(lines, rb, zb, axis, planes)
    return {
        "spacing": spacing,
        "r_axis": r_axis,
        "axis": axis,
        "iota_axis": iota_axis,
        "planes": planes,
        "iota_surface": iota_surface,
        "misfit": misfit,
        "flux": flux,
        "input": interior_input(knot, rb, zb, axis, flux, planes),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--knot", choices=list(KNOTS), default="trefoil")
    name = parser.parse_args().knot
    knot = KNOTS[name]
    start = time.time()
    k = knotatron(name)
    flux = k["flux"]
    print(
        f"{name}: {knot.coils} coils at least {k['spacing']:.3f} m apart; magnetic axis "
        f"through R = {k['r_axis']:.6f} m on the midplane of phi = 0, iota "
        f"{abs(k['iota_axis']):.5f} per toroidal turn next to it; surface traced from "
        f"{knot.offset * 100:.0f} cm outboard of the axis with iota "
        f"{abs(k['iota_surface']):.5f}, fitted within {k['misfit']:.1e} m, toroidal "
        f"flux {abs(flux.mean()):.5f} Wb, the same through every plane to "
        f"{np.ptp(flux) / abs(flux.mean()):.1e} ({time.time() - start:.0f} s)"
    )
    for p0 in (0.0, *knot.pressures):
        wout = vmecpp.run(
            k["input"].model_copy(update={"pres_scale": p0}),
            max_threads=4,
            verbose=False,
        ).wout
        shift = np.hypot(*(axis_curve(wout, k["planes"]) - k["axis"]).T).max()
        iota = np.asarray(wout.iotaf)
        print(
            f"p0 = {p0:5.0f} Pa, beta {wout.betatotal:.1e}: {wout.itfsq} iterations, "
            f"magnetic axis within {shift:.1e} m of the traced one, iota "
            f"{abs(iota[0]):.5f} on the axis and {abs(iota[-1]):.5f} on the boundary"
        )


if __name__ == "__main__":
    main()
