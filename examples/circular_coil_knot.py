# SPDX-FileCopyrightText: 2026-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Knotted stellarators wound from identical circular coils, searched for a quasi-
isodynamic field and solved as VMEC++ domains over several transits.

Every coil is the same planar circle with the same current. A design is the curve the
coils are centred on, their places along it and the law by which their axes tilt from
its tangent. In the vacuum field of the coils the magnetic scalar potential is the
Boozer toroidal angle, so a field line traced in steps of the potential gives the field
strength in Boozer coordinates and the bounce integrals of the effective ripple of
Nemov et al., Phys. Plasmas 6, 4622 (1999), without an equilibrium. ``search`` moves a
design by least squares on those integrals and on residuals that vanish where every
field line has one well between the highest and the lowest field strength of its
surface. ``star`` winds a knot whose field periods are plane curves, on which the
radial drift along a field line is odd about the middle of a period. The interior of a
traced flux surface is solved as in ``knotted_stellarator.py``, and its wout gives the
effective ripple again.
"""

import argparse
import dataclasses
import time
import types
from multiprocessing import Pool

import numpy as np
from island_interior import axis_curve
from knotted_stellarator import (
    FieldLines,
    Knot,
    flux_surface,
    interior_input,
    toroidal_flux,
)
from scipy.optimize import brentq, least_squares
from scipy.special import ellipe, ellipk
from simsopt.field import BiotSavart, coils_via_symmetries

import vmecpp

MU0 = 4.0e-7 * np.pi
# of the search, in units of the coil radius of its start: the start of the surface
# outboard of the axis and the least distances between coils and from coil to surface
OFFSET, CLEARANCE, ROOM = 0.225, 0.18, 0.45
MIRROR = 0.2  # largest mirror ratio on the surface
IOTA = 0.08  # smallest rotational transform per toroidal turn
SHAPE = 0.1  # weight of the residuals of the wells against those of the drift
PERIODS = 30  # field periods of the line of the search
PENALTIES = 2.0  # largest sum of squares of the penalties of a start drawn at random


class Loops:
    """The field of identical circular coils with one current, in closed form."""

    def __init__(self, centres, axes, radius, current):
        self.centres = np.asarray(centres, dtype=float)
        self.axes = np.asarray(axes, dtype=float)
        self.radius, self.current = float(radius), float(current)
        self._points = np.zeros((0, 3))

    def _cylindrical(self, points):
        """The offsets of the points from each coil: along its axis, across it, and the
        square of the distance across."""
        d = np.asarray(points, dtype=float)[:, None, :] - self.centres[None]
        z = np.einsum("pnk,nk->pn", d, self.axes)
        across = d - z[..., None] * self.axes[None]
        return z, across, np.einsum("pnk,pnk->pn", across, across)

    def b(self, points):
        """B at the points, [point, 3]."""
        a = self.radius
        z, across, rho2 = self._cylindrical(points)
        rho = np.sqrt(rho2)
        far = (a + rho) ** 2 + z**2
        near = (a - rho) ** 2 + z**2
        m = 4.0 * a * rho / far
        k, e = ellipk(m), ellipe(m)
        scale = MU0 * self.current / (2.0 * np.pi * np.sqrt(far))
        along = scale * (k + (a * a - rho2 - z**2) / near * e)
        # the radial part over rho, which is finite on the axis of a coil
        on_axis = rho < 1.0e-6 * a
        radial = np.where(
            on_axis,
            0.75 * MU0 * self.current * a * a * z / (a * a + z**2) ** 2.5,
            scale
            * z
            / np.where(on_axis, 1.0, rho2)
            * (-k + (a * a + rho2 + z**2) / near * e),
        )
        return (along[..., None] * self.axes[None] + radial[..., None] * across).sum(
            axis=1
        )

    def distance(self, points):
        """The distance from each point to each coil, [point, coil]."""
        z, _, rho2 = self._cylindrical(points)
        return np.sqrt((np.sqrt(rho2) - self.radius) ** 2 + z**2)

    def clearance(self, samples=48):
        """The smallest distance between two coils."""
        angle = 2.0 * np.pi * np.arange(samples) / samples
        first = np.cross(self.axes, np.roll(self.axes, 1, axis=-1))
        first /= np.linalg.norm(first, axis=-1, keepdims=True)
        second = np.cross(self.axes, first)
        circles = self.centres[:, None] + self.radius * (
            np.cos(angle)[None, :, None] * first[:, None]
            + np.sin(angle)[None, :, None] * second[:, None]
        )
        gaps = self.distance(circles.reshape(-1, 3)).reshape(len(circles), samples, -1)
        gaps[np.arange(len(circles)), :, np.arange(len(circles))] = np.inf
        return float(gaps.min())

    # the interface of the fields that knotted_stellarator.FieldLines traces
    def set_points(self, points):
        self._points = np.asarray(points, dtype=float)

    def B(self):
        return self.b(self._points)


@dataclasses.dataclass(frozen=True)
class Design:
    """Identical circular coils along the curve R = r0 + sum r[m] cos(m q z), Z = -sum
    zs[m] sin(m q z), phi = p z + sum phis[m] sin(m q z) in metres, with m from 1.

    Coil k sits at z = z0 + sum places[m] sin(m q z0), z0 = 2 pi k / coils. Its axis is
    the tangent tilted by sum outward[m] sin(m q z) toward the direction away from the
    circle R = r0, Z = 0 and by sum sideways[m] cos(m q z), m from 0, toward their cross
    product, which keeps the coils stellarator symmetric.
    """

    p: int  # toroidal transits, after which the curve closes
    q: int  # field periods
    per_period: int  # coils of a field period
    radius: float  # of every coil
    r: tuple[float, ...] = (1.0,)
    zs: tuple[float, ...] = (1.0,)
    phis: tuple[float, ...] = ()
    places: tuple[float, ...] = ()
    outward: tuple[float, ...] = ()
    sideways: tuple[float, ...] = ()
    r0: float = 2.0
    current: float = 1.0e6

    @property
    def coils(self):
        return self.q * self.per_period

    @property
    def period(self):
        """Toroidal angle of a field period."""
        return 2.0 * np.pi * self.p / self.q

    @property
    def period_potential(self):
        """The magnetic scalar potential that a field line gains in a field period."""
        return MU0 * self.per_period * self.current

    @property
    def vector(self):
        """The parameters that :func:`search` moves."""
        return np.array(
            [
                *self.r,
                *self.zs,
                *self.phis,
                self.radius,
                *self.places,
                *self.outward,
                *self.sideways,
            ]
        )

    def moved(self, x):
        """The design with the parameters x."""
        x = [float(v) for v in x]
        out, at = {}, 0
        for name in ("r", "zs", "phis", "radius", "places", "outward", "sideways"):
            if name == "radius":
                out[name] = x[at]
                at += 1
            else:
                count = len(getattr(self, name))
                out[name] = tuple(x[at : at + count])
                at += count
        return dataclasses.replace(self, **out)

    def _series(self, function, amplitudes, z, first=1):
        """The sum of amplitudes[m] function(m q z), m from first."""
        orders = np.arange(first, first + len(amplitudes))
        return function(self.q * z[..., None] * orders) @ np.asarray(amplitudes, float)

    def points(self, z):
        """Cartesian points of the curve at the parameter z."""
        z = np.asarray(z, dtype=float)
        radius = self.r0 + self._series(np.cos, self.r, z)
        height = -self._series(np.sin, self.zs, z)
        phi = self.p * z + self._series(np.sin, self.phis, z)
        return np.stack([radius * np.cos(phi), radius * np.sin(phi), height], -1)

    def frame(self, z, h=1.0e-5):
        """The tangent of the curve, the unit vector away from the circle R = r0, Z = 0
        across it, and their cross product."""
        z = np.asarray(z, dtype=float)
        here = self.points(z)
        tangent = (self.points(z + h) - self.points(z - h)) / (2.0 * h)
        tangent /= np.linalg.norm(tangent, axis=-1, keepdims=True)
        angle = np.arctan2(here[..., 1], here[..., 0])
        out = here - self.r0 * np.stack(
            [np.cos(angle), np.sin(angle), np.zeros_like(angle)], -1
        )
        out -= np.sum(out * tangent, axis=-1, keepdims=True) * tangent
        out /= np.linalg.norm(out, axis=-1, keepdims=True)
        return tangent, out, np.cross(tangent, out)

    def loops(self):
        """The coils of the design."""
        z = 2.0 * np.pi * np.arange(self.coils) / self.coils
        z = z + self._series(np.sin, self.places, z)
        tangent, out, side = self.frame(z)
        toward_out = self._series(np.sin, self.outward, z)
        toward_side = self._series(np.cos, self.sideways, z, first=0)
        axes = tangent + toward_out[:, None] * out + toward_side[:, None] * side
        axes /= np.linalg.norm(axes, axis=-1, keepdims=True)
        return Loops(self.points(z), axes, self.radius, self.current)

    def axis(self, field):
        """The magnetic axis of the coils nearest their curve, as :func:`magnetic_axis`
        returns it."""
        return magnetic_axis(
            field,
            self.period_potential,
            4 * self.per_period,
            float(np.hypot(*self.points(0.0)[:2])),
            0.6 * self.radius,
        )


def star(p, q, per_period, radius, inner, height, slow, mirror, order=6):
    """A (p, q) knot whose field periods are plane curves, wound with coils that are
    densest where it passes from one plane to the next.

    The curve is a wave on the faces of the star prism {q / p} about the cylinder of
    radius inner. With v from -pi at one edge of a face to pi at the next,

        y = half (v + slow sin v) / pi,  Z = height sin v,
        R = hypot(inner, y),  phi = phi_c + atan(y / inner),

    half = inner tan(pi p / q) being half the width of a face and phi_c its azimuth.
    The harmonics of R, Z and phi - p z in q z, with v = q z - pi, are cut at order.
    The coils lie 1 + mirror cos(2 pi s) dense along the length s of a period from an
    edge, in units of that length.
    """
    samples = 4096
    z = 2.0 * np.pi / q * np.arange(samples) / samples
    v = q * z - np.pi
    y = inner * np.tan(np.pi * p / q) * (v + slow * np.sin(v)) / np.pi
    r = np.hypot(inner, y)
    turn = np.pi * p / q + np.arctan2(y, inner) - p * z
    angle = q * z[:, None] * np.arange(1, order + 1)
    cos, sin = np.cos(angle), np.sin(angle)
    curve = Design(
        p=p,
        q=q,
        per_period=per_period,
        radius=radius,
        r0=float(r.mean()),
        r=tuple(2.0 * np.mean((r - r.mean())[:, None] * cos, axis=0)),
        zs=tuple(-2.0 * np.mean(height * np.sin(v)[:, None] * sin, axis=0)),
        phis=tuple(2.0 * np.mean(turn[:, None] * sin, axis=0)),
    )
    # the share of the coils of a period up to z, from their density along its length
    ends = 2.0 * np.pi / q * np.arange(samples + 1) / samples
    steps = np.linalg.norm(np.diff(curve.points(ends), axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(steps)]) / steps.sum()
    share = s + mirror * np.sin(2.0 * np.pi * s) / (2.0 * np.pi)
    shift = np.interp(np.arange(samples) / samples, share, ends) - z
    places = tuple(2.0 * np.mean(shift[:, None] * sin, axis=0))
    return dataclasses.replace(curve, places=places)


# the start of the search of the trefoil: the curve of the knotatron of Hudson et al.
# with twice its coils, spread evenly along it
STARTS = {
    "trefoil": Design(
        p=2,
        q=3,
        per_period=24,
        radius=2.0 / 3.0,
        r=(1.0, 0.0, 0.0),
        zs=(1.0, 0.0, 0.0),
        phis=(0.0, 0.0),
        places=(-0.107, 0.0, 0.0),
        outward=(0.0, 0.0),
        sideways=(0.0, 0.0, 0.0),
    ),
}
DESIGNS = {
    "knotatron": Design(p=2, q=3, per_period=24, radius=2.0 / 3.0),
    "trefoil": Design(
        p=2,
        q=3,
        per_period=24,
        radius=0.7879006051522417,
        r=(0.9837925252867619, 0.003902167267083462, 0.001892000789529573),
        zs=(0.9842070992884725, -0.006681125120768963, -0.006300161313929336),
        phis=(-0.043020905169186316, 0.019677780893734016),
        places=(-0.12078657577535272, 0.008553768770652356, 0.002388106854324222),
        outward=(0.03434891847148264, -0.19332167736230776),
        sideways=(-0.02054551837863322, 0.05169127027447562, 0.10858923434013387),
    ),
    # plane periods on the seven-pointed star prism
    "star": star(2, 7, 22, 0.35, 1.87, 0.85, 0.15, 0.2),
}
# the start of the outermost surface of each design outboard of its magnetic axis, in
# metres, mpol, ntor and the planes per period of its interior, and the field periods
# of the lines of its measures. The lines of the star close after 45 field periods.
SURFACES = {
    "knotatron": (0.15, 8, 24, 72, 40),
    "trefoil": (0.15, 8, 24, 72, 40),
    "star": (0.05, 10, 28, 96, 45),
}


def trace(field, start, potential, steps):
    """The field line through start in steps uniform steps of the magnetic scalar
    potential, by RK4: its points, [steps + 1, 3], and the field strength at them."""
    h = potential / steps
    x = np.asarray(start, dtype=float).reshape(1, 3)
    points = np.empty((steps + 1, 3))
    strength = np.empty(steps + 1)

    def slope(x):
        b = field.b(x)
        b2 = float(b[0] @ b[0])
        return b / b2, np.sqrt(b2)

    for i in range(steps):
        k1, strength[i] = slope(x)
        points[i] = x[0]
        k2, _ = slope(x + 0.5 * h * k1)
        k3, _ = slope(x + 0.5 * h * k2)
        k4, _ = slope(x + h * k3)
        x = x + h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    points[steps] = x[0]
    strength[steps] = slope(x)[1]
    return points, strength


def magnetic_axis(field, potential, steps, middle, width):
    """R of the magnetic axis on the line phi = 0, Z = 0, which half a field period
    takes to the line of the next half period, its points over a period and the field
    strength at them.

    potential is the magnetic scalar potential of a field period, and the axis is the
    one nearest R = middle within width of it.
    """
    half = 0.5 * potential

    def miss(r):
        return trace(field, [r, 0.0, 0.0], half, steps // 2)[0][-1, 2]

    grid = middle + width * np.linspace(-1.0, 1.0, 9)
    # a line of the grid may meet a coil
    with np.errstate(all="ignore"):
        values = np.array([miss(r) for r in grid])
    crossings = np.flatnonzero(values[:-1] * values[1:] < 0.0)
    if crossings.size == 0:
        message = "no magnetic axis within the width"
        raise ValueError(message)
    k = crossings[
        np.argmin(np.abs(0.5 * (grid[crossings] + grid[crossings + 1]) - middle))
    ]
    r_axis = brentq(miss, grid[k], grid[k + 1], xtol=1.0e-13)
    points, strength = trace(field, [r_axis, 0.0, 0.0], 2.0 * half, steps)
    return r_axis, points, strength


def drift_line(field, start, normal, potential, steps, spacing=1.0e-5):
    """The field line through start with the gradient of the flux surface label carried
    along it from normal: its points at steps uniform steps of the potential, the
    field strength, the size of that gradient and the radial drift (b x grad B) . grad
    psi / B at them."""
    h = potential / steps
    x = np.asarray(start, dtype=float)
    g = np.asarray(normal, dtype=float)
    stencil = np.concatenate(
        [np.zeros((1, 3)), spacing * np.eye(3), -spacing * np.eye(3)]
    )
    strength, size, drift = (np.empty(steps + 1) for _ in range(3))
    points = np.empty((steps + 1, 3))

    def slopes(x, g):
        b = field.b(x[None] + stencil)
        b2 = np.einsum("ij,ij->i", b, b)
        v = b / b2[:, None]
        # the label is constant along the line, so its gradient obeys g' = -(grad v)^T g
        along = v @ g
        dg = -(along[1:4] - along[4:7]) / (2.0 * spacing)
        size_b = np.sqrt(b2)
        grad_b = (size_b[1:4] - size_b[4:7]) / (2.0 * spacing)
        return v[0], dg, size_b[0], b[0], grad_b

    for i in range(steps + 1):
        v1, g1, size_b, b, grad_b = slopes(x, g)
        g = g - (g @ b) / (b @ b) * b
        strength[i], size[i] = size_b, np.linalg.norm(g)
        drift[i] = np.cross(b, grad_b) @ g / size_b**2
        points[i] = x
        if i == steps:
            break
        v2, g2, *_ = slopes(x + 0.5 * h * v1, g + 0.5 * h * g1)
        v3, g3, *_ = slopes(x + 0.5 * h * v2, g + 0.5 * h * g2)
        v4, g4, *_ = slopes(x + h * v3, g + h * g3)
        x = x + h / 6.0 * (v1 + 2.0 * v2 + 2.0 * v3 + v4)
        g = g + h / 6.0 * (g1 + 2.0 * g2 + 2.0 * g3 + g4)
    return points, strength, size, drift


def rotational_transform(field, r_axis, at, period):
    """The rotational transform per toroidal turn of a field line, from its points at
    the starts of the field periods, of the toroidal angle period."""
    direction = field.b(np.array([[r_axis, 0.0, 0.0]]))[0]
    direction /= np.linalg.norm(direction)
    # the starts of the periods, turned back into the first
    turn = -np.sign(direction[1]) * period * np.arange(len(at))
    x = at[:, 0] * np.cos(turn) - at[:, 1] * np.sin(turn)
    y = at[:, 0] * np.sin(turn) + at[:, 1] * np.cos(turn)
    # across the axis: e_R and e_R x t
    across = np.cross([1.0, 0.0, 0.0], direction)
    angle = np.unwrap(np.arctan2(y * across[1] + at[:, 2] * across[2], x - r_axis))
    return np.polyfit(np.arange(len(at)), angle, 1)[0] / period


def bounce_integrals(strength, drift, weight, levels):
    """The bounce integrals of the wells of a field line.

    For each level b' of B / B0, B0 the mean strength, and each well j of that level
    that the line holds whole,

        H[j] = integral ds / B sqrt(b' - B / B0) (4 B0 / B - 1 / b') drift
        I[j] = integral ds / B sqrt(1 - B / (B0 b'))

    with weight the ds / B of a step at each sample. B / B0 is linear between the
    samples and the other factors are their mean, so the integrals are continuously
    differentiable in the samples. Returns the levels and, for each, the first and the
    last step of its wells, H and I.
    """
    eta = strength * np.sum(weight) / np.sum(weight * strength)
    grid = np.linspace(eta.min(), eta.max(), levels + 2)[1:-1]
    step_weight = 0.5 * (weight[:-1] + weight[1:])
    wells = []
    for level in grid:
        depth = level - eta
        below = np.maximum(depth, 0.0)
        change = np.diff(depth)
        flat = np.abs(change) < 1.0e-14
        # the mean of sqrt(max(b' - B / B0, 0)) over each step
        root = np.where(
            flat,
            np.sqrt(below[:-1]),
            2.0 / 3.0 * np.diff(below**1.5) / np.where(flat, 1.0, change),
        )
        wet = root > 0.0
        starts = np.flatnonzero(wet[1:] & ~wet[:-1]) + 1
        stops = np.flatnonzero(~wet[1:] & wet[:-1]) + 1
        # the wells that the ends of the line cut are left out
        if wet[0]:
            root[: stops[0] if stops.size else None] = 0.0
            stops = stops[1:]
        if wet[-1] and starts.size:
            root[starts[-1] :] = 0.0
            starts = starts[:-1]
        if starts.size == 0:
            wells.append((starts, stops, np.zeros(0), np.zeros(0)))
            continue
        factor = weight * (4.0 / eta - 1.0 / level) * drift
        h = np.add.reduceat(root * 0.5 * (factor[:-1] + factor[1:]), starts)
        i = np.add.reduceat(root * step_weight, starts) / np.sqrt(level)
        wells.append((starts, stops, h, i))
    return grid, wells


def _ripple_scale(size, weight, major_radius, grid):
    step_weight = 0.5 * (weight[:-1] + weight[1:])
    step_size = 0.5 * (weight[:-1] * size[:-1] + weight[1:] * size[1:])
    return (
        np.pi
        * major_radius**2
        / (8.0 * np.sqrt(2.0))
        * np.sum(step_weight)
        / np.sum(step_size) ** 2
        * (grid[1] - grid[0])
    )


def effective_ripple(strength, size, drift, weight, major_radius, levels=120):
    """The effective ripple from the samples of a field line.

    It is (pi R0^2 / (8 sqrt 2) integral ds / B / (integral ds |grad psi| / B)^2
    integral db' sum H[j]^2 / (b'^2 I[j]))^(2/3), with H and I of
    :func:`bounce_integrals`, from the field strength, the size of the gradient of the
    flux surface label, the radial drift and the ds / B of a step at the samples.
    """
    grid, wells = bounce_integrals(strength, drift, weight, levels)
    total = sum(
        float(np.sum(h[i > 0.0] ** 2 / i[i > 0.0])) / level**2
        for level, (_, _, h, i) in zip(grid, wells, strict=True)
    )
    return float((_ripple_scale(size, weight, major_radius, grid) * total) ** (2 / 3))


def drift_residuals(strength, size, drift, weight, major_radius, steps, levels=40):
    """Residuals whose squares sum to the effective ripple to the power 3/2, from the
    data of :func:`effective_ripple` with steps steps to a field period.

    The line is cut into field periods where the strength, averaged over the periods,
    is highest. residual[k, i] is H[j] / (b' sqrt(I[j])) of :func:`bounce_integrals`
    for the well j of level k in period i. Where a period holds several wells of a
    level it is the root of the sum of their squares, with the sign of the largest.
    """
    periods = (len(strength) - 1) // steps
    top = int(np.argmax(strength[: periods * steps].reshape(periods, steps).mean(0)))
    count = periods - 1
    grid, wells = bounce_integrals(strength, drift, weight, levels)
    square = np.zeros((levels, count))
    sign = np.ones((levels, count))
    for k, (starts, stops, h, i) in enumerate(wells):
        period = ((starts + stops) // 2 - top) // steps
        good = (i > 0.0) & (period >= 0) & (period < count)
        value = h[good] / (grid[k] * np.sqrt(i[good]))
        np.add.at(square[k], period[good], value**2)
        order = np.argsort(np.abs(value))
        sign[k, period[good][order]] = np.sign(value[order])
    scale = _ripple_scale(size, weight, major_radius, grid)
    return np.sqrt(scale) * sign * np.sqrt(square)


def well_lines(strength, steps):
    """The field strength of a field line on uniform steps of the Boozer toroidal angle,
    steps to a field period, as one row to a period, from where the strength averaged
    over the periods is highest, scaled to span 0 to 1."""
    periods = (len(strength) - 1) // steps
    top = int(np.argmax(strength[: periods * steps].reshape(periods, steps).mean(0)))
    rows = top + steps * np.arange(periods - 1)[:, None] + np.arange(steps + 1)[None, :]
    lines = strength[rows]
    return (lines - lines.min()) / (lines.max() - lines.min())


def well_residuals(strength, steps, levels=101):
    """Residuals that vanish for a quasi-isodynamic field, from the rows of
    :func:`well_lines` as field lines i of one surface.

        spread[i, j] = fraction of line i below level j - mean of that over the lines
        wells[i]     = half the variation of line i - (its maximum - its minimum)
        ends[i]      = 1 - mean of the two ends of line i
        lowest[i]    = minimum of line i

    The fraction below a level is that of the piecewise linear strength, so the
    residuals are continuous in it.
    """
    lines = well_lines(strength, steps)
    low = np.minimum(lines[:, :-1], lines[:, 1:])[..., None]
    high = np.maximum(lines[:, :-1], lines[:, 1:])[..., None]
    level = (np.arange(levels) + 0.5) / levels
    below = np.clip((level - low) / np.maximum(high - low, 1.0e-300), 0.0, 1.0)
    below = np.where(high > low, below, (low < level).astype(float)).mean(axis=1)
    return {
        "spread": below - below.mean(axis=0),
        "wells": 0.5 * np.abs(np.diff(lines, axis=1)).sum(axis=1)
        - (lines.max(axis=1) - lines.min(axis=1)),
        "ends": 1.0 - 0.5 * (lines[:, 0] + lines[:, -1]),
        "lowest": lines.min(axis=1),
    }


def poloidal_closure(strength, steps, turns, angles=64):
    """The share of the range of the field strength whose contours close poloidally on
    the surface of a field line, from its samples on uniform steps of the Boozer
    toroidal angle, steps to a field period in which it makes turns poloidal turns.

    The line is straight in Boozer angles, so the samples of a toroidal angle lie at
    known poloidal angles, from which the strength is interpolated to a uniform grid.
    The contours between the highest minimum and the lowest maximum over the toroidal
    angle close poloidally. The share is 1 for a quasi-isodynamic field.
    """
    periods = (len(strength) - 1) // steps
    values = strength[: periods * steps].reshape(periods, steps)
    theta = turns * (np.arange(periods)[:, None] + np.arange(steps)[None, :] / steps)
    grid = np.arange(angles) / angles
    surface = np.stack(
        [
            np.interp(grid, theta[:, j] % 1.0, values[:, j], period=1.0)
            for j in range(steps)
        ],
        axis=1,
    )
    lowest_ridge, highest_valley = surface.max(axis=1).min(), surface.min(axis=1).max()
    return float((lowest_ridge - highest_valley) / (surface.max() - surface.min()))


def measures(design, offset, periods=40, levels=120):
    """The axis of a design and the field line from offset outboard of it over periods
    field periods: R of the axis, its length, the mirror ratio on it and on the line,
    the rotational transform per toroidal turn, the smallest distances between coils
    and from the line to a coil, the effective ripple with the length of the axis over
    2 pi as major radius, the share of :func:`poloidal_closure`, the sum of the mean
    squares of :func:`well_residuals`, and the data of the line."""
    field = design.loops()
    steps = 4 * design.per_period
    r_axis, axis, axis_strength = design.axis(field)
    length = float(np.linalg.norm(np.diff(axis, axis=0), axis=1).sum() * design.q)
    potential = periods * design.period_potential
    # the surface is normal to e_R where it crosses the line phi = 0, Z = 0
    points, strength, size, drift = drift_line(
        field, [r_axis + offset, 0.0, 0.0], [1.0, 0.0, 0.0], potential, periods * steps
    )
    if not np.all(np.isfinite(strength)):
        message = "the field line meets a coil"
        raise ValueError(message)
    weight = potential / (periods * steps) / strength**2
    major = length / (2.0 * np.pi)

    def mirror(b):
        return float((b.max() - b.min()) / (b.max() + b.min()))

    iota = float(rotational_transform(field, r_axis, points[::steps], design.period))
    return {
        "r_axis": r_axis,
        "length": length,
        "mirror_axis": mirror(axis_strength),
        "mirror": mirror(strength),
        "iota": iota,
        "clearance": field.clearance(),
        "room": float(field.distance(points[::4]).min()),
        "ripple": effective_ripple(strength, size, drift, weight, major, levels),
        "closure": poloidal_closure(
            strength, steps, iota * design.period / (2.0 * np.pi)
        ),
        "wells": float(
            sum(np.mean(v**2) for v in well_residuals(strength, steps).values())
        ),
        "line": (strength, size, drift, weight, major, steps),
    }


def search_residuals(item):
    """The residuals of :func:`search` for a design: those of the drift on its surface,
    those of its wells, each kind over the root of its number and all times the root of
    SHAPE, and what it lacks of the least distances, exceeds of the mirror ratio and
    lacks of the rotational transform, for a tuple of the design, the coil radius of the
    start and the field periods of the line."""
    design, scale, periods = item
    try:
        with np.errstate(invalid="ignore", divide="ignore"):
            found = measures(design, OFFSET * scale, periods)
    except ValueError:
        return None
    line = found.pop("line")
    penalties = [
        max(0.0, CLEARANCE - found["clearance"] / scale) / CLEARANCE,
        max(0.0, ROOM - found["room"] / scale) / ROOM,
        max(0.0, found["mirror"] - MIRROR) / MIRROR,
        max(0.0, IOTA - abs(found["iota"])) / IOTA,
    ]
    wells = well_residuals(line[0], line[-1])
    return np.concatenate(
        [
            drift_residuals(*line).ravel(),
            *(np.sqrt(SHAPE / v.size) * v.ravel() for v in wells.values()),
            3.0 * np.array(penalties),
        ]
    )


def search_tables(base):
    """The scales and the lower and upper bounds of the parameters of a search about
    base, in the order of Design.vector."""
    radius, r1, turn = base.radius, base.r[0], base.p / base.q

    def harmonics(values, name):
        # the last value holds for the higher harmonics
        count = len(getattr(base, name))
        return np.array([values[min(m, len(values) - 1)] for m in range(count)])

    scale = np.concatenate(
        [
            r1 * harmonics((1.0, 0.2, 0.1), "r"),
            r1 * harmonics((1.0, 0.2, 0.1), "zs"),
            turn * harmonics((0.3, 0.15), "phis"),
            [0.4 * radius],
            harmonics((0.15, 0.15, 0.1), "places") / base.q,
            harmonics((0.2,), "outward"),
            harmonics((0.2,), "sideways"),
        ]
    )
    high = np.concatenate(
        [
            r1 * harmonics((2.0, 0.6, 0.25), "r"),
            r1 * harmonics((2.0, 0.6, 0.25), "zs"),
            turn * harmonics((0.7, 0.35), "phis"),
            [1.5 * radius],
            harmonics((0.42, 0.18, 0.12), "places") / base.q,
            harmonics((0.5,), "outward"),
            harmonics((0.5,), "sideways"),
        ]
    )
    low = -high
    first = [0, len(base.r), len(base.r) + len(base.zs) + len(base.phis)]
    low[first] = 0.3 * r1, 0.3 * r1, 0.6 * radius
    return scale, low, high


def search(start, base=None, evaluations=60, periods=PERIODS, pool=None):
    """The design that least squares on :func:`search_residuals` reaches from start,
    with a finite-difference Jacobian, or None where start has no magnetic axis.

    base, start by default, gives the scales and the bounds of the parameters and the
    coil radius that the lengths of the search are in. The Jacobian is taken on the
    processes of pool where one is given.
    """
    base = base or start
    radius = base.radius
    scale, low, high = search_tables(base)
    x = np.clip(start.vector, low, high)
    first = search_residuals((base.moved(x), radius, periods))
    if first is None:
        return None
    # a design without a magnetic axis costs more than any with one
    failed = np.full(first.size, 1.0e3 / np.sqrt(first.size))

    def residuals(x):
        value = search_residuals((base.moved(x), radius, periods))
        return failed if value is None else value

    def jacobian(x):
        value = residuals(x)
        steps = np.where(x + 2.0e-4 * scale > high, -2.0e-4, 2.0e-4) * scale
        moved = [
            (base.moved(x + steps[i] * np.eye(x.size)[i]), radius, periods)
            for i in range(x.size)
        ]
        columns = (pool.map if pool else map)(search_residuals, moved)
        out = np.zeros((value.size, x.size))
        for i, column in enumerate(columns):
            if column is not None:
                out[:, i] = (column - value) / steps[i]
        return out

    cost = np.inf
    for _ in range(4):
        solution = least_squares(
            residuals,
            x,
            jac=jacobian,
            bounds=(low, high),
            x_scale=scale,
            max_nfev=evaluations,
            xtol=1e-9,
            ftol=1e-9,
            gtol=1e-12,
        )
        x = np.clip(solution.x, low, high)
        if 2.0 * solution.cost > 0.98 * cost:
            break
        cost = 2.0 * solution.cost
    return base.moved(x)


def near_limits(item):
    """Whether the penalties of a design, with the coil radius of its base, stay within
    PENALTIES."""
    design, radius = item
    value = search_residuals((design, radius, PERIODS))
    return value is not None and float(np.sum(value[-4:] ** 2)) <= PENALTIES


def searched(item):
    """The design :func:`search` reaches from a start about a base, and its effective
    ripple on the surface of the search."""
    start, base = item
    found = search(start, base)
    if found is None:
        return None, np.inf
    try:
        return found, measures(found, OFFSET * base.radius)["ripple"]
    except ValueError:
        return None, np.inf


def starts_about(base, count, seed, pool):
    """base and count - 1 designs drawn about it within half the scales of its
    parameters, each :func:`near_limits`."""
    generator = np.random.default_rng(seed)
    scale, low, high = search_tables(base)
    starts = [base]
    while len(starts) < count:
        drawn = [
            base.moved(
                np.clip(
                    base.vector
                    + 0.5 * scale * generator.uniform(-1.0, 1.0, scale.size),
                    low,
                    high,
                )
            )
            for _ in range(count)
        ]
        kept = pool.map(near_limits, [(design, base.radius) for design in drawn])
        starts += [design for design, keep in zip(drawn, kept, strict=True) if keep]
    return starts[:count]


def best_of(base, count, seed, workers):
    """:func:`search` from the designs of :func:`starts_about`, on workers processes:
    the designs found with their effective ripples, the least first.

    A single search takes its Jacobian on the processes.
    """
    with Pool(workers) as pool:
        if count == 1:
            found = search(base, pool=pool)
            return [(found, measures(found, OFFSET * base.radius)["ripple"])]
        starts = starts_about(base, count, seed, pool)
        results = pool.map(searched, [(start, base) for start in starts], chunksize=1)
    return sorted(results, key=lambda result: result[1])


def interior(design, offset, mpol=8, ntor=24, planes=72):
    """The traced axis of a design and the surface from offset outboard of it, their
    rotational transform, and the fixed-boundary input of the interior of the surface
    over the transits of the curve."""
    field = design.loops()
    r_axis, _, _ = design.axis(field)
    knot = Knot(
        p=design.p,
        q=design.q,
        r0=design.r0,
        r_knot=design.r[0],
        coils=design.coils,
        radius=design.radius,
        planes=planes,
        offset=offset,
        mpol=mpol,
        ntor=ntor,
    )
    lines = FieldLines(field, design.period)
    r, z = lines.trace(r_axis, 0.0, planes, 1, 8 * knot.steps)
    axis = np.stack([r[:planes], z[:planes]], -1)
    rb, zb, iota_surface, misfit = flux_surface(knot, lines, r_axis)
    angles = np.arange(planes) * design.period / planes
    flux = toroidal_flux(lines, rb, zb, axis, angles)
    return {
        "axis": axis,
        "planes": angles,
        "iota_surface": iota_surface,
        "misfit": misfit,
        "input": interior_input(knot, rb, zb, axis, flux, angles),
    }


def wout_ripple(wout, js, periods=40, steps=96, levels=120):
    """The effective ripple of the half-grid surface js of a wout, from the field line
    through theta = 0 at phi = 0 over periods field periods, with the length of the
    magnetic axis over 2 pi as major radius."""
    transits = int(wout.toroidal_transits)
    period = 2.0 * np.pi * transits / int(wout.nfp)
    m = np.asarray(wout.xm_nyq, dtype=float)
    n = np.asarray(wout.xn_nyq, dtype=float) / transits
    b_theta_up = np.asarray(wout.bsupumnc)[:, js]
    b_phi_up = np.asarray(wout.bsupvmnc)[:, js]

    def slope(theta, phi):
        c = np.cos(m * theta - n * phi)
        return (b_theta_up @ c) / (b_phi_up @ c)

    count = periods * steps
    h = period / steps
    theta = np.zeros(count + 1)
    for k in range(count):
        phi = k * h
        k1 = slope(theta[k], phi)
        k2 = slope(theta[k] + 0.5 * h * k1, phi + 0.5 * h)
        k3 = slope(theta[k] + 0.5 * h * k2, phi + 0.5 * h)
        k4 = slope(theta[k] + h * k3, phi + h)
        theta[k + 1] = theta[k] + h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    phi = h * np.arange(count + 1)
    angle = m[:, None] * theta[None] - n[:, None] * phi[None]
    cos, sin = np.cos(angle), np.sin(angle)
    bmnc = np.asarray(wout.bmnc)[:, js]
    strength = bmnc @ cos
    jacobian = np.asarray(wout.gmnc)[:, js] @ cos
    # (B x grad B) . grad s / B^2 = (B_theta dB/dphi - B_phi dB/dtheta) / (sqrt(g) B^2)
    drift = (
        (np.asarray(wout.bsubumnc)[:, js] @ cos) * ((n * bmnc) @ sin)
        + (np.asarray(wout.bsubvmnc)[:, js] @ cos) * ((m * bmnc) @ sin)
    ) / (jacobian * strength**2)
    # |grad s| from R and Z between the two full-grid surfaces around js
    m = np.asarray(wout.xm, dtype=float)
    n = np.asarray(wout.xn, dtype=float) / transits
    angle = m[:, None] * theta[None] - n[:, None] * phi[None]
    cos_full, sin_full = np.cos(angle), np.sin(angle)
    rmnc = np.asarray(wout.rmnc)[:, js - 1 : js + 1].mean(axis=1)
    zmns = np.asarray(wout.zmns)[:, js - 1 : js + 1].mean(axis=1)
    r = rmnc @ cos_full
    r_theta, r_phi = -(m * rmnc) @ sin_full, (n * rmnc) @ sin_full
    z_theta, z_phi = (m * zmns) @ cos_full, -(n * zmns) @ cos_full
    size = np.sqrt(
        (r_theta**2 + z_theta**2) * (r_phi**2 + z_phi**2 + r**2)
        - (r_theta * r_phi + z_theta * z_phi) ** 2
    ) / np.abs(jacobian)
    weight = h / np.abs(b_phi_up @ cos)
    turn = 2.0 * np.pi * transits * np.arange(4097) / 4096
    axis = axis_curve(wout, turn)
    axis = np.stack(
        [axis[:, 0] * np.cos(turn), axis[:, 0] * np.sin(turn), axis[:, 1]], -1
    )
    major = np.linalg.norm(np.diff(axis, axis=0), axis=1).sum() / (2.0 * np.pi)
    return effective_ripple(strength, size, drift, weight, major, levels)


def w7x(offsets=(0.05, 0.1, 0.15), periods=40, steps=120):
    """The effective ripple, the share of :func:`poloidal_closure` and the sum of the
    mean squares of :func:`well_residuals` of the standard configuration of W7-X by the
    routines of the knots, from the coils that SIMSOPT ships, on the surfaces from
    offsets outboard of its axis in the bean plane, with 5.5 m as major radius."""
    try:  # SIMSOPT 1.11 and later
        from simsopt.configs import get_data  # noqa: PLC0415

        curves, currents = get_data("w7x")[:2]
    except ImportError:
        from simsopt.configs import get_w7x_data  # noqa: PLC0415

        curves, currents, _ = get_w7x_data()
    biot_savart = BiotSavart(coils_via_symmetries(curves, currents, 5, True))

    def b(points):
        biot_savart.set_points(np.ascontiguousarray(points))
        return np.asarray(biot_savart.B())

    field = types.SimpleNamespace(b=b)
    # the coils of a half period carry the potential of half a field period
    potential = 2.0 * MU0 * sum(abs(current.get_value()) for current in currents)
    r_axis, _, _ = magnetic_axis(field, potential, steps, 5.95, 0.24)
    out = []
    for offset in offsets:
        points, strength, size, drift = drift_line(
            field,
            [r_axis + offset, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            periods * potential,
            periods * steps,
        )
        weight = potential / steps / strength**2
        iota = rotational_transform(field, r_axis, points[::steps], 2.0 * np.pi / 5.0)
        out.append(
            {
                "ripple": effective_ripple(strength, size, drift, weight, 5.5),
                "closure": poloidal_closure(strength, steps, iota / 5.0),
                "wells": float(
                    sum(np.mean(v**2) for v in well_residuals(strength, steps).values())
                ),
            }
        )
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--design", choices=list(DESIGNS), default="trefoil")
    parser.add_argument(
        "--search",
        type=int,
        default=0,
        metavar="COUNT",
        help="search again from the start of the design and COUNT - 1 drawn about it",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--pressure",
        type=float,
        nargs="*",
        default=[2000.0, 8000.0],
        help="p0 in Pa of the interiors with pressure p0 (1 - s)",
    )
    parser.add_argument(
        "--w7x",
        action="store_true",
        help="print the measures of W7-X by the same routines",
    )
    args = parser.parse_args()
    if args.w7x:
        for offset, k in zip((5, 10, 15), w7x(), strict=True):
            print(
                f"W7-X, surface from {offset} cm outboard of the magnetic axis: "
                f"effective ripple {100 * k['ripple']:.2f}%, contours of B closing "
                f"poloidally over {k['closure']:.2f} of its range, residuals of the "
                f"wells {k['wells']:.3f}"
            )
    design = DESIGNS[args.design]
    reach, mpol, ntor, planes, periods = SURFACES[args.design]
    if args.search:
        if args.design not in STARTS:
            parser.error(f"no search starts from {args.design}")
        begin = time.time()
        results = best_of(STARTS[args.design], args.search, args.seed, args.workers)
        for found, ripple in results:
            print(f"effective ripple {100 * ripple:.2f}%: {found}")
        design = results[0][0]
        print(f"{args.search} starts in {time.time() - begin:.0f} s")
    shares = (1 / 3, 2 / 3, 1.0)
    surfaces = [measures(design, share * reach, periods) for share in shares]
    k = surfaces[0]
    print(
        f"{args.design}: {design.coils} coils of radius {design.radius:.3f} m, at "
        f"least {k['clearance']:.3f} m apart; magnetic axis {k['length']:.2f} m long "
        f"through R = {k['r_axis']:.5f} m on the midplane of phi = 0, with mirror "
        f"ratio {k['mirror_axis']:.3f}"
    )
    for share, k in zip(shares, surfaces, strict=True):
        print(
            f"surface from {100 * share * reach:.1f} cm outboard of the axis: iota "
            f"{abs(k['iota']):.4f} per toroidal turn, mirror ratio {k['mirror']:.3f}, "
            f"{k['room']:.3f} m from the coils, effective ripple "
            f"{100 * k['ripple']:.2f}%, contours of B closing poloidally over "
            f"{k['closure']:.2f} of its range, residuals of the wells {k['wells']:.3f}"
        )
    k = interior(design, reach, mpol, ntor, planes)
    for p0 in (0.0, *args.pressure):
        try:
            wout = vmecpp.run(
                k["input"].model_copy(update={"pres_scale": p0}),
                max_threads=4,
                verbose=False,
            ).wout
        except RuntimeError as error:
            print(f"p0 = {p0:5.0f} Pa: {str(error).splitlines()[0][:120]}")
            continue
        shift = np.hypot(*(axis_curve(wout, k["planes"]) - k["axis"]).T).max()
        iota = np.asarray(wout.iotaf)
        ns = int(wout.ns)
        ripple = [wout_ripple(wout, js, periods) for js in (ns // 4, ns // 2, ns - 1)]
        print(
            f"p0 = {p0:5.0f} Pa, beta {wout.betatotal:.1e}: {wout.itfsq} iterations, "
            f"magnetic axis within {shift:.1e} m of the traced one, iota "
            f"{abs(iota[0]):.5f} on the axis and {abs(iota[-1]):.5f} on the boundary, "
            "effective ripple "
            + ", ".join(f"{100 * value:.2f}%" for value in ripple)
            + " at a quarter and half of the flux and on the boundary"
        )


if __name__ == "__main__":
    main()
