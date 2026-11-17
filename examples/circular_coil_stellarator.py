# SPDX-FileCopyrightText: 2026-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""A stellarator wound from identical circular coils, searched for a quasi-isodynamic
field and solved with VMEC++.

Every coil is the same planar circle with the same current. A design is the curve the
coils are centred on, their places along it and the law by which their axes tilt from
its tangent. ``star`` winds a curve whose field periods are plane curves, on which the
radial drift along a field line is odd about the middle of a period. In the vacuum
field of the coils the magnetic scalar potential is the Boozer toroidal angle, so a
field line traced in steps of the potential gives the field strength in Boozer
coordinates and the bounce integrals of the effective ripple of Nemov et al., Phys.
Plasmas 6, 4622 (1999), without an equilibrium. ``search`` moves a design by
Gauss-Newton steps on those integrals, taken along the tree in which the wells of a
field line merge, and on its Pfirsch-Schlueter currents, within limits on the distances
of the coils, the mirror ratio, the rotational transform and the angle at which the
curve crosses the planes of constant toroidal angle. The interior of a traced flux surface is solved as in
``knotted_stellarator.py``, with the pressure raised by hot restarts, and its wout gives
the effective ripple again.
"""

import argparse
import dataclasses
import types
from itertools import pairwise
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
from scipy.optimize import brentq, lsq_linear
from scipy.special import ellipe, ellipk
from simsopt.field import BiotSavart, coils_via_symmetries

import vmecpp

MU0 = 4.0e-7 * np.pi
# of the search: the starts of its surfaces outboard of the magnetic axis and the least
# distances between two coils and from a coil to a surface, in metres
OFFSETS = (0.1, 0.2, 0.3)
CLEARANCE, ROOM = 0.1, 0.15
MIRROR = 0.45  # largest mirror ratio on the magnetic axis
CROSSING = 0.4  # least sine of the angle of the curve to a plane phi = const
TRANSFORM = (0.808, 0.826)  # window of the rotational transform, within 4/5 to 5/6
CURRENTS = 1.0e-3  # weight of <j_par^2> / <j_perp^2> in the merit of the search
PERIODS = 60  # field periods of the lines of the search
SAMPLES = 16  # samples of a field line to a coil
# the series of a design and the largest move of each of their harmonics in a search,
# from the first harmonic (sideways from the constant)
FIELDS = ("r", "zs", "phis", "places", "outward", "sideways")
REACH = {
    "r": (0.9, 0.45, 0.45, 0.24, 0.24, 0.15),
    "zs": (1.2, 0.6, 0.3, 0.15, 0.15, 0.15),
    "phis": (0.6, 0.3, 0.18, 0.18, 0.12, 0.12),
    "places": (0.12, 0.09, 0.06, 0.045, 0.03, 0.03),
    "outward": (1.5,) * 6,
    "sideways": (1.5,) * 6,
}


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

    def separation(self, samples=48):
        """The distance from each coil to each other coil, [coil, coil], infinite on the
        diagonal."""
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
        return gaps.min(axis=1)

    def clearance(self, samples=48):
        """The smallest distance between two coils."""
        return float(self.separation(samples).min())

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

    def padded(self, counts):
        """The design with the series of FIELDS padded with zeros to counts
        harmonics."""
        return dataclasses.replace(
            self,
            **{
                name: getattr(self, name) + (0.0,) * (count - len(getattr(self, name)))
                for name, count in zip(FIELDS, counts, strict=True)
            },
        )

    @property
    def crossing(self):
        """The sine of the least angle between the curve and a plane phi = const."""
        z = 2.0 * np.pi / self.q * np.arange(512) / 512
        points = self.points(z)
        speed = np.linalg.norm(np.gradient(points, z, axis=0), axis=1)
        turn = np.gradient(np.unwrap(np.arctan2(points[:, 1], points[:, 0])), z)
        return float((np.hypot(points[:, 0], points[:, 1]) * turn / speed).min())

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

    def gaps(self, field):
        """The distances from each coil of half a field period to its first and second
        neighbours along the curve, and the smallest from them to any other coil."""
        separation = field.separation()
        rows = np.arange((self.per_period + 1) // 2)
        near = [separation[rows, (rows + shift) % self.coils] for shift in (1, 2)]
        for shift in (-2, -1, 1, 2):
            separation[rows, (rows + shift) % self.coils] = np.inf
        return np.array([*near[0], *near[1], separation[rows].min()])

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
    """A curve of q field periods that closes after p toroidal turns, each period a
    plane curve, wound with coils that are densest where it passes from one plane to the
    next.

    The curve is a wave on the faces of the prism {q / p} about the cylinder of radius
    inner, a regular prism for p = 1 and a star prism, on which it is a knot, for p >
    1. With v from -pi at one edge of a face to pi at the next,

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


# the start of the search: plane periods on the faces of a triangular prism 3.2 m from
# the axis, each a wave that meets the edges at 45 degrees, with 14 coils to a period
START = star(
    1, 3, 14, 0.85, 3.2, 0.6 * 3.2 * np.tan(np.pi / 3.0) / np.pi, 0.4, 0.35, order=3
).padded((6, 6, 6, 6, 5, 6))
# the design that the search has reached from it
FOUND = Design(
    p=1,
    q=3,
    per_period=14,
    radius=0.9050894473330581,
    r0=4.707288105806693,
    r=(
        1.5323190496119048,
        -0.022390967351230997,
        0.08246138367878764,
        -0.005416772725879753,
        -0.02739644200777635,
        0.00601738664050476,
    ),
    zs=(
        0.9377477371684005,
        0.11743951527898722,
        0.013582433804995341,
        -0.0029169402679679554,
        0.008538172421259616,
        -0.011644423345209947,
    ),
    phis=(
        -0.3066386842617297,
        0.058964224323790046,
        0.000865930174982378,
        -0.005244196690062911,
        0.0020951405887634457,
        -0.0022466712862611858,
    ),
    places=(
        -0.06593404982761622,
        0.01670638257520222,
        0.007702256244835227,
        -0.001881104252693904,
        0.00393989646570914,
        -0.0006828850947140823,
    ),
    outward=(
        -0.04041140408497798,
        0.04562280803342292,
        -0.0072834131936026225,
        -0.005224337996109935,
        0.006951225157614378,
    ),
    sideways=(
        -0.017446902899689823,
        0.06854985773835935,
        0.07756039643520769,
        -0.013482010601876995,
        -0.1024662085246131,
        0.006725931659078227,
    ),
)
# the start of the outermost surface of FOUND outboard of its magnetic axis, in metres,
# mpol, ntor and the planes per period of its interior, and the field periods of the
# lines of its measures
SURFACE = (0.3, 12, 24, 96, 120)


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


def drift_lines(field, starts, normals, potential, steps, spacing=1.0e-5):
    """The field lines through starts, [line, 3], with the gradient of the flux surface
    label carried along each from its row of normals: their points at steps uniform
    steps of the potential, [line, steps + 1, 3], and the field strength, the size of
    that gradient and the radial drift (b x grad B) . grad psi / B at them, [line,
    steps + 1]."""
    h = potential / steps
    x = np.array(starts, dtype=float).reshape(-1, 3)
    g = np.array(normals, dtype=float).reshape(-1, 3)
    count = len(x)
    stencil = np.concatenate(
        [np.zeros((1, 3)), spacing * np.eye(3), -spacing * np.eye(3)]
    )
    strength, size, drift = (np.empty((count, steps + 1)) for _ in range(3))
    points = np.empty((count, steps + 1, 3))

    def slopes(x, g):
        b = field.b((x[:, None, :] + stencil[None]).reshape(-1, 3)).reshape(count, 7, 3)
        b2 = np.einsum("lij,lij->li", b, b)
        v = b / b2[..., None]
        # the label is constant along the line, so its gradient obeys g' = -(grad v)^T g
        along = np.einsum("lij,lj->li", v, g)
        dg = -(along[:, 1:4] - along[:, 4:7]) / (2.0 * spacing)
        size_b = np.sqrt(b2)
        grad_b = (size_b[:, 1:4] - size_b[:, 4:7]) / (2.0 * spacing)
        return v[:, 0], dg, size_b[:, 0], b[:, 0], grad_b

    for i in range(steps + 1):
        v1, g1, size_b, b, grad_b = slopes(x, g)
        g = g - (np.sum(g * b, axis=1) / np.sum(b * b, axis=1))[:, None] * b
        strength[:, i], size[:, i] = size_b, np.linalg.norm(g, axis=1)
        drift[:, i] = np.sum(np.cross(b, grad_b) * g, axis=1) / size_b**2
        points[:, i] = x
        if i == steps:
            break
        v2, g2, *_ = slopes(x + 0.5 * h * v1, g + 0.5 * h * g1)
        v3, g3, *_ = slopes(x + 0.5 * h * v2, g + 0.5 * h * g2)
        v4, g4, *_ = slopes(x + h * v3, g + h * g3)
        x = x + h / 6.0 * (v1 + 2.0 * v2 + 2.0 * v3 + v4)
        g = g + h / 6.0 * (g1 + 2.0 * g2 + 2.0 * g3 + g4)
    return points, strength, size, drift


def drift_line(field, start, normal, potential, steps, spacing=1.0e-5):
    """One line of :func:`drift_lines`."""
    lines = drift_lines(field, start, normal, potential, steps, spacing)
    return tuple(values[0] for values in lines)


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


def wells_of(eta):
    """The wells of the piecewise linear eta along a field line, as (birth, death, left,
    right): the levels between which each exists and the samples that bound it.

    A well forms at a minimum of eta or where two wells merge at a maximum between them,
    and ends where it merges with a neighbour. A well that an end of the line cuts is
    left out from the level at which it reaches that end.
    """
    slope = np.sign(np.diff(eta))
    # a flat step takes the slope of the step before it
    for i in np.flatnonzero(slope == 0.0):
        slope[i] = slope[i - 1] if i > 0 else 1.0
    inner = np.flatnonzero((slope[:-1] > 0.0) & (slope[1:] < 0.0)) + 1
    ridges = np.concatenate([[0], inner, [len(eta) - 1]])
    count = len(ridges) - 1
    # the basins between consecutive ridges, merged from the lowest ridge upwards
    birth = [eta[a : b + 1].min() for a, b in pairwise(ridges)]
    whole, alive = [True] * count, [True] * count
    left, right = list(range(-1, count - 1)), list(range(1, count + 1))
    first, last = list(ridges[:-1]), list(ridges[1:])
    out = []
    # ridge k lies between basins k - 1 and k; ridges 0 and count are the ends
    for k in np.argsort(eta[ridges], kind="stable"):
        level = float(eta[ridges[k]])
        if k in (0, count):
            basin = 0 if k == 0 else count - 1
            while not alive[basin]:
                basin = left[basin]
            if whole[basin] and level > birth[basin]:
                out.append((birth[basin], level, first[basin], last[basin]))
            whole[basin], birth[basin] = False, level
            continue
        a = k - 1
        while not alive[a]:
            a = left[a]
        for basin in (a, k):
            if whole[basin] and level > birth[basin]:
                out.append((birth[basin], level, first[basin], last[basin]))
        whole[a], birth[a], last[a] = whole[a] and whole[k], level, last[k]
        alive[k], right[a] = False, right[k]
        if right[k] < count:
            left[right[k]] = a
    return out


def ripple_residuals(strength, size, drift, weight, major_radius, nodes=6):
    """Residuals whose squares sum to the effective ripple to the power 3/2, from the
    field strength, the size of the gradient of the flux surface label, the radial drift
    and the ds / B of a step at the samples of a field line.

    With B / B0 linear between the samples, B0 the mean strength, a well j of
    :func:`wells_of` has at the level b'

        H[j] = integral ds / B sqrt(b' - B / B0) (4 B0 / B - 1 / b') drift
        I[j] = integral ds / B sqrt(1 - B / (B0 b'))

    and the effective ripple of Nemov et al. (1999) is (pi R0^2 / (8 sqrt 2) integral
    ds / B / (integral ds |grad psi| / B)^2 integral db' sum H[j]^2 / (b'^2 I[j]))^(2/3).
    Each well is taken on Gauss nodes between the levels at which it forms and ends,
    so the residuals are continuous in the samples. Returns the samples that bound the
    wells, [well, 2], and the residuals, [well, node].
    """
    eta = strength * np.sum(weight) / np.sum(weight * strength)
    step_weight = 0.5 * (weight[:-1] + weight[1:])
    step_size = 0.5 * (weight[:-1] * size[:-1] + weight[1:] * size[1:])
    scale = (
        np.pi
        * major_radius**2
        / (8.0 * np.sqrt(2.0))
        * np.sum(step_weight)
        / np.sum(step_size) ** 2
    )
    x, share = np.polynomial.legendre.leggauss(nodes)
    x, share = 0.5 * (x + 1.0), 0.5 * share
    wells = wells_of(eta)
    values = np.zeros((len(wells), nodes))
    for index, (birth, death, a, b) in enumerate(wells):
        e, w, d = eta[a : b + 1], weight[a : b + 1], drift[a : b + 1]
        level = (birth + x * (death - birth))[:, None]
        below = np.maximum(level - e, 0.0)
        change = -np.diff(e)
        flat = np.abs(change) < 1.0e-14
        # the mean of sqrt(max(b' - B / B0, 0)) over each step
        root = np.where(
            flat,
            np.sqrt(below[:, :-1]),
            2.0 / 3.0 * np.diff(below**1.5, axis=1) / np.where(flat, 1.0, change),
        )
        factor = w * (4.0 / e - 1.0 / level) * d
        h = np.sum(root * 0.5 * (factor[:, :-1] + factor[:, 1:]), axis=1)
        i = np.sum(root * 0.5 * (w[:-1] + w[1:]), axis=1) / np.sqrt(level[:, 0])
        wet = i > 0.0
        values[index, wet] = (
            np.sqrt(scale * share[wet] * (death - birth))
            * h[wet]
            / (level[wet, 0] * np.sqrt(i[wet]))
        )
    return np.array([well[2:] for well in wells], dtype=int).reshape(-1, 2), values


def effective_ripple(strength, size, drift, weight, major_radius, nodes=12):
    """The effective ripple of a field line, from :func:`ripple_residuals`."""
    values = ripple_residuals(strength, size, drift, weight, major_radius, nodes)[1]
    return float(np.sum(values**2) ** (2.0 / 3.0))


def current_ratio(strength, size, drift, step):
    """<j_par^2> / <j_perp^2> on the surface of a field line at low pressure, from its
    samples in steps step of the potential.

    The diamagnetic current is p' |grad psi| / B and the Pfirsch-Schlueter current p' h
    B, with dh / dPhi = -2 drift / B^3 along the line and no net current. The ratio is 2
    / iota^2 in a torus of large aspect ratio.
    """
    return float(np.sum(parallel_current(strength, size, drift, step) ** 2))


def parallel_current(strength, size, drift, step):
    """The h of :func:`current_ratio` at the samples of a field line, scaled so that its
    squares sum to the ratio."""
    rate = -2.0 * drift / strength**3
    h = step * np.concatenate([[0.0], np.cumsum(0.5 * (rate[1:] + rate[:-1]))])
    h -= h.mean()
    return h / np.sqrt(np.sum(size**2 / strength**4))


def magnetic_well(strength, axis_strength):
    """The vacuum magnetic well of the surface of a field line, 1 - V'(surface) /
    V'(axis), from the field strength of the line and of the magnetic axis on uniform
    steps of the potential.

    V' is the volume per unit of flux, the mean of 1 / B^2 over the potential. The mean
    of the line is taken under the window (1 - cos)^2, which removes the modulation of
    its poloidal turns.
    """
    count = len(strength) - 1
    window = (1.0 - np.cos(2.0 * np.pi * (np.arange(count) + 0.5) / count)) ** 2
    volume = np.sum(window / strength[:-1] ** 2) / np.sum(window)
    return float(1.0 - volume / np.mean(1.0 / axis_strength[:-1] ** 2))


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


def measures_of(design, offsets, periods=120):
    """For each of offsets, the axis of a design and the field line from that far
    outboard of it over periods field periods: R of the axis, its length, the mirror ratio on it and on the line,
    the rotational transform per toroidal turn, the smallest distances between coils
    and from the line to a coil, :meth:`Design.gaps`, the effective ripple with the
    length of the axis over 2 pi as major radius, the share of
    :func:`poloidal_closure`, the sum of the mean squares of :func:`well_residuals`,
    :func:`current_ratio` and :func:`parallel_current`, :func:`magnetic_well`, and
    the data of the line."""
    field = design.loops()
    steps = SAMPLES * design.per_period
    r_axis, axis, axis_strength = design.axis(field)
    length = float(np.linalg.norm(np.diff(axis, axis=0), axis=1).sum() * design.q)
    potential = periods * design.period_potential
    # the surfaces are normal to e_R where they cross the line phi = 0, Z = 0
    lines = drift_lines(
        field,
        [[r_axis + offset, 0.0, 0.0] for offset in offsets],
        [[1.0, 0.0, 0.0]] * len(offsets),
        potential,
        periods * steps,
    )
    step = potential / (periods * steps)
    major = length / (2.0 * np.pi)
    clearance, gaps = field.clearance(), design.gaps(field)

    def mirror(b):
        return float((b.max() - b.min()) / (b.max() + b.min()))

    found = []
    for points, strength, size, drift in zip(*lines, strict=True):
        if not np.all(np.isfinite(strength)):
            message = "the field line meets a coil"
            raise ValueError(message)
        weight = step / strength**2
        iota = float(
            rotational_transform(field, r_axis, points[::steps], design.period)
        )
        found.append(
            {
                "r_axis": r_axis,
                "length": length,
                "mirror_axis": mirror(axis_strength),
                "mirror": mirror(strength),
                "iota": iota,
                "clearance": clearance,
                "gaps": gaps,
                "room": float(field.distance(points[::4]).min()),
                "ripple": effective_ripple(strength, size, drift, weight, major),
                "closure": poloidal_closure(
                    strength, steps, iota * design.period / (2.0 * np.pi)
                ),
                "wells": float(
                    sum(np.mean(v**2) for v in well_residuals(strength, steps).values())
                ),
                "currents": current_ratio(strength, size, drift, step),
                "current": parallel_current(strength, size, drift, step),
                "well": magnetic_well(strength, axis_strength),
                "line": (strength, size, drift, weight, major, steps),
            }
        )
    return found


def measures(design, offset, periods=120):
    """:func:`measures_of` of one offset."""
    return measures_of(design, [offset], periods)[0]


def search_parts(design):
    """What a step of :func:`search` takes of a design, or None where it has no
    magnetic axis or a field line of it meets a coil: the wells of the lines of its
    surfaces and their residuals as :func:`ripple_residuals` gives them, the squares
    summing to the mean over the surfaces of the effective ripple to the power 3/2;
    the Pfirsch-Schlueter currents of the lines, the squares summing to CURRENTS times
    the mean of :func:`current_ratio`; and what the design exceeds its limits by,
    relative to each limit."""
    keys, values, currents, found = [], [], [], []
    try:
        with np.errstate(invalid="ignore", divide="ignore"):
            for k in measures_of(design, OFFSETS, PERIODS):
                wells, residuals = ripple_residuals(*k["line"][:5])
                if not np.all(np.isfinite(residuals)):
                    return None
                keys.append(wells)
                values.append(residuals / np.sqrt(len(OFFSETS)))
                currents.append(
                    np.sqrt(4.0 * CURRENTS / len(OFFSETS)) * k["current"][::4]
                )
                found.append(k)
    except (ValueError, IndexError):
        return None
    limits = [
        *((CLEARANCE - found[0]["gaps"]) / CLEARANCE),
        (ROOM - min(k["room"] for k in found)) / ROOM,
        (found[0]["mirror_axis"] - MIRROR) / MIRROR,
        (CROSSING - design.crossing) / CROSSING,
        (TRANSFORM[0] - min(abs(k["iota"]) for k in found)) / TRANSFORM[0],
        (max(abs(k["iota"]) for k in found) - TRANSFORM[1]) / TRANSFORM[1],
    ]
    return keys, values, np.concatenate(currents), np.array(limits)


def _aligned(keys, values, base):
    """Values in the order of the wells base, a well matched where its bounds lie within
    a sample of those of a base well, with the wells matched."""
    table = {(int(a), int(b)): k for k, (a, b) in enumerate(keys)}
    out = np.zeros((len(base), values.shape[1]))
    matched = np.zeros(len(base), dtype=bool)
    shifts = [(da, db) for da in (0, -1, 1) for db in (0, -1, 1)]
    for k, (a, b) in enumerate(base):
        for da, db in shifts:
            match = table.get((int(a) + da, int(b) + db))
            if match is not None:
                out[k], matched[k] = values[match], True
                break
    return out, matched


def search(start, workers=4, steps=300, report=None):
    """The design that Gauss-Newton steps on :func:`search_parts` reach from start.

    The parameters are the harmonics of the series of start and its coil radius, each
    within REACH of its start. A limit g <= 0 enters a step linearized, as weight (g +
    its change + z) with a slack z >= 0, and the move of the parameters is damped. Each
    step solves that bounded linear least-squares problem at nine dampings, evaluates
    the nine moves on the processes of the pool and keeps the best; its merit is the sum
    of the squares of the residuals and of weight times what exceeds the limits. The
    Jacobian is by differences, a well of the moved design matched to one of the design
    by the samples that bound it. Where four steps in a row gain nothing the damping
    starts again, up to three times, while the merit has fallen by a hundredth since it
    last did. report is called with a line of text at each step.
    """
    dampings = np.array([1.0e-2, 3.0e-2, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0])
    weight, move = 10.0, 5.0e-4
    counts = [len(getattr(start, name)) for name in FIELDS]
    reach = np.array(
        [
            REACH[name][m]
            for name, count in zip(FIELDS, counts, strict=True)
            for m in range(count)
        ]
        + [0.3 * start.radius]
    )
    scale = 0.5 * reach

    def moved(x):
        series, at = {}, 0
        for name, count in zip(FIELDS, counts, strict=True):
            series[name] = tuple(float(v) for v in x[at : at + count])
            at += count
        return dataclasses.replace(start, radius=float(x[at]), **series)

    def merit(parts):
        if parts is None:
            return np.inf
        _, values, currents, limits = parts
        return float(
            sum(np.sum(v**2) for v in values)
            + currents @ currents
            + weight**2 * np.sum(np.maximum(limits, 0.0) ** 2)
        )

    x = np.array([v for name in FIELDS for v in getattr(start, name)] + [start.radius])
    low, high = x - reach, x + reach
    count = x.size
    with Pool(workers) as pool:
        here = search_parts(start)
        cost, damping, misses = merit(here), 0.03, 0
        rounds, before = 4, cost
        for step in range(steps):
            if misses >= 4 or damping > 1.0e10:
                rounds -= 1
                if rounds <= 0 or before - cost < 0.01 * cost:
                    break
                damping, misses, before = 0.03, 0, cost
            keys, values, currents, limits = here
            parts = np.concatenate([v.ravel() for v in values] + [currents])
            slacks = limits.size
            moves = move * scale * np.where(x + move * scale > high, -1.0, 1.0)
            columns = pool.map(
                search_parts,
                [moved(x + moves[i] * np.eye(count)[i]) for i in range(count)],
            )
            j_parts, j_limits = np.zeros((parts.size, count)), np.zeros((slacks, count))
            for i, there in enumerate(columns):
                if there is None:
                    continue
                changes = []
                for wells, residuals, base, mine in zip(
                    there[0], there[1], keys, values, strict=True
                ):
                    change, matched = _aligned(wells, residuals, base)
                    changes.append(
                        np.where(matched[:, None], change - mine, 0.0).ravel()
                    )
                changes.append(there[2] - currents)
                j_parts[:, i] = np.concatenate(changes) / moves[i] * scale[i]
                j_limits[:, i] = (there[3] - limits) / moves[i] * scale[i]
            lower = np.concatenate([(low - x) / scale, np.zeros(slacks)])
            upper = np.concatenate([(high - x) / scale, np.full(slacks, np.inf)])
            right = -np.concatenate([parts, weight * limits, np.zeros(count)])
            trials = []
            for factor in dampings:
                matrix = np.block(
                    [
                        [j_parts, np.zeros((parts.size, slacks))],
                        [weight * j_limits, weight * np.eye(slacks)],
                        [
                            np.sqrt(damping * factor) * np.eye(count),
                            np.zeros((count, slacks)),
                        ],
                    ]
                )
                solution = lsq_linear(
                    matrix, right, bounds=(lower, upper), method="bvls"
                )
                trials.append(np.clip(x + scale * solution.x[:count], low, high))
            found = pool.map(search_parts, [moved(trial) for trial in trials])
            merits = [merit(parts) for parts in found]
            best = int(np.argmin(merits))
            if merits[best] < cost:
                misses = (
                    misses + 1 if cost - merits[best] < 1.0e-5 * merits[best] else 0
                )
                x, cost, here = trials[best], merits[best], found[best]
                damping = float(np.clip(damping * dampings[best], 1.0e-8, 1.0e8))
            else:
                damping *= 300.0
                misses += 1
            if report:
                report(f"step {step}: merit {cost:.6f}, damping {damping:.1e}")
    return moved(x)


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


def wout_ripple(wout, js, periods=120, steps=96):
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
    return effective_ripple(strength, size, drift, weight, major)


def w7x(offsets=(0.05, 0.1, 0.15), periods=40, steps=120):
    """The effective ripple, the share of :func:`poloidal_closure`, the sum of the mean
    squares of :func:`well_residuals` and :func:`current_ratio` of the standard
    configuration of W7-X by the routines of the designs, from the coils that SIMSOPT
    ships, on the surfaces from offsets outboard of its axis in the bean plane, with
    5.5 m as major radius."""
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
                "currents": current_ratio(strength, size, drift, potential / steps),
            }
        )
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--search", action="store_true", help="search again from the start"
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--pressure",
        type=float,
        nargs="*",
        default=[8000.0, 16000.0, 24000.0],
        help="p0 in Pa of the interiors with pressure p0 (1 - s), in rising order",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1.0e-10,
        help="force tolerance of the interiors with pressure",
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
                f"wells {k['wells']:.3f}, <j_par^2> / <j_perp^2> {k['currents']:.2f}"
            )
    design = FOUND
    if args.search:
        design = search(START, args.workers, report=print)
        print(design)
    reach, mpol, ntor, planes, periods = SURFACE
    shares = (1 / 3, 2 / 3, 1.0)
    surfaces = [measures(design, share * reach, periods) for share in shares]
    k = surfaces[0]
    print(
        f"{design.coils} coils of radius {design.radius:.3f} m, at least "
        f"{k['clearance']:.3f} m apart; magnetic axis {k['length']:.2f} m long through "
        f"R = {k['r_axis']:.5f} m on the midplane of phi = 0, with mirror ratio "
        f"{k['mirror_axis']:.3f}"
    )
    for share, k in zip(shares, surfaces, strict=True):
        print(
            f"surface from {100 * share * reach:.0f} cm outboard of the axis: iota "
            f"{abs(k['iota']):.4f} per toroidal turn, mirror ratio {k['mirror']:.3f}, "
            f"{k['room']:.3f} m from the coils, effective ripple "
            f"{100 * k['ripple']:.2f}%, contours of B closing poloidally over "
            f"{k['closure']:.2f} of its range, residuals of the wells {k['wells']:.3f}, "
            f"<j_par^2> / <j_perp^2> {k['currents']:.2f}, vacuum magnetic well "
            f"{100 * k['well']:+.2f}%"
        )
    k = interior(design, reach, mpol, ntor, planes)
    last = None
    for p0 in (0.0, *args.pressure):
        step = k["input"].model_copy(update={"pres_scale": p0})
        if last is not None:
            # a hot restart from the pressure before, which takes one multigrid step
            step = step.model_copy(
                update={
                    "ns_array": step.ns_array[-1:],
                    "ftol_array": np.array([args.tolerance]),
                    "niter_array": step.niter_array[-1:],
                }
            )
        try:
            last = vmecpp.run(step, max_threads=4, verbose=False, restart_from=last)
        except RuntimeError as error:
            print(f"p0 = {p0:5.0f} Pa: {str(error).splitlines()[0][:120]}")
            break
        wout = last.wout
        shift = np.hypot(*(axis_curve(wout, k["planes"]) - k["axis"]).T).max()
        iota = np.asarray(wout.iotaf)
        ns = int(wout.ns)
        ripple = [wout_ripple(wout, js, periods) for js in (ns // 4, ns // 2, ns - 1)]
        mercier = np.asarray(wout.DMerc)[2:-2]
        print(
            f"p0 = {p0:5.0f} Pa, beta {wout.betatotal:.1e}: {wout.itfsq} iterations, "
            f"magnetic axis {shift:.1e} m from the traced one, iota "
            f"{abs(iota[0]):.5f} on the axis and {abs(iota[-1]):.5f} on the boundary, "
            "effective ripple "
            + ", ".join(f"{100 * value:.2f}%" for value in ripple)
            + " at a quarter and half of the flux and on the boundary, Mercier's "
            f"criterion positive on {np.mean(mercier > 0.0):.2f} of the surfaces"
        )


if __name__ == "__main__":
    main()
