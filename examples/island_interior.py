# SPDX-FileCopyrightText: 2026-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""The interior of a W7-X edge island as its own equilibrium over the transits after
which it closes, against field-line tracing and in a finite-beta core's field."""

import argparse
import math
import threading
import time
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline, RectBivariateSpline
from scipy.optimize import brentq, minimize_scalar
from scipy.spatial import cKDTree
from simsopt.field import (
    BiotSavart,
    Coil,
    Current,
    InterpolatedField,
    coils_via_symmetries,
)
from simsopt.field.tracing import compute_fieldlines
from simsopt.geo import RotatedCurve

import vmecpp

NFP = 5
MU0 = 4.0e-7 * np.pi
# interpolation cells and degree, island mpol, boundary harmonics, largest toroidal mode,
# ns; core resolution, casing upsampling, core-field grid and closures traced in it
SETTINGS = {
    "default": {
        "cells": (27, 24, 18),
        "degree": 3,
        "mpol": 8,
        "harmonics": 12,
        "modes": 100,
        "ns": 13,
        "core": (6, 6, [13]),
        "upsampling": (12, 8),
        "grid": (11, 7),
        "core_turns": 60,
    },
    "full": {
        "cells": (54, 48, 36),
        "degree": 4,
        "mpol": 16,
        "harmonics": 16,
        "modes": 120,
        "ns": 51,
        "core": (8, 8, [13, 25]),
        "upsampling": (12, 8),
        "grid": (13, 9),
        "core_turns": 120,
    },
}
# per chain: planar coil current ratio, coil rotation, edge iota, transits, periods, the
# O-point interval, surface offsets, separatrix, axis interval, core offsets, ftol
EDGES = {
    "5/5": {
        "planar": 0.0,
        "rotation": 0.0,
        "iota": 1.0,
        "transits": 1,
        "periods": 1,
        "o_point": (6.20, 6.27),
        "offset": (0.010, 0.016),
        "separatrix": 0.029,
        "axis": (5.90, 6.00),
        "core_offset": 0.22,
        "offset_with_core": 0.010,
        "ftol": 1e-13,
    },
    "5/4": {
        "planar": -0.23,
        "rotation": 0.0,
        "iota": 1.25,
        "transits": 4,
        "periods": 5,
        "o_point": (6.196, 6.2025),
        "offset": (0.0058, 0.0070),
        "separatrix": 0.0103,
        "axis": (5.90, 6.00),
        "core_offset": 0.15,
        "offset_with_core": 0.002,
        "ftol": 1e-11,
    },
    "5/6": {
        "planar": 0.25,
        "rotation": np.pi / NFP,
        "iota": 5.0 / 6.0,
        "transits": 6,
        "periods": 5,
        "o_point": (5.93, 6.00),
        "offset": (0.030, 0.055),
        "separatrix": 0.072,
        "axis": (5.15, 5.25),
        "core_offset": 0.45,
        "offset_with_core": 0.030,
        "ftol": 1e-11,
    },
}
# closures of the island over which its surface is traced in the coil field
SURFACE_TURNS = 150
# island pressures p0 in Pa
PRESSURES = (100.0, 400.0)
CORE_PRESSURE = 5.0e4
RK4_STEPS = 5  # per plane of the island run
# the least spacing in rad of the planes whose crossings SIMSOPT records
MIN_PLANE_SPACING = 0.02
# the least distance of the grid of the core field from the casing points, over their
# spacing
CASING_MARGIN = 1.5


class FieldLines:
    """Field lines of the vacuum field, crossing given toroidal angles."""

    def __init__(self, cells, degree, planar, rotation):
        try:  # SIMSOPT 1.11 and later
            from simsopt.configs import get_data  # noqa: PLC0415

            curves, currents = get_data("w7x")[:2]
        except ImportError:
            from simsopt.configs import get_w7x_data  # noqa: PLC0415

            curves, currents, _ = get_w7x_data()
        if planar != 0.0:
            # the two planar coils follow the five non-planar ones
            i_np = currents[0].get_value()
            currents = [*currents[:5], *(Current(planar * i_np) for _ in range(2))]
        coils = coils_via_symmetries(curves, currents, NFP, True)
        if rotation:
            # turned so that the symmetry plane of the O points lies at phi = 0
            coils = [
                Coil(RotatedCurve(c.curve, -rotation, False), c.current) for c in coils
            ]
        nr, nphi, nz = cells
        self.field = InterpolatedField(
            BiotSavart(coils),
            degree,
            (4.3, 7.0, nr),
            (0.0, 2.0 * np.pi / NFP, nphi),
            (0.0, 1.4, nz),
            True,
            nfp=NFP,
            stellsym=True,
        )
        self.lock = threading.Lock()
        # the lines run toward decreasing or increasing phi with the sign of B_phi
        b_phi = self.b(np.array([6.0]), np.zeros(1), np.zeros(1))[1][0]
        self.sign = -1 if b_phi < 0.0 else 1

    def b(self, r, phi, z):
        """Cylindrical components of B at the points (r, phi, z)."""
        cos, sin = np.cos(phi), np.sin(phi)
        points = np.ascontiguousarray(np.stack([r * cos, r * sin, z], -1))
        # VMEC++ calls the external field of a free-boundary run from several threads
        # at once, and a SIMSOPT field keeps its evaluation points as state
        with self.lock:
            self.field.set_points(points)
            b = np.array(self.field.B())
        return b[:, 0] * cos + b[:, 1] * sin, -b[:, 0] * sin + b[:, 1] * cos, b[:, 2]

    def onto(self, r, phi, z, target, steps=4):
        """(R, Z) where the lines through (r, phi, z) cross the nearby angles target, by
        RK4 in the toroidal angle."""

        def slope(r, phi, z):
            b_r, b_phi, b_z = self.b(r, phi, z)
            return r * b_r / b_phi, r * b_z / b_phi

        h = np.angle(np.exp(1j * (target - phi))) / steps
        for _ in range(steps):
            k1 = slope(r, phi, z)
            k2 = slope(r + 0.5 * h * k1[0], phi + 0.5 * h, z + 0.5 * h * k1[1])
            k3 = slope(r + 0.5 * h * k2[0], phi + 0.5 * h, z + 0.5 * h * k2[1])
            k4 = slope(r + h * k3[0], phi + h, z + h * k3[1])
            r = r + h / 6.0 * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0])
            z = z + h / 6.0 * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1])
            phi = phi + h
        return r, z

    def crossings(self, r0, z0, a, b, count):
        """(R, Z) of the line through (r0, 0, z0) at the toroidal angles 2 pi a s / b
        along B, s = 1, ..., count."""
        # SIMSOPT records crossings only of planes MIN_PLANE_SPACING apart: it traces
        # every stride-th angle and carries the line from there onto those between
        stride = 1
        while 2.0 * np.pi * np.gcd(a * stride, b) / b < MIN_PLANE_SPACING:
            stride += 1
        if stride == 1:
            return self._crossings(r0, z0, a, b, count)
        coarse = count // stride
        r, z = np.empty(count), np.empty(count)
        if coarse:
            rc, zc = self._crossings(r0, z0, a * stride, b, coarse)
            r[stride - 1 :: stride][:coarse], z[stride - 1 :: stride][:coarse] = rc, zc
        s = np.arange(1, count + 1)
        base = s // stride
        start_r = np.where(base > 0, r[np.maximum(base * stride - 1, 0)], r0)
        start_z = np.where(base > 0, z[np.maximum(base * stride - 1, 0)], z0)
        angle = self.sign * 2.0 * np.pi * a / b
        for offset in range(1, stride):
            sel = s - base * stride == offset
            if np.any(sel):
                r[sel], z[sel] = self.onto(
                    start_r[sel],
                    angle * (base[sel] * stride),
                    start_z[sel],
                    angle * s[sel],
                    steps=16 * offset,
                )
        return r, z

    def _crossings(self, r0, z0, a, b, count):
        # in steps of 2 pi / b along the line the angles are the steps a s; it crosses
        # the planes of these angles at every step u whose plane (sign u) mod b is one
        index = np.unique((self.sign * a * np.arange(1, count + 1)) % b)
        steps = np.arange(1, a * count + 1)
        steps = steps[np.isin((self.sign * steps) % b, index)]
        planes = 2.0 * np.pi * index / b
        # SIMSOPT integrates dx/dt = B, a turn of about 2 pi R taking t = 2 pi R / |B|
        _, hits = compute_fieldlines(
            self.field,
            np.array([r0]),
            np.array([z0]),
            tmax=(a * count / b + 1.0) * 2.0 * np.pi * 6.4 / 2.1,
            tol=1e-11,
            phis=list(planes),
        )
        h = np.asarray(hits[0])
        h = h[(h[:, 0] > 0.0) & (h[:, 1] >= 0)][: len(steps)]
        expected = np.searchsorted(index, (self.sign * steps) % b)
        if len(h) < len(steps) or np.any(h[:, 1].astype(int) != expected):
            msg = "the field line does not cross the planes in the expected order"
            raise RuntimeError(msg)
        h = h[steps % a == 0]
        x, y, z = h[:, 2], h[:, 3], h[:, 4]
        target = planes[h[:, 1].astype(int)]
        return self.onto(np.hypot(x, y), np.arctan2(y, x), z, target)


class PlaneTracer:
    """Field lines of the vacuum field plus another field, by RK4 in the toroidal angle
    with steps of a fixed fraction of the spacing of the planes of the island run."""

    def __init__(self, lines, extra, transits, count):
        self.lines, self.extra = lines, extra
        self.sign = lines.sign
        # RK4_STEPS steps between the count planes the island meets over its closure
        self.transits, self.steps = transits, count * RK4_STEPS

    def b(self, r, phi, z):
        b, c = self.lines.b(r, phi, z), self.extra(r, phi, z)
        return b[0] + c[0], b[1] + c[1], b[2] + c[2]

    def crossings(self, r0, z0, a, b, count):
        """(R, Z) of the line through (r0, 0, z0) at the toroidal angles 2 pi a s / b
        along B, s = 1, ..., count."""
        per, rest = divmod(a * self.steps, b * self.transits)
        if rest:
            msg = "the angles do not fall on the steps"
            raise ValueError(msg)
        h = self.sign * 2.0 * np.pi * self.transits / self.steps
        r, z = np.array([float(r0)]), np.array([float(z0)])

        def slope(r, phi, z):
            b_r, b_phi, b_z = self.b(r, np.full(1, phi), z)
            return r * b_r / b_phi, r * b_z / b_phi

        out_r, out_z = np.empty(count), np.empty(count)
        for i in range(count * per):
            phi = i * h
            k1 = slope(r, phi, z)
            k2 = slope(r + 0.5 * h * k1[0], phi + 0.5 * h, z + 0.5 * h * k1[1])
            k3 = slope(r + 0.5 * h * k2[0], phi + 0.5 * h, z + 0.5 * h * k2[1])
            k4 = slope(r + h * k3[0], phi + h, z + h * k3[1])
            r = r + h / 6.0 * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0])
            z = z + h / 6.0 * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1])
            if (i + 1) % per == 0:
                out_r[(i + 1) // per - 1], out_z[(i + 1) // per - 1] = r[0], z[0]
        return out_r, out_z


def circular_angle(r, z, center):
    """The angle about center of the points (r, z) once their covariance has been mapped
    to a circle, unwrapped along them."""
    d = np.stack([r - center[0], z - center[1]], -1)
    values, vectors = np.linalg.eigh(d.T @ d / len(d))
    w = d @ (vectors / np.sqrt(values)) @ vectors.T
    return np.unwrap(np.arctan2(w[:, 1], w[:, 0]))


def o_point(lines, edge):
    """R of the O point at phi = 0 by half-closure symmetry, and the turn per closure of
    a line 0.2 mm away, modulo whole turns and sign."""
    transits = edge["transits"]

    def z_at_half(r):
        return lines.crossings(r, 0.0, transits, 2, 1)[1][0]

    r_o = brentq(z_at_half, *edge["o_point"], xtol=1e-11)
    r, z = lines.crossings(r_o + 2e-4, 0.0, transits, 1, 100)
    angle = circular_angle(r, z, (r_o, 0.0))
    rotation = abs(angle[-1] - angle[0]) / (2.0 * np.pi * (len(angle) - 1))
    return r_o, rotation


def surface_fit(plane, r, z, turns_along, line, harmonics):
    """R and Z of the traced surface against its straight-field-line angle in each
    plane, [plane, harmonic], with the surface iota that makes the fit closest."""
    nplanes = len(line)

    def fit(iota, harmonics):
        theta = 2.0 * np.pi * iota * turns_along
        coef = np.empty((2, nplanes, 2 * harmonics - 1))
        worst = 0.0
        for k in range(nplanes):
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

    # first estimate from the angle about the O-point line, each plane's crossings mapped
    # onto a circle since the cross-sections are long thin ellipses; then a refined scan
    angle = np.empty_like(r)
    for k in range(nplanes):
        sel = plane == k
        angle[sel] = circular_angle(r[sel], z[sel], line[k])
    # the angle of every plane, unwrapped along the line
    angle = np.unwrap(np.mod(angle, 2.0 * np.pi))
    closures = turns_along[-1] - turns_along[0]
    iota0 = (angle[-1] - angle[0]) / (2.0 * np.pi * closures)
    # the estimate is off by a bounded angle over the whole trace
    width = max(2e-3, 0.05 / closures)
    scan = iota0 + np.linspace(-width, width, 41)
    best = int(np.argmin([fit(x, 4)[1] for x in scan]))
    step = scan[1] - scan[0]
    iota = minimize_scalar(
        lambda x: fit(x, 4)[1],
        bounds=(scan[best] - step, scan[best] + step),
        method="bounded",
        options={"xatol": 1e-10},
    ).x
    coef, worst = fit(iota, harmonics)
    return coef, iota, worst


def fourier(values, phi, theta, mpol, ntor, parity):
    """Coefficients of values[plane, theta] in cos (parity 0) or sin (parity 1) of
    (m theta - n phi), [m, ntor + n]."""
    out = np.zeros((mpol, 2 * ntor + 1))
    basis = np.cos if parity == 0 else np.sin
    for m in range(mpol):
        for n in range(-ntor, ntor + 1):
            if m == 0 and n <= 0 and not (n == 0 and parity == 0):
                continue
            weight = (1.0 if (m == 0 and n == 0) else 2.0) / values.size
            out[m, ntor + n] = weight * np.sum(
                values * basis(m * theta[None, :] - n * phi[:, None])
            )
    return out


def surface_curve(wout, js, theta, phi):
    """R and Z of flux surface js at the angles theta in the planes phi, [plane,
    theta]."""
    # xn / toroidal_transits is the toroidal mode number in phi
    angle = (
        np.asarray(wout.xm)[:, None, None] * theta[None, None, :]
        - np.asarray(wout.xn)[:, None, None]
        / wout.toroidal_transits
        * phi[None, :, None]
    )
    return (
        np.einsum("k,kpt->pt", np.asarray(wout.rmnc)[:, js], np.cos(angle)),
        np.einsum("k,kpt->pt", np.asarray(wout.zmns)[:, js], np.sin(angle)),
    )


def axis_curve(wout, phi):
    r, z = surface_curve(wout, 0, np.zeros(1), phi)
    return np.stack([r[:, 0], z[:, 0]], axis=1)


class InteriorMap:
    """R and Z of an interior as functions of rho = sqrt(s), theta and phi; the
    coefficient mn has the toroidal mode number xn[mn] / toroidal_transits in phi."""

    def __init__(self, wout):
        self.m = np.asarray(wout.xm, dtype=float)
        self.n = np.asarray(wout.xn, dtype=float) / wout.toroidal_transits
        rho = np.sqrt(np.linspace(0.0, 1.0, int(wout.ns)))
        # the coefficients of m >= 1 vanish as rho^m on the axis, which a cubic in rho
        # through rho = 0 carries
        self.r = CubicSpline(rho, np.asarray(wout.rmnc).T, axis=0)
        self.z = CubicSpline(rho, np.asarray(wout.zmns).T, axis=0)

    def jet(self, rho, theta, phi):
        """R, Z and their derivatives in rho, theta and phi at the points."""
        angle = self.m * theta[:, None] - self.n * phi[:, None]
        cos, sin = np.cos(angle), np.sin(angle)
        rc, rc_d, zs, zs_d = self.r(rho), self.r(rho, 1), self.z(rho), self.z(rho, 1)
        return {
            "r": np.sum(rc * cos, 1),
            "z": np.sum(zs * sin, 1),
            "r_rho": np.sum(rc_d * cos, 1),
            "z_rho": np.sum(zs_d * sin, 1),
            "r_theta": -np.sum(rc * sin * self.m, 1),
            "z_theta": np.sum(zs * cos * self.m, 1),
            "r_phi": np.sum(rc * sin * self.n, 1),
            "z_phi": -np.sum(zs * cos * self.n, 1),
        }

    def locate(self, r0, phi0, z0):
        """(rho, theta) of the laboratory points (r0, phi0, z0): the nearest point of a
        grid in each plane, refined by Newton steps that do not cross the axis."""
        grid_rho, grid_theta = (
            a.ravel()
            for a in np.meshgrid(
                np.linspace(0.05, 1.0, 24),
                2.0 * np.pi * np.arange(64) / 64,
                indexing="ij",
            )
        )
        rho, theta = np.empty_like(r0), np.empty_like(r0)
        for value in np.unique(phi0):
            sel = phi0 == value
            j = self.jet(grid_rho, grid_theta, np.full_like(grid_rho, value))
            d = (j["r"] - r0[sel, None]) ** 2 + (j["z"] - z0[sel, None]) ** 2
            k = np.argmin(d, axis=1)
            rho[sel], theta[sel] = grid_rho[k], grid_theta[k]
        for _ in range(40):
            j = self.jet(rho, theta, phi0)
            fr, fz = j["r"] - r0, j["z"] - z0
            det = j["r_rho"] * j["z_theta"] - j["r_theta"] * j["z_rho"]
            d_rho = (j["z_theta"] * fr - j["r_theta"] * fz) / det
            d_theta = (j["r_rho"] * fz - j["z_rho"] * fr) / det
            # halve a step that would cross the axis
            scale = np.ones_like(rho)
            cross = d_rho > rho
            scale[cross] = 0.5 * rho[cross] / d_rho[cross]
            rho, theta = rho - scale * d_rho, theta - scale * d_theta
            if np.max(np.hypot(fr, fz)) < 1e-12:
                break
        return rho, theta

    def radial(self, rho, theta, phi):
        """Grad rho / |grad rho| at the points in cylindrical components, with grad rho
        = e_theta x e_phi / sqrt(g)."""
        j = self.jet(rho, theta, phi)
        zero = np.zeros_like(j["r"])
        e_rho = np.stack([j["r_rho"], zero, j["z_rho"]], -1)
        normal = np.cross(
            np.stack([j["r_theta"], zero, j["z_theta"]], -1),
            np.stack([j["r_phi"], j["r"], j["z_phi"]], -1),
        )
        grad = normal / np.sum(e_rho * normal, -1)[:, None]
        return grad / np.linalg.norm(grad, axis=1)[:, None]

    def vectors(self, r, phi, z, b):
        """Rho, theta and the radial and binormal unit vectors at the laboratory points,
        the binormal b x e_r / |b x e_r| for the field b there."""
        rho, theta = self.locate(r, phi, z)
        e_r = self.radial(rho, theta, phi)
        e_b = np.cross(b, e_r)
        return rho, theta, e_r, e_b / np.linalg.norm(e_b, axis=1)[:, None]


def island_coordinates(lines, interior, edge, planes):
    """Rho, radial and binormal unit vectors of the interior on lines traced inside it,
    the binormal b x e_r / |b x e_r| with b along the coil field."""
    imap = InteriorMap(interior["output"].wout)
    nzeta = len(planes)
    count = edge["periods"] * nzeta
    spread, lean = 0.0, 0.0
    for f in (0.25, 0.5, 0.75, 0.95):
        r, z = lines.crossings(
            interior["r_o"] + f * edge["offset"],
            0.0,
            edge["transits"],
            count,
            4 * count,
        )
        phi = planes[(lines.sign * np.arange(1, len(r) + 1)) % nzeta]
        pick = slice(None, None, max(1, len(r) // 200))
        r, phi, z = r[pick], phi[pick], z[pick]
        b = np.stack(lines.b(r, phi, z), -1)
        b /= np.linalg.norm(b, axis=1)[:, None]
        rho, _, e_r, _ = imap.vectors(r, phi, z, b)
        spread = max(spread, float(np.ptp(rho)))
        lean = max(lean, float(np.abs(np.sum(b * e_r, 1)).max()))
    print(
        f"island coordinates: on lines traced from 0.25 to 0.95 of the boundary offset, "
        f"rho varies along each by at most {spread:.1e}, and the radial vector leans "
        f"into the coil field by at most {lean:.1e}"
    )
    return imap


def island(lines, s, planes, label, edge):
    """The O point, a traced surface and the interior of the island, in the field of
    lines."""
    transits, periods = edge["transits"], edge["periods"]
    nzeta = len(planes)
    # the planes cover one field period of the domain; the s-th crossing along the line
    # lies at plane (sign s) mod nzeta, s / (periods nzeta) closures along it
    count = periods * nzeta
    start = time.time()
    r_o, rotation = o_point(lines, edge)
    r, z = lines.crossings(r_o, 0.0, transits, count, nzeta)
    steps = np.arange(1, nzeta + 1)
    line = np.empty((nzeta, 2))
    line[(lines.sign * steps) % nzeta] = np.stack([r, z], -1)
    closures = s["core_turns"] if isinstance(lines, PlaneTracer) else SURFACE_TURNS
    r, z = lines.crossings(r_o + edge["offset"], 0.0, transits, count, closures * count)
    steps = np.arange(1, closures * count + 1)
    plane = (lines.sign * steps) % nzeta
    turns_along = steps / count
    coef, iota_surface, residual = surface_fit(
        plane, r, z, turns_along, line, s["harmonics"]
    )
    print(f"{label}: O point R = {r_o:.6f} m on the midplane at phi = 0, where the")
    print(
        f"  {transits}-transit map turns the lines near it by {rotation:.5f} of a turn;"
    )
    print(
        f"  surface traced from {edge['offset'] * 1e3:.1f} mm outboard of it, of the "
        f"{edge['separatrix'] * 1e3:.1f} mm to the separatrix of the coil field, over"
    )
    print(
        f"  {closures * transits} transits, iota per transit "
        f"{iota_surface / transits:+.6f}, fit in"
    )
    print(f"  {s['harmonics'] - 1} poloidal harmonics within {residual:.1e} m")

    # the boundary on a uniform grid of the straight-field-line angle, and the toroidal
    # flux through each cross-section from B_phi on signed triangles from the O point
    theta = 2.0 * np.pi * np.arange(128) / 128
    harmonics = np.column_stack(
        [np.ones_like(theta)]
        + [f(m * theta) for m in range(1, s["harmonics"]) for f in (np.cos, np.sin)]
    )
    rb, zb = coef[0] @ harmonics.T, coef[1] @ harmonics.T
    flux = np.empty(nzeta)
    for k in range(nzeta):
        p1 = np.stack([rb[k], zb[k]], -1)
        p2 = np.roll(p1, -1, axis=0)
        d1, d2 = p1 - line[k], p2 - line[k]
        area = 0.5 * (d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0])
        area *= np.sign(area.sum())
        mid = np.concatenate([line[k] + 0.5 * d1, line[k] + 0.5 * d2, 0.5 * (p1 + p2)])
        b_phi = lines.b(mid[:, 0], np.full(len(mid), planes[k]), mid[:, 1])[1]
        flux[k] = np.sum(area * b_phi.reshape(3, -1).mean(axis=0))
    print(f"  toroidal flux {flux.mean():.6e} Wb, the same through every plane to")
    print(f"  {np.ptp(flux) / abs(flux.mean()):.1e}")

    mpol, ntor = s["mpol"], s["ntor"]
    # the harmonic n of the run has the toroidal mode number n periods / transits
    angle = planes * periods / transits
    island_input = vmecpp.VmecInput(
        nfp=periods,
        toroidal_transits=transits,
        lasym=False,
        mpol=mpol,
        ntor=ntor,
        nzeta=nzeta,
        ns_array=np.array([s["ns"]]),
        ftol_array=np.array([edge["ftol"]]),
        niter_array=np.array([60000]),
        phiedge=float(flux.mean()),
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
        raxis_c=fourier(line[:, :1], angle, np.zeros(1), 1, ntor, 0)[0, ntor:],
        zaxis_s=fourier(line[:, 1:], angle, np.zeros(1), 1, ntor, 1)[0, ntor:],
    )
    interior = vmecpp.run(island_input, max_threads=4, verbose=False)
    w = interior.wout
    axis = axis_curve(w, planes)
    iota = np.asarray(w.iotaf)
    print(
        f"  island interior: {w.itfsq} iterations ({time.time() - start:.0f} s so far)"
    )
    print(
        f"  magnetic axis against the O-point line: largest distance "
        f"{np.hypot(*(axis - line).T).max():.2e} m"
    )
    # the map fixes the turn about the O point per closure up to its sign and whole
    # turns: the branch nearest that of the traced surface
    near = abs(iota_surface) - edge["iota"] * transits
    turns = np.floor(near) + np.array([-1, 0, 1, 2])
    branches = np.concatenate([turns - rotation, turns + rotation])
    turn = branches[np.argmin(np.abs(branches - near))]
    print(
        f"  iota per transit on the axis {iota[0]:.5f}, the {transits}-transit map "
        f"{edge['iota'] + turn / transits:.5f}; on the boundary {iota[-1]:.5f}, "
        f"the traced surface {abs(iota_surface) / transits:.5f}"
    )
    return {"input": island_input, "output": interior, "line": line, "r_o": r_o}


def magnetic_axis(lines, edge):
    """R of the magnetic axis on the midplane of phi = 0: the line through it crosses
    the midplane again at phi = pi."""
    return brentq(
        lambda r: lines.crossings(r, 0.0, 1, 2, 1)[1][0], *edge["axis"], xtol=1e-11
    )


def core_runs(lines, s, edge):
    """Free-boundary runs of the core at CORE_PRESSURE and without pressure."""
    r_axis = magnetic_axis(lines, edge)
    # toroidal flux of the vacuum surface core_offset outboard of the axis, from B_phi
    # on a fan of triangles from the axis
    r, z = lines.crossings(r_axis + edge["core_offset"], 0.0, 1, 1, 150)
    order = np.argsort(np.arctan2(z, r - r_axis))
    p1 = np.stack([r[order], z[order]], -1)
    p2 = np.roll(p1, -1, axis=0)
    d1, d2 = p1 - [r_axis, 0.0], p2 - [r_axis, 0.0]
    area = 0.5 * np.abs(d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0])
    mid = np.concatenate(
        [[r_axis, 0.0] + 0.5 * d1, [r_axis, 0.0] + 0.5 * d2, 0.5 * (p1 + p2)]
    )
    b_phi = lines.b(mid[:, 0], np.zeros(len(mid)), mid[:, 1])[1]
    phiedge = float(np.sum(area * b_phi.reshape(3, -1).mean(axis=0)))

    mpol, ntor, ns = s["core"]
    base = vmecpp.VmecInput.from_file(Path(__file__).parent / "data" / "w7x.json")
    # its cross-section scaled to the toroidal flux of the core
    shape = np.arange(base.mpol)[:, None] > 0
    scale = np.sqrt(phiedge / base.phiedge)
    base = base.model_copy(
        update={
            "rbc": np.where(shape, base.rbc * scale, base.rbc),
            "zbs": np.where(shape, base.zbs * scale, base.zbs),
        }
    )
    if edge["rotation"]:
        # turned with the coils by half a field period, which changes the sign of the
        # harmonics of odd n
        flip = (-1.0) ** np.arange(base.ntor + 1)
        base = base.model_copy(
            update={
                "rbc": base.rbc * np.concatenate([flip[:0:-1], flip]),
                "zbs": base.zbs * np.concatenate([flip[:0:-1], flip]),
                "raxis_c": base.raxis_c * flip,
                "zaxis_s": base.zaxis_s * flip,
            }
        )
    core = base.model_copy(
        update={
            "lfreeb": True,
            "mpol": mpol,
            "ntor": ntor,
            "nzeta": 36,
            "rbc": vmecpp.VmecInput.resize_2d_coeff(base.rbc, mpol, ntor),
            "zbs": vmecpp.VmecInput.resize_2d_coeff(base.zbs, mpol, ntor),
            "raxis_c": np.resize(np.append(base.raxis_c, np.zeros(ntor)), ntor + 1),
            "zaxis_s": np.resize(np.append(base.zaxis_s, np.zeros(ntor)), ntor + 1),
            "ns_array": np.array(ns),
            "ftol_array": np.array([1e-9] * (len(ns) - 1) + [1e-12]),
            "niter_array": np.array([5000] * (len(ns) - 1) + [40000]),
            "phiedge": phiedge,
            "pmass_type": "power_series",
            "am": np.array([1.0, -1.0]),
            "ncurr": 1,
            "curtor": 0.0,
            "ac": np.array([0.0]),
            "nvacskip": 6,
        }
    )
    runs = []
    for p0 in (CORE_PRESSURE, 0.0):
        runs.append(
            vmecpp.run(
                core.model_copy(update={"pres_scale": p0}), lines.b, verbose=False
            )
        )
    w = runs[0].wout
    print(
        f"core: toroidal flux {phiedge:.4f} Wb; at p0 = {CORE_PRESSURE:.0f} Pa <beta> = "
        f"{float(w.betatotal):.4f}, iota {np.asarray(w.iotaf)[0]:.4f} .. "
        f"{np.asarray(w.iotaf)[-1]:.4f}"
    )
    return runs


def casing(output, upsampling):
    """Points and surface currents n x B dS / (4 pi) on the boundary of a free-boundary
    run over the whole torus, its grid refined by trigonometric interpolation."""
    fb = output.threed1_free_boundary
    v = {k: np.asarray(getattr(fb, k)) for k in ("rb", "zb", "brv", "bphiv", "bzv")}
    nz, nt = v["rb"].shape
    if not output.wout.lasym:
        # theta -> 2 pi - theta, zeta -> -zeta: R, B_phi and B_Z even, Z and B_R odd
        full = 2 * (nt - 1)
        ti, zi = full - np.arange(nt, full), (-np.arange(nz)) % nz
        for k, sign in (("rb", 1), ("zb", -1), ("brv", -1), ("bphiv", 1), ("bzv", 1)):
            v[k] = np.concatenate([v[k], sign * v[k][np.ix_(zi, ti)]], axis=1)

    def refine(f, axis, factor):
        n = f.shape[axis]
        return np.fft.irfft(np.fft.rfft(f, axis=axis), n * factor, axis) * factor

    v = {
        k: refine(refine(np.tile(a, (NFP, 1)), 0, upsampling[0]), 1, upsampling[1])
        for k, a in v.items()
    }
    nphi, ntheta = v["rb"].shape
    phi = np.repeat(2.0 * np.pi * np.arange(nphi)[:, None] / nphi, ntheta, axis=1)
    cos, sin = np.cos(phi), np.sin(phi)
    x = np.stack([v["rb"] * cos, v["rb"] * sin, v["zb"]], -1)

    def derivative(f, axis):
        n = f.shape[axis]
        k = np.fft.fftfreq(n, 1.0 / n)
        k[n // 2] = 0.0
        shape = [1] * f.ndim
        shape[axis] = n
        spectrum = 1j * k.reshape(shape) * np.fft.fft(f, axis=axis)
        return np.real(np.fft.ifft(spectrum, axis=axis))

    normal = np.cross(derivative(x, 1), derivative(x, 0))
    if np.sum(x * normal) < 0.0:
        normal = -normal
    b = np.stack(
        [
            v["brv"] * cos - v["bphiv"] * sin,
            v["brv"] * sin + v["bphiv"] * cos,
            v["bzv"],
        ],
        -1,
    )
    weight = (2.0 * np.pi) ** 2 / (nphi * ntheta) / (4.0 * np.pi)
    # the larger of the toroidal and poloidal distances between neighbouring points
    spacing = max(
        np.median(np.linalg.norm(np.diff(x, axis=0), axis=-1)),
        np.median(np.linalg.norm(np.diff(x, axis=1), axis=-1)),
    )
    currents = np.cross(normal, b) * weight
    return x.reshape(-1, 3), currents.reshape(-1, 3), float(spacing)


def casing_field(points, currents, targets, chunk=64):
    """Field at the targets [n, 3] of the surface currents k at the points y, the sum of
    k x (x - y) / |x - y|^3 as (sum w k) x x - sum w (k x y), w = |x - y|^-3."""
    k_cross_y = np.cross(currents, points)
    y2 = np.einsum("sk,sk->s", points, points)
    out = np.empty_like(targets)
    for i in range(0, len(targets), chunk):
        t = targets[i : i + chunk]
        d2 = np.einsum("tk,tk->t", t, t)[:, None] + y2[None, :] - 2.0 * (t @ points.T)
        w = 1.0 / (d2 * np.sqrt(d2))
        out[i : i + chunk] = np.cross(w @ currents, t) - w @ k_cross_y
    return out


class CoreField:
    """Field of the core's pressure-driven currents near the island, the difference of
    its casing fields with and without pressure."""

    def __init__(self, cores, vacuum_island, planes, s):
        w = vacuum_island["output"].wout
        self.nzeta = len(planes)
        self.transits = w.toroidal_transits
        # the length of one field period of the island's domain in phi
        self.period = 2.0 * np.pi * w.toroidal_transits / w.nfp
        m0 = np.asarray(w.xm) == 0
        self.axis_n = np.asarray(w.xn)[m0] / w.toroidal_transits
        self.axis_r, self.axis_z = np.asarray(w.rmnc)[m0, 0], np.asarray(w.zmns)[m0, 0]
        rb, zb = surface_curve(
            w, -1, np.linspace(0, 2 * np.pi, 64, endpoint=False), planes
        )
        axis = axis_curve(w, planes)
        nu, nv = s["grid"]
        self.frames, targets = [], []
        for j, phi in enumerate(planes):
            d = np.stack([rb[j] - axis[j, 0], zb[j] - axis[j, 1]], -1)
            _, vectors = np.linalg.eigh(d.T @ d)
            along, across = d @ vectors[:, 1], d @ vectors[:, 0]
            # the cross-section bends with the flux surfaces about the main axis
            terms = np.column_stack([along**2, along, np.ones_like(along)])
            bend = np.linalg.lstsq(terms, across, rcond=None)[0][0]
            reach = np.array(
                [np.abs(across - bend * along**2).max(), np.abs(along).max()]
            )
            # beyond the island boundary by a fifth and 6 mm, within which the island
            # moves
            extent = 1.2 * reach + 0.006
            u = np.linspace(-extent[1], extent[1], nu)
            v = np.linspace(-extent[0], extent[0], nv)
            uu, vv = (a.ravel() for a in np.meshgrid(u, v, indexing="ij"))
            rz = (
                axis[j]
                + uu[:, None] * vectors[:, 1]
                + (vv + bend * uu**2)[:, None] * vectors[:, 0]
            )
            targets.append(
                np.stack([rz[:, 0] * np.cos(phi), rz[:, 0] * np.sin(phi), rz[:, 1]], -1)
            )
            self.frames.append((vectors[:, 1], vectors[:, 0], u, v, bend))
        targets = np.concatenate(targets)
        # the trapezoidal casing sum needs point spacing well below the distance to the
        # grid, so the boundary grid is refined until it is CASING_MARGIN spacings
        upsampling = np.array(s["upsampling"])
        while True:
            sums = [casing(core, upsampling) for core in cores]
            self.spacing = max(spacing for _, _, spacing in sums)
            self.distance = min(
                float(cKDTree(points).query(targets)[0].min()) for points, _, _ in sums
            )
            if self.distance >= CASING_MARGIN * self.spacing:
                break
            factor = CASING_MARGIN * self.spacing / self.distance
            if factor * upsampling.max() > 4 * max(s["upsampling"]):
                msg = (
                    f"the grid of the core field comes within {self.distance:.3f} m "
                    "of the core boundary; a smaller core_offset or offset_with_core "
                    "keeps them apart"
                )
                raise ValueError(msg)
            upsampling = np.ceil(upsampling * factor).astype(int)
        b = np.zeros_like(targets)
        for (points, currents, _), sign in zip(sums, (1.0, -1.0), strict=True):
            b += sign * casing_field(points, currents, targets)
        phi = np.repeat(planes, nu * nv)
        b_cyl = np.stack(
            [
                b[:, 0] * np.cos(phi) + b[:, 1] * np.sin(phi),
                -b[:, 0] * np.sin(phi) + b[:, 1] * np.cos(phi),
                b[:, 2],
            ],
            -1,
        ).reshape(len(planes), nu, nv, 3)
        self.splines = [
            [RectBivariateSpline(f[2], f[3], b_cyl[j, :, :, k]) for k in range(3)]
            for j, f in enumerate(self.frames)
        ]
        self.magnitude = np.linalg.norm(b, axis=1)

    def _axis(self, phi):
        angle = -self.axis_n[:, None] * phi[None, :]
        return self.axis_r @ np.cos(angle), self.axis_z @ np.sin(angle)

    def __call__(self, r, phi, z):
        r, phi, z = (
            a.ravel()
            for a in np.broadcast_arrays(
                *(np.asarray(a, dtype=float) for a in (r, phi, z))
            )
        )
        # an island that closes after several transits passes each plane that often:
        # take the passage whose axis is nearest
        passes = (
            np.mod(phi, 2.0 * np.pi)[None, :]
            + 2.0 * np.pi * np.arange(self.transits)[:, None]
        )
        distance = np.empty_like(passes)
        for i, p in enumerate(passes):
            axis_r, axis_z = self._axis(p)
            distance[i] = np.hypot(r - axis_r, z - axis_z)
        along = passes[np.argmin(distance, axis=0), np.arange(len(r))]
        axis_r, axis_z = self._axis(along)
        offset = np.stack([r - axis_r, z - axis_z], -1)
        # the planes of one field period of the domain, which repeats along the island
        x = np.mod(along, self.period) * self.nzeta / self.period
        j0 = np.floor(x).astype(int)
        t = x - j0
        out = np.zeros((3, r.size))
        nodes = (-1, 0, 1, 2)
        for a in nodes:
            weight = np.prod([(t - b) / (a - b) for b in nodes if b != a], axis=0)
            plane = (j0 + a) % self.nzeta
            for j in np.unique(plane):
                sel = plane == j
                e1, e2, _, _, bend = self.frames[j]
                u = offset[sel] @ e1
                v = offset[sel] @ e2 - bend * u**2
                for k in range(3):
                    out[k, sel] += weight[sel] * self.splines[j][k].ev(u, v)
        return out[0], out[1], out[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--full", action="store_true", help="resolve the coil ripple")
    parser.add_argument("--edge", choices=list(EDGES), default="5/5")
    args = parser.parse_args()
    s = dict(SETTINGS["full" if args.full else "default"])
    edge = dict(EDGES[args.edge])
    edge["offset"] = edge["offset"][1 if args.full else 0]
    # the harmonic n has the toroidal mode number n periods / transits in phi
    s["ntor"] = math.ceil(s["modes"] * edge["transits"] / edge["periods"])
    nzeta = 2 * s["ntor"] + 8
    # the planes of one field period of the island's domain
    planes = (
        2.0 * np.pi * edge["transits"] * np.arange(nzeta) / (edge["periods"] * nzeta)
    )

    lines = FieldLines(s["cells"], s["degree"], edge["planar"], edge["rotation"])
    vacuum = island(lines, s, planes, "coil field", edge)
    island_coordinates(lines, vacuum, edge, planes)

    if args.full:
        # the slowest interior modes converge slowly, so every pressure run and the
        # vacuum reference start from one initial guess and take equal iterations
        reference = vacuum["input"].model_copy(
            update={
                "ns_array": np.array([13]),
                "ftol_array": np.array([1e-20]),
                "niter_array": np.array([15000]),
                "return_outputs_even_if_not_converged": True,
            }
        )
        axis0 = axis_curve(
            vmecpp.run(reference, max_threads=4, verbose=False).wout, planes
        )
        for p0 in PRESSURES:
            run = vmecpp.run(
                reference.model_copy(update={"pres_scale": p0}),
                max_threads=4,
                verbose=False,
            )
            shift = np.hypot(*(axis_curve(run.wout, planes) - axis0).T)
            beta = 2.0 * MU0 * p0 / float(run.wout.b0) ** 2
            print(
                f"p0 = {p0:.0f} Pa in the island (beta {beta:.1e} on the axis): magnetic "
                f"axis shifted by up to {shift.max():.2e} m"
            )

    # the surface traced with the core, and the vacuum island traced from the same
    # offset as the reference for the displacement
    inner = {**edge, "offset": edge["offset_with_core"]}
    reference = (
        vacuum
        if edge["offset_with_core"] == edge["offset"]
        else island(lines, s, planes, "coil field", inner)
    )
    start = time.time()
    cores = core_runs(lines, s, edge)
    field = CoreField(cores, reference, planes, s)
    print(
        f"field of the pressure-driven core currents around the island: "
        f"{field.magnitude.min():.1e} .. {field.magnitude.max():.1e} T, from casing "
        f"points {field.spacing * 1e3:.0f} mm apart, the nearest "
        f"{field.distance * 1e3:.0f} mm from the grid ({time.time() - start:.0f} s)"
    )
    tracer = PlaneTracer(lines, field, edge["transits"], edge["periods"] * nzeta)
    coupled = island(tracer, s, planes, "with the core", inner)
    shift = coupled["line"] - reference["line"]
    axis_shift = axis_curve(coupled["output"].wout, planes) - axis_curve(
        reference["output"].wout, planes
    )
    print(
        f"island moved by the core: O point by "
        f"{coupled['r_o'] - reference['r_o']:+.2e} m at phi = 0, O-point line by up to "
        f"{np.hypot(*shift.T).max():.2e} m, magnetic axis of the interior by up to "
        f"{np.hypot(*axis_shift.T).max():.2e} m"
    )


if __name__ == "__main__":
    main()
