# SPDX-FileCopyrightText: 2026-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""The interior of a magnetic island as an equilibrium of its own.

VMEC++ describes a plasma by flux surfaces nested about one magnetic axis, so an island
chain enters a run only as a distortion of those surfaces. Inside one island, though,
the field lines lie on surfaces nested about the island's O-point line: the island
interior is such a region in its own right, with the O-point line as its magnetic axis
and an island flux surface as its boundary. This example computes the interior of one
island of the 5/5 chain at the edge of the W7-X standard configuration, which
J. C. Schmitt et al., Plasma Phys. Control. Fusion 64, 055022 (2022), solve in vacuum as
a fixed-boundary VMEC run.

1. The vacuum field of the W7-X coils of SIMSOPT, interpolated over one field period.
2. The O point on the outboard midplane of the bean-shaped plane phi = 0. The field is
   stellarator symmetric, so the line through a point of that midplane closes after one
   toroidal turn exactly when it crosses the midplane again at phi = pi, a root in one
   variable. SIMSOPT places a crossing on its line but only near the plane, so each
   crossing is carried along the line onto the plane.
3. An island flux surface, traced from a point 6 mm outboard of the O point (the island
   reaches about 10 mm to either side of it there). The line turns about the O-point
   line iota times per toroidal turn, so its crossing after tau turns has the
   straight-field-line angle 2 pi iota tau on that surface, and R and Z against that
   angle need only a few poloidal harmonics in each plane.
4. The interior, solved with that surface as fixed boundary: nfp = 1, since an island
   closes after one toroidal turn and the five islands are images of each other under
   the field-period rotation, and stellarator symmetric about the bean plane, since the
   island and the straight-field-line angle are. Its magnetic axis is compared with the
   traced O-point line and its rotational transform with tracing: on the axis, the
   rotation angle of the one-turn map at the O point; on the boundary, the iota of the
   traced surface. Both count a full turn of the island's orientation, which follows the
   poloidal direction about the main axis, besides the slow turn about the O point.
5. With --full, pressure p0 (1 - s) in the island, which shifts its magnetic axis.
6. The island in the field of a finite-beta core plasma inside the chain. The core is a
   free-boundary run in the coil field (nfp = 5, stellarator symmetric) with the
   toroidal flux of the vacuum flux surface 0.22 m outboard of the magnetic axis, at
   pressure p0 (1 - s) and without pressure. Outside it, its currents make the field of
   the surface current n x B / mu_0 on its boundary, B the vacuum field there (the
   virtual casing principle). A free-boundary boundary is a flux surface of the coil
   field only to the resolution of its run, and the run without pressure carries that
   error in its casing field as well; the difference of the two keeps the field of the
   pressure-driven currents. Steps 2 to 4 are repeated in the coil field plus that
   field, traced by RK4 in the toroidal angle.

The default resolution keeps the run short. With --full, the O-point line and the
boundary are resolved to the toroidal harmonics of the ripple of the 50 coils, n = 50.
"""

import argparse
import threading
import time
from pathlib import Path

import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import brentq, minimize_scalar
from simsopt.field import BiotSavart, InterpolatedField, coils_via_symmetries
from simsopt.field.tracing import compute_fieldlines

import vmecpp

NFP = 5
MU0 = 4.0e-7 * np.pi
# grid cells (R, phi over a period, Z) and polynomial degree of the interpolated field,
# the resolution of the island run, the resolution of the core run, the upsampling of
# its boundary grid for the casing sum (toroidal, poloidal), the points per plane of
# the grid of its field around the island, and the turns of the surface traced in it
SETTINGS = {
    "default": {
        "cells": (27, 24, 18),
        "degree": 3,
        "mpol": 5,
        "ntor": 30,
        "ns": 13,
        "core": (6, 6, [13]),
        "upsampling": (12, 8),
        "grid": (11, 7),
        "core_turns": 60,
    },
    "full": {
        "cells": (54, 48, 36),
        "degree": 4,
        "mpol": 6,
        "ntor": 60,
        "ns": 25,
        "core": (8, 8, [13, 25]),
        "upsampling": (12, 8),
        "grid": (13, 9),
        "core_turns": 120,
    },
}
SURFACE_OFFSET = 0.006
SURFACE_TURNS = 150
HARMONICS = 8
# island pressures p0 in Pa
PRESSURES = (100.0, 400.0)
CORE_OFFSET = 0.22
CORE_PRESSURE = 5.0e4
RK4_STEPS = 8  # per plane of the island run


class FieldLines:
    """Field lines of the vacuum field, crossing given toroidal planes."""

    def __init__(self, cells, degree):
        try:  # SIMSOPT 1.11 and later
            from simsopt.configs import get_data  # noqa: PLC0415

            curves, currents = get_data("w7x")[:2]
        except ImportError:
            from simsopt.configs import get_w7x_data  # noqa: PLC0415

            curves, currents, _ = get_w7x_data()
        coils = coils_via_symmetries(curves, currents, NFP, True)
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

    def crossings(self, r0, z0, turns, planes):
        """Crossings of the line through (r0, 0, z0) with the planes over the given
        number of turns, in the order the line reaches them: (plane index, R, Z)."""
        # SIMSOPT integrates dx/dt = B, a turn of about 2 pi R taking t = 2 pi R / |B|
        _, hits = compute_fieldlines(
            self.field,
            np.array([r0]),
            np.array([z0]),
            tmax=(turns + 0.5) * 2.0 * np.pi * 6.4 / 2.3,
            tol=1e-11,
            phis=list(planes),
        )
        h = np.asarray(hits[0])
        h = h[(h[:, 0] > 0.0) & (h[:, 1] >= 0)]
        count = int(np.ceil(turns * len(planes)))
        plane = h[:count, 1].astype(int)
        x, y, z = h[:count, 2], h[:count, 3], h[:count, 4]
        r, z = self.onto(np.hypot(x, y), np.arctan2(y, x), z, np.asarray(planes)[plane])
        return plane, r, z


class PlaneTracer:
    """Field lines of the vacuum field plus another field, by RK4 in the toroidal angle
    with steps that fall on the planes of the island run, the angle pi and 0."""

    def __init__(self, lines, extra, nzeta):
        self.lines, self.extra = lines, extra
        self.steps = nzeta * RK4_STEPS  # per turn
        b_phi = lines.b(np.array([6.0]), np.zeros(1), np.zeros(1))[1][0]
        self.sign = -1 if b_phi < 0.0 else 1

    def b(self, r, phi, z):
        b, c = self.lines.b(r, phi, z), self.extra(r, phi, z)
        return b[0] + c[0], b[1] + c[1], b[2] + c[2]

    def crossings(self, r0, z0, turns, planes):
        step_of = {
            round(float(np.mod(p, 2.0 * np.pi)) * self.steps / (2.0 * np.pi))
            % self.steps: k
            for k, p in enumerate(planes)
        }
        count = int(np.ceil(turns * len(planes)))
        h = self.sign * 2.0 * np.pi / self.steps
        r, z = np.array([float(r0)]), np.array([float(z0)])

        def slope(r, phi, z):
            b_r, b_phi, b_z = self.b(r, np.full(1, phi), z)
            return r * b_r / b_phi, r * b_z / b_phi

        out, i = [], 0
        while len(out) < count:
            phi = i * h
            k1 = slope(r, phi, z)
            k2 = slope(r + 0.5 * h * k1[0], phi + 0.5 * h, z + 0.5 * h * k1[1])
            k3 = slope(r + 0.5 * h * k2[0], phi + 0.5 * h, z + 0.5 * h * k2[1])
            k4 = slope(r + h * k3[0], phi + h, z + h * k3[1])
            r = r + h / 6.0 * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0])
            z = z + h / 6.0 * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1])
            i += 1
            k = step_of.get((self.sign * i) % self.steps)
            if k is not None:
                out.append((k, r[0], z[0]))
        plane, rr, zz = (np.array(v) for v in zip(*out, strict=True))
        return plane.astype(int), rr, zz


def o_point(lines):
    """R of the O point on the outboard midplane of the bean plane, from the line that
    crosses the midplane again at phi = pi, and the rotation angle of the one-turn map
    there per turn."""

    def z_at_half_turn(r):
        return lines.crossings(r, 0.0, 1, [np.pi])[2][0]

    r_o = brentq(z_at_half_turn, 6.20, 6.27, xtol=1e-11)

    def one_turn(r, z):
        _, rr, zz = lines.crossings(r, z, 1, [0.0])
        return np.array([rr[0], zz[0]])

    eps = 1e-5
    jacobian = np.column_stack(
        [
            (one_turn(r_o + eps, 0.0) - one_turn(r_o - eps, 0.0)) / (2 * eps),
            (one_turn(r_o, eps) - one_turn(r_o, -eps)) / (2 * eps),
        ]
    )
    rotation = np.arccos(0.5 * np.trace(jacobian)) / (2.0 * np.pi)
    return r_o, rotation


def surface_fit(plane, r, z, turns_along, line):
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

    # first estimate from the angle about the O-point line, then the best of a scan
    # around it, refined
    angle = np.unwrap(np.arctan2(z - line[plane, 1], r - line[plane, 0]))
    iota0 = (angle[-1] - angle[0]) / (2.0 * np.pi * (turns_along[-1] - turns_along[0]))
    scan = iota0 + np.linspace(-2e-3, 2e-3, 41)
    best = int(np.argmin([fit(x, 4)[1] for x in scan]))
    step = scan[1] - scan[0]
    iota = minimize_scalar(
        lambda x: fit(x, 4)[1],
        bounds=(scan[best] - step, scan[best] + step),
        method="bounded",
        options={"xatol": 1e-10},
    ).x
    coef, worst = fit(iota, HARMONICS)
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
    angle = (
        np.asarray(wout.xm)[:, None, None] * theta[None, None, :]
        - np.asarray(wout.xn)[:, None, None] * phi[None, :, None]
    )
    return (
        np.einsum("k,kpt->pt", np.asarray(wout.rmnc)[:, js], np.cos(angle)),
        np.einsum("k,kpt->pt", np.asarray(wout.zmns)[:, js], np.sin(angle)),
    )


def axis_curve(wout, phi):
    r, z = surface_curve(wout, 0, np.zeros(1), phi)
    return np.stack([r[:, 0], z[:, 0]], axis=1)


def island(lines, s, planes, label):
    """Steps 2 to 4 in the field the line tracer follows."""
    nzeta = len(planes)
    start = time.time()
    r_o, rotation = o_point(lines)
    # the O-point line, one turn, and the island surface; the line runs toward
    # decreasing phi, so crossing i (from 0) lies (i + 1) / nzeta turns along it
    plane, r, z = lines.crossings(r_o, 0.0, 1, planes)
    line = np.empty((nzeta, 2))
    line[plane] = np.stack([r, z], -1)
    turns = s["core_turns"] if isinstance(lines, PlaneTracer) else SURFACE_TURNS
    plane, r, z = lines.crossings(r_o + SURFACE_OFFSET, 0.0, turns, planes)
    if np.any(plane != (nzeta - 1 - np.arange(len(plane))) % nzeta):
        msg = "the field line does not cross the planes in the expected order"
        raise RuntimeError(msg)
    turns_along = (np.arange(len(plane)) + 1) / nzeta
    coef, iota_surface, residual = surface_fit(plane, r, z, turns_along, line)
    print(f"{label}: O point R = {r_o:.6f} m on the midplane at phi = 0, where the")
    print(
        f"  one-turn map turns the lines near it by {rotation:.5f} of a turn; surface"
    )
    print(f"  traced over {turns} turns, iota {iota_surface:+.6f}, fit in")
    print(f"  {HARMONICS - 1} poloidal harmonics within {residual:.1e} m")

    # the boundary on a uniform grid of the straight-field-line angle, and the toroidal
    # flux through each cross-section from B_phi on a fan of triangles from the O point
    theta = 2.0 * np.pi * np.arange(128) / 128
    harmonics = np.column_stack(
        [np.ones_like(theta)]
        + [f(m * theta) for m in range(1, HARMONICS) for f in (np.cos, np.sin)]
    )
    rb, zb = coef[0] @ harmonics.T, coef[1] @ harmonics.T
    flux = np.empty(nzeta)
    for k in range(nzeta):
        p1 = np.stack([rb[k], zb[k]], -1)
        p2 = np.roll(p1, -1, axis=0)
        d1, d2 = p1 - line[k], p2 - line[k]
        area = 0.5 * np.abs(d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0])
        mid = np.concatenate([line[k] + 0.5 * d1, line[k] + 0.5 * d2, 0.5 * (p1 + p2)])
        b_phi = lines.b(mid[:, 0], np.full(len(mid), planes[k]), mid[:, 1])[1]
        flux[k] = np.sum(area * b_phi.reshape(3, -1).mean(axis=0))
    print(f"  toroidal flux {flux.mean():.6e} Wb, the same through every plane to")
    print(f"  {np.ptp(flux) / abs(flux.mean()):.1e}")

    mpol, ntor = s["mpol"], s["ntor"]
    island_input = vmecpp.VmecInput(
        nfp=1,
        lasym=False,
        mpol=mpol,
        ntor=ntor,
        nzeta=nzeta,
        ns_array=np.array([s["ns"]]),
        ftol_array=np.array([1e-13]),
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
        rbc=fourier(rb, planes, theta, mpol, ntor, 0),
        zbs=fourier(zb, planes, theta, mpol, ntor, 1),
        # VMEC++ takes the axis as R = sum raxis_c cos(n phi), Z = -sum zaxis_s sin(n phi)
        raxis_c=fourier(line[:, :1], planes, np.zeros(1), 1, ntor, 0)[0, ntor:],
        zaxis_s=fourier(line[:, 1:], planes, np.zeros(1), 1, ntor, 1)[0, ntor:],
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
    print(
        f"  iota on the axis {iota[0]:.5f}, the one-turn map {1.0 + rotation:.5f}; "
        f"on the boundary {iota[-1]:.5f}, the traced surface {abs(iota_surface):.5f}"
    )
    return {"input": island_input, "output": interior, "line": line, "r_o": r_o}


def magnetic_axis(lines):
    """R of the magnetic axis on the midplane of the bean plane: the line through it
    crosses the midplane again at phi = pi."""
    return brentq(
        lambda r: lines.crossings(r, 0.0, 1, [np.pi])[2][0], 5.90, 6.00, xtol=1e-11
    )


def core_runs(lines, s):
    """Free-boundary runs of the core at CORE_PRESSURE and without pressure."""
    r_axis = magnetic_axis(lines)
    # toroidal flux of the vacuum surface CORE_OFFSET outboard of the axis, from B_phi
    # on a fan of triangles from the axis
    _, r, z = lines.crossings(r_axis + CORE_OFFSET, 0.0, 150, [0.0])
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
    """Field at the targets [n, 3] of the surface currents at the points, the sum of.

    k x (x - y) / |x - y|^3 written as (sum w k) x x - sum w (k x y) with
    w = |x - y|^-3, and the smallest distance from a target to a point.
    """
    k_cross_y = np.cross(currents, points)
    y2 = np.einsum("sk,sk->s", points, points)
    out = np.empty_like(targets)
    nearest = np.inf
    for i in range(0, len(targets), chunk):
        t = targets[i : i + chunk]
        d2 = np.einsum("tk,tk->t", t, t)[:, None] + y2[None, :] - 2.0 * (t @ points.T)
        nearest = min(nearest, float(d2.min()))
        w = 1.0 / (d2 * np.sqrt(d2))
        out[i : i + chunk] = np.cross(w @ currents, t) - w @ k_cross_y
    return out, np.sqrt(nearest)


class CoreField:
    """Field of the pressure-driven currents of the core near the island: the
    difference of the casing fields of the core with and without pressure, on a grid
    centred on the vacuum island axis along the principal axes of the island boundary
    in each plane, interpolated by bicubic splines in the plane and by cubic Lagrange
    weights between planes at the same offset from the axis."""

    def __init__(self, cores, vacuum_island, planes, s):
        w = vacuum_island["output"].wout
        self.nzeta = len(planes)
        m0 = np.asarray(w.xm) == 0
        self.axis_n = np.asarray(w.xn)[m0]
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
            reach = np.abs(d @ vectors).max(axis=0)
            # beyond the island boundary by a fifth and 6 mm: the island moves within
            # this, and the grid stays as far from the core boundary as the casing
            # sum's quadrature needs
            extent = 1.2 * reach + 0.006
            u = np.linspace(-extent[1], extent[1], nu)
            v = np.linspace(-extent[0], extent[0], nv)
            uu, vv = np.meshgrid(u, v, indexing="ij")
            rz = (
                axis[j]
                + uu.ravel()[:, None] * vectors[:, 1]
                + vv.ravel()[:, None] * vectors[:, 0]
            )
            targets.append(
                np.stack([rz[:, 0] * np.cos(phi), rz[:, 0] * np.sin(phi), rz[:, 1]], -1)
            )
            self.frames.append((vectors[:, 1], vectors[:, 0], u, v))
        targets = np.concatenate(targets)
        b = np.zeros_like(targets)
        # the trapezoidal rule of the casing sum is accurate where the spacing of its
        # points is well below their distance from the grid
        self.distance, self.spacing = np.inf, 0.0
        for core, sign in zip(cores, (1.0, -1.0), strict=True):
            points, currents, spacing = casing(core, s["upsampling"])
            field, distance = casing_field(points, currents, targets)
            b += sign * field
            self.distance = min(self.distance, distance)
            self.spacing = max(self.spacing, spacing)
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

    def __call__(self, r, phi, z):
        r, phi, z = np.broadcast_arrays(
            *(np.asarray(a, dtype=float) for a in (r, phi, z))
        )
        x = np.mod(phi, 2.0 * np.pi) * self.nzeta / (2.0 * np.pi)
        j0 = np.floor(x).astype(int)
        t = x - j0
        angle = -self.axis_n[:, None] * phi.ravel()[None, :]
        offset = np.stack(
            [
                r.ravel() - self.axis_r @ np.cos(angle),
                z.ravel() - self.axis_z @ np.sin(angle),
            ],
            -1,
        )
        j0, t = j0.ravel(), t.ravel()
        out = np.zeros((3, r.size))
        nodes = (-1, 0, 1, 2)
        for a in nodes:
            weight = np.prod([(t - b) / (a - b) for b in nodes if b != a], axis=0)
            plane = (j0 + a) % self.nzeta
            for j in np.unique(plane):
                sel = plane == j
                e1, e2 = self.frames[j][0], self.frames[j][1]
                u, v = offset[sel] @ e1, offset[sel] @ e2
                for k in range(3):
                    out[k, sel] += weight[sel] * self.splines[j][k].ev(u, v)
        return out[0], out[1], out[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--full", action="store_true", help="resolve the coil ripple")
    args = parser.parse_args()
    s = SETTINGS["full" if args.full else "default"]
    nzeta = 2 * s["ntor"] + 8
    planes = 2.0 * np.pi * np.arange(nzeta) / nzeta

    lines = FieldLines(s["cells"], s["degree"])
    vacuum = island(lines, s, planes, "coil field")

    if args.full:
        # The slowest modes of the interior converge slowly: between ftol 1e-11 and
        # 1e-15 the vacuum axis still moves by 0.07 mm. Every pressure run and the
        # vacuum reference therefore start from the same initial guess and take the
        # same number of iterations, which brings the residuals to about 1e-15.
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

    start = time.time()
    cores = core_runs(lines, s)
    field = CoreField(cores, vacuum, planes, s)
    print(
        f"field of the pressure-driven core currents around the island: "
        f"{field.magnitude.min():.1e} .. {field.magnitude.max():.1e} T, from casing "
        f"points {field.spacing * 1e3:.0f} mm apart, the nearest "
        f"{field.distance * 1e3:.0f} mm from the grid ({time.time() - start:.0f} s)"
    )
    coupled = island(PlaneTracer(lines, field, nzeta), s, planes, "with the core")
    shift = coupled["line"] - vacuum["line"]
    axis_shift = axis_curve(coupled["output"].wout, planes) - axis_curve(
        vacuum["output"].wout, planes
    )
    print(
        f"island moved by the core: O point by {coupled['r_o'] - vacuum['r_o']:+.2e} m at "
        f"phi = 0, O-point line by up to {np.hypot(*shift.T).max():.2e} m, magnetic axis "
        f"of the interior by up to {np.hypot(*axis_shift.T).max():.2e} m"
    )


if __name__ == "__main__":
    main()
