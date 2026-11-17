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

The default resolution keeps the run short. With --full, the O-point line and the
boundary are resolved to the toroidal harmonics of the ripple of the 50 coils, n = 50.
"""

import argparse
import time

import numpy as np
from scipy.optimize import brentq, minimize_scalar
from simsopt.field import BiotSavart, InterpolatedField, coils_via_symmetries
from simsopt.field.tracing import compute_fieldlines

import vmecpp

NFP = 5
MU0 = 4.0e-7 * np.pi
# grid cells (R, phi over a period, Z) and polynomial degree of the interpolated field,
# and the resolution of the island run
SETTINGS = {
    "default": {"cells": (27, 24, 18), "degree": 3, "mpol": 5, "ntor": 30, "ns": 13},
    "full": {"cells": (54, 48, 36), "degree": 4, "mpol": 6, "ntor": 60, "ns": 25},
}
SURFACE_OFFSET = 0.006
SURFACE_TURNS = 150
HARMONICS = 8
# island pressures p0 in Pa
PRESSURES = (100.0, 400.0)


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

    def b(self, r, phi, z):
        """Cylindrical components of B at the points (r, phi, z)."""
        cos, sin = np.cos(phi), np.sin(phi)
        self.field.set_points(np.ascontiguousarray(np.stack([r * cos, r * sin, z], -1)))
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


def axis_curve(wout, phi):
    angle = -np.asarray(wout.xn)[:, None] * phi[None, :]
    return np.stack(
        [
            np.asarray(wout.rmnc)[:, 0] @ np.cos(angle),
            np.asarray(wout.zmns)[:, 0] @ np.sin(angle),
        ],
        axis=1,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--full", action="store_true", help="resolve the coil ripple")
    args = parser.parse_args()
    s = SETTINGS["full" if args.full else "default"]
    nzeta = 2 * s["ntor"] + 8
    planes = 2.0 * np.pi * np.arange(nzeta) / nzeta

    start = time.time()
    lines = FieldLines(s["cells"], s["degree"])
    r_o, rotation = o_point(lines)
    print(f"O point R = {r_o:.6f} m on the midplane at phi = 0; the one-turn map turns")
    print(
        f"the lines near it by {rotation:.5f} of a turn ({time.time() - start:.0f} s)"
    )

    # the O-point line, one turn, and the island surface; the line runs toward
    # decreasing phi, so crossing i (from 0) lies (i + 1) / nzeta turns along it
    plane, r, z = lines.crossings(r_o, 0.0, 1, planes)
    line = np.empty((nzeta, 2))
    line[plane] = np.stack([r, z], -1)
    plane, r, z = lines.crossings(r_o + SURFACE_OFFSET, 0.0, SURFACE_TURNS, planes)
    if np.any(plane != (nzeta - 1 - np.arange(len(plane))) % nzeta):
        msg = "the field line does not cross the planes in the expected order"
        raise RuntimeError(msg)
    turns_along = (np.arange(len(plane)) + 1) / nzeta
    coef, iota_surface, residual = surface_fit(plane, r, z, turns_along, line)
    print(
        f"surface traced over {SURFACE_TURNS} turns: iota {iota_surface:+.6f}, fit in"
    )
    print(f"{HARMONICS - 1} poloidal harmonics within {residual:.1e} m")

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
    print(f"toroidal flux {flux.mean():.6e} Wb, the same through every plane to")
    print(f"{np.ptp(flux) / abs(flux.mean()):.1e}")

    mpol, ntor = s["mpol"], s["ntor"]
    island = vmecpp.VmecInput(
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
    start = time.time()
    vacuum = vmecpp.run(island, max_threads=4, verbose=False)
    w = vacuum.wout
    axis = axis_curve(w, planes)
    iota = np.asarray(w.iotaf)
    print(f"island interior: {w.itfsq} iterations, {time.time() - start:.0f} s")
    print(
        f"  magnetic axis against the O-point line: largest distance "
        f"{np.hypot(*(axis - line).T).max():.2e} m"
    )
    print(
        f"  iota on the axis {iota[0]:.5f}, the one-turn map {1.0 + rotation:.5f}; "
        f"on the boundary {iota[-1]:.5f}, the traced surface {abs(iota_surface):.5f}"
    )

    if not args.full:
        return
    # The slowest modes of the interior converge slowly: between ftol 1e-11 and 1e-15
    # the vacuum axis still moves by 0.07 mm. Every pressure run and the vacuum
    # reference therefore start from the same initial guess and take the same number of
    # iterations, which brings the residuals to about 1e-15.
    reference = island.model_copy(
        update={
            "ns_array": np.array([13]),
            "ftol_array": np.array([1e-20]),
            "niter_array": np.array([15000]),
            "return_outputs_even_if_not_converged": True,
        }
    )
    axis0 = axis_curve(vmecpp.run(reference, max_threads=4, verbose=False).wout, planes)
    for p0 in PRESSURES:
        run = vmecpp.run(
            reference.model_copy(update={"pres_scale": p0}),
            max_threads=4,
            verbose=False,
        )
        shift = np.hypot(*(axis_curve(run.wout, planes) - axis0).T)
        beta = 2.0 * MU0 * p0 / float(run.wout.b0) ** 2
        print(
            f"p0 = {p0:.0f} Pa (beta {beta:.1e} on the axis): magnetic axis shifted "
            f"by up to {shift.max():.2e} m"
        )


if __name__ == "__main__":
    main()
