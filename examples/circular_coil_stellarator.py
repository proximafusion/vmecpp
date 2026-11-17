# SPDX-FileCopyrightText: 2026-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""A quasi-isodynamic stellarator of three field periods wound from 42 identical
circular coils with one current, solved with VMEC++.

The coils of half a field period are given. Stellarator symmetry and the field periods
give the others, and SIMSOPT gives their field. VMEC++ solves the interior of the flux
surface traced 30 cm outboard of the magnetic axis, as in ``knotted_stellarator.py``.
Hot restarts raise the pressure. The effective ripple of Nemov et al., Phys. Plasmas 6,
4622 (1999), and the poloidal closure of the contours of B come from the wout. W7-X,
traced and solved the same way, is the reference. ``--free-boundary`` also solves the
plasma in the field of the coils.
"""

import argparse
import dataclasses
import threading

import numpy as np
from island_interior import axis_curve
from knotted_stellarator import (
    FieldLines,
    Knot,
    flux_surface,
    interior_input,
    magnetic_axis,
    toroidal_flux,
)
from simsopt.field import BiotSavart, Coil, Current, coils_via_symmetries
from simsopt.geo import CurveXYZFourier

import vmecpp

NFP = 3
RADIUS = 0.9377896443164089  # coil radius in m
CURRENT = 1.0e6  # coil current in A
# Centres in m of the coils from phi = 0 to the middle of a field period. The first and
# the last coil are their own images under stellarator symmetry.
CENTRES = np.array(
    [
        [6.276680420603, 0.0, 0.0],
        [6.090668525339, 0.320300080515, -0.495529500718],
        [5.684280413785, 0.564832446215, -0.818403440426],
        [5.159930926567, 0.748644647913, -0.922334786404],
        [4.640202457094, 1.005462973747, -0.907093122598],
        [3.890029448045, 1.434033797722, -0.691744113792],
        [2.641524098782, 2.071612546433, -0.360565901180],
        [1.552216589063, 2.688517996609, 0.0],
    ]
)
# Unit normals of the same coils. A positive current drives the field at a centre along
# the normal.
NORMALS = np.array(
    [
        [0.0, 0.533642846213, -0.845709945954],
        [-0.487224608606, 0.465340751107, -0.738965605511],
        [-0.821227988964, 0.279711625452, -0.497338915358],
        [-0.951775312807, 0.290013512136, -0.100079551908],
        [-0.780132587343, 0.592540035920, 0.200722325607],
        [-0.770055402785, 0.549247937393, 0.324563368099],
        [-0.829105600776, 0.493118061372, 0.263473870262],
        [-0.835825681311, 0.482564182100, 0.261777081914],
    ]
)
# The interior of the design. The coil centres lie near R = r_0 + r_{knot} \cos(3 z).
# The magnetic axis crosses phi = 0, Z = 0 within the bracket axis. The boundary is
# traced from offset outboard of the axis.
KNOT = Knot(
    p=1,
    q=NFP,
    r0=4.71,
    r_knot=1.53,
    coils=42,
    radius=RADIUS,
    axis=(5.95, 6.07),
    planes=96,
    offset=0.3,
    mpol=12,
    ntor=24,
)
# The interior of W7-X in the standard configuration, inside its 5/5 islands.
W7X = dataclasses.replace(
    KNOT, q=5, coils=50, axis=(5.90, 6.00), planes=48, offset=0.22, ntor=12
)
PRESSURES = (8000.0, 16000.0, 24000.0)  # p_0 in Pa of the profile p_0 (1 - s)


def circle(centre, normal):
    """A coil of radius RADIUS about centre.

    A positive current turns about normal.
    """
    first = np.cross(normal, [0.3, 0.5, 0.8])
    first /= np.linalg.norm(first)
    second = np.cross(normal, first)
    curve = CurveXYZFourier(64, 1)
    # c_0 + s_1 \sin(2 \pi t) + c_1 \cos(2 \pi t) in each coordinate
    curve.x = np.concatenate(
        [[centre[k], RADIUS * second[k], RADIUS * first[k]] for k in range(3)]
    )
    return Coil(curve, Current(CURRENT))


def coils():
    """The 42 coils of the design."""
    # Stellarator symmetry maps (x, y, z) to (x, -y, -z) and n to (-n_x, n_y, n_z).
    flip = np.array([1.0, -1.0, -1.0])
    centres = np.concatenate([CENTRES, flip * CENTRES[1:-1]])
    normals = np.concatenate([NORMALS, -flip * NORMALS[1:-1]])
    out = []
    for k in range(NFP):
        cos, sin = np.cos(2.0 * np.pi * k / NFP), np.sin(2.0 * np.pi * k / NFP)
        turn = np.array([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]])
        out += [
            circle(turn @ c, turn @ n) for c, n in zip(centres, normals, strict=True)
        ]
    return out


def w7x_coils():
    """The 50 coils of W7-X in the standard configuration."""
    try:  # SIMSOPT 1.11 and later
        from simsopt.configs import get_data  # noqa: PLC0415

        curves, currents = get_data("w7x")[:2]
    except ImportError:
        from simsopt.configs import get_w7x_data  # noqa: PLC0415

        curves, currents, _ = get_w7x_data()
    return coils_via_symmetries(curves, currents, 5, True)


def interior(field, knot):
    """The traced magnetic axis and boundary, their rotational transform and the input
    of VMEC++ for the interior of the boundary."""
    lines = FieldLines(field, knot.period)
    r_axis, axis, iota_axis = magnetic_axis(knot, lines)
    planes = np.arange(knot.planes) * knot.period / knot.planes
    rb, zb, iota_surface, misfit = flux_surface(knot, lines, r_axis)
    flux = toroidal_flux(lines, rb, zb, axis, planes)
    return {
        "r_axis": r_axis,
        "axis": axis,
        "iota_axis": iota_axis,
        "planes": planes,
        "iota_surface": iota_surface,
        "misfit": misfit,
        "input": interior_input(knot, rb, zb, axis, flux, planes),
    }


def effective_ripple(strength, size, drift, weight, major_radius, levels=4000):
    """The effective ripple of a field line.

    The inputs are B, |\\nabla \\psi|, the radial drift v_d and the ds / B of a step at
    the samples of the line. B_0 is the mean of B along the line. b = B / B_0 is linear
    between the samples. A well j of the line at the level b' has

        H_j = \\int ds / B \\sqrt{b' - b} (4 / b - 1 / b') v_d,
        I_j = \\int ds / B \\sqrt{1 - b / b'}.

    The effective ripple is

        (\\pi R_0^2 / (8 \\sqrt{2}) \\int ds / B / (\\int ds |\\nabla \\psi| / B)^2
        \\int db' \\sum_j H_j^2 / (b'^2 I_j))^{2/3}.

    The integral over b' is a midpoint sum over levels intervals. A well cut by an end
    of the line is left out.
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
    change = -np.diff(eta)
    flat = np.abs(change) < 1.0e-14
    width = (eta.max() - eta.min()) / levels
    total = 0.0
    for level in eta.min() + width * (np.arange(levels) + 0.5):
        below = np.maximum(level - eta, 0.0)
        # the mean of \sqrt{\max(b' - b, 0)} over each step
        root = np.where(
            flat,
            np.sqrt(below[:-1]),
            2.0 / 3.0 * np.diff(below**1.5) / np.where(flat, 1.0, change),
        )
        wet = root > 0.0
        starts = np.flatnonzero(wet[1:] & ~wet[:-1]) + 1
        stops = np.flatnonzero(~wet[1:] & wet[:-1]) + 1
        if wet[0]:
            stops = stops[1:]
        count = min(len(starts), len(stops))
        if count == 0:
            continue
        factor = weight * (4.0 / eta - 1.0 / level) * drift
        h = np.concatenate([[0.0], np.cumsum(root * 0.5 * (factor[:-1] + factor[1:]))])
        i = np.concatenate([[0.0], np.cumsum(root * step_weight)])
        a, b = starts[:count], stops[:count]
        well_h, well_i = h[b] - h[a], (i[b] - i[a]) / np.sqrt(level)
        total += width * np.sum(well_h**2 / (level**2 * well_i))
    return float((scale * total) ** (2.0 / 3.0))


def wout_ripple(wout, js, periods=120, steps=96):
    """The effective ripple of the half-grid surface js of a wout.

    The field line starts at theta = 0, phi = 0 and runs over periods field periods. The
    major radius is the length of the magnetic axis over 2 pi.
    """
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
    # (B \times \nabla B) \cdot \nabla s / B^2
    #     = (B_\theta \partial_\phi B - B_\phi \partial_\theta B) / (\sqrt{g} B^2)
    drift = (
        (np.asarray(wout.bsubumnc)[:, js] @ cos) * ((n * bmnc) @ sin)
        + (np.asarray(wout.bsubvmnc)[:, js] @ cos) * ((m * bmnc) @ sin)
    ) / (jacobian * strength**2)
    # |\nabla s| from R and Z on the full-grid surfaces around js
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


def closure(strength):
    """The share of the range of B on a surface whose contours close poloidally.

    strength is B on a grid [theta, phi] over a field period. The contours close
    poloidally between the largest minimum of B along the lines of constant theta and
    the smallest maximum. The share is 1 for a quasi-isodynamic field.
    """
    lowest_ridge = strength.max(axis=1).min()
    highest_valley = strength.min(axis=1).max()
    return float((lowest_ridge - highest_valley) / (strength.max() - strength.min()))


def wout_closure(wout, js, angles=128):
    """:func:`closure` of the half-grid surface js of a wout."""
    transits = int(wout.toroidal_transits)
    theta = 2.0 * np.pi * np.arange(angles) / angles
    phi = 2.0 * np.pi * transits / int(wout.nfp) * np.arange(angles) / angles
    m = np.asarray(wout.xm_nyq, dtype=float)
    n = np.asarray(wout.xn_nyq, dtype=float) / transits
    bmnc = np.asarray(wout.bmnc)[:, js]
    # \cos(m \theta - n \phi) = \cos m\theta \cos n\phi + \sin m\theta \sin n\phi
    poloidal, toroidal = m[None, :] * theta[:, None], n[None, :] * phi[:, None]
    strength = (np.cos(poloidal) * bmnc) @ np.cos(toroidal).T + (
        np.sin(poloidal) * bmnc
    ) @ np.sin(toroidal).T
    return closure(strength)


def measures(wout):
    """The effective ripple and :func:`closure` at s = 1/4, at s = 1/2 and on the
    boundary of a wout."""
    ns = int(wout.ns)
    surfaces = (ns // 4, ns // 2, ns - 1)
    ripple = [wout_ripple(wout, js) for js in surfaces]
    return ripple, [wout_closure(wout, js) for js in surfaces]


def describe(ripple, closed):
    """:func:`measures` as text."""
    return (
        "effective ripple "
        + ", ".join(f"{100 * value:.2f}%" for value in ripple)
        + ", contours of B closing poloidally over "
        + ", ".join(f"{value:.2f}" for value in closed)
        + " of its range, at s = 1/4, s = 1/2 and on the boundary"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--free-boundary",
        action="store_true",
        help="also solve the plasma in the field of the coils",
    )
    args = parser.parse_args()
    field = BiotSavart(coils())
    k = interior(field, KNOT)
    print(
        f"42 coils of radius {RADIUS:.3f} m: magnetic axis through R = "
        f"{k['r_axis']:.5f} m on the midplane of phi = 0 with iota "
        f"{abs(k['iota_axis']):.4f} next to it, boundary traced from 30 cm outboard "
        f"of it with iota {abs(k['iota_surface']):.4f}, fitted within "
        f"{k['misfit']:.1e} m"
    )
    last = None
    for p0 in (0.0, *PRESSURES):
        step = k["input"].model_copy(update={"pres_scale": p0})
        if last is not None:
            # A hot restart takes one multigrid step.
            step = step.model_copy(
                update={
                    "ns_array": step.ns_array[-1:],
                    "ftol_array": np.array([1.0e-10]),
                    "niter_array": step.niter_array[-1:],
                }
            )
        last = vmecpp.run(step, max_threads=4, verbose=False, restart_from=last)
        wout = last.wout
        shift = np.hypot(*(axis_curve(wout, k["planes"]) - k["axis"]).T).max()
        iota = np.asarray(wout.iotaf)
        print(
            f"p0 = {p0:5.0f} Pa, beta {wout.betatotal:.1e}: {wout.itfsq} iterations, "
            f"magnetic axis {shift:.1e} m from the traced one, iota "
            f"{abs(iota[0]):.4f} on the axis and {abs(iota[-1]):.4f} on the boundary, "
            + describe(*measures(wout))
        )
    w7x = interior(BiotSavart(w7x_coils()), W7X)
    wout = vmecpp.run(w7x["input"], max_threads=4, verbose=False).wout
    print(
        f"W7-X, boundary traced from 22 cm outboard of its axis: {wout.itfsq} "
        "iterations, " + describe(*measures(wout))
    )
    if args.free_boundary:
        lock = threading.Lock()

        def external(r, phi, z):
            r, phi, z = (np.asarray(v, dtype=float) for v in (r, phi, z))
            cos, sin = np.cos(phi), np.sin(phi)
            points = np.stack([r * cos, r * sin, z], -1).reshape(-1, 3)
            with lock:
                field.set_points(np.ascontiguousarray(points))
                b = np.array(field.B()).reshape(*r.shape, 3)
            b_r = b[..., 0] * cos + b[..., 1] * sin
            return b_r, b[..., 1] * cos - b[..., 0] * sin, b[..., 2]

        fine = interior(field, dataclasses.replace(KNOT, planes=128, mpol=14, ntor=32))
        last = vmecpp.run(fine["input"], max_threads=4, verbose=False)
        single = {
            "ns_array": fine["input"].ns_array[-1:],
            "ftol_array": np.array([1.0e-9]),
            "niter_array": np.array([20000]),
        }
        for p0 in (0.0, *PRESSURES):
            free = fine["input"].model_copy(
                update={"lfreeb": True, "pres_scale": p0, **single}
            )
            last = vmecpp.run(
                free,
                magnetic_field=external,
                max_threads=4,
                verbose=False,
                restart_from=last,
            )
            wout = last.wout
            shift = np.hypot(*(axis_curve(wout, fine["planes"]) - fine["axis"]).T)
            print(
                f"free boundary, p0 = {p0:5.0f} Pa, beta {wout.betatotal:.1e}: "
                f"{wout.itfsq} iterations, magnetic axis moved {shift.max():.1e} m, "
                + describe(*measures(wout))
            )


if __name__ == "__main__":
    main()
