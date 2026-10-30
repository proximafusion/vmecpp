# SPDX-FileCopyrightText: 2026-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Alfven's Birkeland rope: Taylor-state strands wound about each other along a guide
field, one solved inside a boundary matched to NESTOR's vacuum field."""

import argparse
import dataclasses
import functools
import json
import multiprocessing
import os
import time

import numpy as np
from scipy.optimize import brentq
from scipy.special import j0, j1

import vmecpp

MU0 = 4.0e-7 * np.pi
N = 3  # braid turns per transit
R0 = 1.0  # radius of the rope in m
B0 = 1.0  # guide field at R0 in T
PLANES, THETAS = 48, 48  # planes and polar angles of the boundary
SEGMENTS = 192  # points on the magnetic axis of the strand
SOURCES = 24, 108  # polar and toroidal angles of the current elements of each strand
PENALTY = 10.0  # mode amplitude of a boundary that VMEC++ does not solve
STEP = 1.0e-5  # finite-difference step of the unknowns, in m and T
MAX_STEP = 2.0e-3  # largest change of an unknown in one step, in m and T


@dataclasses.dataclass(frozen=True)
class Rope:
    """The design's helical angle in degrees, helix radius over strand radius and number
    of strands, the shape modes and the flux surfaces."""

    angle: float = 15.0
    spacing: float = 2.0
    strands: int = 2
    m_shape: int = 6
    n_shape: int = 6
    ns: int = 25

    @functools.cached_property
    def helix_radius(self):
        return R0 * np.tan(np.radians(self.angle)) / N

    @functools.cached_property
    def radius(self):
        return self.helix_radius / self.spacing

    @functools.cached_property
    def ka(self):
        """The k a of the Taylor strand whose edge pitch J1 / J0, 2 (d / b) tan psi over
        strands - 1, gives the point-vortex pitch."""
        pitch = 2.0 * self.spacing * np.tan(np.radians(self.angle)) / (self.strands - 1)
        return brentq(lambda x: j1(x) / j0(x) - pitch, 1e-3, 2.4)

    @functools.cached_property
    def phiedge(self):
        """The toroidal flux of the strand whose edge field along it is B0."""
        return 2.0 * np.pi * self.radius**2 * B0 * j1(self.ka) / (self.ka * j0(self.ka))

    @functools.cached_property
    def current(self):
        return self.ka / self.radius * self.phiedge / MU0

    @functools.cached_property
    def modes(self):
        return [
            (m, n)
            for m in range(self.m_shape + 1)
            for n in range(-self.n_shape, self.n_shape + 1)
            if m > 0 or n >= 0
        ]

    @property
    def mpol(self):
        return self.m_shape + 2

    @property
    def ntor(self):
        return self.n_shape + 2


def plane_phi():
    return np.arange(PLANES) * 2.0 * np.pi / (N * PLANES)


def polar_theta():
    return 2.0 * np.pi * np.arange(THETAS) / THETAS


def helix_points(rope, z):
    """Points of the design helix R = R0 + d cos N z, Z = -d sin N z at phi = z."""
    r = R0 + rope.helix_radius * np.cos(N * z)
    return np.stack(
        [r * np.cos(z), r * np.sin(z), -rope.helix_radius * np.sin(N * z)], -1
    )


def frenet(rope, z, h=1e-4):
    """Tangent, principal normal and binormal of the design helix at z."""
    d1 = (helix_points(rope, z + h) - helix_points(rope, z - h)) / (2.0 * h)
    d2 = (
        helix_points(rope, z + h)
        - 2.0 * helix_points(rope, z)
        + helix_points(rope, z - h)
    ) / h**2
    t = d1 / np.linalg.norm(d1, axis=-1, keepdims=True)
    n = d2 - np.sum(d2 * t, axis=-1, keepdims=True) * t
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    return t, n, np.cross(t, n)


def fourier(values, phi, theta, mpol, ntor, parity):
    """Coefficients [m, ntor + n] of values[plane, theta] in cos (parity 0) or sin
    (parity 1) of m theta - n phi."""
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


def boundary(rope, x):
    """R and Z in the planes of a period at sum x_mn cos(m theta - n N phi) from the
    design helix, theta the polar angle about it."""
    phi, theta = plane_phi(), polar_theta()
    rho = sum(
        c * np.cos(m * theta[None, :] - n * N * phi[:, None])
        for c, (m, n) in zip(x, rope.modes, strict=True)
    )
    r_c, z_c = (
        R0 + rope.helix_radius * np.cos(N * phi),
        -rope.helix_radius * np.sin(N * phi),
    )
    return r_c[:, None] + rho * np.cos(theta), z_c[:, None] + rho * np.sin(theta)


def initial_shape(rope):
    """The shape of the circular tube of the strand's radius about the helix in its
    normal planes."""
    phi, theta = plane_phi(), polar_theta()
    psi = 2.0 * np.pi * np.arange(4 * THETAS) / (4 * THETAS)
    rho = np.empty((PLANES, THETAS))
    for k, phi_k in enumerate(phi):

        def tube(z):
            _, normal, binormal = frenet(rope, z)
            return helix_points(rope, z) + rope.radius * (
                np.cos(psi)[:, None] * normal + np.sin(psi)[:, None] * binormal
            )

        def offset(z, phi_k=phi_k):
            x = tube(z)
            return np.angle(np.exp(1j * (np.arctan2(x[:, 1], x[:, 0]) - phi_k)))

        # the parameter of the helix whose normal circle meets the plane phi_k at each
        # psi, by Newton
        z = np.full_like(psi, phi_k)
        for _ in range(20):
            f = offset(z)
            z -= f * 1e-7 / (offset(z + 1e-7) - f)
        x = tube(z)
        dr = np.hypot(x[:, 0], x[:, 1]) - (R0 + rope.helix_radius * np.cos(N * phi_k))
        dz = x[:, 2] + rope.helix_radius * np.sin(N * phi_k)
        order = np.argsort(np.arctan2(dz, dr))
        rho[k] = np.interp(
            theta,
            np.arctan2(dz, dr)[order],
            np.hypot(dr, dz)[order],
            period=2.0 * np.pi,
        )
    out = np.empty(len(rope.modes))
    for i, (m, n) in enumerate(rope.modes):
        weight = (1.0 if (m, n) == (0, 0) else 2.0) / rho.size
        out[i] = weight * np.sum(
            rho * np.cos(m * theta[None, :] - n * N * phi[:, None])
        )
    return out


def strand_input(rope, rb, zb, **update):
    """The fixed-boundary Taylor strand inside the boundary, at zero pressure with the
    enclosed current k Phi / mu0."""
    phi, theta = plane_phi() * N, polar_theta()
    r_c, z_c = R0 + rope.helix_radius * np.cos(phi), -rope.helix_radius * np.sin(phi)
    return vmecpp.VmecInput(
        nfp=N,
        lasym=False,
        mpol=rope.mpol,
        ntor=rope.ntor,
        ntheta=40,
        nzeta=48,
        ns_array=np.array([rope.ns]),
        ftol_array=np.array([1e-13]),
        niter_array=np.array([20000]),
        phiedge=rope.phiedge,
        ncurr=1,
        pcurr_type="power_series",
        ac=np.array([1.0]),
        curtor=rope.current,
        pmass_type="power_series",
        am=np.array([0.0]),
        pres_scale=0.0,
        gamma=0.0,
        rbc=fourier(rb, phi, theta, rope.mpol, rope.ntor, 0),
        zbs=fourier(zb, phi, theta, rope.mpol, rope.ntor, 1),
        raxis_c=fourier(r_c[:, None], phi, np.zeros(1), 1, rope.ntor, 0)[
            0, rope.ntor :
        ],
        zaxis_s=fourier(z_c[:, None], phi, np.zeros(1), 1, rope.ntor, 1)[
            0, rope.ntor :
        ],
        nvacskip=1,
        return_outputs_even_if_not_converged=True,
    ).model_copy(update=update)


def axis_points(wout):
    """Cartesian points of the magnetic axis of a run."""
    phi = 2.0 * np.pi * np.arange(SEGMENTS) / SEGMENTS
    xm, xn = np.asarray(wout.xm), np.asarray(wout.xn)
    axis = xm == 0
    r = np.cos(np.outer(phi, xn[axis])) @ np.asarray(wout.rmnc)[axis, 0]
    z = -np.sin(np.outer(phi, xn[axis])) @ np.asarray(wout.zmns)[axis, 0]
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], -1)


def partners(vectors, strands):
    """The vectors turned about Z by 2 pi j / (strands N) for j = 1 to strands - 1,
    their images on the other strands."""
    out = [vectors[:0]]
    for j in range(1, strands):
        c, s = (
            np.cos(2.0 * np.pi * j / (strands * N)),
            np.sin(2.0 * np.pi * j / (strands * N)),
        )
        out.append(vectors @ np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]]))
    return np.concatenate(out)


def helix_of(points):
    """The mean R of the axis, the radius of the rope, and its mean distance from that
    circle, the helix radius."""
    r, z = np.hypot(points[:, 0], points[:, 1]), points[:, 2]
    return r.mean(), np.hypot(r - r.mean(), z).mean()


def geometry(wout, js, theta, zeta):
    """Points, tangents e_theta and e_zeta, sqrt(g), and sqrt(g) J^theta and sqrt(g)
    J^zeta on the full-grid surfaces js at the angles theta and zeta."""

    def series(coefficients, m, n, kind, d_theta=0, d_zeta=0):
        angle = m[:, None, None] * theta[None, :, None] - n[:, None, None] * zeta
        if (d_theta, d_zeta) == (0, 0):
            base = kind(angle)
        else:
            base = -np.sin(angle) if kind is np.cos else np.cos(angle)
        factor = (m**d_theta * (-n) ** d_zeta)[:, None, None]
        return np.tensordot(coefficients.T, factor * base, axes=1)

    xm, xn = np.asarray(wout.xm, float), np.asarray(wout.xn, float)
    xmb, xnb = np.asarray(wout.xm_nyq, float), np.asarray(wout.xn_nyq, float)
    rmnc, zmns = np.asarray(wout.rmnc)[:, js], np.asarray(wout.zmns)[:, js]
    r = series(rmnc, xm, xn, np.cos)
    z = series(zmns, xm, xn, np.sin)
    r_t, r_z = series(rmnc, xm, xn, np.cos, 1, 0), series(rmnc, xm, xn, np.cos, 0, 1)
    z_t, z_z = series(zmns, xm, xn, np.sin, 1, 0), series(zmns, xm, xn, np.sin, 0, 1)
    cos, sin = np.cos(zeta), np.sin(zeta)
    points = np.stack([r * cos, r * sin, z], -1)
    e_theta = np.stack([r_t * cos, r_t * sin, z_t], -1)
    e_zeta = np.stack([r_z * cos - r * sin, r_z * sin + r * cos, z_z], -1)
    # sqrt(g) on the full grid from the half-grid coefficients of gmnc, extrapolated to
    # the edge and, in its m = 0 modes, to the axis
    gmnc = np.asarray(wout.gmnc)
    full = np.zeros_like(gmnc)
    full[:, 0] = np.where(xmb == 0, 1.5 * gmnc[:, 1] - 0.5 * gmnc[:, 2], 0.0)
    full[:, 1:-1] = 0.5 * (gmnc[:, 1:-1] + gmnc[:, 2:])
    full[:, -1] = 1.5 * gmnc[:, -1] - 0.5 * gmnc[:, -2]
    jacobian = series(full[:, js], xmb, xnb, np.cos)
    cu = series(np.asarray(wout.currumnc)[:, js], xmb, xnb, np.cos)
    cv = series(np.asarray(wout.currvmnc)[:, js], xmb, xnb, np.cos)
    return points, e_theta, e_zeta, jacobian, cu, cv


def currents(wout, ns_step, ntheta, nzeta):
    """Points, current density and Simpson-weighted volume of the elements of a strand
    on an odd number of surfaces ns_step apart, and its toroidal current."""
    ns = int(wout.ns)
    js = np.arange(0, ns, ns_step)
    weight = np.where(np.arange(js.size) % 2 == 1, 4.0, 2.0)
    weight[[0, -1]] = 1.0
    weight *= ns_step / (3.0 * (ns - 1))
    theta = 2.0 * np.pi * np.arange(ntheta) / ntheta
    zeta = 2.0 * np.pi * np.arange(nzeta) / nzeta
    points, e_theta, e_zeta, jacobian, cu, cv = geometry(wout, js, theta, zeta)
    density = (cu / jacobian)[..., None] * e_theta + (cv / jacobian)[..., None] * e_zeta
    cell = (2.0 * np.pi) ** 2 / (ntheta * nzeta)
    volume = np.abs(jacobian) * weight[:, None, None] * cell
    current = (
        float(wout.signgs) * np.sum(weight * cv[:, :, 0].sum(axis=1)) * 2.0 * np.pi
    )
    return points, density, volume, current / ntheta


def biot_savart(points, elements, x):
    """Field at the points x of the current elements J dV at points."""
    out = np.empty_like(x)
    for k in range(0, len(x), 128):
        d = x[k : k + 128, None] - points[None]
        out[k : k + 128] = np.einsum(
            "tsk,ts->tk", np.cross(elements[None], d), np.linalg.norm(d, axis=-1) ** -3
        )
    return MU0 / (4.0 * np.pi) * out


@dataclasses.dataclass
class Field:
    """The other strands as current elements J dV at points, the guide field B0 R0 / R
    along phi and a vertical field."""

    points: np.ndarray
    elements: np.ndarray
    vertical: float

    def __call__(self, r, phi, z):
        r, phi, z = np.broadcast_arrays(
            *(np.asarray(v, dtype=float) for v in (r, phi, z))
        )
        cos, sin = np.cos(phi.ravel()), np.sin(phi.ravel())
        x = np.stack([r.ravel() * cos, r.ravel() * sin, z.ravel()], -1)
        b = biot_savart(self.points, self.elements, x)
        b_r = (b[:, 0] * cos + b[:, 1] * sin).reshape(r.shape)
        b_phi = (b[:, 1] * cos - b[:, 0] * sin).reshape(r.shape) + B0 * R0 / r
        return b_r, b_phi, b[:, 2].reshape(r.shape) + self.vertical


def solve(rope, x, vertical, start=None):
    """The fixed-boundary strand inside the boundary of x, hot-started from start, and
    its first free-boundary iteration in the field of its turned copies."""
    rb, zb = boundary(rope, x)
    fixed = vmecpp.run(
        strand_input(rope, rb, zb), max_threads=1, verbose=False, restart_from=start
    )
    points, density, volume, _ = currents(fixed.wout, 1, *SOURCES)
    elements = (density * volume[..., None]).reshape(-1, 3)
    field = Field(
        partners(points.reshape(-1, 3), rope.strands),
        partners(elements, rope.strands),
        vertical,
    )
    free_input = strand_input(
        rope, rb, zb, lfreeb=True, niter_array=np.array([1]), return_vacuum_field=True
    )
    return fixed, vmecpp.run(
        free_input, field, max_threads=1, verbose=False, restart_from=fixed
    )


@functools.lru_cache
def projection(m_shape, n_shape):
    """The least-squares projection of a function on the solver's boundary grid onto the
    shape modes."""
    k, j = np.meshgrid(np.arange(48), np.arange(21), indexing="ij")
    modes = Rope(m_shape=m_shape, n_shape=n_shape).modes
    grid = np.stack(
        [
            np.cos(m * 2.0 * np.pi * j / 40 - n * 2.0 * np.pi * k / 48).ravel()
            for m, n in modes
        ],
        1,
    )
    return np.linalg.pinv(grid)


def jump(free):
    """The jump in magnetic pressure across the boundary on the solver's grid, plasma
    minus vacuum, in units of B0^2 / 2."""
    fb = free.threed1_free_boundary
    return ((np.asarray(fb.bsqmhdi) - np.asarray(fb.bsqvaci)) / (0.5 * B0**2)).ravel()


def unknowns_to_shape(rope, x0, y):
    """The shape, its shift in R held at x0's so that the rope stays at R0, and the
    vertical field of the unknowns."""
    x = x0.copy()
    free = [i for i, mode in enumerate(rope.modes) if mode != (1, 0)]
    x[free] = y[:-1]
    return x, y[-1]


def evaluate(rope, x0, start, y):
    """The jump in the shape modes of the unknowns y and the fixed-boundary run, or
    PENALTY and None where VMEC++ does not solve the boundary."""
    x, vertical = unknowns_to_shape(rope, x0, y)
    try:
        fixed, free = solve(rope, x, vertical, start)
    except RuntimeError:
        return np.full(len(rope.modes), PENALTY), None
    if fixed.wout.ier_flag != 0:
        return np.full(len(rope.modes), PENALTY), None
    return projection(rope.m_shape, rope.n_shape) @ jump(free), fixed


def residual(rope, x0, start, y):
    """The jump in the shape modes of the unknowns y."""
    return evaluate(rope, x0, start, y)[0]


def match(rope, workers, report=print):
    """Levenberg-Marquardt on the shape and the vertical field until every jump mode
    vanishes, returning the shape, the vertical field and the fixed-boundary run."""
    x0 = initial_shape(rope)
    free = [i for i, mode in enumerate(rope.modes) if mode != (1, 0)]
    # the vertical field of a ring carrying the current of every strand, its minor
    # radius near the helix radius
    vertical = (
        -MU0
        * rope.strands
        * rope.current
        / (4.0 * np.pi * R0)
        * (np.log(8.0 * R0 / (0.88 * max(rope.helix_radius, rope.radius))) - 1.25)
    )
    y = np.concatenate([x0[free], [vertical]])
    f, start = evaluate(rope, x0, None, y)
    damping = 1e-3
    with multiprocessing.get_context("spawn").Pool(workers) as pool:
        for step in range(60):
            report(step, f, y[-1])
            if np.abs(f).max() < 1e-8:
                break
            # the solves of the step start from the strand at y while the jump is large,
            # and from VMEC's initial guess near the root
            hot = start if np.abs(f).max() > 1e-4 else None
            f = residual(rope, x0, hot, y)
            columns = pool.map(
                functools.partial(residual, rope, x0, hot),
                [y + STEP * e for e in np.eye(y.size)],
            )
            jacobian = np.column_stack([(c - f) / STEP for c in columns])
            gradient, normal = jacobian.T @ f, jacobian.T @ jacobian
            for _ in range(12):
                move = -np.linalg.solve(
                    normal + damping * np.diag(np.diag(normal)), gradient
                )
                move *= min(1.0, MAX_STEP / np.abs(move).max())
                trial, state = evaluate(rope, x0, hot, y + move)
                if state is not None and np.sum(trial**2) < np.sum(f**2):
                    y, f, start = y + move, trial, state
                    damping = max(damping / 3.0, 1e-9)
                    break
                damping *= 4.0
            else:
                break
    x, vertical = unknowns_to_shape(rope, x0, y)
    return x, vertical, start


def pitch(rope, wout):
    """The helix radius, the helical angle in degrees, and tan psi over the point-vortex
    pitch (strands - 1) mu0 I / (4 pi d B0)."""
    core, d = helix_of(axis_points(wout))
    psi = np.arctan(N * d / core)
    vortex = (rope.strands - 1) * MU0 * rope.current / (4.0 * np.pi * d * B0)
    return d, np.degrees(psi), np.tan(psi) / vortex


def main():
    parser = argparse.ArgumentParser(description="Alfven's Birkeland rope")
    parser.add_argument(
        "--angle",
        type=float,
        default=15.0,
        help="helical angle of the design in degrees",
    )
    parser.add_argument(
        "--spacing", type=float, default=2.0, help="helix radius over strand radius"
    )
    parser.add_argument("--strands", type=int, default=2)
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--save", help="JSON file for the matched shape")
    args = parser.parse_args()
    rope = Rope(angle=args.angle, spacing=args.spacing, strands=args.strands)
    start = time.time()
    print(
        f"{rope.strands} strands, design angle {rope.angle:g} deg, d/b "
        f"{rope.spacing:g}: helix radius "
        f"{rope.helix_radius:.5f} m, strand radius {rope.radius:.5f} m, k a "
        f"{rope.ka:.4f}, flux {rope.phiedge:.6f} Wb, current "
        f"{rope.current / 1e3:.2f} kA"
    )

    def report(step, f, vertical):
        print(
            f"  {time.time() - start:5.0f} s, step {step}: largest jump mode "
            f"{np.abs(f).max():.2e}, vertical field {vertical:+.5f} T",
            flush=True,
        )

    x, vertical, fixed = match(rope, args.workers, report)
    if args.save:
        matched = {
            "angle": rope.angle,
            "spacing": rope.spacing,
            "strands": rope.strands,
            "vertical": vertical,
            "modes": rope.modes,
            "x": x.tolist(),
        }
        with open(args.save, "w") as handle:
            json.dump(matched, handle, indent=1)
    d, psi, ratio = pitch(rope, fixed.wout)
    taylor = (
        MU0
        * np.asarray(fixed.jxbout.jdotb)[1:-1]
        / np.asarray(fixed.jxbout.bdotb)[1:-1]
    )
    print(
        f"helix radius {d:.5f} m, helical angle {psi:.4f} deg, tan psi over the "
        f"point-vortex pitch {ratio:.5f}, vertical field {vertical:+.6f} T; mu0 J.B / "
        f"B^2 from {taylor.min():.4f} to {taylor.max():.4f} / m against k = "
        f"{rope.ka / rope.radius:.4f} / m"
    )


if __name__ == "__main__":
    main()
