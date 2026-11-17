# SPDX-FileCopyrightText: 2026-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""The Taylor state J = k B / mu0 in a trefoil flux conserver, against Mercier's
rotational transform on its axis from torsion and current."""

import numpy as np
from island_interior import fourier
from knotted_stellarator import KNOTS, frenet

import vmecpp

MU0 = 4.0e-7 * np.pi
KNOT = KNOTS["trefoil"]  # R = 2 + cos 3z, Z = -sin 3z, phi = 2z in m
RADIUS = 0.1  # of the conserver about the knot in its normal planes, in m
K_VALUES = (0.0, 0.1, 0.2)  # k in 1/m
PLANES, THETAS = 72, 64  # planes of a period and polar angles of the conserver
MPOL, NTOR, NS = 8, 16, [13, 25, 49]


def tube(z, psi, radius):
    """Cartesian points at the angle psi from the principal normal on the circle of the
    radius about the knot in its normal plane at z."""
    _, n, b = frenet(KNOT, z)
    cos, sin = np.cos(psi)[:, None], np.sin(psi)[:, None]
    return KNOT.points(z) + radius * (cos * n + sin * b)


def conserver(radius):
    """R and Z of the conserver in the planes of a period on a uniform grid of the polar
    angle about the knot, R and Z of the knot in these planes, and the planes."""
    planes = np.arange(PLANES) * KNOT.period / PLANES
    psi = 2.0 * np.pi * np.arange(4 * THETAS) / (4 * THETAS)
    theta = 2.0 * np.pi * np.arange(THETAS) / THETAS
    rb, zb = np.empty((PLANES, THETAS)), np.empty((PLANES, THETAS))
    centre = np.empty((PLANES, 2))
    for k, phi in enumerate(planes):
        c = KNOT.points(np.array([phi / KNOT.p]))[0]
        centre[k] = np.hypot(c[0], c[1]), c[2]

        def offset(z, phi=phi):
            x = tube(z, psi, radius)
            return np.angle(np.exp(1j * (np.arctan2(x[:, 1], x[:, 0]) - phi)))

        # the knot parameter of the point of the circle at psi in the plane phi, by Newton
        z = np.full_like(psi, phi / KNOT.p)
        for _ in range(20):
            f = offset(z)
            z -= f * 1e-7 / (offset(z + 1e-7) - f)
        x = tube(z, psi, radius)
        r, height = np.hypot(x[:, 0], x[:, 1]), x[:, 2]
        alpha = np.arctan2(height - centre[k, 1], r - centre[k, 0])
        order = np.argsort(alpha)
        rb[k] = np.interp(theta, alpha[order], r[order], period=2.0 * np.pi)
        zb[k] = np.interp(theta, alpha[order], height[order], period=2.0 * np.pi)
    return rb, zb, centre, planes


def mercier(k, samples=1 << 14):
    """Rotational transform per toroidal turn on the axis of a circular conserver, from
    the torsion of the knot, the current k / 2 and the turns of its normal."""
    z = 2.0 * np.pi * np.arange(samples + 1) / samples
    h = 1e-3
    x = [KNOT.points(z + j * h) for j in (-2, -1, 0, 1, 2)]
    d1 = (x[3] - x[1]) / (2.0 * h)
    d2 = (x[3] - 2.0 * x[2] + x[1]) / h**2
    d3 = (x[4] - 2.0 * x[3] + 2.0 * x[1] - x[0]) / (2.0 * h**3)
    cross = np.cross(d1, d2)
    speed = np.linalg.norm(d1, axis=1)
    tau = np.sum(cross * d3, axis=1) / np.sum(cross * cross, axis=1)
    dz = 2.0 * np.pi / samples
    length = np.sum(speed[:-1]) * dz
    torsion = np.sum((tau * speed)[:-1]) * dz
    # turns of the principal normal about the knot in the polar angle of the planes
    _, n, _ = frenet(KNOT, z)
    phi = np.arctan2(x[2][:, 1], x[2][:, 0])
    beta = np.unwrap(np.arctan2(n[:, 2], n[:, 0] * np.cos(phi) + n[:, 1] * np.sin(phi)))
    winding = round((beta[-1] - beta[0]) / (2.0 * np.pi))
    per_closure = winding - (0.5 * k * length - torsion) / (2.0 * np.pi)
    return per_closure / KNOT.p, length, torsion, winding


def taylor_input(k, radius=RADIUS):
    """The fixed-boundary input of the conserver at zero pressure with the enclosed
    current k Phi(s) / mu0 of the Taylor state."""
    rb, zb, centre, planes = conserver(radius)
    angle = planes * KNOT.q / KNOT.p
    theta = 2.0 * np.pi * np.arange(THETAS) / THETAS
    phiedge = np.pi * radius**2
    return vmecpp.VmecInput(
        nfp=KNOT.q,
        toroidal_transits=KNOT.p,
        lasym=False,
        mpol=MPOL,
        ntor=NTOR,
        nzeta=2 * NTOR + 8,
        ns_array=np.array(NS),
        ftol_array=np.array([1e-10] + [1e-12] * (len(NS) - 1)),
        niter_array=np.array([30000] + [60000] * (len(NS) - 1)),
        phiedge=phiedge,
        ncurr=1,
        pcurr_type="power_series",
        ac=np.array([1.0]),
        curtor=k * phiedge / MU0,
        pmass_type="power_series",
        am=np.array([0.0]),
        pres_scale=0.0,
        rbc=fourier(rb, angle, theta, MPOL, NTOR, 0),
        zbs=fourier(zb, angle, theta, MPOL, NTOR, 1),
        raxis_c=fourier(centre[:, :1], angle, np.zeros(1), 1, NTOR, 0)[0, NTOR:],
        zaxis_s=fourier(centre[:, 1:], angle, np.zeros(1), 1, NTOR, 1)[0, NTOR:],
    )


def taylor_state(output):
    """The ratio mu0 <J.B> / <B^2> on the interior surfaces of a run."""
    jxb = output.jxbout
    return MU0 * np.asarray(jxb.jdotb)[1:-1] / np.asarray(jxb.bdotb)[1:-1]


def main():
    _, length, torsion, winding = mercier(0.0)
    print(
        f"trefoil: length {length:.4f} m, total torsion {torsion:.5f}, the normal turns "
        f"{winding} times about it in the planes phi = const"
    )
    for k in K_VALUES:
        output = vmecpp.run(taylor_input(k), max_threads=4, verbose=False)
        wout = output.wout
        mu = taylor_state(output)
        print(
            f"k = {k:.2f} / m: {wout.itfsq} iterations, mu0 J.B / B^2 from {mu.min():.6f} "
            f"to {mu.max():.6f} / m, iota {abs(wout.iotaf[-1]):.5f} on the boundary "
            f"against {abs(mercier(np.mean(mu))[0]):.5f} on the axis of a thin conserver"
        )


if __name__ == "__main__":
    main()
