# SPDX-FileCopyrightText: 2026-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Lundquist's force-free flux rope J = k B / mu0 through field reversal with lrfp,
against the Bessel field B_z = B0 J0(k r), B_theta = B0 J1(k r) of the straight rope."""

import numpy as np
from scipy.optimize import brentq
from scipy.special import j0, j1

import vmecpp

MU0 = 4.0e-7 * np.pi
MINOR = 1.0  # a in m
KA = 3.0  # k a; B_z reverses at k r = 2.405, inside the rope
B0 = 1.0  # B_z on the axis in T
ASPECTS = (10.0, 30.0, 100.0)  # R0 / a
DEGREE = 20  # of the power series of q in the normalized poloidal flux s
NS = [17, 33, 65, 129, 257, 513]


def rope(s):
    """Radius of the rope surface whose poloidal flux is the fraction s of the
    edge's."""
    k = KA / MINOR
    edge = 1.0 - j0(KA)
    return np.array(
        [
            0.0
            if x == 0.0
            else brentq(lambda r, x=x: (1.0 - j0(k * r)) / edge - x, 0.0, MINOR)
            for x in np.atleast_1d(s)
        ]
    )


def q_profile(aspect):
    """Power-series coefficients in s of q of the straight rope at R0 = aspect a, and
    the largest misfit of the series relative to q on the axis."""
    k, r0 = KA / MINOR, aspect * MINOR
    s = np.linspace(0.0, 1.0, 401)
    r = rope(s)
    q = np.where(
        r > 0.0, r * j0(k * r) / (r0 * j1(np.maximum(k * r, 1e-300))), 2.0 / (k * r0)
    )
    coefficients = np.polynomial.Polynomial.fit(s, q, DEGREE).convert().coef
    misfit = np.abs(np.polynomial.polynomial.polyval(s, coefficients) - q).max() / q[0]
    return coefficients, misfit


def lundquist_input(aspect, grids=NS):
    """The fixed-boundary input of the circular rope of the aspect ratio at zero
    pressure, with the q profile of the straight rope in s."""
    k, r0 = KA / MINOR, aspect * MINOR
    return vmecpp.VmecInput(
        lrfp=True,
        mpol=6,
        ntor=0,
        ns_array=np.array(grids),
        ftol_array=np.array([1e-12] * len(grids)),
        niter_array=np.array([80000] * len(grids)),
        phiedge=2.0 * np.pi * B0 * MINOR * j1(KA) / k,
        ncurr=0,
        piota_type="power_series",
        ai=q_profile(aspect)[0],
        pmass_type="power_series",
        am=np.array([0.0]),
        pres_scale=0.0,
        rbc=np.array([[r0], [MINOR], [0.0], [0.0], [0.0], [0.0]]),
        zbs=np.array([[0.0], [MINOR], [0.0], [0.0], [0.0], [0.0]]),
        raxis_c=np.array([r0]),
        zaxis_s=np.zeros(1),
    )


def lundquist(output):
    """The magnitude of mu0 <J.B> / <B^2> on the surfaces with 0 < s < 0.95, s of the
    reversal surface where the toroidal flux peaks, and F = B_z(a) / <B_z>."""
    jxb, wout = output.jxbout, output.wout
    phi = np.asarray(wout.phi)
    s = np.linspace(0.0, 1.0, len(phi))
    inner = (s > 0.0) & (s < 0.95)
    mu = np.abs(MU0 * np.asarray(jxb.jdotb)[inner] / np.asarray(jxb.bdotb)[inner])
    j = int(np.argmax(np.abs(phi)))
    # the vertex of the parabola through the three points about the peak
    a, b, _ = np.polyfit(s[j - 1 : j + 2], phi[j - 1 : j + 2], 2)
    s_reversal = -b / (2.0 * a)
    r0 = float(np.asarray(wout.rmnc)[0, -1])
    # R B_phi from the last two half-grid surfaces to the boundary, over its centre
    bvco = np.asarray(wout.bvco)
    edge_field = (1.5 * bvco[-1] - 0.5 * bvco[-2]) / r0
    mean_field = float(wout.phi[-1]) / (np.pi * MINOR**2)
    return mu, s_reversal, edge_field / mean_field


def main():
    s_reversal = (1.0 - j0(2.404825557695773)) / (1.0 - j0(KA))
    reversal = KA * j0(KA) / (2.0 * j1(KA))
    k = KA / MINOR
    print(
        f"straight rope at k a = {KA}: reversal at s = {s_reversal:.5f}, F = {reversal:.5f}, "
        f"|mu0 J.B / B^2| = k = {k:.5f} / m"
    )
    for aspect in ASPECTS:
        output = vmecpp.run(lundquist_input(aspect), max_threads=1, verbose=False)
        mu, s_rev, f = lundquist(output)
        print(
            f"R0 / a = {aspect:5.0f}: q fitted within {q_profile(aspect)[1]:.1e}, "
            f"{output.wout.itfsq} iterations, |mu0 J.B / B^2| from {mu.min():.5f} to "
            f"{mu.max():.5f} / m for s < 0.95, reversal at s = {s_rev:.5f}, F = {f:.5f}"
        )


if __name__ == "__main__":
    main()
