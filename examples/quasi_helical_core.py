# SPDX-FileCopyrightText: 2026-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Grow a quasi-helical core inside an axisymmetric boundary.

A tokamak of eight field periods with a rotational transform above 8 on its axis holds a
(1, 8) helical core whose axis winds eight times around the torus. The boundary has no
harmonic of n other than 0, and shaping it and the rotational transform lowers the (1,
1) quasisymmetry residuals inside the core.
"""

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
from scipy.optimize import least_squares

import vmecpp
from vmecpp import geometry, qs
from vmecpp.cpp import _vmecpp  # type: ignore[import]

NFP, MPOL, NTOR, NS = 8, 6, 4, 25
# major and minor radius and toroidal flux
R0, A, PHIEDGE = 6.0, 0.5, 1.0
# the axis shift beyond which the normal of the axis turns once per field period
CRITICAL = R0 / (NFP**2 + 1)
# the surfaces inside the core on which the residuals are taken
INSIDE = (0.02, 0.05, 0.1)
# elongation, the m = 2 and m = 3 terms of R and the m = 2 term of Z over the minor
# radius, and iota = iota_0 + iota_1 s
START = np.array([1.3, 0.0, 0.0, 0.0, 8.55, -1.5])
# the shape optimize reaches from START
SHAPED = np.array([1.2821, 0.0, 0.0, 0.0, 8.5219, -1.4903])
LOWER = np.array([1.0, -0.15, -0.15, -0.15, 8.0, -3.0])
UPPER = np.array([2.0, 0.15, 0.15, 0.15, 9.0, -0.6])
# the axis shift of the initial guess of a core
SEED = 0.05


def core_input(shape, seed=SEED):
    """The fixed-boundary input of the axisymmetric boundary ``shape`` at zero pressure,
    with its axis guess shifted helically by ``seed``."""
    kappa, r2, r3, z2, iota_0, iota_1 = shape
    vmec_input = vmecpp.VmecInput(
        nfp=NFP,
        mpol=MPOL,
        ntor=NTOR,
        ns_array=np.array([NS]),
        ftol_array=np.array([1e-11]),
        niter_array=np.array([60000]),
        phiedge=PHIEDGE,
        ncurr=0,
        ai=np.array([iota_0, iota_1]),
        pmass_type="power_series",
        am=np.array([0.0]),
        pres_scale=0.0,
        gamma=0.0,
        always_fix_m1_gauge=True,
        raxis_c=np.array([R0, seed, 0.0, 0.0, 0.0]),
        zaxis_s=np.array([0.0, -seed, 0.0, 0.0, 0.0]),
    )
    vmec_input.rbc = vmecpp.VmecInput.resize_2d_coeff(
        np.array([[R0], [A], [r2 * A], [r3 * A]]), MPOL, NTOR
    )
    vmec_input.zbs = vmecpp.VmecInput.resize_2d_coeff(
        np.array([[0.0], [kappa * A], [z2 * A]]), MPOL, NTOR
    )
    return vmec_input


def solve(shape, seed=SEED, restart=None):
    """The output of VMEC++ and its input, by hot restart from ``restart``."""
    indata = core_input(shape, seed)._to_cpp_vmecindata()
    state = (
        None
        if restart is None
        else _vmecpp.HotRestartState(wout=restart[0].wout, indata=restart[1])
    )
    output = _vmecpp.run(indata, initial_state=state, verbose=_vmecpp.OutputMode.SILENT)
    return output, indata


def axis_shift(output):
    """The n = 1 cosine coefficient of R on the magnetic axis, per field period."""
    xm, xn = np.asarray(output.wout.xm), np.asarray(output.wout.xn)
    return float(np.asarray(output.wout.rmnc)[(xm == 0) & (xn == NFP), 0][0])


def mirror(output):
    """(max - min) / (max + min) along phi of the poloidal mean of |B| on the innermost
    half-grid surface."""
    wout = output.wout
    xm, xn = np.asarray(wout.xm_nyq), np.asarray(wout.xn_nyq, float)
    theta = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
    phi = np.linspace(0.0, 2.0 * np.pi / NFP, 64)
    angle = np.multiply.outer(theta, xm)[:, None, :] - np.multiply.outer(phi, xn)[None]
    along = (np.cos(angle) @ np.asarray(wout.bmnc)[:, 1]).mean(0)
    return (along.max() - along.min()) / (along.max() + along.min())


def residuals(output, helicity_n=1):
    """The quasisymmetry residuals of helicity (1, helicity_n) inside the core."""
    shape = geometry.from_cpp(_vmecpp.make_geometry(output))
    return np.asarray(
        qs.quasisymmetry_residuals(shape, INSIDE, helicity_m=1, helicity_n=helicity_n)
    )


def optimize(start=START, evaluations=150):
    """The shape that lowers the (1, 1) residuals inside the core, with the core held
    above 1.2 CRITICAL."""
    last = [solve(start)]
    size = []

    def objective(shape):
        try:
            output = solve(shape, restart=last[0])
        except RuntimeError:
            return np.ones(size[0])
        shift = abs(axis_shift(output[0]))
        if shift > 1.2 * CRITICAL:
            last[0] = output
        r = np.append(residuals(output[0]), 10.0 * max(0.0, 1.2 - shift / CRITICAL))
        size[:] = [r.size]
        return r

    objective(start)
    return least_squares(
        objective, start, bounds=(LOWER, UPPER), diff_step=1e-3, max_nfev=evaluations
    ).x


def describe(label, output):
    qa, qh = (float(np.sum(residuals(output, n) ** 2)) for n in (0, 1))
    print(
        f"{label}: axis shift {axis_shift(output):+.4f} against {CRITICAL:.4f}, mirror"
        f" {mirror(output):.4f}, QS(1, 0) {qa:.3e}, QS(1, 1) {qh:.3e}"
    )
    return qh


def main():
    describe("without a core", solve(START, seed=0.0)[0])
    unshaped = describe("the core", solve(START)[0])
    core = solve(SHAPED)[0]
    shaped = describe("the core of the shaped boundary", core)
    if abs(axis_shift(core)) < CRITICAL or shaped >= unshaped:
        message = "shaping the boundary loses the core or raises its residuals"
        raise RuntimeError(message)


if __name__ == "__main__":
    main()
