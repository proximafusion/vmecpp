# SPDX-FileCopyrightText: 2026-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Grow a quasi-helical core inside an axisymmetric boundary.

A tokamak of eight field periods with a rotational transform above 8 on its axis holds a
(1, 8) helical core whose axis winds eight times around the torus. The boundary has no
harmonic of n other than 0, and shaping it, the rotational transform and the pressure on
the exact gradients of VMEC++ lowers the (1, 1) quasisymmetry residuals inside the core.
"""

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

import vmecpp
from vmecpp import autodiff, geometry, qs
from vmecpp.cpp import _vmecpp  # type: ignore[import]

NFP, MPOL, NTOR, NS = 8, 6, 4, 25
# major and minor radius and toroidal flux
R0, A, PHIEDGE = 6.0, 0.5, 1.0
# the axis shift beyond which the normal of the axis turns once per field period
CRITICAL = R0 / (NFP**2 + 1)
# the surfaces inside the core on which the residuals are taken
INSIDE = (0.02, 0.05, 0.1)
# the shape: the n = 0 coefficients of R for m = 2 to 5 and of Z for m = 1 to 5 in
# metres, iota_0 to iota_3 of iota = sum_k iota_k s^k, and pres_scale over 1e4
START = np.array([0, 0, 0, 0, 1.3 * A, 0, 0, 0, 0, 8.55, -1.5, 0, 0, 0.0])
# the shape optimize reaches from START
SHAPED = np.concatenate(
    [
        [0.0047, 0.0044, -0.0011, 0.0042],
        [0.6397, -0.0048, 0.0, -0.0006, 0.004],
        [8.5125, -1.5244, -0.0116, -0.0063],
        [0.0],
    ]
)
BOUNDS = (
    [(-0.075, 0.075)] * 4
    + [(0.5, 1.0)]
    + [(-0.075, 0.075)] * 4
    + [(8.0, 9.0)]
    + [(-4.0, 4.0)] * 3
    + [(0.0, 2.0)]
)
# the axis shift of the initial guess of a core
SEED = 0.05


def core_input(shape, seed=SEED):
    """The fixed-boundary input of the axisymmetric boundary ``shape``, with its axis
    guess shifted helically by ``seed``."""
    vmec_input = vmecpp.VmecInput(
        nfp=NFP,
        mpol=MPOL,
        ntor=NTOR,
        ns_array=np.array([NS]),
        ftol_array=np.array([1e-11]),
        niter_array=np.array([60000]),
        phiedge=PHIEDGE,
        ncurr=0,
        ai=np.asarray(shape[9:13], dtype=float),
        pmass_type="power_series",
        am=np.array([1.0, -2.0, 1.0]),
        pres_scale=float(shape[13]) * 1e4,
        gamma=0.0,
        always_fix_m1_gauge=True,
        raxis_c=np.array([R0, seed, 0.0, 0.0, 0.0]),
        zaxis_s=np.array([0.0, -seed, 0.0, 0.0, 0.0]),
    )
    vmec_input.rbc = vmecpp.VmecInput.resize_2d_coeff(
        np.concatenate([[R0, A], shape[0:4]])[:, None], MPOL, NTOR
    )
    vmec_input.zbs = vmecpp.VmecInput.resize_2d_coeff(
        np.concatenate([[0.0], shape[4:9]])[:, None], MPOL, NTOR
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


def optimize(start=START, evaluations=90):
    """The shape that lowers the (1, 1) residuals inside the core by L-BFGS-B on the
    exact gradients of VMEC++, with the core held above 1.2 CRITICAL.

    The gradients need a build with VMECPP_ENABLE_ENZYME.
    """
    template = core_input(start)
    solver = autodiff.make_solver(template)
    boundary = jnp.asarray(np.stack([template.rbc, template.zbs]))

    def objective(shape):
        rows = boundary.at[0, 2:, NTOR].set(shape[0:4]).at[1, 1:, NTOR].set(shape[4:9])
        core = solver(rows, {"ai": shape[9:13], "pres_scale": shape[13] * 1e4})
        helical = qs.quasisymmetry_total(
            core, INSIDE, helicity_m=1, helicity_n=1, ntheta=15, nphi=16
        )
        # the n = 1 cosine coefficient of R on the axis in the product basis
        r = core.r_cc
        shift = jnp.abs(r[0, 1, 0] if r.shape[1] == NTOR + 1 else r[0, 0, 1])
        return helical + 10.0 * jnp.maximum(0.0, 1.2 - shift / CRITICAL) ** 2

    value_and_grad = jax.value_and_grad(objective)

    def fun(shape):
        try:
            value, gradient = value_and_grad(jnp.asarray(shape))
        except (RuntimeError, jax.errors.JaxRuntimeError):
            return 1.0e3, np.zeros_like(shape)
        return float(value), np.asarray(gradient, dtype=float)

    options = {"maxfun": evaluations}
    return minimize(
        fun, start, jac=True, method="L-BFGS-B", bounds=BOUNDS, options=options
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
