# SPDX-FileCopyrightText: 2026-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Reconstruction of the W7-X pressure and current profiles from synthetic diagnostics.

The Jacobians come from the implicit adjoint, which needs a build with
-DVMECPP_ENABLE_ENZYME=ON.

    python examples/equilibrium_reconstruction.py [--taylor] [--coverage N]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import scipy.optimize

import vmecpp

jax.config.update("jax_enable_x64", True)

INPUT = Path(__file__).resolve().parent / "data" / "w7x.json"
MPOL = NTOR = 6
NS = 31

# The unknowns, in units of SCALE, are the pressure coefficients am (Pa), the shape c
# of the current density I'(s) = s + c s^2 and the total toroidal current curtor (A).
NAMES = ("am[0]", "am[1]", "am[2]", "c", "curtor")
SCALE = np.array([1.0e5, 1.0e5, 1.0e5, 1.0, 1.0e4])
TRUTH = np.array([1.0e5, -2.0e5, 1.0e5, -1.0, 2.0e4]) / SCALE
START = np.array([0.6e5, -0.6e5, 0.0, 0.0, 1.5e4]) / SCALE

# The Thomson pressures are taken outboard and inboard of the axis and the MSE pitch
# angles outboard, on the midplane chord of the \phi = 0 plane, at the radii of these
# flux labels in the true equilibrium.
THOMSON_S = np.linspace(0.05, 0.85, 9)
MSE_S = np.linspace(0.1, 0.8, 8)
SIGMA_PRESSURE = 3.0e3  # Pa
SIGMA_PITCH = 1.0e-3  # rad
SIGMA_ENERGY = 0.03  # relative to the true energy
SIGMA_AXIS = 2.0e-3  # m
SIGMA_CURRENT = 5.0e2  # A


def base_input() -> vmecpp.VmecInput:
    source = vmecpp.VmecInput.from_file(INPUT)
    lo, hi = source.ntor - NTOR, source.ntor + NTOR + 1
    return source.model_copy(
        update={
            "mpol": MPOL,
            "ntor": NTOR,
            "rbc": np.asarray(source.rbc)[:MPOL, lo:hi],
            "zbs": np.asarray(source.zbs)[:MPOL, lo:hi],
            "raxis_c": np.asarray(source.raxis_c)[: NTOR + 1],
            "zaxis_s": np.asarray(source.zaxis_s)[: NTOR + 1],
            "ns_array": np.array([NS]),
            "ftol_array": np.array([1.0e-14]),
            "niter_array": np.array([20000]),
            "pres_scale": 1.0,
        }
    )


def profiles(x) -> dict:
    values = jnp.asarray(x) * SCALE
    return {
        "am": values[:3],
        "ac": jnp.stack([jnp.zeros(()), jnp.ones(()), values[3]]),
        "curtor": values[4],
    }


def lagrange(values, first: float, step: float, s):
    """Four-point Lagrange interpolation of values on the grid first + j * step."""
    x = (jnp.asarray(s) - first) / step
    start = jnp.clip(jnp.floor(x).astype(int) - 1, 0, values.shape[0] - 4)
    t = x - start
    weights = jnp.stack(
        [
            -(t - 1) * (t - 2) * (t - 3) / 6,
            t * (t - 2) * (t - 3) / 2,
            -t * (t - 1) * (t - 3) / 2,
            t * (t - 1) * (t - 2) / 6,
        ],
        axis=-1,
    )
    return jnp.sum(weights * values[start[..., None] + jnp.arange(4)], axis=-1)


class Midplane:
    r"""The Fourier series of a wout on the midplane chord of the \phi = 0 plane, at
    \theta = 0 (side +1) or \theta = \pi (side -1), where Z, \partial_\theta R and
    \partial_\phi R vanish."""

    def __init__(self, wout, side: int) -> None:
        self.full = 1.0 / (NS - 1)
        parity = jnp.asarray(float(side) ** np.asarray(wout.xm))
        parity_nyq = jnp.asarray(float(side) ** np.asarray(wout.xm_nyq))
        xm, xn = jnp.asarray(wout.xm, float), jnp.asarray(wout.xn, float)
        self.r = jnp.asarray(wout.rmnc).T @ parity
        self.z_theta = jnp.asarray(wout.zmns).T @ (parity * xm)
        self.z_phi = -(jnp.asarray(wout.zmns).T @ (parity * xn))
        self.b_theta = (jnp.asarray(wout.bsupumnc).T @ parity_nyq)[1:]
        self.b_phi = (jnp.asarray(wout.bsupvmnc).T @ parity_nyq)[1:]

    def radius(self, s):
        return lagrange(self.r, 0.0, self.full, s)

    def flux_label(self, radius):
        """The flux label s with R(s) = radius: Newton without gradient, then one step
        with it, whose derivative is the implicit one at the converged root."""
        frozen = jax.lax.stop_gradient(self.r)

        def r_of(values, s):
            return lagrange(values, 0.0, self.full, s)

        def step(values, s):
            value, slope = jax.jvp(lambda x: r_of(values, x), (s,), (jnp.ones_like(s),))
            return s - (value - radius) / slope

        s = jnp.full(jnp.shape(radius), 0.5)
        for _ in range(40):
            s = jnp.clip(step(frozen, s), 1.0e-6, 1.0)
        return step(self.r, jax.lax.stop_gradient(s))

    def pitch(self, s):
        r"""The pitch angle \arctan(B_Z / B_\phi) of the field, with B_\phi = R B^\phi
        and B_Z = B^\theta \partial_\theta Z + B^\phi \partial_\phi Z."""
        b_theta = lagrange(self.b_theta, 0.5 * self.full, self.full, s)
        b_phi = lagrange(self.b_phi, 0.5 * self.full, self.full, s)
        z_theta = lagrange(self.z_theta, 0.0, self.full, s)
        z_phi = lagrange(self.z_phi, 0.0, self.full, s)
        return jnp.arctan(
            (b_theta * z_theta + b_phi * z_phi) / (self.radius(s) * b_phi)
        )


def pressure(x, s):
    am = jnp.asarray(x)[:3] * SCALE[:3]
    return am[0] + am[1] * s + am[2] * s**2


def solve(x, base):
    return vmecpp.run(base.model_copy(update=profiles(x)), verbose=False).wout


def positions(base) -> dict:
    """The chord radii of the diagnostics, fixed from the true equilibrium."""
    wout = solve(TRUTH, base)
    out, inside = Midplane(wout, +1), Midplane(wout, -1)
    return {
        "thomson_out": np.asarray(out.radius(jnp.asarray(THOMSON_S))),
        "thomson_in": np.asarray(inside.radius(jnp.asarray(THOMSON_S))),
        "mse": np.asarray(out.radius(jnp.asarray(MSE_S))),
    }


def predict(x, base, where):
    wout = solve(x, base)
    out, inside = Midplane(wout, +1), Midplane(wout, -1)
    s_out = out.flux_label(jnp.asarray(where["thomson_out"]))
    s_in = inside.flux_label(jnp.asarray(where["thomson_in"]))
    s_mse = out.flux_label(jnp.asarray(where["mse"]))
    return {
        "pressure": jnp.concatenate([pressure(x, s_out), pressure(x, s_in)]),
        "pitch": out.pitch(s_mse),
        "energy": jnp.atleast_1d(wout.wp),
        "axis": jnp.atleast_1d(out.radius(0.0)),
        "current": jnp.atleast_1d(jnp.asarray(x)[4] * SCALE[4]),
    }


def sigmas(truth_prediction) -> dict:
    return {
        "pressure": SIGMA_PRESSURE,
        "pitch": SIGMA_PITCH,
        "energy": SIGMA_ENERGY * np.abs(np.asarray(truth_prediction["energy"])),
        "axis": SIGMA_AXIS,
        "current": SIGMA_CURRENT,
    }


class Problem:
    """The weighted residuals of one synthetic data set and their exact Jacobian."""

    def __init__(self, base, where, data, sigma) -> None:
        self.base, self.where, self.data, self.sigma = base, where, data, sigma

        def residuals(x):
            prediction = predict(x, base, where)
            return jnp.concatenate(
                [(prediction[k] - data[k]) / sigma[k] for k in sorted(data)]
            )

        self.residuals = residuals
        self.jacobian = jax.jacrev(residuals)

    def fit(self, start) -> scipy.optimize.OptimizeResult:
        return scipy.optimize.least_squares(
            lambda x: np.asarray(self.residuals(jnp.asarray(x))),
            start,
            jac=lambda x: np.asarray(self.jacobian(jnp.asarray(x))),
            method="trf",
            xtol=1.0e-12,
            ftol=1.0e-12,
            gtol=1.0e-12,
            max_nfev=40,
        )


def synthetic(truth_prediction, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    sigma = sigmas(truth_prediction)
    return {
        k: np.asarray(v) + sigma[k] * rng.standard_normal(np.shape(v))
        for k, v in truth_prediction.items()
    }


def reconstruct(base, where, truth_prediction, seed: int):
    data = synthetic(truth_prediction, seed)
    problem = Problem(base, where, data, sigmas(truth_prediction))
    result = problem.fit(START)
    covariance = np.linalg.inv(result.jac.T @ result.jac)
    sigma = np.sqrt(np.diag(covariance))
    chi2 = float(result.fun @ result.fun)
    return result, covariance, sigma, chi2, result.fun.size - result.x.size


def report(result, covariance, sigma, chi2, dof) -> None:
    print(
        f"chi^2 = {chi2:.2f} for {dof} degrees of freedom, {result.nfev} residual "
        f"and {result.njev} Jacobian evaluations"
    )
    print(f"{'':8s} {'truth':>12s} {'estimate':>12s} {'sigma':>10s} {'z':>6s}")
    for k, name in enumerate(NAMES):
        z = (result.x[k] - TRUTH[k]) / sigma[k]
        print(
            f"{name:8s} {TRUTH[k] * SCALE[k]:12.4g} {result.x[k] * SCALE[k]:12.4g} "
            f"{sigma[k] * SCALE[k]:10.3g} {z:6.2f}"
        )
    correlation = covariance / np.outer(sigma, sigma)
    print("correlations:\n" + np.array2string(correlation, precision=2))
    s = np.linspace(0.0, 1.0, 6)
    gradient = jax.vmap(jax.grad(pressure), (None, 0))(
        jnp.asarray(result.x), jnp.asarray(s)
    )
    band = np.sqrt(np.einsum("ij,jk,ik->i", np.asarray(gradient), covariance, gradient))
    for s_k, p, b, t in zip(
        s, pressure(result.x, s), band, pressure(TRUTH, s), strict=True
    ):
        print(f"p({s_k:.1f}) = {p:9.0f} +- {b:6.0f} Pa (truth {t:9.0f})")


def taylor(base, where, truth_prediction) -> None:
    """Remainders of the first-order expansion of the misfit along a random direction,
    against re-solved equilibria; second order shows as a hundredfold drop per decade of
    the step."""
    problem = Problem(
        base,
        where,
        synthetic(truth_prediction, 0),
        sigmas(truth_prediction),
    )
    x = jnp.asarray(START)
    r = np.asarray(problem.residuals(x))
    gradient = np.asarray(problem.jacobian(x)).T @ r
    direction = np.random.default_rng(1).standard_normal(x.size)
    misfit = 0.5 * r @ r
    for step in (1.0e-1, 1.0e-2, 1.0e-3, 1.0e-4):
        shifted = np.asarray(problem.residuals(x + step * direction))
        remainder = abs(0.5 * shifted @ shifted - misfit - step * gradient @ direction)
        print(f"step {step:.0e}: remainder {remainder:.3e}")


def coverage(base, where, truth_prediction, count: int) -> None:
    """Frequentist calibration over noise realizations."""
    z, chi2 = [], []
    for seed in range(count):
        result, _, sigma, value, dof = reconstruct(base, where, truth_prediction, seed)
        z.append((result.x - TRUTH) / sigma)
        chi2.append(value / dof)
        print(
            f"realization {seed}: z = {np.array2string(z[-1], precision=2)}, "
            f"chi^2/dof = {chi2[-1]:.2f}",
            flush=True,
        )
    z = np.abs(np.asarray(z))
    for k, name in enumerate(NAMES):
        print(
            f"{name:8s} |z| < 1: {np.mean(z[:, k] < 1):.2f}, "
            f"|z| < 2: {np.mean(z[:, k] < 2):.2f}"
        )
    print(f"mean chi^2/dof {np.mean(chi2):.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--taylor", action="store_true")
    parser.add_argument("--coverage", type=int, default=0)
    args = parser.parse_args()
    if not vmecpp.has_exact_force_jacobian():
        print("This example needs a build with -DVMECPP_ENABLE_ENZYME=ON.")
        return
    base = base_input()
    where = positions(base)
    truth_prediction = {
        k: np.asarray(v) for k, v in predict(TRUTH, base, where).items()
    }
    if args.taylor:
        taylor(base, where, truth_prediction)
    elif args.coverage:
        coverage(base, where, truth_prediction, args.coverage)
    else:
        report(*reconstruct(base, where, truth_prediction, seed=0))


if __name__ == "__main__":
    main()
