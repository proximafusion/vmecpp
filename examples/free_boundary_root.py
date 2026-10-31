# SPDX-FileCopyrightText: 2026-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Free-boundary equilibria as roots of the pressure jump across the boundary, with the
stability of each shape mode from the eigenvalues of the matching's Jacobian."""

import argparse
import dataclasses
import functools
import multiprocessing
import os

import numpy as np

import vmecpp

MU0 = 4.0e-7 * np.pi
STEP = 1.0e-6  # finite-difference step of the unknowns, in m and T
MAX_STEP = 1.0e-3  # largest change of an unknown in one step, in m and T
PENALTY = 10.0  # jump mode of a boundary that VMEC++ does not solve


@dataclasses.dataclass(frozen=True)
class Matching:
    """A boundary moved along its normal in modes (m, n, parity), 0 cos and 1 sin, in a
    field whose parameters stand in for held modes, None for the input's mgrid."""

    vmec_input: vmecpp.VmecInput
    modes: tuple
    field: object = None
    held: tuple = ()
    b_scale: float = 1.0

    @functools.cached_property
    def sizes(self):
        """The numbers of toroidal and poloidal points of the solver's surfaces."""
        inp = self.vmec_input
        ntheta = 2 * (max(inp.ntheta, 2 * inp.mpol + 6) // 2)
        nzeta = max(inp.nzeta, 2 * inp.ntor + 4) if inp.ntor > 0 else max(inp.nzeta, 1)
        return nzeta, ntheta

    def full_grid(self):
        nzeta, ntheta = self.sizes
        theta = 2.0 * np.pi * np.arange(ntheta) / ntheta
        zeta = 2.0 * np.pi * np.arange(nzeta) / (nzeta * self.vmec_input.nfp)
        return zeta, theta

    @functools.cached_property
    def grid(self):
        """The toroidal and poloidal angles of the boundary points of the solver."""
        zeta, theta = self.full_grid()
        if not self.vmec_input.lasym:
            theta = theta[: len(theta) // 2 + 1]
        return zeta, theta

    @functools.cached_property
    def free(self):
        return [i for i, mode in enumerate(self.modes) if mode not in self.held]

    def basis(self, zeta, theta):
        """The shape modes at the angles, one column each."""
        nfp = self.vmec_input.nfp
        angle = [m * theta[None, :] - n * nfp * zeta[:, None] for m, n, _ in self.modes]
        return np.stack(
            [
                (np.cos if parity == 0 else np.sin)(a).ravel()
                for a, (_, _, parity) in zip(angle, self.modes, strict=True)
            ],
            1,
        )

    @functools.cached_property
    def projection(self):
        """Least-squares projection onto the shape modes of a function on the solver's
        boundary grid, the end points of the half poloidal range at half weight."""
        zeta, theta = self.grid
        weight = np.ones((len(zeta), len(theta)))
        if not self.vmec_input.lasym:
            weight[:, [0, -1]] = 0.5
        root = np.sqrt(weight.ravel())
        return np.linalg.pinv(root[:, None] * self.basis(zeta, theta)) * root

    def reference(self, zeta, theta):
        """R, Z and the unit normal in the toroidal plane of the reference boundary."""
        inp = self.vmec_input
        m = np.arange(inp.mpol)[:, None, None, None]
        n = np.arange(-inp.ntor, inp.ntor + 1)[None, :, None, None]
        angle = m * theta[None, None, None, :] - n * inp.nfp * zeta[None, None, :, None]
        cos, sin = np.cos(angle), np.sin(angle)
        rbc, zbs = inp.rbc[..., None, None], inp.zbs[..., None, None]
        r, z = np.sum(rbc * cos, (0, 1)), np.sum(zbs * sin, (0, 1))
        r_t, z_t = -np.sum(rbc * m * sin, (0, 1)), np.sum(zbs * m * cos, (0, 1))
        if inp.lasym:
            rbs, zbc = inp.rbs[..., None, None], inp.zbc[..., None, None]
            r, z = r + np.sum(rbs * sin, (0, 1)), z + np.sum(zbc * cos, (0, 1))
            r_t = r_t + np.sum(rbs * m * cos, (0, 1))
            z_t = z_t - np.sum(zbc * m * sin, (0, 1))
        # outward whichever way theta runs: the sign of each plane's enclosed area
        sign = np.sign(np.sum(r * z_t - z * r_t, axis=-1, keepdims=True))
        length = sign * np.hypot(r_t, z_t)
        return r, z, z_t / length, -r_t / length

    def boundary(self, x):
        """The Fourier coefficients of the reference boundary moved by sum x_k mode_k
        along its normal, fitted on the solver's full grid."""
        inp = self.vmec_input
        zeta, theta = self.full_grid()
        r, z, n_r, n_z = self.reference(zeta, theta)
        shift = (self.basis(zeta, theta) @ x).reshape(r.shape)
        r, z = r + shift * n_r, z + shift * n_z
        m = np.arange(inp.mpol)[:, None, None, None]
        n = np.arange(-inp.ntor, inp.ntor + 1)[None, :, None, None]
        angle = m * theta[None, None, None, :] - n * inp.nfp * zeta[None, None, :, None]
        weight = np.where((m == 0) & (n == 0), 1.0, 2.0) / r.size
        # one copy of each mode: m = 0 keeps n >= 0
        keep = (m > 0) | (n >= 0)
        coefficients = {}
        for name, values, kind in (
            ("rbc", r, np.cos),
            ("zbs", z, np.sin),
            ("rbs", r, np.sin),
            ("zbc", z, np.cos),
        ):
            if name in ("rbs", "zbc") and not inp.lasym:
                continue
            c = weight * np.sum(values * kind(angle), (2, 3), keepdims=True)
            coefficients[name] = np.where(keep, c, 0.0)[..., 0, 0]
        return coefficients

    def unknowns(self, x, parameters):
        return np.concatenate([np.asarray(x)[self.free], parameters])

    def split(self, x0, y):
        x = np.array(x0, dtype=float)
        x[self.free] = y[: len(self.free)]
        return x, y[len(self.free) :]


def solve(matching, x, parameters, start=None):
    """The fixed-boundary run inside the boundary of x, hot-started from start, and its
    first free-boundary iteration in the external field of the parameters."""
    inp = matching.vmec_input
    # the m = 1 gauge held at the boundary's makes the jump a function of the boundary
    fixed_input = inp.model_copy(
        update={**matching.boundary(x), "lfreeb": False, "always_fix_m1_gauge": True}
    )
    fixed = vmecpp.run(fixed_input, max_threads=1, verbose=False, restart_from=start)
    update = {
        "lfreeb": True,
        "ns_array": inp.ns_array[-1:],
        "ftol_array": inp.ftol_array[-1:],
        "niter_array": np.array([1]),
        "return_vacuum_field": True,
    }
    field = None
    if matching.field is not None:
        update["mgrid_file"] = "NONE"
        field = matching.field(parameters)
    free = vmecpp.run(
        fixed_input.model_copy(update=update),
        field,
        max_threads=1,
        verbose=False,
        restart_from=fixed,
    )
    return fixed, free


def edge_pressure(wout, zeta, theta):
    """The total pressure p + B^2 / 2 of the plasma at the angles on the boundary,
    extrapolated linearly in s from the two outermost half-grid surfaces."""
    angle = (
        wout.xm_nyq[:, None, None] * theta[None, None, :]
        - wout.xn_nyq[:, None, None] * zeta[None, :, None]
    )
    pressure = []
    for j in (wout.ns - 1, wout.ns - 2):
        b = np.tensordot(wout.bmnc[:, j], np.cos(angle), 1)
        if wout.lasym:
            b = b + np.tensordot(wout.bmns[:, j], np.sin(angle), 1)
        pressure.append(0.5 * b**2 + MU0 * wout.pres[j])
    return 1.5 * pressure[0] - 0.5 * pressure[1]


def evaluate(matching, x0, start, y):
    """The jump in total pressure, plasma minus vacuum, in the shape modes and units of
    b_scale^2 / 2, and the fixed run; PENALTY and None where VMEC++ fails."""
    x, parameters = matching.split(x0, y)
    try:
        fixed, free = solve(matching, x, parameters, start)
    except RuntimeError:
        return np.full(len(matching.modes), PENALTY), None
    if fixed.wout.ier_flag != 0:
        return np.full(len(matching.modes), PENALTY), None
    # the vacuum pressure of the first free-boundary evaluation, on the boundary of x
    vacuum = np.asarray(free.threed1_free_boundary.bsqvaci)
    jump = (edge_pressure(fixed.wout, *matching.grid) - vacuum).ravel()
    return matching.projection @ (jump / (0.5 * matching.b_scale**2)), fixed


def residual(matching, x0, start, y):
    return evaluate(matching, x0, start, y)[0]


def jacobian(pool, matching, x0, y, f, start=None):
    columns = pool.map(
        functools.partial(residual, matching, x0, start),
        [y + STEP * e for e in np.eye(y.size)],
    )
    return np.column_stack([(c - f) / STEP for c in columns])


def match(matching, x0, parameters, pool, report=print, steps=40, tol=1e-9):
    """Levenberg-Marquardt on the shape modes and the parameters until every jump mode
    is below tol, returning the shape, the parameters and the fixed-boundary run."""
    y = matching.unknowns(x0, parameters)
    f, start = evaluate(matching, x0, None, y)
    damping = 1e-3
    for step in range(steps):
        report(step, f)
        if np.abs(f).max() < tol:
            break
        hot = start if np.abs(f).max() > 1e-4 else None
        jac = jacobian(pool, matching, x0, y, residual(matching, x0, hot, y), hot)
        gradient, normal = jac.T @ f, jac.T @ jac
        for _ in range(12):
            move = -np.linalg.solve(
                normal + damping * np.diag(np.diag(normal)), gradient
            )
            move *= min(1.0, MAX_STEP / np.abs(move).max())
            trial, state = evaluate(matching, x0, hot, y + move)
            if state is not None and np.sum(trial**2) < np.sum(f**2):
                y, f, start = y + move, trial, state
                damping = max(damping / 3.0, 1e-9)
                break
            damping *= 4.0
        else:
            break
    x, parameters = matching.split(x0, y)
    return x, parameters, start, f


def stiffness(matching, x, pool):
    """Eigenvalues of the matching's Jacobian at x, softest first, and the modes that
    dominate their eigenvectors; a negative eigenvalue is a stable mode."""
    y = np.asarray(x, dtype=float)
    jac = jacobian(pool, matching, x, y, residual(matching, x, None, y))
    values, vectors = np.linalg.eig(jac)
    order = np.argsort(np.abs(values))
    modes = [matching.modes[np.argmax(np.abs(vectors[:, i]))] for i in order]
    return values[order].real, modes


# The tokamak of the example: a circular ring of major radius R0 and minor radius A in
# the field B0 R0 / R along phi and a vertical field of decay index n at (R0, 0).
R0, A, B0 = 2.5, 0.5, 1.0


def decay_field(vertical, decay):
    """B0 R0 / R along phi plus the uniform field and toroidal quadrupole of the flux
    vertical ((4 + 2 n) R^2 - n (R^4 - 4 R^2 Z^2) / R0^2) / 8, decay index n at R0."""

    def field(r, _phi, z):
        b_z = vertical * (1.0 + 0.5 * decay * (1.0 - (r**2 - 2.0 * z**2) / R0**2))
        b_r = -vertical * decay * r * z / R0**2
        return b_r, B0 * R0 / r, b_z

    return field


@dataclasses.dataclass(frozen=True)
class DecayField:
    """The field of decay index decay, its vertical field given or, when None, the first
    parameter."""

    decay: float
    vertical: float | None = None

    def __call__(self, parameters):
        vertical = parameters[0] if self.vertical is None else self.vertical
        return decay_field(vertical, self.decay)


def ring_input(current, lasym, ns=25):
    """The circular tokamak at zero pressure with a uniform current density."""
    mpol = 6
    rbc = np.zeros((mpol, 1))
    zbs = np.zeros((mpol, 1))
    rbc[0, 0], rbc[1, 0], zbs[1, 0] = R0, A, A
    update = {}
    if lasym:
        update = {
            "rbs": np.zeros((mpol, 1)),
            "zbc": np.zeros((mpol, 1)),
            "raxis_s": np.zeros(1),
            "zaxis_c": np.zeros(1),
        }
    return vmecpp.VmecInput(
        nfp=1,
        lasym=lasym,
        mpol=mpol,
        ntor=0,
        ntheta=24,
        nzeta=1,
        ns_array=np.array([ns]),
        ftol_array=np.array([1e-14]),
        niter_array=np.array([20000]),
        phiedge=np.pi * A**2 * B0,
        ncurr=1,
        pcurr_type="power_series",
        ac=np.array([1.0]),
        curtor=current,
        pmass_type="power_series",
        am=np.array([0.0]),
        pres_scale=0.0,
        gamma=0.0,
        rbc=rbc,
        zbs=zbs,
        raxis_c=np.array([R0]),
        zaxis_s=np.array([0.0]),
        nvacskip=1,
        return_outputs_even_if_not_converged=True,
        **update,
    )


def balancing_field(current):
    """The vertical field of a thin ring with uniform current density at R0."""
    return -MU0 * current / (4.0 * np.pi * R0) * (np.log(8.0 * R0 / A) - 1.25)


def ring_matching(current, decay, lasym=True, m_max=3):
    """The ring's matching on modes m <= m_max of both parities where allowed, with the
    radial shift held and the vertical field free."""
    parities = (0, 1) if lasym else (0,)
    modes = tuple(
        (m, 0, p) for p in parities for m in range(m_max + 1) if not (m == 0 and p)
    )
    return Matching(
        ring_input(current, lasym), modes, DecayField(decay), ((1, 0, 0),), B0
    )


def main():
    parser = argparse.ArgumentParser(
        description="free-boundary equilibria of a ring as roots, and the stiffness of "
        "its shape modes against the decay index of its vertical field"
    )
    parser.add_argument("--current", type=float, default=250.0e3)
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument(
        "--decay",
        type=float,
        nargs="+",
        default=[-0.2, -0.1, 0.0, 0.1, 0.2],
        help="decay indices of the vertical field",
    )
    args = parser.parse_args()
    with multiprocessing.get_context("spawn").Pool(args.workers) as pool:
        for decay in args.decay:
            held = ring_matching(args.current, decay)
            x0 = np.zeros(len(held.modes))
            x, parameters, _, f = match(
                held, x0, [balancing_field(args.current)], pool, lambda *_: None
            )
            # in the vertical field found, every shape mode free
            free = dataclasses.replace(
                held, field=DecayField(decay, parameters[0]), held=()
            )
            values, modes = stiffness(free, x, pool)
            print(
                f"decay index {decay:+.2f}: vertical field {parameters[0] * 1e3:+.4f} mT, "
                f"largest jump mode {np.abs(f).max():.0e}; softest eigenvalues "
                + ", ".join(
                    f"{v:+.3e} {m}" for v, m in zip(values[:3], modes[:3], strict=True)
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
