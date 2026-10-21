# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Where a branch of equilibria ends, and how.

Near the end mu_end of a branch of equilibria in a parameter mu, the critical
eigenvalue of the force Jacobian at its equilibria vanishes as |mu - mu_end|^b. At a
pitchfork, where the branch meets a branch of higher symmetry, b = 1; at a fold, where
it meets a branch of saddle points, b = 1/2. Fitting lambda = c (mu - mu_end)^b to
equilibria computed away from the end gives mu_end and the kind of end, without solving
at mu_end, where the descent slows down.

The configuration is a spherical tokamak of major radius 1.5, minor radius 1 and
elongation 1.6, with q = 1 on the axis and low shear out to s = 0.5, and pres_scale as
mu. Its axisymmetric equilibrium is unstable to the m = 1, n = 1 internal kink, and the
helical-core equilibrium the kink leads to ends at a pitchfork, where it meets the
axisymmetric equilibrium, which loses its stability there. A small m = 1, n = 1 ripple
of the boundary breaks the mirror symmetry between the two helical cores, and the
branch of the core the ripple opposes then ends at a fold. Each branch is followed down
in pres_scale by continuation, every solve starting from the equilibrium before.

The eigenvalues are those of the force Jacobian in the coordinates in which VMEC's
preconditioner is the identity, assembled from Hessian-vector products on the free
coefficients of the decomposed state of a ``VmecModel``. Near the end of a branch, the
critical eigenvalue is the smallest one at its equilibria.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit

import vmecpp
from vmecpp.cpp import _vmecpp  # type: ignore[import]

NS = 25
MPOL = 6
NTOR = 1
FTOL = 1.0e-17
RIPPLE = 3.0e-3
PITCHFORK_PATH = [3.3e4, 3.0e4, 2.8e4, 2.7e4]
FOLD_PATH = [6.0e4, 5.2e4, 4.5e4, 4.1e4, 3.9e4, 3.85e4, 3.8e4, 3.775e4]
FOLD_MEASURED = FOLD_PATH[-4:]


def case_input(
    pres_scale: float, ripple: float = 0.0, helical_seed: float = 0.0
) -> vmecpp.VmecInput:
    """The spherical tokamak at pres_scale, with an m = 1, n = 1 boundary ripple of
    amplitude ``ripple`` in R and Z, solved from a magnetic axis guess displaced
    helically by ``helical_seed``."""
    vmec_input = vmecpp.VmecInput(
        lasym=False,
        nfp=1,
        mpol=MPOL,
        ntor=NTOR,
        ns_array=np.array([NS]),
        ftol_array=np.array([FTOL]),
        niter_array=np.array([30000]),
        phiedge=5.0,
        ncurr=0,
        # iota = 1 - (2/3) s^4: q = 1 on the axis and 3 at the boundary
        ai=np.array([1.0, 0.0, 0.0, 0.0, -2.0 / 3.0]),
        pmass_type="power_series",
        am=np.array([1.0, -4.0, 6.0, -4.0, 1.0]),  # p = p0 (1 - s)^4
        pres_scale=pres_scale,
        gamma=0.0,
        always_fix_m1_gauge=True,
        raxis_c=np.array([1.5, helical_seed]),
        zaxis_s=np.array([0.0, -helical_seed]),
    )
    vmec_input.rbc = vmecpp.VmecInput.resize_2d_coeff(
        np.array([[1.5], [1.0]]), MPOL, NTOR
    )
    vmec_input.zbs = vmecpp.VmecInput.resize_2d_coeff(
        np.array([[0.0], [1.6]]), MPOL, NTOR
    )
    vmec_input.rbc[1, NTOR + 1] = ripple
    vmec_input.zbs[1, NTOR + 1] = ripple
    return vmec_input


def create_model(vmec_input: vmecpp.VmecInput):
    return _vmecpp.VmecModel.create(vmec_input._to_cpp_vmecindata(), NS, None)


def solve(vmec_input: vmecpp.VmecInput, x_start: np.ndarray | None = None):
    """The equilibrium the native solver converges to, from the initial guess of the
    input or from the state x_start, as a VmecModel and its state."""
    model = create_model(vmec_input)
    if x_start is not None:
        model.set_state(x_start)
        model.save_backup()
    model.solve()
    model.evaluate(2, 2, False)
    return model, np.asarray(model.get_state()).copy()


def free_coefficients(model) -> np.ndarray:
    """Indices of the state coefficients the solver evolves: those with a nonzero
    preconditioned force at a state without symmetry."""
    model.evaluate(2, 2, True)
    return np.flatnonzero(np.asarray(model.get_forces()) != 0.0)


def odd_n(model) -> np.ndarray:
    """Mask of the state coefficients with odd toroidal mode number n.

    The state holds R cos cos, R sin sin, Z sin cos, Z cos sin, lambda sin cos and
    lambda cos sin, each ordered by surface, then m, then n.
    """
    n = np.indices((6, model.ns, model.mpol, model.ntor + 1))[3].ravel()
    return n % 2 == 1


def preconditioner_factor(model, x: np.ndarray, free: np.ndarray) -> np.ndarray:
    """The Cholesky factor of the preconditioner at x on the free coefficients."""
    model.set_state(x)
    model.evaluate(2, 2, True)
    columns = []
    for i in free:
        unit = np.zeros_like(x)
        unit[i] = 1.0
        # apply_preconditioner returns -M^-1 v
        columns.append(-np.asarray(model.apply_preconditioner(unit))[free])
    inverse_metric = np.array(columns).T
    return np.linalg.cholesky(0.5 * (inverse_metric + inverse_metric.T))


def smallest_eigenvalue(model, x: np.ndarray, free: np.ndarray, coordinates=None):
    """The smallest eigenvalue of the force Jacobian at x, in the coordinates in which
    the preconditioner at x is the identity, on the given coordinates, all of them by
    default."""
    factor = preconditioner_factor(model, x, free)
    model.set_state(x)
    model.evaluate(2, 2, False)
    coordinates = np.arange(free.size) if coordinates is None else coordinates
    jacobian = np.empty((coordinates.size, coordinates.size))
    for column, k in enumerate(coordinates):
        v = np.zeros_like(x)
        v[free] = factor[:, k]
        hv = np.asarray(model.hessian_vector_product(v))[free]
        jacobian[:, column] = (factor.T @ hv)[coordinates]
    return np.linalg.eigvals(jacobian).real.min()


def follow_branch(pres_scales, ripple, helical_seed, measured):
    """Follow the branch reached from ``helical_seed`` at the first pres_scale down
    through pres_scales by continuation, and return the critical eigenvalue at each
    pres_scale in ``measured``."""
    free = free_coefficients(create_model(case_input(pres_scales[0], ripple, 0.3)))
    x = None
    eigenvalues = []
    for pres_scale in pres_scales:
        model, x = solve(case_input(pres_scale, ripple, helical_seed), x)
        if pres_scale in measured:
            eigenvalues.append(smallest_eigenvalue(model, x, free))
    return np.array(eigenvalues)


def fit_branch_end(pres_scales, eigenvalues):
    """The fit lambda = c (mu - mu_end)^b, as mu_end and b.

    The fit is to log lambda, in mu measured from the lowest pres_scale in units of the
    range of pres_scales.
    """
    mu = np.asarray(pres_scales)
    span = mu.max() - mu.min()
    s = (mu - mu.min()) / span

    def law(s, log_c, s_end, b):
        return log_c + b * np.log(s - s_end)

    (_, s_end, b), _ = curve_fit(
        law,
        s,
        np.log(eigenvalues),
        p0=(np.log(eigenvalues.max()), -0.5, 0.75),
        bounds=((-np.inf, -10.0, 0.1), (np.inf, -1.0e-6, 3.0)),
    )
    return mu.min() + s_end * span, b


def axisymmetric_stability_limit(pres_scales):
    """The pres_scale at which the critical eigenvalue of the axisymmetric equilibrium
    reaches zero, extrapolated linearly from two pres_scales above it.

    The critical mode lies in the block of coefficients with odd toroidal mode number.
    """
    probe = create_model(case_input(pres_scales[0], 0.0, 0.3))
    free = free_coefficients(probe)
    odd = np.flatnonzero(odd_n(probe)[free])
    eigenvalues = []
    for pres_scale in pres_scales:
        model, x = solve(case_input(pres_scale))
        eigenvalues.append(smallest_eigenvalue(model, x, free, odd))
    (mu1, mu2), (l1, l2) = pres_scales, eigenvalues
    return mu2 - l2 * (mu2 - mu1) / (l2 - l1)


def pitchfork_case():
    """The critical eigenvalues along the helical-core branch without ripple."""
    return PITCHFORK_PATH, follow_branch(PITCHFORK_PATH, 0.0, 0.3, PITCHFORK_PATH)


def fold_case():
    """The critical eigenvalues along the helical-core branch the ripple opposes."""
    return FOLD_MEASURED, follow_branch(FOLD_PATH, RIPPLE, -0.3, FOLD_MEASURED)


def main():
    for name, (pres_scales, eigenvalues) in (
        ("without ripple", pitchfork_case()),
        ("with ripple", fold_case()),
    ):
        mu_end, b = fit_branch_end(pres_scales, eigenvalues)
        kind = "pitchfork" if b > 0.75 else "fold"
        print(f"{name}:")
        for pres_scale, eigenvalue in zip(pres_scales, eigenvalues, strict=True):
            print(
                f"  pres_scale {pres_scale:.4e}: critical eigenvalue {eigenvalue:.4e}"
            )
        print(
            f"  the branch ends at pres_scale {mu_end:.4e} with b = {b:.2f}: a {kind}"
        )
    limit = axisymmetric_stability_limit([2.8e4, 2.7e4])
    print(f"the axisymmetric equilibrium loses its stability at pres_scale {limit:.4e}")


if __name__ == "__main__":
    main()
