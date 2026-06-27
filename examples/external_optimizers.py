# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
# <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Drive a VMEC++ equilibrium from outside with general-purpose optimizers.

VMEC's equilibrium is the stationary point of its augmented functional (MHD
energy plus the spectral-condensation and lambda constraints). The gradient of
that functional in the decomposed internal basis is the raw, unpreconditioned
force exposed by ``VmecModel.evaluate(precondition=False)`` (see
``tests/test_internal_gradient.py``). Finding the equilibrium is therefore the
root problem F(x) = 0, which any gradient/Hessian-based solver can attack while
reusing VMEC++'s forward model.

This module wires that residual to two solvers and is used both by the benchmark
``main`` below and by ``tests/test_external_optimizers.py``:

* native-style preconditioned descent (heavy-ball on the preconditioned search
  direction, i.e. VMEC's own update), and
* Jacobian-free Newton-Krylov (matrix-free Hessian information).

Both converge to the same equilibrium as the native solver. Quasi-Newton
root-finders without a preconditioner diverge on this stiff system, which is why
VMEC's preconditioner matters; exposing it as an operator is a follow-up.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import newton_krylov

from vmecpp.cpp import _vmecpp  # type: ignore[import]

DEFAULT_INPUT = (
    Path(__file__).resolve().parents[1] / "examples" / "data" / "solovev.json"
)


_BAD_JACOBIAN = 2  # RestartReason::BAD_JACOBIAN (flow_control.h)


def _ensure_valid_initial_jacobian(model, max_reguess: int = 2) -> None:
    """Reguess the magnetic axis until the initial Jacobian is non-singular.

    Inputs that ship no axis (e.g. cma.json, with raxis/zaxis all zero) start from a
    degenerate geometry, so the bare forward model returns at the BAD_JACOBIAN
    checkpoint with zero force. The native solver reguesses the axis on the first
    iterate (vmec.cc SolveEquilibriumLoop); mirror that here via reinitialize() so a
    cold start from any valid INDATA has a defined gradient.
    """
    for _ in range(max_reguess):
        model.evaluate(2, 2, False)
        if model.restart_reason != _BAD_JACOBIAN:
            return
        model.reinitialize()
    model.evaluate(2, 2, False)


def make_model(input_path: Path = DEFAULT_INPUT, ns: int = 11):
    indata = _vmecpp.VmecINDATA.from_file(str(input_path))
    model = _vmecpp.VmecModel.create(indata, ns)
    _ensure_valid_initial_jacobian(model)
    return model


def residual(model):
    """Return F(x) = raw internal-basis force; F(x) = 0 at equilibrium."""

    def F(x):
        model.set_state(np.ascontiguousarray(x, dtype=float))
        model.evaluate(2, 2, False)
        return np.asarray(model.get_forces(), dtype=float)

    return F


@dataclass
class Result:
    name: str
    force_evals: int
    seconds: float
    residual_norm: float
    energy: float


def reference_equilibrium(input_path: Path = DEFAULT_INPUT, ns: int = 11):
    model = make_model(input_path, ns)
    model.solve()
    model.evaluate(2, 2, False)
    return np.asarray(model.get_state(), float), model.mhd_energy


def solve_preconditioned_descent(
    input_path=DEFAULT_INPUT, ns=11, tol=1e-9, delt=0.9, momentum=0.5, max_iter=20000
):
    model = make_model(input_path, ns)
    F = residual(model)
    x = np.asarray(model.get_state(), float).copy()
    v = np.zeros_like(x)
    evals = 0
    t0 = time.perf_counter()
    for _ in range(max_iter):
        if np.linalg.norm(F(x)) < tol:
            break
        evals += 1
        model.set_state(np.ascontiguousarray(x))
        model.evaluate(2, 2, True)  # preconditioned search direction
        fprec = np.asarray(model.get_forces(), float)
        v = momentum * v + delt * fprec
        x = x + delt * v
    model.set_state(np.ascontiguousarray(x))
    model.evaluate(2, 2, False)
    return x, Result(
        "preconditioned descent",
        evals,
        time.perf_counter() - t0,
        float(np.linalg.norm(np.asarray(model.get_forces(), float))),
        model.mhd_energy,
    )


def solve_newton_krylov(input_path=DEFAULT_INPUT, ns=11, tol=1e-9, max_iter=200):
    model = make_model(input_path, ns)
    F = residual(model)
    n = [0]

    def counted(x):
        n[0] += 1
        return F(x)

    x0 = np.asarray(model.get_state(), float)
    t0 = time.perf_counter()
    x = newton_krylov(counted, x0, f_tol=tol, maxiter=max_iter, method="lgmres")
    model.set_state(np.ascontiguousarray(x))
    model.evaluate(2, 2, False)
    return x, Result(
        "Newton-Krylov (JFNK)",
        n[0],
        time.perf_counter() - t0,
        float(np.linalg.norm(np.asarray(model.get_forces(), float))),
        model.mhd_energy,
    )


def main():
    _, w_star = reference_equilibrium()
    print(f"reference equilibrium (native solve): W = {w_star:.8e}\n")
    rows = [solve_preconditioned_descent()[1], solve_newton_krylov()[1]]
    print(
        f"{'optimizer':28s} {'F-evals':>8s} {'time[s]':>8s} "
        f"{'||F||':>10s} {'dW vs ref':>10s}"
    )
    for r in rows:
        print(
            f"{r.name:28s} {r.force_evals:8d} {r.seconds:8.2f} "
            f"{r.residual_norm:10.1e} {abs(r.energy - w_star):10.1e}"
        )


if __name__ == "__main__":
    main()
