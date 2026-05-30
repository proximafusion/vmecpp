# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Couple the Python force-balance iteration to boundary-shape optimization.

Owning the equilibrium iteration in Python (:mod:`vmecpp._iteration`) lets a
stellarator-optimization objective drive the plasma boundary directly: for each
candidate boundary the Python iteration solves the fixed-boundary equilibrium to
force balance, and a derivative-free optimizer minimizes a scalar objective
evaluated on the converged equilibrium. This is the nested (solve-then-evaluate)
structure underlying codes such as SIMSOPT and STELLOPT.

The per-step forward model that :class:`vmecpp.cpp._vmecpp.VmecModel` exposes
additionally makes a single-loop scheme possible (interleaving force-balance
steps with objective-gradient steps on the boundary). :func:`optimize_boundary`
implements the nested form; it is the foundation either approach builds on.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from vmecpp._iteration import iterate

_logger = logging.getLogger(__name__)

# Objective value returned for a boundary the iteration could not bring to force
# balance, so the optimizer steers away from non-convergent regions.
_NONCONVERGENT_PENALTY = 1.0e30


@dataclass
class BoundaryOptimizationResult:
    """Outcome of :func:`optimize_boundary`."""

    x: np.ndarray
    objective: float
    num_evaluations: int
    converged: bool  # whether the best boundary's equilibrium reached force balance


def optimize_boundary(
    base_input,
    set_params: Callable,
    objective: Callable,
    x0,
    *,
    ns: int | None = None,
    iteration_style: str | None = None,
    method: str = "Nelder-Mead",
    verbose: bool = False,
    **minimize_kwargs,
) -> BoundaryOptimizationResult:
    """Minimize a scalar objective over boundary parameters.

    For each candidate optimization vector ``x``, ``set_params(base_input, x)``
    produces a configuration with a modified boundary, the Python force-balance
    iteration (:func:`vmecpp.iterate`) solves it, and ``objective(model, result)``
    is evaluated on the converged equilibrium. A derivative-free optimizer
    minimizes that objective over ``x``.

    Parameters
    ----------
    base_input:
        Base configuration whose boundary is varied.
    set_params:
        ``set_params(base_input, x) -> VmecInput`` mapping the optimization vector
        ``x`` to a configuration with a modified boundary. It must return a new
        (or copied) input rather than mutating ``base_input``.
    objective:
        ``objective(model, result) -> float`` evaluated on the converged
        equilibrium. A non-convergent candidate is given a large penalty so the
        optimizer avoids it.
    x0:
        Initial optimization vector.
    ns, iteration_style:
        Forwarded to :func:`vmecpp.iterate` for every equilibrium solve.
    method, minimize_kwargs:
        Forwarded to :func:`scipy.optimize.minimize`.

    Returns
    -------
    BoundaryOptimizationResult
        The best boundary found, its objective value, the number of equilibrium
        solves, and whether that equilibrium converged.
    """
    x0 = np.atleast_1d(np.asarray(x0, dtype=float))
    n_evaluations = 0
    best = {"x": x0.copy(), "objective": float("inf"), "converged": False}

    def evaluate(x: np.ndarray) -> float:
        nonlocal n_evaluations
        n_evaluations += 1
        vmec_input = set_params(base_input, x)
        model, result = iterate(vmec_input, ns=ns, iteration_style=iteration_style)
        value = (
            float(objective(model, result))
            if result.converged
            else _NONCONVERGENT_PENALTY
        )
        if verbose:
            _logger.info(
                "eval %d: x=%s converged=%d J=%.6e",
                n_evaluations,
                np.array2string(np.atleast_1d(x), precision=5),
                int(result.converged),
                value,
            )
        if value < best["objective"]:
            best.update(
                x=np.array(x, dtype=float),
                objective=value,
                converged=result.converged,
            )
        return value

    minimize(evaluate, x0, method=method, **minimize_kwargs)
    return BoundaryOptimizationResult(
        x=np.atleast_1d(best["x"]),
        objective=float(best["objective"]),
        num_evaluations=n_evaluations,
        converged=bool(best["converged"]),
    )
