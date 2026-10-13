"""Interfaces shared by the adjoint verification work packages.

State vectors and generator rows use ``VmecModel.get_state()`` ordering and
scaling, not physical Geometry coefficients. Class arrays are sorted flat
interior indices (LCFS R/Z entries are prescribed separately). They partition
the interior: ``null`` has zero rows and columns, ``pinned`` has zero rows but
nonzero columns, ``zero_column`` has nonzero rows but zero columns, and
``active`` has both nonzero. Analytic initialization rules distinguish pinned
entries from accidental numerical zeros.

Residuals refer to the unpreconditioned decomposed force from
``evaluate(2, 2, False, True)`` followed by ``get_forces()``. This refreshes
tcon from the state, as does the exact force Jacobian; profile/constraint
reference data remain fixed. Copy the residual before HVPs overwrite the
model's force buffers. Norms are
Euclidean; solve_residual is relative to the active cotangent norm. block_rcond
is the minimum reciprocal condition estimate of the block Schur complements.
cotangent_on_null/pinned are norms before projection or pull-back.
adjoint_weighted_residual is the signed scalar lambda^T F; relabeling overlap
is the norm of the orthogonal projection of the cotangent onto the generator
span, divided by the cotangent norm (zero for a zero cotangent).

Stubs deliberately fail rather than supply unverified numerical results.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


@dataclass(frozen=True)
class ImplicitVjpDiagnostics:
    """One adjoint linearization, in the state/force basis specified above."""

    force_residual: float
    solve_residual: float
    block_rcond: float
    class_sizes: dict[str, int]
    cotangent_on_null: float
    cotangent_on_pinned: float
    adjoint_weighted_residual: float
    max_relabeling_overlap: float | None = None


class Factorization(Protocol):
    """Active-set factorization of H^T, reusable for either orientation."""

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        """Solve H^T y = rhs in active-index ordering."""
        ...

    def solve_transpose(self, rhs: np.ndarray) -> np.ndarray:
        """Solve H y = rhs in active-index ordering."""
        ...


@dataclass(frozen=True)
class NewtonResult:
    """Final full state, force norms including the initial state, and H^T factors.

    ``active`` maps factorization coordinates into the full state. The handle
    must be assembled at the returned state, not at the preceding Newton step.
    ``converged`` distinguishes exhaustion/line-search failure from convergence.
    Newton polishing mutates the model to the returned state.
    """

    state: np.ndarray
    residual_history: np.ndarray
    factorization: Factorization
    active: np.ndarray
    converged: bool


def classify_state_entries(model: Any) -> dict[str, np.ndarray]:
    """Return disjoint interior classes: null, pinned, zero_column, active."""
    del model
    error_message = "WP3: state classification is not implemented"
    raise NotImplementedError(error_message)


def newton_polish(model: Any, max_steps: int, tol: float) -> NewtonResult:
    """Polish the active force residual with backtracking; preserve prescribed DOFs."""
    del model, max_steps, tol
    error_message = "WP5: Newton polishing is not implemented"
    raise NotImplementedError(error_message)


def relabeling_generators(model: Any) -> np.ndarray:
    """Return generator columns in the scaled full-state layout."""
    del model
    error_message = "WP4: relabeling generators are not implemented"
    raise NotImplementedError(error_message)


def corrected_vjp(
    model: Any, x_bar: np.ndarray, method: str
) -> tuple[np.ndarray, float, ImplicitVjpDiagnostics]:
    """Return boundary gradient, additive value correction, and diagnostics.

    ``x_bar`` is the objective cotangent in full-state coordinates. ``method``
    is ``"A"`` (Newton and relinearize) or ``"B"`` (second-order adjoint).
    The value correction is added to J, i.e. it is -lambda^T F.
    """
    del model, x_bar, method
    error_message = "WP7: corrected VJP is not implemented"
    raise NotImplementedError(error_message)
