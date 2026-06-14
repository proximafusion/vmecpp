# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
# <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""The adjoint boundary gradient matches brute-force finite differences.

dJ/d(boundary) from one Hessian solve (implicit-function adjoint) agrees with the
reference gradient obtained by re-converging the interior equilibrium for each perturbed
boundary degree of freedom. J here is the MHD energy of the converged equilibrium.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from vmecpp_adjoint import (
    boundary_gradient,
    finite_difference_boundary_gradient,
    make_model,
    mhd_energy,
    partition,
)


def test_adjoint_matches_finite_difference():
    ns = 11
    model = make_model(ns=ns)
    model.solve()
    x_star = np.asarray(model.get_state(), float).copy()
    interior, boundary = partition(model, ns)

    g_adjoint = boundary_gradient(model, x_star, interior, boundary, mhd_energy)

    # Reference on a representative subset (each requires interior re-solves).
    dofs = [0, 2, 9]
    g_fd = finite_difference_boundary_gradient(
        model, x_star, interior, boundary, mhd_energy, dofs
    )

    scale = max(np.linalg.norm(g_adjoint), 1e-30)
    for j in dofs:
        assert abs(g_adjoint[j] - g_fd[j]) < 1e-3 * scale


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
