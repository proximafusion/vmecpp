"""WP5 algebra checks independent of the Enzyme runtime."""

import numpy as np
import pytest
import scipy.linalg

from vmecpp.autodiff import _BlockTridiagonal


@pytest.mark.parametrize(
    "surface",
    [
        np.array([0, 0, 0]),
        np.array([2, 0, 1, 2, 1, 2, 0]),
        np.array([0, 2, 2]),
    ],
)
def test_both_solve_orientations_match_dense(surface):
    generator = np.random.default_rng(5)
    dense = generator.standard_normal((surface.size, surface.size))
    dense[np.abs(surface[:, None] - surface[None, :]) > 1] = 0.0
    dense += 5.0 * np.eye(surface.size)
    # Force row pivoting in at least one diagonal block.
    first = np.flatnonzero(surface == surface[0])
    if first.size > 1:
        dense[first[0], first[0]] = 0.01
        dense[first[1], first[0]] = 4.0
    matrix = _BlockTridiagonal(surface)
    rows, columns = np.nonzero(dense)
    matrix.set_entries(rows, columns, dense[rows, columns])
    rhs = generator.standard_normal(surface.size)
    np.testing.assert_allclose(matrix.matvec(rhs), dense @ rhs, atol=1e-14)
    matrix.factorize()
    for _ in range(3):
        rhs = generator.standard_normal(surface.size)
        np.testing.assert_allclose(
            matrix.solve(rhs), scipy.linalg.solve(dense, rhs), atol=1e-12
        )
        np.testing.assert_allclose(
            matrix.solve_transpose(rhs),
            scipy.linalg.solve(dense.T, rhs),
            atol=1e-12,
        )
