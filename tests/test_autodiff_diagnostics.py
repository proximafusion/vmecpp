"""WP1 raw-force buffer and evaluation contract."""

import numpy as np
import pytest

from vmecpp.autodiff_diagnostics import raw_force


class _Model:
    always_fix_m1_gauge = True
    need_restart = False

    def __init__(self):
        self.forces = np.array([1.0, 2.0])
        self.calls = []

    def evaluate(self, *args):
        self.calls.append(args)

    def get_forces(self):
        return self.forces


def test_raw_force_refreshes_without_preconditioning_and_owns_buffer():
    model = _Model()
    result = raw_force(model)
    assert model.calls == [(2, 2, False, True)]
    model.forces[:] = 3.0
    np.testing.assert_array_equal(result, [1.0, 2.0])


def test_raw_force_rejects_unpinned_gauge():
    model = _Model()
    model.always_fix_m1_gauge = False
    with pytest.raises(ValueError, match="always_fix_m1_gauge"):
        raw_force(model)
    assert not model.calls


def test_raw_force_rejects_invalid_jacobian():
    model = _Model()
    model.need_restart = True
    with pytest.raises(RuntimeError, match="valid Jacobian"):
        raw_force(model)
