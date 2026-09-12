"""``autodiff.state_layout``'s index sets and their agreement with the private helpers
it replaces."""

from pathlib import Path

import numpy as np
import pytest

import vmecpp
from vmecpp import autodiff
from vmecpp.cpp import _vmecpp  # type: ignore

SOLOVEV = Path(__file__).resolve().parents[1] / "examples" / "data" / "solovev.json"


def _requires_exact_derivatives(model) -> None:
    if not model.has_exact_force_jacobian:
        pytest.skip("needs an Enzyme-enabled build for the exact residual transpose")


def _2d_model(ns: int = 5):
    indata = _vmecpp.VmecINDATA.from_file(str(SOLOVEV))
    model = _vmecpp.VmecModel.create(indata, ns)
    model.solve()
    return model


def _3d_model(ns: int = 5):
    """A genuinely three-dimensional fixed-boundary, ncurr=0 case.

    The 2D case cannot exercise the ``r_ss``, ``z_cs`` and ``lambda_cs``
    blocks at all, and it is exactly the ``z_cs``/``z_cc`` blocks that carry
    the m=1 constraint gauge entries.
    """
    source = vmecpp.VmecInput.from_file(str(SOLOVEV))
    mpol = source.mpol
    rbc = np.zeros((mpol, 3))
    zbs = np.zeros((mpol, 3))
    rbc[:, 1] = np.asarray(source.rbc)[:, 0]
    zbs[:, 1] = np.asarray(source.zbs)[:, 0]
    rbc[1, 2] = 0.01
    zbs[1, 2] = 0.01
    indata = source.model_copy(
        update={
            "ntor": 1,
            "rbc": rbc,
            "zbs": zbs,
            "raxis_c": np.asarray([4.0, 0.0]),
            "zaxis_s": np.asarray([0.0, 0.0]),
            "ns_array": np.asarray([ns]),
            "ftol_array": np.asarray([1.0e-15]),
            "niter_array": np.asarray([4000]),
        }
    )
    model = _vmecpp.VmecModel.create(indata._to_cpp_vmecindata(), ns)
    model.solve()
    return model


@pytest.mark.parametrize("make_model", [_2d_model, _3d_model])
def test_index_sets_partition_the_state(make_model) -> None:
    model = make_model()
    _requires_exact_derivatives(model)
    model.evaluate(2, 2, True)
    layout = autodiff.state_layout(model)

    state_size = int(np.asarray(model.get_state()).size)
    assert np.array_equal(
        np.sort(np.concatenate([layout.boundary, layout.interior])),
        np.arange(state_size),
    )
    assert np.intersect1d(layout.boundary, layout.interior).size == 0

    reassembled_interior = np.sort(
        np.concatenate([layout.structural_zero, layout.gauge, layout.solved])
    )
    assert np.array_equal(reassembled_interior, np.sort(layout.interior))
    for first, second in (
        (layout.structural_zero, layout.gauge),
        (layout.structural_zero, layout.solved),
        (layout.gauge, layout.solved),
    ):
        assert np.intersect1d(first, second).size == 0


@pytest.mark.parametrize("make_model", [_2d_model, _3d_model])
def test_solved_matches_the_previous_structural_nullfree_interior(make_model) -> None:
    """``solved`` must equal the interior DOFs the adjoint solve used before this
    change, minus the m=1 gauge rows introduced by this layout."""
    model = make_model()
    _requires_exact_derivatives(model)
    model.evaluate(2, 2, True)

    interior, _ = autodiff._interior_and_boundary(model)
    nullfree = autodiff._structural_nullfree_interior(model, interior)
    layout = autodiff.state_layout(model)

    expected_solved = np.setdiff1d(nullfree, layout.gauge)
    assert np.array_equal(np.sort(layout.solved), np.sort(expected_solved))


def test_gauge_is_empty_in_two_dimensions() -> None:
    """2D has no lthreed/lasym blocks, so there is nothing for zeroZForceForM1 to
    zero."""
    model = _2d_model()
    _requires_exact_derivatives(model)
    model.evaluate(2, 2, True)
    layout = autodiff.state_layout(model)
    assert layout.gauge.size == 0


def test_gauge_force_rows_are_zero() -> None:
    model = _3d_model()
    _requires_exact_derivatives(model)
    model.evaluate(2, 2, True)
    layout = autodiff.state_layout(model)

    assert layout.gauge.size > 0
    forces = np.asarray(model.get_forces(), dtype=np.float64)
    np.testing.assert_allclose(forces[layout.gauge], 0.0, atol=1.0e-12)


def test_spans_partition_the_flat_state() -> None:
    model = _2d_model()
    layout = autodiff.state_layout(model)
    state_size = int(np.asarray(model.get_state()).size)
    covered = np.zeros(state_size, dtype=bool)
    for span in layout.spans.values():
        assert not covered[span].any()
        covered[span] = True
    assert covered.all()


def test_surface_and_mode_indices_agree_with_the_span_shape() -> None:
    model = _3d_model()
    layout = autodiff.state_layout(model)
    modes_per_surface = model.mpol * (model.ntor + 1)
    for name, span in layout.spans.items():
        span_length = span.stop - span.start
        assert layout.surface[name].shape == (span_length,)
        assert layout.mode[name].shape == (span_length,)
        assert layout.surface[name].max() == model.ns - 1
        assert layout.mode[name].max() == modes_per_surface - 1
