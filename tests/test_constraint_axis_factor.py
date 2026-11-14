"""The constraint force raised next to the axis by VmecInput.tcon_axis_factor."""

from pathlib import Path

import numpy as np
import pytest

import vmecpp
from vmecpp.cpp import _vmecpp  # type: ignore

TEST_DATA = Path(__file__).resolve().parents[1] / "src/vmecpp/cpp/vmecpp/test_data"
CTH_LIKE = TEST_DATA / "cth_like_fixed_bdy.json"
VACUUM = TEST_DATA / "vacuum_axis_stall.json"
"""The exact vacuum field of examples/exact_equilibria.py's Vacuum with k 0.04, m_g
0.05, u0 2 F(pi/4 | m_g) and dpsi 0.24, at mpol 12 and ns 51."""


def _raw_forces(factor: float, ns: int = 21) -> tuple[np.ndarray, np.ndarray]:
    vmec_input = vmecpp.VmecInput.from_file(CTH_LIKE)
    vmec_input.tcon_axis_factor = factor
    model = _vmecpp.VmecModel.create(vmec_input._to_cpp_vmecindata(), ns)
    model.evaluate(2, 2, False)
    size = ns * model.mpol * (model.ntor + 1)
    forces = np.asarray(model.get_forces()).copy()
    return forces.reshape(forces.size // size, ns, -1), np.arange(ns) / (ns - 1.0)


def test_factor_of_one_is_the_default() -> None:
    forces, _ = _raw_forces(1.0)
    vmec_input = vmecpp.VmecInput.from_file(CTH_LIKE)
    model = _vmecpp.VmecModel.create(vmec_input._to_cpp_vmecindata(), 21)
    model.evaluate(2, 2, False)
    np.testing.assert_array_equal(forces.ravel(), np.asarray(model.get_forces()))


def test_factor_changes_the_force_only_next_to_the_axis() -> None:
    reference, s = _raw_forces(1.0)
    raised, _ = _raw_forces(3.0)
    difference = np.abs(raised - reference).max(axis=(0, 2))
    assert difference[(s > 0.0) & (s < 0.1)].min() > 0.0
    # the force on a surface reads its neighbours. The forces change up to one
    # surface past the taper's end.
    assert difference[s > 0.2 + 2.0 / 20].max() == 0.0


@pytest.mark.parametrize("factor", [0.5, np.inf, np.nan])
def test_factor_is_validated(factor: float) -> None:
    vmec_input = vmecpp.VmecInput.from_file(CTH_LIKE)
    vmec_input.tcon_axis_factor = factor
    with pytest.raises(ValueError, match="tcon_axis_factor"):
        vmecpp.run(vmec_input, verbose=False)


def test_factor_round_trips_through_json() -> None:
    vmec_input = vmecpp.VmecInput.from_file(CTH_LIKE)
    vmec_input.tcon_axis_factor = 2.5
    indata = vmec_input._to_cpp_vmecindata()
    assert indata.tcon_axis_factor == 2.5
    assert _vmecpp.VmecINDATA.from_json(indata.to_json()).tcon_axis_factor == 2.5


@pytest.mark.skipif(
    not _vmecpp.VMECPP_ENABLE_ENZYME,
    reason="the exact force Jacobian needs an Enzyme-enabled build",
)
def test_exact_force_jacobian_carries_the_factor() -> None:
    """The exact product against a directional difference of the force with the factor
    set."""
    vmec_input = vmecpp.VmecInput.from_file(CTH_LIKE)
    vmec_input.tcon_axis_factor = 3.0
    model = _vmecpp.VmecModel.create(vmec_input._to_cpp_vmecindata(), 21)
    model.evaluate(2, 2, True)
    state = np.ascontiguousarray(np.array(model.get_state()))
    direction = np.random.default_rng(0).standard_normal(state.size)
    direction = np.ascontiguousarray(direction / np.linalg.norm(direction))
    exact = np.asarray(model.exact_hessian_vector_product(direction))
    model.set_state(state)
    model.evaluate(2, 2, True)
    difference = np.asarray(model.hessian_vector_product(direction))
    np.testing.assert_allclose(
        exact, difference, rtol=0.0, atol=1.0e-4 * np.linalg.norm(difference)
    )


def test_factor_converges_a_stalling_boundary() -> None:
    """An exact vacuum field stalls at the default settings without the factor."""
    vmec_input = vmecpp.VmecInput.from_file(VACUUM)
    vmec_input.return_outputs_even_if_not_converged = True
    stalled = vmecpp.run(vmec_input, verbose=False, max_threads=2).wout
    assert max(stalled.fsqr, stalled.fsqz, stalled.fsql) > stalled.ftolv
    vmec_input.tcon_axis_factor = 2.0
    converged = vmecpp.run(vmec_input, verbose=False, max_threads=2).wout
    assert max(converged.fsqr, converged.fsqz, converged.fsql) <= converged.ftolv
