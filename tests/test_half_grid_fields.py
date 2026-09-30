# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""The half-grid fields and the enclosed current on the ``SolverState`` of the iteration
callback of ``run``, which a closure for the current profile reads and writes."""

import sys
from pathlib import Path

import numpy as np
import pytest

import vmecpp

REPO_ROOT = Path(__file__).resolve().parents[1]
TEST_DATA = REPO_ROOT / "src" / "vmecpp" / "cpp" / "vmecpp" / "test_data"
CTH_LIKE = REPO_ROOT / "examples" / "data" / "cth_like_fixed_bdy.json"
sys.path.insert(0, str(REPO_ROOT / "examples"))
from self_consistent_bootstrap_current import full_surfaces  # type: ignore # noqa: E402


def _input(path: Path, ns: int = 15) -> vmecpp.VmecInput:
    return vmecpp.VmecInput.from_file(path).model_copy(
        update={
            "ns_array": np.asarray([ns]),
            "ftol_array": np.asarray([1.0e-12]),
            "niter_array": np.asarray([4000]),
        }
    )


def _states(vmec_input, num_states=None):
    """The states the callback receives, stopping the run after ``num_states``."""
    states = []

    def callback(state):
        states.append(state)
        return num_states is None or len(states) < num_states

    output = vmecpp.run(vmec_input, verbose=False, iteration_callback=callback)
    return states, output


def _surface_average(values, fields) -> np.ndarray:
    return values @ np.tile(fields.weight, fields.nzeta)


@pytest.mark.parametrize(
    "path",
    [CTH_LIKE, TEST_DATA / "cth_like_fixed_bdy_asym.json"],
    ids=["symmetric", "asymmetric"],
)
def test_half_grid_fields_give_the_profiles_they_define(path) -> None:
    """Buco and bvco are the surface averages of B_theta and B_zeta, vp is signgs times
    that of sqrt(g), and iota phi' is chi' = <sqrt(g) B^theta>, at every iteration."""
    vmec_input = _input(path)
    states, _ = _states(vmec_input, num_states=50)
    assert len(states) == 50
    for state in states:
        fields = state.half_grid
        assert fields.gsqrt.shape == (14, fields.nzeta * fields.ntheta_eff)
        assert fields.ntheta_eff == (
            fields.ntheta_even if vmec_input.lasym else fields.ntheta_even // 2 + 1
        )
        np.testing.assert_allclose(
            np.sum(fields.weight) * fields.nzeta, 1.0, rtol=1e-14
        )
        for average, profile in [
            (_surface_average(fields.bsubu, fields), fields.buco),
            (_surface_average(fields.bsubv, fields), fields.bvco),
            (fields.signgs * _surface_average(fields.gsqrt, fields), fields.vp),
            (
                _surface_average(fields.gsqrt * fields.bsupu, fields),
                fields.iota * fields.phip,
            ),
        ]:
            np.testing.assert_allclose(average, profile, rtol=1e-12, atol=0.0)


def test_half_grid_fields_match_the_wout_at_convergence() -> None:
    states, output = _states(_input(CTH_LIKE))
    assert output.wout.ier_flag == 0
    fields = states[-1].half_grid
    for name, values in [
        ("buco", fields.buco),
        ("bvco", fields.bvco),
        ("iotas", fields.iota),
        ("phips", fields.phip),
        ("vp", fields.vp),
    ]:
        np.testing.assert_allclose(
            np.asarray(getattr(output.wout, name))[1:], values, rtol=1e-11, err_msg=name
        )


def test_curr_h_is_the_current_the_next_evaluations_prescribe() -> None:
    """With ncurr = 1 every force evaluation makes buco equal to curr_h, and the profile
    a callback leaves in curr_h is prescribed from the next iteration on."""
    changed_at = 30
    currents = []
    fields = []
    profile = []

    def callback(state):
        currents.append(state.curr_h.copy())
        fields.append(state.half_grid)
        if len(currents) == changed_at:
            new = 1.3 * state.curr_h + 1.0e-4 * np.linspace(0.0, 1.0, state.curr_h.size)
            state.curr_h[:] = new
            profile.append(new)
        return len(currents) < 2 * changed_at

    vmecpp.run(_input(CTH_LIKE), verbose=False, iteration_callback=callback)

    assert len(currents) == 2 * changed_at
    for current, half_grid in zip(currents, fields, strict=True):
        assert current.shape == (14,)
        np.testing.assert_allclose(half_grid.buco, current, rtol=1e-13, atol=0.0)
    np.testing.assert_array_equal(currents[changed_at - 1], currents[0])
    for current in currents[changed_at:]:
        np.testing.assert_array_equal(current, profile[0])


def test_curr_h_is_empty_with_a_prescribed_iota() -> None:
    vmec_input = vmecpp.VmecInput.from_file(
        REPO_ROOT / "examples" / "data" / "solovev.json"
    )
    assert vmec_input.ncurr == 0
    states, _ = _states(vmec_input, num_states=5)
    assert all(state.curr_h.size == 0 for state in states)


def test_full_surfaces_reflect_a_stellarator_symmetric_grid() -> None:
    """The example's reflection of the stored half of each surface onto the full
    poloidal grid gives back the field a full-grid evaluation would give: sqrt(g) of a
    stellarator-symmetric run is even under (theta, zeta) -> (-theta, -zeta)."""
    states, _ = _states(_input(CTH_LIKE), num_states=1)
    fields = states[0].half_grid
    full = full_surfaces(fields.gsqrt, fields)
    assert full.shape == (fields.ntheta_even, fields.nzeta, 14)
    stored = fields.gsqrt.reshape(14, fields.nzeta, fields.ntheta_eff)
    np.testing.assert_array_equal(full[: fields.ntheta_eff], stored.transpose(2, 1, 0))
    theta = np.arange(fields.ntheta_even)
    zeta = np.arange(fields.nzeta)
    mirror = full[(-theta) % fields.ntheta_even][:, (-zeta) % fields.nzeta]
    np.testing.assert_allclose(full, mirror, rtol=1e-12)
    # the averages over the full grid are those of the stored half with its weights
    np.testing.assert_allclose(
        np.mean(full, axis=(0, 1)), _surface_average(fields.gsqrt, fields), rtol=1e-13
    )
