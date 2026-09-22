# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""The half-grid fields and the enclosed current of ``VmecModel``, the solver state a
closure for the current profile reads and writes during a Python-driven solve."""

import sys
from pathlib import Path

import numpy as np
import pytest

import vmecpp
from vmecpp.cpp import _vmecpp  # type: ignore

REPO_ROOT = Path(__file__).resolve().parents[1]
TEST_DATA = REPO_ROOT / "src" / "vmecpp" / "cpp" / "vmecpp" / "test_data"
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


def _surface_average(values, fields) -> np.ndarray:
    return np.asarray(values) @ np.tile(np.asarray(fields.weight), fields.nzeta)


@pytest.mark.parametrize(
    "path",
    [
        REPO_ROOT / "examples" / "data" / "cth_like_fixed_bdy.json",
        TEST_DATA / "cth_like_fixed_bdy_asym.json",
    ],
)
def test_half_grid_fields_give_the_profiles_they_define(path) -> None:
    """Buco and bvco are the surface averages of B_theta and B_zeta, vp is signgs times
    that of sqrt(g), and iota is chi' / phi', at any state of the iteration."""
    vmec_input = _input(path)
    model = _vmecpp.VmecModel.create(vmec_input._to_cpp_vmecindata(), 15)
    model.evaluate(1, 1, True)
    fields = model.half_grid_fields()

    assert fields.gsqrt.shape == (14, fields.nzeta * fields.ntheta_eff)
    assert fields.ntheta_eff == (
        fields.ntheta_even if vmec_input.lasym else fields.ntheta_even // 2 + 1
    )
    np.testing.assert_allclose(np.sum(fields.weight) * fields.nzeta, 1.0, rtol=1e-14)
    for average, profile in [
        (_surface_average(fields.bsubu, fields), fields.buco),
        (_surface_average(fields.bsubv, fields), fields.bvco),
        (fields.signgs * _surface_average(fields.gsqrt, fields), fields.vp),
    ]:
        np.testing.assert_allclose(average, profile, rtol=1e-13, atol=0.0)
    np.testing.assert_allclose(
        np.asarray(model.chip_h) / np.asarray(fields.phip), fields.iota, rtol=1e-13
    )


def test_half_grid_fields_match_the_wout_at_convergence() -> None:
    vmec_input = _input(REPO_ROOT / "examples" / "data" / "cth_like_fixed_bdy.json")
    model = _vmecpp.VmecModel.create(vmec_input._to_cpp_vmecindata(), 15)
    assert vmecpp.solve_equilibrium(model).converged
    fields = model.half_grid_fields()
    wout = vmecpp.run(vmec_input, verbose=False).wout
    for name, values in [
        ("buco", fields.buco),
        ("bvco", fields.bvco),
        ("iotas", fields.iota),
        ("phips", fields.phip),
        ("vp", fields.vp),
    ]:
        np.testing.assert_allclose(
            np.asarray(getattr(wout, name))[1:], values, rtol=1e-11, err_msg=name
        )


def test_curr_h_is_the_current_the_next_evaluation_imposes() -> None:
    """With ncurr = 1, every evaluation makes buco equal to curr_h, including a profile
    set between evaluations."""
    vmec_input = _input(REPO_ROOT / "examples" / "data" / "cth_like_fixed_bdy.json")
    model = _vmecpp.VmecModel.create(vmec_input._to_cpp_vmecindata(), 15)
    model.evaluate(1, 1, True)
    np.testing.assert_allclose(
        model.half_grid_fields().buco, model.curr_h, rtol=1e-13, atol=0.0
    )

    profile = 1.3 * np.asarray(model.curr_h) + 1.0e-4 * np.linspace(0.0, 1.0, 14)
    model.curr_h = profile
    np.testing.assert_array_equal(model.curr_h, profile)
    model.evaluate(2, 2, True)
    np.testing.assert_allclose(model.half_grid_fields().buco, profile, rtol=1e-13)


def test_curr_h_is_checked() -> None:
    constrained_iota = _vmecpp.VmecModel.create(
        vmecpp.VmecInput.from_file(
            REPO_ROOT / "examples" / "data" / "solovev.json"
        )._to_cpp_vmecindata(),
        5,
    )
    with pytest.raises(RuntimeError, match="ncurr = 1"):
        constrained_iota.curr_h = np.zeros(4)

    vmec_input = _input(REPO_ROOT / "examples" / "data" / "cth_like_fixed_bdy.json")
    model = _vmecpp.VmecModel.create(vmec_input._to_cpp_vmecindata(), 15)
    with pytest.raises(RuntimeError, match="one entry per half-grid surface"):
        model.curr_h = np.zeros(15)


def test_full_surfaces_reflect_a_stellarator_symmetric_grid() -> None:
    """The example's reflection of the stored half of each surface onto the full
    poloidal grid gives back the field a full-grid evaluation would give: sqrt(g) of a
    stellarator-symmetric run is even under (theta, zeta) -> (-theta, -zeta)."""
    vmec_input = _input(REPO_ROOT / "examples" / "data" / "cth_like_fixed_bdy.json")
    model = _vmecpp.VmecModel.create(vmec_input._to_cpp_vmecindata(), 15)
    model.evaluate(1, 1, True)
    fields = model.half_grid_fields()
    full = full_surfaces(fields.gsqrt, fields)
    assert full.shape == (fields.ntheta_even, fields.nzeta, 14)
    stored = np.asarray(fields.gsqrt).reshape(14, fields.nzeta, fields.ntheta_eff)
    np.testing.assert_array_equal(full[: fields.ntheta_eff], stored.transpose(2, 1, 0))
    theta = np.arange(fields.ntheta_even)
    zeta = np.arange(fields.nzeta)
    mirror = full[(-theta) % fields.ntheta_even][:, (-zeta) % fields.nzeta]
    np.testing.assert_allclose(full, mirror, rtol=1e-12)
    # the averages over the full grid are those of the stored half with its weights
    np.testing.assert_allclose(
        np.mean(full, axis=(0, 1)), _surface_average(fields.gsqrt, fields), rtol=1e-13
    )
