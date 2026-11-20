# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Equilibria without a magnetic axis.

With ``VmecInput.inner_boundary_s`` in (0, 1) the plasma fills the shell between the
fixed inner boundary ``rbc_inner``, ``zbs_inner`` and the boundary, and the ``ns``
surfaces divide ``[inner_boundary_s, 1]`` evenly. Cut at one of its surfaces, an
equilibrium is a solution of the shell problem on every surface but the first, where
the shell extrapolates lambda from the two surfaces outside instead of solving for it.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

import vmecpp
from vmecpp import autodiff_wout
from vmecpp.cpp import _vmecpp  # type: ignore

REPO = Path(__file__).resolve().parents[1]
EXAMPLES_DATA = REPO / "examples" / "data"
TEST_DATA = REPO / "src" / "vmecpp" / "cpp" / "vmecpp" / "test_data"


def _input(path: Path, ns: int, ftol: float) -> vmecpp.VmecInput:
    vmec_input = vmecpp.VmecInput.from_file(path)
    vmec_input.ns_array = np.array([ns])
    vmec_input.ftol_array = np.array([ftol])
    vmec_input.niter_array = np.array([100000])
    vmec_input.always_fix_m1_gauge = True
    return vmec_input


def _surface(wout, j: int, mpol: int, ntor: int) -> dict[str, np.ndarray]:
    """Fourier coefficients of surface j in the layout of rbc, zbs, rbs, zbc."""
    coefficients = {
        name: np.zeros((mpol, 2 * ntor + 1)) for name in ("rbc", "zbs", "rbs", "zbc")
    }
    for mn, (m, xn) in enumerate(zip(wout.xm, wout.xn, strict=True)):
        n = int(xn) // int(wout.nfp)
        coefficients["rbc"][int(m), ntor + n] = wout.rmnc[mn, j]
        coefficients["zbs"][int(m), ntor + n] = wout.zmns[mn, j]
        if wout.lasym:
            coefficients["rbs"][int(m), ntor + n] = wout.rmns[mn, j]
            coefficients["zbc"][int(m), ntor + n] = wout.zmnc[mn, j]
    return coefficients


def _shell(full_input: vmecpp.VmecInput, wout, cut: int) -> vmecpp.VmecInput:
    """The input of the shell outside surface cut of the equilibrium wout."""
    ns = int(wout.ns)
    outer = _surface(wout, ns - 1, full_input.mpol, full_input.ntor)
    inner = _surface(wout, cut, full_input.mpol, full_input.ntor)
    shell = full_input.model_copy(deep=True)
    shell.rbc = outer["rbc"]
    shell.zbs = outer["zbs"]
    shell.rbc_inner = inner["rbc"]
    shell.zbs_inner = inner["zbs"]
    if full_input.lasym:
        shell.rbs = outer["rbs"]
        shell.zbc = outer["zbc"]
        shell.rbs_inner = inner["rbs"]
        shell.zbc_inner = inner["zbc"]
    shell.inner_boundary_s = cut / (ns - 1)
    shell.ns_array = np.array([ns - cut])
    return shell


def _relative_difference(shell_wout, full_wout, cut: int) -> float:
    names = ["rmnc", "zmns"] + (["rmns", "zmnc"] if full_wout.lasym else [])
    return max(
        np.abs(
            np.asarray(getattr(shell_wout, name))
            - np.asarray(getattr(full_wout, name))[:, cut:]
        ).max()
        / np.abs(np.asarray(getattr(full_wout, name))).max()
        for name in names
    )


CUT = 5


@pytest.fixture(scope="module")
def full_input() -> vmecpp.VmecInput:
    """cth_like at ns = 25, the equilibrium the shell tests cut at surface CUT."""
    return _input(EXAMPLES_DATA / "cth_like_fixed_bdy.json", 25, ftol=1.0e-12)


@pytest.fixture(scope="module")
def full(full_input: vmecpp.VmecInput) -> vmecpp.VmecOutput:
    return vmecpp.run(full_input, max_threads=1, verbose=False)


@pytest.fixture(scope="module")
def shell_input(
    full_input: vmecpp.VmecInput, full: vmecpp.VmecOutput
) -> vmecpp.VmecInput:
    return _shell(full_input, full.wout, CUT)


@pytest.fixture(scope="module")
def shell(shell_input: vmecpp.VmecInput) -> vmecpp.VmecOutput:
    return vmecpp.run(shell_input, max_threads=1, verbose=False)


@pytest.mark.parametrize(
    "path", [EXAMPLES_DATA / "cth_like_fixed_bdy.json", TEST_DATA / "up_down_asym.json"]
)
def test_restricted_state_has_the_forces_of_the_full_problem(path: Path) -> None:
    """At any state of the full problem restricted to the shell, the shell's forces on
    the surfaces whose half cells exclude surface 0 are the full forces."""
    ns, cut = 25, 5
    full_input = _input(path, ns, ftol=1.0e-12)
    full = vmecpp.run(full_input, max_threads=1, verbose=False)
    shell_input = _shell(full_input, full.wout, cut)

    restart = _vmecpp.HotRestartState(
        wout=full.wout._to_cpp_wout(), indata=full_input._to_cpp_vmecindata()
    )
    full_model = _vmecpp.VmecModel.create(full_input._to_cpp_vmecindata(), ns, restart)
    shell_model = _vmecpp.VmecModel.create(
        shell_input._to_cpp_vmecindata(), ns - cut, None
    )

    lthreed = full_input.ntor > 0
    per_quantity = 1 + lthreed + full_input.lasym + (full_input.lasym and lthreed)
    shape = (3 * per_quantity, -1, full_input.mpol, full_input.ntor + 1)
    lam = slice(2 * per_quantity, 3 * per_quantity)

    # perturb the surfaces strictly inside the shell, away from equilibrium
    state = np.asarray(full_model.get_state()).reshape(shape).copy()
    rng = np.random.default_rng(1)
    state[:, cut + 1 : ns - 1] *= 1.0 + 1.0e-4 * rng.standard_normal(
        state[:, cut + 1 : ns - 1].shape
    )
    restricted = state[:, cut:].copy()
    # lambda is held divided by lamscale, sqrt(1 - s_inner) times smaller for the
    # shell at constant phi'
    lamscale_ratio = np.sqrt(1.0 - cut / (ns - 1))
    restricted[lam] /= lamscale_ratio
    shell_state = np.asarray(shell_model.get_state()).reshape(shape)
    np.testing.assert_allclose(
        restricted[: 2 * per_quantity, 0],
        shell_state[: 2 * per_quantity, 0],
        rtol=0.0,
        atol=1.0e-15,
    )

    full_model.set_state(state.ravel())
    full_model.evaluate(1, 1, False, True)
    shell_model.set_state(restricted.ravel())
    shell_model.evaluate(1, 1, False, True)
    full_forces = np.asarray(full_model.get_forces()).reshape(shape)[:, cut + 2 :]
    shell_forces = np.asarray(shell_model.get_forces()).reshape(shape)[:, 2:]
    shell_forces[lam] /= lamscale_ratio

    scale = np.abs(full_forces).max()
    assert scale > 1.0e-6
    np.testing.assert_allclose(
        shell_forces, full_forces, rtol=0.0, atol=1.0e-10 * scale
    )


@pytest.mark.parametrize(
    ("path", "ns", "tolerance"),
    [
        # three-dimensional, measured 1.3e-5
        (EXAMPLES_DATA / "cth_like_fixed_bdy.json", 25, 2.5e-5),
        # three-dimensional, beta = 4 %, measured 6.3e-5
        (TEST_DATA / "li383_low_res.json", 25, 1.2e-4),
        # three-dimensional, stellarator-asymmetric, measured 1.9e-4
        (TEST_DATA / "cth_like_fixed_bdy_asym.json", 25, 4.0e-4),
    ],
)
def test_shell_reproduces_the_cut_equilibrium(
    path: Path, ns: int, tolerance: float
) -> None:
    cut = round(0.2 * (ns - 1))
    full_input = _input(path, ns, ftol=1.0e-15)
    full = vmecpp.run(full_input, max_threads=1, verbose=False)
    shell = vmecpp.run(_shell(full_input, full.wout, cut), max_threads=2, verbose=False)
    assert _relative_difference(shell.wout, full.wout, cut) < tolerance


def test_difference_from_the_cut_equilibrium_is_second_order() -> None:
    """The shell differs from the cut equilibrium through lambda on its first surface,
    which the hybrid lambda force sees; the difference falls with ns^2."""
    path = TEST_DATA / "up_down_asym.json"
    differences = []
    for ns in (26, 51):
        cut = round(0.2 * (ns - 1))
        full_input = _input(path, ns, ftol=1.0e-16)
        full = vmecpp.run(full_input, max_threads=1, verbose=False)
        shell = vmecpp.run(
            _shell(full_input, full.wout, cut), max_threads=1, verbose=False
        )
        differences.append(_relative_difference(shell.wout, full.wout, cut))
    order = np.log2(differences[0] / differences[1])
    assert 1.6 < order < 2.4


def test_shell_flux_and_volume(full, shell_input, shell) -> None:
    wout = shell.wout
    s_inner = shell_input.inner_boundary_s
    np.testing.assert_allclose(
        wout.phi, np.asarray(full.wout.phi)[CUT:], rtol=1.0e-13, atol=0.0
    )
    np.testing.assert_allclose(wout.phi[0], s_inner * wout.phi[-1], rtol=1.0e-13)
    # the plasma fills the volume between the two boundaries
    delta_s = (1.0 - s_inner) / (int(wout.ns) - 1)
    volume = 4.0 * np.pi**2 * np.sum(np.asarray(wout.vp)[1:]) * delta_s
    np.testing.assert_allclose(wout.volume_p, volume, rtol=1.0e-6)
    assert wout.volume_p < full.wout.volume_p


def test_hot_restart_of_a_shell(full, shell_input, shell) -> None:
    restarted = vmecpp.run(
        shell_input, max_threads=1, verbose=False, restart_from=shell
    )
    assert restarted.wout.itfsq < 0.2 * shell.wout.itfsq
    assert _relative_difference(restarted.wout, shell.wout, 0) < 1.0e-6

    with pytest.raises(ValueError, match="inner_boundary_s"):
        vmecpp.run(
            shell_input.model_copy(update={"ns_array": np.array([int(full.wout.ns)])}),
            max_threads=1,
            verbose=False,
            restart_from=full,
        )


def test_continuation_of_a_shell(full_input, full, shell_input, shell) -> None:
    # at the same resolution only the half-grid lambda is recomputed
    same = vmecpp.interpolate_solution(shell, shell_input)
    np.testing.assert_allclose(
        same.wout.lmns,
        shell.wout.lmns,
        rtol=0.0,
        atol=1.0e-14 * np.abs(shell.wout.lmns).max(),
    )

    schedule = shell_input.model_copy(deep=True)
    schedule.ns_array = np.array([11, int(shell.wout.ns)])
    schedule.ftol_array = np.array([1.0e-8, 1.0e-12])
    schedule.niter_array = np.array([100000, 100000])
    schedule.mpol = [full_input.mpol - 1, full_input.mpol]
    continued = vmecpp.run(schedule, max_threads=1, verbose=False)
    # measured 1.6e-5, set by ftol: the plasma with an axis differs by 2.7e-5
    assert _relative_difference(continued.wout, shell.wout, 0) < 3.0e-5

    with pytest.raises(ValueError, match="inner_boundary_s"):
        vmecpp.interpolate_solution(full, shell_input)


def test_inner_boundary_input_is_validated(full_input, shell_input) -> None:
    cpp_indata = _vmecpp.VmecINDATA.from_json(
        shell_input._to_cpp_vmecindata().to_json()
    )
    assert cpp_indata.inner_boundary_s == shell_input.inner_boundary_s
    np.testing.assert_array_equal(cpp_indata.rbc_inner, shell_input.rbc_inner)
    np.testing.assert_array_equal(cpp_indata.zbs_inner, shell_input.zbs_inner)
    with tempfile.TemporaryDirectory() as tmp_dir:
        for vmec_input in (shell_input, full_input):
            saved = Path(tmp_dir) / "input.json"
            vmec_input.save(saved)
            reloaded = vmecpp.VmecInput.from_file(saved)
            assert reloaded.inner_boundary_s == vmec_input.inner_boundary_s
            if vmec_input.rbc_inner is None:
                assert reloaded.rbc_inner is None
            else:
                np.testing.assert_array_equal(reloaded.rbc_inner, vmec_input.rbc_inner)

    def validate(**update) -> vmecpp.VmecInput:
        return vmecpp.VmecInput.model_validate_json(
            shell_input.model_copy(update=update).model_dump_json()
        )

    with pytest.raises(ValueError, match=r"\[0, 1\)"):
        validate(inner_boundary_s=1.0)
    with pytest.raises(ValueError, match="rbc_inner"):
        validate(rbc_inner=None)
    with pytest.raises(ValueError, match="only used with"):
        validate(inner_boundary_s=0.0)
    with pytest.raises(ValueError, match="lfreeb"):
        validate(lfreeb=True)
    assert validate().inner_boundary_s == shell_input.inner_boundary_s
    with pytest.raises(NotImplementedError, match="inner boundary"):
        autodiff_wout.wout_quantities(None, shell_input)

    # an inner boundary running against the boundary in the poloidal angle
    flipped = shell_input.model_copy(deep=True)
    flipped.zbs_inner = -np.asarray(flipped.zbs_inner)
    with pytest.raises(ValueError, match="opposite poloidal direction"):
        vmecpp.run(flipped, max_threads=1, verbose=False)

    # an inner boundary outside the boundary
    outside = shell_input.model_copy(deep=True)
    for name in ("rbc", "zbs"):
        coefficients = np.array(getattr(shell_input, name), dtype=float)
        coefficients[1:] *= 1.2
        setattr(outside, f"{name}_inner", coefficients)
    with pytest.raises(RuntimeError, match="does not lie inside the boundary"):
        vmecpp.run(outside, max_threads=1, verbose=False)
