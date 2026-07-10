# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Tests for the Python-side resolution interpolation and continuation driver."""

from pathlib import Path

import numpy as np
import pytest

import vmecpp
from vmecpp._continuation import _state_mode_table, _step_input

REPO_ROOT = Path(__file__).parent.parent
TEST_DATA_DIR = REPO_ROOT / "src" / "vmecpp" / "cpp" / "vmecpp" / "test_data"


@pytest.fixture(scope="module")
def solovev_output() -> vmecpp.VmecOutput:
    vmec_input = vmecpp.VmecInput.from_file(TEST_DATA_DIR / "solovev.json")
    return vmecpp.run(vmec_input, verbose=False, max_threads=1)


@pytest.fixture(scope="module")
def cma_input() -> vmecpp.VmecInput:
    return vmecpp.VmecInput.from_file(TEST_DATA_DIR / "cma.json")


@pytest.fixture(scope="module")
def cma_direct(cma_input: vmecpp.VmecInput) -> vmecpp.VmecOutput:
    """The reference equilibrium, solved with the built-in (C++) multi-grid."""
    return vmecpp.run(cma_input, verbose=False, max_threads=1)


def _final_force_residual(output: vmecpp.VmecOutput) -> float:
    return float(np.asarray(output.wout.fsqt)[-1])


# --- interpolation -----------------------------------------------------------


def test_state_mode_table_matches_wout(solovev_output: vmecpp.VmecOutput):
    """The generated state mode table must match the ordering VMEC++ writes out."""
    wout = solovev_output.wout
    xm, xn = _state_mode_table(wout.mpol, wout.ntor, wout.nfp)
    np.testing.assert_array_equal(xm, np.asarray(wout.xm))
    np.testing.assert_array_equal(xn, np.asarray(wout.xn))


def test_interpolate_identity_is_a_noop(solovev_output: vmecpp.VmecOutput):
    """Interpolating to the same resolution reproduces the source geometry."""
    same = vmecpp.interpolate_solution(solovev_output, solovev_output.input)
    for field in ("rmnc", "zmns", "lmns_full"):
        np.testing.assert_allclose(
            np.asarray(getattr(same.wout, field)),
            np.asarray(getattr(solovev_output.wout, field)),
            atol=1e-10,
        )
    np.testing.assert_array_equal(
        np.asarray(same.wout.xm), np.asarray(solovev_output.wout.xm)
    )


def test_interpolate_radial_upsample(solovev_output: vmecpp.VmecOutput):
    """Radial up-sampling keeps the endpoints and produces the requested shape."""
    wout = solovev_output.wout
    ns_new = int(wout.ns) * 2 - 1
    target = _step_input(
        solovev_output.input, ns_new, int(wout.mpol), int(wout.ntor), 1e-12, 1
    )
    guess = vmecpp.interpolate_solution(solovev_output, target)

    assert int(guess.wout.ns) == ns_new
    assert np.asarray(guess.wout.rmnc).shape == (int(guess.wout.mnmax), ns_new)
    # The boundary surface (s=1) is a grid point in both grids and is preserved
    # exactly for every mode.
    np.testing.assert_allclose(
        np.asarray(guess.wout.rmnc)[:, -1], np.asarray(wout.rmnc)[:, -1], atol=1e-10
    )
    # The axis (s=0) is preserved for even-m modes; odd-m modes vanish at the axis.
    even_m = np.asarray(guess.wout.xm) % 2 == 0
    np.testing.assert_allclose(
        np.asarray(guess.wout.rmnc)[even_m, 0],
        np.asarray(wout.rmnc)[even_m, 0],
        atol=1e-10,
    )


def test_interpolate_fourier_pad_and_truncate(solovev_output: vmecpp.VmecOutput):
    """Padding adds zero modes; truncating drops modes; shared modes are carried."""
    wout = solovev_output.wout
    mpol0, ntor0, ns0 = int(wout.mpol), int(wout.ntor), int(wout.ns)

    # Pad to a higher poloidal resolution.
    padded_input = _step_input(solovev_output.input, ns0, mpol0 + 2, ntor0, 1e-12, 1)
    padded = vmecpp.interpolate_solution(solovev_output, padded_input)
    assert int(padded.wout.mpol) == mpol0 + 2

    src_index = {
        (int(m), int(n)): i
        for i, (m, n) in enumerate(
            zip(np.asarray(wout.xm), np.asarray(wout.xn), strict=False)
        )
    }
    pad_xm, pad_xn = np.asarray(padded.wout.xm), np.asarray(padded.wout.xn)
    pad_rmnc = np.asarray(padded.wout.rmnc)
    src_rmnc = np.asarray(wout.rmnc)
    saw_new_zero_mode = False
    for i, (m, n) in enumerate(zip(pad_xm, pad_xn, strict=False)):
        j = src_index.get((int(m), int(n)))
        if j is not None:
            np.testing.assert_allclose(pad_rmnc[i], src_rmnc[j], atol=1e-10)
        else:
            np.testing.assert_allclose(pad_rmnc[i], 0.0, atol=1e-12)
            saw_new_zero_mode = True
    assert saw_new_zero_mode

    # Truncating to a lower poloidal resolution keeps only the surviving modes.
    if mpol0 > 1:
        trunc_input = _step_input(solovev_output.input, ns0, mpol0 - 1, ntor0, 1e-12, 1)
        trunc = vmecpp.interpolate_solution(solovev_output, trunc_input)
        assert np.asarray(trunc.wout.xm).max() <= mpol0 - 2


# --- continuation driver -----------------------------------------------------


def test_mpol_ntor_length_one_sequence_collapses_to_scalar(
    cma_input: vmecpp.VmecInput,
):
    """A length-1 mpol/ntor sequence is equivalent to a scalar and is stored as a plain
    int, so it takes the direct (non-continuation) code path.

    This coercion runs during validation (construction / model_validate), like all of
    VmecInput's other validators; it goes through vmecpp.VmecInput.model_validate here
    rather than a bare attribute assignment for that reason.
    """
    data = cma_input.model_dump(mode="json")
    data["mpol"] = [int(cma_input.mpol)]
    data["ntor"] = [int(cma_input.ntor)]
    single_step = vmecpp.VmecInput.model_validate(data)
    assert isinstance(single_step.mpol, int)
    assert isinstance(single_step.ntor, int)
    assert single_step.mpol == cma_input.mpol
    assert single_step.ntor == cma_input.ntor


def test_mpol_length_mismatch_raises(cma_input: vmecpp.VmecInput):
    """A mpol/ntor schedule that doesn't match ns_array's length is rejected with a
    clear error, rather than silently misaligning steps."""
    mismatched = cma_input.model_copy(deep=True)
    mismatched.ns_array = np.asarray([15, 31], dtype=np.int64)
    mismatched.ftol_array = np.asarray([1e-8, 1e-10], dtype=float)
    mismatched.niter_array = np.asarray([100, 100], dtype=np.int64)
    mismatched.mpol = np.asarray([3, 4, 5], dtype=np.int64)  # 3 entries, ns_array has 2

    with pytest.raises(ValueError, match="mpol"):
        vmecpp.run(mismatched, verbose=False, max_threads=1)


def test_ns_only_continuation_reproduces_direct_multigrid(
    cma_input: vmecpp.VmecInput, cma_direct: vmecpp.VmecOutput
):
    """A continuation schedule with a constant (array-valued) mpol/ntor reaches the same
    equilibrium as the C++ multi-grid: identical resolution, a converged force balance,
    a bit-identical plasma volume, and geometry agreeing at the convergence level."""
    n_steps = len(np.asarray(cma_input.ns_array))
    continuation_input = cma_input.model_copy(deep=True)
    continuation_input.mpol = np.asarray(
        [int(cma_input.mpol)] * n_steps, dtype=np.int64
    )
    continuation_input.ntor = np.asarray(
        [int(cma_input.ntor)] * n_steps, dtype=np.int64
    )

    continued = vmecpp.run(continuation_input, verbose=False, max_threads=1)

    assert int(continued.wout.ns) == int(cma_direct.wout.ns)
    # Converged to a force residual comparable to the direct solve.
    assert _final_force_residual(continued) < 10 * _final_force_residual(cma_direct)
    # The plasma volume is a robust invariant and matches to full precision.
    assert continued.wout.volume == pytest.approx(cma_direct.wout.volume, rel=1e-9)
    # The Fourier geometry agrees at the level the force balance is converged to.
    # The residual reflects the difference between the Python continuation (sqrt(s)
    # radial interpolation plus hot-restart) and the C++ multi-grid; both are valid
    # force-balanced states and the agreement tightens with ftol (see
    # test_continuation_agreement_tightens_with_ftol).
    np.testing.assert_allclose(
        np.asarray(continued.wout.rmnc),
        np.asarray(cma_direct.wout.rmnc),
        atol=4e-3,
        rtol=1e-2,
    )


def test_continuation_agreement_tightens_with_ftol(cma_input: vmecpp.VmecInput):
    """As the force-balance tolerance tightens, the continuation and the direct multi-
    grid converge toward the same geometry -- the signature of a shared equilibrium
    rather than a defect in the continuation."""
    n_steps = len(np.asarray(cma_input.ns_array))

    def max_geometry_diff(ftol_array: list[float]) -> float:
        reference = cma_input.model_copy(deep=True)
        reference.ftol_array = np.asarray(ftol_array, dtype=float)
        direct = vmecpp.run(reference, verbose=False, max_threads=1)

        continuation_input = cma_input.model_copy(deep=True)
        continuation_input.mpol = np.asarray(
            [int(cma_input.mpol)] * n_steps, dtype=np.int64
        )
        continuation_input.ntor = np.asarray(
            [int(cma_input.ntor)] * n_steps, dtype=np.int64
        )
        continuation_input.ftol_array = np.asarray(ftol_array, dtype=float)
        continued = vmecpp.run(continuation_input, verbose=False, max_threads=1)

        return float(
            np.max(
                np.abs(np.asarray(direct.wout.rmnc) - np.asarray(continued.wout.rmnc))
            )
        )

    loose = max_geometry_diff([1e-6, 1e-6])
    tight = max_geometry_diff([1e-8, 1e-11])
    assert tight < loose


def test_fourier_continuation_converges(
    cma_input: vmecpp.VmecInput, cma_direct: vmecpp.VmecOutput
):
    """Increasing the poloidal resolution along the schedule reaches the same
    equilibrium as a direct solve at full resolution."""
    ns_final = int(np.asarray(cma_input.ns_array)[-1])
    mpol_final = int(cma_input.mpol)
    ntor = int(cma_input.ntor)
    ftol_final = float(np.asarray(cma_input.ftol_array)[-1])
    niter_final = int(np.asarray(cma_input.niter_array)[-1])

    continuation_input = cma_input.model_copy(deep=True)
    continuation_input.ns_array = np.asarray([ns_final, ns_final], dtype=np.int64)
    continuation_input.mpol = np.asarray(
        [max(2, mpol_final - 2), mpol_final], dtype=np.int64
    )
    continuation_input.ntor = ntor  # scalar broadcasts to every step
    continuation_input.ftol_array = np.asarray([ftol_final, ftol_final], dtype=float)
    continuation_input.niter_array = np.asarray(
        [niter_final, niter_final], dtype=np.int64
    )

    continued = vmecpp.run(continuation_input, verbose=False, max_threads=1)

    assert int(continued.wout.mpol) == mpol_final
    assert _final_force_residual(continued) < 1e-5
    assert continued.wout.volume == pytest.approx(cma_direct.wout.volume, rel=1e-4)
    # Geometry agrees at the convergence level (see
    # test_continuation_agreement_tightens_with_ftol).
    np.testing.assert_allclose(
        np.asarray(continued.wout.rmnc),
        np.asarray(cma_direct.wout.rmnc),
        atol=4e-3,
        rtol=1e-2,
    )


def test_fourier_continuation_hot_restarts_first_step_from_restart_from(
    cma_input: vmecpp.VmecInput,
):
    """restart_from seeds the first continuation step (interpolated to its resolution)
    instead of a cold start, when input.mpol/.ntor is a sequence."""
    ns_final = int(np.asarray(cma_input.ns_array)[-1])
    mpol_final = int(cma_input.mpol)
    ntor = int(cma_input.ntor)

    warm_start_input = cma_input.model_copy(deep=True)
    warm_start_input.ns_array = np.asarray([ns_final], dtype=np.int64)
    warm_start = vmecpp.run(warm_start_input, verbose=False, max_threads=1)

    continuation_input = cma_input.model_copy(deep=True)
    continuation_input.ns_array = np.asarray([ns_final, ns_final], dtype=np.int64)
    continuation_input.mpol = np.asarray(
        [max(2, mpol_final - 2), mpol_final], dtype=np.int64
    )
    continuation_input.ntor = ntor
    continuation_input.ftol_array = np.asarray([1e-8, 1e-10], dtype=float)
    continuation_input.niter_array = np.asarray([2000, 2000], dtype=np.int64)

    continued = vmecpp.run(
        continuation_input,
        verbose=False,
        max_threads=1,
        restart_from=warm_start,
    )

    assert int(continued.wout.mpol) == mpol_final
    assert _final_force_residual(continued) < 1e-5
