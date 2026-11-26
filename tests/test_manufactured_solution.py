# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
# <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Convergence orders of the VMEC++ discretization, by manufactured solution.

`examples/manufactured_solution.py` makes an analytic mapping the exact solution
of a modified problem, by installing the negative of its continuum ideal-MHD
force as a source term.  The difference between the solver and the mapping is
then a discretization error, and its behaviour under refinement is an order.

These are the orders as invariants: a change that breaks the discretization,
rather than merely disagreeing with the Fortran references, shows up here as a
wrong slope.  The resolutions are the smallest that separate first from second
order; the example itself runs the same measurements further out.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("sympy")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

import manufactured_solution as mms

MPOL = 4
NTOR = 2
NTHETA = 18
NZETA = 16
NS = (13, 25, 49)
SMIN = 0.2


@pytest.fixture(scope="module")
def case_and_model():
    case = mms.build_case(mms.FITTED_P)
    return case, mms.Model(case)


def _order(values):
    """Convergence order between the last two entries, for a doubled grid."""
    return np.log(values[-2] / values[-1]) / np.log(2.0)


def _force_error(case, model, ns):
    """Worst relative deviation of the discrete force from the continuum one."""
    fv, _ = mms.discrete_force(case, ns, MPOL, NTOR, NTHETA, NZETA)
    sgrid = np.linspace(0.0, 1.0, ns)
    fa = mms.hat_force_to_decomposed(
        mms.project_force(model, case, sgrid, MPOL, NTOR, nu=32, nw=32), MPOL
    )
    sel = slice(2, ns - 2)
    keep = sgrid[sel] >= SMIN
    scale = max(float(np.max(np.abs(fa[k][sel]))) for k in mms.spans(False))
    # the solver's force is + dW/dx; the projected continuum force is - dW/dx
    return (
        max(
            float(np.max(np.abs((fv[k][sel] + fa[k][sel])[keep])))
            for k in mms.spans(False)
        )
        / scale
    )


def test_discrete_force_converges_to_the_continuum_force(case_and_model):
    """The force operator is second order in the radial grid away from the axis."""
    case, model = case_and_model
    errors = [_force_error(case, model, ns) for ns in NS]
    assert errors == sorted(errors, reverse=True), errors
    assert _order(errors) > 1.7, errors


def test_discrete_energy_converges_to_the_continuum_energy(case_and_model):
    """The MHD energy is second order in the radial grid."""
    case, model = case_and_model
    exact = mms.continuum_energy(model, nrho=120, nu=32, nw=32)
    errors = []
    for ns in NS:
        m = mms._model_at(case, ns, MPOL, NTOR, NTHETA, NZETA)
        mms.install_state(m, case, ns, MPOL, NTOR)
        m.evaluate(1, 1, False, True)
        errors.append(abs(m.mhd_energy / exact - 1.0))
    assert errors == sorted(errors, reverse=True), errors
    assert _order(errors) > 1.7, errors


def test_angular_truncation_is_geometric():
    """Refining the angular grid removes error geometrically, not algebraically.

    Measured on the unfitted mapping: the fitted one is close to force balance,
    so its force is near zero and a relative measure of it says nothing.
    """
    case = mms.Case()
    ns = 25
    ref, _ = mms.discrete_force(case, ns, MPOL, NTOR, 48, 48)
    scale = max(float(np.max(np.abs(ref[k]))) for k in mms.spans(False))
    errors = []
    for grid in (16, 18, 20):
        fv, _ = mms.discrete_force(case, ns, MPOL, NTOR, grid, grid)
        errors.append(
            max(float(np.max(np.abs(fv[k] - ref[k]))) for k in mms.spans(False)) / scale
        )
    assert errors == sorted(errors, reverse=True), errors
    # two added grid points buy far more than the factor of four a second-order
    # method would give
    assert errors[-2] / errors[-1] > 4.0, errors


def test_source_cancels_the_force_at_the_mapping(case_and_model):
    """Installing the source leaves only the truncation error behind."""
    case, model = case_and_model
    ns = 25
    sgrid = np.linspace(0.0, 1.0, ns)
    fa = mms.project_force(model, case, sgrid, MPOL, NTOR, nu=32, nw=32)
    source = mms.flatten(mms.masked_source(fa, ns))

    _, without = mms.discrete_force(case, ns, MPOL, NTOR, NTHETA, NZETA)
    _, with_source = mms.discrete_force(
        case, ns, MPOL, NTOR, NTHETA, NZETA, source=source
    )
    bare = without.fsqr + without.fsqz + without.fsql
    left = with_source.fsqr + with_source.fsqz + with_source.fsql
    assert left < bare / 1.0e3, (bare, left)


def test_force_source_is_inert_when_unset(case_and_model):
    """A run that never sets a source is unchanged by the hook."""
    case, _ = case_and_model
    ns = 25
    a, _ = mms.discrete_force(case, ns, MPOL, NTOR, NTHETA, NZETA)
    b, _ = mms.discrete_force(case, ns, MPOL, NTOR, NTHETA, NZETA, source=None)
    for k in mms.spans(False):
        np.testing.assert_array_equal(a[k], b[k])


def test_empty_source_is_accepted_and_clears(case_and_model):
    case, model = case_and_model
    ns = 25
    m = mms._model_at(case, ns, MPOL, NTOR, NTHETA, NZETA)
    mms.install_state(m, case, ns, MPOL, NTOR)
    m.evaluate(1, 1, False, True)
    bare = m.fsqr + m.fsqz + m.fsql

    sgrid = np.linspace(0.0, 1.0, ns)
    fa = mms.project_force(model, case, sgrid, MPOL, NTOR, nu=32, nw=32)
    m.set_force_source(mms.flatten(mms.masked_source(fa, ns)))
    mms.install_state(m, case, ns, MPOL, NTOR)
    m.evaluate(1, 1, False, True)
    assert m.fsqr + m.fsqz + m.fsql < bare

    m.set_force_source(np.zeros(0))
    mms.install_state(m, case, ns, MPOL, NTOR)
    m.evaluate(1, 1, False, True)
    assert m.fsqr + m.fsqz + m.fsql == pytest.approx(bare, rel=1e-12)


def test_wrong_source_length_is_rejected(case_and_model):
    case, _ = case_and_model
    ns = 25
    m = mms._model_at(case, ns, MPOL, NTOR, NTHETA, NZETA)
    with pytest.raises(RuntimeError, match="force source has"):
        m.set_force_source(np.ones(7))


def test_source_is_refused_unless_the_input_asks_for_it(case_and_model):
    """A run that did not ask for a source solves ideal MHD, not a modified problem, so
    installing one has to fail rather than change the answer."""
    case, model = case_and_model
    ns = 25
    m = mms._model_at(case, ns, MPOL, NTOR, NTHETA, NZETA, enable_force_source=False)
    sgrid = np.linspace(0.0, 1.0, ns)
    fa = mms.project_force(model, case, sgrid, MPOL, NTOR, nu=32, nw=32)
    with pytest.raises(RuntimeError, match="enable_force_source"):
        m.set_force_source(mms.flatten(mms.masked_source(fa, ns)))


def test_source_survives_a_reinitialization(case_and_model):
    """InitializeRadial rebuilds the per-thread models, which an axis reguess does mid-
    solve; the source has to come back with them."""
    case, model = case_and_model
    ns = 25
    m = mms._model_at(case, ns, MPOL, NTOR, NTHETA, NZETA)
    sgrid = np.linspace(0.0, 1.0, ns)
    fa = mms.project_force(model, case, sgrid, MPOL, NTOR, nu=32, nw=32)
    m.set_force_source(mms.flatten(mms.masked_source(fa, ns)))

    mms.install_state(m, case, ns, MPOL, NTOR)
    m.evaluate(1, 1, False, True)
    before = m.fsqr + m.fsqz + m.fsql

    m.reinitialize()
    mms.install_state(m, case, ns, MPOL, NTOR)
    m.evaluate(1, 1, False, True)
    assert m.fsqr + m.fsqz + m.fsql == pytest.approx(before, rel=1e-9)


def test_constrained_current_force_is_second_order(case_and_model):
    """The same order holds when the solver derives iota from a prescribed toroidal
    current instead of reading an iota profile (ncurr = 1)."""
    case, model = case_and_model
    current = mms.current_constraint(model, case, nknots=33, nu=32, nw=32)
    errors = []
    for ns in NS:
        fv, _ = mms.discrete_force(case, ns, MPOL, NTOR, NTHETA, NZETA, current=current)
        sgrid = np.linspace(0.0, 1.0, ns)
        fa = mms.hat_force_to_decomposed(
            mms.project_force(model, case, sgrid, MPOL, NTOR, nu=32, nw=32), MPOL
        )
        errors.append(max(mms._force_by_parity(fv, fa, ns, MPOL, SMIN, False)))
    assert errors == sorted(errors, reverse=True), errors
    assert _order(errors) > 1.7, errors


def test_free_boundary_force_is_second_order():
    """The free-boundary edge term and the one-sided force at the last radial node
    converge at the same order as the volume.

    only_coils takes the vacuum field straight from the mgrid, and a field bilinear in
    (R, Z) is interpolated exactly, so the vacuum pressure the solver uses is analytic
    and the only error left is the discretization.
    """
    import tempfile  # noqa: PLC0415
    from pathlib import Path as _Path  # noqa: PLC0415

    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as fh:
        mgrid = fh.name
    try:
        case, model, vac, _ = mms.freeb_setup(NZETA, mgrid)
        volume, edge = [], []
        for ns in (13, 25):
            fa = mms.freeb_source(model, case, vac, MPOL, NTOR, ns, nu=32, nw=32)
            m = mms._model_at(case, ns, MPOL, NTOR, NTHETA, NZETA, mgrid=mgrid)
            for iteration in range(1, 6):
                mms.install_state(m, case, ns, MPOL, NTOR)
                m.evaluate(1, iteration, False, True)
            fv = mms.unflatten(m.get_forces(), ns, MPOL, NTOR)
            dec = mms.hat_force_to_decomposed(fa, MPOL)
            volume.append(max(mms._force_by_parity(fv, dec, ns, MPOL, SMIN, False)))
            num = sum(
                float(np.sum((fv[k][ns - 1] + dec[k][ns - 1]) ** 2))
                for k in mms.spans(False)
            )
            den = sum(float(np.sum(dec[k][ns - 1] ** 2)) for k in mms.spans(False))
            edge.append(float(np.sqrt(num / den)))
        assert _order(volume) > 1.7, volume
        assert _order(edge) > 1.7, edge
    finally:
        _Path(mgrid).unlink(missing_ok=True)


def test_asymmetric_path_reproduces_the_symmetric_result(case_and_model):
    """A stellarator-symmetric mapping run through the lasym code path must give the
    same answer, which is the invariant AGENTS.md asks for, here against a continuum
    reference rather than against another run."""
    case, model = case_and_model
    ns = 25
    exact = mms.continuum_energy(model, nrho=120, nu=32, nw=32)
    got = []
    for lasym in (False, True):
        m = mms._model_at(case, ns, MPOL, NTOR, NTHETA, NZETA, lasym=lasym)
        mms.install_state(m, case, ns, MPOL, NTOR, lasym)
        m.evaluate(1, 1, False, True)
        got.append(abs(m.mhd_energy / exact - 1.0))
    assert got[1] == pytest.approx(got[0], rel=1e-10), got


def test_asymmetric_force_is_second_order():
    """The same order holds with genuine non-stellarator-symmetric content."""
    case = mms.build_case(mms.FITTED_P, asym=mms.ASYM)
    model = mms.Model(case)
    errors = []
    for ns in NS:
        fv, _ = mms.discrete_force(case, ns, MPOL, NTOR, NTHETA, NZETA, lasym=True)
        sgrid = np.linspace(0.0, 1.0, ns)
        fa = mms.hat_force_to_decomposed(
            mms.project_force(model, case, sgrid, MPOL, NTOR, nu=32, nw=32, lasym=True),
            MPOL,
        )
        errors.append(max(mms._force_by_parity(fv, fa, ns, MPOL, SMIN, True)))
    assert errors == sorted(errors, reverse=True), errors
    assert _order(errors) > 1.7, errors
