# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
# <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""The unpreconditioned force is the gradient of the MHD energy.

A central finite difference of `mhd_energy` along a displacement of the R and Z
coefficients equals the raw force of `evaluate(precondition=False)` contracted with
that displacement, times the radial integration step 1/(ns - 1) the energy carries
and the force does not. The equilibria prescribe iota, so the energy is taken at
fixed flux functions, and the spectral-condensation weight tcon0 is zero, since the
constraint force it adds is not a gradient of the energy. The displacement leaves
the axis, the first interior surface and the boundary, which the force treats on
their own, and the m = 1 coefficients, which the poloidal-origin gauge couples
between R and Z. This is the relation between the energy whose second variation
decides ideal stability and the force the solver balances, for axisymmetric
equilibria.
"""

from pathlib import Path

import numpy as np
import pytest

from vmecpp.cpp import _vmecpp  # type: ignore

DATA = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "vmecpp"
    / "cpp"
    / "vmecpp"
    / "test_data"
)
NS = 11
NO_RESTART = 1  # RestartReason::NO_RESTART; BAD_JACOBIAN is 2
CASES = ["solovev.json", "circular_tokamak.json"]


def _model(name):
    indata = _vmecpp.VmecINDATA.from_file(str(DATA / name))
    assert indata.ncurr == 0
    assert not indata.lasym
    assert indata.ntor == 0
    indata.tcon0 = 0.0
    return _vmecpp.VmecModel.create(indata, NS)


def _admissible(model, rng):
    """A random displacement of the R and Z coefficients of the surfaces from the second
    interior one to the one below the boundary, m = 1 left out."""
    x = np.asarray(model.get_state(), float)
    per = len(x) // 3 // NS
    nt1 = model.ntor + 1
    xi = np.zeros_like(x)
    for block in (0, 1):  # R and Z; lambda is the last block
        for j in range(2, NS - 1):
            for k in range(per):
                if (k % (model.mpol * nt1)) // nt1 == 1:
                    continue
                i = (block * NS + j) * per + k
                xi[i] = rng.standard_normal() * (abs(x[i]) + 1e-3)
    return xi


def _energy(model, state):
    model.set_state(np.ascontiguousarray(state))
    model.evaluate(2, 2, False)
    # a flipped Jacobian ends the evaluation before the energy is formed
    assert model.restart_reason == NO_RESTART
    return model.mhd_energy


@pytest.mark.parametrize("name", CASES)
def test_energy_difference_is_force(name):
    model = _model(name)
    model.solve()
    x_eq = np.asarray(model.get_state(), float).copy()
    rng = np.random.default_rng(7)
    for _ in range(3):
        # a state displaced from the equilibrium, where the force is not zero
        model.set_state(np.ascontiguousarray(x_eq + 1e-3 * _admissible(model, rng)))
        model.evaluate(2, 2, False)
        assert model.restart_reason == NO_RESTART
        x = np.asarray(model.get_state(), float).copy()
        force = np.asarray(model.get_forces(), float).copy()
        xi = _admissible(model, rng)
        eps = 1e-5
        de = (_energy(model, x + eps * xi) - _energy(model, x - eps * xi)) / (2 * eps)
        contracted = float(force @ xi) / (NS - 1)
        np.testing.assert_allclose(de, contracted, rtol=1e-4)


@pytest.mark.parametrize("name", CASES)
def test_energy_difference_at_equilibrium_vanishes(name):
    """At a converged equilibrium the first variation of the energy along an admissible
    displacement is a thousandth of what it is at a state displaced from it along that
    direction by a thousandth of the coefficients."""
    model = _model(name)
    model.solve()
    x = np.asarray(model.get_state(), float).copy()
    xi = _admissible(model, np.random.default_rng(11))
    eps = 1e-5

    def first_variation(state):
        return (_energy(model, state + eps * xi) - _energy(model, state - eps * xi)) / (
            2 * eps
        )

    assert abs(first_variation(x)) < 1e-3 * abs(first_variation(x + 1e-3 * xi))
