# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
# <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Two properties of the ideal-MHD stability quantities VMEC++ provides.

The second variation of the MHD energy is a symmetric form. Where the
unpreconditioned force is the gradient of the energy (axisymmetric equilibria at
prescribed iota with the spectral-condensation weight zero), its derivative along a
displacement is the energy's Hessian applied to it, so <u, H v> = <v, H u> for any
two admissible displacements; a central difference of the force gives H v.

The geodesic-curvature term of the Mercier criterion,

    DGeod = <G B^2 / |grad s|^3>^2 - <G^2 B^2 / |grad s|^3> <B^2 / |grad s|^3>

in the notation of VMEC's mercier.f90, with G the parallel current density over
B^2, is a Cauchy-Schwarz defect: the square of an average of a product never
exceeds the product of the averages of the squares, so DGeod <= 0 on every surface
whatever the equilibrium. It is the one term of the criterion that can only
stabilize, and a positive value would be an error in how it is assembled.
"""

from pathlib import Path

import numpy as np
import pytest

import vmecpp
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
NO_RESTART = 1
HESSIAN_CASES = ["solovev.json", "circular_tokamak.json"]
MERCIER_CASES = [
    "solovev.json",
    "circular_tokamak.json",
    "cth_like_fixed_bdy.json",
    "cth_like_fixed_bdy_asym.json",
    "cma.json",
    "li383_low_res.json",
    "up_down_asym.json",
]


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
    for block in (0, 1):
        for j in range(2, NS - 1):
            for k in range(per):
                if (k % (model.mpol * nt1)) // nt1 == 1:
                    continue
                i = (block * NS + j) * per + k
                xi[i] = rng.standard_normal() * (abs(x[i]) + 1e-3)
    return xi


def _force(model, state):
    model.set_state(np.ascontiguousarray(state))
    model.evaluate(2, 2, False)
    assert model.restart_reason == NO_RESTART
    return np.asarray(model.get_forces(), float).copy()


@pytest.mark.parametrize("name", HESSIAN_CASES)
def test_energy_hessian_is_symmetric(name):
    model = _model(name)
    model.solve()
    x = np.asarray(model.get_state(), float).copy()
    rng = np.random.default_rng(3)
    eps = 1e-6
    for _ in range(3):
        u = _admissible(model, rng)
        v = _admissible(model, rng)
        Hu = (_force(model, x + eps * u) - _force(model, x - eps * u)) / (2 * eps)
        Hv = (_force(model, x + eps * v) - _force(model, x - eps * v)) / (2 * eps)
        uHv, vHu = float(u @ Hv), float(v @ Hu)
        scale = max(abs(float(u @ Hu)), abs(float(v @ Hv)))
        assert abs(uHv - vHu) <= 1e-6 * scale, (uHv, vHu, scale)


@pytest.mark.parametrize("name", MERCIER_CASES)
def test_geodesic_curvature_term_is_never_positive(name):
    vi = vmecpp.VmecInput.from_file(DATA / name)
    wout = vmecpp.run(vi, verbose=False).wout
    dgeod = np.asarray(wout.DGeod, dtype=float)
    # the axis and the boundary carry no Mercier terms
    interior = dgeod[1:-1]
    scale = float(np.abs(interior).max()) or 1.0
    assert np.all(interior <= 1e-12 * scale), (name, float(interior.max()), scale)
