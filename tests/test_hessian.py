# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
# <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""VmecModel.hessian_vector_product gives the augmented functional's curvature.

The Hessian-vector product is a central directional derivative of the analytic
force (the gradient of VMEC's augmented functional), computed inside VMEC++:
H v = (F(x + eps v) - F(x - eps v)) / (2 eps). It must be linear in v and agree
with an independent finite difference of the force, and it restores the state.
"""

from pathlib import Path

import numpy as np

from vmecpp.cpp import _vmecpp  # type: ignore

SOLOVEV = Path(__file__).resolve().parents[1] / "examples" / "data" / "solovev.json"


def _model(ns: int = 11):
    return _vmecpp.VmecModel.create(_vmecpp.VmecINDATA.from_file(str(SOLOVEV)), ns)


def _raw_force(model, x):
    model.set_state(np.ascontiguousarray(x))
    model.evaluate(2, 2, False)
    return np.asarray(model.get_forces(), float)


def test_hvp_matches_finite_difference():
    m = _model()
    m.evaluate(2, 2, False)
    x = np.asarray(m.get_state(), float)
    rng = np.random.default_rng(0)
    v = rng.standard_normal(x.size)
    v /= np.linalg.norm(v)

    hv = np.asarray(m.hessian_vector_product(np.ascontiguousarray(v)), float)

    eps = 1e-6
    fd = (_raw_force(m, x + eps * v) - _raw_force(m, x - eps * v)) / (2 * eps)
    assert np.linalg.norm(hv - fd) < 1e-5 * np.linalg.norm(fd)


def test_hvp_is_linear():
    m = _model()
    m.evaluate(2, 2, False)
    rng = np.random.default_rng(1)
    v = rng.standard_normal(np.asarray(m.get_state()).size)
    hv = np.asarray(m.hessian_vector_product(np.ascontiguousarray(v)), float)
    hv2 = np.asarray(m.hessian_vector_product(np.ascontiguousarray(2.0 * v)), float)
    assert np.linalg.norm(hv2 - 2.0 * hv) < 1e-9 * np.linalg.norm(hv)


def test_hvp_restores_state():
    m = _model()
    m.evaluate(2, 2, False)
    x0 = np.asarray(m.get_state(), float).copy()
    rng = np.random.default_rng(2)
    v = rng.standard_normal(x0.size)
    m.hessian_vector_product(np.ascontiguousarray(v))
    assert np.allclose(np.asarray(m.get_state(), float), x0)
