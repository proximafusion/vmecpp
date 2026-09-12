# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
# <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""VmecModel accepts arbitrary Python attributes.

``py::class_<VmecModel>`` is registered with ``py::dynamic_attr()``, giving each
instance a ``__dict__``. This lets Python code attach state to a solved model,
for example a cached factorization or metadata, without VMEC++ having to know
about it.
"""

from pathlib import Path

import numpy as np

from vmecpp.cpp import _vmecpp  # type: ignore

SOLOVEV = Path(__file__).resolve().parents[1] / "examples" / "data" / "solovev.json"


def _model(ns: int = 11):
    return _vmecpp.VmecModel.create(_vmecpp.VmecINDATA.from_file(str(SOLOVEV)), ns)


def test_arbitrary_attribute_can_be_set_and_read_back() -> None:
    model = _model()
    model.cached_factorization = {"marker": 42}

    assert model.cached_factorization == {"marker": 42}


def test_dynamic_attribute_does_not_interfere_with_get_state_and_solve() -> None:
    model = _model()
    model.some_metadata = "arbitrary"

    state_before = np.asarray(model.get_state(), dtype=np.float64).copy()
    model.solve()
    state_after = np.asarray(model.get_state(), dtype=np.float64)

    assert model.some_metadata == "arbitrary"
    assert state_before.shape == state_after.shape
    assert np.all(np.isfinite(state_after))
