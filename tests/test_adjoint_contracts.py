"""WP0 interfaces must import without claiming a numerical implementation."""

import numpy as np
import pytest

from vmecpp import _adjoint_contracts as contracts


@pytest.mark.parametrize(
    ("function", "args"),
    [
        (contracts.classify_state_entries, (None,)),
        (contracts.newton_polish, (None, 3, 1e-14)),
        (contracts.relabeling_generators, (None,)),
        (contracts.corrected_vjp, (None, np.zeros(1), "A")),
    ],
)
def test_unimplemented_interfaces_fail_explicitly(function, args):
    with pytest.raises(NotImplementedError):
        function(*args)


def test_diagnostics_fields():
    record = contracts.ImplicitVjpDiagnostics(
        force_residual=1e-8,
        solve_residual=1e-12,
        block_rcond=1e-6,
        class_sizes={"null": 2, "pinned": 1, "zero_column": 0, "active": 3},
        cotangent_on_null=0.0,
        cotangent_on_pinned=0.1,
        adjoint_weighted_residual=-1e-9,
    )
    assert record.max_relabeling_overlap is None
    assert record.class_sizes["active"] == 3
