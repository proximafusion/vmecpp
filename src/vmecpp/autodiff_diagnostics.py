"""Force-definition handoff for adjoint diagnostics and Newton polishing.

``get_forces()`` returns the model's current decomposed force buffer. A full
``evaluate`` preconditions that buffer; exact HVPs overwrite it with Jacobian
products. It is not a persistent raw residual accessor.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def raw_force(model: Any) -> np.ndarray:
    """Refresh and copy the raw decomposed residual differentiated by exact H.

    Equal iteration counters refresh the radial preconditioner and tcon; the invariant-
    residual checkpoint stops before force preconditioning. The pinned gauge flag
    matches the exact HVP default. The state is unchanged except for the native clearing
    of modes above the geometry resolution. A bad Jacobian/restart cannot supply a valid
    residual and is rejected.
    """
    if not model.always_fix_m1_gauge:
        error_message = "raw adjoint residual requires always_fix_m1_gauge=True"
        raise ValueError(error_message)
    model.evaluate(2, 2, False, True)
    if model.need_restart:
        error_message = "raw adjoint residual evaluation requires a valid Jacobian"
        raise RuntimeError(error_message)
    return np.asarray(model.get_forces(), dtype=np.float64).copy()
