# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Drive the VMEC++ force-balance iteration from Python.

VMEC++'s equilibrium iteration (time stepping toward force balance) is ported to
Python in ``vmecpp._iteration``, driving the C++ forward model (flux-surface
geometry -> MHD forces) exposed as ``VmecModel``. Keeping the expensive forward
model in C++ while owning the iteration *logic* in Python is the foundation for
developing alternative iteration schemes without modifying the C++ core.

Here we solve a single-resolution fixed-boundary equilibrium with the Python loop
and report convergence.
"""

from pathlib import Path

import vmecpp

vmec_input = vmecpp.VmecInput.from_file(
    Path("examples") / "data" / "cth_like_fixed_bdy.json"
)

# vmecpp.iterate builds a single-resolution VmecModel (forward model held in C++)
# and runs the ported Python iteration on it.
model, result = vmecpp.iterate(vmec_input)

print(f"converged      : {result.converged}")
print(f"iterations     : {result.num_iterations}")
print(f"axis reguesses : {result.axis_reguesses}")
print(
    f"force residuals: fsqr={result.fsqr:.2e} fsqz={result.fsqz:.2e} "
    f"fsql={result.fsql:.2e}"
)
