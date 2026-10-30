# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Hot-restart from a converged equilibrium."""

from pathlib import Path

import vmecpp

# Load the VMEC++ JSON indata file.
# Its keys have 1:1 correspondence with those in a classic Fortran INDATA file.
vmec_input_filename = Path("examples") / "data" / "cth_like_fixed_bdy.json"
vmec_input = vmecpp.VmecInput.from_file(vmec_input_filename)

# Let's run VMEC++.
# In case of errors or non-convergence, a RuntimeError is raised.
# The OutputQuantities object returned has attributes corresponding
# to the usual outputs: wout, jxbout, mercier, ...
vmec_output = vmecpp.run(vmec_input)
print("  initial volume:", vmec_output.wout.volume_p)

# Now let's perturb the plasma boundary a little bit...
vmec_input.rbc[0, 0] *= 0.8
vmec_input.rbc[1, 0] *= 1.2

# ...and run VMEC++ again, but using its "hot restart" feature:
# passing the previously obtained output_quantities ensures that
# the run starts already close to the equilibrium, so it will take
# very few iterations to converge this time.
perturbed_output = vmecpp.run(vmec_input, restart_from=vmec_output, verbose=False)
print("perturbed volume:", perturbed_output.wout.volume_p)
