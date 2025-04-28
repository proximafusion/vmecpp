# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Hot-restart from a converged equilibrium."""

from pathlib import Path

import vmecpp

# Load the VMEC++ JSON indata file.
# Its keys have 1:1 correspondence with those in a classic Fortran INDATA file.
TEST_DATA_DIR = Path("src") / "vmecpp" / "cpp" / "vmecpp" / "test_data"
vmec_input_filename = TEST_DATA_DIR / "cth_like_free_bdy.json"
coils_fname = TEST_DATA_DIR / "coils.cth_like"
makegrid_params_fname = TEST_DATA_DIR / "makegrid_parameters_cth_like.json"

vmec_input = vmecpp.VmecInput.from_file(vmec_input_filename)
# We don't need an mgrid file, because we are passing the magnetic field as an object in memory
vmec_input.mgrid_file = ""

mgrid_params = vmecpp.MakegridParameters.from_file(makegrid_params_fname)
magnetic_response_table = vmecpp.MagneticFieldResponseTable.from_coils_file(
    coils_fname, mgrid_params
)
# Let's run VMEC++.
# In case of errors or non-convergence, a RuntimeError is raised.
# The OutputQuantities object returned has attributes corresponding
# to the usual outputs: wout, jxbout, mercier, ...
vmec_output = vmecpp.run(vmec_input, magnetic_field=magnetic_response_table)
print("  initial volume:", vmec_output.wout.volume_p)

# Now let's perturb the plasma boundary a little bit...
vmec_input.rbc[0, 0] *= 0.8
vmec_input.rbc[1, 0] *= 1.2

# ...and run VMEC++ again, but using its "hot restart" feature:
# passing the previously obtained output_quantities ensures that
# the run starts already close to the equilibrium, so it will take
# very few iterations to converge this time.
perturbed_output = vmecpp.run(
    vmec_input, magnetic_field=magnetic_response_table, restart_from=vmec_output
)
print("perturbed volume:", perturbed_output.wout.volume_p)
print("Difference:      ", vmec_output.wout.volume_p - perturbed_output.wout.volume_p)
