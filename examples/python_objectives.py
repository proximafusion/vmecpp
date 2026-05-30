# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Stellarator-optimization objectives on a converged VMEC++ equilibrium.

``vmecpp.objectives`` computes standard design targets from a converged
equilibrium: the aspect ratio, the rotational-transform profile, the magnetic
well, the mirror ratio, and the Boozer quasi-symmetry residual (with the
Boozer-coordinate transform it requires). The aspect ratio, iota, magnetic well,
and mirror-ratio field match SIMSOPT to numerical precision; the Boozer transform
is validated by self-consistency. Here we solve a fixed-boundary stellarator and
report each objective.
"""

from pathlib import Path

import vmecpp
from vmecpp import objectives

vmec_output = vmecpp.run(
    vmecpp.VmecInput.from_file(Path("examples") / "data" / "cth_like_fixed_bdy.json"),
    max_threads=1,
    verbose=False,
)

_, iota = objectives.iota_profile(vmec_output)

print(f"aspect ratio            : {objectives.aspect_ratio(vmec_output):.4f}")
print(f"iota on axis / edge     : {iota[0]:.4f} / {iota[-1]:.4f}")
print(f"magnetic well           : {objectives.magnetic_well(vmec_output):+.4e}")
print(f"mirror ratio            : {objectives.mirror_ratio(vmec_output):.4f}")
print(
    "quasi-axisymmetry resid : "
    f"{objectives.quasisymmetry_residual(vmec_output, helicity_m=1, helicity_n=0):.4e}"
)
print(
    f"Boozer self-consistency : {objectives.boozer_roundtrip_residual(vmec_output):.2e}"
)
