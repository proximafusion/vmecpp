# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
import sys
from pathlib import Path

import vmecpp

input_file = Path(sys.argv[1])
input = vmecpp.VmecInput.from_file(input_file)
output = vmecpp.run(input)
wout_file = Path(f"wout_{vmecpp._util.get_vmec_configuration_name(input_file)}.nc")
output.wout.save(wout_file)

print(f"\nOutput written to {wout_file}.")  # noqa: T201
