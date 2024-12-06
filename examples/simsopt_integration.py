# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""How to run VMEC++ within SIMSOPT."""

from pathlib import Path

import vmecpp.simsopt_compat

# NOTE: This resolves to src/vmecpp/cpp/vmecpp/test_data in the repo.
TEST_DATA_DIR = (
    Path(__file__).parent.parent / "src" / "vmecpp" / "cpp" / "vmecpp" / "test_data"
)


def run_vmecpp_with_simsopt():
    input_file = TEST_DATA_DIR / "input.solovev"
    vmec = vmecpp.simsopt_compat.Vmec(input_file)
    print(f"Computed plasma volume: {vmec.volume()}")


if __name__ == "__main__":
    run_vmecpp_with_simsopt()
