#!/usr/bin/env python3
"""Test symmetric mode convergence using Bazel-built library."""

import sys

sys.path.insert(0, "src/vmecpp/python")

# Now import the bazel-built module
from vmecpp import VmecInput
from vmecpp.cpp import _vmecpp


def test_symmetric_circular_tokamak():
    """Test that circular tokamak (symmetric) converges."""
    print("Testing symmetric circular tokamak convergence...")

    # Load circular tokamak (symmetric case)
    vmec_input = VmecInput.from_file(
        "src/vmecpp/cpp/vmecpp/test_data/circular_tokamak.json"
    )

    print(f"Testing circular tokamak with LASYM={vmec_input.lasym}")

    # Run VMEC using the bazel-built module directly
    data_dict = vmec_input.to_dict()

    # Convert Python data to C++ compatible format
    input_data = _vmecpp.VmecInputWrapper()
    for key, value in data_dict.items():
        if hasattr(input_data, key):
            setattr(input_data, key, value)

    # Run VMEC
    try:
        output = _vmecpp.run_vmec(input_data, verbose=True)
        if output:
            print(f"✅ Converged! Energy: {output.wb}")
            return True
        print("❌ Failed to converge")
        return False
    except Exception as e:
        print(f"❌ Exception: {e}")
        return False


if __name__ == "__main__":
    success = test_symmetric_circular_tokamak()
    exit(0 if success else 1)
