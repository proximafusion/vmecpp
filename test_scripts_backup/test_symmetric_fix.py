#!/usr/bin/env python3
"""Test symmetric mode convergence with circular tokamak."""

import vmecpp


def test_symmetric_circular_tokamak():
    """Test that circular tokamak (symmetric) converges."""
    print("Testing symmetric circular tokamak convergence...")

    # Load circular tokamak (symmetric case)
    vmec_input = vmecpp.VmecInput.from_file(
        "src/vmecpp/cpp/vmecpp/test_data/circular_tokamak.json"
    )

    print(f"Testing circular tokamak with LASYM={vmec_input.lasym}")
    try:
        output = vmecpp.run(vmec_input, verbose=True)
        print(f"✅ Converged! Energy: {output.wout.wb}")
        return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False


if __name__ == "__main__":
    success = test_symmetric_circular_tokamak()
    exit(0 if success else 1)
