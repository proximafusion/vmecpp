#!/usr/bin/env python3
"""Debug symmetric mode convergence."""

import vmecpp

vmec_input = vmecpp.VmecInput.from_file(
    "src/vmecpp/cpp/vmecpp/test_data/circular_tokamak.json"
)
print(f"Testing circular tokamak with LASYM={vmec_input.lasym}")

try:
    # Run with verbose to see convergence info
    output = vmecpp.run(vmec_input, verbose=True)
    print(f"✅ Converged! Energy: {output.wout.wb}")
except Exception as e:
    print(f"❌ Failed: {e}")
    # Check if it's actually a convergence issue
    if "did not converge" in str(e):
        print("Issue: VMEC++ did not converge within maximum iterations")
    else:
        print(f"Other error: {type(e).__name__}")
