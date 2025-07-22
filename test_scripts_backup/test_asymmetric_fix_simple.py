#!/usr/bin/env python3
"""Test asymmetric equilibria with existing input file."""

import vmecpp

def test_asymmetric_with_input_file():
    """Test asymmetric tokamak from existing input file."""
    print("Testing asymmetric tokamak from input file...")
    
    # Load the asymmetric tokamak test case
    input_file = "examples/data/input.up_down_asymmetric_tokamak"
    vmec_input = vmecpp.VmecInput.from_file(input_file)
    
    # Verify it's an asymmetric run
    assert vmec_input.lasym is True
    print(f"Loaded asymmetric input: lasym={vmec_input.lasym}")
    print(f"mpol={vmec_input.mpol}, ntor={vmec_input.ntor}")
    
    # Run VMEC
    try:
        result = vmecpp.run(vmec_input, verbose=True)
        
        print(f"\nRun completed! Final force residual: {result.wout.fsqr:.2e}")
        print(f"Number of iterations: {result.wout.iter}")
        print(f"Converged: {result.wout.fsqr < 1e-6}")
        
        return True
        
    except Exception as e:
        print(f"Run failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_asymmetric_with_input_file()
    print(f"\nTest {'PASSED' if success else 'FAILED'}")