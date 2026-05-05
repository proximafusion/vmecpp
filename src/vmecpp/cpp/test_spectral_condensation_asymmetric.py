#!/usr/bin/env python3
"""Test to verify spectral condensation is computed for asymmetric cases."""

import json
import vmecpp

def test_spectral_condensation_asymmetric():
    """Test that spectral condensation arrays are populated for asymmetric cases."""
    
    # Load the symmetric test case and modify it to be asymmetric
    with open("test_symmetric_quick.json", "r") as f:
        data = json.load(f)
    
    # Enable asymmetric mode
    data["lasym"] = True
    
    # Add some asymmetric boundary coefficients
    # Find m=1, n=0 mode
    for i, (m, n) in enumerate(zip(data["xm"], data["xn"])):
        if m == 1 and n == 0:
            # Add small asymmetric perturbations
            data["rbs"][i] = 0.01  # R sin component
            data["zbc"][i] = 0.01  # Z cos component
            break
    
    # Create modified input file
    modified_file = "test_asymmetric_spectral.json"
    with open(modified_file, "w") as f:
        json.dump(data, f, indent=2)
    
    print("Running VMEC with asymmetric configuration...")
    try:
        # Run VMEC with debug output
        vmec = vmecpp.run_vmec(
            input_file=modified_file,
            verbose=5,
            checkpoint="forces"
        )
        print("\nVMEC completed successfully!")
        print("If VMEC runs without 'not spectrally condensed' errors, the fix is working")
        
    except Exception as e:
        print(f"\nError: {e}")
        error_str = str(e)
        if "not spectrally condensed enough" in error_str:
            print("\nCONFIRMED: Spectral condensation is not being computed for asymmetric cases!")
            print("The asymmetric transform needs to compute rCon and zCon arrays")
        elif "spectral" in error_str.lower() or "condens" in error_str.lower():
            print("\nSpectral condensation related error detected")
        raise

if __name__ == "__main__":
    test_spectral_condensation_asymmetric()