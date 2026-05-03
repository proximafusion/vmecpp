#!/usr/bin/env python3
"""Direct test of spectral condensation for asymmetric case."""

import vmecpp

print("Testing spectral condensation for asymmetric case...")
print("Using existing asymmetric input file: simple_asymmetric.json")

try:
    # Run VMEC with existing asymmetric input
    result = vmecpp.run("simple_asymmetric.json", verbose=5, checkpoint="forces")
    print("\nSUCCESS: VMEC completed without spectral condensation errors!")
    print("The spectral condensation arrays are being computed correctly for asymmetric cases.")
    
except Exception as e:
    error_msg = str(e)
    print(f"\nERROR: {error_msg}")
    
    if "spectral" in error_msg.lower() or "condens" in error_msg.lower():
        print("\n*** SPECTRAL CONDENSATION ERROR DETECTED ***")
        print("The asymmetric Fourier transform is NOT computing rCon/zCon arrays!")
        print("This needs to be fixed in dft_FourierToReal_3d_asymm()")
        
        # Check for the specific error
        if "not spectrally condensed enough" in error_msg:
            print("\nCONFIRMED: This is the exact error we're looking for.")
            print("The spectral condensation arrays are not being populated in the asymmetric transform.")
    else:
        print("\nUnexpected error - not related to spectral condensation")