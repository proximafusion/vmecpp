#!/usr/bin/env python3
"""
Test script to reproduce and debug memory corruption in VMEC++
"""

import os
import sys
from vmecpp.cpp import _vmecpp as vmec

def test_memory_corruption():
    """Test case to reproduce memory corruption"""
    
    # Use a simple symmetric case that should work
    input_file = "src/vmecpp/cpp/vmecpp/test_data/solovev.json"
    
    print(f"Testing memory corruption with: {input_file}")
    
    try:
        # Load input
        indata = vmec.VmecINDATAPyWrapper.from_file(input_file)
        print(f"Loaded input: lasym={indata.lasym}, mpol={indata.mpol}, ntor={indata.ntor}")
        
        # Set small number of iterations
        indata.nstep = 10
        
        # Create a proper niter_array
        niter_array = [20] * indata.nstep
        indata.niter_array = niter_array
        
        print(f"nstep={indata.nstep}, niter_array={indata.niter_array}")
        
        # Run VMEC
        print("Starting VMEC run...")
        output = vmec.run(indata, verbose=True)
        
        print(f"Run completed successfully!")
        print(f"Beta = {output.beta}")
        print(f"Aspect ratio = {output.aspectratio}")
        
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Run with Python's memory debugger
    import faulthandler
    faulthandler.enable()
    
    test_memory_corruption()