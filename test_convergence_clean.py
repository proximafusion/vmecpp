#!/usr/bin/env python3
from vmecpp.cpp import _vmecpp as vmec

# Test cases
test_cases = [
    ("src/vmecpp/cpp/vmecpp/test_data/solovev.json", "Solovev (symmetric)"),
    ("src/vmecpp/cpp/vmecpp/test_data/circular_tokamak.json", "Circular tokamak"),
    ("test_asymmetric_proper.json", "Asymmetric tokamak"),
]

for json_file, description in test_cases:
    print(f"\n{'='*60}")
    print(f"Testing: {description}")
    print(f"File: {json_file}")
    print('='*60)
    
    try:
        indata = vmec.VmecINDATAPyWrapper.from_file(json_file)
        
        # Get configuration
        print(f"Config: LASYM={indata.lasym}, NFP={indata.nfp}, MPOL={indata.mpol}, NTOR={indata.ntor}")
        print(f"NS array: {list(indata.ns_array)}")
        print(f"FTOL array: {list(indata.ftol_array)}")
        
        # Run VMEC
        output = vmec.run(indata)
        
        # Check results
        ier_flag = output.wout.ier_flag
        ftolv = output.wout.ftolv
        niter = output.wout.maximum_iterations
        
        if ier_flag == 0:
            print(f"\n✓ CONVERGED")
            print(f"  Final residual: {ftolv:.2e}")
            print(f"  Iterations: {niter}")
            print(f"  Beta: {output.wout.betatot:.4f}")
            print(f"  Aspect ratio: {output.wout.aspect:.4f}")
        else:
            print(f"\n✗ FAILED TO CONVERGE")
            print(f"  Error flag: {ier_flag}")
            print(f"  Final residual: {ftolv:.2e}")
            print(f"  Iterations: {niter}")
            
    except FileNotFoundError:
        print(f"✗ FILE NOT FOUND: {json_file}")
    except Exception as e:
        print(f"✗ ERROR: {e}")