#!/usr/bin/env python3
from vmecpp.cpp import _vmecpp as vmec
import os

# List of all JSON test files
json_files = [
    # Symmetric cases
    ("src/vmecpp/cpp/vmecpp/test_data/solovev.json", "Solovev (symmetric)"),
    ("src/vmecpp/cpp/vmecpp/test_data/circular_tokamak.json", "Circular tokamak"),
    ("src/vmecpp/cpp/vmecpp/test_data/solovev_analytical.json", "Solovev analytical"),
    ("src/vmecpp/cpp/vmecpp/test_data/solovev_no_axis.json", "Solovev no axis"),
    ("src/vmecpp/cpp/vmecpp/test_data/cma.json", "CMA"),
    ("src/vmecpp/cpp/vmecpp/test_data/cth_like_fixed_bdy.json", "CTH-like fixed boundary"),
    ("src/vmecpp/cpp/vmecpp/test_data/cth_like_fixed_bdy_nzeta_37.json", "CTH-like fixed boundary nzeta=37"),
    ("src/vmecpp/cpp/vmecpp/test_data/cth_like_free_bdy.json", "CTH-like free boundary"),
    ("src/vmecpp/cpp/vmecpp/test_data/li383_low_res.json", "LI383 low res"),
    
    # Asymmetric cases
    ("src/vmecpp/cpp/vmecpp/test_data/up_down_asymmetric_tokamak.json", "Up-down asymmetric tokamak"),
    ("src/vmecpp/cpp/vmecpp/test_data/input.up_down_asymmetric_tokamak.json", "Up-down asymmetric tokamak (input.)"),
    ("src/vmecpp/cpp/vmecpp/test_data/up_down_asymmetric_tokamak_simple.json", "Up-down asymmetric tokamak simple"),
]

print("=" * 80)
print("VMEC++ Convergence Test Summary")
print("=" * 80)

results = []

for json_file, description in json_files:
    if not os.path.exists(json_file):
        results.append((description, "FILE NOT FOUND", "-", "-", "-"))
        continue
        
    print(f"\nTesting: {description}")
    print(f"File: {json_file}")
    print("-" * 60)
    
    try:
        indata = vmec.VmecINDATAPyWrapper.from_file(json_file)
        
        # Print configuration
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
            status = "CONVERGED"
            print(f"✓ {status}")
            print(f"  Final residual: {ftolv:.2e}")
            print(f"  Iterations: {niter}")
            beta = output.wout.betatot
            aspect = output.wout.aspect
            print(f"  Beta: {beta:.4f}")
            print(f"  Aspect ratio: {aspect:.4f}")
        else:
            status = f"NOT CONVERGED (ier={ier_flag})"
            print(f"✗ {status}")
            print(f"  Final residual: {ftolv:.2e}")
            print(f"  Iterations: {niter}")
            beta = "-"
            aspect = "-"
            
        results.append((description, status, f"{ftolv:.2e}", str(niter), f"{beta}" if beta != "-" else "-"))
        
    except Exception as e:
        status = f"ERROR: {str(e)[:50]}..."
        print(f"✗ {status}")
        results.append((description, status, "-", "-", "-"))

# Print summary table
print("\n" + "=" * 80)
print("SUMMARY TABLE")
print("=" * 80)
print(f"{'Test Case':<40} {'Status':<25} {'Residual':<12} {'Iter':<8} {'Beta':<8}")
print("-" * 80)

for desc, status, res, niter, beta in results:
    print(f"{desc:<40} {status:<25} {res:<12} {niter:<8} {beta:<8}")

print("=" * 80)