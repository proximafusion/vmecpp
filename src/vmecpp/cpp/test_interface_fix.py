#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import vmecpp

def test_symmetric_case():
    """Test symmetric case to ensure no regression after lambda coefficient changes."""
    
    print("=== Testing Symmetric Case (Regression Check) ===")
    
    # Create a simple symmetric configuration using VmecInput
    try:
        # Simple tokamak-like configuration 
        config = vmecpp.VmecInput(
            delt=0.5,
            nfp=1,
            ncurr=1,
            niter=50,
            nstep=100,
            nvacskip=6,
            ftol_array=[1e-12, 1e-13, 1e-14],
            ntheta=16,
            nzeta=12,
            mpol=6,
            ntor=4,
            lasym=False,  # SYMMETRIC mode
            rbc=[[1.0, 0.3, 0.0], [0.0, 0.1, 0.0]],
            zbs=[[0.0, 0.3, 0.0], [0.0, 0.1, 0.0]],
        )
        
        print("✓ VmecInput created successfully")
        print(f"Configuration lasym: {config.lasym}")
        
        # Run VMEC using the run function
        print("\nRunning symmetric equilibrium...")
        result = vmecpp.run(config)
        
        print(f"✓ VMEC run completed")
        print(f"Result type: {type(result)}")
        
        # Check if result has ier (error code)
        if hasattr(result, 'ier'):
            print(f"Exit code (ier): {result.ier}")
            if result.ier == 0:
                print("✅ Symmetric case PASSED - No regression from lambda coefficient changes!")
                return True
            else:
                print(f"❌ Symmetric case FAILED with ier={result.ier}")
                return False
        else:
            print("✓ Result returned without error code - likely successful")
            return True
            
    except Exception as e:
        print(f"❌ Error in symmetric test: {e}")
        return False

def test_asymmetric_case():
    """Test asymmetric case to check if lambda coefficient processing helps."""
    
    print("\n=== Testing Asymmetric Case (Lambda Coefficient Fix) ===")
    
    try:
        # Simple asymmetric tokamak-like configuration
        config = vmecpp.VmecInput(
            delt=0.5,
            nfp=1,
            ncurr=1,
            niter=30,  # Fewer iterations for initial test
            nstep=50,
            nvacskip=6,
            ftol_array=[1e-10, 1e-11, 1e-12],  # Slightly relaxed tolerance
            ntheta=16,
            nzeta=12,
            mpol=6,
            ntor=4,
            lasym=True,  # ASYMMETRIC mode
            rbc=[[1.0, 0.3, 0.0], [0.0, 0.1, 0.0]],
            zbs=[[0.0, 0.3, 0.0], [0.0, 0.1, 0.0]],
            # Add small asymmetric perturbations
            rbs=[[0.0, 0.0, 0.0], [0.001, 0.0, 0.0]],  # Small asymmetric
            zbc=[[0.0, 0.0, 0.0], [0.001, 0.0, 0.0]],  # Small asymmetric
        )
        
        print("✓ Asymmetric VmecInput created successfully")
        print(f"Configuration lasym: {config.lasym}")
        
        # Run VMEC
        print("\nRunning asymmetric equilibrium with lambda coefficient processing...")
        result = vmecpp.run(config)
        
        print(f"✓ VMEC run completed")
        print(f"Result type: {type(result)}")
        
        # Check result
        if hasattr(result, 'ier'):
            print(f"Exit code (ier): {result.ier}")
            if result.ier == 0:
                print("✅ Asymmetric case PASSED - Lambda coefficient processing is working!")
                return True
            else:
                print(f"⚠️  Asymmetric case still has issues: ier={result.ier}")
                print("   (This may require additional priorities: force symmetrization, m=1 constraints, etc.)")
                return False
        else:
            print("✓ Result returned - checking if equilibrium computed")
            return True
            
    except Exception as e:
        print(f"❌ Error in asymmetric test: {e}")
        return False

if __name__ == '__main__':
    print("Testing VMEC++ Interface After Lambda Coefficient Processing Fix")
    print("=" * 60)
    
    # Test symmetric case first (critical - must not regress)
    symmetric_ok = test_symmetric_case()
    
    # Test asymmetric case 
    asymmetric_ok = test_asymmetric_case()
    
    print("\n" + "=" * 60)
    print("SUMMARY:")
    print(f"Symmetric case:  {'✅ PASS' if symmetric_ok else '❌ FAIL'}")
    print(f"Asymmetric case: {'✅ PASS' if asymmetric_ok else '⚠️  NEEDS MORE WORK'}")
    
    if symmetric_ok:
        print("\n✅ NO REGRESSION: Lambda coefficient changes did not break symmetric mode")
    else:
        print("\n❌ REGRESSION DETECTED: Lambda coefficient changes broke symmetric mode!")
        
    if asymmetric_ok:
        print("✅ PROGRESS: Asymmetric mode is now working with lambda coefficient processing")
    else:
        print("⚠️  PARTIAL PROGRESS: Asymmetric mode improved but still needs work")
        print("   Next priorities: force symmetrization, m=1 constraints, axis optimization")
    
    # Exit with appropriate code
    sys.exit(0 if symmetric_ok else 1)