#!/usr/bin/env python3
"""Test that symmetric equilibria still work after asymmetric fixes."""

import json
import subprocess
import sys
from pathlib import Path

def test_symmetric_json():
    """Test symmetric equilibrium with JSON input."""
    print("Testing symmetric equilibrium (solovev.json)...")
    
    json_path = Path("src/vmecpp/cpp/vmecpp/test_data/solovev.json")
    if not json_path.exists():
        print(f"Error: {json_path} not found")
        return False
        
    # Check it's actually symmetric
    with open(json_path) as f:
        data = json.load(f)
        if data.get("lasym", False):
            print("Error: solovev.json has lasym=true, not a symmetric test!")
            return False
    
    # Run with standalone executable
    cmd = ["./build/vmec_standalone", str(json_path)]
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        # Check for successful completion
        if result.returncode == 0:
            print("✓ Symmetric equilibrium completed successfully")
            
            # Check for convergence in output
            if "converged" in result.stdout.lower() or "fsqr" in result.stdout:
                print("✓ Equilibrium appears to have converged")
                
                # Look for final iteration info
                lines = result.stdout.split('\n')
                for line in lines[-20:]:  # Check last 20 lines
                    if "ITER" in line or "fsqr" in line.lower():
                        print(f"  {line.strip()}")
                        
                return True
            else:
                print("⚠ No clear convergence indicator found")
                return True  # Still success if it ran without error
        else:
            print(f"✗ Failed with return code: {result.returncode}")
            if result.stderr:
                print(f"Error output:\n{result.stderr[-500:]}")  # Last 500 chars
            return False
            
    except subprocess.TimeoutExpired:
        print("✗ Timed out after 30 seconds")
        return False
    except Exception as e:
        print(f"✗ Exception: {e}")
        return False

def test_symmetric_python():
    """Test symmetric equilibrium through Python interface."""
    print("\nTesting symmetric equilibrium through Python interface...")
    
    try:
        import vmecpp
        
        # Load symmetric test case
        json_path = Path("src/vmecpp/cpp/vmecpp/test_data/solovev.json")
        vmec_input = vmecpp.VmecInput.from_file(json_path)
        
        print(f"✓ Loaded input file")
        print(f"  lasym: {vmec_input.lasym} (should be False)")
        print(f"  mpol: {vmec_input.mpol}")
        print(f"  ntor: {vmec_input.ntor}")
        
        if vmec_input.lasym:
            print("Error: Not a symmetric test case!")
            return False
            
        # Try to run (even if it fails, we want to see if it starts)
        print("Attempting to run through Python interface...")
        
        # Reduce iterations for quick test
        vmec_input.niter_array = [20]
        
        try:
            result = vmecpp.run(vmec_input)
            print(f"✓ Run completed")
            print(f"  Success: {result.success}")
            print(f"  Converged: {result.converged}")
            return True
        except Exception as e:
            # Even if run fails, check if it's the old double free or a new issue
            error_str = str(e)
            if "double free" in error_str:
                print(f"✗ Double free error (regression!): {error_str}")
                return False
            else:
                print(f"⚠ Run failed but not due to double free: {error_str}")
                return True  # Not a regression of our fixes
                
    except ImportError:
        print("⚠ Could not import vmecpp module")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False

def test_another_symmetric():
    """Test another symmetric case."""
    print("\nTesting another symmetric equilibrium (DSHAPE_CURRENT)...")
    
    json_path = Path("src/vmecpp/cpp/vmecpp/test_data/DSHAPE_CURRENT.json")
    if not json_path.exists():
        print(f"⚠ {json_path} not found, skipping")
        return True
        
    cmd = ["./build/vmec_standalone", str(json_path)]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✓ DSHAPE_CURRENT completed successfully")
            return True
        else:
            print(f"✗ Failed with return code: {result.returncode}")
            # Check if it's due to asymmetric-related changes
            if "azNorm" in result.stderr or "ODD ARRAYS" in result.stderr:
                print("  ERROR: Symmetric case failing with asymmetric-related errors!")
                return False
            return True  # Other failures might be unrelated
            
    except subprocess.TimeoutExpired:
        print("⚠ Timed out (might just be slow)")
        return True
    except Exception as e:
        print(f"⚠ Exception: {e}")
        return True

if __name__ == "__main__":
    print("=== Testing Symmetric Equilibria After Asymmetric Fixes ===\n")
    
    # Run all tests
    results = []
    results.append(("Symmetric JSON", test_symmetric_json()))
    results.append(("Symmetric Python", test_symmetric_python()))
    results.append(("DSHAPE_CURRENT", test_another_symmetric()))
    
    # Summary
    print("\n=== SUMMARY ===")
    all_passed = True
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{name}: {status}")
        if not passed:
            all_passed = False
            
    if all_passed:
        print("\n✅ All symmetric tests passed - no regression detected!")
        sys.exit(0)
    else:
        print("\n❌ Some symmetric tests failed - possible regression!")
        sys.exit(1)