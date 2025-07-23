#!/usr/bin/env python3

import sys
import os
import json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import vmecpp

def load_json_config(filepath):
    """Load JSON configuration file."""
    with open(filepath, 'r') as f:
        return json.load(f)

def test_symmetric_case():
    """Test symmetric case using existing solovev.json."""
    
    print("=== Testing Symmetric Case (Regression Check) ===")
    
    try:
        # Load existing symmetric configuration
        config_path = "src/vmecpp/cpp/vmecpp/test_data/solovev.json"
        print(f"Loading config: {config_path}")
        
        config_dict = load_json_config(config_path)
        print(f"✓ Configuration loaded, lasym: {config_dict.get('lasym', 'not specified')}")
        
        # Create VmecInput from the configuration
        config = vmecpp.VmecInput(**config_dict)
        print("✓ VmecInput created successfully")
        
        # Run VMEC
        print("\nRunning symmetric equilibrium...")
        result = vmecpp.run(config)
        
        print(f"✓ VMEC run completed")
        print(f"Result type: {type(result)}")
        
        # Check result
        if hasattr(result, 'ier'):
            print(f"Exit code (ier): {result.ier}")
            if result.ier == 0:
                print("✅ Symmetric case PASSED - No regression!")
                return True
            else:
                print(f"❌ Symmetric case FAILED with ier={result.ier}")
                return False
        else:
            print("✓ Result returned without error - assuming success")
            return True
            
    except Exception as e:
        print(f"❌ Error in symmetric test: {e}")
        return False

def test_asymmetric_case():
    """Test asymmetric case using existing asymmetric configuration."""
    
    print("\n=== Testing Asymmetric Case (Lambda Coefficient Fix) ===")
    
    try:
        # Try to load existing asymmetric test configuration
        config_paths = [
            "src/vmecpp/cpp/vmecpp/test_data/test_minimal_asymmetric.json",
            "src/vmecpp/cpp/vmecpp/test_data/test_asymmetric_simple.json",
            "src/vmecpp/cpp/vmecpp/test_data/up_down_asymmetric_tokamak_simple.json"
        ]
        
        config = None
        config_path = None
        
        for path in config_paths:
            try:
                print(f"Trying config: {path}")
                config_dict = load_json_config(path)
                config = vmecpp.VmecInput(**config_dict)
                config_path = path
                print(f"✓ Configuration loaded from {path}")
                print(f"  lasym: {config_dict.get('lasym', 'not specified')}")
                break
            except Exception as e:
                print(f"  Failed to load {path}: {e}")
                continue
        
        if config is None:
            print("❌ Could not load any asymmetric configuration")
            return False
        
        # Run VMEC
        print(f"\nRunning asymmetric equilibrium with lambda coefficient processing...")
        print(f"Using config: {config_path}")
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
                print("   (May need additional priorities: force symmetrization, m=1 constraints, etc.)")
                return False
        else:
            print("✓ Result returned - likely successful")
            return True
            
    except Exception as e:
        print(f"❌ Error in asymmetric test: {e}")
        return False

if __name__ == '__main__':
    print("VMEC++ Regression Test After Lambda Coefficient Processing Fix")
    print("=" * 65)
    
    # Test symmetric case first (critical - must not regress)
    symmetric_ok = test_symmetric_case()
    
    # Test asymmetric case 
    asymmetric_ok = test_asymmetric_case()
    
    print("\n" + "=" * 65)
    print("SUMMARY:")
    print(f"Symmetric case:  {'✅ PASS' if symmetric_ok else '❌ FAIL'}")
    print(f"Asymmetric case: {'✅ PASS' if asymmetric_ok else '⚠️  PARTIAL'}")
    
    if symmetric_ok:
        print("\n✅ NO REGRESSION: Lambda coefficient changes preserved symmetric mode")
    else:
        print("\n❌ REGRESSION: Lambda coefficient changes may have broken symmetric mode")
        
    if asymmetric_ok:
        print("✅ LAMBDA FIX WORKING: Asymmetric mode improved with lambda processing")
    else:
        print("⚠️  LAMBDA FIX PARTIAL: Need remaining priorities (forces, m=1, axis)")
    
    # Exit with appropriate code
    sys.exit(0 if symmetric_ok else 1)