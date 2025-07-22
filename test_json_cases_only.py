#!/usr/bin/env python3
"""
Test only JSON cases to avoid INDATA parsing issues
"""

import glob
import vmecpp
from pathlib import Path

def test_json_case(filepath):
    """Test a single JSON case"""
    name = Path(filepath).stem
    print(f"\n{'='*50}")
    print(f"Testing: {name}")
    
    try:
        # Load the JSON file
        vmec_input = vmecpp.VmecInput.from_file(filepath)
        
        print(f"  lasym={vmec_input.lasym}, mpol={vmec_input.mpol}, ntor={vmec_input.ntor}")
        print(f"  rbc.shape={vmec_input.rbc.shape}")
        
        # Very limited run
        vmec_input.nstep = 1
        vmec_input.niter_array = [5]
        
        # Try to run
        output = vmecpp.run(vmec_input, verbose=False)
        
        print(f"  ✅ SUCCESS! Volume={output.volume_p:.3f}")
        return True, vmec_input.lasym
        
    except Exception as e:
        error_msg = str(e)
        if "azNorm should never be 0.0" in error_msg:
            print(f"  ❌ FAILED: azNorm=0 error")
            return False, vmec_input.lasym if 'vmec_input' in locals() else None
        else:
            print(f"  ❌ FAILED: {error_msg[:80]}...")
            return None, vmec_input.lasym if 'vmec_input' in locals() else None

def main():
    print("=== Testing JSON Cases Only ===")
    
    # Find JSON files
    json_files = sorted(glob.glob("src/vmecpp/cpp/vmecpp/test_data/*.json"))
    print(f"Found {len(json_files)} JSON test cases")
    
    success_count = 0
    aznorm_count = 0
    other_count = 0
    asymmetric_success = 0
    
    for filepath in json_files:
        result, is_asymmetric = test_json_case(filepath)
        if result is True:
            success_count += 1
            if is_asymmetric:
                asymmetric_success += 1
        elif result is False:
            aznorm_count += 1
        else:
            other_count += 1
    
    print(f"\n{'='*50}")
    print("=== SUMMARY ===")
    print(f"Total: {len(json_files)}")
    print(f"✅ Success: {success_count} (including {asymmetric_success} asymmetric)")
    print(f"❌ azNorm errors: {aznorm_count}")
    print(f"❌ Other errors: {other_count}")
    
    if aznorm_count == 0 and asymmetric_success > 0:
        print("\n🎉 azNorm=0 error eliminated for asymmetric cases!")

if __name__ == "__main__":
    main()