#!/usr/bin/env python3
import vmecpp

# Test symmetric case using working solovev.json
print("Testing symmetric case using working JSON file...")

try:
    # Load from working JSON file
    input_data = vmecpp.VmecInput.from_file("/home/ert/code/vmecpp/examples/data/solovev.json")
    
    # Ensure it's symmetric
    input_data.lasym = False
    
    # Ensure asymmetric fields are None for symmetric case
    input_data.rbs = None
    input_data.zbc = None  
    input_data.raxis_s = None
    input_data.zaxis_c = None
    
    print("Running symmetric test from JSON file...")
    result = vmecpp.run(input_data)
    print("SUCCESS: Symmetric test passed")
    print(f"Beta = {result.beta:.6f}")
    
except Exception as e:
    print(f"ERROR: Symmetric test failed: {e}")
    import traceback
    traceback.print_exc()