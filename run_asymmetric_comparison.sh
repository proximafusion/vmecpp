#!/bin/bash

# Quick comparison of asymmetric up-down tokamak
# Run VMEC++ and jVMEC side by side

echo "=== Asymmetric Up-Down Tokamak Comparison ==="

INPUT_FILE="input.up_down_asymmetric_tokamak"

# Create debug directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
DEBUG_DIR="asymmetric_debug_${TIMESTAMP}"
mkdir -p "$DEBUG_DIR"

echo "Debug directory: $DEBUG_DIR"

# Convert input to JSON for VMEC++
echo "Converting input to JSON..."
python3 -c "
import vmecpp
vmec_input = vmecpp.VmecInput.from_file('$INPUT_FILE')
vmec_input.to_json('$DEBUG_DIR/input.json')
print(f'Converted to JSON: mpol={vmec_input.mpol}, ntor={vmec_input.ntor}, lasym={vmec_input.lasym}')
"

# Run VMEC++
echo "Running VMEC++..."
cd "$DEBUG_DIR"
python3 -c "
import vmecpp
try:
    vmec_input = vmecpp.VmecInput.from_file('input.json')
    print(f'Loaded: mpol={vmec_input.mpol}, ntor={vmec_input.ntor}, lasym={vmec_input.lasym}')
    result = vmecpp.run(vmec_input, verbose=True)
    print(f'SUCCESS: Volume={result.volume_p:.4f}, Beta={result.beta:.6f}')
except Exception as e:
    print(f'VMEC++ FAILED: {e}')
" > vmecpp_output.log 2>&1

echo "VMEC++ results in: $DEBUG_DIR/vmecpp_output.log"

# Run jVMEC for comparison
echo "Running jVMEC..."
cd /home/ert/code/jVMEC
java -jar target/jVMEC-1.0-SNAPSHOT.jar "/home/ert/code/vmecpp/$INPUT_FILE" > "/home/ert/code/vmecpp/$DEBUG_DIR/jvmec_output.log" 2>&1

echo "jVMEC results in: $DEBUG_DIR/jvmec_output.log"

# Compare results
cd /home/ert/code/vmecpp
echo "=== COMPARISON SUMMARY ==="
echo "VMEC++ output:"
tail -10 "$DEBUG_DIR/vmecpp_output.log"
echo
echo "jVMEC output:"
tail -10 "$DEBUG_DIR/jvmec_output.log"