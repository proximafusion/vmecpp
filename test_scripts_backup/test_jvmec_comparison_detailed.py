#!/usr/bin/env python3
"""Detailed comparison between VMEC++ and jVMEC for asymmetric equilibria."""

import vmecpp
import numpy as np
import subprocess
import json
import os

def run_jvmec_with_output(input_file, max_iterations=1):
    """Run jVMEC and capture detailed debug output."""
    # Create a temporary input file with debug flags
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    # Create modified input with just 1 iteration for debugging
    temp_input = "temp_jvmec_input"
    with open(temp_input, 'w') as f:
        for line in lines:
            if line.strip().startswith('NITER'):
                f.write(f"  NITER = {max_iterations}\n")
            else:
                f.write(line)
    
    # Run jVMEC and capture output
    try:
        cmd = ["java", "-jar", "jVMEC-1.0.0.jar", temp_input]
        result = subprocess.run(cmd, capture_output=True, text=True)
        print("jVMEC stdout:")
        print(result.stdout)
        print("\njVMEC stderr:")
        print(result.stderr)
        
        # Clean up
        os.remove(temp_input)
        return result.stdout
    except Exception as e:
        print(f"Error running jVMEC: {e}")
        return None

def extract_jvmec_values(output):
    """Extract key values from jVMEC output."""
    values = {}
    lines = output.split('\n')
    
    for i, line in enumerate(lines):
        # Look for Fourier coefficients
        if 'rmncc(' in line or 'rmnsc(' in line:
            print(f"Found coefficient line: {line}")
        
        # Look for geometry values
        if 'R1(' in line or 'Z1(' in line:
            print(f"Found geometry line: {line}")
            
        # Look for derivative values  
        if 'dRdu' in line or 'dZdu' in line:
            print(f"Found derivative line: {line}")
            
        # Look for Jacobian info
        if 'tau' in line or 'gsqrt' in line:
            print(f"Found Jacobian line: {line}")
    
    return values

def test_detailed_comparison():
    print("=== DETAILED VMEC++ vs jVMEC COMPARISON ===\n")
    
    # Load asymmetric tokamak configuration
    vmec_input = vmecpp.VmecInput.from_file("examples/data/input.up_down_asymmetric_tokamak")
    
    # Run with just 1 iteration to see initial state
    vmec_input.niter_array = np.array([1], dtype=np.int64)
    vmec_input.return_outputs_even_if_not_converged = True
    
    print("Configuration:")
    print(f"  lasym = {vmec_input.lasym}")
    print(f"  ns = {vmec_input.ns_array[0]}")
    print(f"  mpol = {vmec_input.mpol}")
    print(f"  ntor = {vmec_input.ntor}")
    
    # Show asymmetric coefficients
    print("\nAsymmetric boundary coefficients:")
    for rbs in vmec_input.rbs:
        if abs(rbs['value']) > 1e-12:
            print(f"  RBS({rbs['m']},{rbs['n']}) = {rbs['value']}")
    for zbs in vmec_input.zbs:
        if abs(zbs['value']) > 1e-12:
            print(f"  ZBS({zbs['m']},{zbs['n']}) = {zbs['value']}")
    for zbc in vmec_input.zbc:
        if abs(zbc['value']) > 1e-12:
            print(f"  ZBC({zbc['m']},{zbc['n']}) = {zbc['value']}")
    
    print("\n--- Running jVMEC ---")
    jvmec_output = run_jvmec_with_output("examples/data/input.up_down_asymmetric_tokamak", max_iterations=1)
    
    print("\n--- Running VMEC++ ---")
    try:
        result = vmecpp.run(vmec_input, verbose=True)
        print(f"VMEC++ completed (fsqr={result.fsqr:.6e}, fsqz={result.fsqz:.6e})")
        
        # Compare Fourier coefficients
        if hasattr(result, 'rmnc') and result.rmnc is not None:
            print("\nVMEC++ Fourier coefficients (first surface):")
            for m in range(min(5, result.rmnc.shape[1])):
                rmnc = result.rmnc[0, m] if m < result.rmnc.shape[1] else 0.0
                zmns = result.zmns[0, m] if m < result.zmns.shape[1] else 0.0
                print(f"  m={m}: rmnc={rmnc:.6f}, zmns={zmns:.6f}")
                
                if hasattr(result, 'rmns') and result.rmns is not None:
                    rmns = result.rmns[0, m] if m < result.rmns.shape[1] else 0.0
                    zmnc = result.zmnc[0, m] if m < result.zmnc.shape[1] else 0.0
                    print(f"       rmns={rmns:.6f}, zmnc={zmnc:.6f}")
        
    except RuntimeError as e:
        print(f"\nVMEC++ failed: {e}")

if __name__ == "__main__":
    test_detailed_comparison()