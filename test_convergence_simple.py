#!/usr/bin/env python3
"""Test convergence after asymmetric fix"""

import os
import sys
import subprocess

# Ensure vmecpp module can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src/vmecpp/python')))

# Build the C++ extension
print("Building vmecpp C++ extension...")
build_result = subprocess.run([sys.executable, 'setup.py', 'build_ext', '--inplace'], 
                            capture_output=True, text=True)
if build_result.returncode != 0:
    print("Build failed!")
    print(build_result.stdout)
    print(build_result.stderr)
    sys.exit(1)

import vmecpp

print("Testing Solovev equilibrium convergence after asymmetric fix...")

# Create vmec object
vmec = vmecpp.vmec.VMEC()

# Load Solovev test input
vmec.indata.read_indata_file("src/vmecpp/cpp/vmecpp/test_data/input.solovev")

# Run VMEC
result = vmec.run()

# Check result
if result and result.fsqr < 1e-14:
    print(f"SUCCESS: Solovev equilibrium converged! fsqr = {result.fsqr}")
    print(f"         Iterations: {result.iter}")
    print(f"         MHD Energy: {result.wdot}")
else:
    print(f"FAILED: Solovev equilibrium did not converge")
    if result:
        print(f"        fsqr = {result.fsqr}")
        print(f"        iter = {result.iter}")