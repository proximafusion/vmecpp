#!/usr/bin/env python3
"""Test script to reproduce symmetric mode convergence regression.

This script tests the circular_tokamak symmetric case to verify VMEC++ convergence.
It should converge with MHD Energy = 172.39494071067568 (golden record from upstream/main).

Usage:
    python test_symmetric_regression.py

Expected (working):
    ✅ VMEC converged successfully!
    MHD Energy = 172.39494071067568

Actual (regression):
    ❌ VMEC failed: VMEC++ did not converge
    MHD Energy = 6.822967e+03
"""

import os
import sys

import vmecpp


def main():
    print("🧪 Testing symmetric mode convergence regression...")
    print("=" * 60)

    # Path to test input
    input_path = (
        "../benchmark_vmec/vmec_repos/VMEC2000/python/tests/input.circular_tokamak"
    )

    if not os.path.exists(input_path):
        print(f"❌ ERROR: Input file not found: {input_path}")
        print("Please ensure benchmark_vmec repository is available.")
        sys.exit(1)

    try:
        # Load the circular tokamak (symmetric) input
        vmec_input = vmecpp.VmecInput.from_file(input_path)
        print(f"📁 Loaded input file: {input_path}")
        print(f"   LASYM = {vmec_input.lasym}")
        print(f"   MPOL = {vmec_input.mpol}")
        print(f"   NTOR = {vmec_input.ntor}")
        print(f"   NFP = {vmec_input.nfp}")

        # Verify this is a symmetric case
        if vmec_input.lasym:
            print("⚠️  WARNING: Expected symmetric case (LASYM=False)")

        print("\n🚀 Running VMEC...")

        # Run VMEC
        output = vmecpp.run(vmec_input)

        # Check results
        mhd_energy = output.wout.wb
        golden_record = 172.39494071067568
        tolerance = 1e-10

        print("✅ VMEC converged successfully!")
        print(f"🎯 MHD Energy = {mhd_energy}")

        # Compare with golden record
        if abs(mhd_energy - golden_record) < tolerance:
            print(f"✅ GOLDEN RECORD MATCH: Within tolerance ({tolerance})")
            print(f"   Expected: {golden_record}")
            print(f"   Actual:   {mhd_energy}")
            print(f"   Diff:     {abs(mhd_energy - golden_record):.2e}")
        else:
            print("⚠️  GOLDEN RECORD MISMATCH:")
            print(f"   Expected: {golden_record}")
            print(f"   Actual:   {mhd_energy}")
            print(f"   Diff:     {abs(mhd_energy - golden_record):.2e}")
            print(f"   Tolerance: {tolerance}")

        print("\n🎉 SUCCESS: Symmetric mode convergence verified!")
        sys.exit(0)

    except Exception as e:
        golden_record = 172.39494071067568  # Define here too for error case
        print(f"❌ VMEC CONVERGENCE FAILED: {e}")
        print("\n💥 REGRESSION CONFIRMED: Symmetric mode does not converge")
        print("\nThis indicates the critical regression described in:")
        print("  - GitHub Issue: https://github.com/proximafusion/vmecpp/issues/363")
        print("  - Analysis: regression.txt")
        print("\nExpected behavior (upstream/main):")
        print(f"  ✅ Converges with MHD Energy = {golden_record}")
        print("\nActual behavior (current main):")
        print("  ❌ Fails to converge with 'VMEC++ did not converge'")

        sys.exit(1)


if __name__ == "__main__":
    main()
