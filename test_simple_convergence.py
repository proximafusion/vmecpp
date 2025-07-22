#!/usr/bin/env python3
"""Simple test to check convergence"""
try:
    from test_solovev_baseline import test_solovev_baseline
    print("Testing convergence after buffer overflow fix...")
    result = test_solovev_baseline()
    if result:
        print("✅ SUCCESS! Fix appears to work!")
    else:
        print("❌ FAILED: Still not converging")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()