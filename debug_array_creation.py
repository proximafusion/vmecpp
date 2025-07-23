#!/usr/bin/env python3
"""Debug array creation for boundary coefficients."""

import json
import numpy as np
from pathlib import Path

def debug_coefficient_conversion():
    """Debug how coefficients are converted from JSON."""
    
    json_path = Path("test_symmetric_quick.json")
    with open(json_path) as f:
        data = json.load(f)
    
    print("=== JSON Data ===")
    print(f"mpol: {data['mpol']}")
    print(f"ntor: {data['ntor']}")
    print(f"rbc: {data['rbc']}")
    print(f"zbs: {data['zbs']}")
    
    # Manual sparse to dense conversion like the Python code does
    print("\n=== Manual Sparse to Dense Conversion ===")
    
    # From _util.py sparse_to_dense_coefficients_implicit
    rbc_sparse = data['rbc']
    mpol_inferred = 0
    ntor_inferred = 0
    for entry in rbc_sparse:
        mpol_inferred = max(mpol_inferred, entry['m'])
        ntor_inferred = max(ntor_inferred, abs(entry['n']))
    
    mpol_inferred += 1  # This is the key line from _util.py:413
    
    print(f"Inferred mpol from rbc: {mpol_inferred}")
    print(f"Inferred ntor from rbc: {ntor_inferred}")
    
    # Create dense array
    dense_rbc = np.zeros((mpol_inferred, 2 * ntor_inferred + 1))
    for entry in rbc_sparse:
        m = entry['m']
        n = entry['n']
        val = entry['value']
        col_idx = n + ntor_inferred
        dense_rbc[m, col_idx] = val
    
    print(f"Dense rbc shape: {dense_rbc.shape}")
    print(f"Dense rbc array:\n{dense_rbc}")
    
    # Expected by C++ validation
    expected_size = data['mpol'] * (2 * data['ntor'] + 1)
    actual_size = dense_rbc.size
    
    print(f"\nC++ validation expects: {expected_size}")
    print(f"Python creates array size: {actual_size}")
    print(f"Match: {expected_size == actual_size}")

if __name__ == "__main__":
    debug_coefficient_conversion()