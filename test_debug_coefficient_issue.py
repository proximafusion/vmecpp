#!/usr/bin/env python3
"""Debug coefficient indexing issue."""

import json
import numpy as np
from pathlib import Path

def analyze_test_case():
    """Analyze the test case structure."""
    
    json_path = Path("test_symmetric_quick.json")
    with open(json_path) as f:
        data = json.load(f)
    
    print("Test case analysis:")
    print(f"  mpol: {data['mpol']}")
    print(f"  ntor: {data['ntor']}")
    print(f"  lasym: {data['lasym']}")
    
    # Count coefficients
    rbc_count = len(data.get('rbc', []))
    zbs_count = len(data.get('zbs', []))
    
    print(f"\nBoundary coefficients:")
    print(f"  rbc entries: {rbc_count}")
    print(f"  zbs entries: {zbs_count}")
    
    # Expected array size
    expected_size = (data['mpol'] + 1) * (2 * data['ntor'] + 1)
    print(f"\nExpected array size: {expected_size}")
    print(f"  Calculation: ({data['mpol']} + 1) * (2 * {data['ntor']} + 1)")
    
    # Check coefficient m values
    print(f"\nrbc coefficients:")
    for coeff in data.get('rbc', []):
        print(f"  m={coeff['m']}, n={coeff['n']}, value={coeff['value']}")
    
    print(f"\nzbs coefficients:")
    for coeff in data.get('zbs', []):
        print(f"  m={coeff['m']}, n={coeff['n']}, value={coeff['value']}")
        
    # Check mnsize calculation
    mnsize = (data['mpol'] + 1) * (data['ntor'] + 1)
    print(f"\nmnsize calculation: ({data['mpol']} + 1) * ({data['ntor']} + 1) = {mnsize}")

if __name__ == "__main__":
    analyze_test_case()