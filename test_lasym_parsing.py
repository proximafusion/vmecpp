#!/usr/bin/env python3
"""Test to verify lasym is being parsed correctly"""

# Read solovev input file text
with open('src/vmecpp/cpp/vmecpp/test_data/input.solovev', 'r') as f:
    content = f.read()
    for line in content.split('\n'):
        if 'LASYM' in line.upper():
            print(f"Solovev text file line: {line.strip()}")
            
# Also check circular tokamak
print("\nCircular tokamak:")
with open('src/vmecpp/cpp/vmecpp/test_data/input.circular_tokamak', 'r') as f:
    content = f.read()
    for line in content.split('\n'):
        if 'LASYM' in line.upper():
            print(f"Circular tokamak file line: {line.strip()}")