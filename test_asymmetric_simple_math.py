#!/usr/bin/env python3
"""Check basic asymmetric Fourier math"""

import numpy as np

# Test configuration
mpol = 3
ntor = 0
ntheta = 12  # Full [0, 2pi]

# Fourier coefficients
RBC = {(0,0): 1.0, (1,0): 0.3}  # Symmetric
ZBS = {(1,0): 0.3}               # Symmetric  
RBS = {(1,0): 0.001}             # Asymmetric

# Compute expected values
theta = np.linspace(0, 2*np.pi, ntheta, endpoint=False)

print("=== EXPECTED VALUES ===")
print(f"Configuration: mpol={mpol}, ntor={ntor}")
print(f"Coefficients:")
print(f"  RBC(0,0) = {RBC.get((0,0), 0)}")
print(f"  RBC(1,0) = {RBC.get((1,0), 0)}")  
print(f"  ZBS(1,0) = {ZBS.get((1,0), 0)}")
print(f"  RBS(1,0) = {RBS.get((1,0), 0)}")
print()

# Compute R and Z
R_expected = np.zeros_like(theta)
Z_expected = np.zeros_like(theta)

# Add symmetric contributions
for m in range(mpol+1):
    n = 0  # axisymmetric
    if (m,n) in RBC:
        R_expected += RBC[(m,n)] * np.cos(m * theta)
    if (m,n) in ZBS:
        Z_expected += ZBS[(m,n)] * np.sin(m * theta)
        
# Add asymmetric contributions  
for m in range(mpol+1):
    n = 0  # axisymmetric
    if (m,n) in RBS:
        R_expected += RBS[(m,n)] * np.sin(m * theta)

print("Expected values at key angles:")
for i, t in enumerate(theta):
    if i % 3 == 0:  # Every 90 degrees
        print(f"  θ={t:.3f} ({t*180/np.pi:.1f}°): R={R_expected[i]:.6f}, Z={Z_expected[i]:.6f}")

print("\nDetailed check at θ=0:")
print(f"  R = RBC(0,0)*cos(0) + RBC(1,0)*cos(0) + RBS(1,0)*sin(0)")
print(f"    = {RBC.get((0,0),0)}*1 + {RBC.get((1,0),0)}*1 + {RBS.get((1,0),0)}*0")
print(f"    = {RBC.get((0,0),0) + RBC.get((1,0),0)}")

print("\nCheck at θ=π/2:")  
print(f"  R = RBC(0,0)*cos(0) + RBC(1,0)*cos(π/2) + RBS(1,0)*sin(π/2)")
print(f"    = {RBC.get((0,0),0)}*1 + {RBC.get((1,0),0)}*0 + {RBS.get((1,0),0)}*1")
print(f"    = {RBC.get((0,0),0) + RBS.get((1,0),0)}")

# What VMEC++ is producing
print("\n=== VMEC++ OUTPUT (from debug) ===")
vmecpp_output = [
    (0.000000, 1.424264, 0.000000),
    (1.570796, 1.001414, 0.424264),
    (3.141593, 0.575736, 0.000000),
    (4.712389, 0.631869, -0.212132),
]

for t, r, z in vmecpp_output:
    i = int(round(t / (2*np.pi) * ntheta))
    if i < len(R_expected):
        r_err = r - R_expected[i]
        z_err = z - Z_expected[i]
        print(f"  θ={t:.3f}: R={r:.6f} (error={r_err:+.6f}), Z={z:.6f} (error={z_err:+.6f})")

# Analysis
print("\n=== ANALYSIS ===")
print("The VMEC++ values suggest:")
print("1. At θ=0: R=1.424264 instead of 1.3")
print("   This is 1.3 * 1.095... ≈ 1.3 * sqrt(1.2)")
print("   Suggests wrong normalization factor")
print("2. Pattern looks like symmetric contribution is scaled incorrectly")