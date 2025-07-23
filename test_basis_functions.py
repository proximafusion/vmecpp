#!/usr/bin/env python3
"""Test basis function values"""

import numpy as np

# At theta=0, all cos(m*theta) = 1, all sin(m*theta) = 0
# With normalization:
# - cosmu[m=0] = cos(0) * mscale[0] = 1 * 1 = 1
# - cosmu[m=1] = cos(0) * mscale[1] = 1 * sqrt(2) = 1.414...
# - sinmu[m=1] = sin(0) * mscale[1] = 0 * sqrt(2) = 0

print("Expected basis function values at theta=0:")
print(f"  cosmu[m=0] = 1.0")
print(f"  cosmu[m=1] = {np.sqrt(2):.6f}")
print(f"  sinmu[m=1] = 0.0")
print()

# For R at theta=0 with our coefficients:
# Symmetric: R = RBC(0,0)*cosmu[0] + RBC(1,0)*cosmu[1]
#              = 1.0 * 1.0 + 0.3 * sqrt(2)
#              = 1.0 + 0.424264...
#              = 1.424264

R_at_0_with_normalization = 1.0 * 1.0 + 0.3 * np.sqrt(2)
print(f"R at theta=0 WITH normalization in basis:")
print(f"  = 1.0 * 1.0 + 0.3 * {np.sqrt(2):.6f}")
print(f"  = {R_at_0_with_normalization:.6f}")
print()

# But we expect R = 1.3, which means we should NOT have normalization in the result
print("This matches VMEC++ output of 1.424264!")
print("But we want 1.3, which means the coefficients should not multiply")
print("with normalized basis functions for the physical values.")
print()

print("The issue: VMEC++ is using normalized basis functions for computing")
print("physical values, but the normalization should only be used internally")
print("for the transforms, not in the final physical values.")