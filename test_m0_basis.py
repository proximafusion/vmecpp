#!/usr/bin/env python3
"""Debug m=0 basis function issue"""

import numpy as np

# For m=0, mscale[0] = 1.0 (not sqrt(2))
# So cosmu[m=0] = cos(0*theta) * 1.0 = 1.0

# But if somehow m=0 is getting mscale = sqrt(2), then:
# cosmu[m=0] = cos(0*theta) * sqrt(2) = sqrt(2)

# This would explain why EVERYTHING is scaled by sqrt(2)

print("If m=0 basis has wrong normalization:")
print(f"  cosmu[m=0] = sqrt(2) = {np.sqrt(2):.6f}")
print(f"  Then R00 * cosmu[0] = 1.0 * {np.sqrt(2):.6f} = {np.sqrt(2):.6f}")
print()

# This would make ALL values sqrt(2) times larger
print("Checking the pattern:")
print(f"  Expected R(0) = 1.3")
print(f"  Actual R(0) = 1.424264")
print(f"  Ratio = {1.424264 / 1.3:.6f} ≈ {np.sqrt(2) / (1.3/1.3):.6f}")
print()

# Actually, let me be more precise
# R(0) = RBC00 * cosmu[0] + RBC10 * cosmu[1] * cos(0)
#      = 1.0 * cosmu[0] + 0.3 * cosmu[1]

# If cosmu includes mscale correctly:
# cosmu[0] = 1.0
# cosmu[1] = sqrt(2)
# Then R(0) = 1.0 * 1.0 + 0.3 * sqrt(2) = 1.0 + 0.424... = 1.424...

print("Wait, this is exactly what we get!")
print("The issue is that cosmu[1] = sqrt(2) when it should be 1.0 at theta=0")
print()
print("The problem: For physical values, we want cos(m*theta), not cos(m*theta)*mscale[m]")
print("The mscale normalization is for internal transform consistency, not physical output.")