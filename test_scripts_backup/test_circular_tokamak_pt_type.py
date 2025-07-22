#!/usr/bin/env python3
"""
Test running VMEC++ with PT_TYPE fields for a circular tokamak
using the direct Python interface
"""

import vmecpp
import numpy as np

print("Creating circular tokamak configuration with PT_TYPE fields...")

# Create input
vmec_input = vmecpp.VmecInput(
    # Grid parameters
    lasym=False,  # Axisymmetric
    nfp=1,        # Field period = 1 for tokamak
    mpol=5,       # Poloidal resolution
    ntor=0,       # No toroidal modes for axisymmetric
    ntheta=0,     # Auto
    nzeta=0,      # Auto
    
    # Multi-grid
    ns_array=[11, 21, 51],  # 3 grid levels
    ftol_array=[1.0e-6, 1.0e-8, 1.0e-12],
    niter_array=[500, 1000, 2000],
    
    # Equilibrium parameters
    phiedge=5.0,  # Toroidal flux
    ncurr=1,      # Use toroidal current profile
    
    # Pressure profile - moderate beta
    pmass_type="power_series",
    am=[1.0, -1.0, 0.0, 0.0, 0.0],  # Parabolic pressure profile
    am_aux_s=[],
    am_aux_f=[],
    pres_scale=1.0e4,  # Pressure scaling
    gamma=0.0,
    spres_ped=1.0,
    
    # Rotational transform profile
    piota_type="power_series", 
    ai=[0.0, 0.0, 0.0, 0.0, 0.0],  # Will be determined by current
    ai_aux_s=[],
    ai_aux_f=[],
    
    # Current density profile
    pcurr_type="power_series",
    ac=[1.0, -1.0, 0.0, 0.0, 0.0],  # Parabolic current profile
    ac_aux_s=[],
    ac_aux_f=[],
    curtor=1.0e6,  # 1 MA total current
    bloat=1.0,
    
    # ANIMEC PT_TYPE fields - Temperature anisotropy
    bcrit=1.0,  # Critical field
    pt_type="power_series",
    at=[1.0, 0.5, -0.5, 0.0, 0.0],  # TPERP/TPAR profile: 1.0 at axis, 1.5 at edge
    ph_type="power_series", 
    ah=[0.2, -0.2, 0.0, 0.0, 0.0],  # Small hot particle pressure
    
    # Fixed boundary
    lfreeb=False,
    mgrid_file="",
    extcur=[],
    nvacskip=1,
    
    # Numerical parameters
    nstep=200,
    aphi=[],
    delt=0.9,
    tcon0=0.9,
    lforbal=False,
    
    # Initial axis guess - circular tokamak
    raxis_c=[3.0],  # Major radius R0 = 3m (only one element for ntor=0)
    zaxis_s=[0.0],  # Z = 0 on midplane (only one element for ntor=0)
    
    # Boundary shape - circular cross-section tokamak
    # R = R0 + a*cos(theta), Z = a*sin(theta)
    # where R0 = 3.0 m (major radius) and a = 1.0 m (minor radius)
    rbc=[
        (0, 0, 3.0),   # R00 = 3.0 (major radius)
        (1, 0, 1.0),   # R10 = 1.0 (minor radius)
    ],
    zbs=[
        (1, 0, 1.0),   # Z10 = 1.0 (minor radius)
    ],
    rbs=None,  # None for stellarator-symmetric
    zbc=None   # None for stellarator-symmetric
)

print("\nInput parameters:")
print(f"  Major radius R0 = 3.0 m")
print(f"  Minor radius a = 1.0 m") 
print(f"  Aspect ratio = 3.0")
print(f"  Total current = 1.0 MA")
print(f"  Pressure profile: parabolic")
print(f"  Current profile: parabolic")

print("\nPT_TYPE anisotropy parameters:")
print(f"  bcrit = {vmec_input.bcrit}")
print(f"  pt_type = '{vmec_input.pt_type}'")
print(f"  at = {vmec_input.at}")
print(f"  ph_type = '{vmec_input.ph_type}'")
print(f"  ah = {vmec_input.ah}")

print("\nRunning VMEC++...")
try:
    result = vmecpp.run(vmec_input, verbose=True)
    
    print("\n✓ VMEC++ completed successfully!")
    print(f"\nConvergence flag: {result.r00_convergence_flag}")
    print(f"Final force residual: {result.fsql:.2e}")
    
    # Extract some key results
    print("\nEquilibrium properties:")
    print(f"  Volume-averaged beta: {result.betatot:.4f}")
    print(f"  On-axis rotational transform: {result.iota[0]:.4f}")
    print(f"  Edge rotational transform: {result.iota[-1]:.4f}")
    print(f"  Magnetic axis R: {result.raxis_symm[0]:.4f} m")
    print(f"  Aspect ratio: {result.aspect:.4f}")
    
    # Check if PT_TYPE fields were preserved
    if hasattr(result, 'input'):
        print("\nPT_TYPE fields preserved in output:")
        print(f"  bcrit = {result.input.bcrit}")
        print(f"  pt_type = '{result.input.pt_type}'")
        
    print("\n✓ Circular tokamak with PT_TYPE anisotropy computed successfully!")
    
except Exception as e:
    print(f"\nError running VMEC++: {e}")
    import traceback
    traceback.print_exc()