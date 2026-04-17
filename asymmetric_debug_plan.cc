// Asymmetric Debug Plan - Selective Debug Output Locations
// Compare with jVMEC implementation

// 1. Enable debug for asymmetric transform in ideal_mhd_model.cc
// Around line 1287 in geometryFromFourier:
#ifdef DEBUG_ASYMMETRIC
if (s_.lasym) {
  std::cout << "DEBUG ASYM: Processing asymmetric equilibrium\n";
  std::cout << "  physical_x.rmnsc size=" << physical_x.rmnsc.size() << "\n";
  std::cout << "  physical_x.zmncc size=" << physical_x.zmncc.size() << "\n";
  // Print first few coefficients
  for (int i = 0; i < std::min(5, (int)physical_x.rmnsc.size()); ++i) {
    std::cout << "  rmnsc[" << i << "]=" << physical_x.rmnsc[i] 
              << " zmncc[" << i << "]=" << physical_x.zmncc[i] << "\n";
  }
}
#endif

// 2. Enable debug in computeJacobian for geometry derivatives
// Around line 1900:
#ifdef DEBUG_JACOBIAN
if (s_.lasym && jH == 1 && (kl == 0 || kl == s_.ntheta/2)) {
  std::cout << "DEBUG JAC: jH=" << jH << " kl=" << kl << "\n";
  std::cout << "  ru12=" << ru12[iHalf] << " zu12=" << zu12[iHalf] << "\n";
  std::cout << "  rs=" << rs[iHalf] << " zs=" << zs[iHalf] << "\n";
  std::cout << "  r12=" << r12[iHalf] << "\n";
  std::cout << "  tau=" << tau[iHalf] << "\n";
}
#endif

// 3. Enable debug in constraint force calculation
// Around line 3515:
#ifdef DEBUG_CONSTRAINT_FORCE
if (jF == 1) {  // Only debug first interior surface
  std::cout << "DEBUG CONSTRAINT: Surface jF=" << jF << "\n";
  double local_arNorm = 0.0, local_azNorm = 0.0;
  for (int kl = 0; kl < std::min(4, s_.nZnT); ++kl) {
    int idx_kl = (jF - r_.nsMinF) * s_.nZnT + kl;
    std::cout << "  kl=" << kl << " ruFull=" << ruFull[idx_kl] 
              << " zuFull=" << zuFull[idx_kl] << "\n";
    local_arNorm += ruFull[idx_kl] * ruFull[idx_kl] * s_.wInt[kl % s_.nThetaEff];
    local_azNorm += zuFull[idx_kl] * zuFull[idx_kl] * s_.wInt[kl % s_.nThetaEff];
  }
  std::cout << "  Partial arNorm=" << local_arNorm << " azNorm=" << local_azNorm << "\n";
}
#endif

// 4. Add debug flag to CMakeLists.txt or compile command:
// -DDEBUG_ASYMMETRIC -DDEBUG_JACOBIAN -DDEBUG_CONSTRAINT_FORCE

// 5. Key comparison points with jVMEC:
// - Check if asymmetric Fourier coefficients are properly loaded
// - Verify geometry derivatives (ru, zu) are computed correctly
// - Ensure zuFull array is populated (not all zeros)
// - Compare tau calculation with jVMEC's computeJacobian
// - Check array indexing and bounds

// 6. Focus areas:
// a) The disabled 2D asymmetric transform (line 4176-4190)
// b) Array initialization for asymmetric components
// c) Proper addition of asymmetric contributions to symmetric arrays
// d) Calculation of derivatives for asymmetric mode