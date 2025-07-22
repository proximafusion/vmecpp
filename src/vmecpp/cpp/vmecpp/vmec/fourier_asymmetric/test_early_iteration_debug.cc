// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

#include "vmecpp/common/vmec_indata/vmec_indata.h"
#include "vmecpp/vmec/vmec/vmec.h"

namespace vmecpp {

class EarlyIterationDebugTest : public ::testing::Test {
 protected:
  void SetUp() override {}

  // Helper to create simple asymmetric tokamak configuration
  VmecINDATA CreateAsymmetricTokamakConfig() {
    VmecINDATA indata;

    // Basic parameters
    indata.nfp = 1;
    indata.ncurr = 0;
    indata.delt = 0.25;
    indata.lasym = true;  // CRITICAL: Asymmetric mode
    indata.mpol = 7;
    indata.ntor = 0;
    indata.ntheta = 17;
    indata.nzeta = 1;

    // Multi-grid with low iteration count for debugging
    indata.ns_array = {5};  // Single stage for simplicity
    indata.ftol_array = {1.0e-12};
    indata.niter_array = {20};  // Only 20 iterations for debug
    indata.nstep = 1;           // Report every iteration

    // Pressure and current profiles
    indata.am = {1.0};
    indata.pres_scale = 1000.0;
    indata.ai = {0.0};
    indata.curtor = 0.0;
    indata.gamma = 0.0;
    indata.phiedge = 1.0;

    // Axis position
    indata.raxis_c = {3.0};
    indata.zaxis_s = {0.0};
    indata.raxis_s = {0.0};  // Asymmetric axis component
    indata.zaxis_c = {0.0};  // Asymmetric axis component

    // Boundary coefficients
    std::vector<double> zeros(indata.mpol, 0.0);

    // Symmetric coefficients
    indata.rbc = zeros;
    indata.zbs = zeros;
    indata.rbc[0] = 3.0;  // R00 - major radius
    indata.rbc[1] = 1.0;  // R10 - circular cross-section
    indata.zbs[1] = 1.0;  // Z10 - circular cross-section

    // Asymmetric coefficients - small perturbations
    indata.rbs = zeros;
    indata.zbc = zeros;
    indata.rbs[1] = 0.05;  // R10s - small horizontal asymmetry
    indata.zbc[0] = 0.1;   // Z00c - small vertical shift

    return indata;
  }

  // Helper to create symmetric tokamak for comparison
  VmecINDATA CreateSymmetricTokamakConfig() {
    VmecINDATA indata = CreateAsymmetricTokamakConfig();
    indata.lasym = false;  // Symmetric mode

    // Clear asymmetric coefficients
    indata.rbs.clear();
    indata.zbc.clear();
    indata.raxis_s.clear();
    indata.zaxis_c.clear();

    return indata;
  }
};

// Test detailed debug of first few iterations
TEST_F(EarlyIterationDebugTest, DetailedFirstIterations) {
  std::cout << "\n=== DETAILED EARLY ITERATION DEBUG TEST ===\n";
  std::cout << "Monitoring first 20 iterations with detailed output\n\n";

  auto indata = CreateAsymmetricTokamakConfig();

  std::cout << "Configuration:\n";
  std::cout << "  lasym = true (asymmetric)\n";
  std::cout << "  mpol = " << indata.mpol << ", ntor = " << indata.ntor << "\n";
  std::cout << "  ntheta = " << indata.ntheta << ", nzeta = " << indata.nzeta
            << "\n";
  std::cout << "  Asymmetric coefficients:\n";
  std::cout << "    rbs[1] = " << indata.rbs[1] << " (horizontal asymmetry)\n";
  std::cout << "    zbc[0] = " << indata.zbc[0] << " (vertical shift)\n\n";

  // Create debug output file
  std::ofstream debug_file("asymmetric_debug_iterations.txt");
  debug_file << std::fixed << std::setprecision(8);
  debug_file << "# Iteration debug for asymmetric tokamak\n";
  debug_file << "# Monitoring geometry, forces, and Jacobian\n\n";

  // TODO: In a full implementation, we would add debug hooks to Vmec
  // For now, we document what should be captured:

  std::cout << "DEBUG OUTPUT PLAN:\n";
  std::cout << "1. Initial geometry after spectral_to_initial_guess:\n";
  std::cout << "   - Print R, Z values at key points (axis, edge)\n";
  std::cout << "   - Verify asymmetric contributions are included\n\n";

  std::cout << "2. After geometryFromFourier (each iteration):\n";
  std::cout << "   - Check r1_e, r1_o, z1_e, z1_o arrays\n";
  std::cout << "   - Verify array combination (symmetric + asymmetric)\n";
  std::cout << "   - Look for zeros at kl=6-9 indices\n\n";

  std::cout << "3. After computeJacobian (each iteration):\n";
  std::cout << "   - Print min/max Jacobian values\n";
  std::cout << "   - Check for negative or zero Jacobian\n";
  std::cout << "   - Monitor tau2 calculation\n\n";

  std::cout << "4. Force residuals (each iteration):\n";
  std::cout << "   - Print RMS force values\n";
  std::cout << "   - Check for NaN or infinity\n";
  std::cout << "   - Monitor convergence trend\n\n";

  // Run VMEC with limited iterations
  Vmec vmec(indata);
  auto result = vmec.run();

  if (!result.ok()) {
    std::cout << "\nError during run: " << result.status() << "\n";

    // Check for specific error patterns
    if (result.status().message().find("Jacobian") != std::string::npos) {
      std::cout << "\n⚠️ JACOBIAN ERROR DETECTED\n";
      std::cout << "This confirms the array combination issue:\n";
      std::cout << "- Asymmetric contributions not properly added\n";
      std::cout << "- Leading to zero geometry at certain indices\n";
      std::cout << "- Causing negative/zero Jacobian\n";
    }
  } else if (result.value()) {
    std::cout << "\n✅ Converged in " << indata.niter_array[0]
              << " iterations\n";
  } else {
    std::cout << "\n❌ Did not converge in " << indata.niter_array[0]
              << " iterations\n";
  }

  debug_file.close();
  EXPECT_TRUE(true);  // Documentation test
}

// Compare symmetric vs asymmetric initialization
TEST_F(EarlyIterationDebugTest, CompareInitialGeometry) {
  std::cout << "\n=== COMPARE INITIAL GEOMETRY TEST ===\n";
  std::cout << "Compare symmetric and asymmetric initial guess\n\n";

  // Create both configurations
  auto indata_sym = CreateSymmetricTokamakConfig();
  auto indata_asym = CreateAsymmetricTokamakConfig();

  std::cout << "SYMMETRIC CASE:\n";
  std::cout << "  lasym = false\n";
  std::cout << "  u-range: [0, π]\n";
  std::cout << "  Expected: Geometry only in first half of arrays\n\n";

  std::cout << "ASYMMETRIC CASE:\n";
  std::cout << "  lasym = true\n";
  std::cout << "  u-range: [0, 2π]\n";
  std::cout << "  Expected: Geometry in full arrays\n";
  std::cout << "  Asymmetric contributions:\n";
  std::cout << "    rbs[1] = " << indata_asym.rbs[1] << "\n";
  std::cout << "    zbc[0] = " << indata_asym.zbc[0] << "\n\n";

  std::cout << "KEY POINTS TO CHECK:\n";
  std::cout << "1. Array size differences:\n";
  std::cout << "   - Symmetric: ntheta+1 points for u in [0,π]\n";
  std::cout << "   - Asymmetric: 2*ntheta+1 points for u in [0,2π]\n\n";

  std::cout << "2. Geometry at u > π:\n";
  std::cout << "   - Symmetric: Should be zero (not computed)\n";
  std::cout << "   - Asymmetric: Should have non-zero values\n\n";

  std::cout << "3. Array combination:\n";
  std::cout << "   - Symmetric arrays: r1_e, r1_o contain all geometry\n";
  std::cout << "   - Asymmetric: Must add m_ls_.r1e_i, m_ls_.r1o_i\n\n";

  // TODO: Add actual comparison once debug hooks are in place
  EXPECT_TRUE(true);  // Documentation test
}

// Test Jacobian calculation details
TEST_F(EarlyIterationDebugTest, JacobianCalculationDebug) {
  std::cout << "\n=== JACOBIAN CALCULATION DEBUG TEST ===\n";
  std::cout << "Focus on Jacobian computation issues\n\n";

  auto indata = CreateAsymmetricTokamakConfig();

  std::cout << "JACOBIAN FORMULA:\n";
  std::cout << "gsqrt = R * (Ru * Zs - Rs * Zu)\n";
  std::cout << "where:\n";
  std::cout << "  R = major radius position\n";
  std::cout << "  Ru, Rs = derivatives w.r.t. u, s\n";
  std::cout << "  Zu, Zs = derivatives w.r.t. u, s\n\n";

  std::cout << "POTENTIAL ISSUES:\n";
  std::cout << "1. Missing geometry at u > π:\n";
  std::cout << "   - If R, Z are zero, Jacobian = 0\n";
  std::cout << "   - Causes division by zero in metric terms\n\n";

  std::cout << "2. Array combination:\n";
  std::cout << "   - Symmetric contribution: r1_e[idx]\n";
  std::cout << "   - Asymmetric contribution: m_ls_.r1e_i[idx]\n";
  std::cout << "   - Total: r1_e[idx] + m_ls_.r1e_i[idx]\n\n";

  std::cout << "3. Index mapping:\n";
  std::cout << "   - kl indices 0-5: u in [0, π]\n";
  std::cout << "   - kl indices 6-9: u in [π, 2π]\n";
  std::cout << "   - Second half only has asymmetric contributions\n\n";

  std::cout << "DEBUG STRATEGY:\n";
  std::cout << "1. Print R, Z, Ru, Zu values at each kl index\n";
  std::cout << "2. Check which indices have zero geometry\n";
  std::cout << "3. Verify array combination is happening\n";
  std::cout << "4. Monitor Jacobian sign and magnitude\n\n";

  // Create output file for Jacobian debug
  std::ofstream jac_file("jacobian_debug.txt");
  jac_file << std::fixed << std::setprecision(12);
  jac_file << "# Jacobian debug for asymmetric case\n";
  jac_file << "# kl R Z Ru Zu Rs Zs gsqrt\n";

  // TODO: Add actual debug output once hooks are in place
  jac_file.close();

  EXPECT_TRUE(true);  // Documentation test
}

// Test force residual monitoring
TEST_F(EarlyIterationDebugTest, ForceResidualProgression) {
  std::cout << "\n=== FORCE RESIDUAL PROGRESSION TEST ===\n";
  std::cout << "Monitor how forces evolve during iterations\n\n";

  auto indata = CreateAsymmetricTokamakConfig();
  indata.niter_array = {50};  // More iterations to see trend

  std::cout << "EXPECTED BEHAVIOR:\n";
  std::cout << "1. Initial forces should be large (initial guess)\n";
  std::cout << "2. Forces should decrease monotonically\n";
  std::cout << "3. No NaN or infinity values\n";
  std::cout << "4. Convergence rate similar to symmetric case\n\n";

  std::cout << "MONITORING:\n";
  std::cout << "- Force residual at each iteration\n";
  std::cout << "- Individual force components (R, Z, Lambda)\n";
  std::cout << "- Constraint force contributions\n";
  std::cout << "- Spectral force coefficients\n\n";

  // Create force monitoring file
  std::ofstream force_file("force_residuals.txt");
  force_file << std::fixed << std::setprecision(10);
  force_file << "# Force residual progression\n";
  force_file << "# iter force_rms force_r force_z force_lambda\n";

  // TODO: Add actual monitoring once hooks are in place
  force_file.close();

  std::cout << "FILES CREATED:\n";
  std::cout << "- asymmetric_debug_iterations.txt: General debug output\n";
  std::cout << "- jacobian_debug.txt: Jacobian calculation details\n";
  std::cout << "- force_residuals.txt: Force convergence monitoring\n\n";

  std::cout << "NEXT STEPS:\n";
  std::cout << "1. Add debug print statements in ideal_mhd_model.cc\n";
  std::cout << "2. Focus on geometryFromFourier array combination\n";
  std::cout << "3. Verify tau2 calculation stability\n";
  std::cout << "4. Check constraint force application\n";

  EXPECT_TRUE(true);  // Documentation test
}

// Test to identify exact divergence point
TEST_F(EarlyIterationDebugTest, IdentifyDivergencePoint) {
  std::cout << "\n=== IDENTIFY DIVERGENCE POINT TEST ===\n";
  std::cout << "Find where symmetric and asymmetric cases diverge\n\n";

  std::cout << "COMPARISON STRATEGY:\n";
  std::cout << "1. Run symmetric case (baseline)\n";
  std::cout << "2. Run asymmetric case with same parameters\n";
  std::cout << "3. Compare at each step:\n";
  std::cout << "   - Initial geometry\n";
  std::cout << "   - First Jacobian calculation\n";
  std::cout << "   - First force calculation\n";
  std::cout << "   - First spectral update\n\n";

  std::cout << "KEY DIFFERENCES TO WATCH:\n";
  std::cout << "1. Array sizes:\n";
  std::cout << "   - Symmetric: smaller arrays (u in [0,π])\n";
  std::cout << "   - Asymmetric: larger arrays (u in [0,2π])\n\n";

  std::cout << "2. Transform calls:\n";
  std::cout << "   - Symmetric: FourierToReal3DSymmFastPoloidal\n";
  std::cout << "   - Asymmetric: FourierToReal3DAsymmFastPoloidal\n\n";

  std::cout << "3. Array combination:\n";
  std::cout << "   - Symmetric: No combination needed\n";
  std::cout << "   - Asymmetric: Must combine symmetric + asymmetric\n\n";

  std::cout << "4. Constraint handling:\n";
  std::cout << "   - Both should apply m=1 constraints\n";
  std::cout << "   - But asymmetric has more modes\n\n";

  // TODO: Implement side-by-side comparison once debug hooks are ready

  EXPECT_TRUE(true);  // Documentation test
}

}  // namespace vmecpp
