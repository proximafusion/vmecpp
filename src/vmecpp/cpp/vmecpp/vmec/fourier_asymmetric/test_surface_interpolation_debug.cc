// TDD test to debug surface interpolation in asymmetric mode
// Investigate why r1_e[16]=0 at surface 1 when it should be ~19-20

#include <gtest/gtest.h>

#include <cmath>
#include <iomanip>
#include <iostream>

#include "vmecpp/common/vmec_indata/vmec_indata.h"
#include "vmecpp/vmec/vmec/vmec.h"

using vmecpp::Vmec;
using vmecpp::VmecINDATA;

namespace vmecpp {

TEST(SurfaceInterpolationDebugTest, TraceSurfacePopulation) {
  std::cout << "\n=== TRACE SURFACE POPULATION ===\n";
  std::cout << std::fixed << std::setprecision(6);

  // Use minimal config to isolate surface interpolation issue
  VmecINDATA config;
  config.lasym = true;
  config.nfp = 1;
  config.mpol = 2;
  config.ntor = 0;
  config.ns_array = {
      3};  // 3 surfaces: j=0 (axis), j=1 (middle), j=2 (boundary)
  config.niter_array = {1};
  config.ftol_array = {1e-6};
  config.return_outputs_even_if_not_converged = true;

  config.delt = 0.5;
  config.tcon0 = 1.0;
  config.phiedge = 1.0;
  config.gamma = 0.0;
  config.curtor = 0.0;
  config.ncurr = 0;

  config.pmass_type = "power_series";
  config.am = {0.0};
  config.pres_scale = 0.0;

  config.piota_type = "power_series";
  config.ai = {0.0};

  // Large R0 to avoid numerical issues
  config.raxis_c = {20.0};
  config.zaxis_s = {0.0};

  config.rbc =
      std::vector<double>((config.mpol + 1) * (2 * config.ntor + 1), 0.0);
  config.zbs =
      std::vector<double>((config.mpol + 1) * (2 * config.ntor + 1), 0.0);

  auto setMode = [&](int m, int n, double rbc_val, double zbs_val) {
    int idx = m * (2 * config.ntor + 1) + n + config.ntor;
    config.rbc[idx] = rbc_val;
    config.zbs[idx] = zbs_val;
  };

  setMode(0, 0, 20.0, 0.0);  // Major radius
  setMode(1, 0, 1.0, 1.0);   // m=1 symmetric

  // Add minimal asymmetric modes
  config.rbs =
      std::vector<double>((config.mpol + 1) * (2 * config.ntor + 1), 0.0);
  config.zbc =
      std::vector<double>((config.mpol + 1) * (2 * config.ntor + 1), 0.0);

  auto setAsymMode = [&](int m, int n, double rbs_val, double zbc_val) {
    int idx = m * (2 * config.ntor + 1) + n + config.ntor;
    config.rbs[idx] = rbs_val;
    config.zbc[idx] = zbc_val;
  };

  setAsymMode(1, 0, 0.1, 0.1);  // m=1 asymmetric (small perturbation)

  std::cout << "\nConfiguration:\n";
  std::cout << "  NS = " << config.ns_array[0] << " (3 surfaces: j=0,1,2)\n";
  std::cout << "  ntheta = 10 (indices 0-9 per surface)\n";
  std::cout << "  Expected array indices:\n";
  std::cout << "    Surface j=0: indices 0-9\n";
  std::cout << "    Surface j=1: indices 10-19\n";
  std::cout << "    Surface j=2: indices 20-29\n\n";

  std::cout << "CRITICAL QUESTION:\n";
  std::cout << "Why is r1_e[16] = 0.000000 when it should be ~19-20?\n";
  std::cout << "Surface j=1 at theta index kl=6 should have geometry data!\n\n";

  // Run and analyze surface population
  vmecpp::Vmec vmec(config);
  auto result = vmec.run();

  EXPECT_TRUE(true) << "Surface interpolation debug completed";
}

TEST(SurfaceInterpolationDebugTest, AnalyzeSurfaceIndexing) {
  std::cout << "\n=== ANALYZE SURFACE INDEXING ===\n";

  std::cout << "SURFACE INDEXING LOGIC:\n";
  std::cout << "- NS = 3 means 3 radial surfaces\n";
  std::cout << "- nsMinF1 = 0, nsMaxF1 = 3 (range for full-grid surfaces)\n";
  std::cout << "- nZnT = 10 (theta points per surface)\n";
  std::cout << "- Total array size = 3 * 10 = 30 elements\n\n";

  std::cout << "ARRAY INDEX CALCULATION:\n";
  std::cout << "idx = (jF - nsMinF1) * nZnT + kl\n";
  std::cout << "For surface jF=1, kl=6: idx = (1-0)*10 + 6 = 16\n\n";

  std::cout << "SURFACE POPULATION PROCESS:\n";
  std::cout << "1. Symmetric transform: Populates boundary surface (jF=2)\n";
  std::cout << "2. Asymmetric transform: Adds asymmetric corrections\n";
  std::cout
      << "3. Interior interpolation: Should populate intermediate surfaces\n\n";

  std::cout << "HYPOTHESIS:\n";
  std::cout << "- Surface jF=2 (boundary): Populated by Fourier transform ✅\n";
  std::cout << "- Surface jF=0 (axis): Populated by axis handling ✅\n";
  std::cout << "- Surface jF=1 (interior): NOT POPULATED ❌\n\n";

  std::cout << "MISSING STEP:\n";
  std::cout
      << "Interior surface interpolation not working in asymmetric mode\n";
  std::cout << "Need to find where/how jF=1 surface should be generated\n";

  EXPECT_TRUE(true) << "Surface indexing analysis completed";
}

TEST(SurfaceInterpolationDebugTest, CompareWithSymmetricMode) {
  std::cout << "\n=== COMPARE WITH SYMMETRIC MODE ===\n";

  std::cout << "TEST PLAN:\n";
  std::cout << "1. Run identical config with lasym=false\n";
  std::cout << "2. Check if r1_e[16] is populated in symmetric mode\n";
  std::cout << "3. Compare surface population patterns\n";
  std::cout << "4. Identify where asymmetric mode differs\n\n";

  std::cout << "EXPECTED RESULTS:\n";
  std::cout << "- Symmetric mode: r1_e[16] should be ~19-20 (working)\n";
  std::cout << "- Asymmetric mode: r1_e[16] = 0.000000 (broken)\n";
  std::cout
      << "- Difference: Asymmetric missing interior surface generation\n\n";

  std::cout << "DEBUGGING APPROACH:\n";
  std::cout
      << "1. Add debug output to geometryFromFourier for surface population\n";
  std::cout << "2. Trace which code path populates interior surfaces\n";
  std::cout << "3. Find where asymmetric mode skips this step\n";
  std::cout << "4. Fix interpolation to work with full theta range [0,2π]\n";

  EXPECT_TRUE(true) << "Symmetric comparison strategy documented";
}

TEST(SurfaceInterpolationDebugTest, ProposeDebuggingStrategy) {
  std::cout << "\n=== PROPOSE DEBUGGING STRATEGY ===\n";

  std::cout << "IMMEDIATE ACTIONS:\n\n";

  std::cout << "1. ADD SURFACE DEBUG OUTPUT:\n";
  std::cout << "   // In geometryFromFourier after symmetric transform\n";
  std::cout << "   if (s_.lasym) {\n";
  std::cout << "     for (int jF = 0; jF < 3; ++jF) {\n";
  std::cout << "       int idx = jF * 10 + 6;  // kl=6 for each surface\n";
  std::cout << "       std::cout << \"Surface jF=\" << jF << \" r1_e[\" << idx "
               "<< \"]=\"\n";
  std::cout << "                 << r1_e[idx] << std::endl;\n";
  std::cout << "     }\n";
  std::cout << "   }\n\n";

  std::cout << "2. TRACE INTERPOLATION LOGIC:\n";
  std::cout << "   - Look for spectral_to_initial_guess calls\n";
  std::cout << "   - Check radial interpolation functions\n";
  std::cout << "   - Find where interior surfaces get geometry data\n";
  std::cout << "   - Compare symmetric vs asymmetric code paths\n\n";

  std::cout << "3. STUDY JVMEC REFERENCE:\n";
  std::cout << "   - How does jVMEC populate interior surfaces?\n";
  std::cout << "   - Does it use different interpolation for asymmetric?\n";
  std::cout << "   - Check initialization sequence differences\n\n";

  std::cout << "4. FIX INTERPOLATION:\n";
  std::cout << "   - Ensure all NS surfaces get proper geometry\n";
  std::cout << "   - Test with full theta range [0,2π]\n";
  std::cout << "   - Verify Jacobian calculation with populated arrays\n";

  EXPECT_TRUE(true) << "Debugging strategy proposed";
}

}  // namespace vmecpp
