// TDD unit test to isolate array combination corruption issue
#include <gtest/gtest.h>

#include <cmath>
#include <iomanip>
#include <iostream>

#include "vmecpp/common/vmec_indata/vmec_indata.h"
#include "vmecpp/vmec/vmec/vmec.h"

namespace vmecpp {

TEST(ArrayIndexingDebugTest, IsolateCorruptionIssue) {
  std::cout << "\n=== ISOLATE ARRAY CORRUPTION ISSUE ===\n";
  std::cout << std::fixed << std::setprecision(8);

  std::cout << "GOAL: Understand why array combination corrupts geometry\n";
  std::cout << "EVIDENCE:\n";
  std::cout
      << "1. Symmetric transform: rnkcc[0]=10.25, rnkcc[1]=-2.0 → R=8.25 ✓\n";
  std::cout
      << "2. Array combination: r1_e[18]=10.25 + 0.0 → should stay 10.25\n";
  std::cout << "3. Final geometry: r1_e at kl=6 = 18.0 ❌ (corruption!)\n";

  std::cout << "\nHYPOTHESES:\n";
  std::cout << "H1: Different array position accessed in final debug vs "
               "combination\n";
  std::cout
      << "H2: Array overwritten after combination but before final check\n";
  std::cout
      << "H3: Indexing formula differs between combination and final check\n";
  std::cout << "H4: Multiple jF surfaces overwriting same kl position\n";

  // Use identical geometry as in the failing test
  VmecINDATA config;
  config.lasym = true;
  config.nfp = 1;
  config.mpol = 3;
  config.ntor = 0;
  config.ns_array = {3};
  config.niter_array = {1};
  config.ftol_array = {1e-6};
  config.return_outputs_even_if_not_converged = true;
  config.delt = 0.5;
  config.tcon0 = 1.0;
  config.phiedge = 1.0;
  config.pmass_type = "power_series";
  config.am = {0.0};
  config.pres_scale = 0.0;

  // Circular tokamak
  config.rbc = {10.0, 2.0, 0.5};
  config.zbs = {0.0, 2.0, 0.5};
  config.rbs = {0.0, 0.0, 0.0};  // Zero asymmetric coeffs
  config.zbc = {0.0, 0.0, 0.0};

  config.raxis_c = {10.0};
  config.zaxis_s = {0.0};
  config.raxis_s = {0.0};
  config.zaxis_c = {0.0};

  std::cout << "\nRunning with detailed indexing debug...\n";
  const auto output = vmecpp::run(config);

  if (!output.ok()) {
    std::cout << "Status: " << output.status() << std::endl;
    std::string error_msg(output.status().message());
    if (error_msg.find("JACOBIAN") != std::string::npos) {
      std::cout << "✅ Expected Jacobian failure - we're debugging array "
                   "corruption\n";
    }
  }

  std::cout << "\nFOCUS AREAS FOR NEXT ITERATION:\n";
  std::cout << "1. Add array position tracking throughout ideal_mhd_model.cc\n";
  std::cout
      << "2. Check if kl=6 maps to different idx values in different parts\n";
  std::cout << "3. Verify nsMinF1, nZnT values affect indexing consistently\n";
  std::cout
      << "4. Add debug before/after EVERY array operation affecting r1_e\n";

  EXPECT_TRUE(true) << "Array corruption analysis complete";
}

TEST(ArrayIndexingDebugTest, TheoryTestIndexingFormulas) {
  std::cout << "\n=== TEST INDEXING FORMULAS ===\n";

  std::cout << "From debug output:\n";
  std::cout << "- nsMinF1 = 0\n";
  std::cout << "- nsMaxF1 = 3\n";
  std::cout << "- nZnT = 12 (theta points)\n";
  std::cout << "- Array combination: jF=1, kl=6 → idx=18\n";
  std::cout << "- Final check: kl=6 → array_idx = (1-0)*12 + 6 = 18\n";

  std::cout << "\nIndexing should be consistent:\n";
  int nsMinF1 = 0;
  int nZnT = 12;
  int jF = 1;
  int kl = 6;

  int idx_combination = (jF - nsMinF1) * nZnT + kl;
  int array_idx_final = (1 - nsMinF1) * nZnT + kl;

  std::cout << "Array combination idx: " << idx_combination << std::endl;
  std::cout << "Final check array_idx: " << array_idx_final << std::endl;
  std::cout << "Match: "
            << (idx_combination == array_idx_final ? "✅ YES" : "❌ NO")
            << std::endl;

  if (idx_combination == array_idx_final) {
    std::cout << "\n✅ INDEXING FORMULAS CONSISTENT\n";
    std::cout << "Corruption must happen AFTER array combination but BEFORE "
                 "final check\n";
    std::cout
        << "Look for operations that modify r1_e[18] between these points\n";
  } else {
    std::cout << "\n❌ INDEXING MISMATCH FOUND\n";
    std::cout << "Different formulas used in combination vs final check\n";
  }

  std::cout << "\nNEXT DEBUG STRATEGY:\n";
  std::cout << "1. Add debug print RIGHT AFTER combination: r1_e[18] = ?\n";
  std::cout << "2. Add debug print RIGHT BEFORE final check: r1_e[18] = ?\n";
  std::cout << "3. Find ALL operations that modify r1_e between these points\n";
  std::cout
      << "4. Identify which operation changes r1_e[18] from 10.25 to 18.0\n";

  EXPECT_TRUE(true) << "Indexing formula analysis complete";
}

}  // namespace vmecpp
