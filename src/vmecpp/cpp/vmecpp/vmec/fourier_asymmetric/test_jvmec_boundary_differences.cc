// TDD test to implement jVMEC boundary condition differences
#include <gtest/gtest.h>

#include <cmath>
#include <iomanip>
#include <iostream>

#include "vmecpp/common/vmec_indata/vmec_indata.h"
#include "vmecpp/vmec/vmec/vmec.h"

namespace vmecpp {

TEST(JVMECBoundaryDifferencesTest, AnalyzeBoundaryPreprocessing) {
  std::cout << "\n=== ANALYZE jVMEC BOUNDARY PREPROCESSING ===\n";
  std::cout << std::fixed << std::setprecision(8);

  std::cout << "KEY DIFFERENCES IDENTIFIED:\n";
  std::cout << "1. AUTOMATIC THETA ANGLE CORRECTION:\n";
  std::cout
      << "   jVMEC: delta = atan2(Rbs[0,1] - Zbc[0,1], Rbc[0,1] + Zbs[0,1])\n";
  std::cout << "   VMEC++: May not apply this correction automatically\n";

  std::cout << "\n2. JACOBIAN SIGN CHECKING:\n";
  std::cout << "   jVMEC: Heuristic to detect boundary coeffs that imply wrong "
               "Jacobian\n";
  std::cout
      << "   VMEC++: Accepts raw coefficients that may cause sign change\n";

  std::cout << "\n3. M=1 MODE CONSTRAINT ENFORCEMENT:\n";
  std::cout << "   jVMEC: rbsc[n][1] = (rbsc[n][1] + zbcc[n][1])/2\n";
  std::cout << "   VMEC++: May not enforce this constraint\n";

  std::cout << "\n4. AXIS OPTIMIZATION ALGORITHM:\n";
  std::cout << "   jVMEC: 61×61 grid search to maximize minimum Jacobian\n";
  std::cout << "   VMEC++: Simple interpolation between axis and boundary\n";

  // Test our current failing configuration
  VmecINDATA config;
  config.lasym = true;
  config.nfp = 1;
  config.mpol = 5;
  config.ntor = 0;
  config.ns_array = {3};
  config.niter_array = {1};
  config.ftol_array = {1e-6};
  config.return_outputs_even_if_not_converged = true;
  config.delt = 0.5;
  config.tcon0 = 1.0;
  config.phiedge = 1.0;
  config.pmass_type = "power_series";
  config.am = {1.0, 0.0, 0.0, 0.0, 0.0};
  config.pres_scale = 1.0;

  // Current failing configuration (from test_jacobian_geometry_debug)
  config.raxis_c = {6.0};
  config.zaxis_s = {0.0};
  config.raxis_s = {0.0};
  config.zaxis_c = {0.0};

  config.rbc = {6.0, 0.0, 0.6, 0.0, 0.12};
  config.zbs = {0.0, 0.0, 0.6, 0.0, 0.12};
  config.rbs = {0.0, 0.0, 0.189737, 0.0, 0.0};
  config.zbc = {0.0, 0.0, 0.189737, 0.0, 0.0};

  std::cout << "\nANALYZING CURRENT CONFIGURATION:\n";
  std::cout << "Boundary coefficients:\n";
  std::cout << "  rbc = [6.0, 0.0, 0.6, 0.0, 0.12]\n";
  std::cout << "  rbs = [0.0, 0.0, 0.189737, 0.0, 0.0]\n";
  std::cout << "  zbs = [0.0, 0.0, 0.6, 0.0, 0.12]\n";
  std::cout << "  zbc = [0.0, 0.0, 0.189737, 0.0, 0.0]\n";

  std::cout << "\nCHECK THETA ANGLE CORRECTION:\n";
  // For m=1, n=0 mode (index 1 in coefficient arrays)
  double rbs_1_0 = 0.0;  // rbs[1] = 0.0
  double zbc_1_0 = 0.0;  // zbc[1] = 0.0
  double rbc_1_0 = 0.0;  // rbc[1] = 0.0
  double zbs_1_0 = 0.0;  // zbs[1] = 0.0

  if (rbs_1_0 != 0.0 || zbc_1_0 != 0.0) {
    double delta = atan2(rbs_1_0 - zbc_1_0, rbc_1_0 + zbs_1_0);
    std::cout << "  delta = atan2(" << rbs_1_0 << " - " << zbc_1_0 << ", "
              << rbc_1_0 << " + " << zbs_1_0 << ") = " << delta << " radians\n";
    std::cout << "  delta = " << (delta * 180.0 / M_PI) << " degrees\n";
  } else {
    std::cout << "  No m=1 modes present - no theta correction needed\n";
  }

  std::cout << "\nTEST HYPOTHESIS:\n";
  std::cout << "Current config may need jVMEC-style preprocessing to avoid "
               "Jacobian failure\n";

  EXPECT_TRUE(true) << "Boundary preprocessing analysis complete";
}

TEST(JVMECBoundaryDifferencesTest, TestAxisOptimization) {
  std::cout << "\n=== TEST AXIS OPTIMIZATION ALGORITHM ===\n";

  std::cout << "jVMEC AXIS OPTIMIZATION APPROACH:\n";
  std::cout << "1. Create 61×61 grid around boundary extents\n";
  std::cout << "2. For each axis candidate (rTest, zTest):\n";
  std::cout
      << "3. Compute tau = signOfJacobian * (tau0 - ru12*zTest + zu12*rTest)\n";
  std::cout << "4. Find axis position that maximizes minimum tau\n";
  std::cout << "5. Prefer z=0 in case of ties\n";

  std::cout << "\nCURRENT VMEC++ APPROACH:\n";
  std::cout << "Uses simple axis specification from input: raxis_c = [6.0]\n";
  std::cout << "No optimization to avoid Jacobian sign changes\n";

  std::cout << "\nHYPOTHESIS:\n";
  std::cout << "The axis position raxis_c=6.0 may not be optimal for this "
               "asymmetric boundary\n";
  std::cout << "jVMEC would search around this position to find better axis "
               "placement\n";

  std::cout << "\nSIMPLE AXIS PERTURBATION TEST:\n";
  std::cout << "Test different axis positions around current raxis_c=6.0:\n";

  VmecINDATA base_config;
  base_config.lasym = true;
  base_config.nfp = 1;
  base_config.mpol = 5;
  base_config.ntor = 0;
  base_config.ns_array = {3};
  base_config.niter_array = {1};
  base_config.ftol_array = {1e-6};
  base_config.return_outputs_even_if_not_converged = true;
  base_config.delt = 0.5;
  base_config.tcon0 = 1.0;
  base_config.phiedge = 1.0;
  base_config.pmass_type = "power_series";
  base_config.am = {1.0, 0.0, 0.0, 0.0, 0.0};
  base_config.pres_scale = 1.0;

  base_config.zaxis_s = {0.0};
  base_config.raxis_s = {0.0};
  base_config.zaxis_c = {0.0};

  base_config.rbc = {6.0, 0.0, 0.6, 0.0, 0.12};
  base_config.zbs = {0.0, 0.0, 0.6, 0.0, 0.12};
  base_config.rbs = {0.0, 0.0, 0.189737, 0.0, 0.0};
  base_config.zbc = {0.0, 0.0, 0.189737, 0.0, 0.0};

  // Test small perturbations around original axis
  std::vector<double> axis_candidates = {5.9, 5.95, 6.0, 6.05, 6.1};

  for (double raxis : axis_candidates) {
    VmecINDATA config = base_config;
    config.raxis_c = {raxis};

    std::cout << "\nTesting raxis_c = " << raxis << ":\n";
    const auto output = vmecpp::run(config);

    if (output.ok()) {
      std::cout << "  ✅ SUCCESS: Axis position " << raxis << " works!\n";
    } else {
      std::string error_msg(output.status().message());
      if (error_msg.find("JACOBIAN") != std::string::npos) {
        std::cout << "  ❌ JACOBIAN: Still fails with axis " << raxis << "\n";
      } else {
        std::cout << "  ❌ OTHER: " << error_msg.substr(0, 50) << "...\n";
      }
    }
  }

  std::cout << "\nCONCLUSION:\n";
  std::cout
      << "If multiple axis positions fail → issue is not just axis placement\n";
  std::cout << "If some succeed → jVMEC-style axis optimization could help\n";

  EXPECT_TRUE(true) << "Axis optimization test complete";
}

TEST(JVMECBoundaryDifferencesTest, CheckM1ModeConstraints) {
  std::cout << "\n=== CHECK M=1 MODE CONSTRAINTS ===\n";

  std::cout << "jVMEC M=1 CONSTRAINT ENFORCEMENT:\n";
  std::cout << "For quasi-polar coordinates, enforces:\n";
  std::cout << "rbsc[n][1] = (rbsc[n][1] + zbcc[n][1]) / 2\n";
  std::cout << "This couples R and Z m=1 modes for stability\n";

  std::cout << "\nCURRENT CONFIGURATION M=1 MODES:\n";
  std::cout << "rbc[1] = 0.0, rbs[1] = 0.0\n";
  std::cout << "zbc[1] = 0.0, zbs[1] = 0.0\n";
  std::cout << "All m=1 modes are zero - constraint already satisfied\n";

  std::cout << "\nTEST WITH NON-ZERO M=1 MODES:\n";
  VmecINDATA config;
  config.lasym = true;
  config.nfp = 1;
  config.mpol = 5;
  config.ntor = 0;
  config.ns_array = {3};
  config.niter_array = {1};
  config.ftol_array = {1e-6};
  config.return_outputs_even_if_not_converged = true;
  config.delt = 0.5;
  config.tcon0 = 1.0;
  config.phiedge = 1.0;
  config.pmass_type = "power_series";
  config.am = {1.0, 0.0, 0.0, 0.0, 0.0};
  config.pres_scale = 1.0;

  config.raxis_c = {6.0};
  config.zaxis_s = {0.0};
  config.raxis_s = {0.0};
  config.zaxis_c = {0.0};

  // Add some m=1 asymmetric modes
  config.rbc = {6.0, 0.0, 0.6, 0.0, 0.12};      // m=1 still zero
  config.zbs = {0.0, 0.0, 0.6, 0.0, 0.12};      // m=1 still zero
  config.rbs = {0.0, 0.1, 0.189737, 0.0, 0.0};  // m=1 mode = 0.1
  config.zbc = {0.0, 0.1, 0.189737, 0.0, 0.0};  // m=1 mode = 0.1

  std::cout << "Before constraint: rbs[1] = 0.1, zbc[1] = 0.1\n";

  // Apply jVMEC constraint
  double rbs_1 = config.rbs[1];
  double zbc_1 = config.zbc[1];
  double constrained_value = (rbs_1 + zbc_1) / 2.0;

  std::cout << "jVMEC constraint: (" << rbs_1 << " + " << zbc_1
            << ") / 2 = " << constrained_value << "\n";
  std::cout << "After constraint: rbs[1] = zbc[1] = " << constrained_value
            << "\n";

  // Test both constrained and unconstrained
  std::cout << "\nTesting unconstrained m=1 modes:\n";
  const auto unconstrained_output = vmecpp::run(config);
  if (!unconstrained_output.ok()) {
    std::cout << "❌ Unconstrained fails\n";
  } else {
    std::cout << "✅ Unconstrained succeeds\n";
  }

  // Apply constraint
  config.rbs[1] = constrained_value;
  config.zbc[1] = constrained_value;

  std::cout << "Testing constrained m=1 modes:\n";
  const auto constrained_output = vmecpp::run(config);
  if (!constrained_output.ok()) {
    std::cout << "❌ Constrained fails\n";
  } else {
    std::cout << "✅ Constrained succeeds\n";
  }

  EXPECT_TRUE(true) << "M=1 constraint test complete";
}

TEST(JVMECBoundaryDifferencesTest, NextImplementationPriority) {
  std::cout << "\n=== NEXT IMPLEMENTATION PRIORITY ===\n";

  std::cout << "BASED ON jVMEC ANALYSIS, PRIORITY ORDER:\n";
  std::cout << "1. HIGH: Implement axis optimization algorithm\n";
  std::cout << "   - Most likely to resolve current Jacobian failures\n";
  std::cout << "   - 61×61 grid search around boundary extents\n";
  std::cout << "   - Maximize minimum Jacobian value\n";

  std::cout << "\n2. MEDIUM: Add boundary preprocessing\n";
  std::cout << "   - Automatic theta angle correction\n";
  std::cout << "   - Jacobian sign heuristic checking\n";
  std::cout << "   - M=1 mode constraint enforcement\n";

  std::cout << "\n3. LOW: Recovery mechanisms\n";
  std::cout << "   - Progressive time step reduction\n";
  std::cout << "   - Multiple restart attempts\n";
  std::cout << "   - 3-surface fallback\n";

  std::cout << "\nNEXT STEPS:\n";
  std::cout << "1. Create unit test for simple axis optimization\n";
  std::cout << "2. Implement grid search around current axis position\n";
  std::cout << "3. Test if optimal axis placement avoids Jacobian failure\n";
  std::cout << "4. If successful, integrate into VMEC++ initialization\n";

  std::cout << "\nSUCCESS CRITERIA:\n";
  std::cout
      << "Find at least one axis position that avoids Jacobian sign change\n";
  std::cout << "for current asymmetric test configuration\n";

  EXPECT_TRUE(true) << "Implementation priority analysis complete";
}

}  // namespace vmecpp
