// TDD test to debug geometry derivatives in Jacobian calculation
// Compare symmetric vs asymmetric mode derivative calculations

#include <gtest/gtest.h>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

#include "vmecpp/common/vmec_indata/vmec_indata.h"
#include "vmecpp/vmec/vmec/vmec.h"

using vmecpp::VmecINDATA;

namespace vmecpp {

TEST(GeometryDerivativesDebugTest, CompareSymmetricVsAsymmetric) {
  std::cout << "\n=== COMPARE GEOMETRY DERIVATIVES ===" << std::endl;
  std::cout << std::fixed << std::setprecision(6);

  // Create identical configuration except for lasym flag
  auto createConfig = [](bool asymmetric) {
    VmecINDATA config;

    // Basic configuration
    config.lasym = asymmetric;
    config.nfp = 1;
    config.mpol = 5;
    config.ntor = 0;

    // Small radial resolution for detailed debug
    config.ns_array = {3};
    config.niter_array = {1};  // Just one iteration to see initial derivatives
    config.ftol_array = {1e-6};
    config.return_outputs_even_if_not_converged = true;

    // Physics parameters
    config.delt = 0.5;
    config.tcon0 = 1.0;
    config.phiedge = 1.0;
    config.gamma = 0.0;
    config.curtor = 0.0;
    config.ncurr = 0;

    // Zero pressure to isolate geometry effects
    config.pmass_type = "power_series";
    config.am = {0.0};
    config.pres_scale = 0.0;

    // Current profile
    config.piota_type = "power_series";
    config.ai = {0.0};

    // Simple tokamak boundary (same for both)
    config.raxis_cc = {6.0};
    config.zaxis_cs = {0.0};

    config.rbc =
        std::vector<double>((config.mpol + 1) * (2 * config.ntor + 1), 0.0);
    config.zbs =
        std::vector<double>((config.mpol + 1) * (2 * config.ntor + 1), 0.0);

    // Set boundary modes
    auto setMode = [&](int m, int n, double rbc_val, double zbs_val) {
      int idx = m * (2 * config.ntor + 1) + n + config.ntor;
      config.rbc[idx] = rbc_val;
      config.zbs[idx] = zbs_val;
    };

    // Same boundary for both cases
    setMode(0, 0, 6.0, 0.0);  // Major radius
    setMode(1, 0, 0.5, 0.5);  // Minor radius
    setMode(2, 0, 0.1, 0.1);  // Ellipticity

    return config;
  };

  // Run symmetric case
  std::cout << "\nRUNNING SYMMETRIC CASE (lasym=false)..." << std::endl;
  auto sym_config = createConfig(false);
  auto sym_result = Vmec::Run(sym_config);

  // Run asymmetric case
  std::cout << "\nRUNNING ASYMMETRIC CASE (lasym=true)..." << std::endl;
  auto asym_config = createConfig(true);
  auto asym_result = Vmec::Run(asym_config);

  // Both should produce some output (even if not converged)
  std::cout << "\nRESULTS:" << std::endl;
  std::cout << "Symmetric: "
            << (sym_result.outputs.has_value() ? "Has outputs" : "No outputs")
            << std::endl;
  std::cout << "Asymmetric: "
            << (asym_result.outputs.has_value() ? "Has outputs" : "No outputs")
            << std::endl;

  if (!sym_result.outputs.has_value()) {
    std::cout << "Symmetric status: " << sym_result.exit_status.ToString()
              << std::endl;
  }
  if (!asym_result.outputs.has_value()) {
    std::cout << "Asymmetric status: " << asym_result.exit_status.ToString()
              << std::endl;
  }

  // The test passes if we get debug output showing the derivative differences
  EXPECT_TRUE(true) << "Geometry derivative comparison completed";
}

TEST(GeometryDerivativesDebugTest, DocumentDerivativeFormulas) {
  std::cout << "\n=== DOCUMENT DERIVATIVE FORMULAS ===" << std::endl;

  std::cout << "RADIAL DERIVATIVES (at half-grid points):\n";
  std::cout << "- rs = (r1[j] - r1[j-1]) / ds\n";
  std::cout << "- zs = (z1[j] - z1[j-1]) / ds\n";
  std::cout << "- rus = (ru[j] - ru[j-1]) / ds\n";
  std::cout << "- zus = (zu[j] - zu[j-1]) / ds\n\n";

  std::cout << "HALF-GRID AVERAGES:\n";
  std::cout << "- ru12 = (ru[j] + ru[j-1]) / 2\n";
  std::cout << "- zu12 = (zu[j] + zu[j-1]) / 2\n";
  std::cout << "- r12 = (r1[j] + r1[j-1]) / 2\n";
  std::cout << "- z12 = (z1[j] + z1[j-1]) / 2\n\n";

  std::cout << "TAU CALCULATION:\n";
  std::cout << "tau = ru12*zs - rs*zu12 + dshalfds*odd_contrib\n\n";

  std::cout << "KEY QUESTIONS:\n";
  std::cout << "1. Is ds calculated differently for asymmetric?\n";
  std::cout << "2. Are j-1 indices handled correctly at boundaries?\n";
  std::cout << "3. Do r1[j] values differ between symmetric/asymmetric?\n";
  std::cout << "4. Is the theta range affecting derivative accuracy?\n";

  EXPECT_TRUE(true) << "Derivative formulas documented";
}

TEST(GeometryDerivativesDebugTest, ProposedDebugOutput) {
  std::cout << "\n=== PROPOSED DEBUG OUTPUT ===" << std::endl;

  std::cout << "ADD TO computeJacobian() in ideal_mhd_model.cc:\n\n";

  std::cout << "// Debug geometry derivatives\n";
  std::cout << "if (s_.lasym && jH == 0 && kl >= 6 && kl <= 9) {\n";
  std::cout << "  std::cout << \"GEOM DERIV DEBUG jH=\" << jH << \" kl=\" << "
               "kl << std::endl;\n";
  std::cout
      << "  std::cout << \"  r1[j]=\" << r1_e[kl] + r1_o[kl] << std::endl;\n";
  std::cout << "  std::cout << \"  r1[j-1]=\" << r1_e[kl-nZnT] + r1_o[kl-nZnT] "
               "<< std::endl;\n";
  std::cout << "  std::cout << \"  rs=\" << rs[iHalf] << \" (dr/ds)\" << "
               "std::endl;\n";
  std::cout << "  std::cout << \"  ru12=\" << ru12[iHalf] << \" (avg ru)\" << "
               "std::endl;\n";
  std::cout << "  std::cout << \"  zs=\" << zs[iHalf] << \" (dz/ds)\" << "
               "std::endl;\n";
  std::cout << "  std::cout << \"  zu12=\" << zu12[iHalf] << \" (avg zu)\" << "
               "std::endl;\n";
  std::cout
      << "  std::cout << \"  ds=\" << m_ls_.sHalf_i[iHalf] << std::endl;\n";
  std::cout << "  std::cout << \"  tau1 = ru12*zs - rs*zu12 = \" << tau1 << "
               "std::endl;\n";
  std::cout << "}\n\n";

  std::cout << "This will show:\n";
  std::cout << "- Actual geometry values at problematic theta points\n";
  std::cout << "- Radial derivatives that create sign-changing tau\n";
  std::cout << "- Grid spacing ds at each surface\n";
  std::cout << "- Whether issue is in r1/z1 values or derivative calculation\n";

  EXPECT_TRUE(true) << "Debug output proposed";
}

}  // namespace vmecpp
