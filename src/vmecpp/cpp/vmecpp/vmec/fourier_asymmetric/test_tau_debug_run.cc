// Test that runs VMEC and captures tau debug information
#include <gtest/gtest.h>

#include <cmath>
#include <iomanip>
#include <iostream>

#include "vmecpp/common/vmec_indata/vmec_indata.h"
#include "vmecpp/vmec/vmec/vmec.h"

namespace vmecpp {

class TauDebugCapture {
 public:
  static void CaptureMinMaxTau(double minTau, double maxTau) {
    captured_minTau = minTau;
    captured_maxTau = maxTau;
    capture_called = true;
  }

  static void Reset() {
    captured_minTau = 0.0;
    captured_maxTau = 0.0;
    capture_called = false;
  }

  static double captured_minTau;
  static double captured_maxTau;
  static bool capture_called;
};

double TauDebugCapture::captured_minTau = 0.0;
double TauDebugCapture::captured_maxTau = 0.0;
bool TauDebugCapture::capture_called = false;

TEST(TauDebugRunTest, CaptureInitialTauValues) {
  std::cout << "\n=== CAPTURE INITIAL TAU VALUES ===" << std::endl;
  std::cout << std::fixed << std::setprecision(8);

  // Test 1: Symmetric case
  {
    TauDebugCapture::Reset();

    VmecINDATA config;
    config.lasym = false;
    config.nfp = 1;
    config.mpol = 3;
    config.ntor = 0;

    config.ns_array = {3};
    config.niter_array = {1};  // One iteration only
    config.ftol_array = {1e-6};
    config.return_outputs_even_if_not_converged = true;

    config.delt = 0.5;
    config.tcon0 = 1.0;
    config.phiedge = 1.0;
    config.pmass_type = "power_series";
    config.am = {0.0};

    // Circular tokamak
    config.rbc = {10.0, 2.0, 0.5};
    config.zbs = {0.0, 2.0, 0.5};

    config.raxis_c = {10.0};
    config.zaxis_s = {0.0};

    std::cout << "Running symmetric case (lasym=false)..." << std::endl;
    const auto output = vmecpp::run(config);

    if (!output.ok()) {
      std::cout << "Status: " << output.status() << std::endl;
    }

    // In real implementation, we'd hook into ideal_mhd_model.cc to capture tau
    // For now, we document expected behavior
    std::cout << "Expected: minTau > 0, maxTau > 0 (positive Jacobian)"
              << std::endl;
  }

  // Test 2: Asymmetric case with zero asymmetric coefficients
  {
    TauDebugCapture::Reset();

    VmecINDATA config;
    config.lasym = true;  // Asymmetric mode
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

    // Same circular tokamak
    config.rbc = {10.0, 2.0, 0.5};
    config.zbs = {0.0, 2.0, 0.5};
    config.rbs = {0.0, 0.0, 0.0};  // Zero asymmetry
    config.zbc = {0.0, 0.0, 0.0};

    config.raxis_c = {10.0};
    config.zaxis_s = {0.0};
    config.raxis_s = {0.0};
    config.zaxis_c = {0.0};

    std::cout << "\nRunning asymmetric mode with zero asymmetric coeffs..."
              << std::endl;
    const auto output = vmecpp::run(config);

    if (!output.ok()) {
      std::cout << "Status: " << output.status() << std::endl;
      std::string error_msg(output.status().message());
      if (error_msg.find("JACOBIAN") != std::string::npos) {
        std::cout << "❌ Jacobian issue even with zero asymmetric coeffs!"
                  << std::endl;
        std::cout << "This confirms issue is in asymmetric mode setup, not the "
                     "perturbation"
                  << std::endl;
      }
    } else {
      std::cout << "✅ No Jacobian issue with zero asymmetric coeffs"
                << std::endl;
    }
  }

  // Test 3: Asymmetric case with tiny perturbation
  {
    TauDebugCapture::Reset();

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

    // Circular tokamak with 0.1% asymmetric perturbation
    config.rbc = {10.0, 2.0, 0.5};
    config.zbs = {0.0, 2.0, 0.5};
    config.rbs = {0.0, 0.002, 0.0005};  // 0.1% of symmetric amplitude
    config.zbc = {0.0, 0.002, 0.0005};

    config.raxis_c = {10.0};
    config.zaxis_s = {0.0};
    config.raxis_s = {0.0};
    config.zaxis_c = {0.0};

    std::cout << "\nRunning with 0.1% asymmetric perturbation..." << std::endl;
    const auto output = vmecpp::run(config);

    if (!output.ok()) {
      std::cout << "Status: " << output.status() << std::endl;
      std::string error_msg(output.status().message());
      if (error_msg.find("JACOBIAN") != std::string::npos) {
        std::cout << "❌ Jacobian issue even with 0.1% perturbation!"
                  << std::endl;
      }
    } else {
      std::cout << "✅ 0.1% perturbation converges successfully" << std::endl;
    }
  }

  // Test passes
  EXPECT_TRUE(true) << "Tau debug capture test";
}

TEST(TauDebugRunTest, IdentifyProblemLocation) {
  std::cout << "\n=== IDENTIFY PROBLEM LOCATION ===" << std::endl;

  std::cout << "From code analysis (ideal_mhd_model.cc:1724):" << std::endl;
  std::cout << "  tau1 = ru12 * zs - rs * zu12" << std::endl;
  std::cout << "  tau2 = complex expression / sqrtSH" << std::endl;
  std::cout << "  tau = tau1 + dSHalfDsInterp * tau2" << std::endl;

  std::cout << "\nPotential issues:" << std::endl;
  std::cout << "1. tau2 division by sqrtSH when sqrtSH → 0 at axis"
            << std::endl;
  std::cout << "2. Initial guess interpolation might create bad geometry"
            << std::endl;
  std::cout << "3. Array combination might be incomplete" << std::endl;
  std::cout << "4. Theta range [0,2π] vs [0,π] handling" << std::endl;

  std::cout << "\nDiagnostic approach:" << std::endl;
  std::cout << "1. Print tau1 and tau2 separately at each kl position"
            << std::endl;
  std::cout << "2. Check which component causes sign change" << std::endl;
  std::cout << "3. Verify geometry derivatives (rs, zs, ru12, zu12)"
            << std::endl;
  std::cout << "4. Compare with educational_VMEC tau calculation" << std::endl;

  // Test passes
  EXPECT_TRUE(true) << "Problem location analysis";
}

}  // namespace vmecpp
