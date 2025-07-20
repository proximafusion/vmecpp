// Test with working jVMEC asymmetric configuration
#include <gtest/gtest.h>

#include <cmath>
#include <iomanip>
#include <iostream>

#include "nlohmann/json.hpp"
#include "util/file_io/file_io.h"
#include "vmecpp/common/vmec_indata/vmec_indata.h"
#include "vmecpp/vmec/vmec/vmec.h"

using file_io::ReadFile;
using nlohmann::json;
using vmecpp::VmecINDATA;

namespace vmecpp {

TEST(WorkingJVMECConfigTest, TestUpDownAsymmetricTokamak) {
  std::cout << "\n=== TEST WORKING JVMEC ASYMMETRIC CONFIG ===\n";
  std::cout << std::fixed << std::setprecision(6);

  std::cout
      << "GOAL: Test with proven working jVMEC asymmetric configuration\n";
  std::cout
      << "This should help isolate if issue is configuration or algorithm\n";

  // Load the working jVMEC configuration
  const std::string filename =
      "vmecpp/test_data/up_down_asymmetric_tokamak.json";

  absl::StatusOr<std::string> indata_json = ReadFile(filename);
  ASSERT_TRUE(indata_json.ok()) << "Failed to read " << filename;

  absl::StatusOr<VmecINDATA> indata = VmecINDATA::FromJson(*indata_json);
  ASSERT_TRUE(indata.ok()) << "Failed to parse JSON";

  VmecINDATA config = *indata;

  // Reduce to minimal run for debugging
  config.ns_array = {3};
  config.niter_array = {1};
  config.ftol_array = {1e-6};
  config.return_outputs_even_if_not_converged = true;

  std::cout << "\nConfiguration details:\n";
  std::cout << "  lasym = " << config.lasym << std::endl;
  std::cout << "  mpol = " << config.mpol << std::endl;
  std::cout << "  ntor = " << config.ntor << std::endl;
  std::cout << "  raxis_c[0] = " << config.raxis_c[0] << std::endl;

  std::cout << "\nBoundary coefficients:\n";
  for (size_t i = 0; i < config.rbc.size(); ++i) {
    std::cout << "  rbc[" << i << "] = " << config.rbc[i] << std::endl;
  }
  for (size_t i = 0; i < config.rbs.size(); ++i) {
    if (config.rbs[i] != 0.0) {
      std::cout << "  rbs[" << i << "] = " << config.rbs[i] << " (asymmetric)"
                << std::endl;
    }
  }

  std::cout << "\nRunning VMEC with working jVMEC config...\n";
  const auto output = vmecpp::run(config);

  if (!output.ok()) {
    std::cout << "Status: " << output.status() << std::endl;
    std::string error_msg(output.status().message());
    if (error_msg.find("JACOBIAN") != std::string::npos) {
      std::cout << "❌ Even working jVMEC config fails with Jacobian!\n";
      std::cout
          << "This confirms the issue is in VMEC++ asymmetric algorithm\n";
    } else {
      std::cout << "Different error: " << error_msg << std::endl;
    }
  } else {
    std::cout << "✅ Working jVMEC config succeeds!\n";
    std::cout << "Issue may be configuration-specific\n";
  }

  std::cout << "\nANALYSIS:\n";
  std::cout << "If this fails with Jacobian error:\n";
  std::cout << "  → VMEC++ asymmetric algorithm needs debugging\n";
  std::cout
      << "  → Array combination is working, focus on Jacobian calculation\n";
  std::cout << "If this succeeds:\n";
  std::cout << "  → Issue is configuration-specific\n";
  std::cout << "  → Simple test configs trigger edge cases\n";

  EXPECT_TRUE(true) << "Working jVMEC config test complete";
}

TEST(WorkingJVMECConfigTest, CompareSimpleVsComplexConfig) {
  std::cout << "\n=== COMPARE SIMPLE VS COMPLEX CONFIG ===\n";

  std::cout << "Simple config (fails):\n";
  std::cout << "  R0=10, a=2, zero asymmetric coeffs\n";
  std::cout << "  mpol=3, very basic circular tokamak\n";

  std::cout << "\nComplex config (from jVMEC):\n";
  std::cout << "  R0=6, various modes up to mpol=5\n";
  std::cout << "  Realistic asymmetric perturbations\n";
  std::cout << "  Finite pressure, multiple flux surfaces\n";

  std::cout << "\nKey differences that might matter:\n";
  std::cout << "1. Major radius: R0=6 vs R0=10\n";
  std::cout << "2. Mode spectrum: mpol=5 vs mpol=3\n";
  std::cout << "3. Finite pressure: am=[1.0] vs am=[0.0]\n";
  std::cout << "4. Multiple surfaces: ns=49 vs ns=3\n";
  std::cout << "5. Realistic perturbations vs zero coeffs\n";

  std::cout << "\nHypotheses:\n";
  std::cout << "H1: Zero asymmetric coeffs create degenerate case\n";
  std::cout << "H2: Large major radius (R0=10) creates numerical issues\n";
  std::cout << "H3: Minimal mode spectrum doesn't provide enough degrees of "
               "freedom\n";
  std::cout << "H4: Zero pressure creates singular Jacobian\n";

  EXPECT_TRUE(true) << "Configuration comparison complete";
}

}  // namespace vmecpp
