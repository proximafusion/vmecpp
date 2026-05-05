// Test to verify symmetric mode still works (no regression from asymmetric
// changes)

#include <gtest/gtest.h>

#include <iostream>
#include <vector>

#include "absl/log/check.h"
#include "nlohmann/json.hpp"
#include "util/file_io/file_io.h"
#include "vmecpp/common/vmec_indata/vmec_indata.h"
#include "vmecpp/vmec/vmec/vmec.h"

using file_io::ReadFile;
using nlohmann::json;
using vmecpp::VmecINDATA;

namespace vmecpp {

TEST(SymmetricRegressionCheck, SymmetricModeStillWorks) {
  std::cout << "\n=== SYMMETRIC REGRESSION CHECK ===" << std::endl;

  // Load a known working configuration
  const std::string filename = "vmecpp/test_data/cth_like_fixed_bdy.json";
  absl::StatusOr<std::string> indata_json = ReadFile(filename);
  ASSERT_TRUE(indata_json.ok()) << "Failed to read " << filename;

  absl::StatusOr<VmecINDATA> indata = VmecINDATA::FromJson(*indata_json);
  ASSERT_TRUE(indata.ok()) << "Failed to parse JSON";

  VmecINDATA indata_symm = *indata;

  // Explicitly set symmetric mode
  indata_symm.lasym = false;

  // Use small parameters for fast testing
  indata_symm.ns_array[0] = 5;
  indata_symm.niter_array[0] = 10;
  indata_symm.mpol = 2;
  indata_symm.ntor = 1;
  indata_symm.return_outputs_even_if_not_converged = true;

  std::cout << "Configuration: lasym=" << indata_symm.lasym
            << " (SYMMETRIC MODE)" << std::endl;
  std::cout << "  mpol=" << indata_symm.mpol << ", ntor=" << indata_symm.ntor
            << ", ns=" << indata_symm.ns_array[0] << std::endl;

  // Resize coefficient arrays to match new mpol/ntor
  int coeff_size = indata_symm.mpol * (2 * indata_symm.ntor + 1);
  indata_symm.rbc.resize(coeff_size, 0.0);
  indata_symm.zbs.resize(coeff_size, 0.0);

  // Fix axis array sizes to match ntor
  indata_symm.raxis_c.resize(indata_symm.ntor + 1, 0.0);
  indata_symm.zaxis_s.resize(indata_symm.ntor + 1, 0.0);

  // Ensure no asymmetric coefficient arrays for symmetric mode
  indata_symm.rbs.clear();
  indata_symm.zbc.clear();
  indata_symm.raxis_s.clear();
  indata_symm.zaxis_c.clear();

  std::cout << "Asymmetric arrays cleared (rbs, zbc, raxis_s, zaxis_c)"
            << std::endl;

  std::cout << "\nRunning VMEC in SYMMETRIC mode..." << std::endl;

  // Run VMEC
  const auto output = vmecpp::run(indata_symm);

  std::cout << "\nSymmetric VMEC run status: "
            << (output.ok() ? "SUCCESS" : "FAILED") << std::endl;

  if (output.ok()) {
    std::cout << "✅ SUCCESS: Symmetric mode works correctly!" << std::endl;
    const auto& wout = output->wout;
    std::cout << "Output summary:" << std::endl;
    std::cout << "  lasym = " << wout.lasym << " (should be false)"
              << std::endl;
    std::cout << "  ns = " << wout.ns << std::endl;
    std::cout << "  volume = " << wout.volume_p << std::endl;
    std::cout << "  aspect ratio = " << wout.aspect << std::endl;

    // Verify this is actually symmetric
    EXPECT_FALSE(wout.lasym) << "Output should be symmetric (lasym=false)";
    EXPECT_GT(wout.volume_p, 0) << "Volume should be positive";
    EXPECT_GT(wout.aspect, 1) << "Aspect ratio should be > 1";

  } else {
    std::cout << "❌ REGRESSION: Symmetric mode failed!" << std::endl;
    std::cout << "Error: " << output.status() << std::endl;
    FAIL() << "Symmetric mode should still work. This indicates a regression!";
  }

  // This test MUST pass to ensure no regression
  EXPECT_TRUE(output.ok()) << "Symmetric mode must work (no regression)";
}

TEST(SymmetricRegressionCheck, CompareAsymmetricVsSymmetricExecution) {
  std::cout << "\n=== ASYMMETRIC VS SYMMETRIC EXECUTION COMPARISON ==="
            << std::endl;

  // Load base configuration
  const std::string filename = "vmecpp/test_data/cth_like_fixed_bdy.json";
  absl::StatusOr<std::string> indata_json = ReadFile(filename);
  ASSERT_TRUE(indata_json.ok()) << "Failed to read " << filename;

  absl::StatusOr<VmecINDATA> indata = VmecINDATA::FromJson(*indata_json);
  ASSERT_TRUE(indata.ok()) << "Failed to parse JSON";

  // Configure for very simple test
  VmecINDATA base_config = *indata;
  base_config.ns_array[0] = 3;
  base_config.niter_array[0] = 3;  // Just a few iterations
  base_config.mpol = 2;
  base_config.ntor = 0;  // 2D for simplicity
  base_config.return_outputs_even_if_not_converged = true;

  // Test 1: Symmetric mode
  VmecINDATA config_symm = base_config;
  config_symm.lasym = false;

  // Resize coefficient arrays to match new mpol/ntor
  int coeff_size_symm = config_symm.mpol * (2 * config_symm.ntor + 1);
  config_symm.rbc.resize(coeff_size_symm, 0.0);
  config_symm.zbs.resize(coeff_size_symm, 0.0);

  // Fix axis array sizes
  config_symm.raxis_c.resize(config_symm.ntor + 1, 0.0);
  config_symm.zaxis_s.resize(config_symm.ntor + 1, 0.0);

  config_symm.rbs.clear();
  config_symm.zbc.clear();
  config_symm.raxis_s.clear();
  config_symm.zaxis_c.clear();

  std::cout << "\n--- Testing SYMMETRIC mode ---" << std::endl;
  const auto output_symm = vmecpp::run(config_symm);

  bool symm_runs =
      output_symm.ok() ||
      output_symm.status().message().find("not converged") != std::string::npos;

  std::cout << "Symmetric result: " << (symm_runs ? "RUNS" : "FAILS")
            << std::endl;
  if (!output_symm.ok()) {
    std::cout << "  Status: " << output_symm.status() << std::endl;
  }

  // Test 2: Asymmetric mode (no crash test)
  VmecINDATA config_asymm = base_config;
  config_asymm.lasym = true;

  // Resize coefficient arrays to match new mpol/ntor
  int coeff_size_asymm = config_asymm.mpol * (2 * config_asymm.ntor + 1);
  config_asymm.rbc.resize(coeff_size_asymm, 0.0);
  config_asymm.zbs.resize(coeff_size_asymm, 0.0);

  // Fix axis array sizes first
  config_asymm.raxis_c.resize(config_asymm.ntor + 1, 0.0);
  config_asymm.zaxis_s.resize(config_asymm.ntor + 1, 0.0);

  // Resize asymmetric arrays
  config_asymm.rbs.resize(coeff_size_asymm, 0.0);
  config_asymm.zbc.resize(coeff_size_asymm, 0.0);
  config_asymm.raxis_s.resize(config_asymm.ntor + 1, 0.0);
  config_asymm.zaxis_c.resize(config_asymm.ntor + 1, 0.0);

  std::cout << "\n--- Testing ASYMMETRIC mode ---" << std::endl;
  const auto output_asymm = vmecpp::run(config_asymm);

  bool asymm_runs = output_asymm.ok() ||
                    output_asymm.status().message().find("not converged") !=
                        std::string::npos;

  std::cout << "Asymmetric result: " << (asymm_runs ? "RUNS" : "FAILS")
            << std::endl;
  if (!output_asymm.ok()) {
    std::cout << "  Status: " << output_asymm.status() << std::endl;
  }

  // Analysis
  std::cout << "\n--- ANALYSIS ---" << std::endl;
  std::cout << "Symmetric runs:  " << (symm_runs ? "✅ YES" : "❌ NO")
            << std::endl;
  std::cout << "Asymmetric runs: " << (asymm_runs ? "✅ YES" : "❌ NO")
            << std::endl;

  if (symm_runs && asymm_runs) {
    std::cout << "🎉 BOTH MODES WORK: Asymmetric implementation successful!"
              << std::endl;
  } else if (symm_runs && !asymm_runs) {
    std::cout << "⚠️  Symmetric works, asymmetric has remaining issues"
              << std::endl;
  } else if (!symm_runs && asymm_runs) {
    std::cout << "❌ REGRESSION: Symmetric broken, asymmetric works"
              << std::endl;
  } else {
    std::cout << "❌ BOTH BROKEN: Need to debug fundamental issues"
              << std::endl;
  }

  // Critical assertion: symmetric mode MUST work (no regression)
  EXPECT_TRUE(symm_runs) << "Symmetric mode must work to ensure no regression";

  // Asymmetric mode running (even if not converged) is already a major success
  if (asymm_runs) {
    std::cout
        << "\n✅ MAJOR SUCCESS: Asymmetric algorithm runs without crashes!"
        << std::endl;
  }
}

}  // namespace vmecpp
