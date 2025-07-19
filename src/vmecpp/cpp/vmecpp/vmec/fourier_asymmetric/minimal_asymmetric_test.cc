// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT

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

// Minimal test to debug asymmetric configuration issues
TEST(MinimalAsymmetricTest, SimpleAsymmetricTest) {
  std::cout << "\n=== MINIMAL ASYMMETRIC TEST ===" << std::endl;

  // Load a simple configuration
  const std::string filename = "vmecpp/test_data/cth_like_fixed_bdy.json";
  absl::StatusOr<std::string> indata_json = ReadFile(filename);
  ASSERT_TRUE(indata_json.ok()) << "Failed to read " << filename;

  absl::StatusOr<VmecINDATA> indata = VmecINDATA::FromJson(*indata_json);
  ASSERT_TRUE(indata.ok()) << "Failed to parse JSON";

  VmecINDATA indata_asymm = *indata;

  // Enable asymmetric mode
  indata_asymm.lasym = true;

  // Reduce parameters for faster testing
  indata_asymm.ns_array[0] = 3;      // Very few radial surfaces
  indata_asymm.niter_array[0] = 10;  // Few iterations
  indata_asymm.mpol = 2;             // Reduce poloidal modes
  indata_asymm.ntor = 1;             // Reduce toroidal modes
  indata_asymm.return_outputs_even_if_not_converged = true;

  std::cout << "Configuration: lasym=" << indata_asymm.lasym
            << ", mpol=" << indata_asymm.mpol << ", ntor=" << indata_asymm.ntor
            << ", ns=" << indata_asymm.ns_array[0] << std::endl;

  // Set up a simple pressure profile to avoid NaN issues
  indata_asymm.pmass_type = "power_series";
  indata_asymm.pres_scale = 0.01;  // Small pressure for testing
  indata_asymm.am = {1.0, -1.0};   // Linear profile p(s) = pres_scale * s
  indata_asymm.gamma = 0.0;        // No adiabatic compression

  std::cout << "Modified pressure profile: type=" << indata_asymm.pmass_type
            << ", pres_scale=" << indata_asymm.pres_scale
            << ", am.size()=" << indata_asymm.am.size() << std::endl;

  // Resize coefficient arrays appropriately
  int coeff_size = indata_asymm.mpol * (2 * indata_asymm.ntor + 1);
  indata_asymm.rbc.resize(coeff_size, 0.0);
  indata_asymm.zbs.resize(coeff_size, 0.0);
  indata_asymm.rbs.resize(coeff_size, 0.0);
  indata_asymm.zbc.resize(coeff_size, 0.0);

  // Initialize symmetric coefficients with simple values
  // (m=0, n=0) - constant term
  int idx_00 = 0 * (2 * indata_asymm.ntor + 1) + (0 + indata_asymm.ntor);
  indata_asymm.rbc[idx_00] = 1.0;  // Major radius

  // (m=1, n=0) - simple elliptical cross-section
  int idx_10 = 1 * (2 * indata_asymm.ntor + 1) + (0 + indata_asymm.ntor);
  indata_asymm.rbc[idx_10] = 0.1;  // Minor radius
  indata_asymm.zbs[idx_10] = 0.1;  // Vertical elongation

  // Add tiny asymmetric perturbation only for asymmetric case
  if (indata_asymm.lasym) {
    indata_asymm.rbs[idx_10] = 0.001;  // 1% of symmetric value
    indata_asymm.zbc[idx_10] = 0.001;
  } else {
    // Clear asymmetric coefficient arrays for symmetric case
    indata_asymm.rbs.clear();
    indata_asymm.zbc.clear();
  }

  std::cout << "Coefficients initialized:" << std::endl;
  std::cout << "  rbc[" << idx_00 << "] = " << indata_asymm.rbc[idx_00]
            << " (m=0,n=0)" << std::endl;
  std::cout << "  rbc[" << idx_10 << "] = " << indata_asymm.rbc[idx_10]
            << " (m=1,n=0)" << std::endl;
  std::cout << "  zbs[" << idx_10 << "] = " << indata_asymm.zbs[idx_10]
            << " (m=1,n=0)" << std::endl;

  if (indata_asymm.lasym) {
    std::cout << "  rbs[" << idx_10 << "] = " << indata_asymm.rbs[idx_10]
              << " (m=1,n=0) ASYMMETRIC" << std::endl;
    std::cout << "  zbc[" << idx_10 << "] = " << indata_asymm.zbc[idx_10]
              << " (m=1,n=0) ASYMMETRIC" << std::endl;
  }

  // Initialize axis arrays
  indata_asymm.raxis_c.resize(indata_asymm.ntor + 1, 0.0);
  indata_asymm.zaxis_s.resize(indata_asymm.ntor + 1, 0.0);

  if (indata_asymm.lasym) {
    // Only initialize asymmetric axis arrays for asymmetric case
    indata_asymm.raxis_s.resize(indata_asymm.ntor + 1, 0.0);  // Asymmetric
    indata_asymm.zaxis_c.resize(indata_asymm.ntor + 1, 0.0);  // Asymmetric
  } else {
    // Clear asymmetric arrays for symmetric case
    indata_asymm.raxis_s.clear();
    indata_asymm.zaxis_c.clear();
  }

  // Set axis position
  indata_asymm.raxis_c[0] = 1.0;  // Same as boundary R00

  std::cout << "\nAxis arrays initialized with size " << indata_asymm.ntor + 1
            << std::endl;

  // Check pressure profile values before running VMEC
  std::cout << "\nPressure profile values:" << std::endl;
  std::cout << "  pmass_type = " << indata_asymm.pmass_type << std::endl;
  std::cout << "  pres_scale = " << indata_asymm.pres_scale << std::endl;
  std::cout << "  gamma = " << indata_asymm.gamma << std::endl;
  std::cout << "  am coefficients:" << std::endl;
  for (size_t i = 0; i < indata_asymm.am.size(); ++i) {
    std::cout << "    am[" << i << "] = " << indata_asymm.am[i] << std::endl;
  }

  std::cout << "\nRunning VMEC with asymmetric configuration..." << std::endl;

  // Run VMEC
  const auto output = vmecpp::run(indata_asymm);

  if (output.ok()) {
    std::cout << "\nVMEC run completed successfully!" << std::endl;
    const auto& wout = output->wout;
    std::cout << "Output: ns=" << wout.ns << ", volume=" << wout.volume_p
              << std::endl;
  } else {
    std::cout << "\nVMEC run failed: " << output.status() << std::endl;
  }

  EXPECT_TRUE(output.ok()) << "Asymmetric VMEC run failed: " << output.status();
}

}  // namespace vmecpp
