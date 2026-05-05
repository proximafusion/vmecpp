// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <iostream>
#include <vector>

#include "vmecpp/common/sizes/sizes.h"
#include "vmecpp/common/vmec_indata/vmec_indata.h"
#include "vmecpp/vmec/radial_profiles/radial_profiles.h"

namespace vmecpp {

// Test to debug pressure profile issues with asymmetric configurations
TEST(PressureDebugTest, CheckPressureProfileInitialization) {
  std::cout << "\n=== PRESSURE PROFILE DEBUG TEST ===" << std::endl;

  // Create a simple test configuration
  VmecINDATA indata;
  indata.lasym = true;
  indata.ns_array = {3};  // Few radial surfaces
  indata.mpol = 2;
  indata.ntor = 1;
  indata.nfp = 1;

  // Set up pressure profile
  indata.pmass_type = "power_series";
  indata.am = {1.0, 0.0};  // Simple linear profile
  indata.pres_scale = 1.0;
  indata.gamma = 0.0;  // No adiabatic compression

  std::cout << "Pressure profile configuration:" << std::endl;
  std::cout << "  pmass_type = " << indata.pmass_type << std::endl;
  std::cout << "  am = [";
  for (size_t i = 0; i < indata.am.size(); ++i) {
    std::cout << indata.am[i];
    if (i < indata.am.size() - 1) std::cout << ", ";
  }
  std::cout << "]" << std::endl;
  std::cout << "  pres_scale = " << indata.pres_scale << std::endl;
  std::cout << "  gamma = " << indata.gamma << std::endl;

  // Create radial profiles
  Sizes sizes(indata.lasym, indata.nfp, indata.mpol, indata.ntor, 8, 4);
  sizes.ns = indata.ns_array[0];

  RadialPartitioning radial_partitioning(&sizes);
  RadialProfiles radial_profiles(indata, radial_partitioning);

  std::cout << "\nRadial grid points:" << std::endl;
  std::cout << "  ns = " << sizes.ns << std::endl;
  std::cout << "  nsH = " << radial_partitioning.nsH << std::endl;
  std::cout << "  nsF = " << radial_partitioning.nsF << std::endl;

  // Check pressure values at different radial points
  std::cout << "\nPressure profile evaluation:" << std::endl;

  // Test at several s values
  std::vector<double> s_values = {0.0, 0.25, 0.5, 0.75, 1.0};
  for (double s : s_values) {
    try {
      double p = radial_profiles.evalMassProfile(s);
      std::cout << "  s = " << s << ", pressure = " << p << std::endl;

      if (!std::isfinite(p)) {
        std::cout << "    WARNING: Non-finite pressure value!" << std::endl;
      }
    } catch (const std::exception& e) {
      std::cout << "  s = " << s << ", ERROR: " << e.what() << std::endl;
    }
  }

  // Check pressure on half grid
  std::cout << "\nPressure on half grid (presH):" << std::endl;
  for (int j = 0; j < radial_profiles.presH.size(); ++j) {
    double pH = radial_profiles.presH[j];
    std::cout << "  presH[" << j << "] = " << pH;
    if (!std::isfinite(pH)) {
      std::cout << " (NaN!)";
    }
    std::cout << std::endl;
  }

  // Check for potential indexing issues
  std::cout << "\nIndexing check:" << std::endl;
  std::cout << "  presH.size() = " << radial_profiles.presH.size() << std::endl;
  std::cout << "  Expected accesses: j-1 for j in [1, ns-1]" << std::endl;
  std::cout << "  For ns=3: j=1 -> presH[0], j=2 -> presH[1]" << std::endl;
  std::cout << "  Valid indices: 0 to " << (radial_profiles.presH.size() - 1)
            << std::endl;

  // Verify all pressure values are finite
  bool all_finite = true;
  for (size_t i = 0; i < radial_profiles.presH.size(); ++i) {
    if (!std::isfinite(radial_profiles.presH[i])) {
      all_finite = false;
      break;
    }
  }

  EXPECT_TRUE(all_finite) << "Pressure profile contains non-finite values";
}

}  // namespace vmecpp
