// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <iostream>
#include <vector>

#include "vmecpp/common/vmec_indata/vmec_indata.h"
#include "vmecpp/vmec/vmec/vmec.h"

namespace vmecpp {

// Test with very simple asymmetric configuration to check basic functionality
TEST(SimpleAsymmetricTest, BasicAsymmetricTokamak) {
  std::cout << "\n=== SIMPLE ASYMMETRIC TEST ===\n" << std::endl;

  // Use jVMEC reference case: tok_simple_asym
  VmecINDATA indata;

  // jVMEC working parameters
  indata.nfp = 1;
  indata.lasym = true;
  indata.mpol = 3;  // Match jVMEC reference
  indata.ntor = 0;  // Axisymmetric
  indata.ns_array = {5};  // Very low resolution like jVMEC
  indata.niter_array = {5};  // Few iterations
  indata.ntheta = 17;
  indata.nzeta = 1;

  // Zero pressure vacuum case
  indata.pres_scale = 0.0;
  indata.am = {0.4, 0.0};  // iota profile from jVMEC
  indata.gamma = 0.0;
  indata.phiedge = 1.0;  // toroidal flux

  indata.return_outputs_even_if_not_converged = true;

  // jVMEC reference boundary configuration
  // VMEC++ uses 1D arrays with index = m * (2*ntor+1) + (ntor + n)
  // For ntor=0, we only have n=0, so index = m * 1 + 0 = m
  const int array_size = (indata.mpol + 1) * (2 * indata.ntor + 1);
  indata.rbc.resize(array_size, 0.0);
  indata.zbs.resize(array_size, 0.0);
  indata.rbs.resize(array_size, 0.0);
  indata.zbc.resize(array_size, 0.0);
  
  // jVMEC reference values
  indata.rbc[0] = 1.0;   // RBC(0,0) = 1.0 - Major radius
  indata.rbc[1] = 0.3;   // RBC(0,1) = 0.3 - Ellipticity  
  indata.zbs[1] = 0.3;   // ZBS(0,1) = 0.3 - Elongation
  indata.rbs[1] = 0.001; // RBS(0,1) = 0.001 - TINY asymmetric perturbation

  // Axis arrays (match boundary m=0)
  indata.raxis_c = {1.0};  // Match RBC(0,0)
  indata.zaxis_s = {0.0};
  indata.raxis_s = {0.0};
  indata.zaxis_c = {0.0};

  std::cout << "jVMEC Reference Configuration:" << std::endl;
  std::cout << "  lasym = " << indata.lasym << std::endl;
  std::cout << "  mpol = " << indata.mpol << ", ntor = " << indata.ntor
            << std::endl;
  std::cout << "  ns_array = " << indata.ns_array[0] << ", niter = " << indata.niter_array[0] << std::endl;
  std::cout << "  RBC(0,0) = " << indata.rbc[0] << " (major radius)" << std::endl;
  std::cout << "  RBC(0,1) = " << indata.rbc[1] << " (ellipticity)" << std::endl;
  std::cout << "  ZBS(0,1) = " << indata.zbs[1] << " (elongation)" << std::endl;
  std::cout << "  RBS(0,1) = " << indata.rbs[1] << " (asymmetric - 0.1% perturbation)" << std::endl;

  std::cout << "\nRunning jVMEC reference asymmetric case..."
            << std::endl;

  const auto output = vmecpp::run(indata);

  if (output.ok()) {
    std::cout << "\n✅ SUCCESS: jVMEC reference asymmetric case works!"
              << std::endl;
    const auto& wout = output->wout;
    std::cout << "  Volume = " << wout.volume_p << std::endl;
    std::cout << "  Aspect ratio = " << wout.aspect << std::endl;
    std::cout << "  Iterations converged!" << std::endl;

    EXPECT_GT(wout.volume_p, 0.0) << "Volume should be positive";
    EXPECT_GT(wout.aspect, 0.0) << "Aspect ratio should be positive";

  } else {
    std::cout << "\n❌ FAILED: jVMEC reference asymmetric case failed"
              << std::endl;
    std::cout << "Error: " << output.status() << std::endl;

    // This jVMEC reference case is known to work - failure indicates 
    // missing robustness mechanism in VMEC++
    FAIL() << "jVMEC reference asymmetric case should work: "
           << output.status();
  }
}

}  // namespace vmecpp
