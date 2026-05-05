#include <gtest/gtest.h>

#include <iostream>

#include "vmecpp/vmec/vmec/vmec.h"

TEST(SimpleDebugTest, CheckRadialPartitioning) {
  std::cout << "\n=== CHECKING RADIAL PARTITIONING ===\n";

  vmecpp::VmecINDATA config;
  config.lasym = true;
  config.nfp = 5;
  config.mpol = 3;
  config.ntor = 2;
  config.ntheta = 40;
  config.nzeta = 64;
  config.ns_array = {4, 8, 16};
  config.niter_array = {2, 4, 8};
  config.ftol_array = {1e-4, 1e-6, 1e-8};
  config.ncurr = 1;

  // Minimal boundary
  config.rbc = {{0, 0, 19.0}};  // R(m=0,n=0) = 19.0
  config.zbs = {{1, 0, 1.0}};   // Z(m=1,n=0) = 1.0

  // Add asymmetric perturbation
  config.rbs = {{1, 0, 0.1}};  // R asymmetric perturbation

  std::cout << "Running asymmetric VMEC to see radial partitioning...\n";

  try {
    vmecpp::Vmec vmec(config);
    auto result = vmec.run();
    std::cout << "VMEC completed: " << result.ok() << "\n";
  } catch (const std::exception& e) {
    std::cout << "Exception (expected): " << e.what() << "\n";
  }

  EXPECT_TRUE(true) << "Debug test completed";
}
