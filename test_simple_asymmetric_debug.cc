// Debug test for simple asymmetric case

#include <gtest/gtest.h>
#include <iostream>
#include <vector>
#include <cmath>

#include "vmecpp/common/vmec_indata/vmec_indata.h"
#include "vmecpp/vmec/vmec/vmec.h"

TEST(AsymmetricDebug, SimpleAsymmetricTokamak) {
  vmecpp::VmecINDATA indata;
  
  // Simple asymmetric tokamak parameters
  indata.lasym = true;
  indata.nfp = 1;
  indata.mpol = 3;
  indata.ntor = 0;
  indata.ns_array = {5};
  indata.ftol_array = {1e-6};
  indata.niter_array = {5};
  indata.delt = 0.9;
  indata.tcon0 = 1.0;
  indata.phiedge = 1.0;
  indata.nstep = 10;
  indata.ncurr = 0;
  indata.lfreeb = false;
  
  // Pressure and current profiles
  indata.pmass_type = "power_series";
  indata.am = {0.0, 1.0, 0.0};
  indata.piota_type = "power_series";
  indata.ai = {0.4, 0.0};
  
  // Axis
  indata.raxis_cc = {1.0, 0.0, 0.0, 0.0};
  indata.zaxis_cs = {0.0, 0.0, 0.0, 0.0};
  
  // Boundary coefficients
  // Symmetric baseline
  indata.rbc[vmecpp::VmecINDATA::ntord + 0][0] = 1.0;  // R00
  indata.rbc[vmecpp::VmecINDATA::ntord + 0][1] = 0.3;  // R01
  indata.zbs[vmecpp::VmecINDATA::ntord + 0][1] = 0.3;  // Z01
  
  // Asymmetric perturbation
  indata.rbs[vmecpp::VmecINDATA::ntord + 0][1] = 0.001;  // Small asymmetric perturbation
  
  std::cout << "\n=== Running simple asymmetric tokamak test ===" << std::endl;
  std::cout << "lasym = " << indata.lasym << std::endl;
  std::cout << "mpol = " << indata.mpol << ", ntor = " << indata.ntor << std::endl;
  std::cout << "ns_array = " << indata.ns_array[0] << std::endl;
  std::cout << "R00 = " << indata.rbc[vmecpp::VmecINDATA::ntord + 0][0] << std::endl;
  std::cout << "R01 = " << indata.rbc[vmecpp::VmecINDATA::ntord + 0][1] << std::endl;
  std::cout << "RBS01 = " << indata.rbs[vmecpp::VmecINDATA::ntord + 0][1] << " (asymmetric)" << std::endl;
  std::cout << "Z01 = " << indata.zbs[vmecpp::VmecINDATA::ntord + 0][1] << std::endl;
  
  // Create VMEC instance and run
  vmecpp::Vmec vmec(/*verbose=*/true);
  
  std::cout << "\nStarting VMEC run..." << std::endl;
  auto status = vmec.run(indata);
  
  if (!status.ok()) {
    std::cout << "VMEC failed with error: " << status.ToString() << std::endl;
    FAIL();
  }
  
  std::cout << "\nVMEC run completed successfully!" << std::endl;
  
  // Get results
  auto result = vmec.getPrimaryOutputs();
  std::cout << "Beta = " << result.volume_average_beta << std::endl;
  std::cout << "MHD Energy = " << result.w_mhd << std::endl;
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}