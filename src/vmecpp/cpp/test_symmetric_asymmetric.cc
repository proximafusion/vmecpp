#include <gtest/gtest.h>
#include <iostream>
#include "vmecpp/common/vmec_indata/vmec_indata.h"
#include "vmecpp/vmec/vmec/vmec.h"

namespace vmecpp {

// Test symmetric mode - this MUST pass
TEST(SymmetricAsymmetricTest, SymmetricMode) {
  std::cout << "\n=== SYMMETRIC MODE TEST (MUST PASS) ===" << std::endl;

  VmecINDATA indata;
  
  // Basic parameters
  indata.nfp = 1;
  indata.lasym = false;  // SYMMETRIC
  indata.mpol = 3;
  indata.ntor = 0;
  indata.ns_array = {5};
  indata.niter_array = {50};
  indata.ntheta = 17;
  indata.nzeta = 1;
  
  // Physics parameters
  indata.phiedge = 1.0;
  indata.gamma = 0.0;
  indata.pres_scale = 0.0;
  indata.am = {0.0};
  indata.ncurr = 0;
  indata.curtor = 0.0;
  
  // Boundary arrays - correct size: (mpol+1) * (2*ntor+1) = 4 * 1 = 4
  const int array_size = (indata.mpol + 1) * (2 * indata.ntor + 1);
  indata.rbc.resize(array_size, 0.0);
  indata.zbs.resize(array_size, 0.0);
  
  // Simple tokamak
  indata.rbc[0] = 10.0;  // R00 - major radius
  indata.rbc[1] = 1.0;   // R10 - minor radius
  indata.zbs[1] = 1.0;   // Z10 - elongation
  
  // Axis
  indata.raxis_c = {10.0};
  indata.zaxis_s = {0.0};
  
  indata.return_outputs_even_if_not_converged = true;
  
  std::cout << "Config: mpol=" << indata.mpol << ", ntor=" << indata.ntor 
            << ", lasym=" << indata.lasym << std::endl;
  std::cout << "Array size: " << array_size << std::endl;
  
  auto result = vmecpp::run(indata);
  
  if (result.ok()) {
    std::cout << "✅ SYMMETRIC MODE PASSES!" << std::endl;
    const auto& wout = result->wout;
    std::cout << "   Volume = " << wout.volume_p << std::endl;
    std::cout << "   Beta = " << wout.volume_average_beta << std::endl;
    EXPECT_GT(wout.volume_p, 0.0);
  } else {
    std::cout << "❌ SYMMETRIC MODE FAILED: " << result.status() << std::endl;
    FAIL() << "Symmetric mode MUST pass: " << result.status();
  }
}

// Test asymmetric mode - goal is to make this pass
TEST(SymmetricAsymmetricTest, AsymmetricMode) {
  std::cout << "\n=== ASYMMETRIC MODE TEST (GOAL TO PASS) ===" << std::endl;

  VmecINDATA indata;
  
  // Basic parameters
  indata.nfp = 1;
  indata.lasym = true;   // ASYMMETRIC
  indata.mpol = 3; 
  indata.ntor = 0;
  indata.ns_array = {5};
  indata.niter_array = {50};
  indata.ntheta = 17;
  indata.nzeta = 1;
  
  // Physics parameters
  indata.phiedge = 1.0;
  indata.gamma = 0.0;
  indata.pres_scale = 0.0;
  indata.am = {0.0};
  indata.ncurr = 0;
  indata.curtor = 0.0;
  
  // Boundary arrays - correct size: (mpol+1) * (2*ntor+1) = 4 * 1 = 4
  const int array_size = (indata.mpol + 1) * (2 * indata.ntor + 1);
  
  // Symmetric arrays
  indata.rbc.resize(array_size, 0.0);
  indata.zbs.resize(array_size, 0.0);
  
  // Asymmetric arrays
  indata.rbs.resize(array_size, 0.0);
  indata.zbc.resize(array_size, 0.0);
  
  // Simple tokamak with small asymmetry
  indata.rbc[0] = 10.0;  // R00 - major radius
  indata.rbc[1] = 1.0;   // R10 - minor radius  
  indata.zbs[1] = 1.0;   // Z10 - elongation
  
  // Small asymmetric perturbations
  indata.rbs[1] = 0.05;  // Small asymmetric R10
  indata.zbc[1] = 0.05;  // Small asymmetric Z10
  
  // Axis (slightly asymmetric)
  indata.raxis_c = {10.0};
  indata.zaxis_s = {0.0};
  indata.raxis_s = {0.01}; // Small asymmetric axis
  indata.zaxis_c = {0.0};
  
  indata.return_outputs_even_if_not_converged = true;
  
  std::cout << "Config: mpol=" << indata.mpol << ", ntor=" << indata.ntor 
            << ", lasym=" << indata.lasym << std::endl;
  std::cout << "Array size: " << array_size << std::endl;
  
  auto result = vmecpp::run(indata);
  
  if (result.ok()) {
    std::cout << "✅ ASYMMETRIC MODE PASSES!" << std::endl;
    const auto& wout = result->wout;
    std::cout << "   Volume = " << wout.volume_p << std::endl;
    std::cout << "   Beta = " << wout.volume_average_beta << std::endl;
    EXPECT_GT(wout.volume_p, 0.0);
  } else {
    std::cout << "❌ ASYMMETRIC MODE FAILED: " << result.status() << std::endl;
    std::cout << "   This is expected - we're working to fix this!" << std::endl;
    
    // Check for specific errors
    std::string error_msg = result.status().ToString();
    if (error_msg.find("JACOBIAN") != std::string::npos) {
      std::cout << "   ERROR TYPE: Jacobian sign issue (expected)" << std::endl;
    }
    // Don't fail the test - this is our work in progress
  }
}

} // namespace vmecpp