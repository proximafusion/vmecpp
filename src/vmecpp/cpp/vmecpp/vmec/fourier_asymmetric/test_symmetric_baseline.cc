#include <gtest/gtest.h>
#include <iostream>

#include "vmecpp/common/vmec_indata/vmec_indata.h"
#include "vmecpp/vmec/vmec/vmec.h"

namespace vmecpp {

TEST(SymmetricBaselineTest, EnsureSymmetricWorks) {
  std::cout << "\n=== SYMMETRIC BASELINE TEST ===\n";
  std::cout << "Testing that symmetric mode MUST PASS (requirement)\n\n";

  // Create simple symmetric tokamak configuration
  VmecINDATA indata;
  indata.lasym = false;  // SYMMETRIC mode
  indata.nfp = 1;
  indata.mpol = 4;
  indata.ntor = 0;
  indata.ntheta = 18;
  indata.nzeta = 1;
  
  // Radial grid
  indata.ns_array = {3, 5};
  indata.ftol_array = {1e-6, 1e-8};
  indata.niter_array = {50, 100};
  
  // Physics parameters
  indata.delt = 0.9;
  indata.phiedge = 1.0;
  indata.gamma = 0.0;
  indata.pres_scale = 0.1;
  
  // Pressure profile
  indata.pmass_type = "power_series";
  indata.am = {1.0, -1.0};
  
  // Boundary shape (simple circular tokamak)
  indata.raxis_c = {10.0};
  indata.zaxis_s = {0.0};
  
  // Set boundary Fourier coefficients
  indata.rbc.resize((indata.mpol + 1) * (indata.ntor + 1), 0);
  indata.zbs.resize((indata.mpol + 1) * (indata.ntor + 1), 0);
  
  // RBC(0,0) = 10.0 (major radius)
  indata.rbc[0] = 10.0;
  
  // RBC(1,0) = 1.0 (minor radius)
  indata.rbc[1] = 1.0;
  
  // ZBS(1,0) = 1.0 (elongation)
  indata.zbs[1] = 1.0;
  
  std::cout << "Running symmetric case with:\n";
  std::cout << "  RBC(0,0) = " << indata.rbc[0] << " (major radius)\n";
  std::cout << "  RBC(1,0) = " << indata.rbc[1] << " (minor radius)\n";
  std::cout << "  ZBS(1,0) = " << indata.zbs[1] << " (elongation)\n";
  
  try {
    auto result = Vmec::run(indata);
    
    std::cout << "\n✓ SYMMETRIC CASE PASSED!\n";
    std::cout << "  Beta = " << result.beta << "\n";
    std::cout << "  MHD Energy = " << result.wmhd << "\n";
    std::cout << "  Aspect Ratio = " << result.aspect_ratio << "\n";
    
    EXPECT_GT(result.wmhd, 0) << "MHD energy should be positive";
    EXPECT_GT(result.aspect_ratio, 0) << "Aspect ratio should be positive";
    
  } catch (const std::exception& e) {
    std::cout << "\n✗ SYMMETRIC CASE FAILED!\n";
    std::cout << "Error: " << e.what() << "\n";
    FAIL() << "Symmetric case MUST pass - this is a requirement!";
  }
}

}  // namespace vmecpp