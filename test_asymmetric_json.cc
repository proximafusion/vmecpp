// Test asymmetric JSON file
#include <gtest/gtest.h>
#include <iostream>
#include "util/json_io/json_io.h"
#include "vmecpp/common/vmec_indata/vmec_indata.h"
#include "vmecpp/vmec/vmec/vmec.h"

TEST(AsymmetricJsonTest, LoadExistingJsonFile) {
  std::cout << "\n=== Testing existing asymmetric JSON file ===" << std::endl;
  
  const std::string filename = "/home/ert/code/vmecpp/src/vmecpp/cpp/vmecpp/test_data/test_asymmetric_simple.json";
  
  // Load JSON
  auto json_result = json_io::LoadJson(filename);
  ASSERT_TRUE(json_result.ok()) << "Failed to load JSON: " << json_result.status();
  
  // Parse to INDATA
  auto indata_result = vmecpp::VmecINDATA::FromJson(json_result.value());
  ASSERT_TRUE(indata_result.ok()) << "Failed to parse INDATA: " << indata_result.status();
  
  auto indata = indata_result.value();
  
  std::cout << "Loaded configuration:" << std::endl;
  std::cout << "  lasym = " << indata.lasym << std::endl;
  std::cout << "  mpol = " << indata.mpol << ", ntor = " << indata.ntor << std::endl;
  std::cout << "  ns_array = " << indata.ns_array[0] << std::endl;
  std::cout << "  rbc size = " << indata.rbc.size() << std::endl;
  std::cout << "  rbs size = " << indata.rbs.size() << std::endl;
  
  // Set more debugging options
  indata.return_outputs_even_if_not_converged = true;
  indata.nstep = 1;  // Print every iteration
  
  std::cout << "\nRunning VMEC++ on asymmetric test case..." << std::endl;
  
  // Create VMEC instance with verbose output
  vmecpp::Vmec vmec(/*verbose=*/true);
  
  auto status = vmec.run(indata);
  
  if (!status.ok()) {
    std::cout << "\nVMEC failed with error: " << status.ToString() << std::endl;
    
    // Check for specific errors
    if (status.message().find("INITIAL JACOBIAN CHANGED SIGN") != std::string::npos) {
      std::cout << "\n⚠️  JACOBIAN SIGN ERROR - This is the key issue!" << std::endl;
      std::cout << "The initial guess interpolation is causing negative Jacobian" << std::endl;
    }
  } else {
    std::cout << "\n✅ SUCCESS! Asymmetric case ran to completion" << std::endl;
    auto result = vmec.getPrimaryOutputs();
    std::cout << "  Volume = " << result.volume_p << std::endl;
    std::cout << "  Beta = " << result.volume_average_beta << std::endl;
  }
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}