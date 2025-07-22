#include <fstream>
#include <iostream>
#include <string>

#include "util/file_io/file_io.h"
#include "vmecpp/vmec/vmec/vmec.h"

int main() {
  std::cout << "DEBUG: Testing VMEC++ crash with asymmetric tokamak"
            << std::endl;

  try {
    // Read the asymmetric tokamak input
    std::string input_file =
        "/home/ert/code/vmecpp/src/vmecpp/cpp/vmecpp/test_data/"
        "up_down_asymmetric_tokamak_simple.json";
    auto maybe_input = file_io::ReadFile(input_file);
    if (!maybe_input.ok()) {
      std::cout << "ERROR: Cannot read input file: " << maybe_input.status()
                << std::endl;
      return 1;
    }

    std::cout << "DEBUG: Successfully read input file" << std::endl;

    // Parse VMEC input
    auto maybe_indata = vmecpp::VmecINDATA::FromJson(*maybe_input);
    if (!maybe_indata.ok()) {
      std::cout << "ERROR: Cannot parse input: " << maybe_indata.status()
                << std::endl;
      return 1;
    }

    std::cout << "DEBUG: Successfully parsed input - lasym="
              << maybe_indata->lasym << std::endl;

    // Create VMEC instance
    vmecpp::Vmec vmec(*maybe_indata);
    std::cout << "DEBUG: Successfully created VMEC instance" << std::endl;

    // Run just one iteration to trigger the crash
    std::cout << "DEBUG: About to run VMEC iteration..." << std::endl;
    auto result = vmec.run(vmecpp::VmecCheckpoint::NONE,
                           1);  // Just 1 iteration to see where it crashes

    if (result.ok()) {
      std::cout << "DEBUG: VMEC iteration completed successfully" << std::endl;
    } else {
      std::cout << "ERROR: VMEC iteration failed: " << result.status()
                << std::endl;
    }

  } catch (const std::exception& e) {
    std::cout << "ERROR: Exception caught: " << e.what() << std::endl;
    return 1;
  } catch (...) {
    std::cout << "ERROR: Unknown exception caught" << std::endl;
    return 1;
  }

  return 0;
}
