#include <fstream>
#include <iostream>

#include "util/file_io/file_io.h"
#include "vmecpp/vmec/vmec/vmec.h"

int main() {
  std::cout << "=== VMEC++ DEBUG OUTPUT CAPTURE ===\n";

  // Load the asymmetric tokamak configuration
  std::string input_file =
      "/home/ert/code/vmecpp/src/vmecpp/cpp/vmecpp/test_data/"
      "up_down_asymmetric_tokamak_simple.json";

  std::cout << "Loading configuration: " << input_file << "\n";

  auto maybe_input = file_io::ReadFile(input_file);
  if (!maybe_input.ok()) {
    std::cerr << "ERROR: Cannot read input file: " << maybe_input.status()
              << "\n";
    return 1;
  }

  auto maybe_indata = vmecpp::VmecINDATA::FromJson(*maybe_input);
  if (!maybe_indata.ok()) {
    std::cerr << "ERROR: Cannot parse JSON: " << maybe_indata.status() << "\n";
    return 1;
  }

  auto config = *maybe_indata;

  std::cout << "Configuration loaded successfully:\n";
  std::cout << "  lasym = " << (config.lasym ? "true" : "false") << "\n";
  std::cout << "  nfp = " << config.nfp << "\n";
  std::cout << "  mpol = " << config.mpol << ", ntor = " << config.ntor << "\n";
  std::cout << "  NS = " << config.ns_array[0] << "\n";
  std::cout << "  niter = " << config.niter_array[0] << "\n";

  std::cout << "\nAsymmetric boundary coefficients:\n";
  int coeff_size = config.mpol * (2 * config.ntor + 1);
  std::cout << "  rbc size: " << config.rbc.size() << " (expected: " << coeff_size
            << ")\n";
  std::cout << "  zbs size: " << config.zbs.size() << "\n";
  std::cout << "  rbs size: " << config.rbs.size() << " (asymmetric)\n";
  std::cout << "  zbc size: " << config.zbc.size() << " (asymmetric)\n";

  // Show a few key coefficients
  if (config.rbc.size() > 0) {
    int idx_00 = 0 * (2 * config.ntor + 1) + config.ntor;  // m=0, n=0
    int idx_10 = 1 * (2 * config.ntor + 1) + config.ntor;  // m=1, n=0
    std::cout << "\nKey coefficients:\n";
    if (idx_00 < static_cast<int>(config.rbc.size())) {
      std::cout << "  R00 = " << config.rbc[idx_00] << " (axis)\n";
    }
    if (idx_10 < static_cast<int>(config.rbc.size())) {
      std::cout << "  R10 = " << config.rbc[idx_10] << " (shape)\n";
      std::cout << "  Z10 = " << config.zbs[idx_10] << " (shape)\n";
    }
    if (idx_10 < static_cast<int>(config.rbs.size())) {
      std::cout << "  R10s = " << config.rbs[idx_10] << " (asymmetric)\n";
    }
    if (idx_10 < static_cast<int>(config.zbc.size())) {
      std::cout << "  Z10c = " << config.zbc[idx_10] << " (asymmetric)\n";
    }
  }

  std::cout << "\n" << std::string(60, '=') << "\n";
  std::cout << "RUNNING VMEC++ WITH DEBUG OUTPUT\n";
  std::cout << std::string(60, '=') << "\n";

  try {
    // Create and run VMEC
    vmecpp::Vmec vmec(config);
    auto result = vmec.run();

    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "VMEC++ EXECUTION COMPLETED\n";
    std::cout << std::string(60, '=') << "\n";

    if (result.ok()) {
      std::cout << "✅ VMEC++ converged successfully!\n";
    } else {
      std::cout << "❌ VMEC++ failed to converge\n";
      std::cout << "Error: " << result.status() << "\n";
    }

  } catch (const std::exception& e) {
    std::cout << "\n❌ EXCEPTION CAUGHT:\n";
    std::cout << "Exception: " << e.what() << "\n";
  }

  std::cout << "\n=== DEBUG OUTPUT CAPTURE COMPLETE ===\n";
  std::cout << "Review the debug output above for:\n";
  std::cout << "1. Array bounds debugging\n";
  std::cout << "2. Tau calculation components (tau1, tau2, odd_contrib)\n";
  std::cout << "3. Geometry derivatives (ru12, zu12, rs, zs)\n";
  std::cout << "4. Min/max tau values and Jacobian check\n";
  std::cout << "5. Any non-finite values or errors\n";

  return 0;
}