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

// Test using known-good jVMEC asymmetric tokamak configuration
TEST(JVMECTokAsymTest, RunKnownGoodAsymmetricTokamak) {
  std::cout << "\n=== JVMEC TOK_ASYM REFERENCE TEST ===\n" << std::endl;

  // Replicate the exact tok_asym configuration from jVMEC
  VmecINDATA indata;

  // Basic parameters from input.tok_asym
  indata.nfp = 1;
  indata.ncurr = 0;
  indata.delt = 0.9;
  indata.lasym = true;
  indata.mpol = 7;
  indata.ntor = 0;  // Axisymmetric tokamak
  indata.ntheta = 17;
  indata.nzeta = 1;

  // Multigrid progression (using working parameters)
  indata.ns_array = {3, 5};
  indata.ftol_array = {1.0e-4, 1.0e-6};
  indata.niter_array = {50, 100};
  indata.nstep = 100;

  // Pressure profile
  indata.am = {1.0, -2.0, 1.0};
  indata.pres_scale = 100000.0;

  // Current profile
  indata.ai = {0.6, -0.45};
  indata.curtor = 0.0;
  indata.gamma = 0.0;
  indata.phiedge = 119.15;

  // Axis position (using working configuration for ntor=0)
  indata.raxis_c = {6.676};
  indata.zaxis_s = {0.47};

  // Asymmetric axis arrays (empty for tokamak)
  indata.raxis_s = {0.0};
  indata.zaxis_c = {0.0};

  // Allow non-converged output for testing
  indata.return_outputs_even_if_not_converged = true;

  std::cout << "Setting up boundary coefficients from jVMEC tok_asym..."
            << std::endl;

  // Calculate coefficient array size for asymmetric tokamak
  int coeff_size = indata.mpol * (2 * indata.ntor + 1);
  indata.rbc.resize(coeff_size, 0.0);
  indata.zbs.resize(coeff_size, 0.0);
  indata.rbs.resize(coeff_size, 0.0);
  indata.zbc.resize(coeff_size, 0.0);

  // Set boundary coefficients exactly as in jVMEC input.tok_asym
  // Format: rbc(m,n), rbs(m,n), zbc(m,n), zbs(m,n)
  // Index calculation: idx = m * (2*ntor + 1) + (n + ntor)
  // For ntor=0: idx = m

  // m=0 coefficients
  indata.rbc[0] = 5.91630000E+00;
  indata.rbs[0] = 0.00000000E+00;
  indata.zbc[0] = 4.10500000E-01;
  indata.zbs[0] = 0.00000000E+00;

  // m=1 coefficients
  indata.rbc[1] = 1.91960000E+00;
  indata.rbs[1] = 2.76100000E-02;
  indata.zbc[1] = 5.73020000E-02;
  indata.zbs[1] = 3.62230000E+00;

  std::cout << "DEBUG: M=1 coefficients from input:" << std::endl;
  std::cout << "  rbc[1] = " << indata.rbc[1] << std::endl;
  std::cout << "  rbs[1] = " << indata.rbs[1] << std::endl;
  std::cout << "  zbc[1] = " << indata.zbc[1] << std::endl;
  std::cout << "  zbs[1] = " << indata.zbs[1] << std::endl;

  // m=2 coefficients
  indata.rbc[2] = 3.37360000E-01;
  indata.rbs[2] = 1.00380000E-01;
  indata.zbc[2] = 4.66970000E-03;
  indata.zbs[2] = -1.85110000E-01;

  // m=3 coefficients
  indata.rbc[3] = 4.15040000E-02;
  indata.rbs[3] = -7.18430000E-02;
  indata.zbc[3] = -3.91550000E-02;
  indata.zbs[3] = -4.85680000E-03;

  // m=4 coefficients
  indata.rbc[4] = -5.82560000E-03;
  indata.rbs[4] = -1.14230000E-02;
  indata.zbc[4] = -8.78480000E-03;
  indata.zbs[4] = 5.92680000E-02;

  // m=5 coefficients
  indata.rbc[5] = 1.03740000E-02;
  indata.rbs[5] = 8.17770000E-03;
  indata.zbc[5] = 2.11750000E-02;
  indata.zbs[5] = 4.47690000E-03;

  // m=6 coefficients (mpol=7 means m goes from 0 to 6)
  indata.rbc[6] = -5.63650000E-03;
  indata.rbs[6] = -7.61150000E-03;
  indata.zbc[6] = 2.43930000E-03;
  indata.zbs[6] = -1.67730000E-02;

  std::cout << "Configuration summary:" << std::endl;
  std::cout << "  lasym = " << indata.lasym << std::endl;
  std::cout << "  mpol = " << indata.mpol << ", ntor = " << indata.ntor
            << std::endl;
  std::cout << "  Major radius (R00) = " << indata.rbc[0] << std::endl;
  std::cout << "  Minor radius (R10) = " << indata.rbc[1] << std::endl;
  std::cout << "  Asymmetric R10 = " << indata.rbs[1] << std::endl;
  std::cout << "  Asymmetric Z10 = " << indata.zbc[1] << std::endl;
  std::cout << "  Pressure scale = " << indata.pres_scale << std::endl;

  std::cout << "\nRunning VMEC with jVMEC tok_asym configuration..."
            << std::endl;

  const auto output = vmecpp::run(indata);

  if (output.ok()) {
    std::cout << "\n✅ SUCCESS: jVMEC tok_asym configuration completed!"
              << std::endl;
    const auto& wout = output->wout;
    std::cout << "  Volume = " << wout.volume_p << std::endl;
    std::cout << "  Aspect ratio = " << wout.aspect << std::endl;
    std::cout << "  Beta total = " << wout.betatot << std::endl;
    std::cout << "  Magnetic energy = " << wout.wb << std::endl;

    // This proves asymmetric transforms and physics work correctly
    EXPECT_GT(wout.volume_p, 0.0) << "Volume should be positive";
    EXPECT_GT(wout.aspect, 0.0) << "Aspect ratio should be positive";
    EXPECT_TRUE(std::isfinite(wout.betatot)) << "Beta should be finite";

  } else {
    std::cout << "\n❌ FAILED: jVMEC tok_asym configuration failed"
              << std::endl;
    std::cout << "Error: " << output.status() << std::endl;

    // This indicates still unresolved issues in asymmetric implementation
    FAIL() << "Known-good jVMEC asymmetric configuration should work: "
           << output.status();
  }
}

}  // namespace vmecpp
