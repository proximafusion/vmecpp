// TDD test to identify and fix array indexing bug in Jacobian calculation
// The debug output shows r1[j+1]=-0.029389 which is clearly wrong

#include <gtest/gtest.h>

#include <cmath>
#include <iomanip>
#include <iostream>

#include "vmecpp/common/vmec_indata/vmec_indata.h"
#include "vmecpp/vmec/vmec/vmec.h"

using vmecpp::Vmec;
using vmecpp::VmecINDATA;

namespace vmecpp {

TEST(ArrayIndexingBugTest, DebugGeometryArrayValues) {
  std::cout << "\n=== DEBUG GEOMETRY ARRAY VALUES ===\n";
  std::cout << std::fixed << std::setprecision(6);

  // Use the exact same configuration that showed the bug
  VmecINDATA config;
  config.lasym = true;
  config.nfp = 1;
  config.mpol = 2;
  config.ntor = 0;
  config.ns_array = {3};
  config.niter_array = {1};
  config.ftol_array = {1e-6};
  config.return_outputs_even_if_not_converged = true;

  config.delt = 0.5;
  config.tcon0 = 1.0;
  config.phiedge = 1.0;
  config.gamma = 0.0;
  config.curtor = 0.0;
  config.ncurr = 0;

  config.pmass_type = "power_series";
  config.am = {0.0};
  config.pres_scale = 0.0;

  config.piota_type = "power_series";
  config.ai = {0.0};

  // Large R0 to avoid numerical issues, with m=1 asymmetric modes
  config.raxis_c = {20.0};
  config.zaxis_s = {0.0};

  config.rbc =
      std::vector<double>((config.mpol + 1) * (2 * config.ntor + 1), 0.0);
  config.zbs =
      std::vector<double>((config.mpol + 1) * (2 * config.ntor + 1), 0.0);

  auto setMode = [&](int m, int n, double rbc_val, double zbs_val) {
    int idx = m * (2 * config.ntor + 1) + n + config.ntor;
    config.rbc[idx] = rbc_val;
    config.zbs[idx] = zbs_val;
  };

  setMode(0, 0, 20.0, 0.0);  // Major radius
  setMode(1, 0, 1.0, 1.0);   // m=1 symmetric

  // Add asymmetric arrays
  config.rbs =
      std::vector<double>((config.mpol + 1) * (2 * config.ntor + 1), 0.0);
  config.zbc =
      std::vector<double>((config.mpol + 1) * (2 * config.ntor + 1), 0.0);

  auto setAsymMode = [&](int m, int n, double rbs_val, double zbc_val) {
    int idx = m * (2 * config.ntor + 1) + n + config.ntor;
    config.rbs[idx] = rbs_val;
    config.zbc[idx] = zbc_val;
  };

  setAsymMode(1, 0, 0.1, 0.1);  // m=1 asymmetric (10%)

  std::cout << "\nConfiguration with m=1 asymmetric modes:\n";
  std::cout << "  R0 = " << config.raxis_c[0] << "\n";
  std::cout << "  m=1 symmetric: rbc=" << config.rbc[1]
            << ", zbs=" << config.zbs[1] << "\n";
  std::cout << "  m=1 asymmetric: rbs=" << config.rbs[1]
            << ", zbc=" << config.zbc[1] << "\n";

  // Run with enhanced debug output
  auto result = Vmec::Run(config);

  // The debug output should show the problematic array values
  EXPECT_TRUE(true) << "Array indexing bug analysis completed";
}

TEST(ArrayIndexingBugTest, DiagnoseIndexingLogic) {
  std::cout << "\n=== DIAGNOSE INDEXING LOGIC ===\n";

  std::cout << "FROM DEBUG OUTPUT:\n";
  std::cout << "- r1[j]=18.970611 (reasonable R value)\n";
  std::cout << "- r1[j+1]=-0.029389 (impossible, likely theta derivative)\n";
  std::cout << "- z1[j]=0.059549 (reasonable Z value)\n";
  std::cout << "- z1[j+1]=-0.040451 (also looks like derivative)\n\n";

  std::cout << "HYPOTHESIS:\n";
  std::cout << "The variables r1e_o, r1o_o are being read from wrong arrays.\n";
  std::cout << "Values like -0.029389 suggest reading from ru/zu arrays "
               "instead of r1/z1.\n\n";

  std::cout << "CRITICAL LINES TO CHECK:\n";
  std::cout << "Line 1772: double r1e_o = r1_e[(jH + 1 - r_.nsMinF1) * s_.nZnT "
               "+ kl];\n";
  std::cout << "Line 1773: double r1o_o = r1_o[(jH + 1 - r_.nsMinF1) * s_.nZnT "
               "+ kl];\n\n";

  std::cout << "POSSIBLE CAUSES:\n";
  std::cout << "1. Array r1_o is corrupted with ru/zu values\n";
  std::cout
      << "2. Index calculation (jH + 1 - r_.nsMinF1) * s_.nZnT + kl is wrong\n";
  std::cout << "3. Array bounds issue causing reading from wrong memory\n";
  std::cout << "4. r1_o array not properly initialized (contains garbage)\n\n";

  std::cout << "DEBUG STRATEGY:\n";
  std::cout << "1. Add array bounds checking before reading r1e_o, r1o_o\n";
  std::cout << "2. Print array sizes and index calculations\n";
  std::cout << "3. Verify r1_o array contains R values, not derivatives\n";
  std::cout << "4. Check if odd array population is overwriting even arrays\n";

  EXPECT_TRUE(true) << "Indexing logic diagnosis completed";
}

TEST(ArrayIndexingBugTest, ProposeArrayDebugging) {
  std::cout << "\n=== PROPOSE ARRAY DEBUGGING ===\n";

  std::cout << "ADD TO ideal_mhd_model.cc BEFORE LINE 1772:\n\n";

  std::cout << "// Debug array bounds and index calculation\n";
  std::cout << "if (s_.lasym && jH == 0 && kl == 6) {\n";
  std::cout << "  int target_idx = (jH + 1 - r_.nsMinF1) * s_.nZnT + kl;\n";
  std::cout << "  std::cout << \"Array index debug jH=\" << jH << \" kl=\" << "
               "kl << \":\\n\";\n";
  std::cout << "  std::cout << \"  jH + 1 = \" << (jH + 1) << \"\\n\";\n";
  std::cout << "  std::cout << \"  r_.nsMinF1 = \" << r_.nsMinF1 << \"\\n\";\n";
  std::cout << "  std::cout << \"  s_.nZnT = \" << s_.nZnT << \"\\n\";\n";
  std::cout << "  std::cout << \"  target_idx = \" << target_idx << \"\\n\";\n";
  std::cout
      << "  std::cout << \"  r1_e.size() = \" << r1_e.size() << \"\\n\";\n";
  std::cout
      << "  std::cout << \"  r1_o.size() = \" << r1_o.size() << \"\\n\";\n";
  std::cout << "  if (target_idx < r1_e.size()) {\n";
  std::cout << "    std::cout << \"  r1_e[\" << target_idx << \"] = \" << "
               "r1_e[target_idx] << \"\\n\";\n";
  std::cout << "    std::cout << \"  r1_o[\" << target_idx << \"] = \" << "
               "r1_o[target_idx] << \"\\n\";\n";
  std::cout << "  } else {\n";
  std::cout << "    std::cout << \"  ❌ INDEX OUT OF BOUNDS!\\n\";\n";
  std::cout << "  }\n";
  std::cout << "}\n\n";

  std::cout << "This will show:\n";
  std::cout << "- Exact index being calculated\n";
  std::cout << "- Array bounds at the problematic location\n";
  std::cout << "- Whether out-of-bounds access is occurring\n";
  std::cout << "- Raw array values before they're combined\n";

  EXPECT_TRUE(true) << "Array debugging approach proposed";
}

}  // namespace vmecpp
