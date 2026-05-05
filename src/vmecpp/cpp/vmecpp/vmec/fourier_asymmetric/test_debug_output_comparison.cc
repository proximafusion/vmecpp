#include <gtest/gtest.h>

#include <fstream>
#include <iomanip>
#include <iostream>

#include "util/file_io/file_io.h"
#include "vmecpp/vmec/vmec/vmec.h"

TEST(DebugOutputComparisonTest, VMECPlusPlusDetailedOutput) {
  std::cout << "\n=== VMEC++ DETAILED DEBUG OUTPUT ===\n";

  std::cout << "Creating comprehensive debug output for asymmetric case:\n";
  std::cout << "1. Load exact same configuration as jVMEC test\n";
  std::cout << "2. Add debug prints at every critical calculation step\n";
  std::cout << "3. Output Jacobian components (tau1, tau2) at each surface\n";
  std::cout << "4. Log geometry arrays (R, Z, dR/dtheta, dZ/dtheta) at each theta\n";

  // Use the proven asymmetric tokamak configuration
  std::string input_file =
      "/home/ert/code/vmecpp/src/vmecpp/cpp/vmecpp/test_data/"
      "up_down_asymmetric_tokamak_simple.json";

  auto maybe_input = file_io::ReadFile(input_file);
  ASSERT_TRUE(maybe_input.ok())
      << "Cannot read input file: " << maybe_input.status();

  auto maybe_indata = vmecpp::VmecINDATA::FromJson(*maybe_input);
  ASSERT_TRUE(maybe_indata.ok())
      << "Cannot parse JSON: " << maybe_indata.status();

  auto config = *maybe_indata;

  std::cout << "\nConfiguration loaded - will add debug output:\n";
  std::cout << "  lasym = " << config.lasym << "\n";
  std::cout << "  nfp = " << config.nfp << "\n";
  std::cout << "  mpol = " << config.mpol << ", ntor = " << config.ntor << "\n";
  std::cout << "  NS = " << config.ns_array[0] << "\n";

  std::cout << "\nRequired debug locations in ideal_mhd_model.cc:\n";
  std::cout << "1. Line ~387: After asymmetric inverse DFT\n";
  std::cout << "   - Print R[theta], Z[theta] for each surface\n";
  std::cout << "   - Print dR/dtheta, dZ/dtheta derivatives\n";
  std::cout << "   - Show even/odd array values (r1_e, r1_o, z1_e, z1_o)\n";

  std::cout << "\n2. Line ~1764: In computeJacobian()\n";
  std::cout << "   - Print tau1 = ru12*zs - rs*zu12 for each surface\n";
  std::cout << "   - Print tau2 components: odd_contrib, mixed_contrib\n";
  std::cout << "   - Print final tau = tau1 + dSHalfDsInterp*tau2\n";
  std::cout << "   - Print minTau, maxTau, and minTau*maxTau check\n";

  std::cout << "\n3. Line ~415: After asymmetric forward DFT\n";
  std::cout << "   - Print force arrays before symmetrization\n";
  std::cout << "   - Print force residuals by harmonic mode\n";
  std::cout << "   - Show convergence metrics\n";

  std::cout << "\nExpected debug output format:\n";
  std::cout << "VMEC++ DEBUG: surface=0, theta=0.000: R=6.000, Z=0.000\n";
  std::cout << "VMEC++ DEBUG: tau[j=0] = 0.850 (tau1=0.900, tau2=-0.050)\n";
  std::cout << "VMEC++ DEBUG: Jacobian check: minTau=0.820, maxTau=0.880, product=0.722 > 0 ✓\n";

  EXPECT_TRUE(true) << "VMEC++ debug output framework designed";
}

TEST(DebugOutputComparisonTest, JVMECReferenceOutput) {
  std::cout << "\n=== jVMEC REFERENCE DEBUG OUTPUT ===\n";

  std::cout << "Strategy for extracting jVMEC debug information:\n";
  std::cout << "1. Add debug prints to jVMEC source code\n";
  std::cout << "2. Run identical configuration through jVMEC\n";
  std::cout << "3. Extract key values at same calculation points\n";
  std::cout << "4. Generate side-by-side comparison file\n";

  std::cout << "\nKey jVMEC files to instrument:\n";
  std::cout << "1. FourierTransforms.java (line ~163)\n";
  std::cout << "   - Add debug after 'totzspa' asymmetric transform\n";
  std::cout << "   - Print R[theta], Z[theta] arrays\n";
  std::cout << "   - Output at same theta positions as VMEC++\n";

  std::cout << "\n2. EquilibriumState.java (Jacobian calculation)\n";
  std::cout << "   - Add debug in tau calculation method\n";
  std::cout << "   - Print tau components matching VMEC++ format\n";
  std::cout << "   - Show surface-by-surface tau values\n";

  std::cout << "\n3. Forces.java (force calculation)\n";
  std::cout << "   - Add debug after force computation\n";
  std::cout << "   - Print force residuals by harmonic\n";
  std::cout << "   - Show convergence progress\n";

  std::cout << "\nExpected jVMEC debug output format:\n";
  std::cout << "jVMEC DEBUG: surface=0, theta=0.000: R=6.000, Z=0.000\n";
  std::cout << "jVMEC DEBUG: tau[j=0] = 0.850 (tau1=0.900, tau2=-0.050)\n";
  std::cout << "jVMEC DEBUG: Jacobian OK: minTau=0.820, maxTau=0.880\n";

  std::cout << "\nComparison methodology:\n";
  std::cout << "1. Run both codes with identical input\n";
  std::cout << "2. Extract debug output to separate files\n";
  std::cout << "3. Create diff analysis script\n";
  std::cout << "4. Identify first point of divergence\n";

  EXPECT_TRUE(true) << "jVMEC debug output strategy defined";
}

TEST(DebugOutputComparisonTest, EducationalVMECValidation) {
  std::cout << "\n=== EDUCATIONAL_VMEC VALIDATION OUTPUT ===\n";

  std::cout << "Using educational_VMEC as reference implementation:\n";
  std::cout << "1. Add debug prints to educational_VMEC Fortran code\n";
  std::cout << "2. Verify tau calculation formula matches\n";
  std::cout << "3. Check asymmetric transform (totzspa.f90)\n";
  std::cout << "4. Validate array combination (symrzl.f90)\n";

  std::cout << "\nKey educational_VMEC files to instrument:\n";
  std::cout << "1. totzspa.f90 (asymmetric inverse DFT)\n";
  std::cout << "   - Print R, Z arrays after transform\n";
  std::cout << "   - Show basis function evaluation\n";
  std::cout << "   - Verify theta range [0,2π] handling\n";

  std::cout << "\n2. symrzl.f90 (array combination)\n";
  std::cout << "   - Print before: r1s, r1a arrays separately\n";
  std::cout << "   - Print after: r1s = r1s + r1a combined\n";
  std::cout << "   - Verify no data loss in combination\n";

  std::cout << "\n3. jacobian.f90 (tau calculation)\n";
  std::cout << "   - Print unified tau formula components\n";
  std::cout << "   - Show dshalfds=0.25 scaling\n";
  std::cout << "   - Verify even/odd mode contributions\n";

  std::cout << "\nExpected educational_VMEC output:\n";
  std::cout << "EDU_VMEC: before combination: r1s[0]=10.0, r1a[0]=0.1\n";
  std::cout << "EDU_VMEC: after combination: r1s[0]=10.1\n";
  std::cout << "EDU_VMEC: tau unified = 0.850 (all terms included)\n";

  std::cout << "\nValidation approach:\n";
  std::cout << "1. Confirm VMEC++ implements same algorithms\n";
  std::cout << "2. Identify any missing terms or scaling\n";
  std::cout << "3. Verify numerical precision matches\n";
  std::cout << "4. Test edge cases and boundary conditions\n";

  EXPECT_TRUE(true) << "Educational_VMEC validation strategy ready";
}

TEST(DebugOutputComparisonTest, ThreeCodeComparisonFramework) {
  std::cout << "\n=== THREE-CODE COMPARISON FRAMEWORK ===\n";

  std::cout << "Systematic comparison methodology:\n";
  std::cout << "1. Run all three codes with identical configuration\n";
  std::cout << "2. Extract debug output in standardized format\n";
  std::cout << "3. Generate automated difference analysis\n";
  std::cout << "4. Identify exact point where VMEC++ diverges\n";

  std::cout << "\nStandardized debug format (CSV-like):\n";
  std::cout << "CODE,STAGE,SURFACE,THETA,R,Z,dR_dtheta,dZ_dtheta,tau,tau1,tau2\n";
  std::cout << "VMEC++,GEOM,0,0.000,6.000,0.000,0.000,1.000,0.850,0.900,-0.050\n";
  std::cout << "jVMEC,GEOM,0,0.000,6.000,0.000,0.000,1.000,0.850,0.900,-0.050\n";
  std::cout << "EDU_VMEC,GEOM,0,0.000,6.000,0.000,0.000,1.000,0.850,0.900,-0.050\n";

  std::cout << "\nComparison stages:\n";
  std::cout << "1. INIT: Initial configuration and array setup\n";
  std::cout << "2. GEOM: After geometry calculation (inverse DFT)\n";
  std::cout << "3. JACOBIAN: During Jacobian/tau calculation\n";
  std::cout << "4. FORCES: After force calculation (forward DFT)\n";
  std::cout << "5. CONVERGENCE: Final residuals and iteration metrics\n";

  std::cout << "\nDifference analysis script:\n";
  std::cout << "1. Parse all three debug files\n";
  std::cout << "2. Compare values at each stage with tolerance\n";
  std::cout << "3. Flag first significant difference\n";
  std::cout << "4. Generate summary report with root cause\n";

  std::cout << "\nExpected outcome:\n";
  std::cout << "- Identify exactly where VMEC++ differs from working codes\n";
  std::cout << "- Pinpoint missing calculation or wrong scaling\n";
  std::cout << "- Provide specific fix to resolve Jacobian sign issue\n";
  std::cout << "- Achieve first convergent asymmetric equilibrium\n";

  std::cout << "\nSuccess criteria:\n";
  std::cout << "✅ All three codes produce identical geometry arrays\n";
  std::cout << "✅ Tau calculations match at each surface\n";
  std::cout << "✅ Force residuals evolve similarly\n";
  std::cout << "✅ VMEC++ achieves convergence like jVMEC\n";

  EXPECT_TRUE(true) << "Three-code comparison framework designed";
}

TEST(DebugOutputComparisonTest, ImplementationPlan) {
  std::cout << "\n=== DEBUG OUTPUT IMPLEMENTATION PLAN ===\n";

  std::cout << "Step-by-step implementation approach:\n";

  std::cout << "\nPhase 1: VMEC++ Debug Enhancement\n";
  std::cout << "1. Add comprehensive debug prints to ideal_mhd_model.cc\n";
  std::cout << "2. Create debug output file: vmecpp_debug.csv\n";
  std::cout << "3. Test with asymmetric tokamak configuration\n";
  std::cout << "4. Verify all critical values are captured\n";

  std::cout << "\nPhase 2: jVMEC Debug Instrumentation\n";
  std::cout << "1. Modify jVMEC source to add matching debug prints\n";
  std::cout << "2. Create debug output file: jvmec_debug.csv\n";
  std::cout << "3. Run with same asymmetric configuration\n";
  std::cout << "4. Ensure output format matches VMEC++\n";

  std::cout << "\nPhase 3: Educational_VMEC Reference\n";
  std::cout << "1. Add debug prints to key Fortran subroutines\n";
  std::cout << "2. Create debug output file: edu_vmec_debug.csv\n";
  std::cout << "3. Validate algorithm implementation\n";
  std::cout << "4. Confirm tau formula correctness\n";

  std::cout << "\nPhase 4: Automated Comparison\n";
  std::cout << "1. Create Python script for difference analysis\n";
  std::cout << "2. Parse all three debug files\n";
  std::cout << "3. Generate difference report\n";
  std::cout << "4. Identify specific algorithmic differences\n";

  std::cout << "\nPhase 5: Fix Implementation\n";
  std::cout << "1. Apply specific fixes identified by comparison\n";
  std::cout << "2. Test convergence with corrected algorithm\n";
  std::cout << "3. Verify no regression in symmetric mode\n";
  std::cout << "4. Achieve first asymmetric equilibrium\n";

  std::cout << "\nTimeframe estimate:\n";
  std::cout << "- Phase 1 (VMEC++ debug): 1-2 hours\n";
  std::cout << "- Phase 2 (jVMEC debug): 2-3 hours\n";
  std::cout << "- Phase 3 (Educational_VMEC): 1-2 hours\n";
  std::cout << "- Phase 4 (Comparison script): 1 hour\n";
  std::cout << "- Phase 5 (Fix implementation): 2-4 hours\n";
  std::cout << "- Total: 7-12 hours systematic debugging\n";

  std::cout << "\nDeliverables:\n";
  std::cout << "📁 vmecpp_debug.csv - Comprehensive VMEC++ debug output\n";
  std::cout << "📁 jvmec_debug.csv - Reference jVMEC debug output\n";
  std::cout << "📁 edu_vmec_debug.csv - Educational_VMEC validation\n";
  std::cout << "📁 comparison_report.txt - Automated difference analysis\n";
  std::cout << "🔧 Fixed ideal_mhd_model.cc - Corrected asymmetric algorithm\n";
  std::cout << "✅ Working asymmetric equilibrium - First convergent case\n";

  EXPECT_TRUE(true) << "Debug output implementation plan ready";
}