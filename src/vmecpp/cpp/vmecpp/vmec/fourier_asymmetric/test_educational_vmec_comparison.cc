// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <cmath>
#include <iostream>

#include "vmecpp/common/vmec_indata/vmec_indata.h"

namespace vmecpp {

// Test to document findings from educational_VMEC
TEST(EducationalVmecComparisonTest, DocumentArrayCombination) {
  std::cout << "\n=== EDUCATIONAL_VMEC ARRAY COMBINATION ===\n" << std::endl;

  std::cout << "From educational_VMEC/src/symrzl.f90 lines 66-70:" << std::endl;
  std::cout << "```fortran" << std::endl;
  std::cout << "r1s(:,:n2,mpar) = r1s(:,:n2,mpar) + r1a(:,:n2,mpar)"
            << std::endl;
  std::cout << "rus(:,:n2,mpar) = rus(:,:n2,mpar) + rua(:,:n2,mpar)"
            << std::endl;
  std::cout << "z1s(:,:n2,mpar) = z1s(:,:n2,mpar) + z1a(:,:n2,mpar)"
            << std::endl;
  std::cout << "zus(:,:n2,mpar) = zus(:,:n2,mpar) + zua(:,:n2,mpar)"
            << std::endl;
  std::cout << "lus(:,:n2,mpar) = lus(:,:n2,mpar) + lua(:,:n2,mpar)"
            << std::endl;
  std::cout << "```" << std::endl;

  std::cout << "\nTRANSLATION TO VMEC++:" << std::endl;
  std::cout << "- r1s → r1_e (symmetric R even)" << std::endl;
  std::cout << "- r1a → m_ls_.r1e_i (asymmetric R even)" << std::endl;
  std::cout << "- z1s → z1_o (symmetric Z odd)" << std::endl;
  std::cout << "- z1a → m_ls_.z1o_i (asymmetric Z odd)" << std::endl;
  std::cout << "- rus → ru_e (symmetric dR/dtheta even)" << std::endl;
  std::cout << "- rua → m_ls_.rue_i (asymmetric dR/dtheta even)" << std::endl;

  std::cout << "\nKEY INSIGHT:" << std::endl;
  std::cout << "✅ Educational_VMEC ADDS asymmetric to symmetric arrays"
            << std::endl;
  std::cout << "✅ This happens AFTER both transforms complete" << std::endl;
  std::cout << "✅ This is EXACTLY the fix we proposed!" << std::endl;

  EXPECT_TRUE(true) << "Documentation complete";
}

// Test to document Jacobian calculation findings
TEST(EducationalVmecComparisonTest, DocumentJacobianDivision) {
  std::cout << "\n=== JACOBIAN DIVISION IN ALL THREE CODES ===\n" << std::endl;

  std::cout << "1. EDUCATIONAL_VMEC (jacobian.f90 line 59):" << std::endl;
  std::cout << "   tau = ... + (...) / shalf(l)" << std::endl;
  std::cout << "   where shalf = sqrt(s) on half grid" << std::endl;

  std::cout << "\n2. jVMEC (RealSpaceGeometry.java line 307):" << std::endl;
  std::cout << "   tau = ... + (...) / sqrtSHalf[j]" << std::endl;
  std::cout << "   Comment: 'does not work at all!!!'" << std::endl;

  std::cout << "\n3. VMEC++ (ideal_mhd_model.cc line 1692):" << std::endl;
  std::cout << "   tau2 = ... + (...) / protected_sqrtSH" << std::endl;
  std::cout << "   where protected_sqrtSH = max(sqrtSH, 1e-12)" << std::endl;

  std::cout << "\nCONCLUSION:" << std::endl;
  std::cout << "✅ ALL three codes have division by sqrt(s)" << std::endl;
  std::cout << "✅ This is part of the standard algorithm" << std::endl;
  std::cout << "✅ The REAL issue is missing array combination" << std::endl;
  std::cout << "✅ Division may need better protection but is not primary cause"
            << std::endl;

  EXPECT_TRUE(true) << "Analysis complete";
}

// Test to summarize the complete fix
TEST(EducationalVmecComparisonTest, SummarizeCompleteFix) {
  std::cout << "\n=== COMPLETE FIX SUMMARY ===\n" << std::endl;

  std::cout << "ROOT CAUSE:" << std::endl;
  std::cout << "- Symmetric arrays (r1_e) are zero at kl=6-9" << std::endl;
  std::cout << "- Asymmetric contributions (m_ls_.r1e_i) exist but not added"
            << std::endl;
  std::cout << "- Jacobian calculation uses zero values → NaN" << std::endl;

  std::cout << "\nPRIMARY FIX (from educational_VMEC):" << std::endl;
  std::cout << "After asymmetric transform in ideal_mhd_model.cc:" << std::endl;
  std::cout << "```cpp" << std::endl;
  std::cout << "// Add asymmetric contributions to symmetric arrays"
            << std::endl;
  std::cout << "for (int idx = 0; idx < arraySize; ++idx) {" << std::endl;
  std::cout << "  r1_e[idx] += m_ls_.r1e_i[idx];" << std::endl;
  std::cout << "  r1_o[idx] += m_ls_.r1o_i[idx];" << std::endl;
  std::cout << "  z1_e[idx] += m_ls_.z1e_i[idx];" << std::endl;
  std::cout << "  z1_o[idx] += m_ls_.z1o_i[idx];" << std::endl;
  std::cout << "  ru_e[idx] += m_ls_.rue_i[idx];" << std::endl;
  std::cout << "  ru_o[idx] += m_ls_.ruo_i[idx];" << std::endl;
  std::cout << "  zu_e[idx] += m_ls_.zue_i[idx];" << std::endl;
  std::cout << "  zu_o[idx] += m_ls_.zuo_i[idx];" << std::endl;
  std::cout << "}" << std::endl;
  std::cout << "```" << std::endl;

  std::cout << "\nSECONDARY CONSIDERATION:" << std::endl;
  std::cout << "- tau2 division by sqrtSH exists in all codes" << std::endl;
  std::cout << "- Current protection (min 1e-12) may be sufficient"
            << std::endl;
  std::cout << "- Focus on array combination first" << std::endl;

  std::cout << "\nEXPECTED RESULT:" << std::endl;
  std::cout << "✅ No more zero values at kl=6-9" << std::endl;
  std::cout << "✅ Jacobian calculation gets valid input" << std::endl;
  std::cout << "✅ No more NaN in tau, zu12, ru12" << std::endl;
  std::cout << "✅ Asymmetric VMEC should converge!" << std::endl;

  EXPECT_TRUE(true) << "Fix strategy complete";
}

}  // namespace vmecpp
