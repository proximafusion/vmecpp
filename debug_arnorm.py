#!/usr/bin/env python3
"""Debug arNorm=0 issue in symmetric case."""

import os
import sys

# Create a simple test case that should populate ru/zu
test_code = '''
#include <iostream>
#include <vector>
#include "vmecpp/common/sizes/sizes.h"
#include "vmecpp/vmec/ideal_mhd_model/ideal_mhd_model.h"

int main() {
    // Create a minimal symmetric configuration
    vmecpp::Sizes sizes(false, 1, 3, 0, 8, 4);  // lasym=false
    
    std::cout << "Sizes created:" << std::endl;
    std::cout << "  lasym = " << sizes.lasym << std::endl;
    std::cout << "  mpol = " << sizes.mpol << std::endl;
    std::cout << "  ntor = " << sizes.ntor << std::endl;
    std::cout << "  nThetaEff = " << sizes.nThetaEff << std::endl;
    std::cout << "  nZeta = " << sizes.nZeta << std::endl;
    std::cout << "  nZnT = " << sizes.nZnT << std::endl;
    
    return 0;
}
'''

# Write test file
with open("test_arnorm_debug.cc", "w") as f:
    f.write(test_code)

print("Created test_arnorm_debug.cc")
print("To compile and run:")
print("  bazel build //vmecpp/common/sizes:sizes")
print("  g++ -I. -I/path/to/bazel-bin test_arnorm_debug.cc")