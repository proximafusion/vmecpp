#!/usr/bin/env python3
"""Add debug output to track arNorm=0 issue."""

import re

# Read the file
with open("src/vmecpp/cpp/vmecpp/vmec/ideal_mhd_model/ideal_mhd_model.cc", "r") as f:
    content = f.read()

# Find the constraintForceMultiplier function and add debug
pattern = r'(for \(int jF = std::max\(jMin, r_\.nsMinF\); jF < r_\.nsMaxF; \+\+jF\) \{\s*\n\s*double arNorm = 0\.0;\s*\n\s*double azNorm = 0\.0;)'

replacement = r'''\1
    // DEBUG: Check if ruFull/zuFull are populated
    if (jF == std::max(jMin, r_.nsMinF)) {
      std::cout << "DEBUG constraintForceMultiplier: jF=" << jF 
                << ", r_.nsMinF=" << r_.nsMinF 
                << ", r_.nsMaxF=" << r_.nsMaxF 
                << ", s_.nZnT=" << s_.nZnT << std::endl;
      for (int kl = 0; kl < std::min(5, s_.nZnT); ++kl) {
        int idx_kl = (jF - r_.nsMinF) * s_.nZnT + kl;
        std::cout << "  kl=" << kl << ", idx=" << idx_kl 
                  << ", ruFull=" << ruFull[idx_kl] 
                  << ", zuFull=" << zuFull[idx_kl] << std::endl;
      }
    }'''

content = re.sub(pattern, replacement, content, flags=re.MULTILINE)

# Write back
with open("src/vmecpp/cpp/vmecpp/vmec/ideal_mhd_model/ideal_mhd_model.cc", "w") as f:
    f.write(content)

print("Added debug output to constraintForceMultiplier")