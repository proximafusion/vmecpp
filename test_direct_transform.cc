#include <iostream>
#include <vector>
#include <cmath>

// Minimal test to understand the transform issue
int main() {
  // Simple case: m=0,n=0 and m=1,n=0 only
  
  // Expected values:
  // R = 1.0*cos(0*theta) + 0.3*cos(1*theta) = 1.0 + 0.3*cos(theta)
  // At theta=0: R = 1.0 + 0.3 = 1.3
  
  // With normalization:
  // cosmu[m=0] = cos(0*theta) * mscale[0] = 1.0 * 1.0 = 1.0
  // cosmu[m=1] = cos(1*theta) * mscale[1] = cos(theta) * sqrt(2)
  // At theta=0: cosmu[m=1] = 1.0 * sqrt(2) = 1.414...
  
  // So if we use:
  // R = rmncc[0] * cosmu[0] + rmncc[1] * cosmu[1]
  // R = 1.0 * 1.0 + 0.3 * 1.414... = 1.0 + 0.424... = 1.424...
  
  std::cout << "Direct calculation:\n";
  std::cout << "cosmu[0] at theta=0 = " << 1.0 << "\n";
  std::cout << "cosmu[1] at theta=0 = " << std::sqrt(2.0) << "\n";
  std::cout << "R = 1.0 * " << 1.0 << " + 0.3 * " << std::sqrt(2.0) 
            << " = " << (1.0 + 0.3 * std::sqrt(2.0)) << "\n";
  
  std::cout << "\nThis matches VMEC++ output of 1.424264\n";
  
  std::cout << "\nThe issue: When using normalized basis functions,\n";
  std::cout << "the coefficients need to be pre-divided by the normalization.\n";
  std::cout << "So rmncc[1] should be 0.3/sqrt(2) = " << (0.3/std::sqrt(2.0)) << "\n";
  std::cout << "Then: R = 1.0 * 1.0 + (0.3/sqrt(2)) * sqrt(2) = 1.0 + 0.3 = 1.3\n";
  
  return 0;
}