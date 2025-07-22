#include <iostream>
#include <string>
#include "vmecpp/common/vmec_indata/vmec_indata.h"
#include "vmecpp/vmec/vmec_checkpoint.h"
#include "vmecpp/run.h"

int main() {
    std::cout << "Testing Solovev equilibrium convergence after asymmetric fix..." << std::endl;
    
    // Create and read input
    vmecpp::VmecINDATA indata;
    std::string input_file = "src/vmecpp/cpp/vmecpp/test_data/input.solovev";
    
    auto status = indata.FromNamelist(input_file);
    if (!status.ok()) {
        std::cerr << "Failed to read input file: " << status.message() << std::endl;
        return 1;
    }
    
    std::cout << "Running VMEC with lasym = " << (indata.lasym ? "T" : "F") << std::endl;
    
    // Run VMEC
    auto result = vmecpp::Run(indata);
    if (!result.ok()) {
        std::cerr << "VMEC run failed: " << result.status().message() << std::endl;
        return 1;
    }
    
    auto& output = *result;
    std::cout << "VMEC completed:" << std::endl;
    std::cout << "  fsqr = " << output.output_quantities.fsqr << std::endl;
    std::cout << "  iter = " << output.output_quantities.iter << std::endl;
    std::cout << "  MHD Energy = " << output.output_quantities.wdot << std::endl;
    
    if (output.output_quantities.fsqr < 1e-14) {
        std::cout << "SUCCESS: Convergence achieved!" << std::endl;
        return 0;
    } else {
        std::cout << "FAILED: Did not converge to required tolerance" << std::endl;
        return 1;
    }
}