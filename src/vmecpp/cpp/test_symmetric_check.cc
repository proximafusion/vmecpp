#include "vmecpp/common/vmec_indata/vmec_indata.h"
#include "vmecpp/vmec/vmec/vmec.h"
#include <iostream>

int main() {
    std::cout << "=== SYMMETRIC MODE TEST ===" << std::endl;
    
    // Simple symmetric tokamak (should work)
    vmecpp::VmecINDATA indata;
    indata.nfp = 1;
    indata.lasym = false;  // SYMMETRIC
    indata.mpol = 2;
    indata.ntor = 0;
    indata.ns_array = {5};
    indata.niter_array = {50};
    indata.ftol_array = {1e-08};
    indata.ntheta = 17;
    indata.nzeta = 1;
    
    indata.pres_scale = 0.0;
    indata.am = {0.0};
    indata.gamma = 0.0;
    indata.phiedge = 1.0;
    
    const int array_size = (indata.mpol + 1) * (2 * indata.ntor + 1);
    indata.rbc.resize(array_size, 0.0);
    indata.zbs.resize(array_size, 0.0);
    
    // Simple symmetric tokamak
    indata.rbc[0] = 1.0;   // R00 - major radius
    indata.rbc[1] = 0.3;   // R10 - minor radius
    indata.zbs[1] = 0.3;   // Z10 - elongation
    
    indata.raxis_c = {1.0};
    indata.zaxis_s = {0.0};
    
    std::cout << "Configuration: Symmetric tokamak" << std::endl;
    std::cout << "  lasym = " << indata.lasym << std::endl;
    std::cout << "  mpol = " << indata.mpol << ", ntor = " << indata.ntor << std::endl;
    
    try {
        const auto output = vmecpp::run(indata);
        if (output.ok()) {
            std::cout << "✅ SYMMETRIC MODE: PASSED" << std::endl;
            const auto& wout = output->wout;
            std::cout << "  Volume = " << wout.volume_p << std::endl;
            std::cout << "  Aspect ratio = " << wout.aspect << std::endl;
            return 0;
        } else {
            std::cout << "❌ SYMMETRIC MODE: FAILED" << std::endl;
            std::cout << "  Status: " << output.status() << std::endl;
            return 1;
        }
    } catch (const std::exception& e) {
        std::cout << "❌ SYMMETRIC MODE: EXCEPTION" << std::endl;
        std::cout << "  Error: " << e.what() << std::endl;
        return 1;
    }
}