#ifndef VMECPP_TOOLS_DFT_BENCH_FAST_POLOIDAL_H_
#define VMECPP_TOOLS_DFT_BENCH_FAST_POLOIDAL_H_

#include "vmecpp/vmec/fourier_forces/fourier_forces.h"
#include "vmecpp_tools/dft_bench/ideal_mhd_data.h"
#include "vmecpp_tools/dft_bench/inputs.h"

namespace dft_bench::fast_poloidal {

void ForcesToFourier3D(vmecpp::FourierForces& m_physical_f,
                       const IdealMHDData& d, const Inputs& inputs);

void FourierToReal3D(IdealMHDData& d, const Inputs& inputs);

}  // namespace dft_bench::fast_poloidal

#endif  // VMECPP_TOOLS_DFT_BENCH_FAST_POLOIDAL_H_
