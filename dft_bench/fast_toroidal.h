#ifndef VMECPP_TOOLS_DFT_BENCH_FAST_TOROIDAL_H_
#define VMECPP_TOOLS_DFT_BENCH_FAST_TOROIDAL_H_

#include "vmecpp/vmec/fourier_forces/fourier_forces.h"
#include "vmecpp_tools/dft_bench/ideal_mhd_data.h"
#include "vmecpp_tools/dft_bench/inputs.h"

namespace dft_bench::fast_toroidal {

// A version of ForcesToFourier3D precedent to low-level optimizations
// geared towards getting compiler auto-vectorization.
// Used as a baseline to gauge the impact of vectorization.
// Corresponds to IdealMHDModel::dft_ForcesToFourier_symm from
// main@3b3790d86dcb849e4ab3c483481db29d952fecfd.
void ForcesToFourier3D(vmecpp::FourierForces& m_physical_f,
                       const IdealMHDData& d, const Inputs& inputs);

}  // namespace dft_bench::fast_toroidal

#endif  //  VMECPP_TOOLS_DFT_BENCH_FAST_TOROIDAL_H_
