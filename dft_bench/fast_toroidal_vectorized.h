#ifndef VMECPP_TOOLS_DFT_BENCH_FAST_TOROIDAL_VECTORIZED_H_
#define VMECPP_TOOLS_DFT_BENCH_FAST_TOROIDAL_VECTORIZED_H_

#include "vmecpp/vmec/fourier_forces/fourier_forces.h"
#include "vmecpp_tools/dft_bench/ideal_mhd_data.h"
#include "vmecpp_tools/dft_bench/inputs.h"

namespace dft_bench::fast_toroidal_vectorized {
// Standalone version of IdealMHDModel::dft_ForcesToFourier_3d_symm
// Correponds to the implementation in main@70b2abe9.
void ForcesToFourier3D(vmecpp::FourierForces& m_physical_f,
                       const IdealMHDData& d, const Inputs& inputs);

// Standalone version of IdealMHDModel::dft_FourierToReal_3d_symm
// Correponds to the implementation in main@70b2abe9.
void FourierToReal3D(dft_bench::IdealMHDData& m_mhd_data, const Inputs& inputs);
}  // namespace dft_bench::fast_toroidal_vectorized

#endif  // VMECPP_TOOLS_DFT_BENCH_FAST_TOROIDAL_VECTORIZED_H_
