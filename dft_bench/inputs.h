#ifndef VMECPP_TOOLS_DFT_BENCH_INPUTS_H_
#define VMECPP_TOOLS_DFT_BENCH_INPUTS_H_

#include <filesystem>

#include "vmecpp/common/flow_control/flow_control.h"
#include "vmecpp/common/fourier_basis_fast_poloidal/fourier_basis_fast_poloidal.h"
#include "vmecpp/common/sizes/sizes.h"
#include "vmecpp/vmec/fourier_geometry/fourier_geometry.h"
#include "vmecpp/vmec/radial_partitioning/radial_partitioning.h"
#include "vmecpp/vmec/radial_profiles/radial_profiles.h"

namespace dft_bench {

struct Inputs {
  vmecpp::RadialPartitioning radial_partitioning;
  vmecpp::Sizes sizes;
  vmecpp::FourierBasisFastPoloidal fourier_basis;
  vmecpp::FlowControl flow_control;
  vmecpp::RadialProfiles radial_profiles;
  vmecpp::FourierGeometry fourier_geometry;

  // Create an Inputs instance with data members consistent with the specified
  // configuration file. Note that the vectors in Inputs::ideal_mhd_data will
  // have the correct sizes but they will only be zero-initialized.
  static Inputs FromConfig(const std::filesystem::path &config);
};

}  // namespace dft_bench

#endif  //  VMECPP_TOOLS_DFT_BENCH_INPUTS_H_
