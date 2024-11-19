#ifndef VMECPP_VMEC_IDEAL_MHD_MODEL_VECTORIZED_DFT_FUNCTIONS_H_
#define VMECPP_VMEC_IDEAL_MHD_MODEL_VECTORIZED_DFT_FUNCTIONS_H_

#include <immintrin.h>  // Intel AVX intrinscis

#include <vector>

#include "vmecpp/vmec/fourier_forces/fourier_forces.h"
#include "vmecpp/vmec/fourier_geometry/fourier_geometry.h"
#include "vmecpp/vmec/ideal_mhd_model/dft_data.h"
#include "vmecpp/vmec/radial_partitioning/radial_partitioning.h"

/*
Code for substitution of similarly named dft functions at runtime
in 'ideal_mhd_model' (without suffix *_avx):
cc_library(
  name = "vectorized_dft_functions",
  srcs = ["vectorized_dft_functions.cc"],
  hdrs = ["vectorized_dft_functions.h", "dft_data.h"], ... (see BUILD.bazel)

Code in this library is hand-optimized (i.e. manually vectorized)
for AVX256 and compiled for a specific architecture
(copts = ["-march=skylake-avx512"]) capable of AVX256 instructions. AVX512
instructions are *NOT* used anyway, since they are clockspeed-reduced on current
Intel processors, which may lead to even reduced speed at the end.
-march=skylake-avx512 is chosen nevertheless, as it includes some -mtune flags
which are beneficial for performance (e.g. register widths etc).
The dispatching either to the original functions in 'ideal_mhd_model', or the
manually tuned ones here with suffix '*_avx' takes place at runtime in
ideal_mhd_model, by employing the '__attribute__((target()))' directive of gcc
*/
namespace vmecpp {

// AVX 256
constexpr int VECTORSIZE = 4;

// Horizontal add, i.e. sum of 4 entries in an 4 x double mm256 register
// AVX512 has instrinsic functions for this and similar, but AVX 512 is not
//  used, as it is actually slower
inline double avx_hadd(__m256d& acc) {
  // horizontal add top lane and bottom lane
  acc = _mm256_hadd_pd(acc, acc);
  // add lanes
  acc = _mm256_add_pd(acc, _mm256_permute2f128_pd(acc, acc, 0x31));
  // extract double
  return _mm256_cvtsd_f64(acc);
}

// This is the avx-tuned version of vmecpp::ForcesToFourier3DSymmFastPoloidal
void ForcesToFourier3DSymmFastPoloidal_avx(
    const RealSpaceForces& d, const std::vector<double>& xmpq,
    const RadialPartitioning& rp, const FlowControl& fc, const Sizes& s,
    const FourierBasisFastPoloidal& fb, int ivac,
    FourierForces& physical_forces);

// This is the avx-tuned version of vmecpp::FourierToReal3DSymmFastPoloidal
void FourierToReal3DSymmFastPoloidal_avx(
    const FourierGeometry& physical_x, const std::vector<double>& xmpq,
    const RadialPartitioning& r, const Sizes& s, const RadialProfiles& rp,
    const FourierBasisFastPoloidal& fb, RealSpaceGeometry& g);

// This is the avx-tuned version of vmecpp::deAliasConstraintForce
void deAliasConstraintForce_avx(
    const vmecpp::RadialPartitioning& rp,
    const vmecpp::FourierBasisFastPoloidal& fb, const vmecpp::Sizes& s_,
    std::vector<double>& faccon, std::vector<double>& tcon,
    std::vector<double>& gConEff, std::vector<double>& gsc,
    std::vector<double>& gcs, std::vector<double>& gCon);

}  // namespace vmecpp

#endif  // VMECPP_VMEC_IDEAL_MHD_MODEL_VECTORIZED_DFT_FUNCTIONS_H_
