#include "benchmark/benchmark.h"
#include "vmecpp/vmec/fourier_forces/fourier_forces.h"
#include "vmecpp_tools/dft_bench/fast_poloidal.h"
#include "vmecpp_tools/dft_bench/fast_toroidal.h"
#include "vmecpp_tools/dft_bench/fast_toroidal_vectorized.h"
#include "vmecpp_tools/dft_bench/ideal_mhd_data.h"
#include "vmecpp_tools/dft_bench/inputs.h"

// A benchmark suite to test alternative implementations of
// IdealMHDModel::forcesToFourier and IdealMHDModel::geometryFromFourier -- two
// of the most computationally expensive functions in VMEC++. The benchmarks are
// single-thread: in a multi-thread run of VMEC++, each thread calls these
// methods independently, so we can benchmark the single-thread invocation in
// isolation.

static constexpr const char* config_path =
    "alphasim_core/test_data/vmecpp/cma.json";

static void BM_ForcesToFourier3DFastToroidal(benchmark::State& state) {
  const auto inputs = dft_bench::Inputs::FromConfig(config_path);
  const dft_bench::IdealMHDData mhd_data(inputs.sizes,
                                         inputs.radial_partitioning);

  vmecpp::FourierForces output_forces(
      &inputs.sizes, &inputs.radial_partitioning, inputs.flow_control.ns);

  for (auto _ : state) {
    dft_bench::fast_toroidal::ForcesToFourier3D(output_forces, mhd_data,
                                                inputs);
  }
}
BENCHMARK(BM_ForcesToFourier3DFastToroidal)->Iterations(1000);

static void BM_ForcesToFourier3DFastToroidalVectorized(
    benchmark::State& state) {
  const auto inputs = dft_bench::Inputs::FromConfig(config_path);
  const dft_bench::IdealMHDData mhd_data(inputs.sizes,
                                         inputs.radial_partitioning);

  vmecpp::FourierForces output_forces(
      &inputs.sizes, &inputs.radial_partitioning, inputs.flow_control.ns);

  for (auto _ : state) {
    dft_bench::fast_toroidal_vectorized::ForcesToFourier3D(output_forces,
                                                           mhd_data, inputs);
  }
}
BENCHMARK(BM_ForcesToFourier3DFastToroidalVectorized)->Iterations(1000);

static void BM_ForcesToFourier3DFastPoloidal(benchmark::State& state) {
  const auto inputs = dft_bench::Inputs::FromConfig(config_path);
  const dft_bench::IdealMHDData mhd_data(inputs.sizes,
                                         inputs.radial_partitioning);

  vmecpp::FourierForces output_forces(
      &inputs.sizes, &inputs.radial_partitioning, inputs.flow_control.ns);

  for (auto _ : state) {
    dft_bench::fast_poloidal::ForcesToFourier3D(output_forces, mhd_data,
                                                inputs);
  }
}
BENCHMARK(BM_ForcesToFourier3DFastPoloidal)->Iterations(1000);

static void BM_FourierToReal3DFastToroidalVectorized(benchmark::State& state) {
  const auto inputs = dft_bench::Inputs::FromConfig(config_path);
  dft_bench::IdealMHDData mhd_data(inputs.sizes, inputs.radial_partitioning);

  for (auto _ : state) {
    dft_bench::fast_toroidal_vectorized::FourierToReal3D(mhd_data, inputs);
  }
}
BENCHMARK(BM_FourierToReal3DFastToroidalVectorized)->Iterations(1000);

static void BM_FourierToReal3DFastPoloidal(benchmark::State& state) {
  const auto inputs = dft_bench::Inputs::FromConfig(config_path);
  dft_bench::IdealMHDData mhd_data(inputs.sizes, inputs.radial_partitioning);

  for (auto _ : state) {
    dft_bench::fast_poloidal::FourierToReal3D(mhd_data, inputs);
  }
}
BENCHMARK(BM_FourierToReal3DFastPoloidal)->Iterations(1000);

BENCHMARK_MAIN();
