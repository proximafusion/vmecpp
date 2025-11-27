// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT

// Benchmark tool for comparing compute backend performance.
//
// Usage:
//   backend_benchmark [options]
//
// Options:
//   --ns=<int>        Number of radial surfaces (default: 50)
//   --mpol=<int>      Number of poloidal modes (default: 12)
//   --ntor=<int>      Number of toroidal modes (default: 12)
//   --nzeta=<int>     Number of toroidal grid points (default: 36)
//   --ntheta=<int>    Number of poloidal grid points (default: 36)
//   --iterations=<int> Number of benchmark iterations (default: 100)
//   --warmup=<int>    Number of warmup iterations (default: 10)
//
// The benchmark measures the time for:
//   1. FourierToReal (inverse DFT)
//   2. ForcesToFourier (forward DFT)

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "vmecpp/common/compute_backend/compute_backend.h"
#include "vmecpp/common/compute_backend/compute_backend_cpu.h"
#include "vmecpp/common/compute_backend/compute_backend_factory.h"
#include "vmecpp/common/flow_control/flow_control.h"
#include "vmecpp/common/fourier_basis_fast_poloidal/fourier_basis_fast_poloidal.h"
#include "vmecpp/common/sizes/sizes.h"
#include "vmecpp/vmec/fourier_forces/fourier_forces.h"
#include "vmecpp/vmec/fourier_geometry/fourier_geometry.h"
#include "vmecpp/vmec/ideal_mhd_model/dft_data.h"
#include "vmecpp/vmec/radial_partitioning/radial_partitioning.h"
#include "vmecpp/vmec/radial_profiles/radial_profiles.h"

namespace {

struct BenchmarkConfig {
  int ns = 50;
  int mpol = 12;
  int ntor = 12;
  int nzeta = 36;
  int ntheta = 36;
  int iterations = 100;
  int warmup = 10;
  int nfp = 5;  // Number of field periods (typical stellarator)
};

struct BenchmarkResult {
  std::string backend_name;
  double fourier_to_real_mean_us;
  double fourier_to_real_std_us;
  double forces_to_fourier_mean_us;
  double forces_to_fourier_std_us;
  double total_mean_us;
  bool available;
};

// Parse command line arguments.
BenchmarkConfig ParseArgs(int argc, char* argv[]) {
  BenchmarkConfig config;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg.rfind("--ns=", 0) == 0) {
      config.ns = std::stoi(arg.substr(5));
    } else if (arg.rfind("--mpol=", 0) == 0) {
      config.mpol = std::stoi(arg.substr(7));
    } else if (arg.rfind("--ntor=", 0) == 0) {
      config.ntor = std::stoi(arg.substr(7));
    } else if (arg.rfind("--nzeta=", 0) == 0) {
      config.nzeta = std::stoi(arg.substr(8));
    } else if (arg.rfind("--ntheta=", 0) == 0) {
      config.ntheta = std::stoi(arg.substr(9));
    } else if (arg.rfind("--iterations=", 0) == 0) {
      config.iterations = std::stoi(arg.substr(13));
    } else if (arg.rfind("--warmup=", 0) == 0) {
      config.warmup = std::stoi(arg.substr(9));
    } else if (arg == "--help" || arg == "-h") {
      std::cout << "Usage: backend_benchmark [options]\n"
                << "\n"
                << "Options:\n"
                << "  --ns=<int>         Number of radial surfaces (default: "
                   "50)\n"
                << "  --mpol=<int>       Number of poloidal modes (default: "
                   "12)\n"
                << "  --ntor=<int>       Number of toroidal modes (default: "
                   "12)\n"
                << "  --nzeta=<int>      Number of toroidal grid points "
                   "(default: 36)\n"
                << "  --ntheta=<int>     Number of poloidal grid points "
                   "(default: 36)\n"
                << "  --iterations=<int> Number of benchmark iterations "
                   "(default: 100)\n"
                << "  --warmup=<int>     Number of warmup iterations (default: "
                   "10)\n";
      std::exit(0);
    }
  }

  return config;
}

// Calculate mean and standard deviation.
std::pair<double, double> CalcStats(const std::vector<double>& times) {
  if (times.empty()) return {0.0, 0.0};

  double sum = std::accumulate(times.begin(), times.end(), 0.0);
  double mean = sum / static_cast<double>(times.size());

  double sq_sum = 0.0;
  for (double t : times) {
    sq_sum += (t - mean) * (t - mean);
  }
  double std_dev = std::sqrt(sq_sum / static_cast<double>(times.size()));

  return {mean, std_dev};
}

// Initialize test data with random values.
void InitializeTestData(vmecpp::FourierGeometry& geometry,
                        vmecpp::FourierForces& forces,
                        std::vector<double>& xmpq, const vmecpp::Sizes& s,
                        const vmecpp::RadialPartitioning& rp) {
  std::mt19937 rng(42);  // Fixed seed for reproducibility
  std::uniform_real_distribution<double> dist(-1.0, 1.0);

  // Initialize Fourier coefficients.
  int coeff_size = (rp.nsMaxF1 - rp.nsMinF1) * s.mpol * (s.ntor + 1);
  geometry.rmncc.resize(coeff_size);
  geometry.rmnss.resize(coeff_size);
  geometry.zmnsc.resize(coeff_size);
  geometry.zmncs.resize(coeff_size);
  geometry.lmnsc.resize(coeff_size);
  geometry.lmncs.resize(coeff_size);

  for (int i = 0; i < coeff_size; ++i) {
    geometry.rmncc[i] = dist(rng) * 0.1;
    geometry.rmnss[i] = dist(rng) * 0.1;
    geometry.zmnsc[i] = dist(rng) * 0.1;
    geometry.zmncs[i] = dist(rng) * 0.1;
    geometry.lmnsc[i] = dist(rng) * 0.01;
    geometry.lmncs[i] = dist(rng) * 0.01;
  }

  // Set major radius baseline.
  for (int j = 0; j < rp.nsMaxF1 - rp.nsMinF1; ++j) {
    geometry.rmncc[j * s.mpol * (s.ntor + 1)] = 5.5;  // Major radius
  }

  // Initialize force output arrays.
  forces.frcc.resize(coeff_size);
  forces.frss.resize(coeff_size);
  forces.fzsc.resize(coeff_size);
  forces.fzcs.resize(coeff_size);
  forces.flsc.resize(coeff_size);
  forces.flcs.resize(coeff_size);

  // Initialize xmpq.
  xmpq.resize(s.mpol);
  for (int m = 0; m < s.mpol; ++m) {
    xmpq[m] = m * (m - 1);
  }
}

// Run benchmark for a single backend.
BenchmarkResult RunBenchmark(vmecpp::ComputeBackend* backend,
                             const BenchmarkConfig& config,
                             const vmecpp::Sizes& s,
                             const vmecpp::FourierBasisFastPoloidal& fb,
                             const vmecpp::RadialPartitioning& rp,
                             const vmecpp::RadialProfiles& profiles,
                             const vmecpp::FlowControl& fc) {
  BenchmarkResult result;
  result.backend_name = backend->GetName();
  result.available = backend->IsAvailable();

  if (!result.available) {
    result.fourier_to_real_mean_us = 0;
    result.fourier_to_real_std_us = 0;
    result.forces_to_fourier_mean_us = 0;
    result.forces_to_fourier_std_us = 0;
    result.total_mean_us = 0;
    return result;
  }

  // Initialize test data.
  vmecpp::FourierGeometry geometry;
  vmecpp::FourierForces forces;
  std::vector<double> xmpq;
  InitializeTestData(geometry, forces, xmpq, s, rp);

  // Allocate real-space geometry output.
  int grid_size = (rp.nsMaxF1 - rp.nsMinF1) * s.nZeta * s.nThetaEff;
  int con_size = (rp.nsMaxFIncludingLcfs - rp.nsMinF) * s.nZeta * s.nThetaEff;

  std::vector<double> r1_e(grid_size), r1_o(grid_size);
  std::vector<double> ru_e(grid_size), ru_o(grid_size);
  std::vector<double> rv_e(grid_size), rv_o(grid_size);
  std::vector<double> z1_e(grid_size), z1_o(grid_size);
  std::vector<double> zu_e(grid_size), zu_o(grid_size);
  std::vector<double> zv_e(grid_size), zv_o(grid_size);
  std::vector<double> lu_e(grid_size), lu_o(grid_size);
  std::vector<double> lv_e(grid_size), lv_o(grid_size);
  std::vector<double> r_con(con_size), z_con(con_size);

  vmecpp::RealSpaceGeometry real_geom{r1_e, r1_o, ru_e, ru_o, rv_e, rv_o,
                                      z1_e, z1_o, zu_e, zu_o, zv_e, zv_o,
                                      lu_e, lu_o, lv_e, lv_o, r_con, z_con};

  // Allocate real-space forces input (dummy data for benchmark).
  int force_grid_size = (rp.nsMaxF - rp.nsMinF) * s.nZeta * s.nThetaEff;
  std::vector<double> armn_e(force_grid_size), armn_o(force_grid_size);
  std::vector<double> azmn_e(force_grid_size), azmn_o(force_grid_size);
  std::vector<double> blmn_e(force_grid_size), blmn_o(force_grid_size);
  std::vector<double> brmn_e(force_grid_size), brmn_o(force_grid_size);
  std::vector<double> bzmn_e(force_grid_size), bzmn_o(force_grid_size);
  std::vector<double> clmn_e(force_grid_size), clmn_o(force_grid_size);
  std::vector<double> crmn_e(force_grid_size), crmn_o(force_grid_size);
  std::vector<double> czmn_e(force_grid_size), czmn_o(force_grid_size);
  std::vector<double> frcon_e(force_grid_size), frcon_o(force_grid_size);
  std::vector<double> fzcon_e(force_grid_size), fzcon_o(force_grid_size);

  // Fill with small random values.
  std::mt19937 rng(42);
  std::uniform_real_distribution<double> dist(-0.01, 0.01);
  for (int i = 0; i < force_grid_size; ++i) {
    armn_e[i] = dist(rng);
    armn_o[i] = dist(rng);
    azmn_e[i] = dist(rng);
    azmn_o[i] = dist(rng);
    blmn_e[i] = dist(rng);
    blmn_o[i] = dist(rng);
    brmn_e[i] = dist(rng);
    brmn_o[i] = dist(rng);
    bzmn_e[i] = dist(rng);
    bzmn_o[i] = dist(rng);
    clmn_e[i] = dist(rng);
    clmn_o[i] = dist(rng);
    crmn_e[i] = dist(rng);
    crmn_o[i] = dist(rng);
    czmn_e[i] = dist(rng);
    czmn_o[i] = dist(rng);
  }

  vmecpp::RealSpaceForces real_forces{armn_e, armn_o, azmn_e, azmn_o,
                                      blmn_e, blmn_o, brmn_e, brmn_o,
                                      bzmn_e, bzmn_o, clmn_e, clmn_o,
                                      crmn_e, crmn_o, czmn_e, czmn_o,
                                      frcon_e, frcon_o, fzcon_e, fzcon_o};

  // Warmup iterations.
  for (int i = 0; i < config.warmup; ++i) {
    backend->FourierToReal(geometry, xmpq, rp, s, profiles, fb, real_geom);
    backend->ForcesToFourier(real_forces, xmpq, rp, fc, s, fb,
                             vmecpp::VacuumPressureState::kOff, forces);
  }
  backend->Synchronize();

  // Benchmark FourierToReal.
  std::vector<double> f2r_times;
  f2r_times.reserve(config.iterations);
  for (int i = 0; i < config.iterations; ++i) {
    auto start = std::chrono::high_resolution_clock::now();
    backend->FourierToReal(geometry, xmpq, rp, s, profiles, fb, real_geom);
    backend->Synchronize();
    auto end = std::chrono::high_resolution_clock::now();
    double us =
        std::chrono::duration<double, std::micro>(end - start).count();
    f2r_times.push_back(us);
  }

  // Benchmark ForcesToFourier.
  std::vector<double> f2f_times;
  f2f_times.reserve(config.iterations);
  for (int i = 0; i < config.iterations; ++i) {
    auto start = std::chrono::high_resolution_clock::now();
    backend->ForcesToFourier(real_forces, xmpq, rp, fc, s, fb,
                             vmecpp::VacuumPressureState::kOff, forces);
    backend->Synchronize();
    auto end = std::chrono::high_resolution_clock::now();
    double us =
        std::chrono::duration<double, std::micro>(end - start).count();
    f2f_times.push_back(us);
  }

  // Calculate statistics.
  auto [f2r_mean, f2r_std] = CalcStats(f2r_times);
  auto [f2f_mean, f2f_std] = CalcStats(f2f_times);

  result.fourier_to_real_mean_us = f2r_mean;
  result.fourier_to_real_std_us = f2r_std;
  result.forces_to_fourier_mean_us = f2f_mean;
  result.forces_to_fourier_std_us = f2f_std;
  result.total_mean_us = f2r_mean + f2f_mean;

  return result;
}

void PrintResults(const std::vector<BenchmarkResult>& results,
                  const BenchmarkConfig& config) {
  std::cout << "\n";
  std::cout << "========================================\n";
  std::cout << "  VMEC++ Compute Backend Benchmark\n";
  std::cout << "========================================\n";
  std::cout << "\n";
  std::cout << "Configuration:\n";
  std::cout << "  Radial surfaces (ns):     " << config.ns << "\n";
  std::cout << "  Poloidal modes (mpol):    " << config.mpol << "\n";
  std::cout << "  Toroidal modes (ntor):    " << config.ntor << "\n";
  std::cout << "  Toroidal grid (nzeta):    " << config.nzeta << "\n";
  std::cout << "  Poloidal grid (ntheta):   " << config.ntheta << "\n";
  std::cout << "  Field periods (nfp):      " << config.nfp << "\n";
  std::cout << "  Iterations:               " << config.iterations << "\n";
  std::cout << "  Warmup iterations:        " << config.warmup << "\n";
  std::cout << "\n";

  // Estimate total operations.
  int64_t grid_points =
      static_cast<int64_t>(config.ns) * config.nzeta * config.ntheta;
  int64_t modes = static_cast<int64_t>(config.mpol) * (config.ntor + 1);
  std::cout << "Problem size:\n";
  std::cout << "  Total grid points:        " << grid_points << "\n";
  std::cout << "  Total Fourier modes:      " << modes << "\n";
  std::cout << "  Grid * modes:             " << grid_points * modes << "\n";
  std::cout << "\n";

  std::cout << "Results (times in microseconds):\n";
  std::cout << "\n";
  std::cout << std::left << std::setw(20) << "Backend" << std::right
            << std::setw(15) << "FourierToReal" << std::setw(15)
            << "ForcesToFourier" << std::setw(15) << "Total"
            << std::setw(12) << "Status" << "\n";
  std::cout << std::string(77, '-') << "\n";

  const BenchmarkResult* cpu_result = nullptr;
  for (const auto& r : results) {
    if (r.backend_name.find("CPU") != std::string::npos) {
      cpu_result = &r;
      break;
    }
  }

  for (const auto& r : results) {
    std::cout << std::left << std::setw(20) << r.backend_name;

    if (!r.available) {
      std::cout << std::right << std::setw(15) << "-" << std::setw(15) << "-"
                << std::setw(15) << "-" << std::setw(12) << "N/A"
                << "\n";
      continue;
    }

    std::cout << std::right << std::fixed << std::setprecision(1)
              << std::setw(15) << r.fourier_to_real_mean_us << std::setw(15)
              << r.forces_to_fourier_mean_us << std::setw(15) << r.total_mean_us
              << std::setw(12) << "OK"
              << "\n";
  }

  // Print speedup if both CPU and CUDA are available.
  if (cpu_result != nullptr && cpu_result->available) {
    std::cout << "\n";
    std::cout << "Speedup vs CPU:\n";
    for (const auto& r : results) {
      if (!r.available || r.backend_name.find("CPU") != std::string::npos) {
        continue;
      }
      double speedup = cpu_result->total_mean_us / r.total_mean_us;
      std::cout << "  " << r.backend_name << ": " << std::fixed
                << std::setprecision(2) << speedup << "x\n";
    }
  }

  std::cout << "\n";
}

}  // namespace

int main(int argc, char* argv[]) {
  BenchmarkConfig config = ParseArgs(argc, argv);

  // Create mock Sizes object.
  vmecpp::Sizes s;
  s.ns = config.ns;
  s.mpol = config.mpol;
  s.ntor = config.ntor;
  s.nZeta = config.nzeta;
  s.nThetaEven = config.ntheta;
  s.nThetaReduced = config.ntheta / 2 + 1;
  s.nThetaEff = s.nThetaReduced;
  s.nZnT = s.nZeta * s.nThetaEff;
  s.nnyq2 = config.ntor;
  s.mnyq2 = config.mpol;
  s.mnsize = config.mpol * (config.ntor + 1);
  s.nfp = config.nfp;
  s.lthreed = true;
  s.lasym = false;

  // Create Fourier basis.
  vmecpp::FourierBasisFastPoloidal fb(&s);

  // Create radial partitioning (single thread, full domain).
  vmecpp::RadialPartitioning rp;
  rp.nsMinF = 0;
  rp.nsMaxF = config.ns;
  rp.nsMinF1 = 0;
  rp.nsMaxF1 = config.ns;
  rp.nsMinH = 0;
  rp.nsMaxH = config.ns - 1;
  rp.nsMaxFIncludingLcfs = config.ns;

  // Create radial profiles.
  vmecpp::RadialProfiles profiles;
  profiles.sqrtSF.resize(config.ns);
  profiles.sqrtSH.resize(config.ns);
  for (int j = 0; j < config.ns; ++j) {
    double s_val = static_cast<double>(j) / static_cast<double>(config.ns - 1);
    profiles.sqrtSF[j] = std::sqrt(s_val);
    profiles.sqrtSH[j] = std::sqrt(s_val + 0.5 / static_cast<double>(config.ns - 1));
  }

  // Create flow control.
  vmecpp::FlowControl fc;
  fc.ns = config.ns;
  fc.lfreeb = false;
  fc.deltaS = 1.0 / static_cast<double>(config.ns - 1);

  std::cout << "Initializing backends...\n";

  // Collect results.
  std::vector<BenchmarkResult> results;

  // CPU backend (always available).
  {
    vmecpp::ComputeBackendCpu cpu_backend;
    std::cout << "  CPU backend: " << cpu_backend.GetName() << "\n";
    results.push_back(
        RunBenchmark(&cpu_backend, config, s, fb, rp, profiles, fc));
  }

  // CUDA backend (if available).
  {
    vmecpp::BackendConfig cuda_config;
    cuda_config.type = vmecpp::BackendType::kCuda;
    auto cuda_result = vmecpp::ComputeBackendFactory::Create(cuda_config);
    if (cuda_result.ok()) {
      auto& cuda_backend = cuda_result.value();
      std::cout << "  CUDA backend: " << cuda_backend->GetName() << "\n";
      results.push_back(
          RunBenchmark(cuda_backend.get(), config, s, fb, rp, profiles, fc));
    } else {
      std::cout << "  CUDA backend: not available\n";
      BenchmarkResult cuda_unavailable;
      cuda_unavailable.backend_name = "CUDA";
      cuda_unavailable.available = false;
      results.push_back(cuda_unavailable);
    }
  }

  PrintResults(results, config);

  return 0;
}
