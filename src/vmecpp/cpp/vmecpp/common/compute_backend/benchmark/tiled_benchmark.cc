// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT

// Benchmark for comparing tiled vs non-tiled GPU execution.
// Measures performance across different problem sizes and tile configurations.

#include <chrono>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

#include "vmecpp/common/compute_backend/compute_backend_cpu.h"
#include "vmecpp/common/compute_backend/cuda/tile_memory_budget.h"
#include "vmecpp/common/compute_backend/cuda/tile_scheduler.h"

namespace vmecpp {

// Benchmark configuration
struct BenchmarkConfig {
  int ns_min = 50;
  int ns_max = 500;
  int ns_step = 50;
  int mpol = 12;
  int ntor = 12;
  int n_zeta = 36;
  int n_theta = 36;
  int iterations = 10;
  int warmup = 3;
  bool verbose = false;
};

// Benchmark result for a single configuration
struct BenchmarkResult {
  int ns;
  int tile_size;
  int num_tiles;
  double time_non_tiled_us;
  double time_tiled_us;
  double overhead_percent;
  size_t memory_per_surface;
  bool tiling_required;
};

// Timer utility
class Timer {
 public:
  void Start() { start_ = std::chrono::high_resolution_clock::now(); }

  double ElapsedMicroseconds() const {
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::micro>(end - start_).count();
  }

 private:
  std::chrono::high_resolution_clock::time_point start_;
};

// Simulate DFT operation for benchmarking (CPU-based)
void SimulateDFT(const std::vector<double>& fourier_coeffs,
                 std::vector<double>& real_space, int ns, int mpol, int ntor,
                 int n_zeta, int n_theta) {
  const int coeff_per_surface = mpol * (ntor + 1);
  const int grid_per_surface = n_zeta * n_theta;

  // Simple simulation of DFT work
  for (int j = 0; j < ns; ++j) {
    const double* coeffs = &fourier_coeffs[j * coeff_per_surface];
    double* grid = &real_space[j * grid_per_surface];

    for (int k = 0; k < n_zeta; ++k) {
      for (int l = 0; l < n_theta; ++l) {
        double sum = 0.0;
        for (int m = 0; m < mpol; ++m) {
          for (int n = 0; n <= ntor; ++n) {
            sum += coeffs[m * (ntor + 1) + n] * std::cos(m * l * 0.1) *
                   std::cos(n * k * 0.1);
          }
        }
        grid[k * n_theta + l] = sum;
      }
    }
  }
}

// Simulate tiled DFT operation
void SimulateTiledDFT(const std::vector<double>& fourier_coeffs,
                      std::vector<double>& real_space, int ns, int mpol,
                      int ntor, int n_zeta, int n_theta, int tile_size) {
  TileSchedulerConfig config;
  config.ns = ns;
  config.tile_size = tile_size;
  config.ns_min = 1;
  config.op_type = TileOperationType::kNoOverlap;

  TileScheduler scheduler(config);

  const int coeff_per_surface = mpol * (ntor + 1);
  const int grid_per_surface = n_zeta * n_theta;

  for (const auto& tile : scheduler.GetTiles()) {
    int start = tile.start_surface - 1;
    int end = tile.end_surface - 1;

    for (int j = start; j < end; ++j) {
      const double* coeffs = &fourier_coeffs[j * coeff_per_surface];
      double* grid = &real_space[j * grid_per_surface];

      for (int k = 0; k < n_zeta; ++k) {
        for (int l = 0; l < n_theta; ++l) {
          double sum = 0.0;
          for (int m = 0; m < mpol; ++m) {
            for (int n = 0; n <= ntor; ++n) {
              sum += coeffs[m * (ntor + 1) + n] * std::cos(m * l * 0.1) *
                     std::cos(n * k * 0.1);
            }
          }
          grid[k * n_theta + l] = sum;
        }
      }
    }
  }
}

// Simulate stencil operation (like Jacobian)
void SimulateStencil(const std::vector<double>& input,
                     std::vector<double>& output, int ns, int n_points) {
  for (int j = 0; j < ns - 1; ++j) {
    for (int k = 0; k < n_points; ++k) {
      int idx = j * n_points + k;
      int idx_next = (j + 1) * n_points + k;
      output[idx] = (input[idx] + input[idx_next]) * 0.5 +
                    std::sin(input[idx]) * std::cos(input[idx_next]);
    }
  }
}

// Simulate tiled stencil operation
void SimulateTiledStencil(const std::vector<double>& input,
                          std::vector<double>& output, int ns, int n_points,
                          int tile_size) {
  TileSchedulerConfig config;
  config.ns = ns;
  config.tile_size = tile_size;
  config.ns_min = 1;
  config.op_type = TileOperationType::kForwardStencil;

  TileScheduler scheduler(config);

  for (const auto& tile : scheduler.GetTiles()) {
    int start = tile.start_surface - 1;
    int end = tile.end_surface - 1;

    for (int j = start; j < end && j < ns - 1; ++j) {
      for (int k = 0; k < n_points; ++k) {
        int idx = j * n_points + k;
        int idx_next = (j + 1) * n_points + k;
        output[idx] = (input[idx] + input[idx_next]) * 0.5 +
                      std::sin(input[idx]) * std::cos(input[idx_next]);
      }
    }
  }
}

// Run benchmarks for a single configuration
BenchmarkResult RunBenchmark(int ns, int tile_size, const BenchmarkConfig& cfg,
                             std::mt19937& rng) {
  BenchmarkResult result;
  result.ns = ns;
  result.tile_size = tile_size;

  const int coeff_size = ns * cfg.mpol * (cfg.ntor + 1);
  const int grid_size = ns * cfg.n_zeta * cfg.n_theta;

  // Calculate memory per surface
  GridSizeParams grid_params{ns, cfg.mpol, cfg.ntor, cfg.n_zeta, cfg.n_theta};
  result.memory_per_surface = TileMemoryBudget::PerSurfaceMemory(grid_params);

  // Generate random data
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  std::vector<double> fourier_coeffs(coeff_size);
  std::vector<double> real_space(grid_size);
  std::vector<double> real_space_tiled(grid_size);

  for (auto& x : fourier_coeffs) {
    x = dist(rng);
  }

  // Warmup
  for (int i = 0; i < cfg.warmup; ++i) {
    SimulateDFT(fourier_coeffs, real_space, ns, cfg.mpol, cfg.ntor, cfg.n_zeta,
                cfg.n_theta);
  }

  // Benchmark non-tiled
  Timer timer;
  timer.Start();
  for (int i = 0; i < cfg.iterations; ++i) {
    SimulateDFT(fourier_coeffs, real_space, ns, cfg.mpol, cfg.ntor, cfg.n_zeta,
                cfg.n_theta);
  }
  result.time_non_tiled_us = timer.ElapsedMicroseconds() / cfg.iterations;

  // Calculate number of tiles
  TileSchedulerConfig sched_config;
  sched_config.ns = ns;
  sched_config.tile_size = tile_size;
  sched_config.ns_min = 1;
  sched_config.op_type = TileOperationType::kNoOverlap;
  TileScheduler scheduler(sched_config);
  result.num_tiles = scheduler.NumTiles();

  // Benchmark tiled
  timer.Start();
  for (int i = 0; i < cfg.iterations; ++i) {
    SimulateTiledDFT(fourier_coeffs, real_space_tiled, ns, cfg.mpol, cfg.ntor,
                     cfg.n_zeta, cfg.n_theta, tile_size);
  }
  result.time_tiled_us = timer.ElapsedMicroseconds() / cfg.iterations;

  // Calculate overhead
  if (result.time_non_tiled_us > 0) {
    result.overhead_percent =
        (result.time_tiled_us / result.time_non_tiled_us - 1.0) * 100.0;
  } else {
    result.overhead_percent = 0.0;
  }

  // Check if tiling would be required (simulated with 8GB GPU)
  size_t simulated_gpu_memory = 8ULL * 1024 * 1024 * 1024;  // 8 GB
  size_t total_memory = result.memory_per_surface * ns;
  result.tiling_required = (total_memory > simulated_gpu_memory * 0.8);

  return result;
}

void PrintHeader() {
  std::cout << "\n";
  std::cout << std::setw(8) << "ns" << std::setw(10) << "tile_size"
            << std::setw(8) << "tiles" << std::setw(14) << "non-tiled(us)"
            << std::setw(12) << "tiled(us)" << std::setw(12) << "overhead(%)"
            << std::setw(12) << "mem/surf" << std::setw(10) << "need_tile"
            << "\n";
  std::cout << std::string(86, '-') << "\n";
}

void PrintResult(const BenchmarkResult& r) {
  auto format_size = [](size_t bytes) -> std::string {
    if (bytes >= 1024 * 1024) {
      char buf[32];
      snprintf(buf, sizeof(buf), "%.1fMB", bytes / (1024.0 * 1024.0));
      return buf;
    } else if (bytes >= 1024) {
      char buf[32];
      snprintf(buf, sizeof(buf), "%.1fKB", bytes / 1024.0);
      return buf;
    }
    return std::to_string(bytes) + "B";
  };

  std::cout << std::setw(8) << r.ns << std::setw(10) << r.tile_size
            << std::setw(8) << r.num_tiles << std::setw(14) << std::fixed
            << std::setprecision(1) << r.time_non_tiled_us << std::setw(12)
            << r.time_tiled_us << std::setw(12) << std::setprecision(2)
            << std::showpos << r.overhead_percent << std::noshowpos
            << std::setw(12) << format_size(r.memory_per_surface)
            << std::setw(10) << (r.tiling_required ? "YES" : "NO") << "\n";
}

void RunAllBenchmarks(const BenchmarkConfig& cfg) {
  std::cout << "========================================\n";
  std::cout << "  VMEC++ Tiled Execution Benchmark\n";
  std::cout << "========================================\n";
  std::cout << "\nConfiguration:\n";
  std::cout << "  mpol=" << cfg.mpol << ", ntor=" << cfg.ntor << "\n";
  std::cout << "  Grid: " << cfg.n_zeta << " x " << cfg.n_theta << "\n";
  std::cout << "  Iterations: " << cfg.iterations << "\n";

  std::mt19937 rng(42);  // Fixed seed for reproducibility

  // Benchmark different tile sizes for each ns
  std::vector<int> tile_sizes = {10, 25, 50, 100};

  PrintHeader();

  for (int ns = cfg.ns_min; ns <= cfg.ns_max; ns += cfg.ns_step) {
    // First, show non-tiled as baseline (tile_size = ns)
    auto baseline = RunBenchmark(ns, ns, cfg, rng);
    baseline.tile_size = ns;
    baseline.num_tiles = 1;
    baseline.overhead_percent = 0.0;
    PrintResult(baseline);

    // Then show various tile sizes
    for (int tile_size : tile_sizes) {
      if (tile_size < ns) {
        auto result = RunBenchmark(ns, tile_size, cfg, rng);
        PrintResult(result);
      }
    }

    std::cout << "\n";
  }

  // Stencil operation benchmark
  std::cout << "\n========================================\n";
  std::cout << "  Stencil Operation Benchmark\n";
  std::cout << "========================================\n";

  const int n_points = cfg.n_zeta * cfg.n_theta;

  std::cout << "\nConfiguration:\n";
  std::cout << "  Points per surface: " << n_points << "\n";
  std::cout << "  Iterations: " << cfg.iterations << "\n";

  std::cout << "\n"
            << std::setw(8) << "ns" << std::setw(10) << "tile_size"
            << std::setw(14) << "non-tiled(us)" << std::setw(12) << "tiled(us)"
            << std::setw(12) << "overhead(%)" << "\n";
  std::cout << std::string(56, '-') << "\n";

  for (int ns = cfg.ns_min; ns <= cfg.ns_max; ns += cfg.ns_step) {
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::vector<double> input(ns * n_points);
    std::vector<double> output(ns * n_points);

    for (auto& x : input) {
      x = dist(rng);
    }

    // Warmup
    for (int i = 0; i < cfg.warmup; ++i) {
      SimulateStencil(input, output, ns, n_points);
    }

    // Benchmark non-tiled
    Timer timer;
    timer.Start();
    for (int i = 0; i < cfg.iterations; ++i) {
      SimulateStencil(input, output, ns, n_points);
    }
    double non_tiled_us = timer.ElapsedMicroseconds() / cfg.iterations;

    // Baseline
    std::cout << std::setw(8) << ns << std::setw(10) << ns << std::setw(14)
              << std::fixed << std::setprecision(1) << non_tiled_us
              << std::setw(12) << non_tiled_us << std::setw(12) << "+0.00"
              << "\n";

    // Tiled versions
    for (int tile_size : tile_sizes) {
      if (tile_size >= ns) continue;

      timer.Start();
      for (int i = 0; i < cfg.iterations; ++i) {
        SimulateTiledStencil(input, output, ns, n_points, tile_size);
      }
      double tiled_us = timer.ElapsedMicroseconds() / cfg.iterations;
      double overhead = (tiled_us / non_tiled_us - 1.0) * 100.0;

      std::cout << std::setw(8) << ns << std::setw(10) << tile_size
                << std::setw(14) << std::fixed << std::setprecision(1)
                << non_tiled_us << std::setw(12) << tiled_us << std::setw(12)
                << std::setprecision(2) << std::showpos << overhead
                << std::noshowpos << "\n";
    }

    std::cout << "\n";
  }
}

}  // namespace vmecpp

void PrintUsage(const char* program) {
  std::cout << "Usage: " << program << " [options]\n";
  std::cout << "\nOptions:\n";
  std::cout << "  --ns-min=<int>     Minimum ns (default: 50)\n";
  std::cout << "  --ns-max=<int>     Maximum ns (default: 500)\n";
  std::cout << "  --ns-step=<int>    Step size for ns (default: 50)\n";
  std::cout << "  --mpol=<int>       Number of poloidal modes (default: 12)\n";
  std::cout << "  --ntor=<int>       Number of toroidal modes (default: 12)\n";
  std::cout << "  --nzeta=<int>      Toroidal grid points (default: 36)\n";
  std::cout << "  --ntheta=<int>     Poloidal grid points (default: 36)\n";
  std::cout << "  --iterations=<int> Benchmark iterations (default: 10)\n";
  std::cout << "  --warmup=<int>     Warmup iterations (default: 3)\n";
  std::cout << "  -v, --verbose      Verbose output\n";
  std::cout << "  -h, --help         Show this help\n";
}

int main(int argc, char* argv[]) {
  vmecpp::BenchmarkConfig cfg;

  // Parse command line arguments
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];

    if (arg == "-h" || arg == "--help") {
      PrintUsage(argv[0]);
      return 0;
    } else if (arg == "-v" || arg == "--verbose") {
      cfg.verbose = true;
    } else if (arg.substr(0, 9) == "--ns-min=") {
      cfg.ns_min = std::stoi(arg.substr(9));
    } else if (arg.substr(0, 9) == "--ns-max=") {
      cfg.ns_max = std::stoi(arg.substr(9));
    } else if (arg.substr(0, 10) == "--ns-step=") {
      cfg.ns_step = std::stoi(arg.substr(10));
    } else if (arg.substr(0, 7) == "--mpol=") {
      cfg.mpol = std::stoi(arg.substr(7));
    } else if (arg.substr(0, 7) == "--ntor=") {
      cfg.ntor = std::stoi(arg.substr(7));
    } else if (arg.substr(0, 8) == "--nzeta=") {
      cfg.n_zeta = std::stoi(arg.substr(8));
    } else if (arg.substr(0, 9) == "--ntheta=") {
      cfg.n_theta = std::stoi(arg.substr(9));
    } else if (arg.substr(0, 13) == "--iterations=") {
      cfg.iterations = std::stoi(arg.substr(13));
    } else if (arg.substr(0, 9) == "--warmup=") {
      cfg.warmup = std::stoi(arg.substr(9));
    } else {
      std::cerr << "Unknown option: " << arg << "\n";
      PrintUsage(argv[0]);
      return 1;
    }
  }

  vmecpp::RunAllBenchmarks(cfg);

  return 0;
}
