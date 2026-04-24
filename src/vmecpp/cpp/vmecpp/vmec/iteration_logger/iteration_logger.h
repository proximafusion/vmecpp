// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#ifndef VMECPP_VMEC_ITERATION_LOGGER_ITERATION_LOGGER_H_
#define VMECPP_VMEC_ITERATION_LOGGER_ITERATION_LOGGER_H_

#include <cstdint>
#include <ostream>
#include <string>
#include <vector>

namespace vmecpp {

// Controls the output format of iteration logging.
// Values are chosen so that the old bool verbose (false=0, true=1) maps to
// kSilent and kLegacy respectively, preserving backward compatibility.
//
// The caller (Python layer) is responsible for choosing the appropriate mode
// based on environment detection (TTY, Jupyter, pipe, etc.).
//
// kSilent:          no output
// kLegacy:          traditional table output (original VMEC++ format)
// kProgress:        multi-line progress bars with ANSI cursor movement (TTY)
// kProgressNonTTY: single-line progress with carriage return (Jupyter, etc.)
enum class OutputMode : std::uint8_t {
  kSilent = 0,
  kLegacy = 1,
  kProgress = 2,
  kProgressNonTTY = 3
};

// Summary data displayed at the end of a VMEC++ run.
struct RunSummary {
  bool converged = false;
  int total_iterations = 0;
  int num_jacobian_resets = 0;
  double fsqr = 0.0;
  double fsqz = 0.0;
  double fsql = 0.0;
  double ftolv = 0.0;
  double betatot = 0.0;
  double betapol = 0.0;
  double betator = 0.0;
  double w_mhd = 0.0;
  double rax = 0.0;
  double aminor = 0.0;
  double rmajor = 0.0;
  double b0 = 0.0;
};

// Handles formatted output of VMEC++ iteration progress.
// Supports four modes: see OutputMode above.
//
// Thread safety: all public methods must be called from a single thread.
// This is already guaranteed by the existing call sites in vmec.cc.
class IterationLogger {
 public:
  IterationLogger(std::ostream& output, OutputMode mode);

  // Called at the start of each multigrid stage.
  void BeginStage(int stage_index, int num_stages, int ns, int mnmax,
                  double ftolv, int niter, bool is_free_boundary);

  // Called every nstep iterations with current convergence data.
  void LogIteration(int iter, double fsqr, double fsqz, double fsql,
                    double fsqr1, double fsqz1, double fsql1, double delt,
                    double rax, double w_mhd, double beta_vol_avg,
                    double vol_avg_m, double delbsq);

  // Called after SolveEquilibrium completes for a stage.
  void EndStage(double mhd_energy);

  // Called once at the very end of a run. Prints the summary table.
  void EndRun(const RunSummary& summary);

 private:
  // Per-stage tracking state for progress display.
  struct StageState {
    int ns = 0;
    int niter = 0;
    int last_iter = 0;
    double ftolv = 0.0;
    double initial_log_fsq = 0.0;
    double last_fsq = 0.0;
    double last_beta = 0.0;
    double last_w_mhd = 0.0;
    double last_rax = 0.0;
    double last_delbsq = 0.0;
    bool initial_fsq_set = false;
    bool completed = false;
  };

  // Compute progress fraction for a stage, considering both FSQ convergence
  // and iteration count toward niter.
  double ComputeProgress(const StageState& stage) const;

  // Render the full multi-line progress display (kProgress mode).
  void RenderProgressDisplay();

  // Render a compact single-line progress display (kProgressNonTTY mode).
  void RenderSingleLineProgress();

  // Format a progress bar of the given width.
  // fraction is clamped to [0, 1].
  // fill_color is applied only to the filled portion of the bar.
  std::string FormatBar(double fraction, int width,
                        const char* fill_color = nullptr) const;

  // Print the post-run summary table.
  void PrintSummaryTable(const RunSummary& summary) const;

  // Legacy mode: print the table header.
  void PrintLegacyHeader() const;

  // Legacy mode: print one iteration data row.
  void PrintLegacyRow(int iter, double fsqr, double fsqz, double fsql,
                      double fsqr1, double fsqz1, double fsql1, double delt,
                      double rax, double w_mhd, double beta_vol_avg,
                      double vol_avg_m, double delbsq) const;

  std::ostream& output_;
  OutputMode mode_;
  bool is_free_boundary_ = false;
  int current_stage_ = -1;
  int num_stages_ = 0;
  std::vector<StageState> stages_;

  // Number of lines printed by the last RenderProgressDisplay call,
  // used to move the cursor back up for in-place rewriting (TTY only).
  int progress_lines_printed_ = 0;
};

}  // namespace vmecpp

#endif  // VMECPP_VMEC_ITERATION_LOGGER_ITERATION_LOGGER_H_
