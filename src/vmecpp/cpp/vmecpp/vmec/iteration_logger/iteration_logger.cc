// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#include "vmecpp/vmec/iteration_logger/iteration_logger.h"

#include <algorithm>
#include <cmath>

#include "absl/strings/str_format.h"

namespace vmecpp {

namespace {

// ANSI escape sequences using standard 8-color palette.
// Colors are determined by the user's terminal theme.
constexpr const char* kBold = "\033[1m";
constexpr const char* kDim = "\033[2m";
constexpr const char* kReset = "\033[0m";
constexpr const char* kGreen = "\033[32m";
constexpr const char* kYellow = "\033[33m";
constexpr const char* kRed = "\033[31m";

// Move cursor up N lines (for in-place rewriting).
std::string CursorUp(int n) {
  if (n <= 0) return "";
  return absl::StrFormat("\033[%dA", n);
}

// Erase from cursor to end of line.
constexpr const char* kClearLine = "\033[K";

}  // namespace

IterationLogger::IterationLogger(std::ostream& output, OutputMode mode)
    : output_(output), mode_(mode) {}

void IterationLogger::BeginStage(int stage_index, int num_stages, int ns,
                                 int mnmax, double ftolv, int niter,
                                 bool is_free_boundary) {
  if (mode_ == OutputMode::kSilent) return;

  current_stage_ = stage_index;
  num_stages_ = num_stages;
  is_free_boundary_ = is_free_boundary;

  // Grow the stages vector if needed (stages may arrive out of order
  // if jacob_off inserts an extra ns=3 stage).
  if (stage_index >= static_cast<int>(stages_.size())) {
    stages_.resize(stage_index + 1);
  }

  StageState& stage = stages_[stage_index];
  stage.ns = ns;
  stage.niter = niter;
  stage.ftolv = ftolv;
  stage.last_iter = 0;
  stage.initial_fsq_set = false;
  stage.completed = false;

  if (mode_ == OutputMode::kLegacy) {
    output_ << absl::StrFormat(
        "\n NS = %d   NO. FOURIER MODES = %d   FTOLV = %9.3e   NITER = %d\n",
        ns, mnmax, ftolv, niter);
    PrintLegacyHeader();
  }
  // In progress mode, the display is rendered on the first LogIteration call.
}

void IterationLogger::LogIteration(int iter, double fsqr, double fsqz,
                                   double fsql, double fsqr1, double fsqz1,
                                   double fsql1, double delt, double rax,
                                   double w_mhd, double beta_vol_avg,
                                   double vol_avg_m, double delbsq) {
  if (mode_ == OutputMode::kSilent) return;

  if (mode_ == OutputMode::kLegacy) {
    PrintLegacyRow(iter, fsqr, fsqz, fsql, fsqr1, fsqz1, fsql1, delt, rax,
                   w_mhd, beta_vol_avg, vol_avg_m, delbsq);
    return;
  }

  // Progress mode: update the current stage state and re-render.
  if (current_stage_ < 0 ||
      current_stage_ >= static_cast<int>(stages_.size())) {
    return;
  }

  StageState& stage = stages_[current_stage_];
  double fsq_total = fsqr + fsqz + fsql;

  if (!stage.initial_fsq_set && fsq_total > 0.0) {
    stage.initial_log_fsq = std::log10(fsq_total);
    stage.initial_fsq_set = true;
  }

  stage.last_iter = iter;
  stage.last_fsq = fsq_total;
  stage.last_beta = beta_vol_avg;
  stage.last_w_mhd = w_mhd;
  stage.last_rax = rax;
  stage.last_delbsq = delbsq;

  if (mode_ == OutputMode::kProgress) {
    RenderProgressDisplay();
  } else {
    RenderSingleLineProgress();
  }
}

void IterationLogger::EndStage(double mhd_energy) {
  if (mode_ == OutputMode::kSilent) return;

  if (current_stage_ >= 0 &&
      current_stage_ < static_cast<int>(stages_.size())) {
    stages_[current_stage_].completed = true;
  }

  if (mode_ == OutputMode::kLegacy) {
    output_ << absl::StrFormat("MHD Energy = %12.6e\n", mhd_energy);
    output_ << std::flush;
    return;
  }

  // Final render for this stage showing completion.
  if (mode_ == OutputMode::kProgress) {
    RenderProgressDisplay();
  } else {
    // Print the final state and move to a new line so it persists.
    RenderSingleLineProgress();
    output_ << "\n";
  }
}

void IterationLogger::EndRun(const RunSummary& summary) {
  if (mode_ == OutputMode::kSilent || mode_ == OutputMode::kLegacy) return;

  PrintSummaryTable(summary);
  output_ << std::flush;
}

// ---------------------------------------------------------------------------
// Progress computation
// ---------------------------------------------------------------------------

double IterationLogger::ComputeProgress(const StageState& stage) const {
  if (stage.completed) return 1.0;

  // FSQ-based progress: how far have we dropped in log-space toward ftolv?
  double fsq_progress = 0.0;
  if (stage.initial_fsq_set && stage.last_fsq > 0.0) {
    double log_current = std::log10(stage.last_fsq);
    double log_target = std::log10(stage.ftolv);
    if (stage.initial_log_fsq > log_target) {
      fsq_progress = (stage.initial_log_fsq - log_current) /
                     (stage.initial_log_fsq - log_target);
    }
  }

  // Iteration-based progress: fraction of niter used.
  double iter_progress = 0.0;
  if (stage.niter > 0) {
    iter_progress =
        static_cast<double>(stage.last_iter) / static_cast<double>(stage.niter);
  }

  // The stage terminates when either condition is met, so use the max.
  double progress = std::max(fsq_progress, iter_progress);
  return std::max(0.0, std::min(1.0, progress));
}

// ---------------------------------------------------------------------------
// Progress mode rendering
// ---------------------------------------------------------------------------

void IterationLogger::RenderProgressDisplay() {
  // Move cursor up to overwrite previous display.
  if (progress_lines_printed_ > 0) {
    output_ << CursorUp(progress_lines_printed_);
  }

  int lines = 0;

  // Line 1: stage selector
  output_ << kClearLine;
  for (int i = 0; i < static_cast<int>(stages_.size()); ++i) {
    if (i > 0) output_ << "  ";
    const StageState& s = stages_[i];

    if (s.completed) {
      output_ << kGreen << kBold;
    } else if (i == current_stage_) {
      output_ << kBold;
    } else {
      output_ << kDim;
    }

    output_ << absl::StrFormat("[%d/%d] NS=%d", i + 1, num_stages_, s.ns);
    output_ << kReset;
  }
  output_ << "\n";
  lines++;

  // Blank separator line.
  output_ << kClearLine << "\n";
  lines++;

  // One block per stage: bar line + metrics line + blank line.
  for (int i = 0; i < static_cast<int>(stages_.size()); ++i) {
    const StageState& s = stages_[i];

    if (!s.initial_fsq_set && !s.completed) {
      // Stage not started yet.
      output_ << kClearLine;
      output_ << absl::StrFormat("  Stage %d (NS=%3d): ", i + 1, s.ns);
      output_ << "[" << kDim << std::string(50, '-') << kReset << "]" << kDim
              << "   0.0%" << kReset << "\n";
      lines++;

      output_ << kClearLine << "  " << kDim << "waiting..." << kReset << "\n";
      lines++;

      output_ << kClearLine << "\n";
      lines++;
      continue;
    }

    double progress_frac = ComputeProgress(s);

    // Color for the filled portion only.
    const char* fill_color = kRed;
    if (progress_frac >= 1.0) {
      fill_color = kGreen;
    } else if (progress_frac > 0.5) {
      fill_color = kYellow;
    }

    // Bar line.
    output_ << kClearLine;
    output_ << absl::StrFormat("  Stage %d (NS=%3d): ", i + 1, s.ns);
    output_ << FormatBar(progress_frac, 50, fill_color);
    output_ << absl::StrFormat(" %5.1f%%", progress_frac * 100.0);
    output_ << "\n";
    lines++;

    // Metrics line.
    output_ << kClearLine << "  " << kDim;
    output_ << absl::StrFormat(
        "iter=%-5d FSQ=%.2e  beta=%.4f  W_MHD=%.4f  "
        "RAX=%.4f",
        s.last_iter, s.last_fsq, s.last_beta, s.last_w_mhd, s.last_rax);
    if (is_free_boundary_) {
      output_ << absl::StrFormat("  DELBSQ=%.3e", s.last_delbsq);
    }
    output_ << kReset << "\n";
    lines++;

    // Blank line between stages.
    output_ << kClearLine << "\n";
    lines++;
  }

  output_ << std::flush;
  progress_lines_printed_ = lines;
}

void IterationLogger::RenderSingleLineProgress() {
  if (current_stage_ < 0 ||
      current_stage_ >= static_cast<int>(stages_.size())) {
    return;
  }

  const StageState& s = stages_[current_stage_];
  double progress_frac = ComputeProgress(s);

  const char* fill_color = kRed;
  if (progress_frac >= 1.0) {
    fill_color = kGreen;
  } else if (progress_frac > 0.5) {
    fill_color = kYellow;
  }

  // Use carriage return to overwrite the current line in-place.
  output_ << "\r";
  output_ << absl::StrFormat("Stage %d/%d [ns=%d] ", current_stage_ + 1,
                             num_stages_, s.ns);
  output_ << FormatBar(progress_frac, 30, fill_color);
  output_ << absl::StrFormat(" %5.1f%%  FSQ=%.2e", progress_frac * 100.0,
                             s.last_fsq);
  if (is_free_boundary_) {
    output_ << absl::StrFormat("  delbsq=%.2e", s.last_delbsq);
  }

  // Pad with spaces to clear any leftover characters from a longer line.
  output_ << "    ";
  output_ << std::flush;
}

std::string IterationLogger::FormatBar(double fraction, int width,
                                       const char* fill_color) const {
  fraction = std::max(0.0, std::min(1.0, fraction));
  int filled = static_cast<int>(fraction * width);
  std::string bar = "[";
  if (fill_color != nullptr && filled > 0) {
    bar += fill_color;
  }
  bar.append(filled, '=');
  if (fill_color != nullptr && filled > 0) {
    bar += kReset;
  }
  bar.append(width - filled, '-');
  bar += "]";
  return bar;
}

// ---------------------------------------------------------------------------
// Summary table
// ---------------------------------------------------------------------------

void IterationLogger::PrintSummaryTable(const RunSummary& summary) const {
  // Use ANSI codes in both progress modes (TTY and non-TTY both support them).
  const bool use_ansi =
      (mode_ == OutputMode::kProgress || mode_ == OutputMode::kProgressNonTTY);
  const char* bold = use_ansi ? kBold : "";
  const char* green = use_ansi ? kGreen : "";
  const char* red = use_ansi ? kRed : "";
  const char* reset = use_ansi ? kReset : "";

  // Unicode box-drawing characters (rounded corners, like rich ROUNDED style).
  // U+256D, U+256E, U+256F, U+2570, U+2500, U+2502
  const char* tl = "\xe2\x95\xad";  // top-left corner
  const char* tr = "\xe2\x95\xae";  // top-right corner
  const char* bl = "\xe2\x95\xb0";  // bottom-left corner
  const char* br = "\xe2\x95\xaf";  // bottom-right corner
  const char* h = "\xe2\x94\x80";   // horizontal line
  const char* v = "\xe2\x94\x82";   // vertical line

  const int table_width = 38;
  // Build horizontal borders:
  std::string hline;
  for (int i = 0; i < table_width; ++i) hline += h;

  std::string top_border = std::string(tl) + hline + tr;
  std::string bot_border = std::string(bl) + hline + br;

  // Helper: print a row with vertical bars, padded to table_width visible
  // characters. content must be exactly table_width visible characters wide
  // (excluding ANSI codes).
  auto row = [&](const std::string& content) {
    output_ << v << content << v << "\n";
  };
  auto blank_row = [&]() { row(std::string(table_width, ' ')); };

  // U+251C left-tee, U+2524 right-tee for the separator line.
  const char* lt = "\xe2\x94\x9c";
  const char* rt = "\xe2\x94\xa4";
  std::string sep_border = std::string(lt) + hline + rt;

  output_ << "\n" << top_border << "\n";
  row(absl::StrFormat("          %sVMEC++ Run Summary%14s", bold, reset));
  output_ << sep_border << "\n";

  if (summary.converged) {
    row(absl::StrFormat(" %-15s %s%sCONVERGED%s            ", "Status", green,
                        bold, reset));
  } else {
    row(absl::StrFormat(" %-15s %s%sNOT CONVERGED%s        ", "Status", red,
                        bold, reset));
  }

  row(absl::StrFormat(" %-15s %-20d ", "Jacobian resets",
                      summary.num_jacobian_resets));
  blank_row();

  row(absl::StrFormat(" %-15s %-20.4e ", "Final FSQR", summary.fsqr));
  row(absl::StrFormat(" %-15s %-20.4e ", "Final FSQZ", summary.fsqz));
  row(absl::StrFormat(" %-15s %-20.4e ", "Final FSQL", summary.fsql));
  blank_row();

  row(absl::StrFormat(" %-14s %8.4f %%             ", "β total   ",
                      100 * summary.betatot));
  row(absl::StrFormat(" %-14s %8.4f %%             ", "β poloidal",
                      100 * summary.betapol));
  row(absl::StrFormat(" %-14s %8.4f %%             ", "β toroidal",
                      100 * summary.betator));
  blank_row();

  row(absl::StrFormat(" %-14s % -20.4f  ", "R_major", summary.rmajor));
  row(absl::StrFormat(" %-14s % -20.4f  ", "a_minor", summary.aminor));
  row(absl::StrFormat(" %-14s % -20.4f  ", "B0", summary.b0));

  output_ << bot_border << "\n";
}

// ---------------------------------------------------------------------------
// Legacy mode output (reproduces original VMEC++ format)
// ---------------------------------------------------------------------------

void IterationLogger::PrintLegacyHeader() const {
  output_ << '\n';
  if (is_free_boundary_) {
    output_ << " ITER |    FSQR     FSQZ     FSQL    |    fsqr     fsqz      "
               "fsql   |   DELT   |  RAX(v=0) |    W_MHD   |   <BETA>   |  "
               "<M>  |  DELBSQ  \n";
    output_ << "------+------------------------------+-----------------------"
               "-------+----------+-----------+------------+------------+----"
               "---+----------\n";
  } else {
    output_ << " ITER |    FSQR     FSQZ     FSQL    |    fsqr     fsqz    "
               "  fsql  "
               " |   DELT   |  RAX(v=0) |    W_MHD   |   <BETA>   |  <M>  \n";
    output_ << "------+------------------------------+---------------------"
               "--------"
               "-+----------+-----------+------------+------------+-------\n";
  }
}

void IterationLogger::PrintLegacyRow(int iter, double fsqr, double fsqz,
                                     double fsql, double fsqr1, double fsqz1,
                                     double fsql1, double delt, double rax,
                                     double w_mhd, double beta_vol_avg,
                                     double vol_avg_m, double delbsq) const {
  if (is_free_boundary_) {
    output_ << absl::StrFormat(
        "%5d | %.2e  %.2e  %.2e | %.2e  %.2e  %.2e | %.2e | "
        "%.3e | %.4e | %.4e | %5.3f | %.3e\n",
        iter, fsqr, fsqz, fsql, fsqr1, fsqz1, fsql1, delt, rax, w_mhd,
        beta_vol_avg, vol_avg_m, delbsq);
  } else {
    output_ << absl::StrFormat(
        "%5d | %.2e  %.2e  %.2e | %.2e  %.2e  %.2e | %.2e | "
        "%.3e | %.4e | %.4e | %5.3f\n",
        iter, fsqr, fsqz, fsql, fsqr1, fsqz1, fsql1, delt, rax, w_mhd,
        beta_vol_avg, vol_avg_m);
  }
}

}  // namespace vmecpp
