// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#ifndef VMECPP_COMMON_FLOW_CONTROL_FLOW_CONTROL_H_
#define VMECPP_COMMON_FLOW_CONTROL_FLOW_CONTROL_H_

#include <cstdint>
#include <optional>
#include <vector>

namespace vmecpp {

// enumerates values of `restart_reason`
// was `irst` = 1, 2, 3, 4 in Fortran VMEC
enum class RestartReason : std::uint8_t {
  // irst == 1, no restart required, instead make backup of current state vector
  // when calling Vmec::RestartIteration
  NO_RESTART = 1,

  // irst == 2, bad Jacobian, flux surfaces are overlapping
  BAD_JACOBIAN = 2,

  // irst == 3, bad progress, residuals not decaying as expected
  BAD_PROGRESS = 3,

  // irst == 4, huge initial forces, flux surfaces are too close to each other
  // (but not overlapping yet)
  HUGE_INITIAL_FORCES = 4
};

RestartReason RestartReasonFromInt(int restart_reason);

class FlowControl {
 public:
  // ns4: number of iterations between update of radial preconditioner matrix
  static constexpr int kPreconditionerUpdateInterval = 25;

  FlowControl(bool lfreeb, double delt, int num_grids,
              std::optional<int> max_threads = std::nullopt);

  int max_threads() const;

  const bool lfreeb;

  // was called `irst` in Fortran VMEC
  RestartReason restart_reason;

  // current ns in algorithm
  int ns;

  int neqs;
  int neqs_old;

  bool haveToFlipTheta;

  int ijacob;

  int multi_ns_grid;

  // ------ current multi-grid step settings

  // radial resolution of current multi-grid step
  int nsval;

  // radial grid spacing of flux surfaces: 1.0 / (ns - 1.0)
  double deltaS;

  // current force tolerance
  double ftolv;

  // current maximum number of iterations
  int niterv;

  // --------- end: current multi-grid step settings

  int num_surfaces_to_distribute;

  // for initialize_radial and interp
  int ns_min;
  int ns_old;

  double delt0r;

  // Cumulative force residuals (radial, vertical and lambda)
  // Populated by `evalFResInvar`
  double fsqr, fsqz, fsql;

  // Time-trace of the invariant force residuals during convergence
  // fsqt = (force_residual_r + force_residual_z + force_residual_lambda)
  std::vector<double> force_residual_r;
  std::vector<double> force_residual_z;
  std::vector<double> force_residual_lambda;

  // Preconditioned cumulative force residuals (radial, vertical and lambda)
  // Populated by `evalFResPrecd`
  double fsqr1, fsqz1, fsql1;
  double fsq;

  std::vector<double> mhd_energy;

  // Time-trace of the force at the vacuum boundary (only for free-boundary)
  std::vector<double> delbsq;
  // Time-trace of the restart reasons, for debugging purposes. Each restart is
  // a pair of <iteration, reason> (e.g. to see how many jacobian resets
  // occurred)
  std::vector<RestartReason> restart_reasons;

  double res0;

  std::vector<double> fResInvar;
  std::vector<double> fResPrecd;

 private:
  const int max_threads_;
};

}  // namespace vmecpp

#endif  // VMECPP_COMMON_FLOW_CONTROL_FLOW_CONTROL_H_
