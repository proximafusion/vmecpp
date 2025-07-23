// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#ifndef VMECPP_COMMON_UTIL_UTIL_H_
#define VMECPP_COMMON_UTIL_UTIL_H_

#include <Eigen/Dense>  // VectorXd, Matrix
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <span>
#include <sstream>
#include <string>
#include <vector>

#include "absl/log/check.h"
#include "vmecpp/vmec/vmec_constants/vmec_algorithm_constants.h"

#ifdef _OPENMP
#include <omp.h>
#endif  // _OPENMP

namespace vmecpp {

// Eigen defaults to column-major ordering, but we prefer row-major for two
// reasons:
// - the data will be mostly handled as numpy arrays via pybind11, and numpy's
// default is row-major
// - lots of code in output_quantities.cc expects row-major when iterating over
// elements with a linear index
using RowMatrixXd =
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

inline Eigen::VectorXd ToEigenVector(const std::vector<double> &v) {
  return Eigen::Map<const Eigen::VectorXd>(v.data(),
                                           static_cast<Eigen::Index>(v.size()));
}

inline Eigen::VectorXi ToEigenVector(const std::vector<int> &v) {
  return Eigen::Map<const Eigen::VectorXi>(v.data(),
                                           static_cast<Eigen::Index>(v.size()));
}

inline vmecpp::RowMatrixXd ToEigenMatrix(const std::vector<double> &v,
                                         Eigen::Index size1,
                                         Eigen::Index size2) {
  return Eigen::Map<const vmecpp::RowMatrixXd>(v.data(), size1, size2);
}

// Convert a rectangular nested STL vector to the corresponding Eigen matrix
inline vmecpp::RowMatrixXd ToEigenMatrix(
    const std::vector<std::vector<double>> &v) {
  const std::size_t outer_size = v.size();
  CHECK_GT(outer_size, 0u);
  const std::size_t inner_size = v[0].size();
  for (const auto &row : v) {
    CHECK_EQ(row.size(), inner_size);
  }

  vmecpp::RowMatrixXd m(outer_size, inner_size);

  for (int i = 0; i < m.rows(); ++i) {
    for (int j = 0; j < m.cols(); ++j) {
      m(i, j) = v[i][j];
    }
  }

  return m;
}

enum class VmecCheckpoint : std::uint8_t {
  NONE = 0,

  // ------ initial guess and static members
  RADIAL_PROFILES_EVAL,
  SPECTRAL_CONSTRAINT,
  SETUP_INITIAL_STATE,

  // ------ related to updateEnergy
  FOURIER_GEOMETRY_TO_START_WITH,
  INV_DFT_GEOMETRY,
  JACOBIAN,
  METRIC,
  VOLUME,
  B_CONTRA,
  B_CO,
  ENERGY,

  // ------ related to updateForces
  RADIAL_FORCE_BALANCE,
  HYBRID_LAMBDA_FORCE,
  REALSPACE_FORCES,
  UPDATE_RADIAL_PRECONDITIONER,
  UPDATE_FORCE_NORMS,
  UPDATE_TCON,
  ALIAS,
  FWD_DFT_FORCES,

  // ------ Nestor free-boundary contribution
  VAC1_VACUUM,
  VAC1_SURFACE,
  VAC1_BEXTERN,
  VAC1_ANALYT,
  VAC1_GREENF,
  VAC1_FOURP,
  VAC1_FOURI_SYMM,
  VAC1_FOURI_KV_DFT,
  VAC1_FOURI_KU_DFT,
  VAC1_SOLVER,
  VAC1_BSQVAC,

  RBSQ,

  // ------ back in updateFwdModel
  PHYSICAL_FORCES,
  INVARIANT_RESIDUALS,
  APPLY_M1_PRECONDITIONER,
  ASSEMBLE_RZ_PRECONDITIONER,
  APPLY_RADIAL_PRECONDITIONER,
  PRECONDITIONED_RESIDUALS,

  // ------ closing the iteration loop
  PRINTOUT,
  EVOLVE,

  // ------ interpolation between multi-grid steps
  INTERP,

  // ------ computation of output quantities
  BCOVAR_FILEOUT,
  BSS,
  LOWPASS_BCOVARIANT,
  EXTRAPOLATE_BSUBS,
  JXBOUT,
  MERCIER,
  THREED1_FIRST_TABLE,
  THREED1_GEOMAG,
  THREED1_VOLUMETRICS,
  THREED1_AXIS,
  THREED1_BETAS,
  THREED1_SHAFRANOV_INTEGRALS
};

enum class VmecStatus : std::uint8_t {
  // no fatal error but convergence was not reached
  NORMAL_TERMINATION = 0,
  BAD_JACOBIAN = 1,
  // 2 TODO(jons): not used anymore?
  NCURR_NE_1_BLOAT_NE_1 = 3,
  JACOBIAN_75_TIMES_BAD = 4,
  INPUT_PARSING_ERROR = 5,
  // 6 TODO(jons): not used anymore?
  PHIEDGE_WRONG_SIGN = 7,
  NS_ERROR = 8,    // NS ARRAY MUST NOT BE ALL ZEROES
  MISC_ERROR = 9,  // can happen in mgrid_mod
  VAC_VMEC_ITOR_MISMATCH = 10,
  // everything went well, VMEC++ converged
  SUCCESSFUL_TERMINATION = 11
};

enum class VacuumPressureState : std::int8_t {
  // No vacuum pressure
  kOff = -1,

  // No vacuum pressure yet, force free-boundary update, but ignore the result
  // in this force-balance computation.
  kInitializing = 0,

  // vacuum pressure turned on.
  // soft restart equilibrium calculation by returning BAD_JACOBIAN in the
  // process of reducing rCon0,zCon0 *= 0.9;
  kInitialized = 1,

  // vacuum pressure turned on
  // in the process of reducing rCon0,zCon0 *= 0.9;
  kActive = 2
};

int VmecStatusCode(const VmecStatus vmec_status);

std::string VmecStatusAsString(const VmecStatus vmec_status);

// vacuum magnetic permeability in Vs/Am (CODATA-2018)
// TODO(jons): In the long term, we should use the CODATA value,
// as it is the official value after re-definition of the SI system.
// However, for now, use the old definition for 1:1 comparison against Fortran
// VMEC.
// static constexpr double MU_0 = 1.25663706212e-6;
static constexpr double MU_0 = 4.0e-7 * M_PI;

// ----------------------
// simple math
int signum(int x);

// ----------------------
// tri-diagonal solvers

// Solve a tri-diagonal system of equations:
// for k in range(nRHS):
//
//   a[j]*x[k,j+1] + d[j]*x[k,j] + b[j]*x[k,j-1] = c[k,j] for j = jMin, jMin+1,
//   ..., (jMax-1)
//
// a,d,b contain the tri-diagonal matrix and is modified in-place
// c     contains the RHS on entry and the solution vectors on exit
void TridiagonalSolveSerial(std::vector<double> &m_a, std::vector<double> &m_d,
                            std::vector<double> &m_b,
                            std::vector<std::vector<double>> &m_c, int jMin,
                            int jMax, int nRHS);

// OpenMP-enabled tri-diagonal solver
//
// Solve a tri-diagonal system of equations:
//
// for k in range(nRHS):
//   b[j]*x[k,j-1] + d[j]*x[k,j] + a[j]*x[k,j+1] = c[k,j] for j = jMin, jMin+1,
// ..., (jMax-1)
//
// using the Thomas algorithm from:
//   https://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm
// This works in-place and does not need extra arrays !
//
// a,d,b contain the tri-diagonal matrix and is modified in-place
// c     contains the RHS on entry and the solution vectors on exit
void TridiagonalSolveOpenMP(
    std::vector<double> &m_ar, std::vector<double> &m_dr,
    std::vector<double> &m_br, std::vector<std::span<double>> &m_cr,
    std::vector<double> &m_az, std::vector<double> &m_dz,
    std::vector<double> &m_bz, std::vector<std::span<double>> &m_cz,
    const std::vector<int> &jMin, int jMax, int mnmax, int nRHS,
    std::vector<std::mutex> &m_mutices, int ncpu, int myid, int nsMinF,
    int nsMaxF, std::vector<double> &m_handover_ar,
    std::vector<std::vector<double>> &m_handover_cr,
    std::vector<double> &m_handover_az,
    std::vector<std::vector<double>> &m_handover_cz);

// ----------------------
// VMEC-specific

// Compute the maximum allowed number of threads for a VMEC++ run with given
// radial resolution and adjust the number of OpenMP threads accordingly.
int vmec_adjust_num_threads(int max_threads, int num_surfaces_to_distribute);

}  // namespace vmecpp

#endif  // VMECPP_COMMON_UTIL_UTIL_H_
