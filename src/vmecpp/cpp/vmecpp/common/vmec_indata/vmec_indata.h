// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#ifndef VMECPP_COMMON_VMEC_INDATA_VMEC_INDATA_H_
#define VMECPP_COMMON_VMEC_INDATA_VMEC_INDATA_H_

#include <string>
#include <vector>

#include "H5Cpp.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"

namespace vmecpp {

// default number of flux surfaces
static constexpr int kNsDefault = 31;

// default force tolerance for convergence
static constexpr double kFTolDefault = 1.0e-10;

// default maximum number of iterations
static constexpr int kNIterDefault = 100;

enum class FreeBoundaryMethod : std::uint8_t {
  // use the NEumann Solver for TORoidal systems
  // for the free-boundary force contribution
  NESTOR,

  // use the Boundary Integral Equation Solver for Toroidal systems
  // for the free-boundary force contribution
  BIEST
};

int FreeBoundaryMethodCode(FreeBoundaryMethod free_boundary_method);
absl::StatusOr<FreeBoundaryMethod> FreeBoundaryMethodFromString(
    const std::string& free_boundary_method_string);
std::string ToString(FreeBoundaryMethod free_boundary_method);

// Use this to switch the overall program flow/iteration style
// between VMEC 8.52 (Golden Reference for V&V, and what educational_VMEC is
// based on) and PARVMEC (~same as hiddenSymmetries/VMEC2000) - version 9.0.
enum class IterationStyle : std::uint8_t {
  // VMEC 8.52 (Golden Reference for V&V, and what educational_VMEC is based on)
  VMEC_8_52,

  // PARVMEC (~same as hiddenSymmetries/VMEC2000) - version 9.0
  PARVMEC
};

int IterationStyleCode(IterationStyle iteration_style);
absl::StatusOr<IterationStyle> IterationStyleFromString(
    const std::string& iteration_style_string);
std::string ToString(IterationStyle iteration_style);

// INDATA: user-provided inputs for a stand-alone VMEC run
class VmecINDATA {
 public:
  // ---------------------------------
  // numerical resolution, symmetry assumption

  // flag to indicate non-stellarator-symmetry
  bool lasym;

  // number of toroidal field periods (=1 for Tokamak)
  int nfp;

  // number of poloidal Fourier harmonics; m = 0, 1, ..., (mpol-1)
  int mpol;

  // number of toroidal Fourier harmonics; n = -ntor, -ntor+1, ..., -1, 0, 1,
  // ..., ntor-1, ntor
  int ntor;

  // number of poloidal grid points; if odd: is rounded to next smaller even
  // number
  int ntheta;

  // number of toroidal grid points; must match nzeta of mgrid file if using
  // free-boundary
  int nzeta;

  // ---------------------------------
  // multi-grid steps

  // [numGrids] number of flux surfaces per multigrid step
  std::vector<int> ns_array;

  // [numGrids] requested force tolerance for convergence per multigrid step
  std::vector<double> ftol_array;

  // [numGrids] maximum number of iterations per multigrid step
  std::vector<int> niter_array;

  // ---------------------------------
  // global physics parameters

  // total enclosed toroidal magnetic flux in Vs == Wb
  double phiedge;

  // select constraint on iota or enclosed toroidal current profiles
  // 0: constrained-iota; 1: constrained-current
  int ncurr;

  // ---------------------------------
  // mass / pressure profile

  // parametrization of mass/pressure profile
  std::string pmass_type;

  // [amLen] mass/pressure profile coefficients
  std::vector<double> am;

  // [am_auxLen] spline mass/pressure profile: knot locations in s
  std::vector<double> am_aux_s;

  // [am_auxLen] spline mass/pressure profile: values at knots
  std::vector<double> am_aux_f;

  // global scaling factor for mass/pressure profile
  double pres_scale;

  // adiabatic index
  double gamma;

  // location of pressure pedestal in s
  double spres_ped;

  // ---------------------------------
  // (initial guess for) iota profile

  // parametrization of iota profile
  std::string piota_type;

  // [aiLen] iota profile coefficients
  std::vector<double> ai;

  // [ai_auxLen] spline iota profile: knot locations in s
  std::vector<double> ai_aux_s;

  // [ai_auxLen] spline iota profile: values at knots
  std::vector<double> ai_aux_f;

  // ---------------------------------
  // enclosed toroidal current profile

  // parametrization of toroidal current profile
  std::string pcurr_type;

  // enclosed toroidal current profile coefficients
  std::vector<double> ac;

  // [ac_auxLen] spline toroidal current profile: knot locations in s
  std::vector<double> ac_aux_s;

  // [ac_auxLen] spline toroidal current profile: values at knots
  std::vector<double> ac_aux_f;

  // toroidal current in A
  double curtor;

  // bloating factor (for constrained toroidal current)
  double bloat;

  // ---------------------------------
  // free-boundary parameters

  // flag to indicate free-boundary
  bool lfreeb;

  // full path for vacuum Green's function data
  std::string mgrid_file;

  // [extcurLen] coil currents in A
  std::vector<double> extcur;

  // number of iterations between full vacuum calculations
  int nvacskip;

  // indicates which method to use
  // for the free-boundary force contribution
  FreeBoundaryMethod free_boundary_method;

  // ---------------------------------
  // tweaking parameters

  // printout interval
  int nstep;

  // [aphiLen] radial flux zoning profile coefficients
  std::vector<double> aphi;

  // initial value for artificial time step in iterative solver
  double delt;

  // constraint force scaling factor for ns --> 0
  double tcon0;

  // hack: directly compute innermost flux surface geometry from radial force
  // balance
  bool lforbal;

  // allows to switch between VMEC 8.52 and PARVMEC iteration style
  // default: VMEC 8.52 (Golden Reference for V&V, and what educational_VMEC is
  // based on)
  IterationStyle iteration_style;

  // If true, return the outputs even if VMEC++ did not converge.
  // Otherwise (default = false), an absl::InternalError will be returned.
  bool return_outputs_even_if_not_converged;

  // ---------------------------------
  // initial guess for magnetic axis

  // [ntor+1] magnetic axis coefficients for R ~ cos(n*v); stellarator-symmetric
  std::vector<double> raxis_c;

  // [ntor+1] magnetic axis coefficients for Z ~ sin(n*v); stellarator-symmetric
  std::vector<double> zaxis_s;

  // [ntor+1] magnetic axis coefficients for R ~ sin(n*v);
  // non-stellarator-symmetric
  // TODO(jurasic) make theses optional after the Eigen3 refactor
  std::vector<double> raxis_s;

  // [ntor+1] magnetic axis coefficients for Z ~ cos(n*v);
  // non-stellarator-symmetric
  // TODO(jurasic) make theses optional after the Eigen3 refactor
  std::vector<double> zaxis_c;

  // ---------------------------------
  // (initial guess for) boundary shape

  // [mpol*(2*ntor+1)] boundary coefficients for R ~ cos(m*u - n*v);
  // stellarator-symmetric
  std::vector<double> rbc;

  // [mpol*(2*ntor+1)] boundary coefficients for Z ~ sin(m*u - n*v);
  // stellarator-symmetric
  std::vector<double> zbs;

  // [mpol*(2*ntor+1)] boundary coefficients for R ~ sin(m*u - n*v);
  // non-stellarator-symmetric
  // TODO(jurasic) make theses optional after the Eigen3 refactor
  std::vector<double> rbs;

  // [mpol*(2*ntor+1)] boundary coefficients for Z ~ cos(m*u - n*v);
  // non-stellarator-symmetric
  // TODO(jurasic) make theses optional after the Eigen3 refactor
  std::vector<double> zbc;

  // Construct a VmecINDATA instance with default values, except for profile
  // coefficients or knot locations/values. Axis geometry coefficients and
  // boundary geometry coefficients are zero-initialized.
  VmecINDATA();

  bool operator==(const VmecINDATA&) const = default;
  bool operator!=(const VmecINDATA& o) const { return !(*this == o); }

  // Write object to the specified HDF5 file, under key "indata".
  absl::Status WriteTo(H5::H5File& file) const;

  // Load contents of `from_file` into the specified instance.
  // The file is expected to have the same schema as the one produced by
  // WriteTo.
  static absl::Status LoadInto(VmecINDATA& m_obj, H5::H5File& from_file);

  static absl::StatusOr<VmecINDATA> FromJson(const std::string& indata_json);

  absl::StatusOr<std::string> ToJson() const;
};

absl::Status IsConsistent(const VmecINDATA& vmec_indata,
                          bool enable_info_messages);

}  // namespace vmecpp

#endif  // VMECPP_COMMON_VMEC_INDATA_VMEC_INDATA_H_
