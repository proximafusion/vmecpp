// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#ifndef VMECPP_VMEC_PYBIND11_VMEC_INDATA_PYWRAPPER_H_
#define VMECPP_VMEC_PYBIND11_VMEC_INDATA_PYWRAPPER_H_

#include <Eigen/Dense>
#include <filesystem>
#include <optional>
#include <string>

#include "vmecpp/common/util/util.h"  // RowMatrixXd, ToEigenVector, ToEigenMatrix
#include "vmecpp/common/vmec_indata/vmec_indata.h"

namespace vmecpp {

// A Python-friendly equivalent of VmecINDATA.
// - Eigen matrices and vectors are used instead of STL types for better
//   pybind11 bindings (Eigen types are automatically exposed as numpy arrays)
// - In case of error, exceptions are thrown instead of aborting via CHECKs:
//   this is not the preferred style in C++ but it integrates better with Python
// VmecINDATAPyWrapper can be copied to/from VmecINDATA.
// This adapter can be removed once all of VMEC++ switches to Eigen, tracked at
// https://github.com/proximafusion/repo/issues/2593.
class VmecINDATAPyWrapper {
 public:
  // see VmecINDATA for data member docs.
  bool lasym;
  int nfp;
  int mpol;
  int ntor;
  int ntheta;
  int nzeta;
  Eigen::VectorXi ns_array;
  Eigen::VectorXd ftol_array;
  Eigen::VectorXi niter_array;
  double phiedge;
  int ncurr;
  std::string pmass_type;
  Eigen::VectorXd am;
  Eigen::VectorXd am_aux_s;
  Eigen::VectorXd am_aux_f;
  double pres_scale;
  double gamma;
  double spres_ped;
  std::string piota_type;
  Eigen::VectorXd ai;
  Eigen::VectorXd ai_aux_s;
  Eigen::VectorXd ai_aux_f;
  std::string pcurr_type;
  Eigen::VectorXd ac;
  Eigen::VectorXd ac_aux_s;
  Eigen::VectorXd ac_aux_f;
  double curtor;
  double bloat;
  bool lfreeb;
  std::string mgrid_file;
  Eigen::VectorXd extcur;
  int nvacskip;
  FreeBoundaryMethod free_boundary_method;
  int nstep;
  Eigen::VectorXd aphi;
  double delt;
  double tcon0;
  bool lforbal;
  bool return_outputs_even_if_not_converged;
  Eigen::VectorXd raxis_c;
  Eigen::VectorXd zaxis_s;
  std::optional<Eigen::VectorXd> raxis_s;
  std::optional<Eigen::VectorXd> zaxis_c;
  RowMatrixXd rbc;
  RowMatrixXd zbs;
  std::optional<RowMatrixXd> rbs;
  std::optional<RowMatrixXd> zbc;

  VmecINDATAPyWrapper();
  explicit VmecINDATAPyWrapper(const VmecINDATA& indata);
  explicit operator VmecINDATA() const;

  // Set new values for indata's mpol and/or ntor.
  // Related quantities such as raxis_c and rbc are zero-padded or truncated as
  // needed.
  void SetMpolNtor(int new_mpol, int new_ntor);

  std::string ToJson() const;
  static VmecINDATAPyWrapper FromFile(
      const std::filesystem::path& indata_json_file_path);
  static VmecINDATAPyWrapper FromJson(const std::string& indata_json);

  VmecINDATAPyWrapper Copy() const;
};

}  // namespace vmecpp

#endif  // VMECPP_VMEC_PYBIND11_VMEC_INDATA_PYWRAPPER_H_
