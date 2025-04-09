// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#include "vmecpp/vmec/pybind11/vmec_indata_pywrapper.h"

#include <algorithm>  // min
#include <stdexcept>
#include <string>

#include "util/file_io/file_io.h"

namespace {
[[noreturn]] void ErrorToException(const absl::Status& s,
                                   const std::string& where) {
  const std::string msg =
      "There was an error " + where + ":\n" + std::string(s.message());
  throw std::runtime_error(msg);
}
}  // namespace

namespace vmecpp {

VmecINDATAPyWrapper::VmecINDATAPyWrapper()
    : vmecpp::VmecINDATAPyWrapper(VmecINDATA()) {}

VmecINDATAPyWrapper::VmecINDATAPyWrapper(const VmecINDATA& indata)
    : lasym(indata.lasym),
      nfp(indata.nfp),
      mpol(indata.mpol),
      ntor(indata.ntor),
      ntheta(indata.ntheta),
      nzeta(indata.nzeta),
      ns_array(ToEigenVector(indata.ns_array)),
      ftol_array(ToEigenVector(indata.ftol_array)),
      niter_array(ToEigenVector(indata.niter_array)),
      phiedge(indata.phiedge),
      ncurr(indata.ncurr),
      pmass_type(indata.pmass_type),
      am(ToEigenVector(indata.am)),
      am_aux_s(ToEigenVector(indata.am_aux_s)),
      am_aux_f(ToEigenVector(indata.am_aux_f)),
      pres_scale(indata.pres_scale),
      gamma(indata.gamma),
      spres_ped(indata.spres_ped),
      piota_type(indata.piota_type),
      ai(ToEigenVector(indata.ai)),
      ai_aux_s(ToEigenVector(indata.ai_aux_s)),
      ai_aux_f(ToEigenVector(indata.ai_aux_f)),
      pcurr_type(indata.pcurr_type),
      ac(ToEigenVector(indata.ac)),
      ac_aux_s(ToEigenVector(indata.ac_aux_s)),
      ac_aux_f(ToEigenVector(indata.ac_aux_f)),
      curtor(indata.curtor),
      bloat(indata.bloat),
      lfreeb(indata.lfreeb),
      mgrid_file(indata.mgrid_file),
      extcur(ToEigenVector(indata.extcur)),
      nvacskip(indata.nvacskip),
      free_boundary_method(indata.free_boundary_method),
      nstep(indata.nstep),
      aphi(ToEigenVector(indata.aphi)),
      delt(indata.delt),
      tcon0(indata.tcon0),
      lforbal(indata.lforbal),
      return_outputs_even_if_not_converged(
          indata.return_outputs_even_if_not_converged),
      raxis_c(ToEigenVector(indata.raxis_c)),
      zaxis_s(ToEigenVector(indata.zaxis_s)),
      rbc(ToEigenMatrix(indata.rbc, mpol, 2 * ntor + 1)),
      zbs(ToEigenMatrix(indata.zbs, mpol, 2 * ntor + 1)) {
  if (lasym) {
    raxis_s = ToEigenVector(indata.raxis_s);
    zaxis_c = ToEigenVector(indata.zaxis_c);
    rbs = ToEigenMatrix(indata.rbs, mpol, 2 * ntor + 1);
    zbc = ToEigenMatrix(indata.zbc, mpol, 2 * ntor + 1);
  }
}

VmecINDATAPyWrapper::operator VmecINDATA() const {
  VmecINDATA indata;

  indata.lasym = lasym;
  indata.nfp = nfp;
  indata.mpol = mpol;
  indata.ntor = ntor;
  indata.ntheta = ntheta;
  indata.nzeta = nzeta;
  indata.ns_array.assign(ns_array.begin(), ns_array.end());
  indata.ftol_array.assign(ftol_array.begin(), ftol_array.end());
  indata.niter_array.assign(niter_array.begin(), niter_array.end());
  indata.phiedge = phiedge;
  indata.ncurr = ncurr;
  indata.pmass_type = pmass_type;
  indata.am.assign(am.begin(), am.end());
  indata.am_aux_s.assign(am_aux_s.begin(), am_aux_s.end());
  indata.am_aux_f.assign(am_aux_f.begin(), am_aux_f.end());
  indata.pres_scale = pres_scale;
  indata.gamma = gamma;
  indata.spres_ped = spres_ped;
  indata.piota_type = piota_type;
  indata.ai.assign(ai.begin(), ai.end());
  indata.ai_aux_s.assign(ai_aux_s.begin(), ai_aux_s.end());
  indata.ai_aux_f.assign(ai_aux_f.begin(), ai_aux_f.end());
  indata.pcurr_type = pcurr_type;
  indata.ac.assign(ac.begin(), ac.end());
  indata.ac_aux_s.assign(ac_aux_s.begin(), ac_aux_s.end());
  indata.ac_aux_f.assign(ac_aux_f.begin(), ac_aux_f.end());
  indata.curtor = curtor;
  indata.bloat = bloat;
  indata.lfreeb = lfreeb;
  indata.mgrid_file = mgrid_file;
  indata.extcur.assign(extcur.begin(), extcur.end());
  indata.nvacskip = nvacskip;
  indata.free_boundary_method = free_boundary_method;
  indata.nstep = nstep;
  indata.aphi.assign(aphi.begin(), aphi.end());
  indata.delt = delt;
  indata.tcon0 = tcon0;
  indata.lforbal = lforbal;
  indata.return_outputs_even_if_not_converged =
      return_outputs_even_if_not_converged;
  indata.raxis_c.assign(raxis_c.begin(), raxis_c.end());
  indata.zaxis_s.assign(zaxis_s.begin(), zaxis_s.end());
  if (raxis_s.has_value()) {
    indata.raxis_s.assign(raxis_s.value().begin(), raxis_s.value().end());
  }
  if (zaxis_c.has_value()) {
    indata.zaxis_c.assign(zaxis_c.value().begin(), zaxis_c.value().end());
  }

  const auto rbc_flat = rbc.reshaped<Eigen::RowMajor>();
  indata.rbc.assign(rbc_flat.begin(), rbc_flat.end());
  const auto zbs_flat = zbs.reshaped<Eigen::RowMajor>();
  indata.zbs.assign(zbs_flat.begin(), zbs_flat.end());
  if (rbs.has_value()) {
    const auto rbs_flat = rbs.value().reshaped<Eigen::RowMajor>();
    indata.rbs.assign(rbs_flat.begin(), rbs_flat.end());
  }
  if (zbc.has_value()) {
    const auto zbc_flat = zbc.value().reshaped<Eigen::RowMajor>();
    indata.zbc.assign(zbc_flat.begin(), zbc_flat.end());
  }
  return indata;
}

void VmecINDATAPyWrapper::SetMpolNtor(int new_mpol, int new_ntor) {
  using Eigen::VectorXd;

  const bool both_same_as_before = (new_mpol == mpol && new_ntor == ntor);
  if (both_same_as_before) {
    return;  // nothing to do
  }

  VectorXd old_axis_fc = raxis_c;
  const auto shortest_range = Eigen::seq(0, std::min(ntor, new_ntor));

  raxis_c = VectorXd::Zero(new_ntor + 1);
  // Copy back pre-existing elements
  raxis_c(shortest_range) = old_axis_fc(shortest_range);

  old_axis_fc = zaxis_s;
  zaxis_s = VectorXd::Zero(new_ntor + 1);
  zaxis_s(shortest_range) = old_axis_fc(shortest_range);

  if (lasym) {
    old_axis_fc = raxis_s.value();
    raxis_s = VectorXd::Zero(new_ntor + 1);
    (*raxis_s)(shortest_range) = old_axis_fc(shortest_range);

    old_axis_fc = zaxis_c.value();
    zaxis_c = VectorXd::Zero(new_ntor + 1);
    (*zaxis_c)(shortest_range) = old_axis_fc(shortest_range);
  }

  auto resized_2d_coeff = [this, new_mpol, new_ntor](const auto& coeff) {
    const int new_nmax = 2 * new_ntor + 1;
    RowMatrixXd resized_coeff = RowMatrixXd::Zero(new_mpol, new_nmax);

    // copy the original values at the appropriate indices
    const int smaller_ntor = std::min(ntor, new_ntor);
    const int smaller_mpol = std::min(mpol, new_mpol);
    for (int m = 0; m < smaller_mpol; ++m) {
      for (int n = -smaller_ntor; n <= smaller_ntor; ++n) {
        resized_coeff(m, (n + new_ntor)) = coeff(m, n + ntor);
      }
    }

    return resized_coeff;
  };

  rbc = resized_2d_coeff(rbc);
  zbs = resized_2d_coeff(zbs);
  if (lasym) {
    rbs = resized_2d_coeff(rbs.value());
    zbc = resized_2d_coeff(zbc.value());
  }

  mpol = new_mpol;
  ntor = new_ntor;
}

std::string VmecINDATAPyWrapper::ToJson() const {
  const absl::StatusOr<std::string> json = VmecINDATA(*this).ToJson();

  if (!json.ok()) {
    ErrorToException(json.status(), "converting VmecINDATA to JSON");
  }

  return *json;
}

VmecINDATAPyWrapper VmecINDATAPyWrapper::FromFile(
    const std::filesystem::path& indata_json_file_path) {
  absl::StatusOr<std::string> indata_json =
      file_io::ReadFile(indata_json_file_path);

  if (!indata_json.ok()) {
    ErrorToException(
        indata_json.status(),
        "reading JSON file '" + indata_json_file_path.string() + "'");
  }

  absl::StatusOr<VmecINDATA> vmec_indata = VmecINDATA::FromJson(*indata_json);
  if (!vmec_indata.ok()) {
    ErrorToException(vmec_indata.status(),
                     "creating VmecINDATA object from JSON (input file was '" +
                         indata_json_file_path.string() + "')");
  }

  return VmecINDATAPyWrapper(*vmec_indata);
}

VmecINDATAPyWrapper VmecINDATAPyWrapper::FromJson(
    const std::string& indata_json) {
  absl::StatusOr<VmecINDATA> vmec_indata = VmecINDATA::FromJson(indata_json);
  if (!vmec_indata.ok()) {
    ErrorToException(vmec_indata.status(),
                     "creating VmecINDATA object from JSON ");
  }

  return VmecINDATAPyWrapper(*vmec_indata);
}

VmecINDATAPyWrapper VmecINDATAPyWrapper::Copy() const { return *this; }

}  // namespace vmecpp
