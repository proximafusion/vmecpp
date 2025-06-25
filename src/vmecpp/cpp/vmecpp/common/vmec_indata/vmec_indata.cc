// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#include "vmecpp/common/vmec_indata/vmec_indata.h"

#include <algorithm>
#include <cassert>
#include <string>
#include <vector>

#include "H5Cpp.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "nlohmann/json.hpp"
#include "util/hdf5_io/hdf5_io.h"
#include "util/json_io/json_io.h"
#include "vmecpp/common/util/util.h"
#include "vmecpp/common/vmec_indata/boundary_from_json.h"

namespace vmecpp {

using nlohmann::json;

using json_io::JsonReadBool;
using json_io::JsonReadDouble;
using json_io::JsonReadInt;
using json_io::JsonReadString;
using json_io::JsonReadVectorDouble;
using json_io::JsonReadVectorInt;

int FreeBoundaryMethodCode(FreeBoundaryMethod free_boundary_method) {
  // from https://stackoverflow.com/a/11421471
  return static_cast<std::underlying_type_t<FreeBoundaryMethod>>(
      free_boundary_method);
}  // FreeBoundaryMethodCode

absl::StatusOr<FreeBoundaryMethod> FreeBoundaryMethodFromString(
    const std::string& free_boundary_method_string) {
  if (free_boundary_method_string == "nestor") {
    return FreeBoundaryMethod::NESTOR;
  } else if (free_boundary_method_string == "biest") {
    return FreeBoundaryMethod::BIEST;
  }
  return absl::NotFoundError(absl::StrCat("free boundary method named '",
                                          free_boundary_method_string,
                                          "' not known"));
}  // FreeBoundaryMethodFromString

std::string ToString(FreeBoundaryMethod free_boundary_method) {
  switch (free_boundary_method) {
    case FreeBoundaryMethod::NESTOR:
      return "nestor";
    case FreeBoundaryMethod::BIEST:
      return "biest";
    default:
      LOG(FATAL)
          << "no string conversion implemented yet for FreeBoundaryMethod code "
          << FreeBoundaryMethodCode(free_boundary_method);
  }
}  // ToString

int IterationStyleCode(IterationStyle iteration_style) {
  // from https://stackoverflow.com/a/11421471
  return static_cast<std::underlying_type_t<IterationStyle>>(iteration_style);
}  // IterationStyleCode

absl::StatusOr<IterationStyle> IterationStyleFromString(
    const std::string& iteration_style_string) {
  if (iteration_style_string == "vmec_8_52") {
    return IterationStyle::VMEC_8_52;
  } else if (iteration_style_string == "parvmec") {
    return IterationStyle::PARVMEC;
  }
  return absl::NotFoundError(absl::StrCat(
      "iteration style named '", iteration_style_string, "' not known"));
}  // IterationStyleFromString

std::string ToString(IterationStyle iteration_style) {
  switch (iteration_style) {
    case IterationStyle::VMEC_8_52:
      return "vmec_8_52";
    case IterationStyle::PARVMEC:
      return "parvmec";
    default:
      LOG(FATAL)
          << "no string conversion implemented yet for IterationStyle code "
          << IterationStyleCode(iteration_style);
  }
}  // ToString

VmecINDATA::VmecINDATA() {
  // numerical resolution, symmetry assumption
  lasym = false;
  nfp = 1;
  mpol = 6;
  ntor = 0;
  ntheta = 0;
  nzeta = 0;

  // multi-grid steps
  ns_array = {kNsDefault};
  ftol_array = {kFTolDefault};
  niter_array = {kNIterDefault};

  // global physics parameters
  phiedge = 1.0;
  ncurr = 0;

  // mass / pressure profile
  pmass_type = "power_series";
  // am left empty
  // am_aux_s left empty
  // am_aux_f left empty
  pres_scale = 1.0;
  gamma = 0.0;
  spres_ped = 1.0;

  // (initial guess for) iota profile
  piota_type = "power_series";
  // ai left empty
  // ai_aux_s left empty
  // ai_aux_f left empty

  // enclosed toroidal current profile
  pcurr_type = "power_series";
  // ac left empty
  // ac_aux_s left empty
  // ac_aux_f left empty
  curtor = 0.0;
  bloat = 1.0;

  // free-boundary parameters
  lfreeb = false;
  mgrid_file = "NONE";  // default from Fortran VMEC via indata2json
  // extcur is left empty
  nvacskip = 1;
  free_boundary_method = FreeBoundaryMethod::NESTOR;

  // tweaking parameters
  nstep = 10;
  aphi = {1.0};
  delt = 1.0;
  tcon0 = 1.0;
  lforbal = false;
  iteration_style = IterationStyle::VMEC_8_52;
  return_outputs_even_if_not_converged = false;

  // zero-initialized magnetic axis
  raxis_c.resize(ntor + 1);
  zaxis_s.resize(ntor + 1);
  if (lasym) {
    raxis_s.resize(ntor + 1);
    zaxis_c.resize(ntor + 1);
  }

  // zero-initialized boundary shape
  const int bdy_size = mpol * (2 * ntor + 1);
  rbc.resize(bdy_size);
  zbs.resize(bdy_size);
  if (lasym) {
    rbs.resize(bdy_size);
    zbc.resize(bdy_size);
  }
}

// Write object to the specified HDF5 file, under key "indata".
absl::Status VmecINDATA::WriteTo(H5::H5File& file) const {
  using hdf5_io::WriteH5Dataset;
  file.createGroup("/indata");

  // scalars and strings
  WriteH5Dataset(lasym, "/indata/lasym", file);
  WriteH5Dataset(nfp, "/indata/nfp", file);
  WriteH5Dataset(mpol, "/indata/mpol", file);
  WriteH5Dataset(ntor, "/indata/ntor", file);
  WriteH5Dataset(ntheta, "/indata/ntheta", file);
  WriteH5Dataset(nzeta, "/indata/nzeta", file);
  WriteH5Dataset(phiedge, "/indata/phiedge", file);
  WriteH5Dataset(ncurr, "/indata/ncurr", file);
  WriteH5Dataset(pmass_type, "/indata/pmass_type", file);
  WriteH5Dataset(pres_scale, "/indata/pres_scale", file);
  WriteH5Dataset(gamma, "/indata/gamma", file);
  WriteH5Dataset(spres_ped, "/indata/spres_ped", file);
  WriteH5Dataset(piota_type, "/indata/piota_type", file);
  WriteH5Dataset(pcurr_type, "/indata/pcurr_type", file);
  WriteH5Dataset(curtor, "/indata/curtor", file);
  WriteH5Dataset(bloat, "/indata/bloat", file);
  WriteH5Dataset(lfreeb, "/indata/lfreeb", file);
  WriteH5Dataset(mgrid_file, "/indata/mgrid_file", file);
  WriteH5Dataset(nvacskip, "/indata/nvacskip", file);

  // special treatment for enums
  WriteH5Dataset(ToString(free_boundary_method), "/indata/free_boundary_method",
                 file);
  WriteH5Dataset(ToString(iteration_style), "/indata/iteration_style", file);

  WriteH5Dataset(nstep, "/indata/nstep", file);
  WriteH5Dataset(delt, "/indata/delt", file);
  WriteH5Dataset(tcon0, "/indata/tcon0", file);
  WriteH5Dataset(lforbal, "/indata/lforbal", file);
  WriteH5Dataset(return_outputs_even_if_not_converged,
                 "/indata/return_outputs_even_if_not_converged", file);

  // 1D arrays
  WriteH5Dataset(ns_array, "/indata/ns_array", file);
  WriteH5Dataset(ftol_array, "/indata/ftol_array", file);
  WriteH5Dataset(niter_array, "/indata/niter_array", file);
  WriteH5Dataset(am, "/indata/am", file);
  WriteH5Dataset(am_aux_s, "/indata/am_aux_s", file);
  WriteH5Dataset(am_aux_f, "/indata/am_aux_f", file);
  WriteH5Dataset(ai, "/indata/ai", file);
  WriteH5Dataset(ai_aux_s, "/indata/ai_aux_s", file);
  WriteH5Dataset(ai_aux_f, "/indata/ai_aux_f", file);
  WriteH5Dataset(ac, "/indata/ac", file);
  WriteH5Dataset(ac_aux_s, "/indata/ac_aux_s", file);
  WriteH5Dataset(ac_aux_f, "/indata/ac_aux_f", file);
  WriteH5Dataset(extcur, "/indata/extcur", file);
  WriteH5Dataset(aphi, "/indata/aphi", file);
  WriteH5Dataset(raxis_c, "/indata/raxis_c", file);
  WriteH5Dataset(zaxis_s, "/indata/zaxis_s", file);
  WriteH5Dataset(raxis_s, "/indata/raxis_s", file);
  WriteH5Dataset(zaxis_c, "/indata/zaxis_c", file);

  // 2D matrices (represented as 1D std::vectors)
  // All have dimensions (mpol, 2*ntor+1)
  const auto rbc_view =
      Eigen::Map<const RowMatrixXd>(rbc.data(), mpol, 2 * ntor + 1);
  WriteH5Dataset(rbc_view, "/indata/rbc", file);
  const auto zbs_view =
      Eigen::Map<const RowMatrixXd>(zbs.data(), mpol, 2 * ntor + 1);
  WriteH5Dataset(zbs_view, "/indata/zbs", file);

  const int mpol_asym = lasym ? mpol : 0;
  const int ntor2p1_asym = lasym ? 2 * ntor + 1 : 0;
  const auto rbs_view =
      Eigen::Map<const RowMatrixXd>(rbs.data(), mpol_asym, ntor2p1_asym);
  WriteH5Dataset(rbs_view, "/indata/rbs", file);
  const auto zbc_view =
      Eigen::Map<const RowMatrixXd>(zbc.data(), mpol_asym, ntor2p1_asym);
  WriteH5Dataset(zbc_view, "/indata/zbc", file);

  return absl::OkStatus();
}

// Load contents of `from_file` into the specified instance.
// The file is expected to have the same schema as the one produced by
// WriteTo.
absl::Status VmecINDATA::LoadInto(VmecINDATA& m_indata, H5::H5File& from_file) {
  using hdf5_io::ReadH5Dataset;

  // scalars
  ReadH5Dataset(m_indata.lasym, "/indata/lasym", from_file);
  ReadH5Dataset(m_indata.nfp, "/indata/nfp", from_file);
  ReadH5Dataset(m_indata.mpol, "/indata/mpol", from_file);
  ReadH5Dataset(m_indata.ntor, "/indata/ntor", from_file);
  ReadH5Dataset(m_indata.ntheta, "/indata/ntheta", from_file);
  ReadH5Dataset(m_indata.nzeta, "/indata/nzeta", from_file);
  ReadH5Dataset(m_indata.phiedge, "/indata/phiedge", from_file);
  ReadH5Dataset(m_indata.ncurr, "/indata/ncurr", from_file);
  ReadH5Dataset(m_indata.pmass_type, "/indata/pmass_type", from_file);
  ReadH5Dataset(m_indata.pres_scale, "/indata/pres_scale", from_file);
  ReadH5Dataset(m_indata.gamma, "/indata/gamma", from_file);
  ReadH5Dataset(m_indata.spres_ped, "/indata/spres_ped", from_file);
  ReadH5Dataset(m_indata.piota_type, "/indata/piota_type", from_file);
  ReadH5Dataset(m_indata.pcurr_type, "/indata/pcurr_type", from_file);
  ReadH5Dataset(m_indata.curtor, "/indata/curtor", from_file);
  ReadH5Dataset(m_indata.bloat, "/indata/bloat", from_file);
  ReadH5Dataset(m_indata.lfreeb, "/indata/lfreeb", from_file);
  ReadH5Dataset(m_indata.mgrid_file, "/indata/mgrid_file", from_file);
  ReadH5Dataset(m_indata.nvacskip, "/indata/nvacskip", from_file);

  // special treatment for enums
  std::string fbdy_method_str;
  ReadH5Dataset(fbdy_method_str, "/indata/free_boundary_method", from_file);
  const auto maybe_fbdy_method = FreeBoundaryMethodFromString(fbdy_method_str);
  if (!maybe_fbdy_method.ok()) {
    return maybe_fbdy_method.status();
  }
  m_indata.free_boundary_method = maybe_fbdy_method.value();

  // Legacy way of checking for dataset existence
  // TODO(jons) replace with from_file.nameExists when we get a newer HDF5
  // version in pip wheels
  if (H5Lexists(from_file.getId(), "/indata/iteration_style", 0) == 1) {
    std::string iteration_style_str;
    ReadH5Dataset(iteration_style_str, "/indata/iteration_style", from_file);
    const auto maybe_iteration_style =
        IterationStyleFromString(iteration_style_str);
    if (!maybe_iteration_style.ok()) {
      return maybe_iteration_style.status();
    }
    m_indata.iteration_style = maybe_iteration_style.value();
  } else {
    // fall back to default value
    m_indata.iteration_style = IterationStyle::VMEC_8_52;
  }

  ReadH5Dataset(m_indata.nstep, "/indata/nstep", from_file);
  ReadH5Dataset(m_indata.delt, "/indata/delt", from_file);
  ReadH5Dataset(m_indata.tcon0, "/indata/tcon0", from_file);
  ReadH5Dataset(m_indata.lforbal, "/indata/lforbal", from_file);

  // Legacy way of checking for dataset existence
  if (H5Lexists(from_file.getId(),
                "/indata/return_outputs_even_if_not_converged", 0) == 1) {
    ReadH5Dataset(m_indata.return_outputs_even_if_not_converged,
                  "/indata/return_outputs_even_if_not_converged", from_file);
  } else {
    m_indata.return_outputs_even_if_not_converged = false;
  }

  // 1D arrays
  ReadH5Dataset(m_indata.ns_array, "/indata/ns_array", from_file);
  ReadH5Dataset(m_indata.ftol_array, "/indata/ftol_array", from_file);
  ReadH5Dataset(m_indata.niter_array, "/indata/niter_array", from_file);
  ReadH5Dataset(m_indata.am, "/indata/am", from_file);
  ReadH5Dataset(m_indata.am_aux_s, "/indata/am_aux_s", from_file);
  ReadH5Dataset(m_indata.am_aux_f, "/indata/am_aux_f", from_file);
  ReadH5Dataset(m_indata.ai, "/indata/ai", from_file);
  ReadH5Dataset(m_indata.ai_aux_s, "/indata/ai_aux_s", from_file);
  ReadH5Dataset(m_indata.ai_aux_f, "/indata/ai_aux_f", from_file);
  ReadH5Dataset(m_indata.ac, "/indata/ac", from_file);
  ReadH5Dataset(m_indata.ac_aux_s, "/indata/ac_aux_s", from_file);
  ReadH5Dataset(m_indata.ac_aux_f, "/indata/ac_aux_f", from_file);
  ReadH5Dataset(m_indata.extcur, "/indata/extcur", from_file);
  ReadH5Dataset(m_indata.aphi, "/indata/aphi", from_file);
  ReadH5Dataset(m_indata.raxis_c, "/indata/raxis_c", from_file);
  ReadH5Dataset(m_indata.zaxis_s, "/indata/zaxis_s", from_file);
  ReadH5Dataset(m_indata.raxis_s, "/indata/raxis_s", from_file);
  ReadH5Dataset(m_indata.zaxis_c, "/indata/zaxis_c", from_file);

  // 2D matrices (represented as 1D std::vectors)
  // All have dimensions (mpol, 2*ntor+1)
  // NOTE: we read into a RowMatrixXd and then copy into std::vectors for
  // simplicity. In the future we expect VmecINDATA's data members will switch
  // to Eigen types and the extra copy will evaporate.
  RowMatrixXd tmp_matrix;
  const int linear_size = m_indata.mpol * (2 * m_indata.ntor + 1);

  ReadH5Dataset(tmp_matrix, "/indata/rbc", from_file);
  assert(tmp_matrix.size() == linear_size);
  m_indata.rbc.resize(linear_size);
  std::copy(tmp_matrix.data(), tmp_matrix.data() + tmp_matrix.size(),
            m_indata.rbc.begin());

  ReadH5Dataset(tmp_matrix, "/indata/zbs", from_file);
  assert(tmp_matrix.size() == linear_size);
  m_indata.zbs.resize(linear_size);
  std::copy(tmp_matrix.data(), tmp_matrix.data() + tmp_matrix.size(),
            m_indata.zbs.begin());

  if (m_indata.lasym) {
    ReadH5Dataset(tmp_matrix, "/indata/rbs", from_file);
    assert(tmp_matrix.size() == linear_size);
    m_indata.rbs.resize(linear_size);
    std::copy(tmp_matrix.data(), tmp_matrix.data() + tmp_matrix.size(),
              m_indata.rbs.begin());
    ReadH5Dataset(tmp_matrix, "/indata/zbc", from_file);
    assert(tmp_matrix.size() == linear_size);
    m_indata.zbc.resize(linear_size);
    std::copy(tmp_matrix.data(), tmp_matrix.data() + tmp_matrix.size(),
              m_indata.zbc.begin());
  }

  return absl::OkStatus();
}

absl::StatusOr<VmecINDATA> VmecINDATA::FromJson(
    const std::string& indata_json) {
  json j = json::parse(indata_json);

  if (!j.is_object()) {
    return absl::InvalidArgumentError("root JSON element is not an object");
  }

  // Similar to Fortran VMEC, we start with default values
  // that are overwritten by what is in the input file.
  VmecINDATA vmec_indata;

  // -----------------------------------------------

  auto maybe_lasym = JsonReadBool(j, "lasym");
  if (!maybe_lasym.ok()) {
    return maybe_lasym.status();
  }
  if (maybe_lasym->has_value()) {
    vmec_indata.lasym = maybe_lasym->value();
  }

  auto maybe_nfp = JsonReadInt(j, "nfp");
  if (!maybe_nfp.ok()) {
    return maybe_nfp.status();
  }
  if (maybe_nfp->has_value()) {
    vmec_indata.nfp = maybe_nfp->value();
  }

  auto maybe_mpol = JsonReadInt(j, "mpol");
  if (!maybe_mpol.ok()) {
    return maybe_mpol.status();
  }
  if (maybe_mpol->has_value()) {
    vmec_indata.mpol = maybe_mpol->value();
  }

  auto maybe_ntor = JsonReadInt(j, "ntor");
  if (!maybe_ntor.ok()) {
    return maybe_ntor.status();
  }
  if (maybe_ntor->has_value()) {
    vmec_indata.ntor = maybe_ntor->value();
  }

  auto maybe_ntheta = JsonReadInt(j, "ntheta");
  if (!maybe_ntheta.ok()) {
    return maybe_ntheta.status();
  }
  if (maybe_ntheta->has_value()) {
    vmec_indata.ntheta = maybe_ntheta->value();
  }

  auto maybe_nzeta = JsonReadInt(j, "nzeta");
  if (!maybe_nzeta.ok()) {
    return maybe_nzeta.status();
  }
  if (maybe_nzeta->has_value()) {
    vmec_indata.nzeta = maybe_nzeta->value();
  }

  // -----------------------------------------------

  auto maybe_ns_array = JsonReadVectorInt(j, "ns_array");
  if (!maybe_ns_array.ok()) {
    return maybe_ns_array.status();
  }
  if (maybe_ns_array->has_value()) {
    vmec_indata.ns_array = maybe_ns_array->value();
  }

  auto maybe_ftol_array = JsonReadVectorDouble(j, "ftol_array");
  if (!maybe_ftol_array.ok()) {
    return maybe_ftol_array.status();
  }
  if (maybe_ftol_array->has_value()) {
    vmec_indata.ftol_array = maybe_ftol_array->value();
  }

  if (vmec_indata.ftol_array.size() != vmec_indata.ns_array.size()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "length of ftol_array (%ld) does not match length of ns_array (%ld)\n",
        vmec_indata.ftol_array.size(), vmec_indata.ns_array.size()));
  }

  auto maybe_niter_array = JsonReadVectorInt(j, "niter_array");
  if (!maybe_niter_array.ok()) {
    return maybe_niter_array.status();
  }
  if (maybe_niter_array->has_value()) {
    vmec_indata.niter_array = maybe_niter_array->value();
  }

  if (vmec_indata.niter_array.size() != vmec_indata.ns_array.size()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "length of niter_array (%ld) does not match length of ns_array (%ld)\n",
        vmec_indata.niter_array.size(), vmec_indata.ns_array.size()));
  }

  // -----------------------------------------------

  auto maybe_phiedge = JsonReadDouble(j, "phiedge");
  if (!maybe_phiedge.ok()) {
    return maybe_phiedge.status();
  }
  if (maybe_phiedge->has_value()) {
    vmec_indata.phiedge = maybe_phiedge->value();
  }

  auto maybe_ncurr = JsonReadInt(j, "ncurr");
  if (!maybe_ncurr.ok()) {
    return maybe_ncurr.status();
  }
  if (maybe_ncurr->has_value()) {
    vmec_indata.ncurr = maybe_ncurr->value();
  }

  // -----------------------------------------------

  auto maybe_pmass_type = JsonReadString(j, "pmass_type");
  if (!maybe_pmass_type.ok()) {
    return maybe_pmass_type.status();
  }
  if (maybe_pmass_type->has_value()) {
    vmec_indata.pmass_type = maybe_pmass_type->value();
  }

  auto maybe_am = JsonReadVectorDouble(j, "am");
  if (!maybe_am.ok()) {
    return maybe_am.status();
  }
  if (maybe_am->has_value()) {
    vmec_indata.am = maybe_am->value();
  }

  auto maybe_am_aux_s = JsonReadVectorDouble(j, "am_aux_s");
  if (!maybe_am_aux_s.ok()) {
    return maybe_am_aux_s.status();
  }
  if (maybe_am_aux_s->has_value()) {
    vmec_indata.am_aux_s = maybe_am_aux_s->value();
  }

  auto maybe_am_aux_f = JsonReadVectorDouble(j, "am_aux_f");
  if (!maybe_am_aux_f.ok()) {
    return maybe_am_aux_f.status();
  }
  if (maybe_am_aux_f->has_value()) {
    vmec_indata.am_aux_f = maybe_am_aux_f->value();
  }

  if (vmec_indata.am_aux_f.size() != vmec_indata.am_aux_s.size()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "length of am_aux_f (%ld) does not match length of am_aux_s (%ld)\n",
        vmec_indata.am_aux_f.size(), vmec_indata.am_aux_s.size()));
  }

  auto maybe_pres_scale = JsonReadDouble(j, "pres_scale");
  if (!maybe_pres_scale.ok()) {
    return maybe_pres_scale.status();
  }
  if (maybe_pres_scale->has_value()) {
    vmec_indata.pres_scale = maybe_pres_scale->value();
  }

  auto maybe_gamma = JsonReadDouble(j, "gamma");
  if (!maybe_gamma.ok()) {
    return maybe_gamma.status();
  }
  if (maybe_gamma->has_value()) {
    vmec_indata.gamma = maybe_gamma->value();
  }

  auto maybe_spres_ped = JsonReadDouble(j, "spres_ped");
  if (!maybe_spres_ped.ok()) {
    return maybe_spres_ped.status();
  }
  if (maybe_spres_ped->has_value()) {
    vmec_indata.spres_ped = maybe_spres_ped->value();
  }

  // -----------------------------------------------

  auto maybe_piota_type = JsonReadString(j, "piota_type");
  if (!maybe_piota_type.ok()) {
    return maybe_piota_type.status();
  }
  if (maybe_piota_type->has_value()) {
    vmec_indata.piota_type = maybe_piota_type->value();
  }

  auto maybe_ai = JsonReadVectorDouble(j, "ai");
  if (!maybe_ai.ok()) {
    return maybe_ai.status();
  }
  if (maybe_ai->has_value()) {
    vmec_indata.ai = maybe_ai->value();
  }

  auto maybe_ai_aux_s = JsonReadVectorDouble(j, "ai_aux_s");
  if (!maybe_ai_aux_s.ok()) {
    return maybe_ai_aux_s.status();
  }
  if (maybe_ai_aux_s->has_value()) {
    vmec_indata.ai_aux_s = maybe_ai_aux_s->value();
  }

  auto maybe_ai_aux_f = JsonReadVectorDouble(j, "ai_aux_f");
  if (!maybe_ai_aux_f.ok()) {
    return maybe_ai_aux_f.status();
  }
  if (maybe_ai_aux_f->has_value()) {
    vmec_indata.ai_aux_f = maybe_ai_aux_f->value();
  }

  if (vmec_indata.ai_aux_f.size() != vmec_indata.ai_aux_s.size()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "length of ai_aux_f (%ld) does not match length of ai_aux_s (%ld)\n",
        vmec_indata.ai_aux_f.size(), vmec_indata.ai_aux_s.size()));
  }

  // -----------------------------------------------

  auto maybe_pcurr_type = JsonReadString(j, "pcurr_type");
  if (!maybe_pcurr_type.ok()) {
    return maybe_pcurr_type.status();
  }
  if (maybe_pcurr_type->has_value()) {
    vmec_indata.pcurr_type = maybe_pcurr_type->value();
  }

  auto maybe_ac = JsonReadVectorDouble(j, "ac");
  if (!maybe_ac.ok()) {
    return maybe_ac.status();
  }
  if (maybe_ac->has_value()) {
    vmec_indata.ac = maybe_ac->value();
  }

  auto maybe_ac_aux_s = JsonReadVectorDouble(j, "ac_aux_s");
  if (!maybe_ac_aux_s.ok()) {
    return maybe_ac_aux_s.status();
  }
  if (maybe_ac_aux_s->has_value()) {
    vmec_indata.ac_aux_s = maybe_ac_aux_s->value();
  }

  auto maybe_ac_aux_f = JsonReadVectorDouble(j, "ac_aux_f");
  if (!maybe_ac_aux_f.ok()) {
    return maybe_ac_aux_f.status();
  }
  if (maybe_ac_aux_f->has_value()) {
    vmec_indata.ac_aux_f = maybe_ac_aux_f->value();
  }

  if (vmec_indata.ac_aux_f.size() != vmec_indata.ac_aux_s.size()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "length of ac_aux_f (%ld) does not match length of ac_aux_s (%ld)\n",
        vmec_indata.ac_aux_f.size(), vmec_indata.ac_aux_s.size()));
  }

  auto maybe_curtor = JsonReadDouble(j, "curtor");
  if (!maybe_curtor.ok()) {
    return maybe_curtor.status();
  }
  if (maybe_curtor->has_value()) {
    vmec_indata.curtor = maybe_curtor->value();
  }

  auto maybe_bloat = JsonReadDouble(j, "bloat");
  if (!maybe_bloat.ok()) {
    return maybe_bloat.status();
  }
  if (maybe_bloat->has_value()) {
    vmec_indata.bloat = maybe_bloat->value();
  }

  // -----------------------------------------------

  auto maybe_lfreeb = JsonReadBool(j, "lfreeb");
  if (!maybe_lfreeb.ok()) {
    return maybe_lfreeb.status();
  }
  if (maybe_lfreeb->has_value()) {
    vmec_indata.lfreeb = maybe_lfreeb->value();
  }

  auto maybe_mgrid_file = JsonReadString(j, "mgrid_file");
  if (!maybe_mgrid_file.ok()) {
    return maybe_mgrid_file.status();
  }
  if (maybe_mgrid_file->has_value()) {
    vmec_indata.mgrid_file = maybe_mgrid_file->value();
  }

  auto maybe_extcur = JsonReadVectorDouble(j, "extcur");
  if (!maybe_extcur.ok()) {
    return maybe_extcur.status();
  }
  if (maybe_extcur->has_value()) {
    vmec_indata.extcur = maybe_extcur->value();
  }

  auto maybe_nvacskip = JsonReadInt(j, "nvacskip");
  if (!maybe_nvacskip.ok()) {
    return maybe_nvacskip.status();
  }
  if (maybe_nvacskip->has_value()) {
    vmec_indata.nvacskip = maybe_nvacskip->value();
  }

  auto maybe_free_boundary_method = JsonReadString(j, "free_boundary_method");
  if (!maybe_free_boundary_method.ok()) {
    return maybe_free_boundary_method.status();
  }
  if (maybe_free_boundary_method->has_value()) {
    absl::StatusOr<FreeBoundaryMethod> status_or_free_boundary_method =
        FreeBoundaryMethodFromString(maybe_free_boundary_method->value());
    if (status_or_free_boundary_method.ok()) {
      vmec_indata.free_boundary_method = status_or_free_boundary_method.value();
    } else {
      return status_or_free_boundary_method.status();
    }
  }

  // -----------------------------------------------

  auto maybe_nstep = JsonReadInt(j, "nstep");
  if (!maybe_nstep.ok()) {
    return maybe_nstep.status();
  }
  if (maybe_nstep->has_value()) {
    vmec_indata.nstep = maybe_nstep->value();
  }

  auto maybe_aphi = JsonReadVectorDouble(j, "aphi");
  if (!maybe_aphi.ok()) {
    return maybe_aphi.status();
  }
  if (maybe_aphi->has_value()) {
    vmec_indata.aphi = maybe_aphi->value();
  }

  auto maybe_delt = JsonReadDouble(j, "delt");
  if (!maybe_delt.ok()) {
    return maybe_delt.status();
  }
  if (maybe_delt->has_value()) {
    vmec_indata.delt = maybe_delt->value();
  }

  auto maybe_tcon0 = JsonReadDouble(j, "tcon0");
  if (!maybe_tcon0.ok()) {
    return maybe_tcon0.status();
  }
  if (maybe_tcon0->has_value()) {
    vmec_indata.tcon0 = maybe_tcon0->value();
  }

  auto maybe_lforbal = JsonReadBool(j, "lforbal");
  if (!maybe_lforbal.ok()) {
    return maybe_lforbal.status();
  }
  if (maybe_lforbal->has_value()) {
    vmec_indata.lforbal = maybe_lforbal->value();
  }

  auto maybe_iteration_style = JsonReadString(j, "iteration_style");
  if (!maybe_iteration_style.ok()) {
    return maybe_iteration_style.status();
  }
  if (maybe_iteration_style->has_value()) {
    absl::StatusOr<IterationStyle> status_or_iteration_style =
        IterationStyleFromString(maybe_iteration_style->value());
    if (status_or_iteration_style.ok()) {
      vmec_indata.iteration_style = status_or_iteration_style.value();
    } else {
      return status_or_iteration_style.status();
    }
  }

  auto maybe_return_outputs_even_if_not_converged =
      JsonReadBool(j, "return_outputs_even_if_not_converged");
  if (!maybe_return_outputs_even_if_not_converged.ok()) {
    return maybe_return_outputs_even_if_not_converged.status();
  }
  if (maybe_return_outputs_even_if_not_converged->has_value()) {
    vmec_indata.return_outputs_even_if_not_converged =
        maybe_return_outputs_even_if_not_converged->value();
  }

  // -----------------------------------------------

  auto maybe_raxis_c = JsonReadVectorDouble(j, "raxis_c");
  if (!maybe_raxis_c.ok()) {
    return maybe_raxis_c.status();
  }
  if (maybe_raxis_c->has_value()) {
    vmec_indata.raxis_c = maybe_raxis_c->value();
  }

  if (vmec_indata.raxis_c.size() !=
      static_cast<std::size_t>(vmec_indata.ntor) + 1) {
    return absl::InvalidArgumentError(
        absl::StrFormat("length of raxis_c (%ld) does not match ntor+1 (%d)\n",
                        vmec_indata.raxis_c.size(), vmec_indata.ntor + 1));
  }

  auto maybe_zaxis_s = JsonReadVectorDouble(j, "zaxis_s");
  if (!maybe_zaxis_s.ok()) {
    return maybe_zaxis_s.status();
  }
  if (maybe_zaxis_s->has_value()) {
    vmec_indata.zaxis_s = maybe_zaxis_s->value();
  }

  if (vmec_indata.zaxis_s.size() !=
      static_cast<std::size_t>(vmec_indata.ntor) + 1) {
    return absl::InvalidArgumentError(
        absl::StrFormat("length of zaxis_s (%ld) does not match ntor+1 (%d)\n",
                        vmec_indata.zaxis_s.size(), vmec_indata.ntor + 1));
  }

  if (vmec_indata.lasym) {
    auto maybe_raxis_s = JsonReadVectorDouble(j, "raxis_s");
    if (!maybe_raxis_s.ok()) {
      return maybe_raxis_s.status();
    }
    if (maybe_raxis_s->has_value()) {
      vmec_indata.raxis_s = maybe_raxis_s->value();
    }

    if (vmec_indata.raxis_s.size() !=
        static_cast<std::size_t>(vmec_indata.ntor) + 1) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "length of raxis_s (%ld) does not match ntor+1 (%d)\n",
          vmec_indata.raxis_s.size(), vmec_indata.ntor + 1));
    }

    auto maybe_zaxis_c = JsonReadVectorDouble(j, "zaxis_c");
    if (!maybe_zaxis_c.ok()) {
      return maybe_zaxis_c.status();
    }
    if (maybe_zaxis_c->has_value()) {
      vmec_indata.zaxis_c = maybe_zaxis_c->value();
    }

    if (vmec_indata.zaxis_c.size() !=
        static_cast<std::size_t>(vmec_indata.ntor) + 1) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "length of zaxis_c (%ld) does not match ntor+1 (%d)\n",
          vmec_indata.zaxis_c.size(), vmec_indata.ntor + 1));
    }
  }

  // -----------------------------------------------

  auto maybe_rbc = BoundaryCoefficient::FromJson(j, "rbc");
  if (!maybe_rbc.ok()) {
    return maybe_rbc.status();
  }
  if (maybe_rbc->has_value()) {
    vmec_indata.rbc.resize(vmec_indata.mpol * (2 * vmec_indata.ntor + 1), 0.0);
    std::vector<BoundaryCoefficient> entries = maybe_rbc->value();
    for (const BoundaryCoefficient& entry : entries) {
      if (entry.m > vmec_indata.mpol - 1) {
        LOG(INFO) << absl::StrFormat(
            "Ignoring rbc entry with m = %d, since m is larger than (mpol - 1) "
            "= %d",
            entry.m, vmec_indata.mpol - 1);
        continue;
      }
      if (std::abs(entry.n) > vmec_indata.ntor) {
        LOG(INFO) << absl::StrFormat(
            "Ignoring rbc entry with n = %d, since |n| is larger than ntor = "
            "%d",
            entry.n, vmec_indata.ntor);
        continue;
      }

      // Fortran order along n: -ntor, -ntor+1, ..., -1, 0, 1, ..., ntor-1, ntor
      const int index_along_n = vmec_indata.ntor + entry.n;

      const int flat_index =
          entry.m * (2 * vmec_indata.ntor + 1) + index_along_n;

      vmec_indata.rbc[flat_index] = entry.value;
    }
  }

  auto maybe_zbs = BoundaryCoefficient::FromJson(j, "zbs");
  if (!maybe_zbs.ok()) {
    return maybe_zbs.status();
  }
  if (maybe_zbs->has_value()) {
    vmec_indata.zbs.resize(vmec_indata.mpol * (2 * vmec_indata.ntor + 1), 0.0);
    std::vector<BoundaryCoefficient> entries = maybe_zbs->value();
    for (const BoundaryCoefficient& entry : entries) {
      if (entry.m > vmec_indata.mpol - 1) {
        LOG(INFO) << absl::StrFormat(
            "Ignoring zbs entry with m = %d, since m is larger than (mpol - 1) "
            "= %d",
            entry.m, vmec_indata.mpol - 1);
        continue;
      }
      if (std::abs(entry.n) > vmec_indata.ntor) {
        LOG(INFO) << absl::StrFormat(
            "Ignoring zbs entry with n = %d, since |n| is larger than ntor = "
            "%d",
            entry.n, vmec_indata.ntor);
        continue;
      }

      // Fortran order along n: -ntor, -ntor+1, ..., -1, 0, 1, ..., ntor-1, ntor
      const int index_along_n = vmec_indata.ntor + entry.n;

      const int flat_index =
          entry.m * (2 * vmec_indata.ntor + 1) + index_along_n;

      vmec_indata.zbs[flat_index] = entry.value;
    }
  }

  if (vmec_indata.lasym) {
    auto maybe_rbs = BoundaryCoefficient::FromJson(j, "rbs");
    if (!maybe_rbs.ok()) {
      return maybe_rbs.status();
    }
    if (maybe_rbs->has_value()) {
      vmec_indata.rbs.resize(vmec_indata.mpol * (2 * vmec_indata.ntor + 1),
                             0.0);
      std::vector<BoundaryCoefficient> entries = maybe_rbs->value();
      for (const BoundaryCoefficient& entry : entries) {
        if (entry.m > vmec_indata.mpol - 1) {
          LOG(INFO) << absl::StrFormat(
              "Ignoring rbs entry with m = %d, since m is larger than (mpol - "
              "1) = %d",
              entry.m, vmec_indata.mpol - 1);
          continue;
        }
        if (std::abs(entry.n) > vmec_indata.ntor) {
          LOG(INFO) << absl::StrFormat(
              "Ignoring rbs entry with n = %d, since |n| is larger than ntor = "
              "%d",
              entry.n, vmec_indata.ntor);
          continue;
        }

        // Fortran order along n: -ntor, -ntor+1, ..., -1, 0, 1, ..., ntor-1,
        // ntor
        const int index_along_n = vmec_indata.ntor + entry.n;

        const int flat_index =
            entry.m * (2 * vmec_indata.ntor + 1) + index_along_n;

        vmec_indata.rbs[flat_index] = entry.value;
      }
    }

    auto maybe_zbc = BoundaryCoefficient::FromJson(j, "zbc");
    if (!maybe_zbc.ok()) {
      return maybe_zbc.status();
    }
    if (maybe_zbc->has_value()) {
      vmec_indata.zbc.resize(vmec_indata.mpol * (2 * vmec_indata.ntor + 1),
                             0.0);
      std::vector<BoundaryCoefficient> entries = maybe_zbc->value();
      for (const BoundaryCoefficient& entry : entries) {
        if (entry.m > vmec_indata.mpol - 1) {
          LOG(INFO) << absl::StrFormat(
              "Ignoring zbc entry with m = %d, since m is larger than (mpol - "
              "1) = %d",
              entry.m, vmec_indata.mpol - 1);
          continue;
        }
        if (std::abs(entry.n) > vmec_indata.ntor) {
          LOG(INFO) << absl::StrFormat(
              "Ignoring zbc entry with n = %d, since |n| is larger than ntor = "
              "%d",
              entry.n, vmec_indata.ntor);
          continue;
        }

        // Fortran order along n: -ntor, -ntor+1, ..., -1, 0, 1, ..., ntor-1,
        // ntor
        const int index_along_n = vmec_indata.ntor + entry.n;

        const int flat_index =
            entry.m * (2 * vmec_indata.ntor + 1) + index_along_n;

        vmec_indata.zbc[flat_index] = entry.value;
      }
    }
  }

  static constexpr bool kEnableInfoMessages = false;
  absl::Status consistency_check_status =
      IsConsistent(vmec_indata, kEnableInfoMessages);
  if (!consistency_check_status.ok()) {
    return consistency_check_status;
  }

  return vmec_indata;
}

absl::StatusOr<std::string> VmecINDATA::ToJson() const {
  nlohmann::json output;

  // Numerical Resolution and Symmetry Assumptions
  output["lasym"] = lasym;
  output["nfp"] = nfp;
  output["mpol"] = mpol;
  output["ntor"] = ntor;
  output["ntheta"] = ntheta;
  output["nzeta"] = nzeta;

  // Multi-Grid Steps
  output["ns_array"] = ns_array;
  output["ftol_array"] = ftol_array;
  output["niter_array"] = niter_array;

  // Global Physics Parameters
  output["phiedge"] = phiedge;
  output["ncurr"] = ncurr;

  // Profile of Mass or Pressure
  output["pmass_type"] = pmass_type;
  output["am"] = am;
  output["am_aux_s"] = am_aux_s;
  output["am_aux_f"] = am_aux_f;
  output["pres_scale"] = pres_scale;
  output["gamma"] = gamma;
  output["spres_ped"] = spres_ped;

  // (Initial Guess for) Rotational Transform Profile
  output["piota_type"] = piota_type;
  output["ai"] = ai;
  output["ai_aux_s"] = ai_aux_s;
  output["ai_aux_f"] = ai_aux_f;

  // (Initial Guess for) Toroidal Current Profile
  output["pcurr_type"] = pcurr_type;
  output["ac"] = ac;
  output["ac_aux_s"] = ac_aux_s;
  output["ac_aux_f"] = ac_aux_f;
  output["curtor"] = curtor;
  output["bloat"] = bloat;

  // Free-Boundary Parameters
  output["lfreeb"] = lfreeb;
  output["mgrid_file"] = mgrid_file;
  output["extcur"] = extcur;
  output["nvacskip"] = nvacskip;
  output["free_boundary_method"] = ToString(free_boundary_method);

  // Tweaking Parameters
  output["nstep"] = nstep;
  output["aphi"] = aphi;
  output["delt"] = delt;
  output["tcon0"] = tcon0;
  output["lforbal"] = lforbal;
  output["iteration_style"] = ToString(iteration_style);
  output["return_outputs_even_if_not_converged"] =
      return_outputs_even_if_not_converged;

  // Initial Guess for Magnetic Axis Geometry
  output["raxis_c"] = raxis_c;
  output["zaxis_s"] = zaxis_s;
  if (lasym) {
    output["raxis_s"] = raxis_s;
    output["zaxis_c"] = zaxis_c;
  }

  // (Initial Guess for) Boundary Geometry
  output["rbc"] = std::vector<nlohmann::json>();
  output["zbs"] = std::vector<nlohmann::json>();
  if (lasym) {
    output["rbs"] = std::vector<nlohmann::json>();
    output["zbc"] = std::vector<nlohmann::json>();
  }
  nlohmann::json tmp_obj;
  for (int m = 0; m < mpol; ++m) {
    for (int n = 0; n < 2 * ntor + 1; ++n) {
      const int idx_mn = m * (2 * ntor + 1) + n;

      tmp_obj["m"] = m;
      tmp_obj["n"] = n - ntor;
      tmp_obj["value"] = rbc[idx_mn];

      auto push_nonzero = [&output, &tmp_obj](const std::string& key,
                                              double value) {
        tmp_obj["value"] = value;
        if (tmp_obj["value"] != 0.0) {
          output[key].push_back(tmp_obj);
        }
      };

      push_nonzero("rbc", rbc[idx_mn]);
      push_nonzero("zbs", zbs[idx_mn]);
      if (lasym) {
        // we also have non-stellarator-symmetric components
        push_nonzero("rbs", rbs[idx_mn]);
        push_nonzero("zbc", zbc[idx_mn]);
      }
    }
  }

  return output.dump();
}

absl::Status IsConsistent(const VmecINDATA& vmec_indata,
                          bool enable_info_messages) {
  // lasym can be true or false and both are valid

  if (vmec_indata.nfp <= 0) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "input variable 'nfp' needs to be > 0, but is %d", vmec_indata.nfp));
  }

  if (vmec_indata.mpol < 2) {
    return absl::InvalidArgumentError(
        absl::StrFormat("input variable 'mpol' needs to be >= 2, but is %d\n",
                        vmec_indata.mpol));
  }

  if (vmec_indata.ntor < 0) {
    return absl::InvalidArgumentError(
        absl::StrFormat("input variable 'ntor' needs to be >= 0, but is %d\n",
                        vmec_indata.ntor));
  }

  if (vmec_indata.ntheta < 0) {
    return absl::InvalidArgumentError(
        absl::StrFormat("input variable 'ntheta' needs to be >= 0, but is %d\n",
                        vmec_indata.ntheta));
  }

  if (vmec_indata.nzeta < 0) {
    return absl::InvalidArgumentError(
        absl::StrFormat("input variable 'nzeta' needs to be >= 0, but is %d\n",
                        vmec_indata.nzeta));
  }

  /* --------------------------------- */

  const int NS_MIN = 3;

  // ns_array
  // * needs to have non-zero element count
  // * entries need to be integers
  // * entries need to be at least NS_MIN
  // * entries need to be monotonically increasing or stay constant
  if (vmec_indata.ns_array.size() >
      0) {  // make sure that at least one multigrid step is specified
    int largestNumSurfacesSoFar = -1;
    for (size_t idx = 0; idx < vmec_indata.ns_array.size(); ++idx) {
      int ns = vmec_indata.ns_array[idx];
      if (ns >= NS_MIN) {  // ensure minimum number of surfaces
        if (ns < largestNumSurfacesSoFar) {  // make sure that ns is
                                             // monotonically increasing
          return absl::InvalidArgumentError(
              absl::StrFormat("input variable 'ns_array' needs to increase "
                              "monotonically or stay constant, but entries %ld "
                              "and %ld are %d and %d, respectively",
                              idx - 1, idx, vmec_indata.ns_array[idx - 1], ns));
        } else if (ns > largestNumSurfacesSoFar) {
          largestNumSurfacesSoFar = ns;
        }
      } else {
        return absl::InvalidArgumentError(
            absl::StrFormat("values in input variable 'ns_array' need to be at "
                            "least %d, but value %d was found at index %ld\n",
                            NS_MIN, ns, idx));
      }
    }
  } else {
    return absl::InvalidArgumentError(
        absl::StrFormat("input variable 'ns_array' needs to have at least one "
                        "entry, but size is %ld\n",
                        vmec_indata.ns_array.size()));
  }

  // ftol_array
  for (size_t idx = 0; idx < vmec_indata.ns_array.size(); ++idx) {
    double ftol = vmec_indata.ftol_array[idx];
    if (ftol < 0.0) {
      return absl::InvalidArgumentError(
          absl::StrFormat("values in input variable 'ftol_array' need to be "
                          "positive, but value %e was found at index %ld\n",
                          ftol, idx));
    }
  }

  // niter_array
  for (size_t idx = 0; idx < vmec_indata.ns_array.size(); ++idx) {
    int niter = vmec_indata.niter_array[idx];
    if (niter <= 0) {
      return absl::InvalidArgumentError(
          absl::StrFormat("values in input variable 'niter_array' need to be "
                          "positive, but value %d was found at index %ld\n",
                          niter, idx));
    }
  }

  /* --------------------------------- */

  // phiedge
  if (vmec_indata.phiedge == 0.0) {
    return absl::InvalidArgumentError(
        absl::StrFormat("input variable 'phiedge' must not be 0.0\n"));
  }

  // ncurr
  if (vmec_indata.ncurr != 0 && vmec_indata.ncurr != 1) {
    return absl::InvalidArgumentError(
        absl::StrFormat("input variable 'ncurr' must be 0 or 1, but is %d\n",
                        vmec_indata.ncurr));
  }

  /* --------------------------------- */

  // pmass_type
  // TODO(jons): check for allowed value

  // am
  // TODO(jons): must be given for parameterized profiles

  // am_aux_s
  // am_aux_f
  // TODO(jons): must be given for spline data profiles

  // pres_scale
  if (vmec_indata.pres_scale < 0) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "input variable 'pres_scale' must be positive, but was %g\n",
        vmec_indata.pres_scale));
  }

  // adiabatic_index
  if (vmec_indata.gamma == 1.0) {
    return absl::InvalidArgumentError(
        absl::StrFormat("input variable 'adiabatic_index' must not be 1.0\n"));
  }

  // spres_ped
  // * pressure pedestal position cannot be at magnetic axis or negative
  if (vmec_indata.spres_ped <= 0.0 || vmec_indata.spres_ped > 1) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "input variable 'spres_ped' must be > 0 and <= 1, but is %g\n",
        vmec_indata.spres_ped));
  }

  /* --------------------------------- */

  if (vmec_indata.ncurr == 0) {
    // piota_type
    // TODO(jons): check for allowed value

    // ai
    // TODO(jons): must be given for parameterized profiles

    // ai_aux_s
    // ai_aux_f
    // TODO(jons): must be given for spline data profiles

    if (vmec_indata.bloat != 1.0) {
      // bloat != 1 is only allowed when ncurr == 1 (constrained toroidal
      // current)
      // --> for constrained iota profile, must have bloat == 1.0
      return absl::InvalidArgumentError(absl::StrFormat(
          "'bloat' must be 1.0 for ncurr == 0 (constrained-iota), but is %g\n",
          vmec_indata.bloat));
    }

  } else if (vmec_indata.ncurr == 1) {
    // pcurr_type
    // TODO(jons): check for allowed value

    // ac
    // TODO(jons): must be given for parameterized profiles

    // ac_aux_s
    // ac_aux_f
    // TODO(jons): must be given for spline data profiles

    // curtor --> any value is ok

    // bloat --> any value is ok
  }

  /* --------------------------------- */

  // lfreeb
  // nothing to check here: lfreeb can be true or false and both are valid...
  if (vmec_indata.lfreeb) {
    // mgrid_file
    // TODO(jons): if mgrid read, check for consistent nzeta

    // extcur
    // TODO(jons): check that number of coil currents matches number of response
    // tables in mgrid file

    // nvacskip
    if (vmec_indata.nvacskip < 1) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "input variable nvacskip needs to be > 0, but is %d\n",
          vmec_indata.nvacskip));
    }

    // free_boundary_method
    // For the current state of the code, we only accept NESTOR,
    // but in the future [TODO(jons)] also all other (implemented) enum values
    // are valid.
    if (vmec_indata.free_boundary_method != FreeBoundaryMethod::NESTOR) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "input variable 'free_boundary_method' must be 'nestor', but is %s\n",
          ToString(vmec_indata.free_boundary_method)));
    }
  }

  /* --------------------------------- */

  // nstep
  if (vmec_indata.nstep <= 0) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "input variable 'nstep' must be > 0, but is %d\n", vmec_indata.nstep));
  }

  // aphi
  // assume data is ok; will see when the physics starts to run...

  // delt
  if (vmec_indata.delt <= 0.0 || vmec_indata.delt > 1.0) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "input variable 'delt' has to be in the range ]0.0, 1.0], but is %g\n",
        vmec_indata.delt));
  }

  // tcon0
  if (vmec_indata.tcon0 < 0.0 || vmec_indata.tcon0 > 1.0) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "input variable 'tcon0' has to be in the range [0.0, 1.0], but is %g\n",
        vmec_indata.tcon0));
  }

  // lforbal
  // nothing to check here: lforbal can be true or false and both are valid...

  // iteration_style
  // For the current state of the code, we only accept VMEC_8_52,
  // but in the future [TODO(jons)] also all other (implemented) enum values
  // are valid.
  if (vmec_indata.iteration_style != IterationStyle::VMEC_8_52) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "input variable 'iteration_style' must be 'vmec_8_52', but is %s\n",
        ToString(vmec_indata.iteration_style)));
  }

  // return_outputs_even_if_not_converged
  // nothing to check here: return_outputs_even_if_not_converged can be true or
  // false and both are valid...

  /* --------------------------------- */

  // only check sizes are ok; will see about the contents when physics start...
  const std::size_t expected_axis_size_symm = vmec_indata.ntor + 1;
  // raxis_c
  if (vmec_indata.raxis_c.size() != expected_axis_size_symm) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "input variable 'raxis_c' has wrong size: should be %i, but it is %i.",
        expected_axis_size_symm, vmec_indata.raxis_c.size()));
  }
  // zaxis_s
  if (vmec_indata.zaxis_s.size() != expected_axis_size_symm) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "input variable 'zaxis_s' has wrong size: should be %i, but it is %i.",
        expected_axis_size_symm, vmec_indata.zaxis_s.size()));
  }

  const std::size_t expected_axis_size_asym =
      vmec_indata.lasym ? vmec_indata.ntor + 1 : 0;
  // raxis_s
  if (vmec_indata.raxis_s.size() != expected_axis_size_asym) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "input variable 'raxis_s' has wrong size: should be %i, but it is %i.",
        expected_axis_size_asym, vmec_indata.raxis_s.size()));
  }
  // zaxis_c
  if (vmec_indata.zaxis_c.size() != expected_axis_size_asym) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "input variable 'zaxis_c' has wrong size: should be %i, but it is %i.",
        expected_axis_size_asym, vmec_indata.zaxis_c.size()));
  }

  /* --------------------------------- */

  // only check sizes are ok; will see when the physics starts to run...

  const std::size_t expected_bdy_size_symm =
      vmec_indata.mpol * (2 * vmec_indata.ntor + 1);
  // rbc
  if (vmec_indata.rbc.size() != expected_bdy_size_symm) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "input variable 'rbc' has wrong size: should be %i, but it is %i.",
        expected_bdy_size_symm, vmec_indata.rbc.size()));
  }
  // zbs
  if (vmec_indata.zbs.size() != expected_bdy_size_symm) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "input variable 'zbs' has wrong size: should be %i, but it is %i.",
        expected_bdy_size_symm, vmec_indata.zbs.size()));
  }

  const std::size_t expected_bdy_size_asym = 0;

  // rbs
  if (vmec_indata.rbs.size() != expected_bdy_size_asym) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "input variable 'rbs' has wrong size: should be %i, but it is %i.",
        expected_bdy_size_asym, vmec_indata.rbs.size()));
  }
  // zbc
  if (vmec_indata.zbc.size() != expected_bdy_size_asym) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "input variable 'zbc' has wrong size: should be %i, but it is %i.",
        expected_bdy_size_asym, vmec_indata.zbc.size()));
  }

  // zbs
  if (enable_info_messages && vmec_indata.zbs[0] != 0.0) {
    // The (0,0) mode of a sin(m*u-n*v) term is irrelevant, so see if anything
    // was specified there. The linear index of (0,0) mode is 0.
    LOG(INFO) << absl::StrFormat(
        "ignoring irrelevant zbs entry for m=0, n=0: %g\n", vmec_indata.zbs[0]);
  }

  // raxis_cc
  // zaxis_cs
  if (enable_info_messages && vmec_indata.zaxis_s[0] != 0.0) {
    // The n=0 mode of a sin(n*v) term is irrelevant, so see if anything was
    // specified there.
    LOG(INFO) << absl::StrFormat(
        "ignoring irrelevant zaxis_s entry for n=0: %g\n",
        vmec_indata.zaxis_s[0]);
  }

  if (vmec_indata.lasym) {
    // rbs
    if (enable_info_messages && vmec_indata.rbs[0] != 0.0) {
      // The (0,0) mode of a sin(m*u-n*v) term is irrelevant, so see if anything
      // was specified there. The linear index of (0,0) mode is 0.
      LOG(INFO) << absl::StrFormat(
          "ignoring irrelevant rbs entry for m=0, n=0: %g\n",
          vmec_indata.rbs[0]);
    }

    // zbc

    // raxis_cs
    if (enable_info_messages && vmec_indata.raxis_s[0] != 0.0) {
      // The n=0 mode of a sin(n*v) term is irrelevant, so see if anything was
      // specified there.
      LOG(INFO) << absl::StrFormat(
          "ignoring irrelevant raxis_s entry for n=0: %g\n",
          vmec_indata.raxis_s[0]);
    }

    // zaxis_cc
  }

  return absl::OkStatus();
}

}  // namespace vmecpp
