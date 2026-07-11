// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
//
// Standalone driver for the free-boundary solvers: evaluates the vacuum
// magnetic pressure |B|^2/2 on the plasma boundary given in `rbc`/`zbs` of a
// VMEC++ JSON input file, using the solver selected by
// `free_boundary_method` (optionally overridden on the command line).
// Intended for A/B comparisons between NESTOR and BIEST outside the VMEC
// iteration.

#include <cstdlib>  // EXIT_SUCCESS, EXIT_FAILURE
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif  // _OPENMP

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "nlohmann/json.hpp"
#include "util/file_io/file_io.h"
#include "vmecpp/common/fourier_basis_fast_toroidal/fourier_basis_fast_toroidal.h"
#include "vmecpp/common/vmec_indata/vmec_indata.h"
#include "vmecpp/free_boundary/biest/biest.h"
#include "vmecpp/free_boundary/mgrid_provider/mgrid_provider.h"
#include "vmecpp/free_boundary/nestor/nestor.h"
#include "vmecpp/free_boundary/tangential_partitioning/tangential_partitioning.h"
#include "vmecpp/free_boundary/vac2/vac2.h"

using file_io::ReadFile;
using nlohmann::json;
using vmecpp::VmecINDATA;

int main(int argc, char** argv) {
  // use VMEC convention of left-handed coordinate system
  static constexpr int kSignOfJacobian = -1;

  if (argc != 2 && argc != 3) {
    std::cerr << "Compute |B|^2/2 on the boundary given by rbc/zbs using the "
                 "free-boundary solver selected in 'free_boundary_method' of "
                 "the input file (optionally overridden by the second "
                 "argument).\n"
              << "Usage: " << argv[0]
              << " <vmecpp_input_file.json> [nestor|biest|only_coils]\n";
    return EXIT_FAILURE;
  }

  const std::filesystem::path vmecpp_indata_filename(argv[1]);

  absl::StatusOr<std::string> maybe_indata_json =
      ReadFile(vmecpp_indata_filename);
  CHECK_OK(maybe_indata_json)
      << "Could not read input file '" << vmecpp_indata_filename
      << "': " << maybe_indata_json.status();
  const std::string& indata_json = *maybe_indata_json;

  absl::StatusOr<VmecINDATA> maybe_vmec_indata =
      VmecINDATA::FromJson(indata_json);
  CHECK_OK(maybe_vmec_indata)
      << "Could not parse input file '" << vmecpp_indata_filename
      << "' into VmecINDATA: " << maybe_vmec_indata.status();
  VmecINDATA vmec_indata = *maybe_vmec_indata;

  if (argc == 3) {
    const auto maybe_method =
        vmecpp::FreeBoundaryMethodFromString(std::string(argv[2]));
    CHECK_OK(maybe_method) << "invalid free-boundary method '" << argv[2]
                           << "'";
    vmec_indata.free_boundary_method = maybe_method.value();
  }

  vmecpp::Sizes s(vmec_indata);

  // build raxis, zaxis from raxis_c, zaxis_s in VMEC INDATA
  // -> evaluate Fourier series in toroidal direction
  std::vector<double> axis_r(s.nZeta);
  std::vector<double> axis_z(s.nZeta);
  const double delta_phi = 2.0 * M_PI / (s.nfp * s.nZeta);
  for (int k = 0; k < s.nZeta; ++k) {
    for (int n = 0; n < s.ntor + 1; ++n) {
      axis_r[k] += vmec_indata.raxis_c[n] * std::cos(k * n * s.nfp * delta_phi);
      axis_z[k] += vmec_indata.zaxis_s[n] * std::sin(k * n * s.nfp * delta_phi);
    }
  }

  // build rmnc, zmns from rbc, zbs in VMEC INDATA
  std::vector<double> rmnc(s.mnmax);
  std::vector<double> zmns(s.mnmax);
  int mn = 0;
  int m = 0;
  for (int n = 0; n < s.ntor + 1; ++n) {
    rmnc[mn] = vmec_indata.rbc(m, s.ntor + n);
    zmns[mn] = vmec_indata.zbs(m, s.ntor + n);

    mn++;
  }
  for (m = 1; m < s.mpol; ++m) {
    for (int n = -s.ntor; n < s.ntor + 1; ++n) {
      rmnc[mn] = vmec_indata.rbc(m, s.ntor + n);
      zmns[mn] = vmec_indata.zbs(m, s.ntor + n);

      mn++;
    }
  }

  CHECK_EQ(mn, s.mnmax)
      << "counting error in assigning boundary Fourier coefficients";

  vmecpp::FourierBasisFastToroidal fb(&s);

  std::vector<double> rCC(s.mnsize);
  std::vector<double> rSS(s.mnsize);
  std::vector<double> zSC(s.mnsize);
  std::vector<double> zCS(s.mnsize);

  // only needed for lasym=true, which is not supported here yet
  std::vector<double> rSC;
  std::vector<double> rCS;
  std::vector<double> zCC;
  std::vector<double> zSS;

  // convert rmnc, ... from the input into rCC, ...
  fb.cos_to_cc_ss(rmnc, rCC, rSS, s.ntor, s.mpol);
  fb.sin_to_sc_cs(zmns, zSC, zCS, s.ntor, s.mpol);

  // ------------------------------------------------------

  vmecpp::MGridProvider mgrid;
  absl::Status status =
      mgrid.LoadFile(vmec_indata.mgrid_file, vmec_indata.extcur);
  CHECK_OK(status);

  // tangential Fourier resolution for Nestor:
  // one more poloidal mode than VMEC
  const int nf = s.ntor;      // 0 : ntor
  const int mf = s.mpol + 1;  // 0 : (mpol + 1)

  const int mnpd = (2 * nf + 1) * (mf + 1);
  std::vector<double> matrixShare(mnpd * mnpd);
  std::vector<int> iPiv(mnpd);
  std::vector<double> bvecShare(mnpd);

  // shared scratch for BIEST and Vac2
  std::vector<double> vac2RubShare(s.nZnT), vac2RvbShare(s.nZnT);
  std::vector<double> vac2ZubShare(s.nZnT), vac2ZvbShare(s.nZnT);
  std::vector<double> vac2BsqOut(s.nThetaEven * s.nZeta);
  std::vector<double> vac2PotU(s.nThetaEven * s.nZeta);
  std::vector<double> vac2PotV(s.nThetaEven * s.nZeta);
  std::vector<double> coilBrShare(s.nZnT);
  std::vector<double> coilBpShare(s.nZnT);
  std::vector<double> coilBzShare(s.nZnT);
  std::vector<double> bPlasmaShare(3 * s.nZeta * s.nThetaEven);

  // this will contain |B|^2/2 when the solver is done
  std::vector<double> bSqVacShare(s.nZnT);

  std::vector<double> vacuum_b_r(s.nZnT);
  std::vector<double> vacuum_b_phi(s.nZnT);
  std::vector<double> vacuum_b_z(s.nZnT);

  // bsubuvac in Fortran VMEC and the rest of VMEC++
  double net_toroidal_current_from_solution = 0.0;

  // bsubvvac in Fortran VMEC and the rest of VMEC++
  double net_poloidal_current_from_solution = 0.0;

  // this makes sure that we always run the full computation
  const int ivacskip = 0;

#ifdef _OPENMP
  const int num_threads = omp_get_max_threads();
#pragma omp parallel
  {
    const int thread_id = omp_get_thread_num();
#else
  const int num_threads = 1;
  {
    const int thread_id = 0;
#endif  // _OPENMP

    vmecpp::TangentialPartitioning tp(s.nZnT, num_threads, thread_id);

    std::unique_ptr<vmecpp::FreeBoundaryBase> solver;
    if (vmec_indata.free_boundary_method ==
        vmecpp::FreeBoundaryMethod::NESTOR) {
      solver = std::make_unique<vmecpp::Nestor>(
          &s, &tp, &mgrid, matrixShare, bvecShare, bSqVacShare, iPiv,
          vacuum_b_r, vacuum_b_phi, vacuum_b_z);
    } else if (vmec_indata.free_boundary_method ==
               vmecpp::FreeBoundaryMethod::BIEST) {
      solver = std::make_unique<vmecpp::Biest>(
          &s, &tp, &mgrid, vmec_indata.biest_accuracy_digits, coilBrShare,
          coilBpShare, coilBzShare, bPlasmaShare, bSqVacShare, vacuum_b_r,
          vacuum_b_phi, vacuum_b_z);
    } else if (vmec_indata.free_boundary_method ==
               vmecpp::FreeBoundaryMethod::VAC2) {
      solver = std::make_unique<vmecpp::Vac2>(
          &s, &tp, &mgrid, coilBrShare, coilBpShare, coilBzShare, vac2RubShare,
          vac2RvbShare, vac2ZubShare, vac2ZvbShare, vac2BsqOut, vac2PotU,
          vac2PotV, bSqVacShare, vacuum_b_r, vacuum_b_phi, vacuum_b_z);
    } else {
      LOG(FATAL) << "free-boundary method not supported by this tool: "
                 << vmecpp::ToString(vmec_indata.free_boundary_method);
    }

    // NOTE: accumulation into `net_toroidal_current_from_solution` and
    // `net_poloidal_current_from_solution` by only a single thread is
    // already handled within the solver's `update()`.
    solver->update(rCC, rSS, rSC, rCS, zSC, zCS, zCC, zSS, kSignOfJacobian,
                   axis_r, axis_z, &net_toroidal_current_from_solution,
                   &net_poloidal_current_from_solution, vmec_indata.curtor,
                   ivacskip);
    // at this point we expect to have the full solution in bSqVacShare
  }  // pragma omp parallel

  LOG(INFO) << "net toroidal current from solution: "
            << net_toroidal_current_from_solution / vmecpp::MU_0 << " A";
  LOG(INFO) << "net poloidal current R*Btor from solution: "
            << net_poloidal_current_from_solution;

  // write |B|^2/2 and net currents to JSON file
  json freeboundary_outputs;

  // resolution parameters
  freeboundary_outputs["lasym"] = vmec_indata.lasym;
  freeboundary_outputs["nfp"] = vmec_indata.nfp;
  freeboundary_outputs["mpol"] = s.mpol;
  freeboundary_outputs["ntor"] = s.ntor;
  freeboundary_outputs["nThetaEven"] = s.nThetaEven;
  freeboundary_outputs["nThetaReduced"] = s.nThetaReduced;
  freeboundary_outputs["nThetaEff"] = s.nThetaEff;
  freeboundary_outputs["nZeta"] = s.nZeta;
  freeboundary_outputs["signgs"] = kSignOfJacobian;
  freeboundary_outputs["free_boundary_method"] =
      vmecpp::ToString(vmec_indata.free_boundary_method);
  freeboundary_outputs["biest_accuracy_digits"] =
      vmec_indata.biest_accuracy_digits;

  // scalar outputs
  freeboundary_outputs["net_toroidal_current_from_solution"] =
      net_toroidal_current_from_solution;
  freeboundary_outputs["net_poloidal_current_from_solution"] =
      net_poloidal_current_from_solution;

  // vector outputs over theta = [0, pi] and phi = [0, 2 pi / nfp).
  // The toroidal index is fast, i.e., `bSqVacShare` is indexed as
  // `l * s.nZeta + k` for l = 0, 1, ..., (s.nThetaEff - 1) and
  // k = 0, 1, ..., (nZeta - 1). Convert to a two-dimensional array for
  // writing; the toroidal index is fast, so make that the second (inner) one.
  auto to_2d = [&](const std::vector<double>& flat) {
    std::vector<std::vector<double>> result(s.nThetaEff);
    for (int l = 0; l < s.nThetaEff; ++l) {
      result[l].resize(s.nZeta);
      for (int k = 0; k < s.nZeta; ++k) {
        result[l][k] = flat[l * s.nZeta + k];
      }
    }
    return result;
  };
  freeboundary_outputs["bsq_vac"] = to_2d(bSqVacShare);
  freeboundary_outputs["vacuum_b_r"] = to_2d(vacuum_b_r);
  freeboundary_outputs["vacuum_b_phi"] = to_2d(vacuum_b_phi);
  freeboundary_outputs["vacuum_b_z"] = to_2d(vacuum_b_z);

  // write JSON to output file
  std::filesystem::path output_filename(vmecpp_indata_filename);
  auto method_name = vmecpp::ToString(vmec_indata.free_boundary_method);
  output_filename.replace_extension(
      absl::StrCat(".", method_name, "_out.json"));
  std::ofstream output_file(output_filename);
  output_file << freeboundary_outputs;

  LOG(INFO) << "wrote " << output_filename;

  return EXIT_SUCCESS;
}
