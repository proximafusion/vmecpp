// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#include "vmecpp/free_boundary/mgrid_provider/mgrid_provider.h"

#include <netcdf.h>

#include <algorithm>
#include <cfloat>  // DBL_MAX
#include <cstdio>
#include <fstream>
#include <iostream>
#include <vector>

#include "absl/log/check.h"
#include "absl/strings/str_format.h"
#include "util/netcdf_io/netcdf_io.h"
#include "vmecpp/common/makegrid_lib/makegrid_lib.h"
#include "vmecpp/common/util/util.h"

namespace vmecpp {

using netcdf_io::NetcdfReadArray3D;
using netcdf_io::NetcdfReadDouble;
using netcdf_io::NetcdfReadInt;
using netcdf_io::NetcdfReadString;

MGridProvider::MGridProvider() {
  nfp = -1;

  numR = -1;
  minR = 0.0;
  maxR = 0.0;
  deltaR = 0.0;

  numZ = -1;
  minZ = 0.0;
  maxZ = 0.0;
  deltaZ = 0.0;

  numPhi = -1;

  nextcur = -1;

  has_mgrid_loaded_ = false;

  has_fixed_field_ = false;

  mgrid_mode = "";
}

absl::Status MGridProvider::LoadFile(const std::filesystem::path& filename,
                                     const std::vector<double>& coil_currents) {
  {  // try to open file in order to check if it is accessible
    if (!std::filesystem::exists(filename)) {
      return absl::NotFoundError(
          absl::StrFormat("Could not find '%s'.", filename.string()));
    }
    std::ifstream fp(filename);
    if (!fp.is_open()) {
      return absl::PermissionDeniedError(absl::StrFormat(
          "Mgrid file '%s' exists, but could not be opened for reading.",
          filename.string()));
    }
  }

  int ncid = 0;
  if (nc_open(filename.c_str(), NC_NOWRITE, &ncid) != NC_NOERR) {
    return absl::InternalError(
        absl::StrFormat("NetCDF couldn't open '%s', despite passing "
                        "preconditions. The file may be corrupted.",
                        filename.string()));
  }

  // TODO(jurasic) All of these should be handled with abseil status, but
  // terminate on error with absl::CHECK instead.
  nfp = NetcdfReadInt(ncid, "nfp");

  numR = NetcdfReadInt(ncid, "ir");
  minR = NetcdfReadDouble(ncid, "rmin");
  maxR = NetcdfReadDouble(ncid, "rmax");
  deltaR = (maxR - minR) / (numR - 1.0);

  numZ = NetcdfReadInt(ncid, "jz");
  minZ = NetcdfReadDouble(ncid, "zmin");
  maxZ = NetcdfReadDouble(ncid, "zmax");
  deltaZ = (maxZ - minZ) / (numZ - 1.0);

  numPhi = NetcdfReadInt(ncid, "kp");

  nextcur = NetcdfReadInt(ncid, "nextcur");
  if (coil_currents.size() != static_cast<std::size_t>(nextcur)) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Number of currents %d does not match number of mgrid "
                        "coil fields nextcur=%d.",
                        coil_currents.size(), nextcur));
  }
  mgrid_mode = NetcdfReadString(ncid, "mgrid_mode");

  // Resize and make sure that the accumulation arrays are reset to zeros
  // if they contained previous contents from an earlier call to this routine.
  bR.resize(numPhi * numZ * numR, 0.0);
  bP.resize(numPhi * numZ * numR, 0.0);
  bZ.resize(numPhi * numZ * numR, 0.0);

  // combine coil contributions, weighted by coil currents
  for (int i = 0; i < nextcur; ++i) {
    // for each coil group:
    // get 3d double array "br_%03d", "bp_%03d", "bz_%03d"
    // from i=1, 2, ..., nextcur

    std::string br_variable = absl::StrFormat("br_%03d", i + 1);
    std::vector<std::vector<std::vector<double> > > b_r_contribution =
        NetcdfReadArray3D(ncid, br_variable);

    std::string bp_variable = absl::StrFormat("bp_%03d", i + 1);
    std::vector<std::vector<std::vector<double> > > b_p_contribution =
        NetcdfReadArray3D(ncid, bp_variable);

    std::string bz_variable = absl::StrFormat("bz_%03d", i + 1);
    std::vector<std::vector<std::vector<double> > > b_z_contribution =
        NetcdfReadArray3D(ncid, bz_variable);

    for (int index_phi = 0; index_phi < numPhi; ++index_phi) {
      for (int index_z = 0; index_z < numZ; ++index_z) {
        for (int index_r = 0; index_r < numR; ++index_r) {
          const int linear_index =
              (index_phi * numZ + index_z) * numR + index_r;

          bR[linear_index] +=
              b_r_contribution[index_phi][index_z][index_r] * coil_currents[i];
          bP[linear_index] +=
              b_p_contribution[index_phi][index_z][index_r] * coil_currents[i];
          bZ[linear_index] +=
              b_z_contribution[index_phi][index_z][index_r] * coil_currents[i];
        }  // index_r
      }  // index_z
    }  // index_phi
  }  // nextcur

  if (nc_close(ncid) != NC_NOERR) {
    return absl::InternalError("Failed to close NetCDF file.");
  }

  has_mgrid_loaded_ = true;
  has_fixed_field_ = false;

  return absl::Status();
}

absl::Status MGridProvider::LoadFields(
    const makegrid::MakegridParameters& mgrid_params,
    const makegrid::MagneticFieldResponseTable& magnetic_response_table,
    const std::vector<double>& coil_currents) {
  if (coil_currents.size() !=
      static_cast<std::size_t>((magnetic_response_table.b_p.rows()))) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Number of currents %d does not match number of coil fields in the "
        "response table %d.",
        coil_currents.size(), magnetic_response_table.b_p.rows()));
  }

  nfp = mgrid_params.number_of_field_periods;

  numR = mgrid_params.number_of_r_grid_points;
  minR = mgrid_params.r_grid_minimum;
  maxR = mgrid_params.r_grid_maximum;
  deltaR = (maxR - minR) / (numR - 1.0);

  numZ = mgrid_params.number_of_z_grid_points;
  minZ = mgrid_params.z_grid_minimum;
  maxZ = mgrid_params.z_grid_maximum;
  deltaZ = (maxZ - minZ) / (numZ - 1.0);

  numPhi = mgrid_params.number_of_phi_grid_points;

  nextcur = static_cast<int>(coil_currents.size());

  if (mgrid_params.normalize_by_currents) {
    mgrid_mode = "S";
  } else {
    mgrid_mode = "R";
  }

  // TODO(eguiraud): factor out this part that is duplicated
  const int num_grid_points = numPhi * numZ * numR;
  bR.resize(num_grid_points, 0.0);
  bP.resize(num_grid_points, 0.0);
  bZ.resize(num_grid_points, 0.0);

  // combine coil contributions, weighted by coil currents
  for (int i = 0; i < nextcur; ++i) {
    for (int linear_index = 0; linear_index < num_grid_points; ++linear_index) {
      bR[linear_index] +=
          magnetic_response_table.b_r(i, linear_index) * coil_currents[i];
      bP[linear_index] +=
          magnetic_response_table.b_p(i, linear_index) * coil_currents[i];
      bZ[linear_index] +=
          magnetic_response_table.b_z(i, linear_index) * coil_currents[i];
    }  // linear_index
  }  // nextcur

  has_mgrid_loaded_ = true;
  has_fixed_field_ = false;

  return absl::OkStatus();
}

void MGridProvider::SetFixedMagneticField(const std::vector<double>& fixed_br,
                                          const std::vector<double>& fixed_bp,
                                          const std::vector<double>& fixed_bz) {
  // copy into local storage
  fixed_br_ = fixed_br;
  fixed_bp_ = fixed_bp;
  fixed_bz_ = fixed_bz;

  has_mgrid_loaded_ = true;
  has_fixed_field_ = true;
}  // SetFixedMagneticField

// interpolate mgrid file at current flux surface
void MGridProvider::interpolate(int ztMin, int ztMax, int nZeta,
                                const std::vector<double>& rLCFS,
                                const std::vector<double>& zLCFS,
                                std::vector<double>& m_interpBr,
                                std::vector<double>& m_interpBp,
                                std::vector<double>& m_interpBz) const {
  CHECK(has_mgrid_loaded_) << "no mgrid loaded";

  if (has_fixed_field_) {
    // quick return: just copy into target storage

    for (int kl = ztMin; kl < ztMax; ++kl) {
      m_interpBr[kl - ztMin] = fixed_br_[kl];
      m_interpBp[kl - ztMin] = fixed_bp_[kl];
      m_interpBz[kl - ztMin] = fixed_bz_[kl];
    }  // kl

    return;
  }

  double min_r = DBL_MAX;
  double max_r = -DBL_MAX;

  double min_z = DBL_MAX;
  double max_z = -DBL_MAX;

  bool exceedGridSizeR = false;
  bool exceedGridSizeZ = false;
  for (int kl = ztMin; kl < ztMax; ++kl) {
    int k = kl % nZeta;

    min_r = std::min(min_r, rLCFS[kl]);
    max_r = std::max(max_r, rLCFS[kl]);

    min_z = std::min(min_z, zLCFS[kl]);
    max_z = std::max(max_z, zLCFS[kl]);

    // check if plasma boundary exceeds pre-computed grid
    if (rLCFS[kl] < minR || rLCFS[kl] > maxR) {
      exceedGridSizeR = true;
    }
    if (zLCFS[kl] < minZ || zLCFS[kl] > maxZ) {
      exceedGridSizeZ = true;
    }

    // crop to available grid
    double r = std::max(minR, std::min(rLCFS[kl], maxR));
    double z = std::max(minZ, std::min(zLCFS[kl], maxZ));

    // DETERMINE INTEGER INDICES (IR,JZ) FOR LOWER LEFT R, Z CORNER GRID POINT
    int ir = static_cast<int>(floor((r - minR) / deltaR));
    int jz = static_cast<int>(floor((z - minZ) / deltaZ));
    int ir1 = std::min(numR - 1, ir + 1);
    int jz1 = std::min(numZ - 1, jz + 1);

    // COMPUTE RI, ZJ AND PR, QZ AT GRID POINT (IR , JZ)
    double ri = minR + ir * deltaR;
    double zj = minZ + jz * deltaZ;
    double pr = (r - ri) / deltaR;
    double qz = (z - zj) / deltaZ;

    // COMPUTE WEIGHTS WIJ FOR 4 CORNER GRID POINTS
    double w22 = pr * qz;                //    p *   q
    double w21 = pr - w22;               //    p *(1-q) = p - p*q
    double w12 = qz - w22;               // (1-p)*   q  = q - p*q
    double w11 = 1.0 + w22 - (pr + qz);  // (1-p)*(1-q) = 1 + p*q - (p + q)

    // COMPUTE B FIELD AT R, PHI, Z BY INTERPOLATION
    int kj_i_ = (k * numZ + jz) * numR + ir;
    int kj1i_ = (k * numZ + jz1) * numR + ir;
    int kj_i1 = (k * numZ + jz) * numR + ir1;
    int kj1i1 = (k * numZ + jz1) * numR + ir1;

    m_interpBr[kl - ztMin] =
        w11 * bR[kj_i_] + w12 * bR[kj1i_] + w21 * bR[kj_i1] + w22 * bR[kj1i1];
    m_interpBp[kl - ztMin] =
        w11 * bP[kj_i_] + w12 * bP[kj1i_] + w21 * bP[kj_i1] + w22 * bP[kj1i1];
    m_interpBz[kl - ztMin] =
        w11 * bZ[kj_i_] + w12 * bZ[kj1i_] + w21 * bZ[kj_i1] + w22 * bZ[kj1i1];
  }  // kl

  if (exceedGridSizeR || exceedGridSizeZ) {
    // TODO(jons): automatically evaluate B outside of grid based on coil
    // definitions and Biot-Savart
    // --> will only get slower, but more robust (and accurate?)
    // --> would also require to always have coil geometry inside mgrid file for
    // on-the-fly re-evaluation...
    // NOTE: This is not suppressed by the `verbose` flag (vmec.cc:Vmec), since
    // it is considered an error message.
    std::cerr << "WARNING: Plasma Boundary exceeded Vacuum Grid Size\n";

    if (exceedGridSizeR) {
      std::cout << absl::StrFormat("  R: min = % .3e  max = % .3e\n", min_r,
                                   max_r);
    }

    if (exceedGridSizeZ) {
      std::cout << absl::StrFormat("  Z: min = % .3e  max = % .3e\n", min_z,
                                   max_z);
    }
  }

#ifdef _OPENMP
#pragma omp barrier
#endif  // _OPENMP
}

}  // namespace vmecpp
