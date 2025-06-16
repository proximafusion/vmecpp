// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#include "vmecpp/free_boundary/external_magnetic_field/external_magnetic_field.h"

#include "abscab/abscab.hh"
#include "absl/algorithm/container.h"

namespace vmecpp {

ExternalMagneticField::ExternalMagneticField(const Sizes* s,
                                             const TangentialPartitioning* tp,
                                             const SurfaceGeometry* sg,
                                             const MGridProvider* mgrid)
    : s_(*s), tp_(*tp), sg_(*sg), mgrid_(*mgrid) {
  // nzeta points per each of the nfp field periods,
  // and then one more point to close the loop
  axisXYZ.resize(3 * (s_.nZeta * s_.nfp + 1));

  // thread-local tangential grid point range
  const int numLocal = tp_.ztMax - tp_.ztMin;

  surfaceXYZ.resize(3 * numLocal);
  bCoilsXYZ.resize(3 * numLocal);

  interpBr.resize(numLocal);
  interpBp.resize(numLocal);
  interpBz.resize(numLocal);

  curtorBr.resize(numLocal);
  curtorBp.resize(numLocal);
  curtorBz.resize(numLocal);

  bSubU.resize(numLocal);
  bSubV.resize(numLocal);
  bDotN.resize(numLocal);
}

// rAxis, zAxis are provided over a single module
void ExternalMagneticField::update(const std::span<const double> rAxis,
                                   const std::span<const double> zAxis,
                                   double netToroidalCurrent) {
#ifdef _OPENMP
#pragma omp barrier
#endif  // _OPENMP

  mgrid_.interpolate(tp_.ztMin, tp_.ztMax, s_.nZeta, sg_.r1b, sg_.z1b, interpBr,
                     interpBp, interpBz);

#ifdef _OPENMP
#pragma omp barrier
#endif  // _OPENMP

  if (kUseAbscabForAxisCurrent) {
    AddAxisCurrentFieldAbscab(rAxis, zAxis, netToroidalCurrent);
  } else {
    AddAxisCurrentFieldSimple(rAxis, zAxis, netToroidalCurrent);
  }

#ifdef _OPENMP
#pragma omp barrier
#endif  // _OPENMP

  covariantAndNormalComponents();

#ifdef _OPENMP
#pragma omp barrier
#endif  // _OPENMP
}

// add in contribution from net toroidal current along magnetic axis
void ExternalMagneticField::AddAxisCurrentFieldAbscab(
    const std::span<const double> rAxis, const std::span<const double> zAxis,
    double netToroidalCurrent) {
  // copy over axis geometry in first module
  // and convert to Cartesian coordinates
  for (int k = 0; k < s_.nZeta; ++k) {
    axisXYZ[k * 3 + 0] = rAxis[k] * sg_.cos_phi[k];
    axisXYZ[k * 3 + 1] = rAxis[k] * sg_.sin_phi[k];
    axisXYZ[k * 3 + 2] = zAxis[k];
  }  // k

  // rotate into other modules
  for (int p = 1; p < s_.nfp; ++p) {
    for (int k = 0; k < s_.nZeta; ++k) {
      axisXYZ[(p * s_.nZeta + k) * 3 + 0] =
          sg_.cos_per[p] * axisXYZ[k * 3 + 0] -
          sg_.sin_per[p] * axisXYZ[k * 3 + 1];
      axisXYZ[(p * s_.nZeta + k) * 3 + 1] =
          sg_.sin_per[p] * axisXYZ[k * 3 + 0] +
          sg_.cos_per[p] * axisXYZ[k * 3 + 1];
      axisXYZ[(p * s_.nZeta + k) * 3 + 2] = zAxis[k];
    }  // k
  }  // field periods

  // close the loop
  axisXYZ[s_.nZeta * s_.nfp * 3 + 0] = axisXYZ[0];
  axisXYZ[s_.nZeta * s_.nfp * 3 + 1] = axisXYZ[1];
  axisXYZ[s_.nZeta * s_.nfp * 3 + 2] = axisXYZ[2];

  // convert points on surface into surfaceXYZ
  // TODO(jons): might be useful to have interface in abscab,
  // where x, y and z can be specified as three separate arrays
  // (or transpose points array to remove interleave)
  // NOTE: Cannot use rcosuv, rsinuv here,
  // since they are only needed (and thus update) for a full update!
  for (int kl = tp_.ztMin; kl < tp_.ztMax; ++kl) {
    const int k = kl % s_.nZeta;
    surfaceXYZ[(kl - tp_.ztMin) * 3 + 0] = sg_.r1b[kl] * sg_.cos_phi[k];
    surfaceXYZ[(kl - tp_.ztMin) * 3 + 1] = sg_.r1b[kl] * sg_.sin_phi[k];
    surfaceXYZ[(kl - tp_.ztMin) * 3 + 2] = sg_.z1b[kl];
  }  // kl

  // TODO(jons): use callback for providing Cartesian axis geometry
  // --> VertexSupplier

  // store into member for being able to debug it
  axis_current = netToroidalCurrent;

  // Initialize target storage for axis-current magnetic field to zero,
  // since ABSCAB only adds to whatever is in there.
  // If we don't do this, the axis current contributions effectively pile up,
  // iteration after iteration (don't ask how I know about this...).
  const int numLocal = tp_.ztMax - tp_.ztMin;
  absl::c_fill_n(bCoilsXYZ, 3 * numLocal, 0);

  // compute magnetic field due to line current along magnetic axis
  int numProcessors = 1;  // Nestor itself is already parallelized via OpenMP
  abscab::magneticFieldPolygonFilament(s_.nZeta * s_.nfp + 1, axisXYZ.data(),
                                       axis_current, tp_.ztMax - tp_.ztMin,
                                       surfaceXYZ.data(), bCoilsXYZ.data(),
                                       numProcessors);

  // transform bCoilsXYZ into cylindrical coordinates
  for (int kl = tp_.ztMin; kl < tp_.ztMax; ++kl) {
    int k = kl % s_.nZeta;

    double _bX = bCoilsXYZ[3 * (kl - tp_.ztMin) + 0];
    double _bY = bCoilsXYZ[3 * (kl - tp_.ztMin) + 1];
    double _bZ = bCoilsXYZ[3 * (kl - tp_.ztMin) + 2];

    double _bR = sg_.cos_phi[k] * _bX + sg_.sin_phi[k] * _bY;
    double _bP = sg_.cos_phi[k] * _bY - sg_.sin_phi[k] * _bX;

    curtorBr[kl - tp_.ztMin] = _bR;
    curtorBp[kl - tp_.ztMin] = _bP;
    curtorBz[kl - tp_.ztMin] = _bZ;
  }  // kl
}

void ExternalMagneticField::AddAxisCurrentFieldSimple(
    const std::span<const double> rAxis, const std::span<const double> zAxis,
    double netToroidalCurrent) {
  // copy over axis geometry in first module
  // and convert to Cartesian coordinates
  for (int k = 0; k < s_.nZeta; ++k) {
    axisXYZ[k * 3 + 0] = rAxis[k] * sg_.cos_phi[k];
    axisXYZ[k * 3 + 1] = rAxis[k] * sg_.sin_phi[k];
    axisXYZ[k * 3 + 2] = zAxis[k];
  }  // k

  // rotate into other modules
  for (int p = 1; p < s_.nfp; ++p) {
    for (int k = 0; k < s_.nZeta; ++k) {
      axisXYZ[(p * s_.nZeta + k) * 3 + 0] =
          sg_.cos_per[p] * axisXYZ[k * 3 + 0] -
          sg_.sin_per[p] * axisXYZ[k * 3 + 1];
      axisXYZ[(p * s_.nZeta + k) * 3 + 1] =
          sg_.sin_per[p] * axisXYZ[k * 3 + 0] +
          sg_.cos_per[p] * axisXYZ[k * 3 + 1];
      axisXYZ[(p * s_.nZeta + k) * 3 + 2] = zAxis[k];
    }  // k
  }  // field periods

  // close the loop
  axisXYZ[s_.nZeta * s_.nfp * 3 + 0] = axisXYZ[0];
  axisXYZ[s_.nZeta * s_.nfp * 3 + 1] = axisXYZ[1];
  axisXYZ[s_.nZeta * s_.nfp * 3 + 2] = axisXYZ[2];

  // convert points on surface into surfaceXYZ
  // TODO(jons): might be useful to have interface in abscab,
  // where x, y and z can be specified as three separate arrays
  // (or transpose points array to remove interleave)
  // NOTE: Cannot use rcosuv, rsinuv here,
  // since they are only needed (and thus update) for a full update!
  for (int kl = tp_.ztMin; kl < tp_.ztMax; ++kl) {
    const int k = kl % s_.nZeta;
    surfaceXYZ[(kl - tp_.ztMin) * 3 + 0] = sg_.r1b[kl] * sg_.cos_phi[k];
    surfaceXYZ[(kl - tp_.ztMin) * 3 + 1] = sg_.r1b[kl] * sg_.sin_phi[k];
    surfaceXYZ[(kl - tp_.ztMin) * 3 + 2] = sg_.z1b[kl];
  }  // kl

  // store into member for being able to debug it
  axis_current = netToroidalCurrent;

  // -------
  // Here, we have the (static in the context of this method) axis geometry
  // (axisXYZ) as well as the geometry of the evaluation locations (surfaceXYZ)
  // done.
  // -------

  // Initialize target storage for axis-current magnetic field to zero.
  // If we don't do this, the axis current contributions effectively pile up,
  // iteration after iteration (don't ask how I know about this...).
  const int numLocal = tp_.ztMax - tp_.ztMin;
  absl::c_fill_n(bCoilsXYZ, 3 * numLocal, 0);

  // 1.0e-7 == mu0/4 pi
  // NOTE: The factor of 2 comes from the Hanson-Hirshman Biot-Savart formula,
  // which is Eqn. (8) in Hanson & Hirshman (2002) [Physics of Plasmas 9, 4410].
  const double magnetic_field_scale = 1.0e-7 * netToroidalCurrent * 2.0;

  for (int source_index = 0; source_index < s_.nZeta * s_.nfp; ++source_index) {
    const double segment_dx =
        axisXYZ[(source_index + 1) * 3 + 0] - axisXYZ[source_index * 3 + 0];
    const double segment_dy =
        axisXYZ[(source_index + 1) * 3 + 1] - axisXYZ[source_index * 3 + 1];
    const double segment_dz =
        axisXYZ[(source_index + 1) * 3 + 2] - axisXYZ[source_index * 3 + 2];

    const double segment_length =
        std::sqrt(segment_dx * segment_dx + segment_dy * segment_dy +
                  segment_dz * segment_dz);

    for (int kl = tp_.ztMin; kl < tp_.ztMax; ++kl) {
      const int kl_local = kl - tp_.ztMin;

      const double r_i_x =
          surfaceXYZ[kl_local * 3 + 0] - axisXYZ[source_index * 3 + 0];
      const double r_i_y =
          surfaceXYZ[kl_local * 3 + 1] - axisXYZ[source_index * 3 + 1];
      const double r_i_z =
          surfaceXYZ[kl_local * 3 + 2] - axisXYZ[source_index * 3 + 2];
      const double r_i =
          std::sqrt(r_i_x * r_i_x + r_i_y * r_i_y + r_i_z * r_i_z);

      const double r_f_x =
          surfaceXYZ[kl_local * 3 + 0] - axisXYZ[(source_index + 1) * 3 + 0];
      const double r_f_y =
          surfaceXYZ[kl_local * 3 + 1] - axisXYZ[(source_index + 1) * 3 + 1];
      const double r_f_z =
          surfaceXYZ[kl_local * 3 + 2] - axisXYZ[(source_index + 1) * 3 + 2];
      const double r_f =
          std::sqrt(r_f_x * r_f_x + r_f_y * r_f_y + r_f_z * r_f_z);

      const double r_i_plus_r_f = r_i + r_f;

      const double magnetic_field_magnitude =
          magnetic_field_scale * r_i_plus_r_f /
          (r_i * r_f *
           (r_i_plus_r_f * r_i_plus_r_f - segment_length * segment_length));

      // cross product of L*hat(eps)==dvec with Ri_vec,
      // scaled by magnetic field magnitude
      bCoilsXYZ[kl_local * 3 + 0] +=
          magnetic_field_magnitude * (segment_dy * r_i_z - segment_dz * r_i_y);
      bCoilsXYZ[kl_local * 3 + 1] +=
          magnetic_field_magnitude * (segment_dz * r_i_x - segment_dx * r_i_z);
      bCoilsXYZ[kl_local * 3 + 2] +=
          magnetic_field_magnitude * (segment_dx * r_i_y - segment_dy * r_i_x);
    }  // kl
  }  // source_index

  // -------
  // Here, we have the magnetic field from the axis current (bCoilsXYZ) done.
  // -------

  // transform bCoilsXYZ into cylindrical coordinates
  for (int kl = tp_.ztMin; kl < tp_.ztMax; ++kl) {
    int k = kl % s_.nZeta;

    double _bX = bCoilsXYZ[3 * (kl - tp_.ztMin) + 0];
    double _bY = bCoilsXYZ[3 * (kl - tp_.ztMin) + 1];
    double _bZ = bCoilsXYZ[3 * (kl - tp_.ztMin) + 2];

    double _bR = sg_.cos_phi[k] * _bX + sg_.sin_phi[k] * _bY;
    double _bP = sg_.cos_phi[k] * _bY - sg_.sin_phi[k] * _bX;

    curtorBr[kl - tp_.ztMin] = _bR;
    curtorBp[kl - tp_.ztMin] = _bP;
    curtorBz[kl - tp_.ztMin] = _bZ;
  }  // kl
}

// compute bSubU, bSubV: covariant components of external magnetic field
// and bDotN: normal component of external magnetic field
void ExternalMagneticField::covariantAndNormalComponents() {
  for (int kl = tp_.ztMin; kl < tp_.ztMax; ++kl) {
    // add contributions together
    // --> helps in debugging to have them separate until here
    double fullBr = interpBr[kl - tp_.ztMin] + curtorBr[kl - tp_.ztMin];
    double fullBp = interpBp[kl - tp_.ztMin] + curtorBp[kl - tp_.ztMin];
    double fullBz = interpBz[kl - tp_.ztMin] + curtorBz[kl - tp_.ztMin];

    // covariant components
    bSubU[kl - tp_.ztMin] =
        fullBr * sg_.rub[kl - tp_.ztMin] + fullBz * sg_.zub[kl - tp_.ztMin];
    bSubV[kl - tp_.ztMin] = fullBr * sg_.rvb[kl - tp_.ztMin] +
                            fullBz * sg_.zvb[kl - tp_.ztMin] +
                            fullBp * sg_.r1b[kl];

    // normal component
    bDotN[kl - tp_.ztMin] =
        -(fullBr * sg_.snr[kl - tp_.ztMin] + fullBp * sg_.snv[kl - tp_.ztMin] +
          fullBz * sg_.snz[kl - tp_.ztMin]);
  }  // kl
}

}  // namespace vmecpp
