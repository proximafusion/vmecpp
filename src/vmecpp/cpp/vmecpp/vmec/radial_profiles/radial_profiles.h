// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#ifndef VMECPP_VMEC_RADIAL_PROFILES_RADIAL_PROFILES_H_
#define VMECPP_VMEC_RADIAL_PROFILES_RADIAL_PROFILES_H_

#include <Eigen/Dense>
#include <cfloat>
#include <cmath>
#include <string>

#include "vmecpp/common/flow_control/flow_control.h"
#include "vmecpp/common/util/util.h"
#include "vmecpp/common/vmec_indata/vmec_indata.h"
#include "vmecpp/vmec/handover_storage/handover_storage.h"
#include "vmecpp/vmec/profile_parameterization_data/profile_parameterization_data.h"
#include "vmecpp/vmec/radial_partitioning/radial_partitioning.h"
#include "vmecpp/vmec/vmec_constants/vmec_constants.h"

namespace vmecpp {

/* radial profiles: only dependent on number of surfaces */
class RadialProfiles {
 public:
  RadialProfiles(const RadialPartitioning* s, HandoverStorage* m_h,
                 const VmecINDATA* id, const FlowControl* fc,
                 int signOfJacobian, real_t pDamp);

  // update profile parameterizations based on p****_type strings
  void setupInputProfiles();

  // Call this for every thread, one after another (ideally from a single
  // thread), providing thread_id = 0, 1, ..., (num_threads-1), where
  // num_threads is what is used in RadialPartitioning.
  void evalRadialProfiles(bool haveToFlipTheta, VmecConstants& m_vmecconst);

  ProfileParameterization findParameterization(const std::string& name,
                                               ProfileType intendedType);
  std::string profileTypeToString(ProfileType profileType);

  void computeMagneticFluxes();
  real_t torfluxDeriv(real_t x);
  real_t torflux(real_t x);
  real_t polfluxDeriv(real_t x);
  real_t polflux(real_t x);

  real_t evalMassProfile(real_t x);
  real_t evalIotaProfile(real_t x);
  real_t evalCurrProfile(real_t x);

  // Evaluate the radial profile function specified by the given
  // parameterization, which can be either an analytical function (in which case
  // it is parameterized by `coeffs`) or a spline interpolation (in which case
  // the data to be interpolated is given by `splineKnots` and `splineValues`).
  // The flag `shouldIntegrate` indicated whether the profile should be radially
  // integrated when evaluating it. The pressure and iota profiles are used
  // directly as they are specified, hence for their parameterizations one
  // should set `shouldIntegrate` to `false`. The toroidal current profile can
  // be specified both via its radial derivative and as a profile of the
  // enclosed toroidal current (hence the integral of the derivative). Thus,
  // some parameterizations of the current profile need to be radially
  // integrated and some not, which is what `shouldIntegrate` is then used for.
  // Which profile parameterization is integrated and which not is documented in
  // the body of `RadialProfiles::setupProfileParameterizations`, where `I`
  // refers to the profile parameterization specifying the enclosed toroidal
  // current profile already (hence no integration is needed), and `I-prime`
  // indicating that the given profile parameterization needs to be integrated.
  // The implementation of deciding which profile to integrate is in the body of
  // `RadialProfiles::evalProfileFunction`.
  // In the end, for the current profile, this method should always return the
  // enclosed toroidal current profile.
  // TODO(jons): This function is wearing way too many hats. Chunk it up.
  real_t evalProfileFunction(
      const ProfileParameterization& param,
      const Eigen::Matrix<real_t, Eigen::Dynamic, 1>& coeffs,
      const Eigen::Matrix<real_t, Eigen::Dynamic, 1>& splineKnots,
      const Eigen::Matrix<real_t, Eigen::Dynamic, 1>& splineValues,
      bool shouldIntegrate, real_t normX);

  real_t evalPowerSeries(const Eigen::Matrix<real_t, Eigen::Dynamic, 1>& coeffs,
                         real_t x, bool should_integrate);
  real_t evalPowerSeriesI(
      const Eigen::Matrix<real_t, Eigen::Dynamic, 1>& coeffs, real_t x);
  real_t evalGaussTrunc(const Eigen::Matrix<real_t, Eigen::Dynamic, 1>& coeffs,
                        real_t x);
  real_t evalSumAtan(const Eigen::Matrix<real_t, Eigen::Dynamic, 1>& coeffs,
                     real_t x);
  real_t evalTwoLorentz(const Eigen::Matrix<real_t, Eigen::Dynamic, 1>& coeffs,
                        real_t x);
  real_t evalTwoPower(const Eigen::Matrix<real_t, Eigen::Dynamic, 1>& coeffs,
                      real_t x, bool shouldIntegrate);
  real_t evalTwoPowerGs(const Eigen::Matrix<real_t, Eigen::Dynamic, 1>& coeffs,
                        real_t x);
  real_t evalAkima(const Eigen::Matrix<real_t, Eigen::Dynamic, 1>& splineKnots,
                   const Eigen::Matrix<real_t, Eigen::Dynamic, 1>& splineValues,
                   real_t x);
  real_t evalAkimaIntegrated(
      const Eigen::Matrix<real_t, Eigen::Dynamic, 1>& splineKnots,
      const Eigen::Matrix<real_t, Eigen::Dynamic, 1>& splineValues, real_t x);
  real_t evalCubic(const Eigen::Matrix<real_t, Eigen::Dynamic, 1>& splineKnots,
                   const Eigen::Matrix<real_t, Eigen::Dynamic, 1>& splineValues,
                   real_t x);
  real_t evalCubicIntegrated(
      const Eigen::Matrix<real_t, Eigen::Dynamic, 1>& splineKnots,
      const Eigen::Matrix<real_t, Eigen::Dynamic, 1>& splineValues, real_t x);
  real_t evalPedestal(const Eigen::Matrix<real_t, Eigen::Dynamic, 1>& coeffs,
                      real_t x);
  real_t evalRational(const Eigen::Matrix<real_t, Eigen::Dynamic, 1>& coeffs,
                      real_t x);
  real_t evalLineSegment(
      const Eigen::Matrix<real_t, Eigen::Dynamic, 1>& splineKnots,
      const Eigen::Matrix<real_t, Eigen::Dynamic, 1>& splineValues, real_t x);
  real_t evalLineSegmentIntegrated(
      const Eigen::Matrix<real_t, Eigen::Dynamic, 1>& splineKnots,
      const Eigen::Matrix<real_t, Eigen::Dynamic, 1>& splineValues, real_t x);
  real_t evalNiceQuadratic(
      const Eigen::Matrix<real_t, Eigen::Dynamic, 1>& coeffs, real_t x);

  // Accumulate contributions to volume-averaged spectral width <M>.
  void AccumulateVolumeAveragedSpectralWidth() const;

  // part 1: radial profiles directly defined by user inputs

  ProfileParameterization pmassType;
  ProfileParameterization piotaType;
  ProfileParameterization pcurrType;

  // half-grid
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> phipH;
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> chipH;
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> iotaH;
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> currH;
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> massH;
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> sqrtSH;

  // full-grid
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> phipF;
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> chipF;
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> iotaF;
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> sqrtSF;
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> radialBlending;

  // ---------------------------------

  real_t currv;
  real_t Itor;

  real_t maxToroidalFlux;
  real_t maxPoloidalFlux;

  real_t pressureScalingFactor;

  /** sm[j] = sqrt(s_{j-1/2}) / sqrt(s_j) for all force-j (numFull) */
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> sm;

  /** sp[j] = sqrt(s_{j+1/2}) / sqrt(s_j) for all force-j (numFull) */
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> sp;

  // part 2: derived radial profiles

  /** differential volume, half-grid */
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> dVdsH;

  /** kinetic pressure, half-grid */
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> presH;

  /** enclosed poloidal current, half-grid */
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> bvcoH;

  /** enclosed toroidal current, half-grid */
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> bucoH;

  /** poloidal current density, interior full-grid */
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> jcuruF;

  /** toroidal current density, interior full-grid */
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> jcurvF;

  /** pressure gradient, interior full-grid */
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> presgradF;

  /** differential volume, interior full-grid */
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> dVdsF;

  /** radial force balance residual, interior full-grid */
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> equiF;

  // [nsMinF1 ... nsMaxF1] surface-averaged spectral width profile on full-grid
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> spectral_width;

  // ---------------------------------

  /** [ns x 2] 1/sqrtSF for odd-m; 1 for even-m */
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> scalxc;

 private:
  const RadialPartitioning& r_;
  HandoverStorage& m_h_;
  const VmecINDATA& id_;
  const FlowControl& fc_;

  const int signOfJacobian;
  const real_t pDamp;

  std::vector<ProfileParameterizationData> ALL_PARAMS;

  /** one entry for every value of ProfileParameterization */
  void setupProfileParameterizations();
};

}  // namespace vmecpp

#endif  // VMECPP_VMEC_RADIAL_PROFILES_RADIAL_PROFILES_H_
