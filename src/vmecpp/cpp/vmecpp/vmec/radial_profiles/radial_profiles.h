// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#ifndef VMECPP_VMEC_RADIAL_PROFILES_RADIAL_PROFILES_H_
#define VMECPP_VMEC_RADIAL_PROFILES_RADIAL_PROFILES_H_

#include <cfloat>
#include <cmath>
#include <string>
#include <vector>

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
                 int signOfJacobian, double pDamp);

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
  double torfluxDeriv(double x);
  double torflux(double x);
  double polfluxDeriv(double x);
  double polflux(double x);

  double evalMassProfile(double x);
  double evalIotaProfile(double x);
  double evalCurrProfile(double x);

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
  double evalProfileFunction(const ProfileParameterization& param,
                             const std::vector<double>& coeffs,
                             const std::vector<double>& splineKnots,
                             const std::vector<double>& splineValues,
                             bool shouldIntegrate, double normX);

  double evalPowerSeries(const std::vector<double>& coeffs, double x,
                         bool should_integrate);
  double evalPowerSeriesI(const std::vector<double>& coeffs, double x);
  double evalGaussTrunc(const std::vector<double>& coeffs, double x);
  double evalSumAtan(const std::vector<double>& coeffs, double x);
  double evalTwoLorentz(const std::vector<double>& coeffs, double x);
  double evalTwoPower(const std::vector<double>& coeffs, double x,
                      bool shouldIntegrate);
  double evalTwoPowerGs(const std::vector<double>& coeffs, double x);
  double evalAkima(const std::vector<double>& splineKnots,
                   const std::vector<double>& splineValues, double x);
  double evalAkimaIntegrated(const std::vector<double>& splineKnots,
                             const std::vector<double>& splineValues, double x);
  double evalCubic(const std::vector<double>& splineKnots,
                   const std::vector<double>& splineValues, double x);
  double evalCubicIntegrated(const std::vector<double>& splineKnots,
                             const std::vector<double>& splineValues, double x);
  double evalPedestal(const std::vector<double>& coeffs, double x);
  double evalRational(const std::vector<double>& coeffs, double x);
  double evalLineSegment(const std::vector<double>& splineKnots,
                         const std::vector<double>& splineValues, double x);
  double evalLineSegmentIntegrated(const std::vector<double>& splineKnots,
                                   const std::vector<double>& splineValues,
                                   double x);
  double evalNiceQuadratic(const std::vector<double>& coeffs, double x);

  // Accumulate contributions to volume-averaged spectral width <M>.
  void AccumulateVolumeAveragedSpectralWidth() const;

  // part 1: radial profiles directly defined by user inputs

  ProfileParameterization pmassType;
  ProfileParameterization piotaType;
  ProfileParameterization pcurrType;

  // half-grid
  std::vector<double> phipH;
  std::vector<double> chipH;
  std::vector<double> iotaH;
  std::vector<double> currH;
  std::vector<double> massH;
  std::vector<double> sqrtSH;

  // full-grid
  std::vector<double> phipF;
  std::vector<double> chipF;
  std::vector<double> iotaF;
  std::vector<double> sqrtSF;
  std::vector<double> radialBlending;

  // ---------------------------------

  double currv;
  double Itor;

  double maxToroidalFlux;
  double maxPoloidalFlux;

  double pressureScalingFactor;

  /** sm[j] = sqrt(s_{j-1/2}) / sqrt(s_j) for all force-j (numFull) */
  std::vector<double> sm;

  /** sp[j] = sqrt(s_{j+1/2}) / sqrt(s_j) for all force-j (numFull) */
  std::vector<double> sp;

  // part 2: derived radial profiles

  /** differential volume, half-grid */
  std::vector<double> dVdsH;

  /** kinetic pressure, half-grid */
  std::vector<double> presH;

  /** enclosed poloidal current, half-grid */
  std::vector<double> bvcoH;

  /** enclosed toroidal current, half-grid */
  std::vector<double> bucoH;

  /** poloidal current density, interior full-grid */
  std::vector<double> jcuruF;

  /** toroidal current density, interior full-grid */
  std::vector<double> jcurvF;

  /** pressure gradient, interior full-grid */
  std::vector<double> presgradF;

  /** differential volume, interior full-grid */
  std::vector<double> dVdsF;

  /** radial force balance residual, interior full-grid */
  std::vector<double> equiF;

  // [nsMinF1 ... nsMaxF1] surface-averaged spectral width profile on full-grid
  std::vector<double> spectral_width;

  // ---------------------------------

  /** [ns x 2] 1/sqrtSF for odd-m; 1 for even-m */
  std::vector<double> scalxc;

 private:
  const RadialPartitioning& r_;
  HandoverStorage& m_h_;
  const VmecINDATA& id_;
  const FlowControl& fc_;

  const int signOfJacobian;
  const double pDamp;

  std::vector<ProfileParameterizationData> ALL_PARAMS;

  /** one entry for every value of ProfileParameterization */
  void setupProfileParameterizations();
};

}  // namespace vmecpp

#endif  // VMECPP_VMEC_RADIAL_PROFILES_RADIAL_PROFILES_H_
