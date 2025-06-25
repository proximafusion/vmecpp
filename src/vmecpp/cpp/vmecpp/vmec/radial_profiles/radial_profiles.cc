// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#include "vmecpp/vmec/radial_profiles/radial_profiles.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

#include "absl/log/log.h"
#include "absl/strings/str_format.h"
#include "vmecpp/vmec/profile_parameterization_data/profile_parameterization_data.h"
#include "vmecpp/vmec/vmec_constants/vmec_algorithm_constants.h"
#include "vmecpp/vmec/vmec_constants/vmec_constants.h"

using vmecpp::vmec_algorithm_constants::kEvenParity;
using vmecpp::vmec_algorithm_constants::kOddParity;

namespace vmecpp {

RadialProfiles::RadialProfiles(const RadialPartitioning* r,
                               HandoverStorage* m_h, const VmecINDATA* id,
                               const FlowControl* fc, int signOfJacobian,
                               double pDamp)
    : r_(*r),
      m_h_(*m_h),
      id_(*id),
      fc_(*fc),
      signOfJacobian(signOfJacobian),
      pDamp(pDamp) {
  maxToroidalFlux = 0.0;
  maxPoloidalFlux = 0.0;

  currv = 0.0;
  Itor = 0.0;
  pressureScalingFactor = 0.0;

  pmassType = ProfileParameterization::INVALID_PARAM;
  piotaType = ProfileParameterization::INVALID_PARAM;
  pcurrType = ProfileParameterization::INVALID_PARAM;

  setupProfileParameterizations();

  // half-grid
  phipH.resize(r_.nsMaxH - r_.nsMinH);
  chipH.resize(r_.nsMaxH - r_.nsMinH);
  iotaH.resize(r_.nsMaxH - r_.nsMinH);
  currH.resize(r_.nsMaxH - r_.nsMinH);
  massH.resize(r_.nsMaxH - r_.nsMinH);
  sqrtSH.resize(r_.nsMaxH - r_.nsMinH);

  // full-grid
  phipF.resize(r_.nsMaxF1 - r_.nsMinF1);
  chipF.resize(r_.nsMaxF1 - r_.nsMinF1);
  iotaF.resize(r_.nsMaxF1 - r_.nsMinF1);
  sqrtSF.resize(r_.nsMaxF1 - r_.nsMinF1);
  radialBlending.resize(r_.nsMaxF1 - r_.nsMinF1);

  // ---------------------------------

  dVdsH.resize(r_.nsMaxH - r_.nsMinH);
  presH.resize(r_.nsMaxH - r_.nsMinH);
  bucoH.resize(r_.nsMaxH - r_.nsMinH);
  bvcoH.resize(r_.nsMaxH - r_.nsMinH);

  jcuruF.resize(r_.nsMaxFi - r_.nsMinFi);
  jcurvF.resize(r_.nsMaxFi - r_.nsMinFi);
  presgradF.resize(r_.nsMaxFi - r_.nsMinFi);
  dVdsF.resize(r_.nsMaxFi - r_.nsMinFi);
  equiF.resize(r_.nsMaxFi - r_.nsMinFi);

  spectral_width.resize(r_.nsMaxF1 - r_.nsMinF1, 0.0);

  // ---------------------------------

  scalxc.resize((r_.nsMaxF1 - r_.nsMinF1) * 2);

  sm.resize(r_.nsMaxH - r_.nsMinH, 0.0);
  sp.resize(r_.nsMaxH - r_.nsMinH, 0.0);
}

void RadialProfiles::setupInputProfiles() {
  pmassType = findParameterization(id_.pmass_type, ProfileType::PRESSURE);
  piotaType = findParameterization(id_.piota_type, ProfileType::IOTA);
  pcurrType = findParameterization(id_.pcurr_type, ProfileType::CURRENT);

  pressureScalingFactor = MU_0 * id_.pres_scale;

  computeMagneticFluxes();
}

void RadialProfiles::setupProfileParameterizations() {
  // clang-format off
  //                       current | iota | pressure
  //                       --------+------+---------
  // INVALID_PARAM                 |      |
  // POWER_SERIES          I-prime |   X  |     X
  // POWER_SERIES_I        I       |      |
  // GAUSS_TRUNC           I-prime |      |     X
  // SUM_ATAN              I       |   X  |
  // TWO_LORENTZ                   |      |     X
  // TWO_POWER             I-prime |      |     X
  // TWO_POWER_GS          I-prime |      |     X
  // AKIMA_SPLINE                  |   X  |     X
  // AKIMA_SPLINE_I        I       |      |
  // AKIMA_SPLINE_IP       I-prime |      |
  // CUBIC_SPLINE                  |   X  |     X
  // CUBIC_SPLINE_I        I       |      |
  // CUBIC_SPLINE_IP       I-prime |      |
  // PEDESTAL              I       |      |     X
  // RATIONAL              I       |   X  |     X
  // LINE_SEGMENT                  |   X  |     X
  // LINE_SEGMENT_I        I       |      |
  // LINE_SEGMENT_IP       I-prime |      |
  // NICE_QUADRATIC                |   X  |
  // SUM_COSSQ_S           I-prime |      |
  // SUM_COSSQ_SQRTS       I-prime |      |
  // SUM_COSSQ_S_FREE      I-prime |      |
  // clang-format on

  ALL_PARAMS.reserve(NUM_PARAM);
  ALL_PARAMS.emplace_back("---invalid---", /*allowedForPres=*/false,
                          /*allowedForCurr*/ false, /*allowedForIota*/ false,
                          /*needsSplineData*/ false);
  ALL_PARAMS.emplace_back("power_series", /*allowedForPres=*/true,
                          /*allowedForCurr*/ true, /*allowedForIota*/ true,
                          /*needsSplineData*/ false);
  ALL_PARAMS.emplace_back("power_series_i", /*allowedForPres=*/false,
                          /*allowedForCurr*/ true, /*allowedForIota*/ false,
                          /*needsSplineData*/ false);
  ALL_PARAMS.emplace_back("gauss_trunc", /*allowedForPres=*/true,
                          /*allowedForCurr*/ true, /*allowedForIota*/ false,
                          /*needsSplineData*/ false);
  ALL_PARAMS.emplace_back("sum_atan", /*allowedForPres=*/false,
                          /*allowedForCurr*/ true, /*allowedForIota*/ true,
                          /*needsSplineData*/ false);
  ALL_PARAMS.emplace_back("two_lorentz", /*allowedForPres=*/true,
                          /*allowedForCurr*/ false, /*allowedForIota*/ false,
                          /*needsSplineData*/ false);
  ALL_PARAMS.emplace_back("two_power", /*allowedForPres=*/true,
                          /*allowedForCurr*/ true, /*allowedForIota*/ false,
                          /*needsSplineData*/ false);
  ALL_PARAMS.emplace_back("two_power_gs", /*allowedForPres=*/true,
                          /*allowedForCurr*/ true, /*allowedForIota*/ false,
                          /*needsSplineData*/ false);
  ALL_PARAMS.emplace_back("akima_spline", /*allowedForPres=*/true,
                          /*allowedForCurr*/ false, /*allowedForIota*/ true,
                          /*needsSplineData*/ true);
  ALL_PARAMS.emplace_back("akima_spline_i", /*allowedForPres=*/false,
                          /*allowedForCurr*/ true, /*allowedForIota*/ false,
                          /*needsSplineData*/ true);
  ALL_PARAMS.emplace_back("akima_spline_ip", /*allowedForPres=*/false,
                          /*allowedForCurr*/ true, /*allowedForIota*/ false,
                          /*needsSplineData*/ true);
  ALL_PARAMS.emplace_back("cubic_spline", /*allowedForPres=*/true,
                          /*allowedForCurr*/ false, /*allowedForIota*/ true,
                          /*needsSplineData*/ true);
  ALL_PARAMS.emplace_back("cubic_spline_i", /*allowedForPres=*/false,
                          /*allowedForCurr*/ true, /*allowedForIota*/ false,
                          /*needsSplineData*/ true);
  ALL_PARAMS.emplace_back("cubic_spline_ip", /*allowedForPres=*/false,
                          /*allowedForCurr*/ true, /*allowedForIota*/ false,
                          /*needsSplineData*/ true);
  ALL_PARAMS.emplace_back("pedestal", /*allowedForPres=*/true,
                          /*allowedForCurr*/ true, /*allowedForIota*/ false,
                          /*needsSplineData*/ false);
  ALL_PARAMS.emplace_back("rational", /*allowedForPres=*/true,
                          /*allowedForCurr*/ true, /*allowedForIota*/ true,
                          /*needsSplineData*/ false);
  ALL_PARAMS.emplace_back("line_segment", /*allowedForPres=*/true,
                          /*allowedForCurr*/ false, /*allowedForIota*/ true,
                          /*needsSplineData*/ true);
  ALL_PARAMS.emplace_back("line_segment_i", /*allowedForPres=*/false,
                          /*allowedForCurr*/ true, /*allowedForIota*/ false,
                          /*needsSplineData*/ true);
  ALL_PARAMS.emplace_back("line_segment_ip", /*allowedForPres=*/false,
                          /*allowedForCurr*/ true, /*allowedForIota*/ false,
                          /*needsSplineData*/ true);
  ALL_PARAMS.emplace_back("nice_quadratic", /*allowedForPres=*/false,
                          /*allowedForCurr*/ false, /*allowedForIota*/ true,
                          /*needsSplineData*/ false);
  ALL_PARAMS.emplace_back("sum_cossq_s", /*allowedForPres=*/false,
                          /*allowedForCurr*/ true, /*allowedForIota*/ false,
                          /*needsSplineData*/ false);
  ALL_PARAMS.emplace_back("sum_cossq_sqrts", /*allowedForPres=*/false,
                          /*allowedForCurr*/ true, /*allowedForIota*/ false,
                          /*needsSplineData*/ false);
  ALL_PARAMS.emplace_back("sum_cossq_s_free", /*allowedForPres=*/false,
                          /*allowedForCurr*/ true, /*allowedForIota*/ false,
                          /*needsSplineData*/ false);
}

ProfileParameterization RadialProfiles::findParameterization(
    const std::string& name, ProfileType intendedType) {
  for (int i = 0; i < NUM_PARAM; ++i) {
    if (name == ALL_PARAMS[i].Name()) {
      bool isApplicable = false;
      switch (intendedType) {
        case ProfileType::PRESSURE:
          isApplicable = ALL_PARAMS[i].IsAllowedFor().pres;
          break;
        case ProfileType::CURRENT:
          isApplicable = ALL_PARAMS[i].IsAllowedFor().curr;
          break;
        case ProfileType::IOTA:
          isApplicable = ALL_PARAMS[i].IsAllowedFor().iota;
          break;
        default:
          std::cerr << absl::StrFormat("unknown profile: %s",
                                       profileTypeToString(intendedType))
                    << '\n';
          break;
      }

      if (!isApplicable) {
        std::cerr << absl::StrFormat(
                         "profile name '%s' is not applicable for %s profile",
                         ALL_PARAMS[i].Name(),
                         profileTypeToString(intendedType))
                  << '\n';
        return ProfileParameterization::INVALID_PARAM;
      }

      return static_cast<ProfileParameterization>(i);
    }
  }
  return ProfileParameterization::INVALID_PARAM;
}

std::string RadialProfiles::profileTypeToString(ProfileType profileType) {
  switch (profileType) {
    case ProfileType::PRESSURE:
      return "pressure";
    case ProfileType::CURRENT:
      return "current";
    case ProfileType::IOTA:
      return "iota";
    default:
      return "<unknown>";
  }
}

/** Compute the maximum toroidal and poloidal magnetic fluxes. */
void RadialProfiles::computeMagneticFluxes() {
  maxToroidalFlux = signOfJacobian * id_.phiedge / (2.0 * M_PI);
  double edgeToroidalFluxFromProfile = torflux(1.0);
  if (edgeToroidalFluxFromProfile != 0.0) {
    maxToroidalFlux /= edgeToroidalFluxFromProfile;
  }

  // only required for lRFP == true (TODO) ...later...
  // This assumes that the same scaling factor (=phiedge) is used for phi' and
  // chi'.
  maxPoloidalFlux = maxToroidalFlux;
  double edgePoloidalFluxFromProfile = polflux(1.0);
  if (edgePoloidalFluxFromProfile != 0.0) {
    maxPoloidalFlux /= edgePoloidalFluxFromProfile;
  }
}

/**
 * Evaluate the derivative of the aphi polynomial at position x
 * @param x evaluation position
 * @return d(aphi)/dx at x
 */
double RadialProfiles::torfluxDeriv(double x) {
  double torflux_deriv = 0.0;
  for (int i = static_cast<int>(id_.aphi.size()) - 1; i >= 0; i--) {
    torflux_deriv = x * torflux_deriv + (i + 1) * id_.aphi[i];
  }
  return torflux_deriv;
}

/**
 * Compute radial profile of enclosed toroidal flux
 * by trapezoidal quadrature.
 * TODO: why not directly eval the aphi polynomial here?
 *
 * @param x
 * @return
 */
double RadialProfiles::torflux(double x) {
  //  trapezoidal integration outwards from 0 to x in N=100 steps
  const int N = 100;
  double delta_x = x / N;

  double torflux = 0.0;
  for (int i = 0; i <= N; ++i) {
    double contribution = torfluxDeriv(i * delta_x);
    if (i == 0 || i == N) {
      torflux += 0.5 * contribution;
    } else {
      torflux += contribution;
    }
  }
  torflux *= delta_x;

  return torflux;
}

/**
 * TOKAMAK/STELLARATOR: d(chi)/ds = iota * d(phi)/ds
 *
 * @param x
 * @return
 */
double RadialProfiles::polfluxDeriv(double x) {
  // figure out what toroidal flux x corresponds to
  double tf = std::min(torflux(x), 1.0);

  // profiles are always specified in toroidal flux --> eval at toroidal flux
  // corresponding to x
  double iota = evalIotaProfile(tf);

  double polflux_deriv = iota * torfluxDeriv(x);
  return polflux_deriv;
}

/**
 * Compute radial profile of enclosed poloidal flux
 * by trapezoidal quadrature.
 *
 * @param x
 * @return
 */
double RadialProfiles::polflux(double x) {
  // trapezoidal integration outwards from 0 to x in N=100 steps
  const int N = 100;
  double delta_x = x / N;

  double polflux = 0.0;
  for (int i = 0; i <= N; ++i) {
    double contribution = polfluxDeriv(i * delta_x);
    if (i == 0 || i == N) {
      polflux += 0.5 * contribution;
    } else {
      polflux += contribution;
    }
  }
  polflux *= delta_x;

  return polflux;
}

double RadialProfiles::evalMassProfile(double x) {
  // apply bloating factor
  // only allowed for current and pressure profile (checked in
  // VmecIndata.sanitize())
  double normX = std::min(fabs(x * id_.bloat), 1.0);

  double p = evalProfileFunction(pmassType, id_.am, id_.am_aux_s, id_.am_aux_f,
                                 /*shouldIntegrate=*/false, normX);

  return p * pressureScalingFactor;
}

double RadialProfiles::evalIotaProfile(double x) {
  double p = evalProfileFunction(piotaType, id_.ai, id_.ai_aux_s, id_.ai_aux_f,
                                 /*shouldIntegrate=*/false, x);

  return p;
}

double RadialProfiles::evalCurrProfile(double x) {
  // apply bloating factor
  // only allowed for current and pressure profile (checked in
  // VmecIndata.sanitize())
  double normX = std::min(std::abs(x * id_.bloat), 1.0);

  double p = evalProfileFunction(pcurrType, id_.ac, id_.ac_aux_s, id_.ac_aux_f,
                                 /*shouldIntegrate=*/true, normX);

  return p;
}

double RadialProfiles::evalProfileFunction(
    const ProfileParameterization& param, const std::vector<double>& coeffs,
    const std::vector<double>& splineKnots,
    const std::vector<double>& splineValues, bool shouldIntegrate,
    double normX) {
  switch (param) {
    case ProfileParameterization::POWER_SERIES:
      return evalPowerSeries(coeffs, normX, shouldIntegrate);
    case ProfileParameterization::POWER_SERIES_I:
      return evalPowerSeriesI(coeffs, normX);
    case ProfileParameterization::GAUSS_TRUNC:
      return evalGaussTrunc(coeffs, normX);
    case ProfileParameterization::SUM_ATAN:
      return evalSumAtan(coeffs, normX);
    case ProfileParameterization::TWO_LORENTZ:
      return evalTwoLorentz(coeffs, normX);
    case ProfileParameterization::TWO_POWER:
      return evalTwoPower(coeffs, normX, shouldIntegrate);
    case ProfileParameterization::TWO_POWER_GS:
      return evalTwoPowerGs(coeffs, normX);
    case ProfileParameterization::AKIMA_SPLINE:
    case ProfileParameterization::AKIMA_SPLINE_I:
      return evalAkima(splineKnots, splineValues, normX);
    case ProfileParameterization::AKIMA_SPLINE_IP:
      return evalAkimaIntegrated(splineKnots, splineValues, normX);
    case ProfileParameterization::CUBIC_SPLINE:
    case ProfileParameterization::CUBIC_SPLINE_I:
      return evalCubic(splineKnots, splineValues, normX);
    case ProfileParameterization::CUBIC_SPLINE_IP:
      return evalCubicIntegrated(splineKnots, splineValues, normX);
    case ProfileParameterization::PEDESTAL:
      return evalPedestal(coeffs, normX);
    case ProfileParameterization::RATIONAL:
      return evalRational(coeffs, normX);
    case ProfileParameterization::LINE_SEGMENT:
    case ProfileParameterization::LINE_SEGMENT_I:
      return evalLineSegment(splineKnots, splineValues, normX);
    case ProfileParameterization::LINE_SEGMENT_IP:
      return evalLineSegmentIntegrated(splineKnots, splineValues, normX);
    case ProfileParameterization::NICE_QUADRATIC:
      return evalNiceQuadratic(coeffs, normX);
    case ProfileParameterization::SUM_COSSQ_S:
    case ProfileParameterization::SUM_COSSQ_SQRTS:
    case ProfileParameterization::SUM_COSSQ_S_FREE:
    default:
      std::cerr << absl::StrFormat(
                       "profile parameterization '%s' not implemented yet",
                       ALL_PARAMS[static_cast<int>(param)].Name())
                << '\n';
  }

  return 0.0;
}

double RadialProfiles::evalPowerSeries(const std::vector<double>& coeffs,
                                       double x, bool should_integrate) {
  double ret = 0.0;
  int i = static_cast<int>(coeffs.size()) - 1;
  for (auto it = coeffs.rbegin(); it != coeffs.rend(); ++it) {
    const double coeff = *it;
    if (should_integrate) {
      ret = x * ret + coeff / (i + 1);
    } else {
      ret = x * ret + coeff;
    }
    i--;
  }
  if (should_integrate) {
    ret *= x;
  }
  return ret;
}

double RadialProfiles::evalPowerSeriesI(const std::vector<double>& coeffs,
                                        double x) {
  double ret = 0.0;
  for (auto it = coeffs.rbegin(); it != coeffs.rend(); ++it) {
    const double coeff = *it;
    ret = (ret + coeff) * x;
  }
  return ret;
}

double RadialProfiles::evalGaussTrunc(const std::vector<double>& coeffs,
                                      double x) {
  // TODO(jons): implement `gauss_trunc`
  (void)coeffs;
  (void)x;
  return 0.0;
}

double RadialProfiles::evalSumAtan(const std::vector<double>& coeffs,
                                   double x) {
  double ret = 0.0;

  if (coeffs.size() > 0) {
    ret = coeffs[0];
  }

  if (x >= 1.0) {
    if (coeffs.size() >= 2) {
      ret += coeffs[1];
    }
    if (coeffs.size() >= 6) {
      ret += coeffs[5];
    }
    if (coeffs.size() >= 10) {
      ret += coeffs[9];
    }
    if (coeffs.size() >= 14) {
      ret += coeffs[13];
    }
    if (coeffs.size() >= 18) {
      ret += coeffs[17];
    }
  } else {
    if (coeffs.size() >= 5) {
      ret += coeffs[1] * std::atan(coeffs[2] * std::pow(x, coeffs[3]) /
                                   std::pow(1 - x, coeffs[4]));
    }
    if (coeffs.size() >= 9) {
      ret += coeffs[5] * std::atan(coeffs[6] * std::pow(x, coeffs[7]) /
                                   std::pow(1 - x, coeffs[8]));
    }
    if (coeffs.size() >= 13) {
      ret += coeffs[9] * std::atan(coeffs[10] * std::pow(x, coeffs[11]) /
                                   std::pow(1 - x, coeffs[12]));
    }
    if (coeffs.size() >= 17) {
      ret += coeffs[13] * std::atan(coeffs[14] * std::pow(x, coeffs[15]) /
                                    std::pow(1 - x, coeffs[16]));
    }
    if (coeffs.size() >= 21) {
      ret += coeffs[17] * std::atan(coeffs[18] * std::pow(x, coeffs[19]) /
                                    std::pow(1 - x, coeffs[20]));
    }
  }

  return ret;
}

double RadialProfiles::evalTwoLorentz(const std::vector<double>& coeffs,
                                      double x) {
  if (coeffs.size() < 8) {
    LOG(WARNING)
        << "too few coefficients for 'two_lorentz' profile; need 8, got "
        << coeffs.size() << "\n";
    return 0.0;
  }
  double ret = 0.0;
  ret =
      coeffs[0] *
      (coeffs[1] *
           (1.0 /
                std::pow(1.0 + std::pow(x / (coeffs[2] * coeffs[2]), coeffs[3]),
                         coeffs[4]) -
            1.0 / std::pow(
                      1.0 + std::pow(1.0 / (coeffs[2] * coeffs[2]), coeffs[3]),
                      coeffs[4])) /
           (1.0 - 1.0 / std::pow(1.0 + std::pow(1.0 / (coeffs[2] * coeffs[2]),
                                                coeffs[3]),
                                 coeffs[4])) +
       (1.0 - coeffs[1]) *
           (1.0 /
                std::pow(1.0 + std::pow(x / (coeffs[5] * coeffs[5]), coeffs[6]),
                         coeffs[7]) -
            1.0 / std::pow(
                      1.0 + std::pow(1.0 / (coeffs[5] * coeffs[5]), coeffs[6]),
                      coeffs[7])) /
           (1.0 - 1.0 / std::pow(1.0 + std::pow(1.0 / (coeffs[5] * coeffs[5]),
                                                coeffs[6]),
                                 coeffs[7])));
  return ret;
}

double RadialProfiles::evalTwoPower(const std::vector<double>& coeffs, double x,
                                    bool shouldIntegrate) {
  if (coeffs.size() < 3) {
    LOG(WARNING) << "too few coefficients for 'two_power' profile; need 3, got "
                 << coeffs.size() << "\n";
    return 0.0;
  }
  double ret = 0.0;
  if (!shouldIntegrate) {
    // p(s) = am(0) * [1 - s**am(1)]**am(2)
    ret = coeffs[0] * std::pow(1.0 - std::pow(x, coeffs[1]), coeffs[2]);
  } else {
    // regular 10-pt Gauss-Legendre quadrature from pcurr() in Fortran VMEC
    int ngl = 10;
    std::array glx = {0.01304673574141414, 0.06746831665550774,
                      0.1602952158504878,  0.2833023029353764,
                      0.4255628305091844,  0.5744371694908156,
                      0.7166976970646236,  0.8397047841495122,
                      0.9325316833444923,  0.9869532642585859};
    std::array glw = {0.03333567215434407, 0.0747256745752903,
                      0.1095431812579910,  0.1346333596549982,
                      0.1477621123573764,  0.1477621123573764,
                      0.1346333596549982,  0.1095431812579910,
                      0.0747256745752903,  0.03333567215434407};

    ret = 0.0;
    for (int i = 0; i < ngl; ++i) {
      const double xp = x * glx[i];
      ret += glw[i] * coeffs[0] *
             std::pow(1.0 - std::pow(xp, coeffs[1]), coeffs[2]);
    }
    ret *= x;
  }
  return ret;
}

double RadialProfiles::evalTwoPowerGs(const std::vector<double>& coeffs,
                                      double x) {
  // TODO(jons): implement `two_power_gs`
  (void)coeffs;
  (void)x;
  return 0.0;
}

double RadialProfiles::evalAkima(const std::vector<double>& splineKnots,
                                 const std::vector<double>& splineValues,
                                 double x) {
  // TODO(jons): implement `akima_spline`
  (void)splineKnots;
  (void)splineValues;
  (void)x;
  return 0.0;
}

double RadialProfiles::evalAkimaIntegrated(
    const std::vector<double>& splineKnots,
    const std::vector<double>& splineValues, double x) {
  // TODO(jons): implement `akima_spline_i`
  (void)splineKnots;
  (void)splineValues;
  (void)x;
  return 0.0;
}

double RadialProfiles::evalCubic(const std::vector<double>& splineKnots,
                                 const std::vector<double>& splineValues,
                                 double x) {
  // TODO(jons): implement `cubic_spline`
  (void)splineKnots;
  (void)splineValues;
  (void)x;
  return 0.0;
}

double RadialProfiles::evalCubicIntegrated(
    const std::vector<double>& splineKnots,
    const std::vector<double>& splineValues, double x) {
  // TODO(jons): implement `cubic_spline_i`
  (void)splineKnots;
  (void)splineValues;
  (void)x;

  return 0.0;
}

double RadialProfiles::evalPedestal(const std::vector<double>& coeffs,
                                    double x) {
  // TODO(jons): implement `pedestal`
  (void)coeffs;
  (void)x;
  return 0.0;
}

double RadialProfiles::evalRational(const std::vector<double>& coeffs,
                                    double x) {
  // TODO(jons): implement `rational`
  (void)coeffs;
  (void)x;
  return 0.0;
}

double RadialProfiles::evalLineSegment(const std::vector<double>& splineKnots,
                                       const std::vector<double>& splineValues,
                                       double x) {
  // TODO(jons): implement `line_segment`
  (void)splineKnots;
  (void)splineValues;
  (void)x;
  return 0.0;
}

double RadialProfiles::evalLineSegmentIntegrated(
    const std::vector<double>& splineKnots,
    const std::vector<double>& splineValues, double x) {
  // TODO(jons): implement `line_segment_i`
  (void)splineKnots;
  (void)splineValues;
  (void)x;
  return 0.0;
}

double RadialProfiles::evalNiceQuadratic(const std::vector<double>& coeffs,
                                         double x) {
  // TODO(jons): implement `nice_quadratic`
  (void)coeffs;
  (void)x;
  return 0.0;
}

void RadialProfiles::evalRadialProfiles(bool haveToFlipTheta,
                                        VmecConstants& m_vmecconst) {
  // R_00 of initial boundary is ~ major radius
  // and used as normalization factor for mass profile
  const double r00 = id_.rbc[0];

  // scale profile of enclosed toroidal current to prescribed net toroidal
  // current,
  currv = MU_0 * id_.curtor;

  // only scale current profile if value at edge is "sufficiently non-zero"
  // outermost value of enclosed current profile -->
  const double edgeCurrent = evalCurrProfile(1.0);
  Itor = 0.0;
  // TODO(jons): eps*currv instead of eps*curtor?
  if (std::abs(edgeCurrent) > std::abs(DBL_EPSILON * id_.curtor)) {
    // FACTOR OF SIGNGS NEEDED HERE, SINCE MATCH IS MADE TO LINE INTEGRAL OF
    // BSUBU (IN GETIOTA) ~ SIGNGS * CURTOR
    Itor = signOfJacobian * currv / (2.0 * M_PI * edgeCurrent);
  }

  double local_rmsPhiP = 0.0;

  // half-grid
  for (int jH = r_.nsMinH; jH < r_.nsMaxH; ++jH) {
    const double halfGridPos = (jH + 0.5) / (fc_.ns - 1.0);

    // normalized toroidal flux array
    sqrtSH[jH - r_.nsMinH] = std::sqrt(halfGridPos);

    phipH[jH - r_.nsMinH] = maxToroidalFlux * torfluxDeriv(halfGridPos);
    chipH[jH - r_.nsMinH] = maxToroidalFlux * polfluxDeriv(halfGridPos);

    const double toroidalFlux = std::min(torflux(halfGridPos), 1.0);
    iotaH[jH - r_.nsMinH] = evalIotaProfile(toroidalFlux);
    currH[jH - r_.nsMinH] = evalCurrProfile(toroidalFlux);

    if (haveToFlipTheta) {
      chipH[jH - r_.nsMinH] *= -1.0;
      iotaH[jH - r_.nsMinH] *= -1.0;
    }

    // here effectively:
    // Itor == mu0/(2 * pi) * curtor * signgs
    // and then includes also 1/edgeCurrent to normalize profile of enclosed
    // toroidal current
    currH[jH - r_.nsMinH] *= Itor;

    // mass profile

    // effectively vpnorm == phipH[jH]
    const double vpnorm = maxToroidalFlux * torfluxDeriv(halfGridPos);
    const double massEvalPos = std::min(halfGridPos, id_.spres_ped);
    // if (massEvalPos > id_.spres_ped) {
    //     massEvalPos = id_.spres_ped;
    // }
    const double massEvalTorFlux = std::min(torflux(massEvalPos), 1.0);
    const double mass = evalMassProfile(massEvalTorFlux);
    massH[jH - r_.nsMinH] = mass * std::pow(std::abs(vpnorm) * r00, id_.gamma);

    // This must be done over UNIQUE half-grid points !!!
    // --> The standard partitioning has half-grid points between
    //     neighboring ranks that are handled by both ranks.
    if (jH < r_.nsMaxH - 1 || jH == fc_.ns - 2) {
      local_rmsPhiP += phipH[jH - r_.nsMinH] * phipH[jH - r_.nsMinH];
    }
  }

  // global accumulation of rmsPhiP
  m_vmecconst.rmsPhiP += local_rmsPhiP;

  // ------------------------------------------

  const double sqrtS1 = sqrt(1.0 / (fc_.ns - 1));

  // full-grid
  for (int jF1 = r_.nsMinF1; jF1 < r_.nsMaxF1; ++jF1) {
    const double fullGridPos = jF1 / (fc_.ns - 1.0);

    // normalized toroidal flux array
    sqrtSF[jF1 - r_.nsMinF1] = std::sqrt(fullGridPos);

    phipF[jF1 - r_.nsMinF1] = maxToroidalFlux * torfluxDeriv(fullGridPos);
    chipF[jF1 - r_.nsMinF1] = maxToroidalFlux * polfluxDeriv(fullGridPos);

    const double toroidalFlux = std::min(torflux(fullGridPos), 1.0);
    iotaF[jF1 - r_.nsMinF1] = evalIotaProfile(toroidalFlux);

    // TODO(jons): this is still weird
    // TODO(jons): The particular amount of 2*pDamp can sometimes be adjusted in
    // the range 0...1 to yield faster convergence.
    radialBlending[jF1 - r_.nsMinF1] = 2.0 * pDamp * (1.0 - fullGridPos);

    // even-m: no scaling
    scalxc[(jF1 - r_.nsMinF1) * 2 + kEvenParity] = 1.0;

    // odd-m: factor out 1/sqrt(s)
    // This is Eqn. (8c) in Hirshman, Schwenn & Nuehrenberg (1990).
    // The innermost full-grid point (== magnetic axis)
    // gets sqrt(s) of the innermost actual flux surface at j=1.
    // This is a constant extrapolation towards the axis.
    scalxc[(jF1 - r_.nsMinF1) * 2 + kOddParity] =
        1.0 / std::max(sqrtSF[jF1 - r_.nsMinF1], sqrtS1);
  }

  // avoid round-off errors
  if (r_.nsMaxF1 == fc_.ns) {
    sqrtSF[fc_.ns - 1 - r_.nsMinF1] = 1.0;
  }

  for (int jH = r_.nsMinH; jH < r_.nsMaxH; ++jH) {
    const int jFi = jH;
    const int jFo = jH + 1;
    sm[jH - r_.nsMinH] = sqrtSH[jH - r_.nsMinH] / sqrtSF[jFo - r_.nsMinF1];
    if (jH > 0) {
      sp[jH - r_.nsMinH] = sqrtSH[jH - r_.nsMinH] / sqrtSF[jFi - r_.nsMinF1];
    }
  }  // jH
  if (r_.nsMinH == 0) {
    // cannot divide by sqrtSF[0]==0, so extrapolate as constant from next point
    sp[0] = sm[0];
  }
}  // evalRadialProfiles

void RadialProfiles::AccumulateVolumeAveragedSpectralWidth() const {
  SpectralWidthContribution spectral_width_contribution = {.numerator = 0.0,
                                                           .denominator = 0.0};

  for (int jH = r_.nsMinH; jH < r_.nsMaxH; ++jH) {
    // This must be done over UNIQUE half-grid points !!!
    // --> The standard partitioning has half-grid points between
    //     neighboring ranks that are handled by both ranks.
    if (jH < r_.nsMaxH - 1 || jH == fc_.ns - 2) {
      const int jFi = jH;
      const int jFo = jH + 1;

      const double spectral_width_on_half_grid =
          (spectral_width[jFo - r_.nsMinF1] +
           spectral_width[jFi - r_.nsMinF1]) /
          2.0;
      spectral_width_contribution.numerator +=
          spectral_width_on_half_grid * dVdsH[jH - r_.nsMinH];

      spectral_width_contribution.denominator += dVdsH[jH - r_.nsMinH];
    }
  }  // jH

#ifdef _OPENMP
#pragma omp critical
#endif  // _OPENMP

  m_h_.RegisterSpectralWidthContribution(spectral_width_contribution);

#ifdef _OPENMP
#pragma omp barrier
#endif  // _OPENMP
}  // AccumulateVolumeAveragedSpectralWidth

}  // namespace vmecpp
