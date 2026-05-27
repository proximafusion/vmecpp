// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
//
// Unit tests for the profile evaluators that were ported from educational_VMEC
// (profile_functions.f, functions.f, spline_akima.f, spline_akima_int.f,
// spline_cubic.f, spline_cubic_int.f). The checks are hand-derivable invariants
// of the underlying math: spline node reproduction, polynomial exactness of the
// splines, integrals of known profiles, the functions.f self-test value for
// two_power_gs, and the analytic endpoints/limits of the closed-form profiles.
#include "vmecpp/vmec/radial_profiles/radial_profiles.h"

#include <cmath>
#include <initializer_list>
#include <memory>

#include "gtest/gtest.h"
#include "vmecpp/common/flow_control/flow_control.h"
#include "vmecpp/common/sizes/sizes.h"
#include "vmecpp/common/vmec_indata/vmec_indata.h"
#include "vmecpp/vmec/handover_storage/handover_storage.h"
#include "vmecpp/vmec/radial_partitioning/radial_partitioning.h"

namespace vmecpp {

namespace {
Eigen::VectorXd Vec(std::initializer_list<double> xs) {
  Eigen::VectorXd v(static_cast<Eigen::Index>(xs.size()));
  Eigen::Index i = 0;
  for (double x : xs) v[i++] = x;
  return v;
}
constexpr bool kIntegrate = true;
constexpr bool kDirect = false;
}  // namespace

// The profile evaluators are pure functions of their arguments, but they are
// (non-static) members, so the fixture stands up a minimal RadialProfiles to
// call them through.
class RadialProfilesTest : public ::testing::Test {
 protected:
  RadialProfilesTest()
      : sizes_(indata_),
        flow_control_(/*lfreeb=*/false, /*delt=*/0.9,
                      /*num_grids=*/1),
        handover_(&sizes_) {
    partitioning_.adjustRadialPartitioning(/*num_threads=*/1, /*thread_id=*/0,
                                           /*ns=*/15, /*lfreeb=*/false);
    profiles_ = std::make_unique<RadialProfiles>(
        &partitioning_, &handover_, &indata_, &flow_control_,
        /*signOfJacobian=*/-1, /*pDamp=*/0.05);
  }

  VmecINDATA indata_;
  Sizes sizes_;
  FlowControl flow_control_;
  HandoverStorage handover_;
  RadialPartitioning partitioning_;
  std::unique_ptr<RadialProfiles> profiles_;
};

// ---- splines: the interpolant must pass through the data exactly -----------
TEST_F(RadialProfilesTest, SplineNodeReproduction) {
  const Eigen::VectorXd knots = Vec({0.0, 0.2, 0.45, 0.7, 1.0});
  const Eigen::VectorXd values = Vec({0.3, 0.7, 0.55, 0.9, 0.1});
  for (int i = 0; i < knots.size(); ++i) {
    EXPECT_NEAR(profiles_->evalAkima(knots, values, knots[i]), values[i], 1e-12)
        << "akima node " << i;
    EXPECT_NEAR(profiles_->evalCubic(knots, values, knots[i]), values[i], 1e-12)
        << "cubic node " << i;
  }
}

// ---- splines reproduce a linear profile exactly ----------------------------
TEST_F(RadialProfilesTest, SplineLinearExactness) {
  const Eigen::VectorXd knots = Vec({0.0, 0.2, 0.45, 0.7, 1.0});
  auto lin = [](double s) { return 2.0 + 3.0 * s; };
  const Eigen::VectorXd values =
      Vec({lin(0.0), lin(0.2), lin(0.45), lin(0.7), lin(1.0)});
  for (double x : {0.1, 0.33, 0.62, 0.88}) {
    EXPECT_NEAR(profiles_->evalAkima(knots, values, x), lin(x), 1e-9);
    EXPECT_NEAR(profiles_->evalCubic(knots, values, x), lin(x), 1e-9);
  }
}

// ---- integral of a linear profile over [0,1] is a + b/2 --------------------
TEST_F(RadialProfilesTest, SplineIntegralOfLinearProfile) {
  const double a = 2.0;
  const double b = 3.0;
  const Eigen::VectorXd knots = Vec({0.0, 0.2, 0.45, 0.7, 1.0});
  const Eigen::VectorXd values =
      Vec({a + b * 0.0, a + b * 0.2, a + b * 0.45, a + b * 0.7, a + b * 1.0});
  const double expected = a + b / 2.0;  // 3.5
  EXPECT_NEAR(profiles_->evalAkimaIntegrated(knots, values, 1.0), expected,
              1e-9);
  EXPECT_NEAR(profiles_->evalCubicIntegrated(knots, values, 1.0), expected,
              1e-9);
}

// ---- the integrated spline equals the numerical integral of the eval -------
TEST_F(RadialProfilesTest, IntegratedSplineMatchesNumericIntegral) {
  const Eigen::VectorXd knots = Vec({0.0, 0.2, 0.45, 0.7, 1.0});
  const Eigen::VectorXd values = Vec({0.3, 0.7, 0.55, 0.9, 0.1});
  const double upper = 0.62;
  const int kSteps = 100000;
  auto trapezoid = [&](auto eval) {
    double acc = 0.0;
    for (int i = 0; i < kSteps; ++i) {
      const double xa = upper * i / kSteps;
      const double xb = upper * (i + 1) / kSteps;
      acc +=
          0.5 * (eval(knots, values, xa) + eval(knots, values, xb)) * (xb - xa);
    }
    return acc;
  };
  EXPECT_NEAR(profiles_->evalAkimaIntegrated(knots, values, upper),
              trapezoid([&](auto& k, auto& v, double x) {
                return profiles_->evalAkima(k, v, x);
              }),
              1e-5);
  EXPECT_NEAR(profiles_->evalCubicIntegrated(knots, values, upper),
              trapezoid([&](auto& k, auto& v, double x) {
                return profiles_->evalCubic(k, v, x);
              }),
              1e-5);
}

// ---- splines with too few points return 0 (and warn) ----------------------
TEST_F(RadialProfilesTest, SplineTooFewPointsReturnsZero) {
  const Eigen::VectorXd knots = Vec({0.0, 0.5, 1.0});
  const Eigen::VectorXd values = Vec({0.1, 0.2, 0.3});
  EXPECT_EQ(profiles_->evalAkima(knots, values, 0.5), 0.0);
  EXPECT_EQ(profiles_->evalCubic(knots, values, 0.5), 0.0);
  EXPECT_EQ(profiles_->evalAkimaIntegrated(knots, values, 0.5), 0.0);
  EXPECT_EQ(profiles_->evalCubicIntegrated(knots, values, 0.5), 0.0);
}

// ---- corrected Akima right edge: the interpolant is reflection-symmetric ---
// educational_VMEC's spline_akima.f reuses the left-edge curvature on the right
// edge, so its interpolant depends on the orientation of the data. The
// corrected right-edge curvature restores reflection symmetry: interpolating
// the mirror image of the data at mirrored query points reproduces the original
// interpolant, including in the right-edge region the fix touches. This holds
// only with the corrected edge; the reused-cl behavior breaks it.
TEST_F(RadialProfilesTest, AkimaIsReflectionSymmetric) {
  const Eigen::VectorXd knots = Vec({0.0, 0.2, 0.45, 0.7, 1.0});
  const Eigen::VectorXd values = Vec({0.3, 0.7, 0.55, 0.9, 0.1});
  // mirror about the knot span [0, 1]: reverse the nodes and map x -> 1 - x.
  const int n = static_cast<int>(knots.size());
  Eigen::VectorXd mirrored_knots(n);
  Eigen::VectorXd mirrored_values(n);
  for (int i = 0; i < n; ++i) {
    mirrored_knots[i] = 1.0 - knots[n - 1 - i];
    mirrored_values[i] = values[n - 1 - i];
  }
  for (double x : {0.05, 0.18, 0.5, 0.83, 0.96}) {
    EXPECT_NEAR(profiles_->evalAkima(knots, values, x),
                profiles_->evalAkima(mirrored_knots, mirrored_values, 1.0 - x),
                1e-12)
        << "akima reflection mismatch at x=" << x;
  }
}

// ---- gauss_trunc -----------------------------------------------------------
TEST_F(RadialProfilesTest, GaussTruncPressureNormalizedToAxis) {
  // pressure form is normalized so that p(0) = am(0).
  const Eigen::VectorXd c = Vec({0.5, 0.3});
  EXPECT_NEAR(profiles_->evalGaussTrunc(c, 0.0, kDirect), 0.5, 1e-9);
}

TEST_F(RadialProfilesTest, GaussTruncCurrentIsIntegralOfIprime) {
  const Eigen::VectorXd c = Vec({0.7, 0.4});
  const double a0 = c[0];
  const double a1 = c[1];
  const double edge = std::exp(-std::pow(1.0 / a1, 2));
  auto iprime = [&](double s) {
    return a0 * (std::exp(-std::pow(s / a1, 2)) - edge);
  };
  const double upper = 0.7;
  const int kSteps = 100000;
  double expected = 0.0;
  for (int i = 0; i < kSteps; ++i) {
    expected +=
        0.5 * (iprime(upper * i / kSteps) + iprime(upper * (i + 1) / kSteps)) *
        (upper / kSteps);
  }
  EXPECT_NEAR(profiles_->evalGaussTrunc(c, upper, kIntegrate), expected, 1e-6);
}

// ---- two_power_gs ----------------------------------------------------------
TEST_F(RadialProfilesTest, TwoPowerGsFunctionsDotFSelfTest) {
  // functions.f: two_power_gs(0.8, {1,1,0,1,0.8,0.1}) == 2.
  const Eigen::VectorXd c = Vec({1.0, 1.0, 0.0, 1.0, 0.8, 0.1});
  EXPECT_NEAR(profiles_->evalTwoPowerGs(c, 0.8, kDirect), 2.0, 1e-12);
}

TEST_F(RadialProfilesTest, TwoPowerGsReducesToTwoPower) {
  // With no Gaussian terms, two_power_gs == two_power = c0*(1-x^c1)^c2.
  const Eigen::VectorXd c = Vec({2.0, 3.0, 1.5});
  const double x = 0.5;
  const double expected = 2.0 * std::pow(1.0 - std::pow(x, 3.0), 1.5);
  EXPECT_NEAR(profiles_->evalTwoPowerGs(c, x, kDirect), expected, 1e-12);
}

// ---- nice_quadratic --------------------------------------------------------
TEST_F(RadialProfilesTest, NiceQuadratic) {
  const Eigen::VectorXd c = Vec({0.5, 1.5, 0.2});
  EXPECT_NEAR(profiles_->evalNiceQuadratic(c, 0.0), 0.5, 1e-12);  // a0
  EXPECT_NEAR(profiles_->evalNiceQuadratic(c, 1.0), 1.5, 1e-12);  // a1
  // a0*0.5 + a1*0.5 + 4*a2*0.25 = 0.25 + 0.75 + 0.2 = 1.2
  EXPECT_NEAR(profiles_->evalNiceQuadratic(c, 0.5), 1.2, 1e-12);
}

// ---- rational --------------------------------------------------------------
TEST_F(RadialProfilesTest, Rational) {
  // numerator = 1 + 2x (c0,c1), denominator = 3 (c10) -> (1 + 2x)/3.
  Eigen::VectorXd c = Eigen::VectorXd::Zero(12);
  c[0] = 1.0;
  c[1] = 2.0;
  c[10] = 3.0;
  EXPECT_NEAR(profiles_->evalRational(c, 0.5), (1.0 + 2.0 * 0.5) / 3.0, 1e-12);
  EXPECT_NEAR(profiles_->evalRational(c, 0.0), 1.0 / 3.0, 1e-12);
}

// ---- numerical agreement with compiled educational_VMEC Fortran ------------
// The golden values below are produced by fref/fref_driver.f90, which links the
// real educational_VMEC routines (spline_cubic.f, spline_cubic_int.f,
// functions.f, profile_functions.f) and prints their output at exactly the
// inputs used here. See that driver (plus fref/stubs.f90, the minimal module
// shim) to regenerate. The pure-arithmetic cubic spline agrees to ~1e-13; the
// closed-form profiles use exp/pow/tanh whose last ulp differs between the
// Windows libm (golden) and the Linux libm (this test), so the tolerance is
// held at 1e-11, still many orders of magnitude tighter than any indexing or
// formula error would produce.
//
// The Akima spline is intentionally absent here: its right-edge curvature is
// corrected relative to spline_akima.f, so it no longer reproduces that
// routine's edge values by design. It is validated by SplineNodeReproduction,
// SplineLinearExactness, IntegratedSplineMatchesNumericIntegral, and
// AkimaIsReflectionSymmetric instead.
TEST_F(RadialProfilesTest, MatchesFortranReference) {
  constexpr double kTol = 1e-11;
  struct Pt {
    double x;
    double golden;
  };

  // splines through non-uniform, non-monotonic data
  const Eigen::VectorXd knots = Vec({0.0, 0.2, 0.45, 0.7, 1.0});
  const Eigen::VectorXd values = Vec({0.3, 0.7, 0.55, 0.9, 0.1});

  for (const Pt& p :
       {Pt{0.10, 5.71628096527232010e-01}, Pt{0.33, 6.14644817745175098e-01},
        Pt{0.50, 5.98234681104998223e-01}, Pt{0.62, 8.11570261229004775e-01},
        Pt{0.88, 6.01700812156143461e-01}}) {
    EXPECT_NEAR(profiles_->evalCubic(knots, values, p.x), p.golden, kTol)
        << "cubic x=" << p.x;
  }
  for (const Pt& p :
       {Pt{0.33, 1.96931323531569308e-01}, Pt{0.62, 3.77727303980826490e-01},
        Pt{1.00, 6.35921583752995900e-01}}) {
    EXPECT_NEAR(profiles_->evalCubicIntegrated(knots, values, p.x), p.golden,
                kTol)
        << "cubic_int x=" << p.x;
  }

  // gauss_trunc, ac/am = {0.7, 0.4}
  const Eigen::VectorXd gauss = Vec({0.7, 0.4});
  for (const Pt& p :
       {Pt{0.30, 1.76063280436117658e-01}, Pt{0.60, 2.38921959477756629e-01},
        Pt{0.90, 2.46564389349754637e-01}}) {
    EXPECT_NEAR(profiles_->evalGaussTrunc(gauss, p.x, kIntegrate), p.golden,
                kTol)
        << "gauss_trunc current x=" << p.x;
  }
  for (const Pt& p :
       {Pt{0.00, 6.99999999999999956e-01}, Pt{0.30, 3.98265492683955280e-01},
        Pt{0.60, 7.25682289356532340e-02}, Pt{0.90, 3.08543920275165763e-03}}) {
    EXPECT_NEAR(profiles_->evalGaussTrunc(gauss, p.x, kDirect), p.golden, kTol)
        << "gauss_trunc pressure x=" << p.x;
  }

  // two_power_gs with two active Gaussian groups
  const Eigen::VectorXd tpg =
      Vec({1.2, 2.0, 1.5, 0.5, 0.3, 0.2, 0.4, 0.7, 0.15});
  for (const Pt& p :
       {Pt{0.20, 1.56825752881167446e+00}, Pt{0.50, 9.75482824018030881e-01},
        Pt{0.85, 2.01277370289491664e-01}}) {
    EXPECT_NEAR(profiles_->evalTwoPowerGs(tpg, p.x, kDirect), p.golden, kTol)
        << "two_power_gs x=" << p.x;
  }

  // pedestal current: backbone {0.4, 0.2} plus three active tanh terms
  Eigen::VectorXd pedCur = Eigen::VectorXd::Zero(21);
  pedCur[0] = 0.4;
  pedCur[1] = 0.2;
  pedCur[8] = 0.3;
  pedCur[10] = 0.5;
  pedCur[11] = 0.1;
  pedCur[13] = 0.15;
  pedCur[15] = 0.6;
  pedCur[16] = 0.05;
  pedCur[17] = 0.1;
  pedCur[19] = 0.7;
  pedCur[20] = 0.08;
  for (const Pt& p :
       {Pt{0.30, 3.90279651281347062e-01}, Pt{0.55, 5.90587303540200281e-01},
        Pt{0.80, 1.16872771235070450e+00}}) {
    EXPECT_NEAR(profiles_->evalPedestal(pedCur, p.x, kIntegrate), p.golden,
                kTol)
        << "pedestal current x=" << p.x;
  }

  // pedestal pressure: backbone {0.1, -0.05, 0.02} plus one active tanh term
  Eigen::VectorXd pedPrs = Eigen::VectorXd::Zero(21);
  pedPrs[0] = 0.1;
  pedPrs[1] = -0.05;
  pedPrs[2] = 0.02;
  pedPrs[17] = 0.08;
  pedPrs[18] = 0.5;
  pedPrs[19] = 0.1;
  for (const Pt& p :
       {Pt{0.30, 9.71283377363868078e-02}, Pt{0.50, 8.00201912648781377e-02},
        Pt{0.70, 7.48001132593014983e-02}}) {
    EXPECT_NEAR(profiles_->evalPedestal(pedPrs, p.x, kDirect), p.golden, kTol)
        << "pedestal pressure x=" << p.x;
  }

  // rational: numerator 1 + 0.5x + 0.3x^2 over denominator 2 + 0.4x
  Eigen::VectorXd rat = Eigen::VectorXd::Zero(12);
  rat[0] = 1.0;
  rat[1] = 0.5;
  rat[2] = 0.3;
  rat[10] = 2.0;
  rat[11] = 0.4;
  for (const Pt& p :
       {Pt{0.00, 5.00000000000000000e-01}, Pt{0.40, 5.77777777777777724e-01},
        Pt{0.90, 7.17372881355932246e-01}}) {
    EXPECT_NEAR(profiles_->evalRational(rat, p.x), p.golden, kTol)
        << "rational x=" << p.x;
  }

  // nice_quadratic, ai = {0.5, 1.5, 0.2}
  const Eigen::VectorXd nq = Vec({0.5, 1.5, 0.2});
  for (const Pt& p :
       {Pt{0.20, 8.28000000000000069e-01}, Pt{0.50, 1.19999999999999996e+00},
        Pt{0.80, 1.42800000000000038e+00}}) {
    EXPECT_NEAR(profiles_->evalNiceQuadratic(nq, p.x), p.golden, kTol)
        << "nice_quadratic x=" << p.x;
  }
}

// ---- pedestal: polynomial backbone with the tanh terms disabled ------------
TEST_F(RadialProfilesTest, PedestalPressureBackbone) {
  // am(0..15) polynomial; am(19) <= 0 disables the tanh pedestal term.
  Eigen::VectorXd c = Eigen::VectorXd::Zero(21);
  c[0] = 0.1;
  c[1] = -0.05;
  c[2] = 0.02;
  c[19] = -1.0;
  auto poly = [](double x) { return 0.1 - 0.05 * x + 0.02 * x * x; };
  EXPECT_NEAR(profiles_->evalPedestal(c, 0.6, kDirect), poly(0.6), 1e-12);
}

TEST_F(RadialProfilesTest, PedestalCurrentBackbone) {
  // ac(0..7) integrated polynomial; ac(11) <= 0 and ac(13)=ac(17)=0 leave only
  // the enclosed-current backbone I(s) = 0.4 s + 0.1 s^2 from I'(s) = 0.4 + 0.2
  // s.
  Eigen::VectorXd c = Eigen::VectorXd::Zero(21);
  c[0] = 0.4;
  c[1] = 0.2;
  c[11] = -1.0;
  auto enclosed = [](double x) { return 0.4 * x + 0.1 * x * x; };
  EXPECT_NEAR(profiles_->evalPedestal(c, 0.6, kIntegrate), enclosed(0.6),
              1e-12);
}

// ---- profiles reach the evaluators through the production dispatch ---------
// MatchesFortranReference and the other cases call the evaluators directly. The
// two tests below instead drive the wiring the solver uses: a VmecINDATA
// profile-type string resolves through findParameterization, and
// evalMass/Iota/CurrProfile forward the coefficient and spline-aux arrays into
// evalProfileFunction, which dispatches to the matching evaluator. No leaf test
// can see this seam, because the leaves are handed their knots and coefficients
// directly. The spline case in particular confirms that am_aux_s / am_aux_f are
// passed whole as the spline knots and values (VMEC++ stores them trimmed to
// the provided knot count, unlike the 101-padded Fortran arrays) and that the
// current profile takes the integrated-spline branch.
TEST_F(RadialProfilesTest, SplineProfilesRouteThroughDispatch) {
  indata_.bloat = 1.0;
  indata_.pres_scale = 1.0;
  indata_.pmass_type = "cubic_spline";
  indata_.am_aux_s = Vec({0.0, 0.25, 0.5, 0.75, 1.0});
  indata_.am_aux_f = Vec({1.2, 1.0, 0.7, 0.35, 0.05});
  indata_.piota_type = "cubic_spline";
  indata_.ai_aux_s = Vec({0.0, 0.25, 0.5, 0.75, 1.0});
  indata_.ai_aux_f = Vec({0.9, 0.85, 0.7, 0.5, 0.3});
  indata_.pcurr_type = "akima_spline_ip";
  indata_.ac_aux_s = Vec({0.0, 0.25, 0.5, 0.75, 1.0});
  indata_.ac_aux_f = Vec({0.0, 0.3, 0.5, 0.4, 0.1});
  profiles_->setupInputProfiles();

  for (double x : {0.08, 0.3, 0.55, 0.82, 0.97}) {
    const double normX = std::min(std::abs(x * indata_.bloat), 1.0);
    EXPECT_NEAR(
        profiles_->evalMassProfile(x),
        profiles_->pressureScalingFactor *
            profiles_->evalCubic(indata_.am_aux_s, indata_.am_aux_f, normX),
        1e-12)
        << "cubic_spline pressure x=" << x;
    EXPECT_NEAR(profiles_->evalIotaProfile(x),
                profiles_->evalCubic(indata_.ai_aux_s, indata_.ai_aux_f, x),
                1e-12)
        << "cubic_spline iota x=" << x;
    EXPECT_NEAR(profiles_->evalCurrProfile(x),
                profiles_->evalAkimaIntegrated(indata_.ac_aux_s,
                                               indata_.ac_aux_f, normX),
                1e-12)
        << "akima_spline_ip current x=" << x;
  }
}

TEST_F(RadialProfilesTest, AnalyticProfilesRouteThroughDispatch) {
  indata_.bloat = 1.0;
  indata_.pres_scale = 1.0;
  indata_.pmass_type = "gauss_trunc";
  indata_.am = Vec({0.7, 0.4});
  indata_.piota_type = "nice_quadratic";
  indata_.ai = Vec({0.5, 1.5, 0.2});
  indata_.pcurr_type = "gauss_trunc";
  indata_.ac = Vec({0.7, 0.4});
  profiles_->setupInputProfiles();

  for (double x : {0.1, 0.4, 0.7, 0.95}) {
    const double normX = std::min(std::abs(x * indata_.bloat), 1.0);
    EXPECT_NEAR(profiles_->evalMassProfile(x),
                profiles_->pressureScalingFactor *
                    profiles_->evalGaussTrunc(indata_.am, normX, kDirect),
                1e-12)
        << "gauss_trunc pressure x=" << x;
    EXPECT_NEAR(profiles_->evalIotaProfile(x),
                profiles_->evalNiceQuadratic(indata_.ai, x), 1e-12)
        << "nice_quadratic iota x=" << x;
    EXPECT_NEAR(profiles_->evalCurrProfile(x),
                profiles_->evalGaussTrunc(indata_.ac, normX, kIntegrate), 1e-12)
        << "gauss_trunc current x=" << x;
  }
}

}  // namespace vmecpp
