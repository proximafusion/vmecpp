// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT

// Validates the analytic adjoint ComputeQsHarmonicsVjp against the forward
// transform by the directional identity <u, J v> == <v, J^T u>, with J v formed
// by central finite differences of ComputeQsHarmonics. The QS harmonics map is
// linear in every field but |B| (and that one nonlinearity is closed form), so
// the analytic VJP is exact; this is the check that it is, with no autodiff.

#include <cmath>
#include <numeric>
#include <random>
#include <vector>

#include "gtest/gtest.h"
#include "vmecpp/vmec/ideal_mhd_model/qs_harmonics_kernel.h"

namespace vmecpp {
namespace {

struct QsSetup {
  QsHarmonicsConfig c;
  int npts, nh;
  std::vector<int> xm_nyq, xn_nyq;
  std::vector<double> mscale, nscale, cosmui, sinmui, cosnv, sinnv;
};

QsSetup MakeSetup() {
  const int nsH = 6, nZeta = 4, nThetaR = 5, nThetaEff = nThetaR;
  const int nfp = 1, mnyq = 3, nnyq = 2, nnyq2 = 2;
  QsSetup s;
  for (int m = 0; m <= mnyq; ++m)
    for (int n = -nnyq; n <= nnyq; ++n) {
      if (m == 0 && n < 0) continue;
      s.xm_nyq.push_back(m);
      s.xn_nyq.push_back(n * nfp);
    }
  const int mnmax_nyq = static_cast<int>(s.xm_nyq.size());
  s.npts = nsH * nZeta * nThetaEff;
  s.nh = mnmax_nyq * nsH;

  const double pi = std::acos(-1.0);
  s.mscale.assign(mnyq + 1, std::sqrt(2.0));
  s.mscale[0] = 1.0;
  s.nscale.assign(nnyq2 + 1, std::sqrt(2.0));
  s.nscale[0] = 1.0;
  s.cosmui.resize(nThetaR * (mnyq + 1));
  s.sinmui.resize(nThetaR * (mnyq + 1));
  for (int m = 0; m <= mnyq; ++m)
    for (int l = 0; l < nThetaR; ++l) {
      const double th = pi * l / (nThetaR - 1);
      s.cosmui[m * nThetaR + l] = std::cos(m * th) * (2.0 / (nThetaR - 1));
      s.sinmui[m * nThetaR + l] = std::sin(m * th) * (2.0 / (nThetaR - 1));
    }
  const int nnv = nnyq2 + 1;
  s.cosnv.resize(nZeta * nnv);
  s.sinnv.resize(nZeta * nnv);
  for (int k = 0; k < nZeta; ++k)
    for (int n = 0; n < nnv; ++n) {
      const double ze = 2.0 * pi * k / nZeta;
      s.cosnv[k * nnv + n] = std::cos(n * ze) / nZeta;
      s.sinnv[k * nnv + n] = std::sin(n * ze) / nZeta;
    }

  s.c.nsH = nsH;
  s.c.nZeta = nZeta;
  s.c.nThetaReduced = nThetaR;
  s.c.nThetaEff = nThetaEff;
  s.c.nnyq2 = nnyq2;
  s.c.mnyq = mnyq;
  s.c.nnyq = nnyq;
  s.c.mnmax_nyq = mnmax_nyq;
  s.c.tmult = 0.5;
  s.c.xm_nyq = s.xm_nyq.data();
  s.c.xn_nyq = s.xn_nyq.data();
  s.c.nfp = nfp;
  s.c.mscale = s.mscale.data();
  s.c.nscale = s.nscale.data();
  s.c.cosmui = s.cosmui.data();
  s.c.sinmui = s.sinmui.data();
  s.c.cosnv = s.cosnv.data();
  s.c.sinnv = s.sinnv.data();
  return s;
}

// Concatenated harmonics from the six field buffers, used to FD the forward
// map.
std::vector<double> Forward(const QsSetup& s, const std::vector<double>& gsqrt,
                            const std::vector<double>& tp,
                            const std::vector<double>& presH,
                            const std::vector<double>& bsupu,
                            const std::vector<double>& bsupv,
                            const std::vector<double>& bsubu,
                            const std::vector<double>& bsubv) {
  std::vector<double> h(6 * s.nh);
  ComputeQsHarmonics(gsqrt.data(), tp.data(), presH.data(), bsupu.data(),
                     bsupv.data(), bsubu.data(), bsubv.data(), h.data(),
                     h.data() + s.nh, h.data() + 2 * s.nh, h.data() + 3 * s.nh,
                     h.data() + 4 * s.nh, h.data() + 5 * s.nh, &s.c);
  return h;
}

TEST(QsHarmonicsVjp, AdjointIdentityMatchesFiniteDifference) {
  const QsSetup s = MakeSetup();
  std::mt19937 rng(7);
  std::uniform_real_distribution<double> d(0.5, 1.5), sgn(-1.0, 1.0);

  std::vector<double> gsqrt(s.npts), tp(s.npts), bsupu(s.npts), bsupv(s.npts),
      bsubu(s.npts), bsubv(s.npts), presH(s.c.nsH);
  for (int i = 0; i < s.npts; ++i) {
    gsqrt[i] = d(rng);
    tp[i] = 2.0 + d(rng);  // keep total_pressure - presH > 0 for modB
    bsupu[i] = d(rng);
    bsupv[i] = d(rng);
    bsubu[i] = d(rng);
    bsubv[i] = d(rng);
  }
  for (int j = 0; j < s.c.nsH; ++j) presH[j] = 0.2 + 0.1 * d(rng);

  // Random harmonic cotangent u and field direction v (6 fields, presH fixed).
  std::vector<double> u(6 * s.nh);
  for (double& x : u) x = sgn(rng);
  auto rv = [&]() {
    std::vector<double> v(s.npts);
    for (double& x : v) x = sgn(rng);
    return v;
  };
  const std::vector<double> vg = rv(), vtp = rv(), vpu = rv(), vpv = rv(),
                            vbu = rv(), vbv = rv();

  // J^T u via the analytic adjoint.
  std::vector<double> gb(s.npts), tpb(s.npts), pub(s.npts), pvb(s.npts),
      bub(s.npts), bvb(s.npts);
  // Cotangent blocks in output order: gmnc, bmnc, bsubumnc, bsubvmnc, bsupumnc,
  // bsupvmnc (blocks 0..5), matching the ComputeQsHarmonicsVjp argument order.
  ComputeQsHarmonicsVjp(u.data(), u.data() + s.nh, u.data() + 2 * s.nh,
                        u.data() + 3 * s.nh, u.data() + 4 * s.nh,
                        u.data() + 5 * s.nh, tp.data(), presH.data(), gb.data(),
                        tpb.data(), pub.data(), pvb.data(), bub.data(),
                        bvb.data(), &s.c);
  double vtu = 0.0;
  for (int i = 0; i < s.npts; ++i)
    vtu += vg[i] * gb[i] + vtp[i] * tpb[i] + vpu[i] * pub[i] + vpv[i] * pvb[i] +
           vbu[i] * bub[i] + vbv[i] * bvb[i];

  // <u, J v> with J v by central finite differences of the forward map.
  const double h = 1e-6;
  auto perturb = [&](double sign) {
    std::vector<double> g = gsqrt, t = tp, pu = bsupu, pv = bsupv, bu = bsubu,
                        bv = bsubv;
    for (int i = 0; i < s.npts; ++i) {
      g[i] += sign * h * vg[i];
      t[i] += sign * h * vtp[i];
      pu[i] += sign * h * vpu[i];
      pv[i] += sign * h * vpv[i];
      bu[i] += sign * h * vbu[i];
      bv[i] += sign * h * vbv[i];
    }
    return Forward(s, g, t, presH, pu, pv, bu, bv);
  };
  const std::vector<double> hp = perturb(+1.0), hm = perturb(-1.0);
  double utjv = 0.0;
  for (int i = 0; i < 6 * s.nh; ++i) utjv += u[i] * (hp[i] - hm[i]) / (2 * h);

  const double scale = std::fabs(utjv) + 1e-300;
  EXPECT_NEAR(vtu, utjv, 1e-7 * scale)
      << "adjoint identity <v,J^T u> vs <u,J v>";
}

}  // namespace
}  // namespace vmecpp
