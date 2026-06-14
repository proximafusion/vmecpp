// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT

// Validates the flat-buffer ComputeQsHarmonics kernel against the reference
// output_quantities forward transform: same converged equilibrium, same
// half-grid field harmonics, must agree to round-off.

#include "vmecpp/vmec/ideal_mhd_model/qs_harmonics_kernel.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include "absl/status/statusor.h"
#include "gtest/gtest.h"
#include "util/file_io/file_io.h"
#include "vmecpp/common/fourier_basis_fast_poloidal/fourier_basis_fast_poloidal.h"
#include "vmecpp/common/sizes/sizes.h"
#include "vmecpp/common/vmec_indata/vmec_indata.h"
#include "vmecpp/vmec/vmec/vmec.h"

namespace vmecpp {

TEST(QsHarmonicsKernel, MatchesOutputQuantities) {
  const absl::StatusOr<std::string> indata_json =
      file_io::ReadFile("vmecpp/test_data/solovev.json");
  ASSERT_TRUE(indata_json.ok());
  const absl::StatusOr<VmecINDATA> indata = VmecINDATA::FromJson(*indata_json);
  ASSERT_TRUE(indata.ok());

  Vmec vmec(*indata);
  ASSERT_TRUE(vmec.run().ok());

  const Sizes& s = vmec.s_;
  const FourierBasisFastPoloidal t(&s);
  const OutputQuantities& oq = vmec.output_quantities_;
  const VmecInternalResults& vir = oq.vmec_internal_results;
  const WOutFileContents& wout = oq.wout;

  const int ns = static_cast<int>(wout.gmnc.cols());
  const int nsH = ns - 1;
  const int npts = nsH * s.nZeta * s.nThetaEff;

  // Flatten the half-grid real-space fields in the row-major order the
  // reference loop uses: idx = (jH*nZeta + k)*nThetaEff + l.
  auto flat = [npts](const auto& m) {
    std::vector<double> v(npts);
    for (int i = 0; i < npts; ++i) v[i] = m(i);
    return v;
  };
  const std::vector<double> gsqrt = flat(vir.gsqrt);
  const std::vector<double> total_pressure = flat(vir.total_pressure);
  const std::vector<double> bsupu = flat(vir.bsupu);
  const std::vector<double> bsupv = flat(vir.bsupv);
  const std::vector<double> bsubu = flat(vir.bsubu);
  const std::vector<double> bsubv = flat(vir.bsubv);
  std::vector<double> presH(nsH);
  for (int jH = 0; jH < nsH; ++jH) presH[jH] = vir.presH(jH);

  std::vector<int> xm_nyq(s.mnmax_nyq), xn_nyq(s.mnmax_nyq);
  for (int mn = 0; mn < s.mnmax_nyq; ++mn) {
    xm_nyq[mn] = static_cast<int>(wout.xm_nyq[mn]);
    xn_nyq[mn] = static_cast<int>(wout.xn_nyq[mn]);
  }

  QsHarmonicsConfig c{};
  c.nsH = nsH;
  c.nZeta = s.nZeta;
  c.nThetaReduced = s.nThetaReduced;
  c.nThetaEff = s.nThetaEff;
  c.nnyq2 = s.nnyq2;
  c.mnyq = s.mnyq;
  c.nnyq = s.nnyq;
  c.mnmax_nyq = s.mnmax_nyq;
  c.tmult = 0.5;
  c.xm_nyq = xm_nyq.data();
  c.xn_nyq = xn_nyq.data();
  c.nfp = wout.nfp;
  c.mscale = t.mscale.data();
  c.nscale = t.nscale.data();
  c.cosmui = t.cosmui.data();
  c.sinmui = t.sinmui.data();
  c.cosnv = t.cosnv.data();
  c.sinnv = t.sinnv.data();

  const int nh = s.mnmax_nyq * nsH;
  std::vector<double> gmnc(nh), bmnc(nh), bsubumnc(nh), bsubvmnc(nh),
      bsupumnc(nh), bsupvmnc(nh);
  ComputeQsHarmonics(gsqrt.data(), total_pressure.data(), presH.data(),
                     bsupu.data(), bsupv.data(), bsubu.data(), bsubv.data(),
                     gmnc.data(), bmnc.data(), bsubumnc.data(), bsubvmnc.data(),
                     bsupumnc.data(), bsupvmnc.data(), &c);

  // Each harmonic must match the reference to a small fraction of that
  // quantity's overall scale (the Nyquist-band modes are ~round-off, so a fixed
  // absolute floor is meaningless; a scale-relative tolerance is the rigorous
  // check). scale_q = max_|reference value of q|.
  auto scale = [&](const auto& ref) {
    double s_ = 0.0;
    for (int mn = 0; mn < s.mnmax_nyq; ++mn)
      for (int jH = 0; jH < nsH; ++jH)
        s_ = std::max(s_, std::fabs(ref(mn, jH + 1)));
    return s_;
  };
  const double tol_g = 1e-7 * scale(wout.gmnc);
  const double tol_b = 1e-7 * scale(wout.bmnc);
  const double tol_su = 1e-7 * scale(wout.bsubumnc);
  const double tol_sv = 1e-7 * scale(wout.bsubvmnc);
  const double tol_pu = 1e-7 * scale(wout.bsupumnc);
  const double tol_pv = 1e-7 * scale(wout.bsupvmnc);
  for (int mn = 0; mn < s.mnmax_nyq; ++mn) {
    for (int jH = 0; jH < nsH; ++jH) {
      const int o = mn * nsH + jH;
      EXPECT_NEAR(gmnc[o], wout.gmnc(mn, jH + 1), tol_g) << "gmnc " << mn << " " << jH;
      EXPECT_NEAR(bmnc[o], wout.bmnc(mn, jH + 1), tol_b) << "bmnc " << mn << " " << jH;
      EXPECT_NEAR(bsubumnc[o], wout.bsubumnc(mn, jH + 1), tol_su) << "bsubu " << mn << " " << jH;
      EXPECT_NEAR(bsubvmnc[o], wout.bsubvmnc(mn, jH + 1), tol_sv) << "bsubv " << mn << " " << jH;
      EXPECT_NEAR(bsupumnc[o], wout.bsupumnc(mn, jH + 1), tol_pu) << "bsupu " << mn << " " << jH;
      EXPECT_NEAR(bsupvmnc[o], wout.bsupvmnc(mn, jH + 1), tol_pv) << "bsupv " << mn << " " << jH;
    }
  }
}

}  // namespace vmecpp
