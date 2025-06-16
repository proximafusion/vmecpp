// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#include "vmecpp/free_boundary/nestor/nestor.h"

#include "absl/algorithm/container.h"

namespace vmecpp {

Nestor::Nestor(const Sizes* s, const TangentialPartitioning* tp,
               const MGridProvider* mgrid, std::span<double> matrixShare,
               std::span<double> bvecShare, std::span<double> bSqVacShare,
               std::span<int> iPiv, std::span<double> vacuum_b_r_share,
               std::span<double> vacuum_b_phi_share,
               std::span<double> vacuum_b_z_share)
    : FreeBoundaryBase(s, tp, mgrid, bSqVacShare, vacuum_b_r_share,
                       vacuum_b_phi_share, vacuum_b_z_share),
      nf(s_.ntor),
      mf(s_.mpol + 1),
      si_(s, &fb_, tp, &sg_, nf, mf),
      ri_(s, tp, &sg_),
      ls_(s, &fb_, tp, nf, mf, matrixShare, iPiv, bvecShare),
      bvecShare(bvecShare) {
  int numLocal = tp_.ztMax - tp_.ztMin;

  potU.resize(numLocal);
  potV.resize(numLocal);

  bSubU.resize(numLocal);
  bSubV.resize(numLocal);
}

bool Nestor::update(
    const std::span<const double> rCC, const std::span<const double> rSS,
    const std::span<const double> rSC, const std::span<const double> rCS,
    const std::span<const double> zSC, const std::span<const double> zCS,
    const std::span<const double> zCC, const std::span<const double> zSS,
    int signOfJacobian, const std::span<const double> rAxis,
    const std::span<const double> zAxis, double* bSubUVac, double* bSubVVac,
    double netToroidalCurrent, int ivacskip,
    const VmecCheckpoint& vmec_checkpoint, bool at_checkpoint_iteration) {
  if (vmec_checkpoint == VmecCheckpoint::VAC1_VACUUM &&
      at_checkpoint_iteration) {
    return true;
  }

  bool fullUpdate = (ivacskip == 0);

  sg_.update(rCC, rSS, rSC, rCS, zSC, zCS, zCC, zSS, signOfJacobian,
             fullUpdate);
  if (vmec_checkpoint == VmecCheckpoint::VAC1_SURFACE &&
      at_checkpoint_iteration) {
    return true;
  }

  ef_.update(rAxis, zAxis, netToroidalCurrent);
  if (vmec_checkpoint == VmecCheckpoint::VAC1_BEXTERN &&
      at_checkpoint_iteration) {
    return true;
  }

  si_.update(ef_.bDotN, fullUpdate);
  if (vmec_checkpoint == VmecCheckpoint::VAC1_ANALYT &&
      at_checkpoint_iteration) {
    return true;
  }

  if (fullUpdate) {
    ri_.update(ef_.bDotN);
    if (vmec_checkpoint == VmecCheckpoint::VAC1_GREENF &&
        at_checkpoint_iteration) {
      return true;
    }

    ls_.TransformGreensFunctionDerivative(ri_.greenp);
    if (vmec_checkpoint == VmecCheckpoint::VAC1_FOURP &&
        at_checkpoint_iteration) {
      return true;
    }

    ls_.SymmetriseSourceTerm(ri_.gstore);
    if (vmec_checkpoint == VmecCheckpoint::VAC1_FOURI_SYMM &&
        at_checkpoint_iteration) {
      return true;
    }

    ls_.AccumulateFullGrpmn(si_.grpmn_sin);
    ls_.PerformToroidalFourierTransforms();
    if (vmec_checkpoint == VmecCheckpoint::VAC1_FOURI_KV_DFT &&
        at_checkpoint_iteration) {
      return true;
    }

    ls_.PerformPoloidalFourierTransforms();
    ls_.BuildMatrix();
    if (vmec_checkpoint == VmecCheckpoint::VAC1_FOURI_KU_DFT &&
        at_checkpoint_iteration) {
      return true;
    }

#ifdef _OPENMP
#pragma omp single
#endif  // _OPENMP
    ls_.DecomposeMatrix();
#ifdef _OPENMP
#pragma omp barrier
#endif  // _OPENMP
  }  // fullUpdate

  // virtual checkpoint, if maximum_iterations before next full Nestor update
  if ((vmec_checkpoint == VmecCheckpoint::VAC1_GREENF ||
       vmec_checkpoint == VmecCheckpoint::VAC1_FOURP ||
       vmec_checkpoint == VmecCheckpoint::VAC1_FOURI_SYMM ||
       vmec_checkpoint == VmecCheckpoint::VAC1_FOURI_KV_DFT ||
       vmec_checkpoint == VmecCheckpoint::VAC1_FOURI_KU_DFT ||
       vmec_checkpoint == VmecCheckpoint::UPDATE_TCON) &&
      at_checkpoint_iteration) {
    return true;
  }

  ls_.SolveForPotential(si_.bvec_sin);

  if (vmec_checkpoint == VmecCheckpoint::VAC1_SOLVER &&
      at_checkpoint_iteration) {
    return true;
  }

  // thread-local tangential grid point range
  const int mnpd = (mf + 1) * (2 * nf + 1);
  const int numLocal = tp_.ztMax - tp_.ztMin;

  absl::c_fill_n(potU, numLocal, 0);
  absl::c_fill_n(potV, numLocal, 0);

  // inv-DFT with tangential derivatives
  for (int kl = tp_.ztMin; kl < tp_.ztMax; ++kl) {
    const int l = kl / s_.nZeta;
    const int k = kl % s_.nZeta;
    for (int mn = 0; mn < mnpd; ++mn) {
      const int n = mn / (mf + 1) - nf;  // -nf:nf
      const int m = mn % (mf + 1);

      const int abs_n = std::abs(n);
      const int sign_n = signum(n);

      const int idx_lm = l * (s_.mnyq2 + 1) + m;
      const double cosmu = fb_.cosmu[idx_lm] / fb_.mscale[m];
      const double sinmu = fb_.sinmu[idx_lm] / fb_.mscale[m];

      const int idx_nk = abs_n * s_.nZeta + k;
      const double cosnv = fb_.cosnv[idx_nk] / fb_.nscale[abs_n];
      const double sinnv = fb_.sinnv[idx_nk] / fb_.nscale[abs_n];

      const double cos_mu_nv = cosmu * cosnv + sign_n * sinmu * sinnv;

      potU[kl - tp_.ztMin] += bvecShare[mn] * m * cos_mu_nv;
      potV[kl - tp_.ztMin] += bvecShare[mn] * (-n * s_.nfp) * cos_mu_nv;
    }  // mn
  }  // kl

  // compute net covariant magnetic field components on surface
  double local_bSubUVac = 0.0;
  double local_bSubVVac = 0.0;
  for (int kl = tp_.ztMin; kl < tp_.ztMax; ++kl) {
    bSubU[kl - tp_.ztMin] = potU[kl - tp_.ztMin] + ef_.bSubU[kl - tp_.ztMin];
    bSubV[kl - tp_.ztMin] = potV[kl - tp_.ztMin] + ef_.bSubV[kl - tp_.ztMin];

    int l = kl / s_.nZeta;
    local_bSubUVac += bSubU[kl - tp_.ztMin] * s_.wInt[l];
    local_bSubVVac += bSubV[kl - tp_.ztMin] * s_.wInt[l];
  }
  local_bSubUVac *= signOfJacobian * 2.0 * M_PI;

#ifdef _OPENMP
#pragma omp single
#endif  // _OPENMP
  {
    *bSubUVac = 0.0;
    *bSubVVac = 0.0;
  }
#ifdef _OPENMP
#pragma omp barrier
#endif  // _OPENMP

#ifdef _OPENMP
#pragma omp critical
#endif  // _OPENMP
  {
    *bSubUVac += local_bSubUVac;
    *bSubVVac += local_bSubVVac;
  }
#ifdef _OPENMP
#pragma omp barrier
#endif  // _OPENMP

  // compute magnetic pressure from co- and contravariant B_vac components
  for (int kl = tp_.ztMin; kl < tp_.ztMax; ++kl) {
    // metric elements, without the Nestors-specific normalizations
    double guu = sg_.guu[kl - tp_.ztMin];
    double guv = sg_.guv[kl - tp_.ztMin] * s_.nfp * 0.5;
    double gvv = sg_.gvv[kl - tp_.ztMin] * s_.nfp * s_.nfp;

    double det = guu * gvv - guv * guv;

    // compute contravariant magnetic field components
    // by inverting the inverse transform (as used in VMEC to go from bContra to
    // bCov)
    double bSupU =
        (gvv * bSubU[kl - tp_.ztMin] - guv * bSubV[kl - tp_.ztMin]) / det;
    double bSupV =
        (-guv * bSubU[kl - tp_.ztMin] + guu * bSubV[kl - tp_.ztMin]) / det;

    // magnetic pressure from vacuum: |B|^2/2
    bSqVacShare[kl] =
        (bSubU[kl - tp_.ztMin] * bSupU + bSubV[kl - tp_.ztMin] * bSupV) * 0.5;

    // cylindrical components of vacuum magnetic field
    vacuum_b_r_share_[kl] =
        sg_.rub[kl - tp_.ztMin] * bSupU + sg_.rvb[kl - tp_.ztMin] * bSupV;
    vacuum_b_phi_share_[kl] = sg_.r1b[kl] * bSupV;
    vacuum_b_z_share_[kl] =
        sg_.zub[kl - tp_.ztMin] * bSupU + sg_.zvb[kl - tp_.ztMin] * bSupV;
  }  // kl

  // ... done ...

  if (vmec_checkpoint == VmecCheckpoint::VAC1_BSQVAC &&
      at_checkpoint_iteration) {
    return true;
  }

#ifdef _OPENMP
#pragma omp barrier
#endif  // _OPENMP

  // TODO(jons): could move bSubUVac, bSubVVac collection here to spare on
  // barrier

  return false;
}

const SingularIntegrals& Nestor::GetSingularIntegrals() const {
  return si_;
}  // GetSingularIntegrals

const RegularizedIntegrals& Nestor::GetRegularizedIntegrals() const {
  return ri_;
}  // GetRegularizedIntegrals

const LaplaceSolver& Nestor::GetLaplaceSolver() const {
  return ls_;
}  // GetLaplaceSolver

}  // namespace vmecpp
