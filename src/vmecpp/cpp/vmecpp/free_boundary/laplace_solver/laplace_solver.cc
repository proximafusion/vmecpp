// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#include "vmecpp/free_boundary/laplace_solver/laplace_solver.h"

#include <iostream>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"

///  LU factorization of a general M-by-N matrix A
extern "C" void dgetrf_(int *m, int *n, double *a, int *lda, int *ipiv,
                        int *info);
///  Solves a system of linear equations using the LU factorization.
extern "C" void dgetrs_(char *transpose, int *num_rows, int *num_columns,
                        double *matrix, int *leading_dim, int *pivot, double *y,
                        int *y_leading_dim, int *info);

namespace vmecpp {

LaplaceSolver::LaplaceSolver(const Sizes *s, const FourierBasisFastToroidal *fb,
                             const TangentialPartitioning *tp, int nf, int mf,
                             std::span<double> matrixShare, std::span<int> iPiv,
                             std::span<double> bvecShare)
    : s_(*s),
      fb_(*fb),
      tp_(*tp),
      nf(nf),
      mf(mf),
      matrixShare(matrixShare),
      iPiv(iPiv),
      bvecShare(bvecShare) {
  // thread-local tangential grid point range
  numLocal = tp_.ztMax - tp_.ztMin;

  grpOdd.resize(s_.nThetaReduced * s_.nZeta);
  if (s_.lasym) {
    grpEvn.resize(s_.nThetaReduced * s_.nZeta);
  }

  const int mnpd = (2 * nf + 1) * (mf + 1);
  grpmn_sin.resize(mnpd * numLocal);
  if (s_.lasym) {
    grpmn_cos.resize(mnpd * numLocal);
  }

  gstore_symm.resize(s_.nThetaReduced * s_.nZeta);

  const int size_b = s_.nThetaReduced * (2 * nf + 1);
  bcos.resize(size_b);
  bsin.resize(size_b);

  const int size_a_temp = (mf + 1) * (2 * nf + 1) * (2 * nf + 1) * s_.nThetaEff;
  actemp.resize(size_a_temp);
  astemp.resize(size_a_temp);

  bvec_sin.resize(mnpd);
  amat_sin_sin.resize(mnpd * mnpd);
}

// fourp()-equivalent
void LaplaceSolver::TransformGreensFunctionDerivative(
    const std::vector<double> &greenp) {
  const int mnpd = (2 * nf + 1) * (mf + 1);
  absl::c_fill_n(grpmn_sin, mnpd * numLocal, 0);

  for (int klp = tp_.ztMin; klp < tp_.ztMax; ++klp) {
    const int klpRel = klp - tp_.ztMin;
    for (int l = 0; l < s_.nThetaReduced; ++l) {
      const int lRev = (s_.nThetaEven - l) % s_.nThetaEven;

      std::vector<double> g1_symm(nf + 1);
      std::vector<double> g2_symm(nf + 1);

      for (int k = 0; k < s_.nZeta; ++k) {
        const int kRev = (s_.nZeta - k) % s_.nZeta;

        const int kl = l * s_.nZeta + k;
        const int klRev = lRev * s_.nZeta + kRev;

        for (int n = 0; n < nf + 1; ++n) {
          const int idx_nk = n * s_.nZeta + k;

          const int klpOff = (klp - tp_.ztMin) * s_.nThetaEven * s_.nZeta;

          const double cosn = fb_.cosnv[idx_nk] / fb_.nscale[n];
          const double sinn = fb_.sinnv[idx_nk] / fb_.nscale[n];

          const double kernel_odd =
              (greenp[klpOff + kl] - greenp[klpOff + klRev]) * 0.5;

          // TODO(jons): finish this when implementing non-stellarator-symmetric
          // code path double kernel_even = 0.0; if (s_.lasym) {
          //     kernel_even = (greenp[klpOff + kl] + greenp[klpOff + klRev]) *
          //     0.5;
          // }

          g1_symm[n] += cosn * kernel_odd;
          g2_symm[n] += sinn * kernel_odd;
        }  // n
      }  // k

      for (int m = 0; m < mf + 1; ++m) {
        const int idx_lm = l * (s_.mnyq2 + 1) + m;

        double cosmui = fb_.cosmui[idx_lm] / fb_.mscale[m];
        double sinmui = fb_.sinmui[idx_lm] / fb_.mscale[m];

        for (int n = 0; n < nf + 1; ++n) {
          const int idx_m_posn = (nf + n) * (mf + 1) + m;
          const int idx_m_negn = (nf - n) * (mf + 1) + m;

          const double gcos_symm = g1_symm[n] * sinmui;
          const double gsin_symm = g2_symm[n] * cosmui;

          grpmn_sin[idx_m_posn * numLocal + klpRel] += gcos_symm - gsin_symm;
          if (n > 0) {
            grpmn_sin[idx_m_negn * numLocal + klpRel] += gcos_symm + gsin_symm;
          }
        }  // n
      }  // m
    }  // l
  }  // kl'
}  // TransformGreensFunctionDerivative

void LaplaceSolver::SymmetriseSourceTerm(const std::vector<double> &gstore) {
  for (int l = 0; l < s_.nThetaReduced; ++l) {
    int lRev = (s_.nThetaEven - l) % s_.nThetaEven;
    for (int k = 0; k < s_.nZeta; ++k) {
      int kRev = (s_.nZeta - k) % s_.nZeta;

      int kl = l * s_.nZeta + k;
      int klRev = lRev * s_.nZeta + kRev;

      // 1/2 for even/odd decompoition
      gstore_symm[kl] = (gstore[kl] - gstore[klRev]) / 2;
    }  // k
  }  // l
}  // SymmetriseSourceTerm

void LaplaceSolver::AccumulateFullGrpmn(
    const std::vector<double> &grpmn_sin_singular) {
  const int mnpd = (mf + 1) * (2 * nf + 1);
  for (int mn = 0; mn < mnpd; ++mn) {
    for (int klp = tp_.ztMin; klp < tp_.ztMax; ++klp) {
      const int klpRel = klp - tp_.ztMin;

      // need scale factor 1/nfp for singular term!
      grpmn_sin[mn * numLocal + klpRel] +=
          grpmn_sin_singular[mn * numLocal + klpRel] / s_.nfp;
    }  // kl'
  }  // mn
}  // AccumulateFullGrpmn

void LaplaceSolver::PerformToroidalFourierTransforms() {
  const int size_b = s_.nThetaReduced * (2 * nf + 1);
  absl::c_fill_n(bcos, size_b, 0);
  absl::c_fill_n(bsin, size_b, 0);

  for (int n = 0; n < nf + 1; ++n) {
    for (int l = 0; l < s_.nThetaReduced; ++l) {
      // PERFORM KV (TOROIDAL ANGLE) TRANSFORM
      // For every n, compute an integral over the toroidal grid index k.
      for (int k = 0; k < s_.nZeta; ++k) {
        const int idx_nk = n * s_.nZeta + k;

        const double cosn = fb_.cosnv[idx_nk] / fb_.nscale[n];
        const double sinn = fb_.sinnv[idx_nk] / fb_.nscale[n];

        const int idx_kl = l * s_.nZeta + k;

        const int idx_l_posn = (nf + n) * s_.nThetaReduced + l;

        bcos[idx_l_posn] += cosn * gstore_symm[idx_kl];
        bsin[idx_l_posn] += sinn * gstore_symm[idx_kl];
      }  // k

      if (n > 0) {
        const int idx_l_posn = (nf + n) * s_.nThetaReduced + l;
        const int idx_l_negn = (nf - n) * s_.nThetaReduced + l;

        bcos[idx_l_negn] = bcos[idx_l_posn];
        bsin[idx_l_negn] = -bsin[idx_l_posn];
      }
    }  // l
  }  // n

  const int mnpd = (mf + 1) * (2 * nf + 1);
  const int size_a_temp = mnpd * (2 * nf + 1) * s_.nThetaEff;
  absl::c_fill_n(actemp, size_a_temp, 0);
  absl::c_fill_n(astemp, size_a_temp, 0);

  // PERFORM KV (TOROIDAL ANGLE) TRANSFORM
  // For every n, compute an integral over the toroidal grid index k.
  for (int mn = 0; mn < mnpd; ++mn) {
    for (int n = 0; n < nf + 1; ++n) {
      for (int klp = tp_.ztMin; klp < tp_.ztMax; ++klp) {
        const int klpRel = klp - tp_.ztMin;
        const int l = klp / s_.nZeta;
        const int k = klp % s_.nZeta;

        const int idx_nk = n * s_.nZeta + k;

        const int idx_a_posn =
            (mn * (2 * nf + 1) + (nf + n)) * s_.nThetaEff + l;

        const double cosn = fb_.cosnv[idx_nk] / fb_.nscale[n];
        const double sinn = fb_.sinnv[idx_nk] / fb_.nscale[n];

        actemp[idx_a_posn] += cosn * grpmn_sin[mn * numLocal + klpRel];
        astemp[idx_a_posn] += sinn * grpmn_sin[mn * numLocal + klpRel];
      }  // kl'
    }  // n
  }  // mn

  for (int mn = 0; mn < mnpd; ++mn) {
    // starting at n=1 includes check for n > 0 already
    for (int n = 1; n < nf + 1; ++n) {
      for (int klp = tp_.ztMin; klp < tp_.ztMax; ++klp) {
        const int l = klp / s_.nZeta;

        const int idx_a_posn =
            (mn * (2 * nf + 1) + (nf + n)) * s_.nThetaEff + l;
        const int idx_a_negn =
            (mn * (2 * nf + 1) + (nf - n)) * s_.nThetaEff + l;

        actemp[idx_a_negn] = actemp[idx_a_posn];
        astemp[idx_a_negn] = -astemp[idx_a_posn];
      }  // klp, effectively l
    }  // n
  }  // mn
}  // PerformToroidalFourierTransforms

void LaplaceSolver::PerformPoloidalFourierTransforms() {
  const int mnpd = (mf + 1) * (2 * nf + 1);
  absl::c_fill_n(bvec_sin, mnpd, 0);
  absl::c_fill_n(amat_sin_sin, mnpd * mnpd, 0);

  for (int all_n = 0; all_n < 2 * nf + 1; ++all_n) {
    for (int m = 0; m < mf + 1; ++m) {
      for (int l = 0; l < s_.nThetaReduced; ++l) {
        const int idx_lm = l * (s_.mnyq2 + 1) + m;

        double cosmui = fb_.cosmui[idx_lm] / fb_.mscale[m];
        double sinmui = fb_.sinmui[idx_lm] / fb_.mscale[m];

        const int idx_l_all_n = all_n * s_.nThetaReduced + l;
        bvec_sin[all_n * (mf + 1) + m] +=
            bcos[idx_l_all_n] * sinmui - bsin[idx_l_all_n] * cosmui;
      }  // l
    }  // m
  }  // all_n

  // -----------------

  for (int mn = 0; mn < mnpd; ++mn) {
    // linear index over all -nf:nf
    for (int all_n = 0; all_n < 2 * nf + 1; ++all_n) {
      // NOTE: This is a little uneconomic,
      // as not all l have been touched by this thread.
      for (int l = 0; l < s_.nThetaReduced; ++l) {
        for (int m = 0; m < mf + 1; ++m) {
          const int idx_lm = l * (s_.mnyq2 + 1) + m;

          const double cosmui = fb_.cosmui[idx_lm] / fb_.mscale[m];
          const double sinmui = fb_.sinmui[idx_lm] / fb_.mscale[m];

          const int idx_atemp = (mn * (2 * nf + 1) + all_n) * s_.nThetaEff + l;

          const int idx_amat = (all_n * (mf + 1) + m) * mnpd + mn;
          amat_sin_sin[idx_amat] +=
              actemp[idx_atemp] * sinmui - astemp[idx_atemp] * cosmui;
        }  // m
      }  // l
    }  // all_n
  }  // mn
}  // PerformPoloidalFourierTransforms

void LaplaceSolver::BuildMatrix() {
  const int mnpd = (mf + 1) * (2 * nf + 1);
#ifdef _OPENMP
#pragma omp single
#endif  // _OPENMP
  absl::c_fill_n(matrixShare, mnpd * mnpd, 0);
#ifdef _OPENMP
#pragma omp barrier
#endif  // _OPENMP

#ifdef _OPENMP
#pragma omp critical
#endif  // _OPENMP
  {
    for (int mn_mnp = 0; mn_mnp < mnpd * mnpd; ++mn_mnp) {
      matrixShare[mn_mnp] += amat_sin_sin[mn_mnp];
    }  // mn * mn'
  }
#ifdef _OPENMP
#pragma omp barrier
#endif  // _OPENMP

#ifdef _OPENMP
#pragma omp single
#endif  // _OPENMP
  {
    // TODO(jons): Get back to only having the minimal set of unique Fourier
    // coefficients in the linear system. set n = [-nf, ..., -1], m=0 elements
    // to zero
    // --> only have unique non-zero Fourier coefficients in linear system!
    for (int mnp = 0; mnp < mnpd; ++mnp) {
      for (int all_n = 0; all_n < nf; ++all_n) {
        const int m = 0;

        matrixShare[(mnp * (2 * nf + 1) + all_n) * (mf + 1) + m] = 0.0;
      }  // all_n
    }  // mn'

    // add diagonal term
    for (int mn = 0; mn < mnpd; ++mn) {
      // TODO(jons): with current normalizations, the diagonal term needs to be
      // 1/2. This could be due to dividing out mscale and nscale, I guess? An
      // indication for this being related to mscale and nscale is that in
      // Fortran VMEC/Nestor, the cos-cos (0,0)-(0,0) mode needs to get an
      // additional factor of 2!
      matrixShare[mn * mnpd + mn] += 0.5;
    }  // mn
  }
#ifdef _OPENMP
#pragma omp barrier
#endif  // _OPENMP
}  // BuildMatrix

void LaplaceSolver::DecomposeMatrix() {
  // use OPENBLAS_NUM_THREADS to set parallelism in OpenBLAS

  int mnpd = (mf + 1) * (2 * nf + 1);

  // NOTE:
  // As soon as LAPACK starts working on `matrixShare`,
  // it is not consistent with the value on entry anymore
  // and thus cannot be used for testing anymore.

  // perform LU factorization of the matrix
  // (only needed when matrix is updated --> every nvacskip iterations)
  int info;
  dgetrf_(&mnpd, &mnpd, matrixShare.data(), &mnpd, iPiv.data(), &info);

  if (info < 0) {
    std::cout << -info << "-th argument to dgetrf is wrong\n";
  } else if (info > 0) {
    std::cout << absl::StrFormat(
        "U(%d,%d) is exactly zero in dgetrf --> singular matrix!\n", info,
        info);
  }

  CHECK_EQ(info, 0) << "dgetrf error";
}  // DecomposeMatrix

void LaplaceSolver::SolveForPotential(
    const std::vector<double> &bvec_sin_singular) {
  int mnpd = (mf + 1) * (2 * nf + 1);
#ifdef _OPENMP
#pragma omp single
#endif  // _OPENMP
  absl::c_fill_n(bvecShare, mnpd, 0);
#ifdef _OPENMP
#pragma omp barrier
#endif  // _OPENMP

#ifdef _OPENMP
#pragma omp critical
#endif  // _OPENMP
  {
    for (int mn = 0; mn < mnpd; ++mn) {
      bvecShare[mn] += bvec_sin[mn] + bvec_sin_singular[mn] / s_.nfp;
    }  // mn
  }
#ifdef _OPENMP
#pragma omp barrier
#endif  // _OPENMP

#ifdef _OPENMP
#pragma omp single
#endif  // _OPENMP
  {
    // TODO(jons): Get back to only having the minimal set of unique Fourier
    // coefficients in the linear system. set n = [-nf, ..., -1], m=0 elements
    // to zero
    // --> only have unique non-zero Fourier coefficients in linear system!
    for (int all_n = 0; all_n < nf; ++all_n) {
      const int m = 0;
      bvecShare[all_n * (mf + 1) + m] = 0.0;
    }

    // use OPENBLAS_NUM_THREADS to set parallelism in OpenBLAS

    // solve for given RHS
    int one = 1;
    int info;
    char no_transpose = 'N';
    dgetrs_(&no_transpose, &mnpd, &one, matrixShare.data(), &mnpd, iPiv.data(),
            bvecShare.data(), &mnpd, &info);

    if (info < 0) {
      std::cout << -info << "-th argument to dgetrs wrong\n";
    }

    CHECK_EQ(info, 0) << "dgetrs error";
  }
#ifdef _OPENMP
#pragma omp barrier
#endif  // _OPENMP
}

}  // namespace vmecpp
