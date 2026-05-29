// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#include "vmecpp/free_boundary/laplace_solver/laplace_solver.h"

#include <cmath>
#include <iostream>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"

///  LU factorization of a general M-by-N matrix A
extern "C" void dgetrf_(int* m, int* n, double* a, int* lda, int* ipiv,
                        int* info);
///  Solves a system of linear equations using the LU factorization.
extern "C" void dgetrs_(char* transpose, int* num_rows, int* num_columns,
                        double* matrix, int* leading_dim, int* pivot, double* y,
                        int* y_leading_dim, int* info);

namespace vmecpp {

LaplaceSolver::LaplaceSolver(const Sizes* s, const FourierBasisFastToroidal* fb,
                             const TangentialPartitioning* tp, int nf, int mf,
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
  grpOdd.setZero();
  if (s_.lasym) {
    grpEvn.resize(s_.nThetaReduced * s_.nZeta);
    grpEvn.setZero();
  }

  const int mnpd = (2 * nf + 1) * (mf + 1);
  grpmn_sin.resize(mnpd * numLocal);
  grpmn_sin.setZero();
  if (s_.lasym) {
    grpmn_cos.resize(mnpd * numLocal);
    grpmn_cos.setZero();
  }

  gstore_symm.resize(s_.nThetaReduced * s_.nZeta);
  gstore_symm.setZero();

  const int size_b = s_.nThetaReduced * (2 * nf + 1);
  bcos.resize(size_b);
  bcos.setZero();
  bsin.resize(size_b);
  bsin.setZero();

  const int size_a_temp = (mf + 1) * (2 * nf + 1) * (2 * nf + 1) * s_.nThetaEff;
  actemp.resize(size_a_temp);
  actemp.setZero();
  astemp.resize(size_a_temp);
  astemp.setZero();

  bvec_sin.resize(mnpd);
  bvec_sin.setZero();
  amat_sin_sin.resize(mnpd * mnpd);
  amat_sin_sin.setZero();

  if (s_.lasym) {
    gstore_asym.resize(s_.nThetaReduced * s_.nZeta);
    gstore_asym.setZero();

    bcos_asym.resize(size_b);
    bcos_asym.setZero();
    bsin_asym.resize(size_b);
    bsin_asym.setZero();

    actemp_cos.resize(size_a_temp);
    actemp_cos.setZero();
    astemp_cos.resize(size_a_temp);
    astemp_cos.setZero();

    bvec_cos.resize(mnpd);
    bvec_cos.setZero();
    amat_sin_cos.resize(mnpd * mnpd);
    amat_sin_cos.setZero();
    amat_cos_sin.resize(mnpd * mnpd);
    amat_cos_sin.setZero();
    amat_cos_cos.resize(mnpd * mnpd);
    amat_cos_cos.setZero();
  }

  // Pre-compute scaled Fourier basis matrices for efficient matrix operations
  // cosnv_scaled[n, k] = cosnv[n * nZeta + k] / nscale[n]
  cosnv_scaled.resize(nf + 1, s_.nZeta);
  sinnv_scaled.resize(nf + 1, s_.nZeta);
  for (int n = 0; n < nf + 1; ++n) {
    const double scale = 1.0 / fb_.nscale[n];
    for (int k = 0; k < s_.nZeta; ++k) {
      const int idx_nk = n * s_.nZeta + k;
      cosnv_scaled(n, k) = fb_.cosnv[idx_nk] * scale;
      sinnv_scaled(n, k) = fb_.sinnv[idx_nk] * scale;
    }
  }

  // cosmui_scaled[l, m] = cosmui[l * (mnyq2+1) + m] / mscale[m]
  cosmui_scaled.resize(s_.nThetaReduced, mf + 1);
  sinmui_scaled.resize(s_.nThetaReduced, mf + 1);
  for (int l = 0; l < s_.nThetaReduced; ++l) {
    for (int m = 0; m < mf + 1; ++m) {
      const int idx_lm = l * (s_.mnyq2 + 1) + m;
      const double scale = 1.0 / fb_.mscale[m];
      cosmui_scaled(l, m) = fb_.cosmui[idx_lm] * scale;
      sinmui_scaled(l, m) = fb_.sinmui[idx_lm] * scale;
    }
  }
}

// fourp()-equivalent
void LaplaceSolver::TransformGreensFunctionDerivative(
    const std::vector<double>& greenp) {
  grpmn_sin.setZero();
  if (s_.lasym) {
    grpmn_cos.setZero();
  }

  // Temporary vectors for toroidal transform results - allocated once outside
  // loops
  Eigen::VectorXd g1_symm(nf + 1);
  Eigen::VectorXd g2_symm(nf + 1);
  Eigen::VectorXd kernel_odd(s_.nZeta);
  Eigen::VectorXd g1_asym(nf + 1);
  Eigen::VectorXd g2_asym(nf + 1);
  Eigen::VectorXd kernel_even(s_.nZeta);

  for (int klp = tp_.ztMin; klp < tp_.ztMax; ++klp) {
    const int klpRel = klp - tp_.ztMin;
    const int klpOff = klpRel * s_.nThetaEven * s_.nZeta;

    for (int l = 0; l < s_.nThetaReduced; ++l) {
      const int lRev = (s_.nThetaEven - l) % s_.nThetaEven;

      for (int k = 0; k < s_.nZeta; ++k) {
        const int kRev = (s_.nZeta - k) % s_.nZeta;
        const int kl = l * s_.nZeta + k;
        const int klRev = lRev * s_.nZeta + kRev;
        // 0.5 factor for even/odd decomposition
        kernel_odd[k] = (greenp[klpOff + kl] - greenp[klpOff + klRev]) * 0.5;
        if (s_.lasym) {
          kernel_even[k] = (greenp[klpOff + kl] + greenp[klpOff + klRev]) * 0.5;
        }
      }

      // Compute toroidal transform using matrix-vector product
      // g1_symm[n] = sum_k cosnv_scaled[n, k] * kernel_odd[k]
      // g2_symm[n] = sum_k sinnv_scaled[n, k] * kernel_odd[k]
      g1_symm.noalias() = cosnv_scaled * kernel_odd;
      g2_symm.noalias() = sinnv_scaled * kernel_odd;
      if (s_.lasym) {
        g1_asym.noalias() = cosnv_scaled * kernel_even;
        g2_asym.noalias() = sinnv_scaled * kernel_even;
      }

      // Compute poloidal transform
      for (int m = 0; m < mf + 1; ++m) {
        const double cosmui = cosmui_scaled(l, m);
        const double sinmui = sinmui_scaled(l, m);

        for (int n = 0; n < nf + 1; ++n) {
          const int idx_m_posn = (nf + n) * (mf + 1) + m;
          const int idx_m_negn = (nf - n) * (mf + 1) + m;

          const double gcos_symm = g1_symm[n] * sinmui;
          const double gsin_symm = g2_symm[n] * cosmui;

          grpmn_sin[idx_m_posn * numLocal + klpRel] += gcos_symm - gsin_symm;
          if (n > 0) {
            grpmn_sin[idx_m_negn * numLocal + klpRel] += gcos_symm + gsin_symm;
          }

          if (s_.lasym) {
            // even kernel maps to cos(mu - nv) and cos(mu + nv) basis:
            // g1_asym*cosmui + g2_asym*sinmui = cos(mu)*cos(nv) +
            // sin(mu)*sin(nv)
            //   = cos(mu - nv)  [posn]
            // g1_asym*cosmui - g2_asym*sinmui = cos(mu)*cos(nv) -
            // sin(mu)*sin(nv)
            //   = cos(mu + nv)  [negn]
            const double gcos_asym = g1_asym[n] * cosmui;
            const double gsin_asym = g2_asym[n] * sinmui;

            grpmn_cos[idx_m_posn * numLocal + klpRel] += gcos_asym + gsin_asym;
            if (n > 0) {
              grpmn_cos[idx_m_negn * numLocal + klpRel] +=
                  gcos_asym - gsin_asym;
            }
          }
        }  // n
      }  // m
    }  // l
  }  // kl'
}  // TransformGreensFunctionDerivative

void LaplaceSolver::SymmetriseSourceTerm(const std::vector<double>& gstore) {
  // gstore_symm receives the half that is anti-symmetric under (u,v) -> (-u,-v)
  // and feeds the sin-basis source. gstore_asym receives the symmetric half
  // and feeds the cos-basis source (lasym only). Both carry a 1/2 from the
  // even/odd decomposition.
  for (int l = 0; l < s_.nThetaReduced; ++l) {
    const int lRev = (s_.nThetaEven - l) % s_.nThetaEven;
    for (int k = 0; k < s_.nZeta; ++k) {
      const int kRev = (s_.nZeta - k) % s_.nZeta;

      const int kl = l * s_.nZeta + k;
      const int klRev = lRev * s_.nZeta + kRev;

      gstore_symm[kl] = (gstore[kl] - gstore[klRev]) * 0.5;
      if (s_.lasym) {
        gstore_asym[kl] = (gstore[kl] + gstore[klRev]) * 0.5;
      }
    }  // k
  }  // l
}  // SymmetriseSourceTerm

void LaplaceSolver::AccumulateFullGrpmn(
    const std::vector<double>& grpmn_sin_singular,
    const std::vector<double>& grpmn_cos_singular) {
  const int mnpd = (mf + 1) * (2 * nf + 1);
  const double inv_nfp = 1.0 / s_.nfp;

  // Use Eigen Map to view std::vector as Eigen vector and vectorize
  Eigen::Map<const Eigen::VectorXd> sin_singular(grpmn_sin_singular.data(),
                                                 mnpd * numLocal);
  grpmn_sin += sin_singular * inv_nfp;

  if (s_.lasym) {
    Eigen::Map<const Eigen::VectorXd> cos_singular(grpmn_cos_singular.data(),
                                                   mnpd * numLocal);
    grpmn_cos += cos_singular * inv_nfp;
  }
}  // AccumulateFullGrpmn

void LaplaceSolver::PerformToroidalFourierTransforms() {
  bcos.setZero();
  bsin.setZero();
  if (s_.lasym) {
    bcos_asym.setZero();
    bsin_asym.setZero();
  }

  // Map gstore_symm as a matrix [nThetaReduced x nZeta] for efficient access
  Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                                 Eigen::RowMajor>>
      gstore_mat(gstore_symm.data(), s_.nThetaReduced, s_.nZeta);

  // First loop: compute bcos and bsin using matrix multiplication
  // bcos_mat[n, l] = sum_k cosnv_scaled[n,k] * gstore_mat[l,k]
  // This is equivalent to: bcos_mat = cosnv_scaled * gstore_mat^T
  Eigen::MatrixXd bcos_mat = cosnv_scaled * gstore_mat.transpose();
  Eigen::MatrixXd bsin_mat = sinnv_scaled * gstore_mat.transpose();

  // Same toroidal transform on the cos-basis source (lasym only).
  Eigen::MatrixXd bcos_mat_asym;
  Eigen::MatrixXd bsin_mat_asym;
  if (s_.lasym) {
    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                                   Eigen::RowMajor>>
        gstore_asym_mat(gstore_asym.data(), s_.nThetaReduced, s_.nZeta);
    bcos_mat_asym = cosnv_scaled * gstore_asym_mat.transpose();
    bsin_mat_asym = sinnv_scaled * gstore_asym_mat.transpose();
  }

  // Copy results to the output arrays with proper indexing
  for (int n = 0; n < nf + 1; ++n) {
    for (int l = 0; l < s_.nThetaReduced; ++l) {
      const int idx_l_posn = (nf + n) * s_.nThetaReduced + l;
      bcos[idx_l_posn] = bcos_mat(n, l);
      bsin[idx_l_posn] = bsin_mat(n, l);

      if (n > 0) {
        const int idx_l_negn = (nf - n) * s_.nThetaReduced + l;
        bcos[idx_l_negn] = bcos_mat(n, l);
        bsin[idx_l_negn] = -bsin_mat(n, l);
      }

      if (s_.lasym) {
        bcos_asym[idx_l_posn] = bcos_mat_asym(n, l);
        bsin_asym[idx_l_posn] = bsin_mat_asym(n, l);
        if (n > 0) {
          const int idx_l_negn = (nf - n) * s_.nThetaReduced + l;
          bcos_asym[idx_l_negn] = bcos_mat_asym(n, l);
          bsin_asym[idx_l_negn] = -bsin_mat_asym(n, l);
        }
      }
    }  // l
  }  // n

  const int mnpd = (mf + 1) * (2 * nf + 1);
  actemp.setZero();
  astemp.setZero();
  if (s_.lasym) {
    actemp_cos.setZero();
    astemp_cos.setZero();
  }

  // PERFORM KV (TOROIDAL ANGLE) TRANSFORM for matrix A
  // The key insight: for each (l, k) pair in the klp range, we accumulate:
  // actemp[(mn, nf+n, l)] += cosnv_scaled(n, k) * grpmn_sin(mn, klpRel)
  //
  // Reorganize as: for each k, accumulate contributions from all (l, mn, n)
  // For a given k, cosnv_scaled(:, k) is a vector of length (nf+1)
  // grpmn_sin is [mnpd x numLocal]

  // For each unique l value in [tp_.ztMin, tp_.ztMax)
  for (int l = 0; l < s_.nThetaEff; ++l) {
    // Find all k values for this l
    for (int k = 0; k < s_.nZeta; ++k) {
      const int klp = l * s_.nZeta + k;
      if (klp < tp_.ztMin || klp >= tp_.ztMax) continue;

      const int klpRel = klp - tp_.ztMin;

      // Get the column of cosnv_scaled and sinnv_scaled for this k
      Eigen::VectorXd cosn_k = cosnv_scaled.col(k);
      Eigen::VectorXd sinn_k = sinnv_scaled.col(k);

      for (int mn = 0; mn < mnpd; ++mn) {
        const double grpmn_sin_val = grpmn_sin[mn * numLocal + klpRel];
        const double grpmn_cos_val =
            s_.lasym ? grpmn_cos[mn * numLocal + klpRel] : 0.0;

        // Vectorized accumulation for all n values
        for (int n = 0; n < nf + 1; ++n) {
          const int idx_a_posn =
              (mn * (2 * nf + 1) + (nf + n)) * s_.nThetaEff + l;
          actemp[idx_a_posn] += cosn_k[n] * grpmn_sin_val;
          astemp[idx_a_posn] += sinn_k[n] * grpmn_sin_val;
          if (s_.lasym) {
            actemp_cos[idx_a_posn] += cosn_k[n] * grpmn_cos_val;
            astemp_cos[idx_a_posn] += sinn_k[n] * grpmn_cos_val;
          }
        }
      }
    }
  }

  // Copy positive n to negative n with sign change
  for (int mn = 0; mn < mnpd; ++mn) {
    for (int n = 1; n < nf + 1; ++n) {
      for (int l = 0; l < s_.nThetaEff; ++l) {
        const int idx_a_posn =
            (mn * (2 * nf + 1) + (nf + n)) * s_.nThetaEff + l;
        const int idx_a_negn =
            (mn * (2 * nf + 1) + (nf - n)) * s_.nThetaEff + l;

        actemp[idx_a_negn] = actemp[idx_a_posn];
        astemp[idx_a_negn] = -astemp[idx_a_posn];
        if (s_.lasym) {
          actemp_cos[idx_a_negn] = actemp_cos[idx_a_posn];
          astemp_cos[idx_a_negn] = -astemp_cos[idx_a_posn];
        }
      }  // klp, effectively l
    }  // n
  }  // mn
}  // PerformToroidalFourierTransforms

void LaplaceSolver::PerformPoloidalFourierTransforms() {
  const int mnpd = (mf + 1) * (2 * nf + 1);
  bvec_sin.setZero();
  amat_sin_sin.setZero();
  if (s_.lasym) {
    bvec_cos.setZero();
    amat_sin_cos.setZero();
    amat_cos_sin.setZero();
    amat_cos_cos.setZero();
  }

  // bvec_sin uses bcos/bsin (from the anti-symmetric source) with the
  // (sinm, -cosm) sin-projection weights. bvec_cos uses bcos_asym/bsin_asym
  // (from the symmetric source) with the (cosm, +sinm) cos-projection
  // weights (Fortran NESTOR/fouri.f90 lines 117-119).
  for (int all_n = 0; all_n < 2 * nf + 1; ++all_n) {
    Eigen::Map<const Eigen::VectorXd> bcos_n(
        bcos.data() + all_n * s_.nThetaReduced, s_.nThetaReduced);
    Eigen::Map<const Eigen::VectorXd> bsin_n(
        bsin.data() + all_n * s_.nThetaReduced, s_.nThetaReduced);

    Eigen::VectorXd result_sin =
        sinmui_scaled.transpose() * bcos_n - cosmui_scaled.transpose() * bsin_n;

    for (int m = 0; m < mf + 1; ++m) {
      bvec_sin[all_n * (mf + 1) + m] = result_sin[m];
    }

    if (s_.lasym) {
      Eigen::Map<const Eigen::VectorXd> bcos_asym_n(
          bcos_asym.data() + all_n * s_.nThetaReduced, s_.nThetaReduced);
      Eigen::Map<const Eigen::VectorXd> bsin_asym_n(
          bsin_asym.data() + all_n * s_.nThetaReduced, s_.nThetaReduced);

      Eigen::VectorXd result_cos = cosmui_scaled.transpose() * bcos_asym_n +
                                   sinmui_scaled.transpose() * bsin_asym_n;
      for (int m = 0; m < mf + 1; ++m) {
        bvec_cos[all_n * (mf + 1) + m] = result_cos[m];
      }
    }
  }  // all_n

  // Matrix blocks. amat_sin_sin (Fortran amatrix(:,:,1)) uses actemp/astemp
  // (sin'-projected kernel) with sin-unprimed weights. amat_sin_cos
  // (amatrix(:,:,2)) uses the same actemp/astemp with cos-unprimed weights.
  // amat_cos_sin (amatrix(:,:,3)) uses actemp_cos/astemp_cos (cos'-projected
  // kernel) with sin-unprimed weights. amat_cos_cos (amatrix(:,:,4)) uses
  // actemp_cos/astemp_cos with cos-unprimed weights. The four blocks are
  // assembled into amatsq in BuildMatrix.
  for (int mn = 0; mn < mnpd; ++mn) {
    for (int all_n = 0; all_n < 2 * nf + 1; ++all_n) {
      Eigen::VectorXd actemp_l(s_.nThetaReduced);
      Eigen::VectorXd astemp_l(s_.nThetaReduced);

      const int base_idx = (mn * (2 * nf + 1) + all_n) * s_.nThetaEff;
      for (int l = 0; l < s_.nThetaReduced; ++l) {
        actemp_l[l] = actemp[base_idx + l];
        astemp_l[l] = astemp[base_idx + l];
      }

      Eigen::VectorXd result_ss = sinmui_scaled.transpose() * actemp_l -
                                  cosmui_scaled.transpose() * astemp_l;

      for (int m = 0; m < mf + 1; ++m) {
        const int idx_amat = (all_n * (mf + 1) + m) * mnpd + mn;
        amat_sin_sin[idx_amat] = result_ss[m];
      }

      if (s_.lasym) {
        Eigen::VectorXd result_sc = cosmui_scaled.transpose() * actemp_l +
                                    sinmui_scaled.transpose() * astemp_l;
        for (int m = 0; m < mf + 1; ++m) {
          const int idx_amat = (all_n * (mf + 1) + m) * mnpd + mn;
          amat_sin_cos[idx_amat] = result_sc[m];
        }

        // The cos-source kernel grpmn_cos is evaluated on the full theta range.
        // Its poloidal projection must cover that range, so fold actemp_cos /
        // astemp_cos about theta -> -theta into the reduced-range parts the
        // reduced-range weights expect (sin-observation <- antisymmetric fold,
        // cos-observation <- symmetric fold; actemp transforms even under the
        // reflection, astemp odd).
        Eigen::VectorXd cac_so(s_.nThetaReduced), cas_so(s_.nThetaReduced);
        Eigen::VectorXd cac_co(s_.nThetaReduced), cas_co(s_.nThetaReduced);
        for (int l = 0; l < s_.nThetaReduced; ++l) {
          const int rl = (s_.nThetaEven - l) % s_.nThetaEven;
          const double ca = actemp_cos[base_idx + l];
          const double car = actemp_cos[base_idx + rl];
          const double cb = astemp_cos[base_idx + l];
          const double cbr = astemp_cos[base_idx + rl];
          cac_so[l] = 0.5 * (ca - car);
          cas_so[l] = 0.5 * (cb + cbr);
          cac_co[l] = 0.5 * (ca + car);
          cas_co[l] = 0.5 * (cb - cbr);
        }

        Eigen::VectorXd result_cs = sinmui_scaled.transpose() * cac_so -
                                    cosmui_scaled.transpose() * cas_so;
        Eigen::VectorXd result_cc = cosmui_scaled.transpose() * cac_co +
                                    sinmui_scaled.transpose() * cas_co;
        for (int m = 0; m < mf + 1; ++m) {
          const int idx_amat = (all_n * (mf + 1) + m) * mnpd + mn;
          amat_cos_sin[idx_amat] = result_cs[m];
          amat_cos_cos[idx_amat] = result_cc[m];
        }
      }
    }  // all_n
  }  // mn
}  // PerformPoloidalFourierTransforms

void LaplaceSolver::BuildMatrix() {
  const int mnpd = (mf + 1) * (2 * nf + 1);
  const int mnpd_dim = s_.lasym ? 2 * mnpd : mnpd;

#ifdef _OPENMP
#pragma omp single
#endif  // _OPENMP
  absl::c_fill_n(matrixShare, mnpd_dim * mnpd_dim, 0);
#ifdef _OPENMP
#pragma omp barrier
#endif  // _OPENMP

  // Each thread accumulates its contribution to amatrix into matrixShare. For
  // lasym = false this is a flat add; for lasym = true the four blocks
  // (sin-sin, sin-cos, cos-sin, cos-cos) are placed into the four quadrants
  // of the column-major 2 * mnpd matrix (Fortran NESTOR/fouri.f90 amatsq).
#ifdef _OPENMP
#pragma omp critical
#endif  // _OPENMP
  {
    if (!s_.lasym) {
      Eigen::Map<Eigen::VectorXd> matrix_map(matrixShare.data(), mnpd * mnpd);
      matrix_map += amat_sin_sin;
    } else {
      const int stride = mnpd_dim;  // column-major leading dim
      for (int j = 0; j < mnpd; ++j) {
        for (int i = 0; i < mnpd; ++i) {
          const int src = j * mnpd + i;
          // top-left: sin-sin'
          matrixShare[i + j * stride] += amat_sin_sin[src];
          // top-right: sin-cos' (Fortran amatrix(:,:,2))
          matrixShare[i + (j + mnpd) * stride] += amat_sin_cos[src];
          // bottom-left: cos-sin' (Fortran amatrix(:,:,3))
          matrixShare[(i + mnpd) + j * stride] += amat_cos_sin[src];
          // bottom-right: cos-cos'
          matrixShare[(i + mnpd) + (j + mnpd) * stride] += amat_cos_cos[src];
        }
      }
    }
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
        // For lasym the matrix is mnpd_dim x mnpd_dim column-major, so the
        // (row, col) entry sits at row + col * mnpd_dim. The Fortran code's
        // amatrix(1:mn0-mf1:mf1, :, :) zero-out applies independently to all
        // ndim^2 blocks; here we mirror that by zeroing the corresponding
        // rows in each block (m=0, all_n<nf is the m=0, n<0 set).
        if (!s_.lasym) {
          matrixShare[(mnp * (2 * nf + 1) + all_n) * (mf + 1) + m] = 0.0;
        } else {
          const int row = all_n * (mf + 1) + m;
          matrixShare[row + mnp * mnpd_dim] = 0.0;
          matrixShare[row + (mnp + mnpd) * mnpd_dim] = 0.0;
          matrixShare[(row + mnpd) + mnp * mnpd_dim] = 0.0;
          matrixShare[(row + mnpd) + (mnp + mnpd) * mnpd_dim] = 0.0;
        }
      }  // all_n
    }  // mn'

    // add diagonal term
    if (!s_.lasym) {
      for (int mn = 0; mn < mnpd; ++mn) {
        matrixShare[mn * mnpd + mn] += 0.5;
      }
    } else {
      // Both sin-sin and cos-cos diagonals carry the +0.5 from the analytic
      // singular contribution. The cos-cos (m=0, n=0) mode gets an extra
      // +0.5 (Fortran NESTOR/fouri.f90 line 183) to match the orthogonality
      // normalization of the cos basis at the (0, 0) mode.
      const int stride = mnpd_dim;
      const int mn0 = nf * (mf + 1);  // (m=0, n=0) in the all_n * (mf+1) + m
                                      // indexing
      for (int mn = 0; mn < mnpd; ++mn) {
        matrixShare[mn + mn * stride] += 0.5;
        matrixShare[(mn + mnpd) + (mn + mnpd) * stride] += 0.5;
      }
      matrixShare[(mn0 + mnpd) + (mn0 + mnpd) * stride] += 0.5;
    }
  }
#ifdef _OPENMP
#pragma omp barrier
#endif  // _OPENMP
}  // BuildMatrix

void LaplaceSolver::DecomposeMatrix() {
  const int mnpd = (mf + 1) * (2 * nf + 1);
  int mnpd_dim = s_.lasym ? 2 * mnpd : mnpd;

  // NOTE:
  // As soon as LAPACK starts working on `matrixShare`,
  // it is not consistent with the value on entry anymore
  // and thus cannot be used for testing anymore.

  // perform LU factorization of the matrix
  // (only needed when matrix is updated --> every nvacskip iterations)
  int info;
  dgetrf_(&mnpd_dim, &mnpd_dim, matrixShare.data(), &mnpd_dim, iPiv.data(),
          &info);

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
    const std::vector<double>& bvec_sin_singular,
    const std::vector<double>& bvec_cos_singular) {
  const int mnpd = (mf + 1) * (2 * nf + 1);
  const int mnpd_dim = s_.lasym ? 2 * mnpd : mnpd;
  const double inv_nfp = 1.0 / s_.nfp;

#ifdef _OPENMP
#pragma omp single
#endif  // _OPENMP
  absl::c_fill_n(bvecShare, mnpd_dim, 0);
#ifdef _OPENMP
#pragma omp barrier
#endif  // _OPENMP

#ifdef _OPENMP
#pragma omp critical
#endif  // _OPENMP
  {
    Eigen::Map<Eigen::VectorXd> bvec_sin_share(bvecShare.data(), mnpd);
    Eigen::Map<const Eigen::VectorXd> singular_sin(bvec_sin_singular.data(),
                                                   mnpd);
    bvec_sin_share += bvec_sin + singular_sin * inv_nfp;

    if (s_.lasym) {
      Eigen::Map<Eigen::VectorXd> bvec_cos_share(bvecShare.data() + mnpd, mnpd);
      Eigen::Map<const Eigen::VectorXd> singular_cos(bvec_cos_singular.data(),
                                                     mnpd);
      bvec_cos_share += bvec_cos + singular_cos * inv_nfp;
    }
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
      if (s_.lasym) {
        bvecShare[mnpd + all_n * (mf + 1) + m] = 0.0;
      }
    }

    // solve for given RHS
    int one = 1;
    int info;
    char no_transpose = 'N';
    int n = mnpd_dim;
    dgetrs_(&no_transpose, &n, &one, matrixShare.data(), &n, iPiv.data(),
            bvecShare.data(), &n, &info);

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
