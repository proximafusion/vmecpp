#ifndef VMECPP_FREE_BOUNDARY_VAC2_VAC2_SOLVER_H_
#define VMECPP_FREE_BOUNDARY_VAC2_VAC2_SOLVER_H_

#include <memory>
#include <vector>

#include "absl/status/statusor.h"

namespace vmecpp {

// Vac2Solver: the Strumberger/Tichmann "vac2" vacuum solver for 3D
// stellarator geometries (see Merkel & Strumberger, arXiv:1508.04911 for
// the formulation context). Port of the Fortran reference
// vac2/{matrix,fourier,foumat,bexmat,solver}.f90.
//
// Pipeline:
//   1. Kernel assembly: build ga, gc, gd on (nu·nv × nu·nv) grid from 3D
//      positions + log-singular subtraction (vac2_matrix.f90).
//   2. Spatial Fourier reduction: reduce each of ga, gc, gd to 6 mode-space
//      matrices (ac, as) × (a, c, d) over the "ip" index first, giving (nuv,
//      mnpot) matrices (vac2_fourier.f90 first pass).
//   3. Second Fourier reduction: reduce each of the 6 to 12 mode×mode
//      matrices (acc/ass/acs/asc for each of a, c, d variants — with "d"
//      being the gvv-weighted kernel and "b" the cross g_{uv} kernel —
//      giving 12 total) (vac2_foumat.f90).
//   4. RHS assembly: benf via Fourier projection of bexn; cpol, ctor from
//      the (m=0, n=0) row of the mode matrices.
//   5. 4-block matrix g(nd, nd) assembly per solver.f90:37-48.
//   6. Symmetrise + Cholesky factor + solve.
//   7. Reconstruct potU, potV on (nu, nv) grid from the mode-space solution
//      with the `-curtor, +curpol` secular base.
//   8. bsqvac = (½ potU² gvv - potU·potV guv + ½ potV² guu) /
//               (guu gvv - guv²).
//
// Memory:
//   - Stages 1+2 are matrix-free: ga/gc/gd are never materialized as
//     nuv × nuv arrays. Instead, per column c, we stream ga/gc/gd row
//     vectors of length nuv, 2D-r2c-FFT them, and discard. Peak storage
//     is dominated by the 6 mode-space arrays (ac, as, bc, bs, dc, ds)
//     sized (nuv × mnpot). At (nu=1024, nv=256, mpol=12, ntor=12):
//     nuv=262144, mnpot=288 → 6 × 262144 × 288 × 8 B ≈ 3.6 GB total,
//     easily fitting on a 32 GB machine.
//   - Requires FFTW3 at compile time (VMECPP_VAC2_HAVE_FFTW3=1).

class Vac2Solver {
 public:
  // mpol: poloidal mode cutoff (inclusive: modes 0..mpol-1).
  // ntor: toroidal mode half-range (inclusive: modes -ntor..ntor).
  // nu:   poloidal grid size (full period).
  // nv:   toroidal grid size per field period.
  // nfp:  number of field periods.
  Vac2Solver(int mpol, int ntor, int nu, int nv, int nfp);

  struct Input {
    // Surface geometry on the (nu · nv) grid, row-major with toroidal-
    // fastest convention: [ku * nv + kv].
    std::vector<double> x, y, z;     // 3D Cartesian position
    std::vector<double> xu, yu, zu;  // ∂/∂u (poloidal, u in [0,1))
    std::vector<double> xv, yv, zv;  // ∂/∂v (toroidal, v in [0,1))
    // Surface metric on the same grid:
    std::vector<double> guu, guv, gvv;  // g_{ij} (raw — with u,v in [0,1))
    // Normal vector (yu·zv - zu·yv, zu·xv - xu·zv, xu·yv - yu·xv) /|...|:
    std::vector<double> snx, sny, snz;

    // External-field normal component B_ext · n̂ on each grid point (source
    // for the scalar-potential BVP):
    std::vector<double> bexn;

    // Scalar-sources:
    double curpol = 0;  // 2π · ⟨R · B_φ⟩ on LCFS
    double curtor = 0;  // μ₀ · I_plasma

    // Toggle: lasym=true is the full asymmetric path (default false).
    bool lasym = false;
  };

  struct Output {
    // Pointwise on the (nu · nv) grid:
    std::vector<double> potU, potV;  // length nu · nv
    std::vector<double> bsqvac;      // length nu · nv

    // Diagnostics (exposed for tests):
    std::vector<double> g;     // [nd × nd] assembled 4-block matrix
    std::vector<double> hs;    // [nd] mode-space solution
    std::vector<double> cpol;  // [nd]
    std::vector<double> ctor;  // [nd]
    std::vector<double> benf;  // [nd]
  };

  // reuse_operator: when true and a previous full Solve cached the
  // geometry-dependent operator (kernel mode matrices, Cholesky factor,
  // cpol/ctor, basis tables), skip the expensive operator assembly and only
  // re-solve for the new right-hand side (bexn, curpol, curtor) on the
  // FROZEN operator, then reconstruct with the fresh metric from `in`.
  // This mirrors NESTOR's partial update (frozen LU + fresh RHS) and costs
  // O(nuv * mnpot) instead of O(nuv^2). Only supported for lasym = false;
  // falls back to a full solve otherwise or when no operator is cached.
  absl::StatusOr<std::unique_ptr<Output>> Solve(const Input& in,
                                                bool reuse_operator = false);

  // Derived dimensions (exposed for callers and tests).
  int mpol() const { return mpol_; }
  int ntor() const { return ntor_; }
  int nu() const { return nu_; }
  int nv() const { return nv_; }
  int nfp() const { return nfp_; }
  int nuv() const { return nu_ * nv_; }
  int mnpot() const { return mnpot_; }
  int nd() const { return nd_; }

 private:
  const int mpol_, ntor_, nu_, nv_, nfp_;
  // Mode layout follows vac2/precal.f90: one n=0 axis (m=0, n=0 is
  // excluded), then rows (m=1..mpol-1) × (n = -ntor..ntor), for a
  // total of:
  //   mnpot = (ntor + 1) + (mpol - 1) · (2·ntor + 1)
  // nd = 2·(mnpot - 1) (one cosine and one sine block, skipping the
  //                     m=0,n=0 entry which is redundant).
  const int mnpot_;
  const int nd_;
  // Precomputed mode-index tables: ma_[j] = m_mode(j), na_[j] = n_mode(j)
  // for j ∈ [0, mnpot_).
  std::vector<int> ma_, na_;
  // Trig tables on the grid: conu_[ku, m], sinu_[ku, m], conv_[kv, n],
  // sinv_[kv, n] for m ∈ [0, mpol_), n ∈ [0, ntor_].
  std::vector<double> conu_, sinu_, conv_, sinv_;

  // Stage 5: benf = Fourier projection of in.bexn using the cached basis
  // tables; zeroes the cosine half for lasym = false.
  void ComputeBenf(const Input& in, Output* out) const;

  // Stages 8 + 9: potU/potV reconstruction from out->hs and bsqvac from the
  // metric in `in`.
  void ReconstructPotentials(const Input& in, Output* out) const;

  // Cached operator state for reuse_operator solves (stellarator-symmetric
  // path only), valid for the boundary geometry of the last full Solve:
  bool has_cached_operator_ = false;
  // [mnpot1 x mnpot1] dpotrf('U')-factored cc block of the g matrix
  std::vector<double> cached_cc_factor_;
  std::vector<double> cached_cpol_;  // [nd]
  std::vector<double> cached_ctor_;  // [nd]
  // basis tables cos/sin(2pi(m u + n v)) on the grid, [nuv x mnpot]
  mutable std::vector<double> cached_basis_c_;
  mutable std::vector<double> cached_basis_s_;
};

}  // namespace vmecpp

#endif  // VMECPP_FREE_BOUNDARY_VAC2_VAC2_SOLVER_H_
