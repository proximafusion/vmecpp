// Minimal BIEST virtual-casing driver for validating the AEGIS on-surface
// quadrature against BIEST's high-order singular quadrature (the golden
// reference). Reads a surface grid + a field B on it, evaluates the generalized
// virtual-casing exterior field B_ext = BiotSavart(n x B) + Laplace(n.B) + B/2
// with BIEST's singular quadrature, and writes B_ext on the same grid.
//
// Input file:  line 1 "Nt Np"; then Nt*Np lines "x y z bx by bz" in t-major,
//              p-minor order (t outer, p inner).
// Output file: Nt*Np lines "bx by bz" (B_ext), same order.
#include <biest.hpp>
#include <fstream>
#include <iostream>
#include <sctl.hpp>
#include <string>

typedef double Real;

// BIEST's generalized virtual-casing principle (from
// applications/.../src/virtual-casing-principle.cpp), verbatim: no Laplace
// Neumann solve, direct singular quadrature.
template <class Real, sctl::Integer UPSAMPLE, sctl::Integer PDIM,
          sctl::Integer RDIM>
void GeneralVirtualCasing(sctl::Vector<Real>& Bext,
                          sctl::Vector<Real>& Bext_eval,
                          const biest::Surface<Real>& S,
                          const sctl::Vector<Real>& B) {
  constexpr sctl::Integer COORD_DIM = 3;
  sctl::Comm comm = sctl::Comm::Self();

  sctl::Long Nt = S.NTor();
  sctl::Long Np = S.NPol();
  sctl::Vector<biest::Surface<Real>> Svec(1);
  Svec[0] = S;

  biest::BoundaryIntegralOp<Real, 3, 3, UPSAMPLE, PDIM, RDIM> BiotSavartFxU(
      comm);
  BiotSavartFxU.SetupSingular(Svec, biest::BiotSavart3D<Real>::FxU());

  biest::BoundaryIntegralOp<Real, 1, 3, UPSAMPLE, PDIM, RDIM> LaplaceFxdU(comm);
  LaplaceFxdU.SetupSingular(Svec, biest::Laplace3D<Real>::FxdU());

  sctl::Vector<Real> dX, normal, area_elem;
  biest::SurfaceOp<Real> SurfOp(comm, Nt, Np);
  {
    SurfOp.Grad2D(dX, S.Coord());
    SurfOp.SurfNormalAreaElem(&normal, &area_elem, dX, &S.Coord());
  }
  {  // Orient the normal outward. SurfNormalAreaElem's sign follows the (t,p)
     // grid orientation, which need not be outward; the divergence theorem
     // (integral of X.n dA = 3V > 0 for an outward normal) fixes it robustly.
    sctl::Long Nn = normal.Dim() / COORD_DIM;
    Real flux = 0;
    for (sctl::Long i = 0; i < Nn; i++) {
      Real xn = 0;
      for (sctl::Integer k = 0; k < COORD_DIM; k++)
        xn += S.Coord()[k * Nn + i] * normal[k * Nn + i];
      flux += xn * area_elem[i];
    }
    if (flux < 0) normal = -1.0 * normal;
  }

  auto DotProd = [](sctl::Vector<Real>& AdotB, const sctl::Vector<Real>& A,
                    const sctl::Vector<Real>& Bv) {
    sctl::Long N = A.Dim() / COORD_DIM;
    if (AdotB.Dim() != N) AdotB.ReInit(N);
    for (sctl::Long i = 0; i < N; i++) {
      Real s = 0;
      for (sctl::Integer k = 0; k < COORD_DIM; k++)
        s += A[k * N + i] * Bv[k * N + i];
      AdotB[i] = s;
    }
  };
  auto CrossProd = [](sctl::Vector<Real>& AcrossB, const sctl::Vector<Real>& A,
                      const sctl::Vector<Real>& Bv) {
    sctl::Long N = A.Dim() / COORD_DIM;
    if (AcrossB.Dim() != COORD_DIM * N) AcrossB.ReInit(COORD_DIM * N);
    for (sctl::Long i = 0; i < N; i++) {
      sctl::StaticArray<Real, COORD_DIM> a, b, c;
      for (sctl::Integer k = 0; k < COORD_DIM; k++) {
        a[k] = A[k * N + i];
        b[k] = Bv[k * N + i];
      }
      c[0] = a[1] * b[2] - b[1] * a[2];
      c[1] = a[2] * b[0] - b[2] * a[0];
      c[2] = a[0] * b[1] - b[0] * a[1];
      for (sctl::Integer k = 0; k < COORD_DIM; k++) AcrossB[k * N + i] = c[k];
    }
  };

  sctl::Vector<Real> BdotN, J, B0, B1;
  DotProd(BdotN, B, normal);
  CrossProd(J, normal, B);
  LaplaceFxdU(B0, BdotN);  // Laplace(n.B)
  BiotSavartFxU(B1, J);    // BiotSavart(n x B)
  // Return both sign conventions so the caller can pick the one matching the
  // exterior limit: GeneralVirtualCasing uses 0.5B + B0 + B1; the app's
  // validated EvalBplasmaOnSurface uses 0.5B - B0 - B1.
  Bext = 0.5 * B + B0 + B1;
  Bext_eval = 0.5 * B - B0 - B1;
}

int main(int argc, char** argv) {
  if (argc < 4) {
    std::cerr << "usage: biest_driver <in> <out_general> <out_eval>\n";
    return 1;
  }
  std::ifstream f(argv[1]);
  if (!f.good()) {
    std::cerr << "cannot open " << argv[1] << "\n";
    return 1;
  }
  sctl::Long Nt, Np;
  f >> Nt >> Np;
  biest::Surface<Real> S(Nt, Np);
  sctl::Vector<Real> B(3 * Nt * Np);
  for (sctl::Long t = 0; t < Nt; t++) {
    for (sctl::Long p = 0; p < Np; p++) {
      Real x, y, z, bx, by, bz;
      f >> x >> y >> z >> bx >> by >> bz;
      S.Coord(t, p, 0) = x;
      S.Coord(t, p, 1) = y;
      S.Coord(t, p, 2) = z;
      B[(0 * Nt + t) * Np + p] = bx;
      B[(1 * Nt + t) * Np + p] = by;
      B[(2 * Nt + t) * Np + p] = bz;
    }
  }

  sctl::Vector<Real> Bext, Bext_eval;
  GeneralVirtualCasing<Real, 2, 30, 60>(Bext, Bext_eval, S, B);

  auto write = [&](const char* path, const sctl::Vector<Real>& V) {
    std::ofstream o(path);
    o.precision(16);
    for (sctl::Long t = 0; t < Nt; t++)
      for (sctl::Long p = 0; p < Np; p++)
        o << V[(0 * Nt + t) * Np + p] << " " << V[(1 * Nt + t) * Np + p] << " "
          << V[(2 * Nt + t) * Np + p] << "\n";
  };
  write(argv[2], Bext);
  write(argv[3], Bext_eval);
  std::cerr << "biest_driver: wrote " << Nt << "x" << Np
            << " B_ext (both conventions)\n";
  return 0;
}
