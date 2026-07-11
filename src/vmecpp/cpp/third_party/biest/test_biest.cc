#include <biest.hpp>
#include <filesystem>
#include <sctl.hpp>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

// This is a googletest-converted version of the following method:
// biest::ExtVacuumFieldTest<double>::test(digits, NFP, surf_Nt, surf_Np,
// surf_type, Nt, Np) The off-surface evaluation test part has been removed in
// order to speed up the test.

using Real = double;

namespace {

// The prebuilt surface geometries (SurfType::W7X etc.) are read from
// "geom/..." relative to the working directory; chdir into the @biest
// runfiles tree where the geom filegroup is staged.
void ChdirToBiestRunfiles() {
  const char* srcdir = std::getenv("TEST_SRCDIR");
  if (srcdir == nullptr) {
    return;
  }
  for (const auto& entry : std::filesystem::directory_iterator(srcdir)) {
    if (std::filesystem::exists(entry.path() / "geom")) {
      std::filesystem::current_path(entry.path());
      return;
    }
  }
}

}  // namespace

TEST(TestBiest, CheckExtVacuumFieldOnSurface) {
  ChdirToBiestRunfiles();
  int digits = 3;
  int NFP = 5;
  long surf_Nt = 20;
  long surf_Np = 20;
  biest::SurfType surf_type = biest::SurfType::W7X;
  long Nt = 20;
  long Np = 20;

  // Construct the surface
  std::vector<Real> X(3 * surf_Nt * surf_Np), X_offsurf;
  {  // Set X_offsurf
    biest::Surface<Real> S(NFP * Nt, Np, biest::SurfType::AxisymCircleWide);
    X_offsurf.assign(S.Coord().begin(), S.Coord().end());
  }
  X = biest::ExtVacuumFieldTest<Real>::SurfaceCoordinates(NFP, surf_Nt, surf_Np,
                                                          surf_type);

  // Generate B field for testing exterior vacuum fields
  std::vector<Real> B, B_pts;
  std::tie(B, B_pts) = biest::ExtVacuumFieldTest<Real>::BFieldData(
      NFP, surf_Nt, surf_Np, X, Nt, Np, X_offsurf);

  // Setup
  biest::ExtVacuumField<Real> vacuum_field;
  vacuum_field.Setup(digits, NFP, surf_Nt, surf_Np, X, Nt, Np);

  // Compute Bplasma field such that Bplasma.n = -BdotN
  std::vector<Real> Bplasma, sigma, J;
  const auto BdotN = vacuum_field.ComputeBdotN(B);
  std::tie(Bplasma, sigma, J) = vacuum_field.ComputeBplasma(BdotN, (Real)-1);

  // evaluate error in infinity norm
  std::vector<Real> Berr = B;
  for (long i = 0; i < (long)Berr.size(); i++) {
    Berr[i] += Bplasma[i];
  }

  Real max_val = 0;
  for (const auto& x : B) {
    max_val = std::max<Real>(max_val, fabs(x));
  }

  Real max_err = 0;
  for (const auto& x : Berr) {
    max_err = std::max<Real>(max_err, fabs(x));
  }

  EXPECT_LT(max_err / max_val, 0.2)
      << "Maximum relative error: " << max_err / max_val;

}  // CheckExtVacuumFieldOnSurface
