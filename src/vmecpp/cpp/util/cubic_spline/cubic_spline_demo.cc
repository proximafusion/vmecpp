#include <charconv>
#include <iomanip>
#include <sstream>

#include "absl/log/check.h"
#include "util/cubic_spline/cubic_spline.h"
#include "util/file_io/file_io.h"

int main(int argc, char** argv) {
  using cubic_spline::CubicSpline;

  // generate data to be interpolated
  int n = 15;
  Eigen::VectorXd knots(n);
  Eigen::VectorXd values(n);
  for (int i = 0; i < n; ++i) {
    // knots[i] = i * 2.0 * M_PI / (n - 1);
    knots[i] = i * M_PI / (n - 1);
    values[i] = std::sin(knots[i]);
  }

  // build spline
  absl::StatusOr<CubicSpline> maybe_s = CubicSpline::Create(
      knots, values, CubicSpline::Boundary(CubicSpline::BcType::kPeriodic),
      CubicSpline::Boundary(CubicSpline::BcType::kPeriodic)
      // CubicSpline::Boundary(CubicSpline::BcType::kSymmetric),
      // CubicSpline::Boundary(CubicSpline::BcType::kPeriodic)
  );
  CHECK_OK(maybe_s);
  CubicSpline& s = maybe_s.value();

  // evaluate spline at a fine grid
  int n_eval = 1501;
  Eigen::VectorXd x_eval(n_eval);
  Eigen::VectorXd y_eval(n_eval);
  for (int i = 0; i < n_eval; ++i) {
    x_eval[i] = i * knots[n - 1] / (n_eval - 1);
    auto maybe_y = s.Evaluate(x_eval[i]);
    CHECK_OK(maybe_y);
    y_eval[i] = maybe_y.value();
  }

  // write original data into a first text file (for loading into Python via
  // numpy.loadtxt)
  std::stringstream original_data;
  for (int i = 0; i < n; ++i) {
    original_data << std::setprecision(17) << knots[i] << " " << values[i]
                  << "\n";
  }
  absl::Status status =
      file_io::WriteFile("original_data.dat", original_data.str());
  CHECK_OK(status);

  // write interpolated data into a second text file (for loading into Python
  // via numpy.loadtxt)
  std::stringstream interpolated_data;
  for (int i = 0; i < n_eval; ++i) {
    interpolated_data << std::setprecision(17) << x_eval[i] << " " << y_eval[i]
                      << "\n";
  }
  status = file_io::WriteFile("interpolated_data.dat", interpolated_data.str());
  CHECK_OK(status);

  return 0;
}
