// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#include "vmecpp/common/composed_types_lib/composed_types_lib.h"

#include <algorithm>  // min
#include <cmath>      // hypot
#include <sstream>
#include <string>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_split.h"

namespace composed_types {

absl::Status IsVector3dFullyPopulated(const Vector3d& vector,
                                      absl::string_view vector_name) {
  if (!vector.has_x()) {
    std::stringstream error_message;
    error_message << vector_name;
    error_message << " has no x component.";
    return absl::NotFoundError(error_message.str());
  }

  if (!vector.has_y()) {
    std::stringstream error_message;
    error_message << vector_name;
    error_message << " has no y component.";
    return absl::NotFoundError(error_message.str());
  }

  if (!vector.has_z()) {
    std::stringstream error_message;
    error_message << vector_name;
    error_message << " has no z component.";
    return absl::NotFoundError(error_message.str());
  }

  return absl::OkStatus();
}

double Length(const Vector3d& vector) {
  return std::hypot(vector.x(), vector.y(), vector.z());
}  // Length

Vector3d ScaleTo(const Vector3d& vector, double desired_length) {
  const double vector_length = Length(vector);
  const double scaling_factor = desired_length / vector_length;
  Vector3d normalized_vector;
  normalized_vector.set_x(vector.x() * scaling_factor);
  normalized_vector.set_y(vector.y() * scaling_factor);
  normalized_vector.set_z(vector.z() * scaling_factor);
  return normalized_vector;
}  // ScaleTo

Vector3d Normalize(const Vector3d& vector) {
  return ScaleTo(vector, 1.0);
}  // Normalize

Vector3d Add(const Vector3d& vector_1, const Vector3d& vector_2) {
  Vector3d sum;
  sum.set_x(vector_1.x() + vector_2.x());
  sum.set_y(vector_1.y() + vector_2.y());
  sum.set_z(vector_1.z() + vector_2.z());
  return sum;
}  // Add

Vector3d Subtract(const Vector3d& vector_1, const Vector3d& vector_2) {
  Vector3d difference;
  difference.set_x(vector_1.x() - vector_2.x());
  difference.set_y(vector_1.y() - vector_2.y());
  difference.set_z(vector_1.z() - vector_2.z());
  return difference;
}  // Subtract

double DotProduct(const Vector3d& vector_1, const Vector3d& vector_2) {
  double dot_product = 0.0;
  dot_product += vector_1.x() * vector_2.x();
  dot_product += vector_1.y() * vector_2.y();
  dot_product += vector_1.z() * vector_2.z();
  return dot_product;
}  // DotProduct

Vector3d CrossProduct(const Vector3d& vector_1, const Vector3d& vector_2) {
  Vector3d cross_product;
  cross_product.set_x(vector_1.y() * vector_2.z() -
                      vector_1.z() * vector_2.y());
  cross_product.set_y(vector_1.z() * vector_2.x() -
                      vector_1.x() * vector_2.z());
  cross_product.set_z(vector_1.x() * vector_2.y() -
                      vector_1.y() * vector_2.x());
  return cross_product;
}  // CrossProduct

Vector3d MostPerpendicularCoordinateAxis(const Vector3d& axis) {
  // projection of `axis` on coordinate axes are simply the Cartesian components
  // of the vector
  // --> find most-perpendicular axis by finding (first occurence of) min `axis`
  // vector component
  const double min_component = std::min(
      std::min(std::abs(axis.x()), std::abs(axis.y())), std::abs(axis.z()));

  Vector3d most_perpendicular_axis;
  if (min_component == std::abs(axis.x())) {
    most_perpendicular_axis.set_x(1.0);
  } else if (min_component == std::abs(axis.y())) {
    most_perpendicular_axis.set_y(1.0);
  } else {
    most_perpendicular_axis.set_z(1.0);
  }

  return most_perpendicular_axis;
}  // MostPerpendicularCoordinateAxis

std::array<Vector3d, 3> OrthonormalFrameAroundAxis(const Vector3d& axis) {
  std::array<Vector3d, 3> orthonormal_frame;

  // first axis is found by making provided `axis` a unit vector
  orthonormal_frame[0] = ScaleTo(axis, 1.0);

  // Obtain second axis, fully perpendicular to `axis`,
  // by subtracting the projection onto `axis` from the most perpendicular axis.
  // The reasoning is that by using the most perpendicular axis,
  // the least amount of catastrophic cancellation will happen.

  // This already has unit length.
  const Vector3d most_perpendicular_axis =
      MostPerpendicularCoordinateAxis(axis);
  const double axis_dot_most_perp =
      DotProduct(orthonormal_frame[0], most_perpendicular_axis);
  orthonormal_frame[1] =
      Add(most_perpendicular_axis,
          ScaleTo(orthonormal_frame[0], -axis_dot_most_perp));

  // third axis is found from cross product of other two axes
  orthonormal_frame[2] =
      CrossProduct(orthonormal_frame[0], orthonormal_frame[1]);

  return orthonormal_frame;
}  // OrthonormalFrameAroundAxis

absl::Status IsFourierCoefficient1DFullyPopulated(
    const FourierCoefficient1D& fourier_coefficient,
    absl::string_view fourier_coefficient_name) {
  if (!fourier_coefficient.has_fc_cos() && !fourier_coefficient.has_fc_sin()) {
    std::stringstream error_message;
    error_message << fourier_coefficient_name;
    error_message << " has neither a cosine nor a sine coefficient.";
    return absl::NotFoundError(error_message.str());
  }

  if (!fourier_coefficient.has_mode_number()) {
    std::stringstream error_message;
    error_message << fourier_coefficient_name;
    error_message << " has no mode number set.";
    return absl::NotFoundError(error_message.str());
  }

  return absl::OkStatus();
}

absl::Status IsFourierCoefficient2DFullyPopulated(
    const FourierCoefficient2D& fourier_coefficient,
    absl::string_view fourier_coefficient_name) {
  if (!fourier_coefficient.has_fc_cos() && !fourier_coefficient.has_fc_sin()) {
    std::stringstream error_message;
    error_message << fourier_coefficient_name;
    error_message << " has neither a cosine nor a sine coefficient.";
    return absl::NotFoundError(error_message.str());
  }

  if (!fourier_coefficient.has_poloidal_mode_number()) {
    std::stringstream error_message;
    error_message << fourier_coefficient_name;
    error_message << " has no poloidal mode number set.";
    return absl::NotFoundError(error_message.str());
  }

  if (!fourier_coefficient.has_toroidal_mode_number()) {
    std::stringstream error_message;
    error_message << fourier_coefficient_name;
    error_message << " has no toroidal mode number set.";
    return absl::NotFoundError(error_message.str());
  }

  return absl::OkStatus();
}

absl::Status IsCurveRZFourierFullyPopulated(const CurveRZFourier& curve) {
  if (curve.r_size() == 0) {
    return absl::NotFoundError(
        "CurveRZFourier has no FourierCoefficient1D for R");
  }

  if (curve.z_size() == 0) {
    return absl::NotFoundError(
        "CurveRZFourier has no FourierCoefficient1D for Z");
  }

  if (curve.r_size() != curve.z_size()) {
    std::stringstream error_message;
    error_message << "CurveRZFourier has different number of Fourier "
                     "coefficients for R (";
    error_message << curve.r_size();
    error_message << ") and z(";
    error_message << curve.z_size();
    error_message << ")";
    return absl::NotFoundError(error_message.str());
  }

  // now we can assume that r and z have the same non-zero length
  const int rz_size = curve.r_size();
  for (int i = 0; i < rz_size; ++i) {
    absl::Status r_status = IsFourierCoefficient1DFullyPopulated(
        curve.r(i), absl::StrFormat("r[%d]", i));
    if (!r_status.ok()) {
      return r_status;
    }

    absl::Status z_status = IsFourierCoefficient1DFullyPopulated(
        curve.z(i), absl::StrFormat("z[%d]", i));
    if (!z_status.ok()) {
      return z_status;
    }

    if (curve.r(i).mode_number() != curve.z(i).mode_number()) {
      return absl::NotFoundError(absl::StrFormat(
          "found different mode numbers at coefficient %d for r(%d) and z(%d)",
          i, curve.r(i).mode_number(), curve.z(i).mode_number()));
    }
  }

  return absl::OkStatus();
}  // IsCurveRZFourierFullyPopulated

absl::StatusOr<CurveRZFourier> CurveRZFourierFromCsv(
    const std::string& axis_coefficients_csv) {
  CurveRZFourier axis_coefficients;

  std::stringstream axis_coefficients_ss(axis_coefficients_csv);

  // read first line and make sure it contains the expected header
  std::string header_line;
  if (!std::getline(axis_coefficients_ss, header_line)) {
    return absl::InvalidArgumentError("cannot read header line");
  }

  if (header_line != "n,raxis_c,zaxis_s,raxis_s,zaxis_c") {
    return absl::NotFoundError(
        "header line 'n,raxis_c,zaxis_s,raxis_s,zaxis_c' not found");
  }

  // read data lines
  for (std::string raw_line; std::getline(axis_coefficients_ss, raw_line);
       /* no-op */) {
    absl::string_view stripped_line = absl::StripAsciiWhitespace(raw_line);
    std::vector<std::string> line_parts = absl::StrSplit(
        stripped_line, absl::ByAnyChar(","), absl::SkipWhitespace());
    if (line_parts.size() == 5) {
      const int n = std::stoi(line_parts[0]);

      FourierCoefficient1D* raxis = axis_coefficients.add_r();
      raxis->set_mode_number(n);

      FourierCoefficient1D* zaxis = axis_coefficients.add_z();
      zaxis->set_mode_number(n);

      raxis->set_fc_cos(std::stod(line_parts[1]));
      zaxis->set_fc_sin(std::stod(line_parts[2]));
      raxis->set_fc_sin(std::stod(line_parts[3]));
      zaxis->set_fc_cos(std::stod(line_parts[4]));
    } else {
      std::stringstream error_message;
      error_message
          << "cannot parse line: '" << stripped_line << "': has "
          << line_parts.size()
          << " parts, but expect 5: n, raxis_c, zaxis_s, raxis_s, zaxis_c";
      return absl::InvalidArgumentError(error_message.str());
    }
  }

  return axis_coefficients;
}  // ReadAxisCoefficientsFromCsv

absl::StatusOr<std::string> CurveRZFourierToCsv(const CurveRZFourier& axis) {
  // first check that surface is fully populated
  absl::Status status = IsCurveRZFourierFullyPopulated(axis);
  if (!status.ok()) {
    return status;
  }

  std::stringstream ss;

  // write header line
  ss << "n,raxis_c,zaxis_s,raxis_s,zaxis_c\n";

  // write all Fourier coefficients
  const int rz_size = axis.r_size();
  for (int i = 0; i < rz_size; ++i) {
    FourierCoefficient1D r = axis.r(i);
    FourierCoefficient1D z = axis.z(i);
    const int n = r.mode_number();
    ss << n << ",";
    ss << std::setprecision(16) << r.fc_cos() << ",";
    ss << std::setprecision(16) << z.fc_sin() << ",";
    ss << std::setprecision(16) << r.fc_sin() << ",";
    ss << std::setprecision(16) << z.fc_cos() << "\n";
  }

  return ss.str();
}  // CurveRZFourierToCsv

absl::StatusOr<std::vector<int>> ModeNumbers(const CurveRZFourier& curve) {
  absl::Status is_fully_populated = IsCurveRZFourierFullyPopulated(curve);
  if (!is_fully_populated.ok()) {
    return is_fully_populated;
  }
  int rz_size = curve.r_size();
  std::vector<int> mode_numbers(rz_size);
  for (int i = 0; i < rz_size; ++i) {
    mode_numbers[i] = curve.r(i).mode_number();
  }
  return mode_numbers;
}

absl::StatusOr<std::vector<double>> CoefficientsRCos(
    const CurveRZFourier& curve) {
  absl::Status is_fully_populated = IsCurveRZFourierFullyPopulated(curve);
  if (!is_fully_populated.ok()) {
    return is_fully_populated;
  }
  int rz_size = curve.r_size();
  std::vector<double> r_cos(rz_size);
  for (int i = 0; i < rz_size; ++i) {
    r_cos[i] = curve.r(i).fc_cos();
  }
  return r_cos;
}

absl::StatusOr<std::vector<double>> CoefficientsZSin(
    const CurveRZFourier& curve) {
  absl::Status is_fully_populated = IsCurveRZFourierFullyPopulated(curve);
  if (!is_fully_populated.ok()) {
    return is_fully_populated;
  }
  int rz_size = curve.r_size();
  std::vector<double> z_sin(rz_size);
  for (int i = 0; i < rz_size; ++i) {
    z_sin[i] = curve.z(i).fc_sin();
  }
  return z_sin;
}

absl::StatusOr<std::vector<double>> CoefficientsRSin(
    const CurveRZFourier& curve) {
  absl::Status is_fully_populated = IsCurveRZFourierFullyPopulated(curve);
  if (!is_fully_populated.ok()) {
    return is_fully_populated;
  }
  int rz_size = curve.r_size();
  std::vector<double> r_sin(rz_size);
  for (int i = 0; i < rz_size; ++i) {
    r_sin[i] = curve.r(i).fc_sin();
  }
  return r_sin;
}

absl::StatusOr<std::vector<double>> CoefficientsZCos(
    const CurveRZFourier& curve) {
  absl::Status is_fully_populated = IsCurveRZFourierFullyPopulated(curve);
  if (!is_fully_populated.ok()) {
    return is_fully_populated;
  }
  int rz_size = curve.r_size();
  std::vector<double> z_cos(rz_size);
  for (int i = 0; i < rz_size; ++i) {
    z_cos[i] = curve.z(i).fc_cos();
  }
  return z_cos;
}

absl::Status IsSurfaceRZFourierFullyPopulated(const SurfaceRZFourier& surface) {
  if (surface.r_size() == 0) {
    return absl::NotFoundError(
        "SurfaceRZFourier has no FourierCoefficient1D for R");
  }

  if (surface.z_size() == 0) {
    return absl::NotFoundError(
        "SurfaceRZFourier has no FourierCoefficient1D for Z");
  }

  if (surface.r_size() != surface.z_size()) {
    std::stringstream error_message;
    error_message << "SurfaceRZFourier has different number of Fourier "
                     "coefficients for R (";
    error_message << surface.r_size();
    error_message << ") and z(";
    error_message << surface.z_size();
    error_message << ")";
    return absl::NotFoundError(error_message.str());
  }

  // now we can assume that r and z have the same non-zero length
  const int rz_size = surface.r_size();
  for (int i = 0; i < rz_size; ++i) {
    absl::Status r_status = IsFourierCoefficient2DFullyPopulated(
        surface.r(i), absl::StrFormat("r[%d]", i));
    if (!r_status.ok()) {
      return r_status;
    }

    absl::Status z_status = IsFourierCoefficient2DFullyPopulated(
        surface.z(i), absl::StrFormat("z[%d]", i));
    if (!z_status.ok()) {
      return z_status;
    }

    if (surface.r(i).poloidal_mode_number() !=
        surface.z(i).poloidal_mode_number()) {
      return absl::NotFoundError(
          absl::StrFormat("found different poloidal mode numbers at "
                          "coefficient %d for r(%d) and z(%d)",
                          i, surface.r(i).poloidal_mode_number(),
                          surface.z(i).poloidal_mode_number()));
    }

    if (surface.r(i).toroidal_mode_number() !=
        surface.z(i).toroidal_mode_number()) {
      return absl::NotFoundError(
          absl::StrFormat("found different toroidal mode numbers at "
                          "coefficient %d for r(%d) and z(%d)",
                          i, surface.r(i).toroidal_mode_number(),
                          surface.z(i).toroidal_mode_number()));
    }
  }

  return absl::OkStatus();
}  // IsSurfaceRZFourierFullyPopulated

absl::StatusOr<SurfaceRZFourier> SurfaceRZFourierFromCsv(
    const std::string& boundary_coefficients_csv) {
  SurfaceRZFourier boundary_coefficients;

  std::stringstream boundary_coefficients_ss(boundary_coefficients_csv);

  // read first line and make sure it contains the expected header
  std::string header_line;
  if (!std::getline(boundary_coefficients_ss, header_line)) {
    return absl::InvalidArgumentError("cannot read header line");
  }

  if (header_line != "n,m,rbc,zbs,rbs,zbc") {
    return absl::NotFoundError("header line 'n,m,rbc,zbs,rbs,zbc' not found");
  }

  // read data lines
  for (std::string raw_line; std::getline(boundary_coefficients_ss, raw_line);
       /* no-op */) {
    absl::string_view stripped_line = absl::StripAsciiWhitespace(raw_line);
    std::vector<std::string> line_parts = absl::StrSplit(
        stripped_line, absl::ByAnyChar(","), absl::SkipWhitespace());
    if (line_parts.size() == 6) {
      const int n = std::stoi(line_parts[0]);
      const int m = std::stoi(line_parts[1]);

      FourierCoefficient2D* rbc_rbs = boundary_coefficients.add_r();
      rbc_rbs->set_poloidal_mode_number(m);
      rbc_rbs->set_toroidal_mode_number(n);

      FourierCoefficient2D* zbs_zbc = boundary_coefficients.add_z();
      zbs_zbc->set_poloidal_mode_number(m);
      zbs_zbc->set_toroidal_mode_number(n);

      rbc_rbs->set_fc_cos(std::stod(line_parts[2]));
      zbs_zbc->set_fc_sin(std::stod(line_parts[3]));
      rbc_rbs->set_fc_sin(std::stod(line_parts[4]));
      zbs_zbc->set_fc_cos(std::stod(line_parts[5]));
    } else {
      std::stringstream error_message;
      error_message << "cannot parse line: '" << stripped_line << "': has "
                    << line_parts.size()
                    << " parts, but expect 6: n, m, rbc, zbs, rbs, zbc";
      return absl::InvalidArgumentError(error_message.str());
    }
  }

  return boundary_coefficients;
}  // ReadBoundaryCoefficientsFromCsv

absl::StatusOr<std::string> SurfaceRZFourierToCsv(
    const SurfaceRZFourier& surface) {
  // first check that surface is fully populated
  absl::Status status = IsSurfaceRZFourierFullyPopulated(surface);
  if (!status.ok()) {
    return status;
  }

  std::stringstream ss;

  // write header line
  ss << "n,m,rbc,zbs,rbs,zbc\n";

  // write all Fourier coefficients
  const int rz_size = surface.r_size();
  for (int i = 0; i < rz_size; ++i) {
    const FourierCoefficient2D& r = surface.r(i);
    const FourierCoefficient2D& z = surface.z(i);

    const int m = r.poloidal_mode_number();
    const int n = r.toroidal_mode_number();

    ss << n << "," << m << ",";
    ss << std::setprecision(16) << r.fc_cos() << ",";
    ss << std::setprecision(16) << z.fc_sin() << ",";
    ss << std::setprecision(16) << r.fc_sin() << ",";
    ss << std::setprecision(16) << z.fc_cos() << "\n";
  }

  return ss.str();
}  // SurfaceRZFourierToCsv

absl::StatusOr<std::vector<int>> PoloidalModeNumbers(
    const SurfaceRZFourier& surface) {
  absl::Status is_fully_populated = IsSurfaceRZFourierFullyPopulated(surface);
  if (!is_fully_populated.ok()) {
    return is_fully_populated;
  }
  int rz_size = surface.r_size();
  std::vector<int> poloidal_mode_numbers(rz_size);
  for (int i = 0; i < rz_size; ++i) {
    poloidal_mode_numbers[i] = surface.r(i).poloidal_mode_number();
  }
  return poloidal_mode_numbers;
}

absl::StatusOr<std::vector<int>> ToroidalModeNumbers(
    const SurfaceRZFourier& surface) {
  absl::Status is_fully_populated = IsSurfaceRZFourierFullyPopulated(surface);
  if (!is_fully_populated.ok()) {
    return is_fully_populated;
  }
  int rz_size = surface.r_size();
  std::vector<int> toroidal_mode_numbers(rz_size);
  for (int i = 0; i < rz_size; ++i) {
    toroidal_mode_numbers[i] = surface.r(i).toroidal_mode_number();
  }
  return toroidal_mode_numbers;
}

absl::StatusOr<std::vector<double>> CoefficientsRCos(
    const SurfaceRZFourier& surface) {
  absl::Status is_fully_populated = IsSurfaceRZFourierFullyPopulated(surface);
  if (!is_fully_populated.ok()) {
    return is_fully_populated;
  }
  int rz_size = surface.r_size();
  std::vector<double> r_cos(rz_size);
  for (int i = 0; i < rz_size; ++i) {
    r_cos[i] = surface.r(i).fc_cos();
  }
  return r_cos;
}

absl::StatusOr<std::vector<double>> CoefficientsZSin(
    const SurfaceRZFourier& surface) {
  absl::Status is_fully_populated = IsSurfaceRZFourierFullyPopulated(surface);
  if (!is_fully_populated.ok()) {
    return is_fully_populated;
  }
  int rz_size = surface.r_size();
  std::vector<double> z_sin(rz_size);
  for (int i = 0; i < rz_size; ++i) {
    z_sin[i] = surface.z(i).fc_sin();
  }
  return z_sin;
}

absl::StatusOr<std::vector<double>> CoefficientsRSin(
    const SurfaceRZFourier& surface) {
  absl::Status is_fully_populated = IsSurfaceRZFourierFullyPopulated(surface);
  if (!is_fully_populated.ok()) {
    return is_fully_populated;
  }
  int rz_size = surface.r_size();
  std::vector<double> r_sin(rz_size);
  for (int i = 0; i < rz_size; ++i) {
    r_sin[i] = surface.r(i).fc_sin();
  }
  return r_sin;
}

absl::StatusOr<std::vector<double>> CoefficientsZCos(
    const SurfaceRZFourier& surface) {
  absl::Status is_fully_populated = IsSurfaceRZFourierFullyPopulated(surface);
  if (!is_fully_populated.ok()) {
    return is_fully_populated;
  }
  int rz_size = surface.r_size();
  std::vector<double> z_cos(rz_size);
  for (int i = 0; i < rz_size; ++i) {
    z_cos[i] = surface.z(i).fc_cos();
  }
  return z_cos;
}

}  // namespace composed_types
