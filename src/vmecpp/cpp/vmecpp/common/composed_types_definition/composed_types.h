#ifndef VMECPP_COMMON_COMPOSED_TYPES_DEFINITION_COMPOSED_TYPES_H_
#define VMECPP_COMMON_COMPOSED_TYPES_DEFINITION_COMPOSED_TYPES_H_

#include <list>

namespace composed_types {

struct Vector3d {
  // Cartesian x component
  bool has_x_ = false;
  double x_ = 0.0;

  // Cartesian y component
  bool has_y_ = false;
  double y_ = 0.0;

  // Cartesian z component
  bool has_z_ = false;
  double z_ = 0.0;

  // x
  bool has_x() const { return has_x_; }
  double x() const { return x_; }
  void set_x(double value) {
    x_ = value;
    has_x_ = true;
  }
  void clear_x() {
    x_ = 0.0;
    has_x_ = false;
  }

  // y
  bool has_y() const { return has_y_; }
  double y() const { return y_; }
  void set_y(double value) {
    y_ = value;
    has_y_ = true;
  }
  void clear_y() {
    y_ = 0.0;
    has_y_ = false;
  }

  // z
  bool has_z() const { return has_z_; }
  double z() const { return z_; }
  void set_z(double value) {
    z_ = value;
    has_z_ = true;
  }
  void clear_z() {
    z_ = 0.0;
    has_z_ = false;
  }

  // Clear the entire structure
  void Clear() {
    clear_x();
    clear_y();
    clear_z();
  }

  // Copies all fields (including presence flags) from another Vector3d
  void CopyFrom(const Vector3d& other) {
    has_x_ = other.has_x_;
    x_ = other.x_;

    has_y_ = other.has_y_;
    y_ = other.y_;

    has_z_ = other.has_z_;
    z_ = other.z_;
  }
};  // Vector3d

struct FourierCoefficient1D {
  // Fourier coefficients for cosine basis
  bool has_fc_cos_ = false;
  double fc_cos_ = 0.0;

  // Fourier coefficients for sine basis
  bool has_fc_sin_ = false;
  double fc_sin_ = 0.0;

  // mode number
  bool has_mode_number_ = false;
  int mode_number_ = 0;

  // fc_cos
  bool has_fc_cos() const { return has_fc_cos_; }
  double fc_cos() const { return fc_cos_; }
  void set_fc_cos(double value) {
    fc_cos_ = value;
    has_fc_cos_ = true;
  }
  void clear_fc_cos() {
    fc_cos_ = 0.0;
    has_fc_cos_ = false;
  }

  // fc_sin
  bool has_fc_sin() const { return has_fc_sin_; }
  double fc_sin() const { return fc_sin_; }
  void set_fc_sin(double value) {
    fc_sin_ = value;
    has_fc_sin_ = true;
  }
  void clear_fc_sin() {
    fc_sin_ = 0.0;
    has_fc_sin_ = false;
  }

  // mode_number
  bool has_mode_number() const { return has_mode_number_; }
  int mode_number() const { return mode_number_; }
  void set_mode_number(int value) {
    mode_number_ = value;
    has_mode_number_ = true;
  }
  void clear_mode_number() {
    mode_number_ = 0;
    has_mode_number_ = false;
  }

  // Clear the entire structure
  void Clear() {
    clear_fc_cos();
    clear_fc_sin();
    clear_mode_number();
  }
};  // FourierCoefficient1D

struct FourierCoefficient2D {
  // Fourier coefficients for cosine basis
  bool has_fc_cos_ = false;
  double fc_cos_ = 0.0;

  // Fourier coefficients for sine basis
  bool has_fc_sin_ = false;
  double fc_sin_ = 0.0;

  // poloidal mode number (typically called m)
  bool has_poloidal_mode_number_ = false;
  int poloidal_mode_number_ = 0;

  // toroidal mode number (typically called n)
  bool has_toroidal_mode_number_ = false;
  int toroidal_mode_number_ = 0;

  // fc_cos
  bool has_fc_cos() const { return has_fc_cos_; }
  double fc_cos() const { return fc_cos_; }
  void set_fc_cos(double value) {
    fc_cos_ = value;
    has_fc_cos_ = true;
  }
  void clear_fc_cos() {
    fc_cos_ = 0.0;
    has_fc_cos_ = false;
  }

  // fc_sin
  bool has_fc_sin() const { return has_fc_sin_; }
  double fc_sin() const { return fc_sin_; }
  void set_fc_sin(double value) {
    fc_sin_ = value;
    has_fc_sin_ = true;
  }
  void clear_fc_sin() {
    fc_sin_ = 0.0;
    has_fc_sin_ = false;
  }

  // poloidal_mode_number
  bool has_poloidal_mode_number() const { return has_poloidal_mode_number_; }
  int poloidal_mode_number() const { return poloidal_mode_number_; }
  void set_poloidal_mode_number(int value) {
    poloidal_mode_number_ = value;
    has_poloidal_mode_number_ = true;
  }
  void clear_poloidal_mode_number() {
    poloidal_mode_number_ = 0;
    has_poloidal_mode_number_ = false;
  }

  // toroidal_mode_number
  bool has_toroidal_mode_number() const { return has_toroidal_mode_number_; }
  int toroidal_mode_number() const { return toroidal_mode_number_; }
  void set_toroidal_mode_number(int value) {
    toroidal_mode_number_ = value;
    has_toroidal_mode_number_ = true;
  }
  void clear_toroidal_mode_number() {
    toroidal_mode_number_ = 0;
    has_toroidal_mode_number_ = false;
  }

  // Clear the entire structure
  void Clear() {
    clear_fc_cos();
    clear_fc_sin();
    clear_poloidal_mode_number();
    clear_toroidal_mode_number();
  }
};  // FourierCoefficient2D

struct CurveRZFourier {
  // Fourier coefficients for R
  std::list<composed_types::FourierCoefficient1D> r_;

  // Fourier coefficients for Z
  std::list<composed_types::FourierCoefficient1D> z_;

  // r
  int r_size() const { return static_cast<int>(r_.size()); }
  const composed_types::FourierCoefficient1D& r(int index) const {
    auto it = r_.cbegin();
    std::advance(it, index);  // no explicit bounds-check for brevity
    return *it;
  }
  composed_types::FourierCoefficient1D* mutable_r(int index) {
    auto it = r_.begin();
    std::advance(it, index);
    return &(*it);
  }
  composed_types::FourierCoefficient1D* add_r() {
    r_.emplace_back();
    auto it = r_.end();
    --it;
    return &(*it);
  }
  const std::list<composed_types::FourierCoefficient1D>& r() const {
    return r_;
  }
  std::list<composed_types::FourierCoefficient1D>* mutable_r() { return &r_; }
  void clear_r() { r_.clear(); }

  // z
  int z_size() const { return static_cast<int>(z_.size()); }
  const composed_types::FourierCoefficient1D& z(int index) const {
    auto it = z_.cbegin();
    std::advance(it, index);
    return *it;
  }
  composed_types::FourierCoefficient1D* mutable_z(int index) {
    auto it = z_.begin();
    std::advance(it, index);
    return &(*it);
  }
  composed_types::FourierCoefficient1D* add_z() {
    z_.emplace_back();
    auto it = z_.end();
    --it;
    return &(*it);
  }
  const std::list<composed_types::FourierCoefficient1D>& z() const {
    return z_;
  }
  std::list<composed_types::FourierCoefficient1D>* mutable_z() { return &z_; }
  void clear_z() { z_.clear(); }

  // Clear the entire structure
  void Clear() {
    clear_r();
    clear_z();
  }
};  // CurveRZFourier

struct SurfaceRZFourier {
  // Fourier coefficients for R
  std::list<composed_types::FourierCoefficient2D> r_;

  // Fourier coefficients for Z
  std::list<composed_types::FourierCoefficient2D> z_;

  // r
  int r_size() const { return static_cast<int>(r_.size()); }
  const composed_types::FourierCoefficient2D& r(int index) const {
    auto it = r_.cbegin();
    std::advance(it, index);  // no explicit bounds-check for brevity
    return *it;
  }
  composed_types::FourierCoefficient2D* mutable_r(int index) {
    auto it = r_.begin();
    std::advance(it, index);
    return &(*it);
  }
  composed_types::FourierCoefficient2D* add_r() {
    r_.emplace_back();
    auto it = r_.end();
    --it;
    return &(*it);
  }
  const std::list<composed_types::FourierCoefficient2D>& r() const {
    return r_;
  }
  std::list<composed_types::FourierCoefficient2D>* mutable_r() { return &r_; }
  void clear_r() { r_.clear(); }

  // z
  int z_size() const { return static_cast<int>(z_.size()); }
  const composed_types::FourierCoefficient2D& z(int index) const {
    auto it = z_.cbegin();
    std::advance(it, index);
    return *it;
  }
  composed_types::FourierCoefficient2D* mutable_z(int index) {
    auto it = z_.begin();
    std::advance(it, index);
    return &(*it);
  }
  composed_types::FourierCoefficient2D* add_z() {
    z_.emplace_back();
    auto it = z_.end();
    --it;
    return &(*it);
  }
  const std::list<composed_types::FourierCoefficient2D>& z() const {
    return z_;
  }
  std::list<composed_types::FourierCoefficient2D>* mutable_z() { return &z_; }
  void clear_z() { z_.clear(); }

  // Clear the entire structure
  void Clear() {
    clear_r();
    clear_z();
  }
};  // SurfaceRZFourier

}  // namespace composed_types

#endif  // VMECPP_COMMON_COMPOSED_TYPES_DEFINITION_COMPOSED_TYPES_H_
