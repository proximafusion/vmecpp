#ifndef VMECPP_COMMON_COMPOSED_TYPES_DEFINITION_COMPOSED_TYPES_H_
#define VMECPP_COMMON_COMPOSED_TYPES_DEFINITION_COMPOSED_TYPES_H_

// FIXME(jons): to be removed in the end
#include "vmecpp/common/composed_types_definition/composed_types.pb.h"

namespace composed_types {

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
}; // FourierCoefficient2D

struct CurveRZFourier {
  // Fourier coefficients for R
  std::list<composed_types::FourierCoefficient1D> r_;

  // Fourier coefficients for Z
  std::list<composed_types::FourierCoefficient1D> z_;

  // r
  int r_size() const {
    return static_cast<int>(r_.size());
  }
  const composed_types::FourierCoefficient1D& r(int index) const {
    auto it = r_.cbegin();
    std::advance(it, index); // no explicit bounds-check for brevity
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
  std::list<composed_types::FourierCoefficient1D>* mutable_r() {
    return &r_;
  }
  void clear_r() {
    r_.clear();
  }

  // z
  int z_size() const {
    return static_cast<int>(z_.size());
  }
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
  std::list<composed_types::FourierCoefficient1D>* mutable_z() {
    return &z_;
  }
  void clear_z() {
    z_.clear();
  }

  // Clear the entire structure
  void Clear() {
    clear_r();
    clear_z();
  }
}; // CurveRZFourier

struct SurfaceRZFourier {
  // Fourier coefficients for R
  std::list<composed_types::FourierCoefficient2D> r_;

  // Fourier coefficients for Z
  std::list<composed_types::FourierCoefficient2D> z_;

  // r
  int r_size() const {
    return static_cast<int>(r_.size());
  }
  const composed_types::FourierCoefficient2D& r(int index) const {
    auto it = r_.cbegin();
    std::advance(it, index); // no explicit bounds-check for brevity
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
  std::list<composed_types::FourierCoefficient2D>* mutable_r() {
    return &r_;
  }
  void clear_r() {
    r_.clear();
  }

  // z
  int z_size() const {
    return static_cast<int>(z_.size());
  }
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
  std::list<composed_types::FourierCoefficient2D>* mutable_z() {
    return &z_;
  }
  void clear_z() {
    z_.clear();
  }

  // Clear the entire structure
  void Clear() {
    clear_r();
    clear_z();
  }
}; // SurfaceRZFourier

} // namespace composed_types

#endif // VMECPP_COMMON_COMPOSED_TYPES_DEFINITION_COMPOSED_TYPES_H_
