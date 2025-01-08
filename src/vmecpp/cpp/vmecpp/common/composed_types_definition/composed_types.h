#ifndef VMECPP_COMMON_COMPOSED_TYPES_DEFINITION_COMPOSED_TYPES_H_
#define VMECPP_COMMON_COMPOSED_TYPES_DEFINITION_COMPOSED_TYPES_H_

// FIXME(jons): to be removed in the end
#include "vmecpp/common/composed_types_definition/composed_types.pb.h"

namespace composed_types {

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
