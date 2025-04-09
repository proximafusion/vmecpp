#ifndef VMECPP_COMMON_MAGNETIC_CONFIGURATION_DEFINITION_MAGNETIC_CONFIGURATION_H_
#define VMECPP_COMMON_MAGNETIC_CONFIGURATION_DEFINITION_MAGNETIC_CONFIGURATION_H_

#include <Eigen/Dense>
#include <cstdint>
#include <list>
#include <memory>  // For std::addressof, std::construct_at
#include <optional>
#include <utility>

#include "vmecpp/common/composed_types_definition/composed_types.h"

namespace magnetics {

struct PolygonFilament {
  // a human-readable name, e.g., for plotting
  std::optional<std::string> name;

  // Cartesian components of filament geometry (struct of vectors layout)
  Eigen::Matrix<double, 3, Eigen::Dynamic, Eigen::RowMajor> vertices;

  // Clear the entire structure
  void Clear() {
    name.reset();
    vertices = Eigen::Matrix<double, 3, 0, Eigen::RowMajor>::Zero();
  }
};  // PolygonFilament

struct CircularFilament {
  // a human-readable name, e.g., for plotting
  std::optional<std::string> name;

  // Cartesian coordinates of the center point of the loop
  composed_types::Vector3d center;

  // Cartesian components of a vector pointing along the normal of the circle
  // around which the current flows
  composed_types::Vector3d normal;

  // radius of the loop
  double radius;

  // Clear the entire structure
  void Clear() { name.reset(); }
};  // CircularFilament

struct InfiniteStraightFilament {
  // a human-readable name, e.g., for plotting
  std::optional<std::string> name;

  // Cartesian coordinates of a point on the filament
  std::optional<composed_types::Vector3d> origin_;

  // Cartesian components of the direction along the filament
  std::optional<composed_types::Vector3d> direction_;

  bool has_origin() const { return origin_.has_value(); }
  const composed_types::Vector3d& origin() const { return origin_.value(); }
  void set_origin(const composed_types::Vector3d& value) { origin_ = value; }

  bool has_direction() const { return direction_.has_value(); }
  const composed_types::Vector3d& direction() const {
    return direction_.value();
  }
  void set_direction(const composed_types::Vector3d& value) {
    direction_ = value;
  }

  // Clear the entire structure
  void Clear() {
    name.reset();
    origin_.reset();
    direction_.reset();
  }
};  // InfiniteStraightFilament

struct CurrentCarrier {
  // oneof type
  enum TypeCase : std::uint8_t {
    kInfiniteStraightFilament = 1,
    kCircularFilament = 2,
    kPolygonFilament = 3,
    kFourierFilament = 4,
    kTypeNotSet = 0
  };

 private:
  TypeCase type_case_ = kTypeNotSet;

  union {
    InfiniteStraightFilament infinite_straight_filament_;
    CircularFilament circular_filament_;
    PolygonFilament polygon_filament_;
  };

 public:
  CurrentCarrier() : type_case_(kTypeNotSet) {}

  ~CurrentCarrier() { Clear(); }

  // Copy constructor
  CurrentCarrier(const CurrentCarrier& other) : type_case_(kTypeNotSet) {
    switch (other.type_case_) {
      case kInfiniteStraightFilament: {
        type_case_ = kInfiniteStraightFilament;
        std::construct_at(std::addressof(infinite_straight_filament_),
                          other.infinite_straight_filament_);
      } break;
      case kCircularFilament: {
        type_case_ = kCircularFilament;
        std::construct_at(std::addressof(circular_filament_),
                          other.circular_filament_);
      } break;
      case kPolygonFilament: {
        type_case_ = kPolygonFilament;
        std::construct_at(std::addressof(polygon_filament_),
                          other.polygon_filament_);
      } break;
      default:
        type_case_ = kTypeNotSet;
        break;
    }
  }

  // Move constructor
  CurrentCarrier(CurrentCarrier&& other) noexcept : type_case_(kTypeNotSet) {
    switch (other.type_case_) {
      case kInfiniteStraightFilament: {
        type_case_ = kInfiniteStraightFilament;
        std::construct_at(std::addressof(infinite_straight_filament_),
                          std::move(other.infinite_straight_filament_));
      } break;
      case kCircularFilament: {
        type_case_ = kCircularFilament;
        std::construct_at(std::addressof(circular_filament_),
                          std::move(other.circular_filament_));
      } break;
      case kPolygonFilament: {
        type_case_ = kPolygonFilament;
        std::construct_at(std::addressof(polygon_filament_),
                          std::move(other.polygon_filament_));
      } break;
      default:
        type_case_ = kTypeNotSet;
        break;
    }
    other.Clear();
  }

  // Copy assignment
  CurrentCarrier& operator=(const CurrentCarrier& other) {
    if (this != &other) {
      Clear();
      switch (other.type_case_) {
        case kInfiniteStraightFilament: {
          type_case_ = kInfiniteStraightFilament;
          std::construct_at(std::addressof(infinite_straight_filament_),
                            other.infinite_straight_filament_);
        } break;
        case kCircularFilament: {
          type_case_ = kCircularFilament;
          std::construct_at(std::addressof(circular_filament_),
                            other.circular_filament_);
        } break;
        case kPolygonFilament: {
          type_case_ = kPolygonFilament;
          std::construct_at(std::addressof(polygon_filament_),
                            other.polygon_filament_);
        } break;
        default:
          type_case_ = kTypeNotSet;
          break;
      }
    }
    return *this;
  }

  // Move assignment
  CurrentCarrier& operator=(CurrentCarrier&& other) noexcept {
    if (this != &other) {
      Clear();
      switch (other.type_case_) {
        case kInfiniteStraightFilament: {
          type_case_ = kInfiniteStraightFilament;
          std::construct_at(std::addressof(infinite_straight_filament_),
                            std::move(other.infinite_straight_filament_));
        } break;
        case kCircularFilament: {
          type_case_ = kCircularFilament;
          std::construct_at(std::addressof(circular_filament_),
                            std::move(other.circular_filament_));
        } break;
        case kPolygonFilament: {
          type_case_ = kPolygonFilament;
          std::construct_at(std::addressof(polygon_filament_),
                            std::move(other.polygon_filament_));
        } break;
        default:
          type_case_ = kTypeNotSet;
          break;
      }
      other.Clear();
    }
    return *this;
  }

  void Clear() {
    switch (type_case_) {
      case kInfiniteStraightFilament:
        infinite_straight_filament_.~InfiniteStraightFilament();
        break;
      case kCircularFilament:
        circular_filament_.~CircularFilament();
        break;
      case kPolygonFilament:
        polygon_filament_.~PolygonFilament();
        break;
      default:
        break;
    }
    type_case_ = kTypeNotSet;
  }

  // InfiniteStraightFilament
  bool has_infinite_straight_filament() const {
    return type_case_ == kInfiniteStraightFilament;
  }
  const InfiniteStraightFilament& infinite_straight_filament() const {
    return infinite_straight_filament_;
  }
  InfiniteStraightFilament* mutable_infinite_straight_filament() {
    if (type_case_ != kInfiniteStraightFilament) {
      Clear();
      type_case_ = kInfiniteStraightFilament;
      std::construct_at(std::addressof(infinite_straight_filament_));
    }
    return &infinite_straight_filament_;
  }
  void set_infinite_straight_filament(const InfiniteStraightFilament& value) {
    Clear();
    type_case_ = kInfiniteStraightFilament;
    std::construct_at(std::addressof(infinite_straight_filament_), value);
  }

  // CircularFilament
  bool has_circular_filament() const { return type_case_ == kCircularFilament; }
  const CircularFilament& circular_filament() const {
    return circular_filament_;
  }
  CircularFilament* mutable_circular_filament() {
    if (type_case_ != kCircularFilament) {
      Clear();
      type_case_ = kCircularFilament;
      std::construct_at(std::addressof(circular_filament_));
    }
    return &circular_filament_;
  }
  void set_circular_filament(const CircularFilament& value) {
    Clear();
    type_case_ = kCircularFilament;
    std::construct_at(std::addressof(circular_filament_), value);
  }

  // PolygonFilament
  bool has_polygon_filament() const { return type_case_ == kPolygonFilament; }
  const PolygonFilament& polygon_filament() const { return polygon_filament_; }
  PolygonFilament* mutable_polygon_filament() {
    if (type_case_ != kPolygonFilament) {
      Clear();
      type_case_ = kPolygonFilament;
      std::construct_at(std::addressof(polygon_filament_));
    }
    return &polygon_filament_;
  }
  void set_polygon_filament(const PolygonFilament& value) {
    Clear();
    type_case_ = kPolygonFilament;
    std::construct_at(std::addressof(polygon_filament_), value);
  }

  TypeCase type_case() const { return type_case_; }
};  // CurrentCarrier

struct Coil {
  // a human-readable name, e.g., for plotting
  std::optional<std::string> name;

  // number of windings == multiplier for current along geometry;
  // num_windings == 1.0 is often assumed if this field is not populated
  // (std::nullopt)
  std::optional<double> num_windings_;

  // objects that define the single-turn geometry of the coil
  std::list<CurrentCarrier> current_carriers_;

  bool has_num_windings() const { return num_windings_.has_value(); }
  double num_windings() const { return num_windings_.value(); }
  void set_num_windings(double value) { num_windings_ = value; }

  int current_carriers_size() const {
    return static_cast<int>(current_carriers_.size());
  }
  const CurrentCarrier& current_carriers(int index) const {
    auto it = current_carriers_.cbegin();
    std::advance(it, index);
    return *it;
  }
  CurrentCarrier* mutable_current_carriers(int index) {
    auto it = current_carriers_.begin();
    std::advance(it, index);
    return &(*it);
  }
  CurrentCarrier* add_current_carriers() {
    current_carriers_.emplace_back();  // Default construct new carrier
    return &current_carriers_.back();
  }
  const std::list<CurrentCarrier>& current_carriers() const {
    return current_carriers_;
  }
  std::list<CurrentCarrier>* mutable_current_carriers() {
    return &current_carriers_;
  }

  // Clear the entire structure
  void Clear() {
    name.reset();
    num_windings_.reset();
    current_carriers_.clear();
  }
};  // Coil

struct SerialCircuit {
  // a human-readable name, e.g., for plotting
  std::optional<std::string> name;

  // current along each of the current carriers
  std::optional<double> current_;

  // objects that define the geometry of coils
  std::list<Coil> coils_;

  bool has_current() const { return current_.has_value(); }
  double current() const { return current_.value(); }
  void set_current(double value) { current_ = value; }

  int coils_size() const { return static_cast<int>(coils_.size()); }
  const Coil& coils(int index) const {
    auto it = coils_.cbegin();
    std::advance(it, index);
    return *it;
  }
  Coil* mutable_coils(int index) {
    auto it = coils_.begin();
    std::advance(it, index);
    return &(*it);
  }
  Coil* add_coils() {
    coils_.emplace_back();  // Default construct new coil
    return &coils_.back();
  }
  const std::list<Coil>& coils() const { return coils_; }
  std::list<Coil>* mutable_coils() { return &coils_; }

  // Clear the entire structure
  void Clear() {
    name.reset();
    current_.reset();
    coils_.clear();
  }
};  // SerialCircuit

struct MagneticConfiguration {
  // a human-readable name, e.g., for plotting
  std::optional<std::string> name;

  // number of field periods of this coil set
  std::optional<int> num_field_periods_;

  // objects that specify geometry and currents of coils
  std::list<SerialCircuit> serial_circuits_;

  int serial_circuits_size() const {
    return static_cast<int>(serial_circuits_.size());
  }
  const SerialCircuit& serial_circuits(int index) const {
    auto it = serial_circuits_.cbegin();
    std::advance(it, index);
    return *it;
  }
  SerialCircuit* mutable_serial_circuits(int index) {
    auto it = serial_circuits_.begin();
    std::advance(it, index);
    return &(*it);
  }
  SerialCircuit* add_serial_circuits() {
    serial_circuits_.emplace_back();
    return &serial_circuits_.back();
  }
  const std::list<SerialCircuit>& serial_circuits() const {
    return serial_circuits_;
  }
  std::list<SerialCircuit>* mutable_serial_circuits() {
    return &serial_circuits_;
  }

  // Clear the entire structure (all fields)
  void Clear() {
    name.reset();
    num_field_periods_.reset();
    serial_circuits_.clear();
  }
};  // MagneticConfiguration

}  // namespace magnetics

#endif  // VMECPP_COMMON_MAGNETIC_CONFIGURATION_DEFINITION_MAGNETIC_CONFIGURATION_H_
