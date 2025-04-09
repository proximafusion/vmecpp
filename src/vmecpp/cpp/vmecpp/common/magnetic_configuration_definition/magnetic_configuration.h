#ifndef VMECPP_COMMON_MAGNETIC_CONFIGURATION_DEFINITION_MAGNETIC_CONFIGURATION_H_
#define VMECPP_COMMON_MAGNETIC_CONFIGURATION_DEFINITION_MAGNETIC_CONFIGURATION_H_

#include <cstdint>
#include <list>
#include <string>

#include "vmecpp/common/composed_types_definition/composed_types.h"

namespace magnetics {

struct PolygonFilament {
  // a human-readable name, e.g., for plotting
  bool has_name_ = false;
  std::string name_;

  // Cartesian components of filament geometry
  std::list<composed_types::Vector3d> vertices_;

  // name
  bool has_name() const { return has_name_; }
  const std::string& name() const { return name_; }
  void set_name(const std::string& value) {
    name_ = value;
    has_name_ = true;
  }
  void set_name(std::string&& value) {
    name_ = std::move(value);
    has_name_ = true;
  }
  void clear_name() {
    name_.clear();
    has_name_ = false;
  }

  // vertices
  int vertices_size() const { return static_cast<int>(vertices_.size()); }
  const composed_types::Vector3d& vertices(int index) const {
    auto it = vertices_.cbegin();
    std::advance(it, index);  // no explicit bounds-check for brevity
    return *it;
  }
  composed_types::Vector3d* mutable_vertices(int index) {
    auto it = vertices_.begin();
    std::advance(it, index);  // no explicit bounds-check for brevity
    return &(*it);
  }
  composed_types::Vector3d* add_vertices() {
    vertices_.emplace_back();
    auto it = vertices_.end();
    --it;
    return &(*it);
  }
  const std::list<composed_types::Vector3d>& vertices() const {
    return vertices_;
  }
  std::list<composed_types::Vector3d>* mutable_vertices() { return &vertices_; }
  void clear_vertices() { vertices_.clear(); }

  // Clear the entire structure
  void Clear() {
    clear_name();
    clear_vertices();
  }
};  // PolygonFilament

struct CircularFilament {
  // a human-readable name, e.g., for plotting
  bool has_name_ = false;
  std::string name_;

  // Cartesian coordinates of the center point of the loop
  bool has_center_ = false;
  composed_types::Vector3d center_;

  // Cartesian components of a vector pointing along the normal of the circle
  // around which the current flows
  bool has_normal_ = false;
  composed_types::Vector3d normal_;

  // radius of the loop
  bool has_radius_ = false;
  double radius_ = 0.0;

  // name
  bool has_name() const { return has_name_; }
  const std::string& name() const { return name_; }
  void set_name(const std::string& value) {
    name_ = value;
    has_name_ = true;
  }
  void set_name(std::string&& value) {
    name_ = std::move(value);
    has_name_ = true;
  }
  void clear_name() {
    name_.clear();
    has_name_ = false;
  }

  // center
  bool has_center() const { return has_center_; }
  const composed_types::Vector3d& center() const { return center_; }
  composed_types::Vector3d* mutable_center() {
    has_center_ = true;
    return &center_;
  }
  void set_center(const composed_types::Vector3d& value) {
    center_ = value;
    has_center_ = true;
  }
  void clear_center() {
    center_ = composed_types::Vector3d();
    has_center_ = false;
  }

  // normal
  bool has_normal() const { return has_normal_; }
  const composed_types::Vector3d& normal() const { return normal_; }
  composed_types::Vector3d* mutable_normal() {
    has_normal_ = true;
    return &normal_;
  }
  void set_normal(const composed_types::Vector3d& value) {
    normal_ = value;
    has_normal_ = true;
  }
  void clear_normal() {
    normal_ = composed_types::Vector3d();
    has_normal_ = false;
  }

  // radius
  bool has_radius() const { return has_radius_; }
  double radius() const { return radius_; }
  void set_radius(double value) {
    radius_ = value;
    has_radius_ = true;
  }
  void clear_radius() {
    radius_ = 0.0;
    has_radius_ = false;
  }

  // Clear the entire structure
  void Clear() {
    clear_name();
    clear_center();
    clear_normal();
    clear_radius();
  }
};  // CircularFilament

struct InfiniteStraightFilament {
  // a human-readable name, e.g., for plotting
  bool has_name_ = false;
  std::string name_;

  // Cartesian coordinates of a point on the filament
  bool has_origin_ = false;
  composed_types::Vector3d origin_;

  // Cartesian components of the direction along the filament
  bool has_direction_ = false;
  composed_types::Vector3d direction_;

  // name
  bool has_name() const { return has_name_; }
  const std::string& name() const { return name_; }
  void set_name(const std::string& value) {
    name_ = value;
    has_name_ = true;
  }
  void set_name(std::string&& value) {
    name_ = std::move(value);
    has_name_ = true;
  }
  void clear_name() {
    name_.clear();
    has_name_ = false;
  }

  // origin
  bool has_origin() const { return has_origin_; }
  const composed_types::Vector3d& origin() const { return origin_; }
  composed_types::Vector3d* mutable_origin() {
    has_origin_ = true;
    return &origin_;
  }
  void set_origin(const composed_types::Vector3d& value) {
    origin_ = value;
    has_origin_ = true;
  }
  void clear_origin() {
    origin_ = composed_types::Vector3d();
    has_origin_ = false;
  }

  // direction
  bool has_direction() const { return has_direction_; }
  const composed_types::Vector3d& direction() const { return direction_; }
  composed_types::Vector3d* mutable_direction() {
    has_direction_ = true;
    return &direction_;
  }
  void set_direction(const composed_types::Vector3d& value) {
    direction_ = value;
    has_direction_ = true;
  }
  void clear_direction() {
    direction_ = composed_types::Vector3d();
    has_direction_ = false;
  }

  // Clear the entire structure
  void Clear() {
    clear_name();
    clear_origin();
    clear_direction();
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
  bool has_name_ = false;
  std::string name_;

  // number of windings == multiplier for current along geometry;
  // num_windings == 1 is assumed if this field is not populated
  bool has_num_windings_ = false;
  double num_windings_ = 0.0;

  // objects that define the single-turn geometry of the coil
  std::list<CurrentCarrier> current_carriers_;

  // name
  bool has_name() const { return has_name_; }
  const std::string& name() const { return name_; }
  void set_name(const std::string& value) {
    name_ = value;
    has_name_ = true;
  }
  void set_name(std::string&& value) {
    name_ = std::move(value);
    has_name_ = true;
  }
  void clear_name() {
    name_.clear();
    has_name_ = false;
  }

  // num_windings
  bool has_num_windings() const { return has_num_windings_; }
  double num_windings() const { return num_windings_; }
  void set_num_windings(double value) {
    num_windings_ = value;
    has_num_windings_ = true;
  }
  void clear_num_windings() {
    num_windings_ = 0.0;
    has_num_windings_ = false;
  }

  // current_carriers
  int current_carriers_size() const {
    return static_cast<int>(current_carriers_.size());
  }
  const CurrentCarrier& current_carriers(int index) const {
    auto it = current_carriers_.cbegin();
    // No bounds checks here for brevity
    std::advance(it, index);
    return *it;
  }
  CurrentCarrier* mutable_current_carriers(int index) {
    auto it = current_carriers_.begin();
    // No bounds checks here for brevity
    std::advance(it, index);
    return &(*it);
  }
  CurrentCarrier* add_current_carriers() {
    current_carriers_.emplace_back();
    auto it = current_carriers_.end();
    // Move to newly added element
    --it;
    return &(*it);
  }
  const std::list<CurrentCarrier>& current_carriers() const {
    return current_carriers_;
  }
  std::list<CurrentCarrier>* mutable_current_carriers() {
    return &current_carriers_;
  }
  void clear_current_carriers() { current_carriers_.clear(); }

  // Clear the entire structure
  void Clear() {
    clear_name();
    clear_num_windings();
    clear_current_carriers();
  }
};  // Coil

struct SerialCircuit {
  // a human-readable name, e.g., for plotting
  bool has_name_ = false;
  std::string name_;

  // current along each of the current carriers
  bool has_current_ = false;
  double current_ = 0.0;

  // objects that define the geometry of coils
  std::list<Coil> coils_;

  // name
  bool has_name() const { return has_name_; }
  const std::string& name() const { return name_; }
  void set_name(const std::string& value) {
    name_ = value;
    has_name_ = true;
  }
  void set_name(std::string&& value) {
    name_ = std::move(value);
    has_name_ = true;
  }
  void clear_name() {
    name_.clear();
    has_name_ = false;
  }

  // current
  bool has_current() const { return has_current_; }
  double current() const { return current_; }
  void set_current(double value) {
    current_ = value;
    has_current_ = true;
  }
  void clear_current() {
    current_ = 0.0;
    has_current_ = false;
  }

  // coils
  int coils_size() const { return static_cast<int>(coils_.size()); }
  const Coil& coils(int index) const {
    auto it = coils_.cbegin();
    // No bounds checks here for brevity
    std::advance(it, index);
    return *it;
  }
  Coil* mutable_coils(int index) {
    auto it = coils_.begin();
    // No bounds checks here for brevity
    std::advance(it, index);
    return &(*it);
  }
  Coil* add_coils() {
    coils_.emplace_back();
    auto it = coils_.end();
    --it;
    return &(*it);
  }
  const std::list<Coil>& coils() const { return coils_; }
  std::list<Coil>* mutable_coils() { return &coils_; }
  void clear_coils() { coils_.clear(); }

  // Clear the entire structure
  void Clear() {
    clear_name();
    clear_current();
    clear_coils();
  }
};  // SerialCircuit

struct MagneticConfiguration {
  // a human-readable name, e.g., for plotting
  bool has_name_ = false;
  std::string name_;

  // number of field periods of this coil set
  bool has_num_field_periods_ = false;
  int num_field_periods_ = 0;

  // objects that specify geometry and currents of coils
  std::list<SerialCircuit> serial_circuits_;

  // name
  bool has_name() const { return has_name_; }
  const std::string& name() const { return name_; }
  void set_name(const std::string& value) {
    name_ = value;
    has_name_ = true;
  }
  void set_name(std::string&& value) {
    name_ = std::move(value);
    has_name_ = true;
  }
  void clear_name() {
    name_.clear();
    has_name_ = false;
  }

  // num_field_periods
  bool has_num_field_periods() const { return has_num_field_periods_; }
  int num_field_periods() const { return num_field_periods_; }
  void set_num_field_periods(int value) {
    num_field_periods_ = value;
    has_num_field_periods_ = true;
  }
  void clear_num_field_periods() {
    num_field_periods_ = 0;
    has_num_field_periods_ = false;
  }

  // serial_circuits
  int serial_circuits_size() const {
    return static_cast<int>(serial_circuits_.size());
  }
  const SerialCircuit& serial_circuits(int index) const {
    auto it = serial_circuits_.cbegin();
    // No bounds checks here for brevity
    std::advance(it, index);
    return *it;
  }
  SerialCircuit* mutable_serial_circuits(int index) {
    auto it = serial_circuits_.begin();
    // No bounds checks here for brevity
    std::advance(it, index);
    return &(*it);
  }
  SerialCircuit* add_serial_circuits() {
    serial_circuits_.emplace_back();
    auto it = serial_circuits_.end();
    --it;
    return &(*it);
  }
  const std::list<SerialCircuit>& serial_circuits() const {
    return serial_circuits_;
  }
  std::list<SerialCircuit>* mutable_serial_circuits() {
    return &serial_circuits_;
  }
  void clear_serial_circuits() { serial_circuits_.clear(); }

  // Clear the entire structure (all fields)
  void Clear() {
    clear_name();
    clear_num_field_periods();
    clear_serial_circuits();
  }
};  // MagneticConfiguration

}  // namespace magnetics

#endif  // VMECPP_COMMON_MAGNETIC_CONFIGURATION_DEFINITION_MAGNETIC_CONFIGURATION_H_
