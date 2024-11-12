#include "vmecpp/common/vmec_indata/boundary_from_json.h"

#include <string>
#include <vector>

#include "absl/strings/str_format.h"
#include "nlohmann/json.hpp"

namespace vmecpp {

using nlohmann::json;

using json_io::JsonReadBool;
using json_io::JsonReadDouble;
using json_io::JsonReadInt;
using json_io::JsonReadString;
using json_io::JsonReadVectorDouble;
using json_io::JsonReadVectorInt;

absl::StatusOr<std::optional<std::vector<BoundaryCoefficient> > >
BoundaryCoefficient::FromJson(const json& j, const std::string& name) {
  if (!j.contains(name)) {
    // not present --> skip
    return std::nullopt;
  }

  if (!j[name].is_array()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("JSON element '%s' is not an array", name));
  }

  std::vector<BoundaryCoefficient> entries;
  int i = 0;
  for (const json& entry : j[name]) {
    if (!entry.is_object()) {
      return absl::InvalidArgumentError(
          absl::StrFormat("JSON entry '%s'[%d] is not an object", name, i));
    }

    auto m = JsonReadInt(entry, "m");
    if (!m.ok()) {
      return m.status();
    }
    if (!m->has_value()) {
      // skip entries where "m" is not specified
      continue;
    }

    auto n = JsonReadInt(entry, "n");
    if (!n.ok()) {
      return n.status();
    }
    if (!n->has_value()) {
      // skip entries where "n" is not specified
      continue;
    }

    auto value = JsonReadDouble(entry, "value");
    if (!value.ok()) {
      return value.status();
    }
    if (!value->has_value()) {
      // skip entries where "value" is not specified
      continue;
    }

    BoundaryCoefficient boundary_coefficient = {
        /*m=*/m->value(), /*n=*/n->value(), /*value=*/value->value()};
    entries.push_back(boundary_coefficient);

    i++;
  }

  return entries;
}  // JsonReadBoundary

}  // namespace vmecpp
