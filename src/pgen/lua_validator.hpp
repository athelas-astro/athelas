#pragma once

#include <string>

#include "sol/sol.hpp"

namespace athelas {

class Validator {
 public:
  explicit Validator(sol::table schema);

  // Validate config in-place (defaults inserted)
  void validate(sol::table config);

 private:
  sol::table schema_;

  void validate_table(sol::table config, sol::table schema, sol::table root,
                      const std::string &prefix);

  auto is_leaf(sol::table node) -> bool;

  auto get_path(sol::table root, const std::string &path) -> sol::object;

  auto required_dep(sol::object rule, sol::table root) -> bool;

  auto subtree_requires(sol::table schema, sol::table root) -> bool;
};

} // namespace athelas
