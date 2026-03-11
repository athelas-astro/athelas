#include "pgen/lua_validator.hpp"

#include <sstream>
#include <stdexcept>

#include "utils/error.hpp"

namespace athelas {

Validator::Validator(sol::table schema) : schema_(schema) {}

void Validator::validate(sol::table config) {
  validate_table(config, schema_, config, "");
}

bool Validator::is_leaf(sol::table node) { return node["doc"].valid(); }

sol::object Validator::get_path(sol::table root, const std::string &path) {
  sol::object current = root;

  std::stringstream ss(path);
  std::string part;

  while (std::getline(ss, part, '.')) {
    if (!current.is<sol::table>()) {
      return sol::nil;
    }

    sol::table tbl = current;

    current = tbl[part];

    if (!current.valid()) {
      return sol::nil;
    }
  }

  return current;
}

auto Validator::subtree_requires(sol::table schema, sol::table root) -> bool {
  for (auto &kv : schema) {
    sol::table node = kv.second;

    if (node["ignore"].valid()) {
      continue;
    }


    if (is_leaf(node)) {
      if (node["required"].valid()) {
        if (required_dep(node["required"], root)) {
          return true;
        }
      }
    } else {
      if (subtree_requires(node, root)) {
        return true;
      }
    }
  }

  return false;
}

auto Validator::required_dep(sol::object rule, sol::table root) -> bool {
  if (rule.is<bool>()) {
    return rule.as<bool>();
  }

  sol::table cond = rule;

  std::string path = cond["when"];

  sol::object val = get_path(root, path);

  if (!val.valid()) {
    return false;
  }

  if (cond["equals"].valid()) {
    return val == cond["equals"];
  }

  if (cond["is_true"].valid()) {
    return val.as<bool>();
  }

  return false;
}

void Validator::validate_table(sol::table config, sol::table schema,
                               sol::table root, const std::string &prefix) {
  for (auto &kv : schema) {
    std::string key = kv.first.as<std::string>();
    sol::table node = kv.second;

    std::string full = prefix.empty() ? key : prefix + "." + key;

    if (node["ignore"].valid()) {
      continue;
    }

    if (is_leaf(node)) {
      sol::object val = config[key];
      bool present = val.valid();

      // insert default
      if (!present && node["default"].valid()) {
        config[key] = node["default"];
        present = true;
      }

      // required check
      if (!present && node["required"].valid()) {
        if (required_dep(node["required"], root)) {
          throw_athelas_error("Missing required field: " + full);
        }
      }
    } else {
      sol::object sub = config[key];

      if (!sub.valid()) {
        if (subtree_requires(node, root)) {
          throw_athelas_error("Missing required table: " + full);
        }

        // optional table: create empty
        config[key] = sol::table(config.lua_state(), sol::create);
        sub = config[key];
      }

      validate_table(sub, node, root, full);
    }
  }
}

} // namespace athelas
