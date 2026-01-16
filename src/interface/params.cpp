#include <string>

#include "interface/params.hpp"

namespace athelas {

auto Params::contains(const std::string &key) const -> bool {
  return params_.contains(key);
}

// Remove a parameter -- note that the caller
// has no guarantee that the key existed.
void Params::remove(const std::string &key) { params_.erase(key); }

// Clear all parameters
void Params::clear() { params_.clear(); }

// Get all parameter keys
auto Params::keys() const -> std::vector<std::string> {
  std::vector<std::string> result;
  result.reserve(params_.size());
  for (const auto &[key, _] : params_) {
    result.push_back(key);
  }
  return result;
}

auto Params::get_type(const std::string &key) const -> std::type_index {
  return params_.at(key)->type();
}

auto get_safe_param(const std::any &a) -> std::optional<ParamValue> {
    if (a.type() == typeid(bool)) {
        return std::any_cast<bool>(a);
    }

    if (a.type() == typeid(int)) {
        return std::any_cast<int>(a);
    }

    if (a.type() == typeid(double)) {
        return std::any_cast<double>(a);
    }

    if (a.type() == typeid(std::string)) {
        return std::any_cast<std::string>(a);
    }

    if (a.type() == typeid(const char *)) {
        return std::string(std::any_cast<const char *>(a));
    }

    if (a.type() == typeid(std::vector<int>)) {
        return std::any_cast<std::vector<int>>(a);
    }

    if (a.type() == typeid(std::vector<double>)) {
        return std::any_cast<std::vector<double>>(a);
    }

    return std::nullopt;
}

} // namespace athelas
