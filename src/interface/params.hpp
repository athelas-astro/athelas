/**
 * @file params.hpp
 * --------------
 *
 * @brief Simple params container
 * @note Provide a convenient wrapper around
 *   std::unordered_map<std::string, std::unique_ptr<std::any>>
 */

#pragma once

#include <any>
#include <string>
#include <typeindex>
#include <unordered_map>
#include <variant>

#include "utils/error.hpp"

namespace athelas {

/**
 * @class Params
 * @note Type safe paramater container with key-valye storage.
 **/
class Params {

 public:
  Params() = default;

  // Copy constructor - deep copy all stored objects
  Params(const Params &other) {
    for (const auto &[key, value_ptr] : other.params_) {
      if (value_ptr) {
        params_[key] = std::make_unique<std::any>(*value_ptr);
      }
    }
  }

  // Move constructor
  Params(Params &&other) noexcept = default;

  // Copy assignment - deep copy all stored objects
  auto operator=(const Params &other) -> Params & {
    if (this != &other) {
      params_.clear();
      for (const auto &[key, value_ptr] : other.params_) {
        if (value_ptr) {
          params_[key] = std::make_unique<std::any>(*value_ptr);
        }
      }
    }
    return *this;
  }

  // Move assignment
  auto operator=(Params &&other) noexcept -> Params & = default;

  ~Params() = default;

  // Add by copy
  template <typename T>
  void add(const std::string &key, const T &value) {
    params_[key] = std::make_unique<std::any>(value);
  }

  // Add by move
  template <typename T>
  void add(const std::string &key, T &&value) {
    params_[key] = std::make_unique<std::any>(std::forward<T>(value));
  }

  // Get value (returns a copy)
  template <typename T>
  auto get(const std::string &key) const -> T {
    auto it = params_.find(key);
    if (it == params_.end() || !it->second) {
      throw_athelas_error("Parameter '" + key + "' not found");
    }
    try {
      return std::any_cast<T>(*(it->second));
    } catch (const std::bad_any_cast &) {
      throw_athelas_error("Type mismatch for parameter '" + key + "'");
    }
  }

  // Get reference to stored value (avoid copy for large objects)
  template <typename T>
  auto get_ref(const std::string &key) const -> const T & {
    auto it = params_.find(key);
    if (it == params_.end() || !it->second) {
      throw_athelas_error("Parameter '" + key + "' not found");
    }
    try {
      return std::any_cast<const T &>(*(it->second));
    } catch (const std::bad_any_cast &) {
      throw_athelas_error("Type mismatch for parameter '" + key + "'");
    }
  }

  // Get mutable reference (for in-place modification)
  template <typename T>
  auto get_mutable_ref(const std::string &key) -> T & {
    auto it = params_.find(key);
    if (it == params_.end() || !it->second) {
      throw_athelas_error("Parameter '" + key + "' not found");
    }
    try {
      return std::any_cast<T &>(*(it->second));
    } catch (const std::bad_any_cast &) {
      throw_athelas_error("Type mismatch for parameter '" + key + "'");
    }
  }

  // Overload get to return value or default as needed.
  template <typename T>
  auto get(const std::string &key, const T &default_value) -> T {
    auto it = params_.find(key);
    if (it == params_.end() || !it->second) {
      add(key, default_value);
      return default_value;
    }
    try {
      return std::any_cast<T>(*(it->second));
    } catch (const std::bad_any_cast &) {
      throw_athelas_error("Type mismatch for parameter '" + key + "'");
    }
  }

  // Check if a parameter exists
  auto contains(const std::string &key) const -> bool;
  void remove(const std::string &key);
  void clear();
  auto keys() const -> std::vector<std::string>;
  auto get_type(const std::string &key) const -> std::type_index;

 private:
  std::unordered_map<std::string, std::unique_ptr<std::any>> params_;
};

using ParamValue = std::variant<
    bool,
    int,
    double,
    std::string,
    std::vector<int>,
    std::vector<double>
>;

auto get_safe_param(const std::any &a) -> std::optional<ParamValue>;

} // namespace athelas
