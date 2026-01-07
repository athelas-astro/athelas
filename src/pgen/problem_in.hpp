/**
 * @file problem_in.cpp
 * --------------
 *
 * @brief Class for loading input deck
 *
 * @details Loads input deck in TOML format.
 *          See: https://github.com/marzer/tomlplusplus
 */

#pragma once

#include <iostream>

#include "toml.hpp"

#include "interface/params.hpp"
#include "utils/error.hpp"

namespace athelas {

class ProblemIn {

 public:
  ProblemIn(const std::string &fn, const std::string &output_dir);

  auto param() -> Params *;
  [[nodiscard]] auto param() const -> Params *;

 private:
  toml::table config_;
  // params obj
  std::unique_ptr<Params> params_;
};

// TODO(astrobarker) move into class
auto check_bc(std::string bc) -> bool;

template <typename T, typename G>
void read_toml_array(T toml_array, G &out_array) {
  long unsigned int index = 0;
  for (const auto &element : *toml_array) {
    if (index < out_array.size()) {
      if (auto elem = element.as_floating_point()) {
        out_array[index] = static_cast<double>(*elem);
      } else {
        std::cerr << "Type mismatch at index " << index << "\n";
        throw_athelas_error(" ! Error reading dirichlet boundary conditions.");
      }
      index++;
    } else {
      std::cerr << "TOML array is larger than C++ array." << "\n";
      throw_athelas_error(" ! Error reading dirichlet boundary conditions.");
    }
  }
}

} // namespace athelas
