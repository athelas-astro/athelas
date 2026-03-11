/**
 * @file problem_in.hpp
 * --------------
 *
 * @brief Class for loading input deck
 *
 * @details Loads input deck in Lua format via sol2.
 *          See: https://github.com/ThePhD/sol2
 */

#pragma once

#include <iostream>

#define SOL_ALL_SAFETIES_ON 1
#include "sol/sol.hpp"

#include "interface/params.hpp"
#include "utils/error.hpp"

namespace athelas {

class ProblemIn {

 public:
  ProblemIn(const std::string &fn, const std::string &output_dir);

  auto param() -> Params *;
  [[nodiscard]] auto param() const -> Params *;

 private:
  sol::state lua_;
  sol::table config_;
  // params obj
  std::unique_ptr<Params> params_;
};

// TODO(astrobarker) move into class
auto check_bc(std::string bc) -> bool;

template <typename G>
void read_lua_array(const sol::table &tbl, G &out_array) {
  const std::size_t expected = out_array.size();
  if (tbl.size() != expected) {
    std::cerr << "Lua array size " << tbl.size()
              << " does not match expected size " << expected << "\n";
    throw_athelas_error(" ! Error reading dirichlet boundary conditions.");
  }
  for (std::size_t i = 0; i < expected; ++i) {
    // Lua arrays are 1-indexed
    sol::optional<double> val = tbl[i + 1];
    if (!val) {
      std::cerr << "Type mismatch or nil at Lua array index " << (i + 1)
                << "\n";
      throw_athelas_error(" ! Error reading dirichlet boundary conditions.");
    }
    out_array[i] = *val;
  }
}

} // namespace athelas
