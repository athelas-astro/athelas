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
#include <string>
#include <utility>
#include <vector>

#define SOL_ALL_SAFETIES_ON 1
#include "sol/sol.hpp"

#include "interface/params.hpp"
#include "utils/error.hpp"

namespace athelas {

class ProblemIn {

 public:
  /** Disambiguates the restart constructor from the Lua-input constructor. */
  struct RestartTag {};

  ProblemIn(
      const std::string &fn, const std::string &output_dir,
      const std::vector<std::pair<std::string, std::string>> &overrides = {});

  /**
   * @brief Construct from an HDF5 .ath restart file (no Lua input deck).
   *
   * Populates Params directly from /params, overrides output.dir with the
   * supplied path, re-derives time.integrator from time.integrator_string,
   * and applies any CLI overrides as Lua expressions (parsed via a private
   * sol::state — no input file is loaded).
   */
  ProblemIn(
      RestartTag, const std::string &h5_filename, const std::string &output_dir,
      const std::vector<std::pair<std::string, std::string>> &overrides = {});

  auto param() -> Params *;
  [[nodiscard]] auto param() const -> Params *;

 private:
  /**
   * @brief Apply a single restart-mode override: parse `expr` as Lua and
   * assign into `params_[key]`. Type is taken from the existing param if
   * present; otherwise inferred from the Lua value.
   */
  void apply_restart_override(const std::string &key, const std::string &expr);

  sol::state lua_;
  sol::table config_;
  // params obj
  std::unique_ptr<Params> params_;
};

template <typename G>
void read_lua_array(const sol::table &tbl, G &out_array) {
  const std::size_t expected = out_array.size();
  if (tbl.size() != expected) {
    std::cerr << "Lua array size " << tbl.size()
              << " does not match expected size " << expected << "\n";
    throw_athelas_error(" ! Error reading Lua array.");
  }
  for (std::size_t i = 0; i < expected; ++i) {
    // Lua arrays are 1-indexed
    sol::optional<double> val = tbl[i + 1];
    if (!val) {
      std::cerr << "Type mismatch or nil at Lua array index " << (i + 1)
                << "\n";
      throw_athelas_error(" ! Error reading Lua array.");
    }
    out_array[i] = *val;
  }
}

} // namespace athelas
