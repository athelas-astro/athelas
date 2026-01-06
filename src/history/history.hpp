#pragma once

#include <fstream>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

#include "geometry/grid.hpp"
#include "polynomial_basis.hpp"
#include "state/state.hpp"

namespace athelas {

class HistoryOutput {
 public:
  // NOTE: We are always passing in two basis objects to our history functions.
  // This hsould be fine -- the driver always has two, even in pure Hydro mode.
  using QuantityFunction = std::function<double(
      const MeshState &, const GridStructure &, const basis::ModalBasis *,
      const basis::ModalBasis *)>;

  explicit HistoryOutput(const std::string &filename,
                         const std::string &output_dir, bool enabled);

  void add_quantity(const std::string &name, QuantityFunction func);

  void write(const MeshState &mesh_state, const GridStructure &grid,
             const basis::ModalBasis *fluid_basis,
             const basis::ModalBasis *rad_basis, double time);

 private:
  bool enabled_;
  bool header_written_;
  std::string filename_;
  std::string output_dir_;
  std::ofstream file_;
  std::unordered_map<std::string, QuantityFunction> quantities_;
  std::vector<std::string> quantity_names_;
};

} // namespace athelas
