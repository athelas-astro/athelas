#pragma once

#include <fstream>
#include <functional>
#include <string>
#include <vector>

#include "geometry/mesh.hpp"
#include "interface/state.hpp"
#include "polynomial_basis.hpp"

namespace athelas {

class HistoryOutput {
 public:
  using QuantityFunction =
      std::function<double(const MeshState &, const Mesh &)>;
  using QuantityGroupFunction =
      std::function<std::vector<double>(const MeshState &, const Mesh &)>;

  explicit HistoryOutput(const std::string &filename,
                         const std::string &output_dir, bool enabled);

  void add_quantity(const std::string &name, QuantityFunction func);
  void add_quantities(const std::vector<std::string> &names,
                      QuantityGroupFunction func);

  void write(const MeshState &mesh_state, const Mesh &mesh, double time);

 private:
  struct QuantityEntry {
    std::vector<std::string> names;
    QuantityGroupFunction values;
  };

  bool enabled_;
  bool header_written_;
  std::string filename_;
  std::string output_dir_;
  std::ofstream file_;
  std::vector<QuantityEntry> quantities_;
};

} // namespace athelas
