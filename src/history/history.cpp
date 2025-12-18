#include <format>
#include <fstream>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

#include "geometry/grid.hpp"
#include "history/history.hpp"
#include "polynomial_basis.hpp"
#include "state/state.hpp"

namespace athelas {

using basis::ModalBasis;

using QuantityFunction =
    std::function<double(const State &, const GridStructure &,
                         const ModalBasis *, const ModalBasis *)>;

HistoryOutput::HistoryOutput(const std::string &filename,
                             const std::string &output_dir, const bool enabled)
    : enabled_(enabled), header_written_(false), filename_(filename),
      output_dir_(output_dir) {
  if (!enabled_) {
    return;
  }
  file_.open(output_dir_ + "/" + filename, std::ios::out | std::ios::app);
}

void HistoryOutput::add_quantity(const std::string &name,
                                 QuantityFunction func) {
  if (!enabled_) {
    return;
  }
  quantities_[name] = std::move(func);
  quantity_names_.push_back(name);
}

void HistoryOutput::write(const State &state, const GridStructure &grid,
                          const ModalBasis *fluid_basis,
                          const ModalBasis *rad_basis, double time) {
  if (!enabled_) {
    return;
  }
  // Only write header if file doesn't already exist
  // We may want to change this behavior in the future for, e.g.,
  // weird restarts. This is where to change that. Just write the header
  // in the constructor and as we add_quantity.
  if (!header_written_) {
    file_ << "# Time [s]";
    for (const auto &name : quantity_names_) {
      file_ << " " << name;
    }
    header_written_ = true;
  }
  file_ << std::format("\n{:.15e}", time);

  for (const auto &name : quantity_names_) {
    const double value = quantities_[name](state, grid, fluid_basis, rad_basis);
    file_ << std::format(" {:.15e}", value);
  }

  // We don't necessarily want to force a flush.
  // file_.flush();
}

} // namespace athelas
