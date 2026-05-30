#include <filesystem>
#include <format>
#include <fstream>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

#include "geometry/mesh.hpp"
#include "history/history.hpp"
#include "interface/state.hpp"

namespace athelas {

using basis::NodalBasis;

using QuantityFunction = std::function<double(const MeshState &, const Mesh &)>;

HistoryOutput::HistoryOutput(const std::string &filename,
                             const std::string &output_dir, const bool enabled)
    : enabled_(enabled), header_written_(false), filename_(filename),
      output_dir_(output_dir) {
  if (!enabled_) {
    return;
  }
  const std::string path = output_dir_ + "/" + filename;
  // Restart case: existing non-empty file already carries a header - don't
  // emit another one. This could lead to unintended behavior, such as if a
  // non-empty filename exists with garbage in it, we will happily append.
  std::error_code ec;
  if (std::filesystem::exists(path, ec) &&
      std::filesystem::file_size(path, ec) > 0) {
    header_written_ = true;
  }
  file_.open(path, std::ios::out | std::ios::app);
}

void HistoryOutput::add_quantity(const std::string &name,
                                 QuantityFunction func) {
  if (!enabled_) {
    return;
  }
  quantities_[name] = std::move(func);
  quantity_names_.push_back(name);
}

void HistoryOutput::write(const MeshState &mesh_state, const Mesh &mesh,
                          double time) {
  if (!enabled_) {
    return;
  }
  // Only write header if file doesn't already exist
  // We may want to change this behavior in the future for, e.g.,
  // weird restarts. This is where to change that. Just write the header
  // in the constructor and as we add_quantity.
  if (!header_written_) {
    file_ << "# 0 Time [s]";
    int i = 1;
    for (const auto &name : quantity_names_) {
      file_ << " " << std::to_string(i) << " " << name;
      ++i;
    }
    header_written_ = true;
  }
  file_ << std::format("\n{:.15e}", time);

  Kokkos::Profiling::pushRegion("IO");
  Kokkos::Profiling::pushRegion("History");
  for (const auto &name : quantity_names_) {
    const double value = quantities_[name](mesh_state, mesh);
    file_ << std::format(" {:.15e}", value);
  }
  Kokkos::Profiling::popRegion();
  Kokkos::Profiling::popRegion();

  file_.flush();
}

} // namespace athelas
