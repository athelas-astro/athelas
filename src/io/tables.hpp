#pragma once

#include <vector>

#include "H5Cpp.h"

#include "kokkos_types.hpp"

#include "spiner/databox.hpp"

namespace athelas {

using DataBox = Spiner::DataBox<double, Spiner::RegularGrid1D<double>>;

struct OpacityTable {
  std::vector<double> X;
  std::vector<double> Z;
  std::vector<double> logT;
  std::vector<double> logR;
  HostArray4D<double> kappa;

  std::size_t nX, nZ, nT, nR;
};

auto load_opacity_table(const std::string &filename) -> DataBox;

// Helper function to read metadata attribute
template <typename T>
auto read_metadata_attr(H5::H5File &file, const std::string &path,
                        const std::string &attr_name) -> T {
  H5::Group group = file.openGroup(path);
  H5::Attribute attr = group.openAttribute(attr_name);

  T value;
  attr.read(attr.getDataType(), &value);

  return value;
}
} // namespace athelas
