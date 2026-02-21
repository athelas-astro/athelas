#include "io/tables.hpp"
#include "H5Cpp.h"
#include "utils/error.hpp"

namespace athelas {

static auto read_1d_double_dataset(H5::H5File &file, const std::string &path)
    -> std::vector<double> {
  H5::DataSet dset = file.openDataSet(path);
  H5::DataSpace space = dset.getSpace();

  if (space.getSimpleExtentNdims() != 1) {
    throw_athelas_error("Dataset is not 1D: " + path);
  }

  hsize_t n;
  space.getSimpleExtentDims(&n, nullptr);

  std::vector<double> v(n);
  dset.read(v.data(), H5::PredType::NATIVE_DOUBLE);

  return v;
}

auto load_opacity_table(const std::string &filename) -> DataBox {
  std::print("# Building opacity table...");
  OpacityTable tab;

  // Open file
  H5::H5File file(filename, H5F_ACC_RDONLY);

  // Read axes
  tab.X = read_1d_double_dataset(file, "opacity/rosseland/table/X");
  tab.Z = read_1d_double_dataset(file, "opacity/rosseland/table/Z");
  tab.logT = read_1d_double_dataset(file, "opacity/rosseland/table/logT");
  tab.logR = read_1d_double_dataset(file, "opacity/rosseland/table/logR");

  // Open kappa dataset
  H5::DataSet dset = file.openDataSet("opacity/rosseland/table/logkappa");
  H5::DataSpace file_space = dset.getSpace();

  int rank = file_space.getSimpleExtentNdims();
  std::vector<hsize_t> dims(rank);
  file_space.getSimpleExtentDims(dims.data());

  tab.nX = dims[0];
  tab.nZ = dims[1];
  tab.nT = dims[2];
  tab.nR = dims[3];

  if (tab.nX != tab.X.size() || tab.nZ != tab.Z.size() ||
      tab.nT != tab.logT.size() || tab.nR != tab.logR.size()) {
    throw_athelas_error("Dimension mismatch between axes and kappa dataset");
  }

  tab.kappa = HostArray4D<double>("kappa", tab.nX, tab.nZ, tab.nT, tab.nR);

  H5::DataSpace mem_space(rank, dims.data());
  dset.read(tab.kappa.data(), H5::PredType::NATIVE_DOUBLE, mem_space,
            file_space);

  DataBox db(tab.nX, tab.nZ, tab.nT, tab.nR);
  db.setRange(0, tab.logR[0], tab.logR.back(), db.dim(1));
  db.setRange(1, tab.logT[0], tab.logT.back(), db.dim(2));
  db.setRange(2, tab.Z[0], tab.Z.back(), db.dim(3));
  db.setRange(3, tab.X[0], tab.X.back(), db.dim(4));

  for (std::size_t l = 0; l < tab.nR; ++l) {
    for (std::size_t k = 0; k < tab.nT; ++k) {
      for (std::size_t j = 0; j < tab.nZ; ++j) {
        for (std::size_t i = 0; i < tab.nX; ++i) {
          db(i, j, k, l) = tab.kappa(i, j, k, l);
        }
      }
    }
  }

  std::println("... complete!\n");
  return db;
}

} // namespace athelas
