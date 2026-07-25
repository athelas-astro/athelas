#include "io/restart.hpp"

#include <array>
#include <string>
#include <vector>

#include "H5Cpp.h"

#include "composition/compdata.hpp"
#include "geometry/mesh.hpp"
#include "interface/state.hpp"
#include "io/io.hpp"
#include "utils/error.hpp"

namespace athelas::io {

RestartReader::RestartReader(const std::string &filename)
    : file_(filename, H5F_ACC_RDONLY) {}

template <typename T>
auto RestartReader::read_scalar(const std::string &path) const -> T {
  H5::DataSet ds = file_.openDataSet(path);
  T value{};
  ds.read(&value, h5_predtype<T>());
  return value;
}

auto RestartReader::read_string(const std::string &path) const -> std::string {
  H5::DataSet ds = file_.openDataSet(path);
  H5::StrType stringtype(H5::PredType::C_S1, H5T_VARIABLE);
  std::string value;
  ds.read(value, stringtype);
  return value;
}

template <typename T>
auto RestartReader::read_vector(const std::string &path) const
    -> std::vector<T> {
  H5::DataSet ds = file_.openDataSet(path);
  H5::DataSpace space = ds.getSpace();
  const int ndims = space.getSimpleExtentNdims();
  if (ndims != 1) {
    throw_athelas_error("Restart: expected 1D dataset at " + path);
  }
  hsize_t n = 0;
  space.getSimpleExtentDims(&n);
  std::vector<T> out(static_cast<size_t>(n));
  ds.read(out.data(), h5_predtype<T>());
  return out;
}

auto RestartReader::has(const std::string &path) const -> bool {
  return H5Lexists(file_.getId(), path.c_str(), H5P_DEFAULT) > 0;
}

auto RestartReader::list_group(const std::string &group_path) const
    -> std::vector<std::string> {
  H5::Group grp = file_.openGroup(group_path);
  const hsize_t n = grp.getNumObjs();
  std::vector<std::string> names;
  names.reserve(n);
  for (hsize_t i = 0; i < n; ++i) {
    names.emplace_back(grp.getObjnameByIdx(i));
  }
  return names;
}

auto RestartReader::has_attribute(const std::string &dataset_path,
                                  const std::string &attr_name) const -> bool {
  H5::DataSet ds = file_.openDataSet(dataset_path);
  return ds.attrExists(attr_name);
}

auto RestartReader::read_string_attribute(const std::string &dataset_path,
                                          const std::string &attr_name) const
    -> std::string {
  H5::DataSet ds = file_.openDataSet(dataset_path);
  H5::Attribute attr = ds.openAttribute(attr_name);
  H5::StrType stringtype(H5::PredType::C_S1, H5T_VARIABLE);
  std::string value;
  attr.read(stringtype, value);
  return value;
}

auto RestartReader::dataset_extent(const std::string &dataset_path) const
    -> std::vector<hsize_t> {
  H5::DataSet ds = file_.openDataSet(dataset_path);
  H5::DataSpace space = ds.getSpace();
  const int ndims = space.getSimpleExtentNdims();
  std::vector<hsize_t> dims(ndims);
  space.getSimpleExtentDims(dims.data());
  return dims;
}

// Explicit instantiations for the scalar/vector types we actually use.
// Bool is intentionally omitted: h5_predtype<bool> isn't defined; the writer
// stores bools as NATIVE_HBOOL and Phase 3 reads them via a dedicated path.
template auto RestartReader::read_scalar<int>(const std::string &) const -> int;
template auto RestartReader::read_scalar<long>(const std::string &) const
    -> long;
template auto RestartReader::read_scalar<double>(const std::string &) const
    -> double;
template auto RestartReader::read_vector<int>(const std::string &) const
    -> std::vector<int>;
template auto RestartReader::read_vector<double>(const std::string &) const
    -> std::vector<double>;

// ---------------------------------------------------------------------------
// High-level helpers
// ---------------------------------------------------------------------------

auto load_info_from_h5(const RestartReader &reader) -> SimInfo {
  return SimInfo{
      .time = reader.read_scalar<double>("/info/time"),
      .dt = reader.read_scalar<double>("/info/dt"),
      .last_cycle = reader.read_scalar<int>("/info/last_cycle"),
      .last_out_h5 = reader.read_scalar<int>("/info/last_out_h5"),
      .last_out_hist = reader.read_scalar<int>("/info/last_out_hist"),
  };
}

auto load_package_restart_scalars(const RestartReader &reader)
    -> PackageRestartState {
  PackageRestartState state;
  if (!reader.has("/package_state")) {
    return state;
  }
  for (const auto &package_name : reader.list_group("/package_state")) {
    PackageRestartScalars scalars;
    const std::string package_path = "/package_state/" + package_name;
    for (const auto &scalar_name : reader.list_group(package_path)) {
      scalars.emplace_back(scalar_name, reader.read_scalar<double>(
                                            package_path + "/" + scalar_name));
    }
    state.emplace(package_name, std::move(scalars));
  }
  return state;
}

void load_grid_from_h5(Mesh &mesh, const RestartReader &reader) {
  reader.read_view(mesh.widths(), "/mesh/dr");
  reader.read_view(mesh.centers(), "/mesh/r");
  reader.read_view(mesh.x_l(), "/mesh/x_l");
  reader.read_view(mesh.nodal_grid(), "/mesh/r_q");
  reader.read_view(mesh.enclosed_mass(), "/mesh/enclosed_mass");
  reader.read_view(mesh.mass(), "/mesh/dm");
  reader.read_view(mesh.dm_deta(), "/mesh/dm_deta");
  reader.read_view(mesh.sqrt_gm(), "/mesh/sqrt_gm");
}

namespace {

// Read a registered field's data into the underlying Kokkos view, dispatching
// on rank and policy. The writer writes stage 0 for staged fields, so we mirror
// that here.
void read_field(const MeshState &mesh_state, const RestartReader &reader,
                const std::string &name) {
  const auto &meta = mesh_state.get_metadata(name);
  const std::string path = "/fields/" + name;

  if (meta.policy == DataPolicy::Staged || meta.policy == DataPolicy::TwoCopy) {
    switch (meta.rank) {
    case 1:
      reader.read_view(mesh_state.get_field<AthelasArray1D<double>>(name, 0),
                       path);
      break;
    case 2:
      reader.read_view(mesh_state.get_field<AthelasArray2D<double>>(name, 0),
                       path);
      break;
    case 3:
      reader.read_view(mesh_state.get_field<AthelasArray3D<double>>(name, 0),
                       path);
      break;
    default:
      throw_athelas_error("Restart: unsupported staged rank for " + name);
    }
  } else {
    switch (meta.rank) {
    case 1:
      reader.read_view(mesh_state.get_field<AthelasArray1D<double>>(name),
                       path);
      break;
    case 2:
      reader.read_view(mesh_state.get_field<AthelasArray2D<double>>(name),
                       path);
      break;
    case 3:
      reader.read_view(mesh_state.get_field<AthelasArray3D<double>>(name),
                       path);
      break;
    case 4:
      reader.read_view(mesh_state.get_field<AthelasArray4D<double>>(name),
                       path);
      break;
    default:
      throw_athelas_error("Restart: unsupported OneCopy rank for " + name);
    }
  }
}

} // namespace

void load_fields_from_h5(const MeshState &mesh_state,
                         const RestartReader &reader) {
  for (const auto &name : mesh_state.list_fields()) {
    // interface (vstar) is regenerated by the next hydro update — the writer
    // skips it, so the file has nothing to read.
    if (name == "interface" || name == "eos_lambda_avg" || name == "dtau_dt" ||
        name == "diagnostics") {
      continue;
    }
    read_field(mesh_state, reader, name);
  }
}

void load_composition_from_h5(atom::CompositionData &comps,
                              const RestartReader &reader) {
  reader.read_view(comps.charge(), "/composition/charge");
  reader.read_view(comps.neutron_number(), "/composition/neutron_number");
  reader.read_view(comps.inverse_atomic_mass(), "/composition/inv_atomic_mass");
  reader.read_view(comps.abar(), "/composition/abar");
  reader.read_view(comps.number_density(), "/composition/number_density");
  reader.read_view(comps.ye(), "/composition/ye");
  reader.read_view(comps.electron_number_density(), "/composition/ne");

  // Restore the species_indexer from the dump rather than re-deriving entries
  // from (Z, N) heuristics — preserves custom aliases pgens may have added.
  auto *indexer = comps.species_indexer();
  if (reader.has("/composition/species_indexer")) {
    for (const auto &name : reader.list_group("/composition/species_indexer")) {
      const int idx =
          reader.read_scalar<int>("/composition/species_indexer/" + name);
      indexer->add(name, idx);
    }
  }
}

void load_ionization_from_h5(atom::IonizationState &ion,
                             const RestartReader &reader) {
  reader.read_view(ion.ybar(), "/ionization/ybar");
  reader.read_view(ion.zbar(), "/ionization/zbar");
  reader.read_view(ion.ionization_fractions(),
                   "/ionization/ionization_fractions");
}

namespace {

// Read the dataset at /params/<key> and add it to params under that key.
// Dispatches by HDF5 type class + dataspace size, with the "param_kind"
// attribute on the dataset distinguishing std::array<double, N> from a
// same-shape vector<double>.
void read_param_dataset(Params &params, const RestartReader &reader,
                        const std::string &key) {
  const std::string path = "/params/" + key;
  H5::DataSet ds = reader.file().openDataSet(path);
  const H5T_class_t type_class = ds.getTypeClass();

  if (type_class == H5T_STRING) {
    params.add(key, reader.read_string(path));
    return;
  }

  H5::DataSpace space = ds.getSpace();
  if (space.getSimpleExtentNdims() != 1) {
    throw_athelas_error("Restart: param '" + key + "' has unexpected rank");
  }
  hsize_t n = 0;
  space.getSimpleExtentDims(&n);

  if (type_class == H5T_INTEGER) {
    H5::IntType itype = ds.getIntType();
    const bool is_bool =
        H5Tequal(itype.getId(), H5T_NATIVE_HBOOL) > 0 || itype.getSize() == 1;
    if (n == 1) {
      if (is_bool) {
        // HBOOL is stored as unsigned char; read it that way then convert.
        unsigned char raw = 0;
        ds.read(&raw, H5::PredType::NATIVE_HBOOL);
        params.add(key, static_cast<bool>(raw));
      } else {
        params.add(key, reader.read_scalar<int>(path));
      }
    } else {
      params.add(key, reader.read_vector<int>(path));
    }
    return;
  }

  if (type_class == H5T_FLOAT) {
    if (n == 1) {
      params.add(key, reader.read_scalar<double>(path));
    } else if (reader.has_attribute(path, "param_kind")) {
      const std::string kind = reader.read_string_attribute(path, "param_kind");
      const auto vec = reader.read_vector<double>(path);
      if (kind == "array_2") {
        if (vec.size() != 2) {
          throw_athelas_error("Restart: '" + key +
                              "' tagged array_2 but size != 2");
        }
        params.add(key, std::array<double, 2>{vec[0], vec[1]});
      } else if (kind == "array_3") {
        if (vec.size() != 3) {
          throw_athelas_error("Restart: '" + key +
                              "' tagged array_3 but size != 3");
        }
        params.add(key, std::array<double, 3>{vec[0], vec[1], vec[2]});
      } else {
        throw_athelas_error("Restart: '" + key + "' has unknown param_kind '" +
                            kind + "'");
      }
    } else {
      params.add(key, reader.read_vector<double>(path));
    }
    return;
  }

  throw_athelas_error("Restart: param '" + key +
                      "' has unsupported HDF5 type class");
}

} // namespace

void load_params_from_h5(Params &params, const RestartReader &reader) {
  for (const auto &key : reader.list_group("/params")) {
    read_param_dataset(params, reader, key);
  }
}

} // namespace athelas::io
