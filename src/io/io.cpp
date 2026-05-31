#include "io/io.hpp"

#include <array>
#include <cstddef>
#include <iomanip>
#include <map>
#include <print>
#include <sstream>
#include <string>
#include <vector>

#include "H5Cpp.h"

#include "build_info.hpp"
#include "geometry/mesh.hpp"
#include "limiters/slope_limiter.hpp"

namespace athelas {

using basis::NodalBasis;

namespace io {

/**
 * Write to standard output some initialization info
 * for the current simulation.
 **/
void print_simulation_parameters(Mesh &mesh, ProblemIn *pin) {
  const int nX = mesh.n_elements();
  const int nNodes = mesh.n_nodes();
  // NOTE: If I properly support more bases again, adjust here.
  const bool rad_enabled = pin->param()->get<bool>("physics.radiation.enabled");
  const bool gravity_enabled =
      pin->param()->get<bool>("physics.gravity.enabled");
  const bool comps_enabled =
      pin->param()->get<bool>("physics.composition.enabled");
  const bool ionization_enabled =
      pin->param()->get<bool>("physics.ionization.enabled");
  const bool heating_enabled =
      pin->param()->get<bool>("physics.heating.active");
  const bool engine_enabled = pin->param()->get<bool>("physics.engine.enabled");

  std::println("# --- General --- ");
  std::println("# Problem Name    : {}",
               pin->param()->get<std::string>("problem.name"));
  std::println("# CFL             : {}",
               pin->param()->get<double>("problem.cfl"));
  std::println("");

  std::println("# --- Grid Parameters --- ");
  std::println("# Mesh Elements  : {}", nX);
  std::println("# Number Nodes   : {}", nNodes);
  std::println("# Lower Boundary : {}", mesh.get_x_l());
  std::println("# Upper Boundary : {}", mesh.get_x_r());
  std::println("");

  std::println("# --- Physics Parameters --- ");
  std::println("# Radiation      : {}", rad_enabled);
  std::println("# Gravity        : {}", gravity_enabled);
  std::println("# Composition    : {}", comps_enabled);
  std::println("# Ionization     : {}", ionization_enabled);
  std::println("# Heating        : {}", heating_enabled);
  std::println("# Engine         : {}", engine_enabled);
  std::println("# EOS            : {}",
               pin->param()->get<std::string>("eos.type"));
  std::println("");

  std::println("# --- Discretization Parameters --- ");
  std::println("# Spatial Order  : {}", pin->param()->get<int>("basis.nnodes"));
  std::println("# Integrator     : {}",
               pin->param()->get<std::string>("time.integrator_string"));
  std::println("");

  std::println("# --- Fluid Parameters --- ");
  std::println("# Inner BC       : {}",
               pin->param()->get<std::string>("fluid.bc.i"));
  std::println("# Outer BC       : {}",
               pin->param()->get<std::string>("fluid.bc.o"));
  std::println("");

  std::println("# --- Fluid Limiter --- ");
  if (pin->param()->get<int>("basis.nnodes") == 1) {
    std::println("# Spatial Order 1: Slope limiter not applied.");
  }
  if (!pin->param()->get<bool>("fluid.limiter.enabled")) {
    std::println("# Limiter Disabled");
  } else {
    const auto limiter_type =
        pin->param()->get<std::string>("fluid.limiter.type");
    std::println("# Limiter        : {}", limiter_type);
  }
  std::println("");

  if (rad_enabled) {
    std::println("# --- Radiation Parameters --- ");
    std::println("# Spatial Order  : {}",
                 pin->param()->get<int>("basis.nnodes"));
    std::println("# Inner BC       : {}",
                 pin->param()->get<std::string>("radiation.bc.i"));
    std::println("# Outer BC       : {}",
                 pin->param()->get<std::string>("radiation.bc.o"));
    std::println("");

    std::println("# --- Radiation Limiter Parameters --- ");
    if (pin->param()->get<int>("basis.nnodes") == 1) {
      std::println("# Spatial Order 1: Slope limiter not applied.");
    }
    if (!pin->param()->get<bool>("radiation.limiter.enabled")) {
      std::println("# Limiter Disabled");
    } else {
      const auto limiter_type =
          pin->param()->get<std::string>("radiation.limiter.type");
      std::println("# Limiter        : {}", limiter_type);
    }
    std::println("");
  }

  if (gravity_enabled) {
    std::println("# --- Gravity Parameters --- ");
    std::println("# Model           : {}",
                 pin->param()->get<std::string>("gravity.model"));
    std::println("");
  }

  if (heating_enabled) {
    std::println("# Nickel         : {}",
                 pin->param()->get<bool>("physics.heating.nickel.enabled"));
    std::println("");
  }
}

/**
 * Write simulation output to disk
 **/

// Helper class for HDF5 output management
class HDF5Writer {
 private:
  H5::H5File file_;
  std::map<std::string, H5::Group> groups_;

 public:
  explicit HDF5Writer(const std::string &filename)
      : file_(filename, H5F_ACC_TRUNC) {}

  // Create group hierarchy
  void create_group(const std::string &path) {
    if (groups_.contains(path)) {
      return;
    }
    // Create parent groups recursively
    size_t pos = path.find_last_of('/');
    if (pos != std::string::npos && pos > 0) {
      std::string parent = path.substr(0, pos);
      create_group(parent);
    }
    groups_[path] = file_.createGroup(path);
  }

  // Write scalar metadata
  template <typename T>
  void write_scalar(const std::string &path, const T &value,
                    const H5::DataType &h5type) {
    std::array<hsize_t, 1> dim = {1};
    H5::DataSpace space(1, dim.data());
    H5::DataSet dataset = file_.createDataSet(path, h5type, space);
    dataset.write(&value, h5type);
  }

  // Write string metadata
  void write_string(const std::string &path, const std::string &value) {
    std::array<hsize_t, 1> dim = {1};
    H5::DataSpace space(1, dim.data());
    H5::StrType stringtype(H5::PredType::C_S1, H5T_VARIABLE);
    H5::DataSet dataset = file_.createDataSet(path, stringtype, space);
    dataset.write(value, stringtype);
  }

  // Write 1D vector metadata
  template <typename T>
  void write_vector(const std::string &path, const std::vector<T> &values,
                    const H5::DataType &h5type) {
    if (values.empty()) {
      // Avoid creating zero-length dataset (optional)
      std::cerr << "Warning: skipping empty vector for path " << path << "\n";
      return;
    }

    // Dataspace: 1D array of size values.size()
    std::array<hsize_t, 1> dims = {values.size()};
    H5::DataSpace space(1, dims.data());

    // Create dataset and write
    H5::DataSet dataset = file_.createDataSet(path, h5type, space);
    dataset.write(values.data(), h5type);
  }

  // Write attribute to a dataset or group
  template <typename T>
  void write_attribute(const std::string &obj_path,
                       const std::string &attr_name, const T &value,
                       const H5::DataType &h5type) {
    H5::DataSet dataset = file_.openDataSet(obj_path);
    std::array<hsize_t, 1> dim = {1};
    H5::DataSpace attr_space(1, dim.data());
    H5::Attribute attr = dataset.createAttribute(attr_name, h5type, attr_space);
    attr.write(h5type, &value);
  }

  // Write string attribute
  void write_string_attribute(const std::string &obj_path,
                              const std::string &attr_name,
                              const std::string &value) {
    H5::DataSet dataset = file_.openDataSet(obj_path);
    H5::StrType stringtype(H5::PredType::C_S1, H5T_VARIABLE);
    std::array<hsize_t, 1> dim = {1};
    H5::DataSpace attr_space(1, dim.data());
    H5::Attribute attr =
        dataset.createAttribute(attr_name, stringtype, attr_space);
    attr.write(stringtype, value);
  }

  template <typename ViewType>
  void write_view(const ViewType &view, const std::string &dataset_name) {
    static_assert(Kokkos::is_view<ViewType>::value,
                  "write_view expects a Kokkos::View");
    using value_type = typename ViewType::value_type;
    std::vector<hsize_t> dims(view.rank());
    for (size_t r = 0; r < view.rank(); ++r) {
      dims[r] = static_cast<hsize_t>(view.extent(r));
    }
    H5::DataSpace file_space(view.rank(), dims.data());
    H5::DataSet dataset = file_.createDataSet(
        dataset_name, h5_predtype<value_type>(), file_space);
    using HostMirror = typename ViewType::HostMirror;
    HostMirror host_view = Kokkos::create_mirror_view(view);
    Kokkos::deep_copy(host_view, view);
    dataset.write(host_view.data(), h5_predtype<value_type>(),
                  file_space, // mem space
                  file_space); // file space
  }

  // Write MeshState field with metadata
  void write_field(const MeshState &mesh_state, const std::string &field_name,
                   const std::string &group_path, int stage = 0) {
    const auto &metadata = mesh_state.get_metadata(field_name);
    const auto policy = policy_string(metadata);
    std::string dataset_path = group_path + "/" + field_name;

    // NOTE: we could support writing all stages.
    // Not sure why though.
    auto data = mesh_state(stage).get_field(field_name);
    write_view(data, dataset_path);

    // Write metadata as attributes
    write_string_attribute(dataset_path, "description", metadata.description);
    write_string_attribute(dataset_path, "policy", policy);
    write_attribute(dataset_path, "rank", metadata.rank,
                    H5::PredType::NATIVE_INT);

    int nvars = mesh_state.nvars(field_name);
    write_attribute(dataset_path, "nvars", nvars, H5::PredType::NATIVE_INT);

    // Write variable names if available
    auto var_names = mesh_state.get_variable_names(field_name);
    if (!var_names.empty()) {
      std::string var_str;
      for (size_t i = 0; i < var_names.size(); ++i) {
        if (i > 0) {
          var_str += ",";
        }
        var_str += var_names[i];
      }
      write_string_attribute(dataset_path, "variables", var_str);
    }
  }

  // Write all fields from MeshState
  void write_all_fields(const MeshState &mesh_state,
                        const std::string &base_group, int stage = 0) {
    create_group(base_group);

    for (const auto &field_name : mesh_state.list_fields()) {
      // skip list
      if (field_name == "interface") {
        continue;
      }
      write_field(mesh_state, field_name, base_group, stage);
    }
  }

  // Write variable metadata
  void write_variable_metadata(
      const MeshState &mesh_state,
      const std::string &base_group = "/metadata/variables") {
    create_group(base_group);

    for (const auto &field_name : mesh_state.list_fields()) {
      auto var_names = mesh_state.get_variable_names(field_name);

      if (var_names.empty()) {
        // No named variables - create generic entries
        int nvars = mesh_state.nvars(field_name);
        for (int i = 0; i < nvars; ++i) {
          std::string var_name = field_name + "_" + std::to_string(i);
          std::string var_path = base_group;
          var_path.append("/").append(var_name);
          create_group(var_path);
          write_string(var_path + "/location", field_name);
          write_scalar(var_path + "/index", i, H5::PredType::NATIVE_INT);
        }
      } else {
        // Named variables
        for (size_t i = 0; i < var_names.size(); ++i) {
          std::string var_path = base_group + "/" + var_names[i];
          create_group(var_path);
          write_string(var_path + "/location", field_name);
          write_scalar(var_path + "/index", static_cast<int>(i),
                       H5::PredType::NATIVE_INT);
        }
      }
    }
  }

  // Write field registry metadata
  void write_field_registry(
      const MeshState &mesh_state,
      const std::string &base_group = "/metadata/field_registry") {
    create_group(base_group);

    for (const auto &field_name : mesh_state.list_fields()) {
      const auto &metadata = mesh_state.get_metadata(field_name);
      const auto policy = policy_string(metadata);
      std::string field_path = base_group;
      field_path.append("/").append(field_name);
      create_group(field_path);

      write_string(field_path + "/description", metadata.description);
      write_string(field_path + "/policy", policy);
      write_scalar(field_path + "/rank", metadata.rank,
                   H5::PredType::NATIVE_INT);
      write_scalar(field_path + "/nvars", mesh_state.nvars(field_name),
                   H5::PredType::NATIVE_INT);
    }
  }
};

void write_output(const MeshState &mesh_state, Mesh &mesh, ProblemIn *pin,
                  const std::string &filename, const SimInfo &info) {
  HDF5Writer writer(filename);

  // Create structure
  writer.create_group("/params");
  writer.create_group("/mesh");
  writer.create_group("/fields");
  writer.create_group("/metadata");
  writer.create_group("/info");
  writer.create_group("/basis");
  if (mesh_state.enabled("radiation")) {
    writer.create_group("/basis/radiation");
  }

  // Write simulation info.
  writer.write_scalar("/info/last_cycle", info.last_cycle,
                      H5::PredType::NATIVE_INT);
  writer.write_scalar("/info/time", info.time, H5::PredType::NATIVE_DOUBLE);
  writer.write_scalar("/info/dt", info.dt, H5::PredType::NATIVE_DOUBLE);
  writer.write_scalar("/info/last_out_h5", info.last_out_h5,
                      H5::PredType::NATIVE_INT);
  writer.write_scalar("/info/last_out_hist", info.last_out_hist,
                      H5::PredType::NATIVE_INT);
  writer.write_scalar("/info/n_stages", mesh_state.n_stages(),
                      H5::PredType::NATIVE_INT);

  // Write the mesh.
  writer.write_view(mesh.widths(), "/mesh/dr");
  writer.write_view(mesh.centers(), "/mesh/r");
  writer.write_view(mesh.x_l(), "/mesh/x_l");
  writer.write_view(mesh.nodal_grid(), "/mesh/r_q");
  writer.write_view(mesh.enclosed_mass(), "/mesh/enclosed_mass");
  writer.write_view(mesh.mass(), "/mesh/dm");
  writer.write_view(mesh.sqrt_gm(), "/mesh/sqrt_gm");

  const auto &fluid_basis = mesh_state.fluid_basis();
  auto phi_fluid = fluid_basis.phi();
  auto dphi_fluid = fluid_basis.dphi();
  writer.write_view(phi_fluid, "/basis/hi");
  writer.write_view(dphi_fluid, "/basis/dphi");
  writer.write_view(mesh.nodes(), "/basis/nodes");
  writer.write_view(mesh.weights(), "/basis/weights");
  if (mesh_state.enabled("radiation")) {
    const auto &basis = mesh_state.rad_basis();
    auto phi = basis.phi();
    auto dphi = basis.dphi();
    writer.write_view(phi, "/basis/radiation/phi");
    writer.write_view(dphi, "/basis/radiation/dphi");
  }

  if (mesh_state.enabled("composition")) {
    const auto *const comps = mesh_state(0).comps();
    auto number_density = comps->number_density();
    auto ye = comps->ye();
    auto species = comps->charge();
    auto neutron_number = comps->neutron_number();
    auto inv_atomic_mass = comps->inverse_atomic_mass();
    auto abar = comps->abar();
    auto ne = comps->electron_number_density();

    writer.create_group("/composition");
    writer.write_view(species, "composition/charge");
    writer.write_view(neutron_number, "composition/neutron_number");
    writer.write_view(inv_atomic_mass, "composition/inv_atomic_mass");
    writer.write_view(abar, "composition/abar");
    writer.write_view(number_density, "composition/number_density");
    writer.write_view(ye, "composition/ye");
    writer.write_view(ne, "composition/ne");

    // Serialize the species_indexer so restart preserves any custom aliases
    // a pgen added (and so we don't need to re-derive standard ones from
    // (Z, N) heuristics). Currently every value is an int (species index).
    auto *indexer = comps->species_indexer();
    writer.create_group("/composition/species_indexer");
    for (const auto &key : indexer->keys()) {
      const std::type_index t = indexer->get_type(key);
      if (t == typeid(int)) {
        writer.write_scalar("/composition/species_indexer/" + key,
                            indexer->get<int>(key), H5::PredType::NATIVE_INT);
      }
      // Other value types intentionally skipped — extend here when needed.
    }
  }

  if (mesh_state.enabled("ionization")) {
    auto *const ionization_state = mesh_state(0).ionization_state();
    auto ybar = ionization_state->ybar();
    auto ionization_fractions = ionization_state->ionization_fractions();
    auto zbars = ionization_state->zbar();

    writer.create_group("/ionization");
    writer.write_view(ybar, "ionization/ybar");
    writer.write_view(zbars, "ionization/zbar");
    writer.write_view(ionization_fractions, "ionization/ionization_fractions");
  }

  // --- deal with params ---
  auto *params = pin->param();
  for (auto const &key : params->keys()) {
    const std::type_index t = params->get_type(key);

    // ---------------------------
    // Dispatch based on type_index
    // ---------------------------
    if (t == typeid(bool)) {
      auto v = params->get<bool>(key);
      writer.write_scalar("params/" + key, v, H5::PredType::NATIVE_HBOOL);
    } else if (t == typeid(int)) {
      auto v = params->get<int>(key);
      writer.write_scalar("params/" + key, v, H5::PredType::NATIVE_INT);
    } else if (t == typeid(double)) {
      auto v = params->get<double>(key);
      writer.write_scalar("params/" + key, v, H5::PredType::NATIVE_DOUBLE);
    } else if (t == typeid(std::string)) {
      auto v = params->get<std::string>(key);
      writer.write_string("params/" + key, v);
    } else if (t == typeid(std::vector<int>)) {
      auto v = params->get<std::vector<int>>(key);
      writer.write_vector("params/" + key, v, H5::PredType::NATIVE_INT);
    } else if (t == typeid(std::vector<double>)) {
      auto v = params->get<std::vector<double>>(key);
      writer.write_vector("params/" + key, v, H5::PredType::NATIVE_DOUBLE);
    } else if (t == typeid(std::array<double, 2>)) {
      auto v = params->get<std::array<double, 2>>(key);
      std::vector<double> as_vec(v.begin(), v.end());
      writer.write_vector("params/" + key, as_vec, H5::PredType::NATIVE_DOUBLE);
      writer.write_string_attribute("params/" + key, "param_kind", "array_2");
    } else if (t == typeid(std::array<double, 3>)) {
      auto v = params->get<std::array<double, 3>>(key);
      std::vector<double> as_vec(v.begin(), v.end());
      writer.write_vector("params/" + key, as_vec, H5::PredType::NATIVE_DOUBLE);
      writer.write_string_attribute("params/" + key, "param_kind", "array_3");
    }
  }

  // Write all fields
  constexpr int stage = 0;
  writer.write_all_fields(mesh_state, "/fields", stage);

  // Write metadata
  writer.write_field_registry(mesh_state);
  writer.write_variable_metadata(mesh_state);

  // build provenance
  writer.create_group("/metadata/build");
  writer.write_string("/metadata/build/git_hash",
                      athelas::build_info::GIT_HASH);
  writer.write_string("/metadata/build/compiler",
                      athelas::build_info::COMPILER);
  writer.write_string("/metadata/build/timestamp", build_info::BUILD_TIMESTAMP);
  writer.write_string("/metadata/build/arch", build_info::ARCH);
  writer.write_string("/metadata/build/os", build_info::OS);
  writer.write_string("/metadata/build/optimization", build_info::OPTIMIZATION);
}

// Generate filename with proper padding
auto generate_filename(const std::string &problem_name,
                       const std::string &output_dir, int i_write,
                       int max_digits = 4) -> std::string {
  std::ostringstream oss;
  oss << output_dir << "/";
  oss << problem_name << "_";

  if (i_write != -1) {
    oss << std::setfill('0') << std::setw(max_digits) << i_write;
  } else {
    oss << "final";
  }

  oss << ".ath";
  return oss.str();
}

/**
 * @brief write to hdf5
 */
void write_output(const MeshState &mesh_state, Mesh &mesh, SlopeLimiter *SL,
                  ProblemIn *pin, const SimInfo &info, int i_write) {
  Kokkos::Profiling::pushRegion("IO");
  Kokkos::Profiling::pushRegion("HDF5");
  Kokkos::Profiling::pushRegion("Out");

  // Generate filename
  static constexpr int max_digits = 6;
  const auto &output_dir = pin->param()->get_ref<std::string>("output.dir");
  const auto &problem_name = pin->param()->get_ref<std::string>("problem.name");
  std::string filename =
      generate_filename(problem_name, output_dir, i_write, max_digits);

  write_output(mesh_state, mesh, pin, filename, info);

  Kokkos::Profiling::popRegion();
  Kokkos::Profiling::popRegion();
  Kokkos::Profiling::popRegion();
}
} // namespace io
} // namespace athelas
