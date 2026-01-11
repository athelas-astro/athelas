#include <array>
#include <cstddef>
#include <iomanip>
#include <map>
#include <print>
#include <sstream>
#include <string>
#include <vector>

#include "H5Cpp.h"

#include "basis/polynomial_basis.hpp"
#include "build_info.hpp"
#include "geometry/grid.hpp"
#include "io/io.hpp"
#include "limiters/slope_limiter.hpp"

namespace athelas {

using basis::ModalBasis;

namespace io {

// using namespace athelas::build_info;

/**
 * Write to standard output some initialization info
 * for the current simulation.
 **/
void print_simulation_parameters(GridStructure &grid, ProblemIn *pin) {
  const int nX = grid.n_elements();
  const int nNodes = grid.n_nodes();
  // NOTE: If I properly support more bases again, adjust here.
  const std::string basis_name = "Legendre";
  const bool rad_enabled = pin->param()->get<bool>("physics.rad_active");
  const bool gravity_enabled =
      pin->param()->get<bool>("physics.gravity_active");
  const bool comps_enabled =
      pin->param()->get<bool>("physics.composition_enabled");
  const bool ionization_enabled =
      pin->param()->get<bool>("physics.ionization_enabled");
  const bool heating_enabled =
      pin->param()->get<bool>("physics.heating.active");

  std::println("# --- General --- ");
  std::println("# Problem Name    : {}",
               pin->param()->get<std::string>("problem.problem"));
  std::println("# CFL             : {}",
               pin->param()->get<double>("problem.cfl"));
  std::println("");

  std::println("# --- Grid Parameters --- ");
  std::println("# Mesh Elements  : {}", nX);
  std::println("# Number Nodes   : {}", nNodes);
  std::println("# Lower Boundary : {}", grid.get_x_l());
  std::println("# Upper Boundary : {}", grid.get_x_r());
  std::println("");

  std::println("# --- Physics Parameters --- ");
  std::println("# Radiation      : {}", rad_enabled);
  std::println("# Gravity        : {}", gravity_enabled);
  std::println("# Composition    : {}", comps_enabled);
  std::println("# Ionization     : {}", ionization_enabled);
  std::println("# Heating        : {}", heating_enabled);
  std::println("# EOS            : {}",
               pin->param()->get<std::string>("eos.type"));
  std::println("");

  std::println("# --- Discretization Parameters --- ");
  std::println("# Basis          : {}", basis_name);
  std::println("# Integrator     : {}",
               pin->param()->get<std::string>("time.integrator_string"));
  std::println("");

  std::println("# --- Fluid Parameters --- ");
  std::println("# Spatial Order  : {}", pin->param()->get<int>("fluid.porder"));
  std::println("# Inner BC       : {}",
               pin->param()->get<std::string>("fluid.bc.i"));
  std::println("# Outer BC       : {}",
               pin->param()->get<std::string>("fluid.bc.o"));
  std::println("");

  std::println("# --- Fluid Limiter --- ");
  if (pin->param()->get<int>("fluid.porder") == 1) {
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
                 pin->param()->get<int>("radiation.porder"));
    std::println("# Inner BC       : {}",
                 pin->param()->get<std::string>("radiation.bc.i"));
    std::println("# Outer BC       : {}",
                 pin->param()->get<std::string>("radiation.bc.o"));
    std::println("");

    std::println("# --- Radiation Limiter Parameters --- ");
    if (pin->param()->get<int>("radiation.porder") == 1) {
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
    std::println("# Modal           : {}",
                 pin->param()->get<std::string>("gravity.modelstring"));
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
    };

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
};

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

  oss << ".h5";
  return oss.str();
}

/**
 * @brief write to hdf5
 *
 * Bad stuff in here.
 */
void write_state(const StageData &stage_data, GridStructure &grid,
                 SlopeLimiter *SL, ProblemIn *pin, double time, int order,
                 int i_write, bool do_rad) {
  Kokkos::Profiling::pushRegion("HDF5 IO");

  const bool ionization_active = stage_data.ionization_enabled();
  const bool composition_active = stage_data.composition_enabled();

  auto uCF = stage_data.get_field("u_cf");
  auto uPF = stage_data.get_field("u_pf");
  auto uAF = stage_data.get_field("u_af");

  // Grid parameters
  const int nX = grid.n_elements();

  // Generate filename
  static constexpr int max_digits = 6;
  const auto &output_dir = pin->param()->get_ref<std::string>("output.dir");
  const auto &problem_name =
      pin->param()->get_ref<std::string>("problem.problem");
  std::string filename =
      generate_filename(problem_name, output_dir, i_write, max_digits);

  // Create HDF5 writer
  HDF5Writer writer(filename);

  // Create group structure
  writer.create_group("/metadata");
  writer.create_group("/metadata/build");
  writer.create_group("/grid");
  writer.create_group("/variables");
  writer.create_group("/parameters");

  // TODO(astrobarker): Figure out what to do with fluid vs rad limiters
  //  if (order > 1 && limiter_active) {
  //    writer.create_group("/limiter");
  //    writer.write_view(limited(SL), filename, "limiter/limited");
  //  }

  if (composition_active) {
    writer.create_group("/composition");
  }

  // write views
  writer.write_view(uCF, "/variables/conserved");
  writer.write_view(uPF, "/variables/primitive");
  writer.write_view(uAF, "/variables/auxiliary");
  writer.write_view(grid.widths(), "/grid/dx");
  writer.write_view(grid.centers(), "/grid/x");
  writer.write_view(grid.nodal_grid(), "/grid/x_nodal");
  writer.write_view(grid.enclosed_mass(), "/grid/enclosed_mass");

  if (composition_active) {
    // const auto mass_fractions = stage_data.mass_fractions("u_cf");
    const auto *const comps = stage_data.comps();
    auto charges = comps->charge();
    auto neutron_numbers = comps->neutron_number();
    auto ye = comps->ye();
    writer.write_view(charges, "/composition/proton_number");
    writer.write_view(neutron_numbers, "/composition/neutron_number");
    // writer.write_view(mass_fractions, "/composition/mass_fractions");
    writer.write_view(ye, "/composition/ye");
  }

  if (ionization_active) {
    auto *const ion_state = stage_data.ionization_state();
    auto ionization_fractions = ion_state->ionization_fractions();
    auto e_ionization = ion_state->e_ion_corr();
    writer.write_view(ionization_fractions,
                      "/composition/ionization_fractions");
    writer.write_view(e_ionization, "/composition/ionization_energy");
  }

  // metadata
  writer.write_scalar("/metadata/nx", nX, H5::PredType::NATIVE_INT);
  writer.write_scalar("/metadata/order", order, H5::PredType::NATIVE_INT);
  writer.write_scalar("/metadata/time", time, H5::PredType::NATIVE_DOUBLE);

  // build information
  writer.write_string("/metadata/build/git_hash",
                      athelas::build_info::GIT_HASH);
  writer.write_string("/metadata/build/compiler",
                      athelas::build_info::COMPILER);
  writer.write_string("/metadata/build/timestamp", build_info::BUILD_TIMESTAMP);
  writer.write_string("/metadata/build/arch", build_info::ARCH);
  writer.write_string("/metadata/build/os", build_info::OS);
  writer.write_string("/metadata/build/optimization", build_info::OPTIMIZATION);

  // deal with params
  const auto keys = pin->param()->keys();
  // probably a more elegant loop pattern
  for (const auto &key : keys) {
    auto t = pin->param()->get_type(key);
    if (t == typeid(std::string)) {
      writer.write_string("parameters/" + key,
                          pin->param()->get<std::string>(key));
    } else if (t == typeid(int)) {
      writer.write_scalar("parameters/" + key, pin->param()->get<int>(key),
                          H5::PredType::NATIVE_INT);
    } else if (t == typeid(double)) {
      writer.write_scalar("parameters/" + key, pin->param()->get<double>(key),
                          H5::PredType::NATIVE_DOUBLE);
    } else {
      // If the type cannot be matched, write a default string.
      writer.write_string("parameters/" + key, "Null");
    }
  }
  Kokkos::Profiling::popRegion();
}

/**
 * Write Modal basis coefficients and mass matrix
 **/
void write_basis(const ModalBasis &basis, const std::string &problem_name) {
  std::string fn = problem_name;
  fn.append("_basis");
  fn.append(".h5");

  const char *filename = fn.c_str();

  // Create HDF5 writer
  HDF5Writer writer(filename);

  // Create group structure
  writer.create_group("/basis");

  writer.write_view(basis.phi(), "/basis/phi");
  writer.write_view(basis.dphi(), "/basis/dphi");
}

} // namespace io
} // namespace athelas
