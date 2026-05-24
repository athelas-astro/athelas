#include "driver.hpp"
#include "basic_types.hpp"
#include "basis/nodal_basis.hpp"
#include "basis/polynomial_basis.hpp"
#include "engines/thermal.hpp"
#include "fluid/hydro_package.hpp"
#include "geometry/geometry_package.hpp"
#include "gravity/gravity_package.hpp"
#include "heating/nickel_package.hpp"
#include "history/quantities.hpp"
#include "interface/packages_base.hpp"
#include "io/io.hpp"
#include "kokkos_types.hpp"
#include "limiters/slope_limiter.hpp"
#include "loop_layout.hpp"
#include "pgen/initialization.hpp"
#include "pgen/problem_in.hpp"
#include "radiation/imex_radhydro_package.hpp"
#include "radiation/implicit_moments_package.hpp"
#include "interface/state.hpp"
#include "timestepper/timestepper.hpp"
#include "utils/error.hpp"

namespace athelas {

using basis::ModalBasis;
using basis::NodalBasis;
using io::write_output, io::print_simulation_parameters;

auto Driver::execute() -> int {
  static const auto nx = pin_->param()->get<int>("problem.nx");
  static const bool rad_active =
      pin_->param()->get<bool>("physics.radiation.enabled");

  // --- Timer ---
  Kokkos::Timer timer_zone_cycles;
  double zc_ws = 0.0; // zone cycles / wall second

  const double nlim = (pin_->param()->get<double>("problem.nlim")) < 0
                          ? std::numeric_limits<double>::infinity()
                          : pin_->param()->get<double>("problem.nlim");
  const auto ncycle_out = pin_->param()->get<int>("output.ncycle_out");
  const auto dt_init = pin_->param()->get<double>("output.dt_init");
  const auto dt_growth_frac =
      pin_->param()->get<double>("output.dt_growth_frac");
  const auto dt_hdf5 = pin_->param()->get<double>("output.dt_hdf5");
  const auto fixed_dt = pin_->param()->contains("output.dt_fixed");

  dt_ = dt_init;
  TimeStepInfo dt_info{.t = time_,
                       .dt = dt_,
                       .dt_coef_implicit = dt_,
                       .dt_coef = dt_,
                       .stage = 0,
                       .cycle = 1};

  // some startup io
  auto sd0 = mesh_state_(0);
  manager_->fill_derived(sd0, grid_, dt_info);
  print_simulation_parameters(grid_, pin_.get());
  write_output(mesh_state_, grid_, &sl_hydro_, pin_.get(), time_, 0);
  history_->write(mesh_state_, grid_, time_);

  // --- Evolution loop ---
  int cycle = 1;
  int i_out_h5 = 1; // output label, start 1
  int i_out_hist = 1; // output hist
  std::println("# Cycle      t       dt       zone_cycles / wall_second");
  while (time_ < t_end_ && cycle <= nlim) {
    dt_info.t = time_;
    dt_info.dt = dt_;
    dt_info.dt_coef_implicit = dt_;
    dt_info.dt_coef = dt_;
    dt_info.stage = 0;
    dt_info.cycle = cycle;

    if (!fixed_dt) {
      dt_ = std::min(manager_->min_timestep(mesh_state_(0), grid_, dt_info),
                     dt_ * dt_growth_frac);
    } else {
      dt_ = pin_->param()->get<double>("output.dt_fixed");
    }

    if (time_ + dt_ > t_end_) {
      dt_ = t_end_ - time_;
    }

    dt_info.dt = dt_;
    dt_info.dt_coef_implicit = dt_;
    dt_info.dt_coef = dt_;

    // This logic could probably be cleaner..
    if (!rad_active) {
      ssprk_.step(manager_.get(), mesh_state_, grid_, dt_info, &sl_hydro_);
    } else {
      try {
        ssprk_.step_imex(manager_.get(), mesh_state_, grid_, dt_info,
                         &sl_hydro_, &sl_rad_);
      } catch (const AthelasError &e) {
        std::cerr << e.what() << "\n";
        return AthelasExitCodes::FAILURE;
      } catch (const std::exception &e) {
        std::cerr << "Library Error: " << e.what() << "\n";
        return AthelasExitCodes::FAILURE;
      }
    }

    // Call the operator split time stepper
    // Likely will need work if implicit physics is split.
    if (operator_split_physics_) {
      dt_info.t = time_;
      dt_info.dt = dt_;
      dt_info.dt_coef_implicit = dt_;
      dt_info.dt_coef = dt_;
      dt_info.stage = 0;
      split_stepper_->step(split_manager_.get(), mesh_state_, grid_, dt_info);
    }

    time_ += dt_;
    dt_info.t = time_;
    dt_info.dt = dt_;
    dt_info.dt_coef_implicit = dt_;
    dt_info.dt_coef = dt_;
    dt_info.stage = 0;
    post_step_work();

    // Write state, other io
    if (time_ >= i_out_h5 * dt_hdf5) {
      manager_->fill_derived(sd0, grid_, dt_info);
      write_output(mesh_state_, grid_, &sl_hydro_, pin_.get(), time_, i_out_h5);
      i_out_h5 += 1;
    }

    // history
    if (time_ >= i_out_hist * pin_->param()->get<double>("output.hist_dt")) {
      history_->write(mesh_state_, grid_, time_);
      i_out_hist += 1;
    }

    // timer
    if (cycle % ncycle_out == 0) {
      zc_ws =
          static_cast<double>(ncycle_out) * nx / timer_zone_cycles.seconds();
      std::println("{} {:.5e} {:.5e} {:.5e}", cycle, time_, dt_, zc_ws);
      timer_zone_cycles.reset();
    }

    ++cycle;
  }

  manager_->fill_derived(sd0, grid_, dt_info);
  write_output(mesh_state_, grid_, &sl_hydro_, pin_.get(), time_, -1);

  return AthelasExitCodes::SUCCESS;
}

void Driver::initialize(ProblemIn *pin) { // NOLINT
  using fluid::HydroPackage;
  using geometry::GeometryPackage;
  using gravity::GravityPackage;
  using nickel::NickelHeatingPackage;
  using radiation::ImplicitRadiationMomentsPackage;
  using thermal_engine::ThermalEnginePackage;

  const auto nx = pin_->param()->get<int>("problem.nx");
  const int nnodes = pin_->param()->get<int>("basis.nnodes");
  // For nodal DG, u_cf is (ix, node, var); for modal, (ix, mode, var)
  const auto cfl =
      compute_cfl(pin_->param()->get<double>("problem.cfl"), nnodes);
  const bool rad_active = pin->param()->get<bool>("physics.radiation.enabled");
  const bool comps_active =
      pin->param()->get<bool>("physics.composition.enabled");

  // --- Set up mesh state ---
  // First set up the conserved fields.
  // Composition var names can't be known yet, so they are just set to
  // comps_0, ...
  int nvars_cons = 3;
  std::vector<std::string> varnames_cons = {"tau", "vel", "fluid_energy"};
  if (rad_active) {
    nvars_cons += 2;
    varnames_cons.emplace_back("rad_energy");
    varnames_cons.emplace_back("rad_momentum");
  }
  if (comps_active) {
    const auto ncomps = pin->param()->get<int>("composition.ncomps");
    nvars_cons += ncomps;
    for (int i = 0; i < ncomps; ++i) {
      const auto str = "comps_" + std::to_string(i);
      varnames_cons.emplace_back(str);
    }
  }
  mesh_state_.register_field("u_cf", DataPolicy::Staged, "Conserved variables",
                             varnames_cons, nx + 2, nnodes, nvars_cons);

  int nvars_aux = 3;
  mesh_state_.register_field("u_af", DataPolicy::OneCopy, "Auxiliary variables",
                             {"pressure", "gas temperature", "sound speed"},
                             nx + 2, nnodes + 2, nvars_aux);
  int nvars_prim = 3;
  mesh_state_.register_field("u_pf", DataPolicy::OneCopy, "Primitive variables",
                             {"density", "momentum", "sie"}, nx + 2, nnodes + 2,
                             nvars_prim);

  if (comps_active) {
    // TODO(astrobarker) [composition] Get rid of x_q nodal mass fractions
    const auto ncomps = pin->param()->get<int>("composition.ncomps");
    mesh_state_.register_field("x_q", DataPolicy::OneCopy,
                               "Nodal mass fractions", nx + 2, nnodes + 2,
                               ncomps);

    mesh_state_.register_field("bulk_composition", DataPolicy::OneCopy,
                               "bulk mass fractions", {"X", "Y", "Z"}, nx + 2,
                               nnodes + 2, 3);
  }

  mesh_state_.register_field("facedata", DataPolicy::Staged,
                             "Misc variable face data", {"vstar"}, nx + 2 + 1,
                             1);

  // auto info = mesh_state_.field_info();

  if (!restart_) {
    initialize_fields(mesh_state_, &grid_, pin);
    auto sd0 = mesh_state_(0);
    auto prims = sd0.get_field("u_pf");
    auto cons = sd0.get_field("u_cf");
    bc::fill_ghost_zones<3>(cons, &grid_, bcs_.get(), {0, 2});
    grid_.compute_mass(cons);

    auto nx = grid_.n_elements();
    const bool rad_active =
        pin_->param()->get<bool>("physics.radiation.enabled");
    auto fluid_basis = std::make_unique<NodalBasis>(prims, &grid_, nnodes, nx);
    mesh_state_.setup_fluid_basis(std::move(fluid_basis));
    if (rad_active) {
      auto radiation_basis =
          std::make_unique<NodalBasis>(prims, &grid_, nnodes, nx);
      mesh_state_.setup_rad_basis(std::move(radiation_basis));
    }
  }

  // now that all is said and done, perform post init work
  // We may need to do this before packages are constructed.
  post_init_work();

  const bool gravity_active =
      pin->param()->get<bool>("physics.gravity.enabled");
  const bool ni_heating_active =
      pin->param()->get<bool>("physics.heating.nickel.enabled");
  const bool geometry =
      pin->param()->get<std::string>("problem.geometry") == "spherical";
  const bool thermal_engine_active =
      pin->param()->get<bool>("physics.engine.thermal.enabled");

  bool split = false;

  const int n_stages = ssprk_.n_stages();
  auto sd0 = mesh_state_(0);

  // --- Init physics package manager ---
  // NOTE: Hydro/RadHydro should be registered first
  const bool pkg_active = true;
  if (rad_active) {
    const auto discretization =
        pin_->param()->get<std::string>("radiation.discretization");
    if (discretization == "implicit") {
      manager_->add_package(
          HydroPackage{pin, n_stages, nnodes, bcs_.get(), cfl, nx, pkg_active});
      manager_->add_package(ImplicitRadiationMomentsPackage{
          pin, n_stages, nnodes, bcs_.get(), nx, pkg_active});
    }
    if (discretization == "explicit") {
      manager_->add_package(RadHydroPackage{pin, n_stages, nnodes, bcs_.get(),
                                            cfl, nx, pkg_active});
    }
  } else [[unlikely]] {
    // pure Hydro
    manager_->add_package(
        HydroPackage{pin, n_stages, nnodes, bcs_.get(), cfl, nx, pkg_active});
  }
  if (gravity_active) {
    if (!pin->param()->get<bool>("physics.gravity.split")) {
      manager_->add_package(
          GravityPackage{pin, pin->param()->get<std::string>("gravity.model"),
                         pin->param()->get<double>("gravity.gval"), cfl,
                         n_stages, pkg_active});
    } else {
      split = true;
      split_manager_->add_package(
          GravityPackage{pin, pin->param()->get<std::string>("gravity.model"),
                         pin->param()->get<double>("gravity.gval"), cfl,
                         n_stages, pkg_active});
    }
  }
  if (ni_heating_active) {
    if (!pin->param()->get<bool>("physics.heating.nickel.split")) {
      manager_->add_package(NickelHeatingPackage{
          pin, sd0.comps()->species_indexer(), n_stages, nnodes, pkg_active});
    } else {
      split = true;
      split_manager_->add_package(NickelHeatingPackage{
          pin, sd0.comps()->species_indexer(), n_stages, nnodes, pkg_active});
    }
  }
  if (thermal_engine_active) {
    if (!pin->param()->get<bool>("physics.engine.thermal.split")) {
      manager_->add_package(
          ThermalEnginePackage{pin, sd0, &grid_, n_stages, pkg_active});
    } else {
      split = true;
      split_manager_->add_package(
          ThermalEnginePackage{pin, sd0, &grid_, n_stages, pkg_active});
    }
  }
  // TODO(astrobarker): [split, geometry] Could add option to split.. not
  // important..
  if (geometry) {
    manager_->add_package(GeometryPackage{pin, n_stages, pkg_active});
  }

  // set up operator split stepper
  if (split) {
    split_stepper_ = std::make_unique<OperatorSplitStepper>();
    operator_split_physics_ = true;
  }

  auto registered_pkgs = manager_->get_package_names();
  auto split_pkgs = split_manager_->get_package_names();
  std::print("# Registered Packages ::");
  for (auto name : registered_pkgs) {
    std::print(" {}", name);
  }
  for (auto name : split_pkgs) {
    std::print(" {} (operator split)", name);
  }
  std::print("\n\n");

  // --- Fill ghosts and apply limiters to initial condition ---
  auto ucf = sd0.get_field("u_cf");
  bc::fill_ghost_zones<3>(ucf, &grid_, bcs_.get(), {0, 2});
  if (rad_active) {
    bc::fill_ghost_zones<2>(ucf, &grid_, bcs_.get(), {3, 4});
  }
  auto cons = sd0.get_field("u_cf");
  apply_slope_limiter(&sl_hydro_, cons, grid_, sd0.fluid_basis(), sd0.eos());
  bel::apply_bound_enforcing_limiter(sd0, grid_);

  // --- Add history outputs ---
  // NOTE: Could be nice to have gravitational energy added
  // to total, conditionally.
  history_->add_quantity("Total Mass [g]", analysis::total_mass);
  history_->add_quantity("Total Energy [erg]", analysis::total_energy);
  history_->add_quantity("Total Fluid Energy [erg]",
                         analysis::total_fluid_energy);
  history_->add_quantity("Total Fluid Momentum [g cm / s]",
                         analysis::total_fluid_momentum);
  history_->add_quantity("Total Internal Energy [erg]",
                         analysis::total_internal_energy);
  history_->add_quantity("Total Kinetic Energy [erg]",
                         analysis::total_kinetic_energy);
  history_->add_quantity("Total Momentum [g cm / s]", analysis::total_momentum);

  if (gravity_active) {
    history_->add_quantity("Total Gravitational Energy [erg]",
                           analysis::total_gravitational_energy);
  }

  if (rad_active) {
    history_->add_quantity("Total Radiation Momentum [g cm / s]",
                           analysis::total_rad_momentum);
    history_->add_quantity("Total Radiation Energy [erg]",
                           analysis::total_rad_energy);
  }

  // total nickel56, cobalt56, iron56
  if (ni_heating_active) {
    history_->add_quantity("Total 56Ni Mass [g]", analysis::total_mass_ni56);
    history_->add_quantity("Total 56Co Mass [g]", analysis::total_mass_co56);
    history_->add_quantity("Total 56Fe Mass [g]", analysis::total_mass_fe56);
  }
}

/**
 * @brief Perform post initialization checks, calculations
 * Currently:
 * - Call grid's compute enclosed mass.
 * - If composition is enabled
 *   - Compute inv_atomic_mass
 * - If ionization if enabled
 *   - Ensure that if neutrons are present
 *     - saha_ncomps < ncomps
 */
void Driver::post_init_work() {
  auto sd0 = mesh_state_(0);
  auto cons = sd0.get_field("u_cf");
  const bool comps_active = mesh_state_.enabled("composition");
  const bool ionization_active = mesh_state_.enabled("ionization");

  static const IndexRange ib(grid_.domain<Domain::Interior>());

  grid_.compute_mass(cons);
  grid_.compute_mass_r(cons);
  grid_.compute_center_of_mass(cons);

  // If we are doing some kind of mass cut, that mass needs to be included
  // in the enclosed mass.
  const bool do_mass_cut = pin_->param()->contains("problem.params.mass_cut");
  if (do_mass_cut) {
    const auto mc =
        pin_->param()->get<double>("problem.params.mass_cut"); // in M_{\odot}
    auto menc = grid_.enclosed_mass();
    const int nNodes = grid_.n_nodes();
    athelas::par_for(
        DEFAULT_FLAT_LOOP_PATTERN, "post_init_work :: Adjust enclosed mass",
        DevExecSpace(), ib.s, ib.e, KOKKOS_LAMBDA(const int i) {
          for (int q = 0; q < nNodes; q++) {
            menc(i, q) += mc * constants::M_sun;
          }
        });
  }

  // This is a weird one. When ionization is active and needed to be computed
  // in the pgen, then this _must_ be computed there. That is the case
  // for the progenitor pgen, for example. It is here as a safety and
  // convenience for any composition-enabled problems that don't need
  // Saha solved in the pgen.
  if (comps_active) {
    auto *comps = mesh_state_(0).comps();
    const auto ncomps = pin_->param()->get<int>("composition.ncomps");
    auto inv_atomic_mass = comps->inverse_atomic_mass();
    auto charge = comps->charge();
    auto neutron_number = comps->neutron_number();
    const IndexRange eb(ncomps);
    athelas::par_for(
        DEFAULT_FLAT_LOOP_PATTERN, "post_init_work::inv_atomic_mass",
        DevExecSpace(), eb.s, eb.e, KOKKOS_LAMBDA(const int e) {
          inv_atomic_mass(e) = 1.0 / (charge(e) + neutron_number(e));
        });
  }

  if (ionization_active) {
    // Likely checked elsewhere
    athelas_requires(comps_active,
                     "Ionization cannot be enabled without composition!");
    auto *comps = mesh_state_(0).comps();
    auto *species_indexer = comps->species_indexer();
    const bool neut_present = species_indexer->contains("neut");
    if (neut_present) {
      const auto saha_ncomps = pin_->param()->get<int>("ionization.ncomps");
      const auto ncomps = pin_->param()->get<int>("composition.ncomps");
      athelas_requires(
          saha_ncomps < ncomps,
          "Ionization is enabled and neutrons are present in the composition. "
          "Please set [ionization.ncomps] < [composition.ncomps].");
    }
  }
} // post_init_work

/**
 * @brief post timestep work
 * @note Does not include IO work.
 * Contains:
 *  - Checks for package enable/disable
 *  - In debug mode calls check_state
 */
void Driver::post_step_work() {
  bool &thermal_engine_active =
      pin_->param()->get_mutable_ref<bool>("physics.engine.thermal.enabled");

  // Check if we need to disable any packages
  // I wonder if this should be internal to packages
  if (thermal_engine_active) {
    using thermal_engine::ThermalEnginePackage;
    static const auto tend_te =
        pin_->param()->get<double>("physics.engine.thermal.tend");
    static const bool split_te =
        pin_->param()->get<bool>("physics.engine.thermal.split");
    if (time_ >= tend_te && !split_te) {
      manager_->get_package<ThermalEnginePackage>("ThermalEngine")
          ->set_active(false);
      thermal_engine_active = false;
    }
    if (time_ >= tend_te && split_te) {
      split_manager_->get_package<ThermalEnginePackage>("ThermalEngine")
          ->set_active(false);
      thermal_engine_active = false;
    }
  }

#ifdef ATHELAS_DEBUG
  auto sd0 = mesh_state_(0);
  const bool rad_active = pin_->param()->get<bool>("physics.radiation.enabled");
  try {
    check_state(sd0, grid_.get_ihi(), rad_active);
  } catch (const AthelasError &e) {
    std::cerr << e.what() << "\n";
    std::println("!!! Bad State found, writing _final_ output file ...");
    write_output(mesh_state_, grid_, &sl_hydro_, pin_.get(), time_, -1);
  }
#endif
} // post_step_work

} // namespace athelas
