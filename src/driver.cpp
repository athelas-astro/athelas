#include "driver.hpp"
#include "basic_types.hpp"
#include "basis/polynomial_basis.hpp"
#include "fluid/hydro_package.hpp"
#include "geometry/geometry_package.hpp"
#include "gravity/gravity_package.hpp"
#include "heating/nickel_package.hpp"
#include "history/quantities.hpp"
#include "initialization.hpp"
#include "interface/packages_base.hpp"
#include "io/io.hpp"
#include "limiters/slope_limiter.hpp"
#include "pgen/problem_in.hpp"
#include "state/state.hpp"
#include "thermal.hpp"
#include "timestepper/timestepper.hpp"
#include "utils/error.hpp"

namespace athelas {

using basis::ModalBasis;
using io::write_basis, io::write_state, io::print_simulation_parameters;

auto Driver::execute() -> int {
  static const auto nx = pin_->param()->get<int>("problem.nx");
  static const bool rad_active = pin_->param()->get<bool>("physics.rad_active");

  // --- Timer ---
  Kokkos::Timer timer_zone_cycles;
  double zc_ws = 0.0; // zone cycles / wall second

  const double nlim = (pin_->param()->get<double>("problem.nlim")) == -1
                          ? std::numeric_limits<double>::infinity()
                          : pin_->param()->get<double>("problem.nlim");
  const auto ncycle_out = pin_->param()->get<int>("output.ncycle_out");
  const auto dt_init = pin_->param()->get<double>("output.initial_dt");
  const auto dt_init_frac = pin_->param()->get<double>("output.dt_init_frac");
  const auto dt_hdf5 = pin_->param()->get<double>("output.dt_hdf5");

  dt_ = dt_init;
  TimeStepInfo dt_info{.t = time_,
                       .dt = dt_,
                       .dt_coef_implicit = dt_,
                       .dt_coef = dt_,
                       .stage = -1};

  // some startup io
  manager_->fill_derived(state_.get(), grid_, dt_info);
  write_basis(fluid_basis_.get(),
              pin_->param()->get<std::string>("problem.problem"));
  print_simulation_parameters(grid_, pin_.get());
  write_state(state_.get(), grid_, &sl_hydro_, pin_.get(), time_,
              pin_->param()->get<int>("fluid.porder"), 0, rad_active);
  history_->write(*state_, grid_, fluid_basis_.get(), radiation_basis_.get(),
                  time_);

  // misc
  bool &thermal_engine_active =
      pin_->param()->get_mutable_ref<bool>("physics.engine.thermal.enabled");

  // --- Evolution loop ---
  int iStep = 0;
  int i_out_h5 = 1; // output label, start 1
  int i_out_hist = 1; // output hist
  std::println("# Step    t       dt       zone_cycles / wall_second");
  while (time_ < t_end_ && iStep <= nlim) {

    dt_ = std::min(manager_->min_timestep(state_.get(), grid_,
                                          {.t = time_, .dt = dt_, .stage = 0}),
                   dt_ * dt_init_frac);
    if (time_ + dt_ > t_end_) {
      dt_ = t_end_ - time_;
    }

    // This logic could probably be cleaner..
    if (!rad_active) {
      ssprk_.step(manager_.get(), state_.get(), grid_, time_, dt_, &sl_hydro_);
    } else {
      try {
        ssprk_.step_imex(manager_.get(), state_.get(), grid_, time_, dt_,
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
      split_stepper_->step(split_manager_.get(), state_.get(), grid_, time_,
                           dt_);
    }

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
    try {
      check_state(state_.get(), grid_.get_ihi(), rad_active);
    } catch (const AthelasError &e) {
      std::cerr << e.what() << std::endl;
      std::println("!!! Bad State found, writing _final_ output file ...");
      write_state(state_.get(), grid_, &sl_hydro_, pin_.get(), time_,
                  pin_->param()->get<int>("fluid.porder"), -1, rad_active);
      return AthelasExitCodes::FAILURE;
    }
#endif

    time_ += dt_;

    // Write state, other io
    if (time_ >= i_out_h5 * dt_hdf5) {
      manager_->fill_derived(state_.get(), grid_, dt_info);
      write_state(state_.get(), grid_, &sl_hydro_, pin_.get(), time_,
                  fluid_basis_->order(), i_out_h5, rad_active);
      i_out_h5 += 1;
    }

    if (time_ >= i_out_hist * pin_->param()->get<double>("output.hist_dt")) {
      history_->write(*state_, grid_, fluid_basis_.get(),
                      radiation_basis_.get(), time_);
      i_out_hist += 1;
    }

    // timer
    if (iStep % ncycle_out == 0) {
      zc_ws =
          static_cast<double>(ncycle_out) * nx / timer_zone_cycles.seconds();
      std::println("{} {:.5e} {:.5e} {:.5e}", iStep, time_, dt_, zc_ws);
      timer_zone_cycles.reset();
    }

    iStep++;
  }

  manager_->fill_derived(state_.get(), grid_, dt_info);
  write_state(state_.get(), grid_, &sl_hydro_, pin_.get(), time_,
              pin_->param()->get<int>("fluid.porder"), -1, rad_active);

  return AthelasExitCodes::SUCCESS;
}

void Driver::initialize(ProblemIn *pin) { // NOLINT
  using fluid::HydroPackage;
  using geometry::GeometryPackage;
  using gravity::GravityPackage;
  using nickel::NickelHeatingPackage;
  using thermal_engine::ThermalEnginePackage;

  const auto nx = pin_->param()->get<int>("problem.nx");
  const int max_order =
      std::max(pin_->param()->get<int>("fluid.porder"),
               pin_->param()->get<int>("radiation.porder", 1));
  const auto cfl =
      compute_cfl(pin_->param()->get<double>("problem.cfl"), max_order);

  if (!restart_) {
    // The pattern here is annoying and due to a chicken-and-egg
    // pattern between problem generation and basis construction.
    // Some problems, like Shu-Osher, need the basis at setup
    // to perform the L2 projection from nodal to modal
    // representation. Basis construction, however, requires the
    // nodal density field as density weighted inner products are used.
    // So here, the firist initialize_fields call may only populate nodal
    // density in uPF. Then bases are constructed. Then, the second
    // initialize_fields call populates the conserved variables.
    // For simple cases, like Sod, the layering is redundant, as
    // the bases are never used.
    initialize_fields(state_.get(), &grid_, eos_.get(), pin);

    // --- Datastructure for modal basis ---
    static const bool rad_active =
        pin_->param()->get<bool>("physics.rad_active");
    fluid_basis_ = std::make_unique<ModalBasis>(
        poly_basis::legendre, state_->u_pf(), &grid_,
        pin->param()->get<int>("fluid.porder"),
        pin->param()->get<int>("fluid.nnodes"),
        pin->param()->get<int>("problem.nx"), true);
    if (rad_active) {
      radiation_basis_ = std::make_unique<ModalBasis>(
          poly_basis::legendre, state_->u_pf(), &grid_,
          pin->param()->get<int>("radiation.porder"),
          pin->param()->get<int>("radiation.nnodes"),
          pin->param()->get<int>("problem.nx"), false);
    }

    // --- Phase 2: Re-initialize with modal projection ---
    // This will use the nodal density from Phase 1 to construct proper modal
    // coefficients
    initialize_fields(state_.get(), &grid_, eos_.get(), pin, fluid_basis_.get(),
                      radiation_basis_.get());
  }

  const bool rad_active = pin->param()->get<bool>("physics.rad_active");
  const bool gravity_active = pin->param()->get<bool>("physics.gravity_active");
  const bool ni_heating_active =
      pin->param()->get<bool>("physics.heating.nickel.enabled");
  const bool geometry =
      pin->param()->get<std::string>("problem.geometry") == "spherical";
  const bool thermal_engine_active =
      pin->param()->get<bool>("physics.engine.thermal.enabled");

  bool split = false;

  const int n_stages = ssprk_.n_stages();

  // --- Init physics package manager ---
  // NOTE: Hydro/RadHydro should be registered first
  if (rad_active) {
    manager_->add_package(RadHydroPackage{
        pin, n_stages, eos_.get(), opac_.get(), fluid_basis_.get(),
        radiation_basis_.get(), bcs_.get(), cfl, nx, true});
  } else [[unlikely]] {
    // pure Hydro
    manager_->add_package(HydroPackage{pin, n_stages, eos_.get(),
                                       fluid_basis_.get(), bcs_.get(), cfl, nx,
                                       true});
  }
  if (gravity_active) {
    if (!pin->param()->get<bool>("physics.gravity.split")) {
      manager_->add_package(
          GravityPackage{pin, pin->param()->get<GravityModel>("gravity.model"),
                         pin->param()->get<double>("gravity.gval"),
                         fluid_basis_.get(), cfl, n_stages, true});
    } else {
      split = true;
      split_manager_->add_package(
          GravityPackage{pin, pin->param()->get<GravityModel>("gravity.model"),
                         pin->param()->get<double>("gravity.gval"),
                         fluid_basis_.get(), cfl, n_stages, true});
    }
  }
  if (ni_heating_active) {
    if (!pin->param()->get<bool>("physics.heating.nickel.split")) {
      manager_->add_package(NickelHeatingPackage{
          pin, fluid_basis_.get(), state_->comps()->species_indexer(), n_stages,
          true});
    } else {
      split = true;
      split_manager_->add_package(NickelHeatingPackage{
          pin, fluid_basis_.get(), state_->comps()->species_indexer(), n_stages,
          true});
    }
  }
  if (thermal_engine_active) {
    if (!pin->param()->get<bool>("physics.engine.thermal.split")) {
      manager_->add_package(ThermalEnginePackage{
          pin, state_.get(), &grid_, fluid_basis_.get(), n_stages, true});
    } else {
      split = true;
      split_manager_->add_package(ThermalEnginePackage{
          pin, state_.get(), &grid_, fluid_basis_.get(), n_stages, true});
    }
  }
  // TODO(astrobarker): [split] Could add option to split.. not important..
  if (geometry) {
    manager_->add_package(
        GeometryPackage{pin, fluid_basis_.get(), n_stages, true});
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

  // --- slope limiter to initial condition ---
  apply_slope_limiter(&sl_hydro_, state_->u_cf(), &grid_, fluid_basis_.get(),
                      eos_.get());

  // --- Add history outputs ---
  // NOTE: Could be nice to have gravitational energy added
  // to total, conditionally.
  history_->add_quantity("Total Mass [g]", analysis::total_mass);
  history_->add_quantity("Total Fluid Energy [erg]",
                         analysis::total_fluid_energy);
  history_->add_quantity("Total Internal Energy [erg]",
                         analysis::total_internal_energy);
  history_->add_quantity("Total Kinetic Energy [erg]",
                         analysis::total_kinetic_energy);

  if (gravity_active) {
    history_->add_quantity("Total Gravitational Energy [erg]",
                           analysis::total_gravitational_energy);
  }

  if (rad_active) {
    history_->add_quantity("Total Radiation Momentum [g cm / s]",
                           analysis::total_rad_momentum);
    history_->add_quantity("Total Momentum [g cm / s]",
                           analysis::total_momentum);
    history_->add_quantity("Total Radiation Energy [erg]",
                           analysis::total_rad_energy);
    history_->add_quantity("Total Energy [erg]", analysis::total_energy);
  }
  history_->add_quantity("Total Fluid Momentum [g cm / s]",
                         analysis::total_fluid_momentum);

  // total nickel56, cobalt56, iron56
  if (ni_heating_active) {
    history_->add_quantity("Total 56Ni Mass [g]", analysis::total_mass_ni56);
    history_->add_quantity("Total 56Co Mass [g]", analysis::total_mass_co56);
    history_->add_quantity("Total 56Fe Mass [g]", analysis::total_mass_fe56);
  }
}

} // namespace athelas
