#include "driver.hpp"
#include "basic_types.hpp"
#include "basis/nodal_basis.hpp"
#include "basis/polynomial_basis.hpp"
#include "fluid/hydro_package.hpp"
#include "geometry/geometry_package.hpp"
#include "gravity/gravity_package.hpp"
#include "heating/nickel_package.hpp"
#include "history/quantities.hpp"
#include "initialization.hpp"
#include "interface/packages_base.hpp"
#include "io/io.hpp"
#include "kokkos_types.hpp"
#include "limiters/slope_limiter.hpp"
#include "loop_layout.hpp"
#include "pgen/problem_in.hpp"
#include "state/state.hpp"
#include "thermal.hpp"
#include "timestepper/timestepper.hpp"
#include "utils/error.hpp"

namespace athelas {

using basis::ModalBasis;
using basis::NodalBasis;
using io::write_output, io::print_simulation_parameters;

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
  const auto dt_init = pin_->param()->get<double>("output.dt_init");
  const auto dt_init_frac = pin_->param()->get<double>("output.dt_init_frac");
  const auto dt_hdf5 = pin_->param()->get<double>("output.dt_hdf5");
  const auto fixed_dt = pin_->param()->contains("output.dt_fixed");

  dt_ = dt_init;
  TimeStepInfo dt_info{.t = time_,
                       .dt = dt_,
                       .dt_coef_implicit = dt_,
                       .dt_coef = dt_,
                       .stage = 0};

  // some startup io
  auto sd0 = mesh_state_(0);
  manager_->fill_derived(sd0, grid_, dt_info);
  print_simulation_parameters(grid_, pin_.get());
  write_output(mesh_state_, grid_, &sl_hydro_, pin_.get(), time_, 0);
  history_->write(mesh_state_, grid_, time_);

  // --- Evolution loop ---
  int iStep = 0;
  int i_out_h5 = 1; // output label, start 1
  int i_out_hist = 1; // output hist
  std::println("# Step    t       dt       zone_cycles / wall_second");
  while (time_ < t_end_ && iStep <= nlim) {
    if (!fixed_dt) {
      dt_ =
          std::min(manager_->min_timestep(mesh_state_(0), grid_,
                                          {.t = time_, .dt = dt_, .stage = 0}),
                   dt_ * dt_init_frac);
    } else {
      dt_ = pin_->param()->get<double>("output.dt_fixed");
    }
    if (time_ + dt_ > t_end_) {
      dt_ = t_end_ - time_;
    }

    // This logic could probably be cleaner..
    if (!rad_active) {
      ssprk_.step(manager_.get(), mesh_state_, grid_, time_, dt_, &sl_hydro_);
    } else {
      try {
        ssprk_.step_imex(manager_.get(), mesh_state_, grid_, time_, dt_,
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
      split_stepper_->step(split_manager_.get(), mesh_state_, grid_, time_,
                           dt_);
    }

    time_ += dt_;
    post_step_work();

    // Write state, other io
    if (time_ >= i_out_h5 * dt_hdf5) {
      manager_->fill_derived(sd0, grid_, dt_info);
      write_output(mesh_state_, grid_, &sl_hydro_, pin_.get(), time_, i_out_h5);
      i_out_h5 += 1;
    }

    if (time_ >= i_out_hist * pin_->param()->get<double>("output.hist_dt")) {
      history_->write(mesh_state_, grid_, time_);
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

  manager_->fill_derived(sd0, grid_, dt_info);
  write_output(mesh_state_, grid_, &sl_hydro_, pin_.get(), time_, -1);

  return AthelasExitCodes::SUCCESS;
}

void Driver::initialize(ProblemIn *pin) { // NOLINT
  using fluid::HydroPackage;
  using geometry::GeometryPackage;
  using gravity::GravityPackage;
  using nickel::NickelHeatingPackage;
  using thermal_engine::ThermalEnginePackage;

  const auto nx = pin_->param()->get<int>("problem.nx");
  const int nnodes = pin_->param()->get<int>("basis.nnodes");
  // For nodal DG, u_cf is (ix, node, var); for modal, (ix, mode, var)
  const auto cfl =
      compute_cfl(pin_->param()->get<double>("problem.cfl"), nnodes);
  const bool rad_active = pin->param()->get<bool>("physics.rad_active");
  const bool comps_active =
      pin->param()->get<bool>("physics.composition_enabled");

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
    const auto ncomps = pin->param()->get<int>("composition.ncomps");
    mesh_state_.register_field("x_q", DataPolicy::OneCopy,
                               "Nodal mass fractions", nx + 2, nnodes + 2,
                               ncomps);
  }

  // auto info = mesh_state_.field_info();

  if (!restart_) {
    initialize_fields(mesh_state_, &grid_, pin);
    auto sd0 = mesh_state_(0);
    auto prims = sd0.get_field("u_pf");
    auto cons = sd0.get_field("u_cf");
    bc::fill_ghost_zones<3>(cons, &grid_, bcs_.get(), {0, 2});
    grid_.compute_mass(cons);

    auto nx = grid_.n_elements();
    const bool rad_active = pin_->param()->get<bool>("physics.rad_active");
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

  const bool gravity_active = pin->param()->get<bool>("physics.gravity_active");
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
    manager_->add_package(RadHydroPackage{pin, n_stages, nnodes, bcs_.get(),
                                          cfl, nx, pkg_active});
  } else [[unlikely]] {
    // pure Hydro
    manager_->add_package(
        HydroPackage{pin, n_stages, nnodes, bcs_.get(), cfl, nx, pkg_active});
  }
  if (gravity_active) {
    if (!pin->param()->get<bool>("physics.gravity.split")) {
      manager_->add_package(
          GravityPackage{pin, pin->param()->get<GravityModel>("gravity.model"),
                         pin->param()->get<double>("gravity.gval"), cfl,
                         n_stages, pkg_active});
    } else {
      split = true;
      split_manager_->add_package(
          GravityPackage{pin, pin->param()->get<GravityModel>("gravity.model"),
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
  const bool comps_active = mesh_state_.composition_enabled();
  const bool ionization_active = mesh_state_.ionization_enabled();

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
  const bool ionization_enabled = mesh_state_.ionization_enabled();
  if (ionization_enabled) {
    static const IndexRange ib(grid_.domain<Domain::Interior>());
    static const IndexRange qb(grid_.n_nodes());

    const auto &eos = mesh_state_.eos();
    auto ucf = mesh_state_(0).get_field("u_cf");

    const auto *const comps = mesh_state_.comps();
    auto number_density = comps->number_density();
    auto ye = comps->ye();

    const auto *const ionization_states = mesh_state_.ionization_state();
    auto ybar = ionization_states->ybar();
    auto e_ion_corr = ionization_states->e_ion_corr();
    auto sigma1 = ionization_states->sigma1();
    auto sigma2 = ionization_states->sigma2();
    auto sigma3 = ionization_states->sigma3();
    athelas::par_for(
        DEFAULT_FLAT_LOOP_PATTERN, "Fixup", DevExecSpace(), ib.s, ib.e,
        KOKKOS_LAMBDA(const int i) {
          for (int q = qb.s; q <= qb.e; ++q) {
            const double rho = 1.0 / ucf(i, q, vars::cons::SpecificVolume);
            const double vel = ucf(i, q, vars::cons::Velocity);
            const double emt = ucf(i, q, vars::cons::Energy);
            const double sie = emt - 0.5 * vel * vel;
            eos::EOSLambda lambda;
            lambda.data[1] = ye(i, q + 1);
            lambda.data[6] = e_ion_corr(i, q + 1);
            const double emin = eos::min_sie(eos, rho, lambda.ptr());
            if (sie <= emin) {
              double sie_fix = 1.1 * emin;
              ucf(i, q, vars::cons::Energy) = sie_fix + 0.5 * vel * vel;
              std::println("FIXUP i sie sie_min siefix {} {:.8e} {:.8e} {:.5e}",
                           i, sie, emin, sie_fix);
            }
          }
        });
  }

#ifdef ATHELAS_DEBUG
  auto sd0 = mesh_state_(0);
  const bool rad_active = pin_->param()->get<bool>("physics.rad_active");
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
