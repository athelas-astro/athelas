#include "driver.hpp"
#include "basic_types.hpp"
#include "basis/nodal_basis.hpp"
#include "basis/polynomial_basis.hpp"
#include "bc/boundary_conditions.hpp"
#include "engines/thermal.hpp"
#include "fluid/hydro_package.hpp"
#include "geometry/geometry_package.hpp"
#include "gravity/gravity_package.hpp"
#include "heating/nickel_package.hpp"
#include "history/quantities.hpp"
#include "interface/packages_base.hpp"
#include "interface/state.hpp"
#include "io/io.hpp"
#include "io/restart.hpp"
#include "kokkos_types.hpp"
#include "limiters/bound_enforcing_limiter.hpp"
#include "limiters/slope_limiter.hpp"
#include "loop_layout.hpp"
#include "pgen/pgen.hpp"
#include "pgen/problem_in.hpp"
#include "radiation/imex_radhydro_package.hpp"
#include "radiation/implicit_moments_package.hpp"
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

  if (!restart_) {
    dt_ = dt_init;
  }
  TimeStepInfo dt_info{.t = time_,
                       .dt = dt_,
                       .dt_coef_implicit = dt_,
                       .dt_coef = dt_,
                       .stage = 0,
                       .cycle = 1};

  // some startup io
  auto sd0 = mesh_state_(0);
  manager_->fill_derived(sd0, dt_info);
  print_simulation_parameters(mesh_state_.mesh(), pin_.get());
  // Initial dump has file index 0 and is followed by initial history entry
  // 0 on the next line, so all "last_*" counters land at 0 here — restart
  // from this dump then resumes at cycle/h5/hist = 1, matching a fresh-run
  // loop init. Skipped on restart since the source .ath already covers it.
  io::SimInfo info{.time = time_,
                   .dt = dt_,
                   .last_cycle = 0,
                   .last_out_h5 = 0,
                   .last_out_hist = 0};
  if (!restart_) {
    write_output(mesh_state_, mesh_state_.mesh(), &sl_hydro_, pin_.get(), info,
                 0);
    history_->write(mesh_state_, mesh_state_.mesh(), time_);
  }

  // --- Evolution loop ---
  // SimInfo stores "last completed" counters; restart resumes at last + 1.
  int cycle = restart_ ? restart_info_.last_cycle + 1 : 1;
  int i_out_h5 = restart_ ? restart_info_.last_out_h5 + 1 : 1;
  int i_out_hist = restart_ ? restart_info_.last_out_hist + 1 : 1;
  std::println("# Cycle      t       dt       zone_cycles / wall_second");
  while (time_ < t_end_ && cycle <= nlim) {
    dt_info.t = time_;
    dt_info.dt = dt_;
    dt_info.dt_coef_implicit = dt_;
    dt_info.dt_coef = dt_;
    dt_info.stage = 0;
    dt_info.cycle = cycle;

    if (!fixed_dt) {
      dt_ = std::min(manager_->min_timestep(mesh_state_(0), dt_info),
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
      ssprk_.step(manager_.get(), mesh_state_, dt_info, &sl_hydro_);
    } else {
      try {
        ssprk_.step_imex(manager_.get(), mesh_state_, dt_info, &sl_hydro_,
                         &sl_rad_);
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
      split_stepper_->step(split_manager_.get(), mesh_state_, dt_info);
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
      manager_->fill_derived(sd0, dt_info);
      // History is checked AFTER the HDF5 dump, so i_out_hist here is still
      // the pre-history-fire value. Record the index of the most-recently-
      // written history entry so restart's "next = recorded + 1" rule works
      // regardless of whether HDF5 and history cadences align. If history is
      // also due this cycle, the about-to-be-written entry counts as the
      // most-recent (= i_out_hist); otherwise the most-recent is one less
      // than the pending value.
      const double hist_dt = pin_->param()->get<double>("output.hist_dt");
      const bool hist_due_this_cycle = time_ >= i_out_hist * hist_dt;
      const int last_out_hist =
          hist_due_this_cycle ? i_out_hist : i_out_hist - 1;
      info = {.time = time_,
              .dt = dt_,
              .last_cycle = cycle,
              .last_out_h5 = i_out_h5,
              .last_out_hist = last_out_hist};
      write_output(mesh_state_, mesh_state_.mesh(), &sl_hydro_, pin_.get(),
                   info, i_out_h5);
      i_out_h5 += 1;
    }

    // history
    if (time_ >= i_out_hist * pin_->param()->get<double>("output.hist_dt")) {
      history_->write(mesh_state_, mesh_state_.mesh(), time_);
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

  manager_->fill_derived(sd0, dt_info);
  // Loop variables are post-increment here ("next-pending"). Normalize to the
  // "last completed" SimInfo convention so restart-from-_final (with extended
  // tf or nlim) doesn't skip a cycle / HDF5 / history index.
  info = {.time = time_,
          .dt = dt_,
          .last_cycle = cycle - 1,
          .last_out_h5 = i_out_h5 - 1,
          .last_out_hist = i_out_hist - 1};
  write_output(mesh_state_, mesh_state_.mesh(), &sl_hydro_, pin_.get(), info,
               -1);

  return AthelasExitCodes::SUCCESS;
}

void Driver::initialize(ProblemIn *pin) { // NOLINT
  using fluid::HydroPackage;
  using geometry::GeometryPackage;
  using gravity::GravityPackage;
  using nickel::NickelHeatingPackage;
  using radiation::ImplicitRadiationMomentsPackage;
  using radiation::RadHydroPackage;
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
  mesh_state_.register_field("u_cf", DataPolicy::TwoCopy, "Conserved variables",
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
    initialize_fields(mesh_state_, &mesh_state_.mesh(), pin);
    auto cons = mesh_state_(0).get_field("u_cf");
    bc::fill_ghost_zones<3>(cons, &mesh_state_.mesh(), bcs_.get(), {0, 2});
    mesh_state_.mesh().compute_mass(cons);
  } else {
    // Restart path: rebuild MeshState/mesh from the .ath file rather than the
    // problem generator. Composition / ionization wiring must happen before
    // fields are loaded and before any package queries comps()/ion_state().
    io::RestartReader reader(restart_filename_);
    restart_info_ = io::load_info_from_h5(reader);

    if (comps_active) {
      const auto ncomps = pin->param()->get<int>("composition.ncomps");
      auto comps =
          std::make_shared<atom::CompositionData>(nx + 2, nnodes, ncomps);
      mesh_state_.setup_composition(comps);
      io::load_composition_from_h5(*comps, reader);
    }

    const bool ionization_active =
        pin->param()->get<bool>("physics.ionization.enabled");
    if (ionization_active) {
      // n_states (= max_charge + 1) isn't a param — read it from the
      // shape of the dumped ionization_fractions array.
      const auto ifrac_dims =
          reader.dataset_extent("/ionization/ionization_fractions");
      athelas_requires(ifrac_dims.size() == 4,
                       "Restart: /ionization/ionization_fractions must be 4D");
      const int n_states = static_cast<int>(ifrac_dims[3]);
      const auto ncomps = pin->param()->get<int>("composition.ncomps");
      const auto saha_ncomps = pin->param()->get<int>("ionization.ncomps");
      const auto fn_ion =
          pin->param()->get<std::string>("ionization.fn_ionization");
      const auto fn_deg =
          pin->param()->get<std::string>("ionization.fn_degeneracy");
      const auto solver = pin->param()->get<std::string>("ionization.solver");
      auto ion = std::make_shared<atom::IonizationState>(
          nx + 2, nnodes, ncomps, n_states, saha_ncomps, fn_ion, fn_deg,
          solver);
      mesh_state_.setup_ionization(ion);
      // Loaded zbar/ionization_fractions serve as the Saha solver's initial
      // guess on the first fill_derived call — required for bit-exact restart.
      io::load_ionization_from_h5(*ion, reader);
    }

    io::load_fields_from_h5(mesh_state_, reader);
    io::load_grid_from_h5(mesh_state_.mesh(), reader);

    time_ = restart_info_.time;
    dt_ = restart_info_.dt;
    std::println("# Restart resumed at t = {:.5e}, cycle = {}, dt = {:.5e}",
                 time_, restart_info_.last_cycle, dt_);
  }

  // Basis construction is identical for both paths once u_pf is in place
  // (pgen-populated or restart-loaded).
  auto prims = mesh_state_(0).get_field("u_pf");
  mesh_state_.setup_fluid_basis(
      std::make_unique<NodalBasis>(prims, &mesh_state_.mesh(), nnodes, nx));
  if (rad_active) {
    mesh_state_.setup_rad_basis(
        std::make_unique<NodalBasis>(prims, &mesh_state_.mesh(), nnodes, nx));
  }

  // post_init_work recomputes mesh mass / center-of-mass and applies the
  // mass-cut adjustment. For restart those values came from the .ath file
  // and must not be re-derived (mass cut would double-count, and recomputing
  // would slightly perturb a checkpointed state).
  if (!restart_) {
    post_init_work();
  }

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
      manager_->add_package(ThermalEnginePackage{pin, sd0, &mesh_state_.mesh(),
                                                 n_stages, pkg_active});
    } else {
      split = true;
      split_manager_->add_package(ThermalEnginePackage{
          pin, sd0, &mesh_state_.mesh(), n_stages, pkg_active});
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
  // Restart state is already post-step-valid: ghost zones and limited values
  // came from the .ath file, so re-running the limiter would alter the
  // checkpointed state.
  if (!restart_) {
    auto ucf = sd0.get_field("u_cf");
    bc::fill_ghost_zones<3>(ucf, &mesh_state_.mesh(), bcs_.get(), {0, 2});
    if (rad_active) {
      bc::fill_ghost_zones<2>(ucf, &mesh_state_.mesh(), bcs_.get(), {3, 4});
    }
    auto cons = sd0.get_field("u_cf");
    apply_slope_limiter(&sl_hydro_, cons, sd0, sd0.fluid_basis(), sd0.eos());
    bel::apply_bound_enforcing_limiter(sd0);
  }

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
 * - Call mesh's compute enclosed mass.
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

  static const IndexRange ib(mesh_state_.mesh().domain<Domain::Interior>());

  mesh_state_.mesh().compute_mass(cons);
  mesh_state_.mesh().compute_mass_r(cons);
  mesh_state_.mesh().compute_center_of_mass(cons);

  // If we are doing some kind of mass cut, that mass needs to be included
  // in the enclosed mass.
  const bool do_mass_cut = pin_->param()->contains("problem.params.mass_cut");
  if (do_mass_cut) {
    const auto mc =
        pin_->param()->get<double>("problem.params.mass_cut"); // in M_{\odot}
    auto menc = mesh_state_.mesh().enclosed_mass();
    const int nNodes = mesh_state_.mesh().n_nodes();
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
    check_state(sd0, mesh_state_.mesh().get_ihi(), rad_active);
  } catch (const AthelasError &e) {
    std::cerr << e.what() << "\n";
    std::println("!!! Bad State found, writing _final_ output file ...");
    const io::SimInfo info{.time = time_,
                           .dt = dt_,
                           .last_cycle = -1,
                           .last_out_h5 = -1,
                           .last_out_hist = -1};
    write_output(mesh_state_, mesh_state_.mesh(), &sl_hydro_, pin_.get(), info,
                 -1);
  }
#endif
} // post_step_work

} // namespace athelas
