/**
 * @file timestepper.hpp
 * --------------
 *
 * @brief Primary time marching routine.
 *
 * @details Timestppers for hydro and rad hydro.
 *          Uses explicit for transport terms and implicit for coupling.
 *
 * TODO(astrobaker) move to calling step<fluid> / step<radhydro>
 */

#pragma once

#include "basic_types.hpp"
#include "bc/boundary_conditions_base.hpp"
#include "eos/eos_variant.hpp"
#include "fluid/hydro_package.hpp"
#include "interface/packages_base.hpp"
#include "kokkos_abstraction.hpp"
#include "kokkos_types.hpp"
#include "limiters/bound_enforcing_limiter.hpp"
#include "limiters/slope_limiter.hpp"
#include "loop_layout.hpp"
#include "problem_in.hpp"
#include "radiation/radhydro_package.hpp"
#include "state/state.hpp"
#include "timestepper/tableau.hpp"

namespace athelas {

using bc::BoundaryConditions;
using fluid::HydroPackage;
using radiation::RadHydroPackage;

class TimeStepper {

 public:
  // TODO(astrobarker): Is it possible to initialize grid_s_ from grid directly?
  TimeStepper(const ProblemIn *pin, GridStructure *grid, eos::EOS *eos);

  void initialize_timestepper();

  /**
   * Update fluid solution with SSPRK methods
   **/
  void step(PackageManager *pkgs, State *state, GridStructure &grid,
            const double t, const double dt, SlopeLimiter *sl_hydro) {

    // hydro explicit update
    update_fluid_explicit(pkgs, state, grid, t, dt, sl_hydro);
  }

  /**
   * Explicit fluid update with SSPRK methods
   **/
  void update_fluid_explicit(PackageManager *pkgs, State *state,
                             GridStructure &grid, const double t,
                             const double dt, SlopeLimiter *sl_hydro) {

    const auto &order = state->p_order();

    auto U = state->u_cf();
    auto U_s = state->u_cf_stages();

    const int nvars = state->n_cf();
    static const IndexRange ib(grid.domain<Domain::Entire>());
    static const IndexRange kb(order);
    static const IndexRange vb(nvars);

    grid_s_[0] = grid;

    TimeStepInfo dt_info{
        .t = t, .dt = dt, .dt_a = dt, .dt_coef = dt, .stage = 0};

    for (int iS = 0; iS < nStages_; ++iS) {
      auto left_interface = grid_s_[iS].x_l();
      dt_info.stage = iS;
      // re-zero the summation variables `SumVar`
      athelas::par_for(
          DEFAULT_LOOP_PATTERN, "Timestepper :: EX :: Reset sumvar",
          DevExecSpace(), ib.s, ib.e, kb.s, kb.e,
          KOKKOS_CLASS_LAMBDA(const int i, const int k) {
            for (int v = vb.s; v <= vb.e; ++v) {
              SumVar_U_(i, k, v) = U(i, k, v);
            }
            stage_data_(iS, i) = left_interface(i);
          });

      // --- Inner update loop ---

      for (int j = 0; j < iS; ++j) {
        dt_info.stage = j;
        dt_info.t = t + integrator_.explicit_tableau.c_i(j) * dt;
        pkgs->fill_derived(state, grid_s_[j], dt_info);
        pkgs->update_explicit(state, grid_s_[j], dt_info);

        // inner sum
        const double dt_a_ex = dt * integrator_.explicit_tableau.a_ij(iS, j);
        dt_info.dt_coef = dt_a_ex;
        pkgs->apply_delta(SumVar_U_, dt_info);

        athelas::par_for(
            DEFAULT_FLAT_LOOP_PATTERN, "Timestepper :: EX :: grid",
            DevExecSpace(), ib.s, ib.e, KOKKOS_CLASS_LAMBDA(const int i) {
              stage_data_(iS, i) +=
                  dt_a_ex *
                  pkgs->get_package<HydroPackage>("Hydro")->get_flux_u(j, i);
            });
      } // End inner loop

      // set U_s
      athelas::par_for(
          DEFAULT_LOOP_PATTERN, "Timestepper :: EX :: Set Us", DevExecSpace(),
          ib.s, ib.e, kb.s, kb.e, // vb.s, vb.e,
          KOKKOS_CLASS_LAMBDA(const int i, const int k) {
            for (int v = vb.s; v <= vb.e; ++v) {
              U_s(iS, i, k, v) = SumVar_U_(i, k, v);
            }
          });

      auto stage_data_j = Kokkos::subview(stage_data_, iS, Kokkos::ALL);
      grid_s_[iS] = grid;
      grid_s_[iS].update_grid(stage_data_j);

      auto Us_j =
          Kokkos::subview(U_s, iS, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
      apply_slope_limiter(sl_hydro, Us_j, &grid_s_[iS],
                          pkgs->get_package<HydroPackage>("Hydro")->basis(),
                          eos_);
      bel::apply_bound_enforcing_limiter(
          Us_j, pkgs->get_package<HydroPackage>("Hydro")->basis());
    } // end outer loop

    for (int iS = 0; iS < nStages_; ++iS) {
      dt_info.stage = iS;
      dt_info.t = t + integrator_.explicit_tableau.c_i(iS) * dt;

      pkgs->fill_derived(state, grid_s_[iS], dt_info);
      pkgs->update_explicit(state, grid_s_[iS], dt_info);

      const double dt_b_ex = dt * integrator_.explicit_tableau.b_i(iS);
      dt_info.dt_coef = dt_b_ex;
      pkgs->apply_delta(U, dt_info);

      athelas::par_for(
          DEFAULT_FLAT_LOOP_PATTERN, "Timestepper :: EX :: Finalize grid",
          DevExecSpace(), ib.s, ib.e, KOKKOS_CLASS_LAMBDA(const int i) {
            stage_data_(0, i) +=
                dt_b_ex *
                pkgs->get_package<HydroPackage>("Hydro")->get_flux_u(iS, i);
          });
      auto stage_data_j = Kokkos::subview(stage_data_, 0, Kokkos::ALL);
      grid_s_[iS] = grid;
      grid_s_[iS].update_grid(stage_data_j);
    }

    grid = grid_s_[nStages_ - 1];
    apply_slope_limiter(sl_hydro, U, &grid,
                        pkgs->get_package<HydroPackage>("Hydro")->basis(),
                        eos_);
    bel::apply_bound_enforcing_limiter(
        U, pkgs->get_package<HydroPackage>("Hydro")->basis());
  }

  /**
   * Update rad hydro solution with SSPRK methods
   **/
  void step_imex(PackageManager *pkgs, State *state, GridStructure &grid,
                 const double t, const double dt, SlopeLimiter *sl_hydro,
                 SlopeLimiter *sl_rad) {

    update_rad_hydro_imex(pkgs, state, grid, t, dt, sl_hydro, sl_rad);
  }

  /**
   * Fully coupled IMEX rad hydro update with SSPRK methods
   **/
  void update_rad_hydro_imex(PackageManager *pkgs, State *state,
                             GridStructure &grid, const double t,
                             const double dt, SlopeLimiter *sl_hydro,
                             SlopeLimiter *sl_rad) {

    const auto &order = state->p_order();

    auto uCF = state->u_cf();
    auto U_s = state->u_cf_stages();

    const int nvars = state->n_cf();
    static const IndexRange ib(grid.domain<Domain::Entire>());
    static const IndexRange kb(order);
    static const IndexRange vb(nvars);

    grid_s_[0] = grid;

    // TODO(astrobarker) pass in time
    TimeStepInfo dt_info{.t = t, .dt = dt, .dt_a = dt, .stage = 0};

    for (int iS = 0; iS < nStages_; ++iS) {
      auto left_interface = grid_s_[iS].x_l();
      dt_info.stage = iS;
      dt_info.t = t + integrator_.explicit_tableau.c_i(iS) * dt;
      athelas::par_for(
          DEFAULT_LOOP_PATTERN, "Timestepper :: IMEX :: Reset sumvar",
          DevExecSpace(), ib.s, ib.e, kb.s, kb.e, vb.s, vb.e,
          KOKKOS_CLASS_LAMBDA(const int i, const int k, const int v) {
            SumVar_U_(i, k, v) = uCF(i, k, v);
            stage_data_(iS, i) = left_interface(i);
          });

      // --- Inner update loop ---

      for (int j = 0; j < iS; ++j) {
        dt_info.stage = j;
        dt_info.t = t + integrator_.explicit_tableau.c_i(j) * dt;
        const double dt_a = dt * integrator_.explicit_tableau.a_ij(iS, j);
        const double dt_a_im = dt * integrator_.implicit_tableau.a_ij(iS, j);

        pkgs->fill_derived(state, grid_s_[j], dt_info);
        pkgs->update_explicit(state, grid_s_[j], dt_info);

        dt_info.dt_coef = dt_a;
        pkgs->apply_delta(SumVar_U_, dt_info);

        pkgs->update_implicit(state, grid_s_[j], dt_info);

        dt_info.dt_coef = dt_a_im;
        pkgs->apply_delta(SumVar_U_, dt_info);

        athelas::par_for(
            DEFAULT_FLAT_LOOP_PATTERN, "Timestepper :: IMEX :: Update grid",
            DevExecSpace(), ib.s, ib.e, KOKKOS_CLASS_LAMBDA(const int i) {
              stage_data_(iS, i) +=
                  dt_a * pkgs->get_package<RadHydroPackage>("RadHydro")
                             ->get_flux_u(j, i);
            });
      } // End inner loop

      auto stage_data_j = Kokkos::subview(stage_data_, iS, Kokkos::ALL);
      grid_s_[iS] = grid;
      grid_s_[iS].update_grid(stage_data_j);

      // set U_s
      athelas::par_for(
          DEFAULT_LOOP_PATTERN, "Timestepper :: IMEX :: Update Us",
          DevExecSpace(), ib.s, ib.e, kb.s, kb.e, vb.s, vb.e,
          KOKKOS_CLASS_LAMBDA(const int i, const int k, const int v) {
            U_s(iS, i, k, v) = SumVar_U_(i, k, v);
          });

      // NOTE: The limiting strategies in this function will fail if
      // the pkg does not have access to a rad_basis and fluid_basis
      auto Us_j =
          Kokkos::subview(U_s, iS, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
      // limiting madness
      apply_slope_limiter(
          sl_hydro, Us_j, &grid_s_[iS],
          pkgs->get_package<RadHydroPackage>("RadHydro")->fluid_basis(), eos_);
      apply_slope_limiter(
          sl_rad, Us_j, &grid_s_[iS],
          pkgs->get_package<RadHydroPackage>("RadHydro")->rad_basis(), eos_);
      apply_slope_limiter(
          sl_rad, SumVar_U_, &grid_s_[iS],
          pkgs->get_package<RadHydroPackage>("RadHydro")->rad_basis(), eos_);
      apply_slope_limiter(
          sl_hydro, SumVar_U_, &grid_s_[iS],
          pkgs->get_package<RadHydroPackage>("RadHydro")->fluid_basis(), eos_);
      bel::apply_bound_enforcing_limiter(
          Us_j, pkgs->get_package<RadHydroPackage>("RadHydro")->fluid_basis());
      bel::apply_bound_enforcing_limiter_rad(
          Us_j, pkgs->get_package<RadHydroPackage>("RadHydro")->rad_basis());
      bel::apply_bound_enforcing_limiter_rad(
          SumVar_U_,
          pkgs->get_package<RadHydroPackage>("RadHydro")->rad_basis());
      bel::apply_bound_enforcing_limiter(
          SumVar_U_,
          pkgs->get_package<RadHydroPackage>("RadHydro")->fluid_basis());

      // implicit update
      dt_info.stage = iS;
      dt_info.t = t + integrator_.explicit_tableau.c_i(iS) * dt;
      dt_info.dt_a = dt * integrator_.implicit_tableau.a_ij(iS, iS);
      pkgs->fill_derived(state, grid_s_[iS], dt_info);
      pkgs->update_implicit_iterative(state, SumVar_U_, grid_s_[iS], dt_info);
      pkgs->fill_derived(state, grid_s_[iS], dt_info);

      // set U_s after iterative solve
      athelas::par_for(
          DEFAULT_LOOP_PATTERN, "Timestepper :: IMEX :: Set Us implicit",
          DevExecSpace(), ib.s, ib.e, kb.s, kb.e,
          KOKKOS_CLASS_LAMBDA(const int i, const int k) {
            for (int v = 1; v <= vb.e; ++v) {
              U_s(iS, i, k, v) = Us_j(i, k, v);
            }
          });

      // TODO(astrobarker): slope limit rad
      apply_slope_limiter(
          sl_hydro, Us_j, &grid_s_[iS],
          pkgs->get_package<RadHydroPackage>("RadHydro")->fluid_basis(), eos_);
      apply_slope_limiter(
          sl_rad, Us_j, &grid_s_[iS],
          pkgs->get_package<RadHydroPackage>("RadHydro")->rad_basis(), eos_);
      bel::apply_bound_enforcing_limiter(
          Us_j, pkgs->get_package<RadHydroPackage>("RadHydro")->fluid_basis());
      bel::apply_bound_enforcing_limiter_rad(
          Us_j, pkgs->get_package<RadHydroPackage>("RadHydro")->rad_basis());
    } // end outer loop

    for (int iS = 0; iS < nStages_; ++iS) {
      dt_info.stage = iS;
      dt_info.t = t + integrator_.explicit_tableau.c_i(iS) * dt;
      const double dt_b = dt * integrator_.explicit_tableau.b_i(iS);
      const double dt_b_im = dt * integrator_.implicit_tableau.b_i(iS);

      pkgs->fill_derived(state, grid_s_[iS], dt_info);
      pkgs->update_explicit(state, grid_s_[iS], dt_info);

      dt_info.dt_coef = dt_b;
      pkgs->apply_delta_explicit(uCF, dt_info);

      pkgs->update_implicit(state, grid_s_[iS], dt_info);

      dt_info.dt_coef = dt_b_im;
      pkgs->apply_delta_implicit(uCF, dt_info);

      athelas::par_for(
          DEFAULT_FLAT_LOOP_PATTERN, "Timestepper :: IMEX :: Finalize grid",
          DevExecSpace(), ib.s, ib.e, KOKKOS_CLASS_LAMBDA(const int i) {
            stage_data_(0, i) +=
                dt_b * pkgs->get_package<RadHydroPackage>("RadHydro")
                           ->get_flux_u(iS, i);
          });
      auto stage_data_j = Kokkos::subview(stage_data_, 0, Kokkos::ALL); // HERE
      grid_s_[iS] = grid;
      grid_s_[iS].update_grid(stage_data_j);
    }

    // TODO(astrobarker): slope limit rad
    grid = grid_s_[nStages_ - 1];
    apply_slope_limiter(
        sl_hydro, uCF, &grid,
        pkgs->get_package<RadHydroPackage>("RadHydro")->fluid_basis(), eos_);
    apply_slope_limiter(
        sl_rad, uCF, &grid,
        pkgs->get_package<RadHydroPackage>("RadHydro")->rad_basis(), eos_);
    bel::apply_bound_enforcing_limiter(
        uCF, pkgs->get_package<RadHydroPackage>("RadHydro")->fluid_basis());
    bel::apply_bound_enforcing_limiter_rad(
        uCF, pkgs->get_package<RadHydroPackage>("RadHydro")->rad_basis());
  }

  [[nodiscard]] auto n_stages() const noexcept -> int;
  [[nodiscard]] static auto nvars_evolved(const ProblemIn *pin) noexcept -> int;

 private:
  int nvars_evolved_;
  int mSize_;

  // tableaus
  RKIntegrator integrator_;

  int nStages_;
  int tOrder_;

  // Hold stage data
  AthelasArray3D<double> SumVar_U_;
  std::vector<GridStructure> grid_s_;

  // stage_data_ Holds cell left interface positions
  AthelasArray2D<double> stage_data_;

  // Variables to pass to update step
  eos::EOS *eos_;
};

} // namespace athelas
