/**
 * @file timestepper.hpp
 * --------------
 *
 * @brief Primary time marching routine.
 *
 * @details Timestppers for hydro and rad hydro.
 *          Uses explicit for transport terms and implicit for coupling.
 *
 * TODO(astrobarker) move to calling step<fluid> / step<radhydro>
 */

#pragma once

#include "basic_types.hpp"
#include "bc/boundary_conditions.hpp"
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
  TimeStepper(const ProblemIn *pin, GridStructure *grid);

  void initialize_timestepper();

  /**
   * Update fluid solution with SSPRK methods
   **/
  void step(PackageManager *pkgs, MeshState &mesh_state, GridStructure &grid,
            const double t, const double dt, SlopeLimiter *sl_hydro) {
    // hydro explicit update
    update_fluid_explicit(pkgs, mesh_state, grid, t, dt, sl_hydro);
  }

  /**
   * Explicit fluid update with SSPRK methods
   **/
  void update_fluid_explicit(PackageManager *pkgs, MeshState &mesh_state,
                             GridStructure &grid, const double t,
                             const double dt, SlopeLimiter *sl_hydro) {
    static const int nvars = mesh_state.nvars("u_cf");
    static const IndexRange ib(grid.domain<Domain::Entire>());
    static const IndexRange qb(grid.n_nodes());
    static const IndexRange vb(nvars);

    grid_s_[0] = grid;

    TimeStepInfo dt_info{.t = t, .dt = dt, .stage = 0};

    const auto &fluid_basis = mesh_state.fluid_basis();
    const auto &eos = mesh_state.eos();

    auto u0 = mesh_state(0).get_field("u_cf");
    for (int iS = 0; iS < nStages_; ++iS) {
      auto left_interface = grid_s_[iS].x_l();
      dt_info.stage = iS;

      // re-set the summation variables `SumVar`
      auto stage_data = mesh_state.stage(iS);
      auto u = stage_data.get_field("u_cf");
      athelas::par_for(
          DEFAULT_LOOP_PATTERN, "Timestepper :: EX :: Reset sumvar",
          DevExecSpace(), ib.s, ib.e, qb.s, qb.e,
          KOKKOS_CLASS_LAMBDA(const int i, const int q) {
            for (int v = vb.s; v <= vb.e; ++v) {
              SumVar_U_(i, q, v) = u0(i, q, v);
            }
            x_l_sumvar_(iS, i) = left_interface(i);
          });

      // --- Inner update loop ---

      for (int j = 0; j < iS; ++j) {
        dt_info.stage = j;
        dt_info.t = t + integrator_.explicit_tableau.c_i(j) * dt;
        const double dt_a_ex = dt * integrator_.explicit_tableau.a_ij(iS, j);
        dt_info.dt_coef = dt_a_ex;

        pkgs->apply_delta(SumVar_U_, dt_info);

        athelas::par_for(
            DEFAULT_FLAT_LOOP_PATTERN, "Timestepper :: EX :: grid",
            DevExecSpace(), ib.s, ib.e, KOKKOS_CLASS_LAMBDA(const int i) {
              x_l_sumvar_(iS, i) +=
                  dt_a_ex *
                  pkgs->get_package<HydroPackage>("Hydro")->get_flux_u(j, i);
            });
      } // End inner loop

      // set U_s
      athelas::par_for(
          DEFAULT_LOOP_PATTERN, "Timestepper :: EX :: Set Us", DevExecSpace(),
          ib.s, ib.e, qb.s, qb.e,
          KOKKOS_CLASS_LAMBDA(const int i, const int q) {
            for (int v = vb.s; v <= vb.e; ++v) {
              u(i, q, v) = SumVar_U_(i, q, v);
            }
          });

      auto x_l_sumvar_s = Kokkos::subview(x_l_sumvar_, iS, Kokkos::ALL);
      grid_s_[iS] = grid;
      grid_s_[iS].update_grid(x_l_sumvar_s);

      apply_slope_limiter(sl_hydro, u, &grid_s_[iS], fluid_basis, eos);
      bel::apply_bound_enforcing_limiter(stage_data, grid_s_[iS]);

      dt_info.stage = iS;
      pkgs->fill_derived(stage_data, grid_s_[iS], dt_info);
      pkgs->update_explicit(stage_data, grid_s_[iS], dt_info);
    } // end outer loop

    // --- Final U^n update ---

    for (int iS = 0; iS < nStages_; ++iS) {
      dt_info.stage = iS;
      dt_info.t = t + integrator_.explicit_tableau.c_i(iS) * dt;
      const double dt_b_ex = dt * integrator_.explicit_tableau.b_i(iS);
      dt_info.dt_coef = dt_b_ex;

      pkgs->apply_delta(u0, dt_info);

      athelas::par_for(
          DEFAULT_FLAT_LOOP_PATTERN, "Timestepper :: EX :: Finalize grid",
          DevExecSpace(), ib.s, ib.e, KOKKOS_CLASS_LAMBDA(const int i) {
            x_l_sumvar_(0, i) +=
                dt_b_ex *
                pkgs->get_package<HydroPackage>("Hydro")->get_flux_u(iS, i);
          });
      auto x_l_sumvar_j = Kokkos::subview(x_l_sumvar_, 0, Kokkos::ALL);
      grid_s_[iS] = grid;
      grid_s_[iS].update_grid(x_l_sumvar_j);
    }

    auto sd0 = mesh_state(0);
    grid = grid_s_[nStages_ - 1];
    apply_slope_limiter(sl_hydro, u0, &grid, sd0.fluid_basis(), sd0.eos());
    bel::apply_bound_enforcing_limiter(sd0, grid);

    pkgs->zero_delta();
  }

  /**
   * Update rad hydro solution with SSPRK methods
   **/
  void step_imex(PackageManager *pkgs, MeshState &mesh_state,
                 GridStructure &grid, const double t, const double dt,
                 SlopeLimiter *sl_hydro, SlopeLimiter *sl_rad) {

    update_rad_hydro_imex(pkgs, mesh_state, grid, t, dt, sl_hydro, sl_rad);
  }

  /**
   * Fully coupled IMEX rad hydro update with SSPRK methods
   **/
  void update_rad_hydro_imex(PackageManager *pkgs, MeshState &mesh_state,
                             GridStructure &grid, const double t,
                             const double dt, SlopeLimiter *sl_hydro,
                             SlopeLimiter *sl_rad) {
    static const int nnodes = grid.n_nodes();

    static const int nvars = mesh_state.nvars("u_cf");
    static const IndexRange ib(grid.domain<Domain::Entire>());
    static const IndexRange qb(nnodes);
    static const IndexRange vb(nvars);

    grid_s_[0] = grid;

    TimeStepInfo dt_info{.t = t, .dt = dt, .stage = 0};

    const auto &fluid_basis = mesh_state.fluid_basis();
    const auto &rad_basis = mesh_state.rad_basis();
    const auto &eos = mesh_state.eos();

    auto u0 = mesh_state(0).get_field("u_cf");
    for (int iS = 0; iS < nStages_; ++iS) {
      auto left_interface = grid_s_[iS].x_l();
      dt_info.stage = iS;
      dt_info.t = t + integrator_.explicit_tableau.c_i(iS) * dt;
      auto stage_data = mesh_state.stage(iS);
      auto u = stage_data.get_field("u_cf");
      athelas::par_for(
          DEFAULT_LOOP_PATTERN, "Timestepper :: IMEX :: Reset sumvar",
          DevExecSpace(), ib.s, ib.e, qb.s, qb.e, vb.s, vb.e,
          KOKKOS_CLASS_LAMBDA(const int i, const int q, const int v) {
            SumVar_U_(i, q, v) = u0(i, q, v);
            x_l_sumvar_(iS, i) = left_interface(i);
          });

      // --- Inner update loop ---

      for (int j = 0; j < iS; ++j) {
        dt_info.stage = j;
        dt_info.t = t + integrator_.explicit_tableau.c_i(j) * dt;
        const double dt_a = dt * integrator_.explicit_tableau.a_ij(iS, j);
        const double dt_a_im = dt * integrator_.implicit_tableau.a_ij(iS, j);
        dt_info.dt_coef = dt_a;
        dt_info.dt_coef_implicit = dt_a_im;

        pkgs->apply_delta(SumVar_U_, dt_info);

        athelas::par_for(
            DEFAULT_FLAT_LOOP_PATTERN, "Timestepper :: IMEX :: Update grid",
            DevExecSpace(), ib.s, ib.e, KOKKOS_CLASS_LAMBDA(const int i) {
              x_l_sumvar_(iS, i) +=
                  dt_a * pkgs->get_package<RadHydroPackage>("RadHydro")
                             ->get_flux_u(j, i);
            });
      } // End inner loop

      auto x_l_sumvar_j = Kokkos::subview(x_l_sumvar_, iS, Kokkos::ALL);
      grid_s_[iS] = grid;
      grid_s_[iS].update_grid(x_l_sumvar_j);

      // set U_s (stage data)
      athelas::par_for(
          DEFAULT_LOOP_PATTERN, "Timestepper :: IMEX :: Update Us",
          DevExecSpace(), ib.s, ib.e, qb.s, qb.e, vb.s, vb.e,
          KOKKOS_CLASS_LAMBDA(const int i, const int q, const int v) {
            u(i, q, v) = SumVar_U_(i, q, v);
          });

      // NOTE: The limiting strategies in this function will fail if
      // the pkg does not have access to a rad_basis and fluid_basis
      // limiting madness
      apply_slope_limiter(sl_hydro, u, &grid_s_[iS], fluid_basis, eos);
      apply_slope_limiter(sl_rad, u, &grid_s_[iS], rad_basis, eos);
      bel::apply_bound_enforcing_limiter(stage_data, grid_s_[iS]);
      bel::apply_bound_enforcing_limiter_rad(stage_data, grid_s_[iS]);

      // set U_s (stage data)
      athelas::par_for(
          DEFAULT_LOOP_PATTERN, "Timestepper :: IMEX :: Update Us",
          DevExecSpace(), ib.s, ib.e, qb.s, qb.e, vb.s, vb.e,
          KOKKOS_CLASS_LAMBDA(const int i, const int q, const int v) {
            SumVar_U_(i, q, v) = u(i, q, v);
          });

      // implicit update
      dt_info.stage = iS;
      dt_info.t = t + integrator_.explicit_tableau.c_i(iS) * dt;
      dt_info.dt_coef = dt * integrator_.implicit_tableau.a_ij(iS, iS);

      // Need a fill derived?
      // pkgs->fill_derived(stage_data, grid_s_[iS], dt_info);
      if (dt_info.dt_coef != 0.0) {
      pkgs->update_implicit_iterative(stage_data, SumVar_U_, grid_s_[iS],
                                      dt_info);
      }

      apply_slope_limiter(sl_hydro, u, &grid_s_[iS], fluid_basis, eos);
      apply_slope_limiter(sl_rad, u, &grid_s_[iS], rad_basis, eos);
      bel::apply_bound_enforcing_limiter(stage_data, grid_s_[iS]);
      bel::apply_bound_enforcing_limiter_rad(stage_data, grid_s_[iS]);

      dt_info.stage = iS;
      pkgs->fill_derived(stage_data, grid_s_[iS], dt_info);
      pkgs->update_explicit(stage_data, grid_s_[iS], dt_info);
      pkgs->update_implicit(stage_data, grid_s_[iS], dt_info);
    } // end outer loop

    for (int iS = 0; iS < nStages_; ++iS) {
      dt_info.stage = iS;
      dt_info.t = t + integrator_.explicit_tableau.c_i(iS) * dt;
      const double dt_b = dt * integrator_.explicit_tableau.b_i(iS);
      const double dt_b_im = dt * integrator_.implicit_tableau.b_i(iS);
      dt_info.dt_coef = dt_b;
      dt_info.dt_coef_implicit = dt_b_im;

      pkgs->apply_delta(u0, dt_info);

      athelas::par_for(
          DEFAULT_FLAT_LOOP_PATTERN, "Timestepper :: IMEX :: Finalize grid",
          DevExecSpace(), ib.s, ib.e, KOKKOS_CLASS_LAMBDA(const int i) {
            x_l_sumvar_(0, i) +=
                dt_b * pkgs->get_package<RadHydroPackage>("RadHydro")
                           ->get_flux_u(iS, i);
          });
      auto x_l_sumvar_j = Kokkos::subview(x_l_sumvar_, 0, Kokkos::ALL);
      grid_s_[iS] = grid;
      grid_s_[iS].update_grid(x_l_sumvar_j);
    }

    auto sd0 = mesh_state(0);
    grid = grid_s_[nStages_ - 1];
    apply_slope_limiter(sl_hydro, u0, &grid, fluid_basis, eos);
    apply_slope_limiter(sl_rad, u0, &grid, rad_basis, eos);
    bel::apply_bound_enforcing_limiter(sd0, grid);
    bel::apply_bound_enforcing_limiter_rad(sd0, grid);

    pkgs->zero_delta();
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

  // x_l_sumvar_ Holds cell left interface positions
  AthelasArray2D<double> x_l_sumvar_;

  // Variables to pass to update step
  eos::EOS *eos_;
};

} // namespace athelas
