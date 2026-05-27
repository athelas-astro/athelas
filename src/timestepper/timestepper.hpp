/**
 * @file timestepper.hpp
 * --------------
 *
 * @brief Primary time marching routine.
 *
 * @details Timestppers for hydro and rad hydro.
 *          Uses explicit for transport terms and implicit for coupling.
 */

#pragma once

#include <vector>

#include "basic_types.hpp"
#include "geometry/grid.hpp"
#include "kokkos_types.hpp"
#include "limiters/slope_limiter.hpp"
#include "timestepper/tableau.hpp"

namespace athelas {

class MeshState;
class PackageManager;
class ProblemIn;

class TimeStepper {

 public:
  // TODO(astrobarker): Is it possible to initialize grid_s_ from grid directly?
  TimeStepper(const ProblemIn *pin, GridStructure *grid);

  void initialize_timestepper();

  /**
   * Update fluid solution with SSPRK methods
   **/
  void step(PackageManager *pkgs, MeshState &mesh_state, GridStructure &grid,
            TimeStepInfo &dt_info, SlopeLimiter *sl_hydro);

  /**
   * Explicit fluid update with SSPRK methods
   **/
  void update_fluid_explicit(PackageManager *pkgs, MeshState &mesh_state,
                             GridStructure &grid, TimeStepInfo &dt_info,
                             SlopeLimiter *sl_hydro);

  /**
   * Update rad hydro solution with SSPRK methods
   **/
  void step_imex(PackageManager *pkgs, MeshState &mesh_state,
                 GridStructure &grid, TimeStepInfo &dt_info,
                 SlopeLimiter *sl_hydro, SlopeLimiter *sl_rad);

  /**
   * Fully coupled IMEX rad hydro update with SSPRK methods
   **/
  void update_rad_hydro_imex(PackageManager *pkgs, MeshState &mesh_state,
                             GridStructure &grid, TimeStepInfo &dt_info,
                             SlopeLimiter *sl_hydro, SlopeLimiter *sl_rad);

  [[nodiscard]] auto n_stages() const noexcept -> int;
  [[nodiscard]] static auto nvars_evolved(const ProblemIn *pin) noexcept -> int;

 private:
  void seed_stage_grids(const GridStructure &grid);
  void reset_stage_sumvar(int stage, AthelasArray3D<double> u0,
                          const IndexRange &ib, const IndexRange &qb,
                          const IndexRange &vb, const char *label);
  void accumulate_grid_motion(MeshState &mesh_state, int sum_stage,
                              int data_stage, double dt_coef,
                              const IndexRange &ib, const char *label);
  void update_stage_grid(const GridStructure &grid, int grid_stage,
                         int sum_stage);

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
};

} // namespace athelas
