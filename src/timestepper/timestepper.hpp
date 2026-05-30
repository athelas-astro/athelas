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

#include "geometry/mesh.hpp"
#include "kokkos_types.hpp"
#include "limiters/slope_limiter.hpp"
#include "timestepper/tableau.hpp"

namespace athelas {

class MeshState;
class PackageManager;
class ProblemIn;
struct IndexRange;
struct TimeStepInfo;

class TimeStepper {

 public:
  // TODO(astrobarker): Is it possible to initialize mesh_s_ from mesh directly?
  TimeStepper(const ProblemIn *pin, Mesh *mesh);

  void initialize_timestepper();

  /**
   * Update fluid solution with SSPRK methods
   **/
  void step(PackageManager *pkgs, MeshState &mesh_state, Mesh &mesh,
            TimeStepInfo &dt_info, SlopeLimiter *sl_hydro);

  /**
   * Explicit fluid update with SSPRK methods
   **/
  void update_fluid_explicit(PackageManager *pkgs, MeshState &mesh_state,
                             Mesh &mesh, TimeStepInfo &dt_info,
                             SlopeLimiter *sl_hydro);

  /**
   * Update rad hydro solution with SSPRK methods
   **/
  void step_imex(PackageManager *pkgs, MeshState &mesh_state, Mesh &mesh,
                 TimeStepInfo &dt_info, SlopeLimiter *sl_hydro,
                 SlopeLimiter *sl_rad);

  /**
   * Fully coupled IMEX rad hydro update with SSPRK methods
   **/
  void update_rad_hydro_imex(PackageManager *pkgs, MeshState &mesh_state,
                             Mesh &mesh, TimeStepInfo &dt_info,
                             SlopeLimiter *sl_hydro, SlopeLimiter *sl_rad);

  [[nodiscard]] auto n_stages() const noexcept -> int;
  [[nodiscard]] static auto nvars_evolved(const ProblemIn *pin) noexcept -> int;
  // TEMPORARY: derive the RK stage count from the input without a constructed
  // TimeStepper. Needed only because MeshState (which owns the mesh) must be
  // built before the stepper, yet must be sized with nstages. Remove once the
  // per-stage meshes (mesh_s_) move out of TimeStepper into MeshState's staged
  // data, which decouples the construction order.
  [[nodiscard]] static auto compute_n_stages(const ProblemIn *pin) -> int;

 private:
  void seed_stage_meshes(const Mesh &mesh);
  void reset_stage_sumvar(int stage, AthelasArray3D<double> u0,
                          const IndexRange &ib, const IndexRange &qb,
                          const IndexRange &vb, const char *label);
  void accumulate_grid_motion(MeshState &mesh_state, int sum_stage,
                              int data_stage, double dt_coef,
                              const IndexRange &ib, const char *label);
  void update_stage_mesh(const Mesh &mesh, int mesh_stage, int sum_stage);

  int nvars_evolved_;
  int mSize_;

  // tableaus
  RKIntegrator integrator_;

  int nStages_;
  int tOrder_;

  // Hold stage data
  AthelasArray3D<double> SumVar_U_;
  std::vector<Mesh> mesh_s_;

  // x_l_sumvar_ Holds cell left interface positions
  AthelasArray2D<double> x_l_sumvar_;
};

} // namespace athelas
