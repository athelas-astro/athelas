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
  explicit TimeStepper(const ProblemIn *pin);

  void initialize_timestepper();

  /**
   * Update fluid solution with SSPRK methods
   **/
  void step(PackageManager *pkgs, MeshState &mesh_state, TimeStepInfo &dt_info,
            SlopeLimiter *sl_hydro);

  /**
   * Explicit fluid update with SSPRK methods
   **/
  void update_fluid_explicit(PackageManager *pkgs, MeshState &mesh_state,
                             TimeStepInfo &dt_info, SlopeLimiter *sl_hydro);

  /**
   * Update rad hydro solution with SSPRK methods
   **/
  void step_imex(PackageManager *pkgs, MeshState &mesh_state,
                 TimeStepInfo &dt_info, SlopeLimiter *sl_hydro,
                 SlopeLimiter *sl_rad);

  /**
   * Fully coupled IMEX rad hydro update with SSPRK methods
   **/
  void update_rad_hydro_imex(PackageManager *pkgs, MeshState &mesh_state,
                             TimeStepInfo &dt_info, SlopeLimiter *sl_hydro,
                             SlopeLimiter *sl_rad);

  [[nodiscard]] auto n_stages() const noexcept -> int;
  [[nodiscard]] static auto nvars_evolved(const ProblemIn *pin) noexcept -> int;

 private:
  void reset_stage_sumvar(AthelasArray3D<double> u0, const IndexRange &ib,
                          const IndexRange &qb, const IndexRange &vb,
                          const char *label);
  void update_stage_mesh(MeshState &mesh_state, int stage,
                         AthelasArray3D<double> evolved);

  // Inner-face radius buffer for mesh reconstruction. The interior grid is
  // recovered from tau; the inner face is the single position degree of freedom
  // tau cannot supply, so it is advanced Lagrangianly by v*(ilo) with the same
  // RK accumulation as the state (one scalar per stage).
  void reset_x_inner_buffer(const Mesh &mesh, int stage);
  void accumulate_x_inner_buffer(MeshState &mesh_state, int sum_stage,
                                 int data_stage, double dt_coef);
  [[nodiscard]] auto x_inner_buffer(int stage) -> double;

  int nvars_evolved_;
  int mSize_;
  int nNodes_;

  // tableaus
  RKIntegrator integrator_;

  int nStages_;
  int tOrder_;

  // Hold stage data
  AthelasArray3D<double> SumVar_U_;
  AthelasArray3D<double> u0_buffer_;

  // Per-stage accumulated inner-face radius for mesh reconstruction.
  AthelasArray1D<double> x_inner_sumvar_;
};

} // namespace athelas
