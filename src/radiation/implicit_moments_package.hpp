#pragma once

#include "Kokkos_Macros.hpp"
#include "basic_types.hpp"
#include "bc/boundary_conditions_base.hpp"
#include "geometry/mesh.hpp"
#include "interface/params.hpp"
#include "interface/state.hpp"
#include "pgen/problem_in.hpp"
#include "radiation/rad_utilities.hpp"

namespace athelas::radiation {

void radiation_source_implicit(const StageData &stage_data,
                               AthelasArray3D<double> R,
                               AthelasArray4D<double> delta, const Mesh &mesh,
                               const TimeStepInfo &dt_info);

using bc::BoundaryConditions;

// Scratch views used by the implicit transport solver. Grouped by purpose
// so the package class doesn't carry a flat list of ~13 loose member views.
// All allocated once in the package constructor and reused across calls.

// Face states (filled by evaluate_residual) and the per-face flux Jacobian
// blocks used by the matrix-build kernel.
struct TransportFaceScratch {
  AthelasArray2D<double> u_f_l; // left  side at face i: (face_idx, var)
  AthelasArray2D<double> u_f_r; // right side at face i
  AthelasArray2D<double> flux_num; // numerical fluxes at faces
  AthelasArray3D<double> A_minus; // dF_hat/dU_L per interior face
  AthelasArray3D<double> A_plus; // dF_hat/dU_R per interior face
};

// Block-tridiagonal Newton system and Thomas-solver workspace.
struct BlockTridiagSolver {
  AthelasArray3D<double> mat_diag;
  AthelasArray3D<double> mat_upper;
  AthelasArray3D<double> mat_lower;
  AthelasArray2D<double> b; // rhs / Newton step delta after solve
  AthelasArray3D<double> W; // Thomas scratch
  AthelasArray2D<double> Y;
  AthelasArray2D<double> Bi_lu;
};

// Newton iterate and line-search trial state.
struct NewtonScratch {
  // (nx+2, nq, 5) storing (tau, vel, ener, specific er, specific fr).
  AthelasArray3D<double> u_rad_work;

  // Line-search trial: same layout as u_rad_work, holds the damped step's
  // proposed state for residual evaluation.
  AthelasArray3D<double> u_rad_trial;

  // Trial residual at u_rad_trial, same layout as solver_.b.
  AthelasArray2D<double> ls_b_trial;
};

// History needed for the radiation-physics-based dt restriction.
struct ImplicitTimestepHistory {
  AthelasArray2D<double> e_rad_old;
  AthelasArray2D<double> f_rad_old;
};

/**
 * @brief Implicit radiation moments
 * Used for fully implicit transport
 */
class ImplicitRadiationMomentsPackage {
 public:
  ImplicitRadiationMomentsPackage(const ProblemIn * /*pin*/, int n_stages,
                                  int nq, BoundaryConditions *bcs, int nx,
                                  bool active = true);

  void update_implicit(const StageData &stage_data,
                       AthelasArray3D<double> ustar,
                       const TimeStepInfo &dt_info);

  // Compute the implicit-transport residual b_out = -R(U), where
  // R = M (U - U*) - dt_aii * (T(U) + S(U)), T is the DG transport operator
  // (volume + surface), and S are the sources.
  // Side effect: refills u_f_l_, u_f_r_, flux_num_ from U after refreshing
  // copy-only halo cells.
  void evaluate_residual(AthelasArray2D<double> b_out, AthelasArray3D<double> U,
                         AthelasArray3D<double> ustar,
                         const StageData &stage_data, const Mesh &mesh,
                         double dt_aii);

  void apply_delta(AthelasArray3D<double> lhs,
                   const TimeStepInfo &dt_info) const;

  void zero_delta() const noexcept;

  [[nodiscard]] auto min_timestep(const StageData &stage_data,
                                  const TimeStepInfo & /*dt_info*/) const
      -> double;

  [[nodiscard]] auto name() const noexcept -> std::string_view;

  [[nodiscard]] auto is_active() const noexcept -> bool;

  void fill_derived(StageData &stage_data, const TimeStepInfo &dt_info) const;

  void set_active(bool active);

  [[nodiscard]] static constexpr auto num_vars() noexcept -> int {
    return NUM_VARS_;
  }

 private:
  bool active_;

  BoundaryConditions *bcs_;

  // Grouped scratch (see struct definitions above).
  TransportFaceScratch faces_;
  BlockTridiagSolver solver_;
  NewtonScratch newton_;
  ImplicitTimestepHistory dt_cache_;

  AthelasArray4D<double> delta_;

  // package params
  Params params_;

  // constants
  static constexpr int NUM_VARS_ = 4;
};
} // namespace athelas::radiation
