#pragma once

#include "Kokkos_Macros.hpp"
#include "basic_types.hpp"
#include "bc/boundary_conditions_base.hpp"
#include "geometry/grid.hpp"
#include "interface/params.hpp"
#include "interface/state.hpp"
#include "pgen/problem_in.hpp"
#include "radiation/rad_utilities.hpp"

namespace athelas::radiation {

void radiation_source_implicit(const StageData &stage_data,
                               AthelasArray3D<double> R,
                               AthelasArray4D<double> delta,
                               const GridStructure &grid,
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
  // Boundary-face Jacobians split so the kernel can apply distinct phi
  // factors. A_bndry = direct (interior) part; A_bndry_ghost = ghost-side
  // part, multiplied by d_bndry (BC variable-Jacobian) in the kernel.
  AthelasArray2D<double> A_bndry;
  AthelasArray2D<double> A_bndry_ghost;
  AthelasArray2D<double> d_bndry;
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
                       AthelasArray3D<double> ustar, const GridStructure &grid,
                       const TimeStepInfo &dt_info);

  // Compute the implicit-transport residual b_out = -R(U), where
  // R = M (U - U*) - dt_aii * (T(U) + S(U)), T is the DG transport operator
  // (volume + surface), and S are the sources.
  // Side effect: refills u_f_l_, u_f_r_, flux_num_
  // from U after applying ghost-zone BCs to U.
  void evaluate_residual(AthelasArray2D<double> b_out, AthelasArray3D<double> U,
                         AthelasArray3D<double> ustar,
                         const StageData &stage_data, const GridStructure &grid,
                         double dt_aii);

  void apply_delta(AthelasArray3D<double> lhs,
                   const TimeStepInfo &dt_info) const;

  void zero_delta() const noexcept;

  [[nodiscard]] auto min_timestep(const StageData & /*stage_data*/,
                                  const GridStructure &grid,
                                  const TimeStepInfo & /*dt_info*/) const
      -> double;

  [[nodiscard]] auto name() const noexcept -> std::string_view;

  [[nodiscard]] auto is_active() const noexcept -> bool;

  void fill_derived(StageData &stage_data, const GridStructure &grid,
                    const TimeStepInfo &dt_info) const;

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

// Split flux Jacobian at a boundary face into two independent pieces that
// the caller multiplies by separate phi factors:
//
//   A_direct = dF_hat/dU_interior (the side of the face facing the interior)
//   A_ghost  = dF_hat/dU_ghost    (the side facing the ghost, BEFORE applying
//                                  the BC's variable-Jacobian D)
//
// The caller is responsible for combining A_ghost with D and for using the
// correct column-side phi/node-mapping for each piece, because:
//   - Direct uses basis-eval at the interior cell's near-face edge.
//   - Ghost uses basis-eval at the ghost cell's near-face edge (the *other*
//     edge from the interior's perspective) and may also need a node
//     permutation for reflecting/Marshak BCs.
//
// Variables on second axis of ufl/ufr: 0 = specific volume, 1 = specific
// radiation energy density, 2 = specific radiation flux.
//
// TODO(astrobarker): add the dalpha/dU contribution generally — here in
// boundary_jacobian *and* in the A_minus_ / A_plus_ assembly in
// update_implicit — so the analytic Jacobian is exact.
KOKKOS_FUNCTION
template <Boundary Loc>
void boundary_jacobian(AthelasArray2D<double> A_direct,
                       AthelasArray2D<double> A_ghost,
                       AthelasArray2D<double> ufl, AthelasArray2D<double> ufr,
                       const double vstar) {
  using math::utils::sgn;
  constexpr double c = constants::c_cgs;
  constexpr double c2 = c * c;
  if constexpr (Loc == Boundary::Interior) {
    // Inner face: interior on R side, ghost on L side.
    // A_direct uses the A_plus shape (-alpha), A_ghost uses A_minus (+alpha).
    // alpha must use BOTH ghost (L) and interior (R) states to match the
    // alpha actually used in the LLF flux at this face.
    constexpr int i_inner = 1;
    const double alpha = rad_wavespeed(ufl(i_inner, 1), ufr(i_inner, 1),
                                       ufl(i_inner, 2), ufr(i_inner, 2), vstar);

    // Direct (interior, R side).
    double f = flux_factor(ufr(i_inner, 1), ufr(i_inner, 2));
    double chi = eddington_factor(f);
    double chi_prime = eddington_factor_prime(f);
    A_direct(0, 0) = 0.5 * (-vstar - alpha);
    A_direct(0, 1) = 0.5;
    A_direct(1, 0) = 0.5 * c2 * (chi - f * chi_prime);
    A_direct(1, 1) =
        0.5 * (c * chi_prime * sgn(ufr(i_inner, 2)) - vstar - alpha);

    // Ghost (L side).
    f = flux_factor(ufl(i_inner, 1), ufl(i_inner, 2));
    chi = eddington_factor(f);
    chi_prime = eddington_factor_prime(f);
    A_ghost(0, 0) = 0.5 * (-vstar + alpha);
    A_ghost(0, 1) = 0.5;
    A_ghost(1, 0) = 0.5 * c2 * (chi - f * chi_prime);
    A_ghost(1, 1) =
        0.5 * (c * chi_prime * sgn(ufl(i_inner, 2)) - vstar + alpha);
  }
  if constexpr (Loc == Boundary::Exterior) {
    // Outer face: interior on L side, ghost on R side.
    // A_direct uses A_minus (+alpha), A_ghost uses A_plus (-alpha).
    static const int i_outer = static_cast<int>(ufl.extent(0)) - 1;
    const double alpha = rad_wavespeed(ufl(i_outer, 1), ufr(i_outer, 1),
                                       ufl(i_outer, 2), ufr(i_outer, 2), vstar);

    // Direct (interior, L side).
    double f = flux_factor(ufl(i_outer, 1), ufl(i_outer, 2));
    double chi = eddington_factor(f);
    double chi_prime = eddington_factor_prime(f);
    A_direct(0, 0) = 0.5 * (-vstar + alpha);
    A_direct(0, 1) = 0.5;
    A_direct(1, 0) = 0.5 * c2 * (chi - f * chi_prime);
    A_direct(1, 1) =
        0.5 * (c * chi_prime * sgn(ufl(i_outer, 2)) - vstar + alpha);

    // Ghost (R side).
    f = flux_factor(ufr(i_outer, 1), ufr(i_outer, 2));
    chi = eddington_factor(f);
    chi_prime = eddington_factor_prime(f);
    A_ghost(0, 0) = 0.5 * (-vstar - alpha);
    A_ghost(0, 1) = 0.5;
    A_ghost(1, 0) = 0.5 * c2 * (chi - f * chi_prime);
    A_ghost(1, 1) =
        0.5 * (c * chi_prime * sgn(ufr(i_outer, 2)) - vstar - alpha);
  }
}
} // namespace athelas::radiation
