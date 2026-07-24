/**
 * @file gravity_package.hpp
 * --------------
 *
 * @brief Gravitational source package
 **/

#pragma once

#include "basic_types.hpp"
#include "basis/nodal_basis.hpp"
#include "bc/boundary_conditions_base.hpp"
#include "geometry/mesh.hpp"
#include "interface/packages_base.hpp"
#include "interface/state.hpp"
#include "pgen/problem_in.hpp"

namespace athelas::gravity {
namespace pkg_vars {
constexpr int Velocity = 0;
constexpr int Energy = 1;
} // namespace pkg_vars

using bc::BoundaryConditions;

class GravityPackage {
 public:
  GravityPackage(const ProblemIn *pin, const std::string &model, double gval,
                 double cfl, int n_stages, bool active = true);

  [[nodiscard]] auto update_explicit(const StageData &stage_data,
                                     const TimeStepInfo &dt_info) const
      -> UpdateStatus;

  template <GravityModel Model>
  void gravity_update(const Mesh &mesh, const basis::NodalBasis &basis,
                      AthelasArray2D<double> interface,
                      AthelasArray2D<double> dtau_dt, int stage,
                      int idx_vstar) const;

  // Direct, self-contained gravity source for operator-split mode. Each node gets
  // the pointwise acceleration g and an exact frozen-mesh kinetic-energy kick:
  // dE / dt = g * (v + 0.5 dt g). Generic apply_delta then gives
  // E_new - E_old = 0.5 [(v + dt g)^2 - v^2], preserving internal energy. This
  // still gives up the coupled form's well-balancing and total-energy
  // conservation: the gravity substep does not move the mesh, so its work is not
  // matched by a change in the discrete potential energy W_h.
  template <GravityModel Model>
  void update_split(const Mesh &mesh, AthelasArray3D<double> evolved, int stage,
                    int idx_vel, double dt) const;

  // Fill tau_dot_mesh_velocity_ with the time derivative of each reconstructed
  // mesh node, from the chain rule of the cumulative-volume integral using the
  // published dtau_dt (branch-aware: high-order cells use the integration
  // matrix, fallback cells the clamped mass-fraction interpolation).
  void compute_mesh_velocity(const Mesh &mesh, AthelasArray2D<double> interface,
                             AthelasArray2D<double> dtau_dt,
                             int idx_vstar) const;

  void apply_delta(AthelasArray3D<double> lhs,
                   const TimeStepInfo &dt_info) const;

  // --- Limiter energy correction (opt-in) ---------------------------------
  // The limiters preserve the mass-weighted cell average of tau, so cell faces
  // do not move, but they reshape tau within a cell and thereby relocate the
  // interior nodes -- changing the discrete gravitational potential energy W_h
  // with no compensating source. When limiter_energy_correction is enabled, the
  // methods below measure that change and return it to the fluid. Usage:
  // snapshot_limiter_radii before the end-of-step limiter block (in the
  // timestepper), apply_limiter_energy_correction after it (in post_step_work).
  // Both are no-ops of the caller's own gating when the correction is off.

  [[nodiscard]] auto corrects_limiter_energy() const noexcept -> bool;

  // Snapshot the current nodal radii. Call immediately before a limiter block,
  // while the mesh still reflects the pre-limiter tau.
  void snapshot_limiter_radii(const Mesh &mesh) const;

  // Call after the mesh has been reconstructed from the limited tau. For each
  // cell it computes dW_i = 4pi sum_q w_q mu_q [phi(r_q^new) - phi(r_q^old)]
  // and returns -dW_i / m_i uniformly to the cell's specific energy, clamped so
  // the specific internal energy e = E - v^2/2 stays at or above the EOS floor
  // min_sie(eos, rho, lambda) at every node; only tau moves the mesh, so this
  // energy change has no geometric feedback. 
  // Accumulates the mesh work, applied
  // correction, and clamp residual into the cumulative diagnostics below.
  void apply_limiter_energy_correction(const StageData &stage_data,
                                       const Mesh &mesh,
                                       AthelasArray3D<double> evolved,
                                       int idx_vel, int idx_energy) const;

  // Cumulative limiter-energy diagnostics, accumulated across steps. 
  // mesh work is the total change in W_h the limiter
  // caused; correction is what was returned to the fluid; clamp residual is
  // what the EOS floor prevented returning.
  [[nodiscard]] auto cumulative_limiter_mesh_work() const noexcept -> double;
  [[nodiscard]] auto cumulative_limiter_energy_correction() const noexcept
      -> double;
  [[nodiscard]] auto cumulative_limiter_energy_clamp_residual() const noexcept
      -> double;

  [[nodiscard]] auto restart_scalars() const -> PackageRestartScalars;
  void load_restart_scalars(const PackageRestartScalars &scalars);

  void zero_delta() const noexcept;

  [[nodiscard]] auto min_timestep(const StageData &stage_data,
                                  const TimeStepInfo & /*dt_info*/) const
      -> double;

  [[nodiscard]] auto name() const noexcept -> std::string_view;

  [[nodiscard]] auto is_active() const noexcept -> bool;

  void fill_derived(StageData &stage_data, const TimeStepInfo &dt_info) const;

  void set_active(bool active);

 private:
  bool active_;
  GravityModel model_;
  double gval_; // constant gravity
  double cfl_;
  bool split_; // operator-split (direct source) vs coupled (weak source)
  bool correct_limiter_energy_;

  AthelasArray4D<double> delta_; // rhs delta [nstages, nx, nq, nvars]
  AthelasArray2D<double> gravity_pressure_;
  mutable AthelasArray2D<double> tau_dot_mesh_velocity_;
  // Nodal radii snapshotted before a limiter block for the energy correction.
  mutable AthelasArray2D<double> limiter_r_old_;

  // Cumulative limiter-energy diagnostics, accumulated by
  // apply_limiter_energy_correction.
  mutable double cumulative_limiter_mesh_work_{};
  mutable double cumulative_limiter_energy_correction_{};
  mutable double cumulative_limiter_energy_clamp_residual_{};

  static constexpr int NUM_VARS_ = 2;
};

} // namespace athelas::gravity
