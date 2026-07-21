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

  // Fill tau_dot_mesh_velocity_ with the time derivative of each reconstructed
  // mesh node, from the chain rule of the cumulative-volume integral using the
  // published dtau_dt (branch-aware: high-order cells use the integration
  // matrix, fallback cells the clamped mass-fraction interpolation).
  void compute_mesh_velocity(const Mesh &mesh, AthelasArray2D<double> interface,
                             AthelasArray2D<double> dtau_dt,
                             int idx_vstar) const;

  void apply_delta(AthelasArray3D<double> lhs,
                   const TimeStepInfo &dt_info) const;

  // --- Limiter energy correction (experimental, flag-gated) ---------------
  // The limiters preserve the mass-weighted cell average of tau, so cell faces
  // do not move, but they reshape tau within a cell and thereby relocate the
  // interior nodes -- changing the discrete gravitational potential energy W_h
  // with no compensating source. These two methods return that energy to the
  // fluid so total energy is conserved across a limiter block.

  [[nodiscard]] auto corrects_limiter_energy() const noexcept -> bool;

  // Snapshot the current nodal radii. Call immediately before a limiter block,
  // while the mesh still reflects the pre-limiter tau.
  void snapshot_limiter_radii(const Mesh &mesh) const;

  // Result of one correction: energies are 4pi-weighted (erg in spherical).
  struct LimiterCorrection {
    double applied;       // total specific-energy change returned to the fluid
    double clamp_residual; // energy the e >= 0 clamp prevented returning
  };

  // Call after the mesh has been reconstructed from the limited tau. For each
  // cell, returns dW_i = 4pi sum_q w_q mu_q [phi(r_q^new) - phi(r_q^old)] to
  // the fluid as a uniform specific-energy shift dE_i = -dW_i / m_i, clamped so
  // the specific internal energy e = E - v^2/2 stays at or above the EOS floor
  // min_sie(eos, rho, lambda) at every node. Only tau moves the mesh, so this
  // energy change has no geometric feedback. Takes StageData for the EOS and
  // ionization state used to evaluate that floor.
  [[nodiscard]] auto
  apply_limiter_energy_correction(const StageData &stage_data, const Mesh &mesh,
                                  AthelasArray3D<double> evolved, int idx_vel,
                                  int idx_energy) const -> LimiterCorrection;

  void zero_delta() const noexcept;

  [[nodiscard]] auto min_timestep(const StageData &stage_data,
                                  const TimeStepInfo & /*dt_info*/) const
      -> double;

  [[nodiscard]] auto name() const noexcept -> std::string_view;

  [[nodiscard]] auto is_active() const noexcept -> bool;

  // Gravity model and constant-model acceleration coefficient, so callers
  // (history, timestepper) can evaluate the same gravitational_potential the
  // package uses.
  [[nodiscard]] auto model() const noexcept -> GravityModel;
  [[nodiscard]] auto gval() const noexcept -> double;

  void fill_derived(StageData &stage_data, const TimeStepInfo &dt_info) const;

  void set_active(bool active);

 private:
  bool active_;
  GravityModel model_;
  double gval_; // constant gravity
  double cfl_;
  bool correct_limiter_energy_;

  AthelasArray4D<double> delta_; // rhs delta [nstages, nx, nq, nvars]
  AthelasArray2D<double> gravity_pressure_;
  mutable AthelasArray2D<double> tau_dot_mesh_velocity_;
  // Nodal radii snapshotted before a limiter block for the energy correction.
  mutable AthelasArray2D<double> limiter_r_old_;

  static constexpr int NUM_VARS_ = 2;
};

} // namespace athelas::gravity
