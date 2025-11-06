/**
 * @file radhydro_package.hpp
 * --------------
 *
 * @brief Radiation hydrodynamics package
 */

#pragma once

#include "Kokkos_Macros.hpp"
#include "basic_types.hpp"
#include "basis/polynomial_basis.hpp"
#include "bc/boundary_conditions_base.hpp"
#include "composition/composition.hpp"
#include "eos/eos_variant.hpp"
#include "geometry/grid.hpp"
#include "opacity/opac_variant.hpp"
#include "pgen/problem_in.hpp"
#include "radiation/rad_utilities.hpp"
#include "solvers/root_finders.hpp"
#include "state/state.hpp"

namespace athelas::radiation {

using bc::BoundaryConditions;

class RadHydroPackage {
 public:
  RadHydroPackage(const ProblemIn * /*pin*/, int n_stages, eos::EOS *eos,
                  Opacity *opac, basis::ModalBasis *fluid_basis,
                  basis::ModalBasis *rad_basis, BoundaryConditions *bcs,
                  double cfl, int nx, bool active = true);

  void update_explicit(const State *const state, const GridStructure &grid,
                       const TimeStepInfo &dt_info) const;
  void update_implicit(const State *const state, const GridStructure &grid,
                       const TimeStepInfo &dt_info) const;
  void update_implicit_iterative(const State *const state,
                                 AthelasArray3D<double> R,
                                 const GridStructure &grid,
                                 const TimeStepInfo &dt_info);

  void apply_delta(AthelasArray3D<double> lhs,
                   const TimeStepInfo &dt_info) const;

  auto
  radhydro_source(const State *const state, const AthelasArray2D<double> uCRH,
                  const AthelasArray1D<double> dx,
                  const AthelasArray1D<double> weights,
                  const AthelasArray3D<double> phi_fluid,
                  const AthelasArray3D<double> phi_rad,
                  const AthelasArray2D<double> inv_mkk_fluid,
                  const AthelasArray2D<double> inv_mkk_rad, int i, int k) const
      -> std::tuple<double, double, double, double>;

  void radhydro_divergence(const State *const state, const GridStructure &grid,
                           int stage) const;

  [[nodiscard]] auto min_timestep(const State *const /*ucf*/,
                                  const GridStructure &grid,
                                  const TimeStepInfo & /*dt_info*/) const
      -> double;

  [[nodiscard]] auto name() const noexcept -> std::string_view;

  [[nodiscard]] auto is_active() const noexcept -> bool;

  void fill_derived(State *state, const GridStructure &grid,
                    const TimeStepInfo &dt_info) const;

  void set_active(bool active);

  [[nodiscard]] auto get_flux_u(int stage, int i) const -> double;
  [[nodiscard]] auto fluid_basis() const -> const basis::ModalBasis *;
  [[nodiscard]] auto rad_basis() const -> const basis::ModalBasis *;

  [[nodiscard]] static constexpr auto num_vars() noexcept -> int {
    return NUM_VARS_;
  }

 private:
  bool active_;

  int nx_;
  double cfl_;

  eos::EOS *eos_;
  Opacity *opac_;
  basis::ModalBasis *fluid_basis_;
  basis::ModalBasis *rad_basis_;
  BoundaryConditions *bcs_;

  // package storage
  AthelasArray2D<double> dFlux_num_; // stores Riemann solutions
  AthelasArray2D<double> u_f_l_; // left faces
  AthelasArray2D<double> u_f_r_; // right faces
  AthelasArray2D<double> flux_u_; // Riemann velocities

  AthelasArray3D<double> delta_; // rhs delta
  AthelasArray3D<double> delta_im_; // rhs delta

  // iterative solver storage
  AthelasArray3D<double> scratch_k_;
  AthelasArray3D<double> scratch_km1_;
  AthelasArray3D<double> scratch_sol_;

  // constants
  static constexpr int NUM_VARS_ = 5;
};

// This is duplicate of above but used differently, in the root finder
// The code needs some refactoring in order to get rid of this version.
KOKKOS_INLINE_FUNCTION
auto compute_increment_radhydro_source(
    const AthelasArray2D<double> uCRH, const int k, const State *const state,
    const AthelasArray1D<double> dx, const AthelasArray1D<double> weights,
    const AthelasArray3D<double> phi_fluid,
    const AthelasArray3D<double> phi_rad,
    const AthelasArray2D<double> inv_mkk_fluid,
    const AthelasArray2D<double> inv_mkk_rad, const eos::EOS *eos,
    const Opacity *opac, const int i)
    -> std::tuple<double, double, double, double> {
  using basis::basis_eval;
  constexpr static double c = constants::c_cgs;
  constexpr static double c2 = c * c;
  static const bool ionization_enabled = state->ionization_enabled();

  static const int nNodes = static_cast<int>(weights.size());
  const double &dr_i = dx(i);

  double local_sum_e_r = 0.0; // radiation energy source
  double local_sum_m_r = 0.0; // radiation momentum (flux) source
  double local_sum_e_g = 0.0; // gas energy source
  double local_sum_m_g = 0.0; // gas momentum (velocity) source
  for (int q = 0; q < nNodes; ++q) {
    const int qp1 = q + 1;
    const double &wq = weights(q);
    const double &phi_rad_kq = phi_rad(i, qp1, k);
    const double &phi_fluid_kq = phi_fluid(i, qp1, k);

    // Note: basis evaluations are awkward here.
    // must be sure to use the correct basis functions.
    const double tau =
        basis_eval(phi_fluid, uCRH, i, vars::cons::SpecificVolume, qp1);
    const double rho = 1.0 / tau;
    const double vel =
        basis_eval(phi_fluid, uCRH, i, vars::cons::Velocity, qp1);
    const double em_t = basis_eval(phi_fluid, uCRH, i, vars::cons::Energy, qp1);

    double lambda[8];
    if (ionization_enabled) {
      atom::paczynski_terms(state, i, qp1, lambda);
    }
    const double t_g = temperature_from_conserved(eos, tau, vel, em_t, lambda);

    // TODO(astrobarker): composition
    // Should I move these into a lambda?
    const double X = 1.0;
    const double Y = 1.0;
    const double Z = 1.0;

    const double kappa_r = rosseland_mean(opac, rho, t_g, X, Y, Z, lambda);
    const double kappa_p = planck_mean(opac, rho, t_g, X, Y, Z, lambda);

    const double E_r = basis_eval(phi_rad, uCRH, i, vars::cons::RadEnergy, qp1);
    const double F_r = basis_eval(phi_rad, uCRH, i, vars::cons::RadFlux, qp1);
    const double P_r = compute_closure(E_r, F_r);

    // 4 force
    const auto [G0, G] =
        radiation_four_force(rho, vel, t_g, kappa_r, kappa_p, E_r, F_r, P_r);

    const double source_e_r = -c * G0;
    const double source_m_r = -c2 * G;
    const double source_e_g = c * G0;
    const double source_m_g = G;

    local_sum_e_r += wq * phi_rad_kq * source_e_r;
    local_sum_m_r += wq * phi_rad_kq * source_m_r;
    local_sum_e_g += wq * phi_fluid_kq * source_e_g;
    local_sum_m_g += wq * phi_fluid_kq * source_m_g;
  }
  // \Delta x / M_kk
  const double dx_o_mkk_fluid = dr_i * inv_mkk_fluid(i, k);
  const double dx_o_mkk_rad = dr_i * inv_mkk_rad(i, k);

  return {local_sum_m_g * dx_o_mkk_fluid, local_sum_e_g * dx_o_mkk_fluid,
          local_sum_e_r * dx_o_mkk_rad, local_sum_m_r * dx_o_mkk_rad};
}

/**
 * @brief Custom root finder for radiation-matter coulpling.
 * This should not live here forever.
 * TODO(astrobarker): port to the new root finders infra
 */
template <typename T, typename... Args>
KOKKOS_INLINE_FUNCTION void fixed_point_radhydro(T R, double dt_a_ii,
                                                 T scratch_n, T scratch_nm1,
                                                 T scratch, Args... args) {
  static_assert(T::rank == 2, "fixed_point_radhydro expects rank-2 views.");
  static constexpr int nvars = 5;

  const int &num_modes = scratch_n.extent(0);

  auto target = [&](T u, const int k) {
    const auto [s_1_k, s_2_k, s_3_k, s_4_k] =
        compute_increment_radhydro_source(u, k, args...);
    return std::make_tuple(R(k, 1) + dt_a_ii * s_1_k, R(k, 2) + dt_a_ii * s_2_k,
                           R(k, 3) + dt_a_ii * s_3_k,
                           R(k, 4) + dt_a_ii * s_4_k);
  };

  for (int k = 0; k < num_modes; ++k) {
    for (int v = 0; v < nvars; ++v) {
      scratch_nm1(k, v) = scratch_n(k, v); // set to initial guess
    }
  }

  static const PhysicalScales scales{.velocity_scale = 1.0e7,
                                     .energy_scale = 1.0e12,
                                     .rad_energy_scale = 1.0e12,
                                     .rad_flux_scale = 1.0e20};

  static RadHydroConvergence<T> convergence_checker(
      scales, root_finders::ABSTOL, root_finders::RELTOL, num_modes);

  unsigned int n = 0;
  bool converged = false;
  while (n <= root_finders::MAX_ITERS && !converged) {
    for (int k = 0; k < num_modes; ++k) {
      const auto [xkp1_1_k, xkp1_2_k, xkp1_3_k, xkp1_4_k] =
          target(scratch_n, k);
      scratch(k, vars::cons::Velocity) = xkp1_1_k; // fluid vel
      scratch(k, vars::cons::Energy) = xkp1_2_k; // fluid energy
      scratch(k, vars::cons::RadEnergy) = xkp1_3_k; // rad energy
      scratch(k, vars::cons::RadFlux) = xkp1_4_k; // rad flux

      // --- update ---
      for (int v = 1; v < nvars; ++v) {
        scratch_nm1(k, v) = scratch_n(k, v);
        scratch_n(k, v) = scratch(k, v);
      }
    }

    converged = convergence_checker.check_convergence(scratch_n, scratch_nm1);
    ++n;
  } // while not converged
}

template <typename T, typename... Args>
KOKKOS_INLINE_FUNCTION void fixed_point_radhydro_aa(T R, double dt_a_ii,
                                                    T scratch_n, T scratch_nm1,
                                                    T scratch, Args... args) {
  using root_finders::alpha_aa, root_finders::residual;
  static_assert(T::rank == 2, "fixed_point_radhydro expects rank-2 views.");
  constexpr static int nvars = 5;

  const int num_modes = scratch_n.extent(0);

  auto target = [&](T u, const int k) {
    const auto [s_1_k, s_2_k, s_3_k, s_4_k] =
        compute_increment_radhydro_source(u, k, args...);
    return std::make_tuple(R(k, 1) + dt_a_ii * s_1_k, R(k, 2) + dt_a_ii * s_2_k,
                           R(k, 3) + dt_a_ii * s_3_k,
                           R(k, 4) + dt_a_ii * s_4_k);
  };

  // --- first fixed point iteration ---
  for (int k = 0; k < num_modes; ++k) {
    const auto [xnp1_1_k, xnp1_2_k, xnp1_3_k, xnp1_4_k] = target(scratch_n, k);
    scratch(k, vars::cons::Velocity) = xnp1_1_k;
    scratch(k, vars::cons::Energy) = xnp1_2_k;
    scratch(k, vars::cons::RadEnergy) = xnp1_3_k;
    scratch(k, vars::cons::RadFlux) = xnp1_4_k;
  }
  for (int k = 0; k < num_modes; ++k) {
    for (int v = 1; v < nvars; ++v) {
      scratch_nm1(k, v) = scratch_n(k, v);
      scratch_n(k, v) = scratch(k, v);
    }
  }

  // Set up physical scales based on your problem
  PhysicalScales scales{};
  scales.velocity_scale = 1e7; // Typical velocity (cm/s)
  scales.energy_scale = 1e12; // Typical energy density
  scales.rad_energy_scale = 1e12; // Typical radiation energy density
  scales.rad_flux_scale = 1e20; // Typical radiation flux

  static RadHydroConvergence<T> convergence_checker(
      scales, root_finders::ABSTOL, root_finders::RELTOL, num_modes);

  bool converged =
      convergence_checker.check_convergence(scratch_n, scratch_nm1);

  if (converged) {
    return;
  }

  unsigned int n = 1;
  while (n <= root_finders::MAX_ITERS && !converged) {
    for (int k = 0; k < num_modes; ++k) {
      const auto [s_1_n, s_2_n, s_3_n, s_4_n] = target(scratch_n, k);
      const auto [s_1_nm1, s_2_nm1, s_3_nm1, s_4_nm1] = target(scratch_nm1, k);

      // residuals
      const auto r_1_n = residual(s_1_n, scratch_n(k, vars::cons::Velocity));
      const auto r_2_n = residual(s_2_n, scratch_n(k, vars::cons::Energy));
      const auto r_3_n = residual(s_3_n, scratch_n(k, vars::cons::RadEnergy));
      const auto r_4_n = residual(s_4_n, scratch_n(k, vars::cons::RadFlux));
      const auto r_1_nm1 =
          residual(s_1_nm1, scratch_nm1(k, vars::cons::Velocity));
      const auto r_2_nm1 =
          residual(s_2_nm1, scratch_nm1(k, vars::cons::Energy));
      const auto r_3_nm1 =
          residual(s_3_nm1, scratch_nm1(k, vars::cons::RadEnergy));
      const auto r_4_nm1 =
          residual(s_4_nm1, scratch_nm1(k, vars::cons::RadFlux));

      // Anderson acceleration alpha
      const auto a_1 = alpha_aa(r_1_n, r_1_nm1);
      const auto a_2 = alpha_aa(r_2_n, r_2_nm1);
      const auto a_3 = alpha_aa(r_3_n, r_3_nm1);
      const auto a_4 = alpha_aa(r_4_n, r_4_nm1);

      // Anderson acceleration update
      const auto xnp1_1_k = a_1 * s_1_nm1 + (1.0 - a_1) * s_1_n;
      const auto xnp1_2_k = a_2 * s_2_nm1 + (1.0 - a_2) * s_2_n;
      const auto xnp1_3_k = a_3 * s_3_nm1 + (1.0 - a_3) * s_3_n;
      const auto xnp1_4_k = a_4 * s_4_nm1 + (1.0 - a_4) * s_4_n;

      scratch(k, vars::cons::Velocity) = xnp1_1_k; // fluid vel
      scratch(k, vars::cons::Energy) = xnp1_2_k; // fluid energy
      scratch(k, vars::cons::RadEnergy) = xnp1_3_k; // rad energy
      scratch(k, vars::cons::RadFlux) = xnp1_4_k; // rad flux

      // --- update ---
      for (int v = 1; v < nvars; ++v) {
        scratch_nm1(k, v) = scratch_n(k, v);
        scratch_n(k, v) = scratch(k, v);
      }
    }

    converged = convergence_checker.check_convergence(scratch_n, scratch_nm1);

    ++n;
  } // while not converged
}

} // namespace athelas::radiation
