#pragma once

#include "Kokkos_Macros.hpp"
#include "basic_types.hpp"
#include "bc/boundary_conditions_base.hpp"
#include "eos/eos_variant.hpp"
#include "geometry/grid.hpp"
#include "opacity/opac_variant.hpp"
#include "pgen/problem_in.hpp"
#include "radiation/rad_utilities.hpp"
#include "root_finder_opts.hpp"
#include "solvers/root_finders.hpp"
#include "state/state.hpp"

namespace athelas::radiation {

struct RadHydroSolverIonizationContent {
  AthelasArray2D<double> number_density;
  AthelasArray2D<double> ye;
  AthelasArray2D<double> ybar;
  AthelasArray2D<double> sigma1;
  AthelasArray2D<double> sigma2;
  AthelasArray2D<double> sigma3;
  AthelasArray2D<double> e_ion_corr;
  AthelasArray3D<double> bulk;
};

using bc::BoundaryConditions;

class RadHydroPackage {
 public:
  RadHydroPackage(const ProblemIn * /*pin*/, int n_stages, int nq,
                  BoundaryConditions *bcs, double cfl, int nx,
                  bool active = true);

  void update_explicit(const StageData &stage_data, const GridStructure &grid,
                       const TimeStepInfo &dt_info) const;
  void update_implicit(const StageData &stage_data, const GridStructure &grid,
                       const TimeStepInfo &dt_info) const;
  void update_implicit_iterative(const StageData &stage_data,
                                 AthelasArray3D<double> R,
                                 const GridStructure &grid,
                                 const TimeStepInfo &dt_info);

  void apply_delta(AthelasArray3D<double> lhs,
                   const TimeStepInfo &dt_info) const;

  void zero_delta() const noexcept;

  void radhydro_divergence(const StageData &stage_data,
                           const GridStructure &grid, int stage) const;

  [[nodiscard]] auto min_timestep(const StageData & /*stage_data*/,
                                  const GridStructure &grid,
                                  const TimeStepInfo & /*dt_info*/) const
      -> double;

  [[nodiscard]] auto name() const noexcept -> std::string_view;

  [[nodiscard]] auto is_active() const noexcept -> bool;

  void fill_derived(StageData &stage_data, const GridStructure &grid,
                    const TimeStepInfo &dt_info) const;

  void set_active(bool active);

  [[nodiscard]] auto get_flux_u(int stage, int i) const -> double;

  [[nodiscard]] static constexpr auto num_vars() noexcept -> int {
    return NUM_VARS_;
  }

 private:
  bool active_;

  int nx_;
  double cfl_;

  BoundaryConditions *bcs_;

  // package storage
  AthelasArray2D<double> dFlux_num_; // stores Riemann solutions
  AthelasArray2D<double> u_f_l_; // left faces
  AthelasArray2D<double> u_f_r_; // right faces
  AthelasArray2D<double> flux_u_; // Riemann velocities

  AthelasArray4D<double> delta_; // rhs delta [nstages, nx, order, nvars]
  AthelasArray4D<double> delta_im_; // rhs delta

  // iterative solver storage
  AthelasArray3D<double> scratch_k_;
  AthelasArray3D<double> scratch_km1_;
  AthelasArray3D<double> scratch_sol_;

  // constants
  static constexpr int NUM_VARS_ = 5;
};

KOKKOS_FUNCTION
template <IonizationPhysics Ionization>
auto compute_increment_radhydro_source_nodal(
    AthelasArray1D<double> uCRH, AthelasArray3D<double> uaf,
    AthelasArray3D<double> phi_fluid, AthelasArray3D<double> phi_rad,
    AthelasArray2D<double> inv_mkk_fluid, AthelasArray2D<double> inv_mkk_rad,
    const eos::EOS &eos, const Opacity &opac, AthelasArray1D<double> dx,
    AthelasArray2D<double> sqrt_gm, AthelasArray1D<double> weights,
    const RadHydroSolverIonizationContent &content, const int i, const int q)
    -> std::tuple<double, double, double, double> {
  using basis::basis_eval;
  constexpr static double c = constants::c_cgs;
  constexpr static double c2 = c * c;

  const double dr_i = dx(i);

  // Set up views
  // These are only allocated when Ionization == Active
  auto number_density = content.number_density;
  auto ye = content.ye;
  auto ybar = content.ybar;
  auto sigma1 = content.sigma1;
  auto sigma2 = content.sigma2;
  auto sigma3 = content.sigma3;
  auto e_ion_corr = content.e_ion_corr;

  double local_sum_e_r = 0.0; // radiation energy source
  double local_sum_m_r = 0.0; // radiation momentum (flux) source
  double local_sum_e_g = 0.0; // gas energy source
  double local_sum_m_g = 0.0; // gas momentum (velocity) source
  eos::EOSLambda lambda;
  const double wq = weights(q);
  const double wq_sqrtgm = wq * sqrt_gm(i, q + 1);

  // Note: basis evaluations are awkward here.
  // must be sure to use the correct basis functions.
  const double tau = uCRH(vars::cons::SpecificVolume);
  const double rho = 1.0 / tau;
  const double vel = uCRH(vars::cons::Velocity);
  const double em_t = uCRH(vars::cons::Energy);

  if constexpr (Ionization == IonizationPhysics::Active) {
    lambda.data[0] = number_density(i, q + 1);
    lambda.data[1] = ye(i, q + 1);
    lambda.data[2] = ybar(i, q + 1);
    lambda.data[3] = sigma1(i, q + 1);
    lambda.data[4] = sigma2(i, q + 1);
    lambda.data[5] = sigma3(i, q + 1);
    lambda.data[6] = e_ion_corr(i, q + 1);
    lambda.data[7] = uaf(i, q + 1, vars::aux::Tgas);
  }
  const double t_g = eos::temperature_from_density_sie(
      eos, rho, em_t - 0.5 * vel * vel, lambda.ptr());
  uaf(i, q, vars::aux::Tgas) = t_g;

  // TODO(astrobarker): composition
  // Should I move these into a lambda?
  const double X = 1.0;
  const double Y = 1.0;
  const double Z = 1.0;

  const double kappa_r = rosseland_mean(opac, rho, t_g, X, Y, Z, lambda.ptr());
  const double kappa_p = planck_mean(opac, rho, t_g, X, Y, Z, lambda.ptr());

  const double E_r = uCRH(vars::cons::RadEnergy) * rho;
  const double F_r = uCRH(vars::cons::RadFlux) * rho;
  const double P_r = compute_closure(E_r, F_r);

  // 4 force
  const auto [G0, G] =
      radiation_four_force(rho, vel, t_g, kappa_r, kappa_p, E_r, F_r, P_r);

  const double source_e_r = -c * G0;
  const double source_m_r = -c2 * G;
  const double source_e_g = c * G0;
  const double source_m_g = G;

  local_sum_e_r += wq_sqrtgm * source_e_r;
  local_sum_m_r += wq_sqrtgm * source_m_r;
  local_sum_e_g += wq_sqrtgm * source_e_g;
  local_sum_m_g += wq_sqrtgm * source_m_g;

  // \Delta x / M_kk
  const double dx_o_mkk_fluid = dr_i * inv_mkk_fluid(i, q);
  const double dx_o_mkk_rad = dr_i * inv_mkk_rad(i, q);

  return {local_sum_m_g * dx_o_mkk_fluid, local_sum_e_g * dx_o_mkk_fluid,
          local_sum_e_r * dx_o_mkk_rad, local_sum_m_r * dx_o_mkk_rad};
}

// This is duplicate of above but used differently, in the root finder
// The code needs some refactoring in order to get rid of this version.
KOKKOS_FUNCTION
template <IonizationPhysics Ionization>
auto compute_increment_radhydro_source(
    AthelasArray2D<double> uCRH, const int k, AthelasArray3D<double> uaf,
    AthelasArray3D<double> phi_fluid, AthelasArray3D<double> phi_rad,
    AthelasArray2D<double> inv_mkk_fluid, AthelasArray2D<double> inv_mkk_rad,
    const eos::EOS &eos, const Opacity &opac, AthelasArray1D<double> dx,
    AthelasArray1D<double> weights,
    const RadHydroSolverIonizationContent &content, const int i)
    -> std::tuple<double, double, double, double> {
  using basis::basis_eval;
  constexpr static double c = constants::c_cgs;
  constexpr static double c2 = c * c;

  static const int nNodes = static_cast<int>(weights.size());
  const double dr_i = dx(i);

  // Set up views
  // These are only allocated when Ionization == Active
  auto number_density = content.number_density;
  auto ye = content.ye;
  auto ybar = content.ybar;
  auto sigma1 = content.sigma1;
  auto sigma2 = content.sigma2;
  auto sigma3 = content.sigma3;
  auto e_ion_corr = content.e_ion_corr;
  auto bulk = content.bulk;

  double local_sum_e_r = 0.0; // radiation energy source
  double local_sum_m_r = 0.0; // radiation momentum (flux) source
  double local_sum_e_g = 0.0; // gas energy source
  double local_sum_m_g = 0.0; // gas momentum (velocity) source
  eos::EOSLambda lambda;
  for (int q = 0; q < nNodes; ++q) {
    const int qp1 = q + 1;
    const double wq = weights(q);
    const double phi_rad_kq = phi_rad(i, qp1, k);
    const double phi_fluid_kq = phi_fluid(i, qp1, k);

    // Note: basis evaluations are awkward here.
    // must be sure to use the correct basis functions.
    const double tau =
        basis_eval(phi_fluid, uCRH, i, vars::cons::SpecificVolume, qp1);
    const double rho = 1.0 / tau;
    const double vel =
        basis_eval(phi_fluid, uCRH, i, vars::cons::Velocity, qp1);
    const double em_t = basis_eval(phi_fluid, uCRH, i, vars::cons::Energy, qp1);

    // Should I move these into a lambda?
    static const int x_idx = 0;
    static const int z_idx = 2;
    double X = 0.0;
    double Z = 0.0;
    if constexpr (Ionization == IonizationPhysics::Active) {
      lambda.data[0] = number_density(i, q);
      lambda.data[1] = ye(i, q);
      lambda.data[2] = ybar(i, q);
      lambda.data[3] = sigma1(i, q);
      lambda.data[4] = sigma2(i, q);
      lambda.data[5] = sigma3(i, q);
      lambda.data[6] = e_ion_corr(i, q);
      lambda.data[7] = uaf(i, q, vars::aux::Tgas);

      X = bulk(i, qp1, x_idx);
      Z = bulk(i, qp1, z_idx);
    }
    const double t_g = eos::temperature_from_density_sie(
        eos, rho, em_t - 0.5 * vel * vel, lambda.ptr());

    const double kappa_r = opac.rosseland_mean(rho, t_g, X, Z, lambda.ptr());
    const double kappa_p = opac.planck_mean(rho, t_g, X, Z, lambda.ptr());

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
template <IonizationPhysics Ionization, typename T, typename... Args>
KOKKOS_INLINE_FUNCTION void fixed_point_radhydro(T R, double dt_a_ii,
                                                 T scratch_n, T scratch_nm1,
                                                 T scratch, Args... args) {
  static_assert(T::rank == 1, "fixed_point_radhydro expects rank-2 views.");
  static constexpr int nvars = 5;

  auto target = [&](T u) {
    const auto [s_1, s_2, s_3, s_4] =
        compute_increment_radhydro_source_nodal<Ionization>(u, args...);
    return std::make_tuple(R(1) + dt_a_ii * s_1, R(2) + dt_a_ii * s_2,
                           R(3) + dt_a_ii * s_3, R(4) + dt_a_ii * s_4);
  };

  for (int v = 0; v < nvars; ++v) {
    scratch_nm1(v) = scratch_n(v); // set to initial guess
  }

  static const PhysicalScales scales{.velocity_scale = 1.0e7,
                                     .energy_scale = 1.0e12,
                                     .rad_energy_scale = 1.0e12,
                                     .rad_flux_scale = 1.0e20};

  static RadHydroConvergence<T> convergence_checker(
      scales, root_finders::ABSTOL, root_finders::RELTOL, 1);

  unsigned int n = 0;
  bool converged = false;
  while (n <= root_finders::MAX_ITERS && !converged) {
    const auto [xkp1_1, xkp1_2, xkp1_3, xkp1_4] = target(scratch_n);
    scratch(vars::cons::Velocity) = xkp1_1; // fluid vel
    scratch(vars::cons::Energy) = xkp1_2; // fluid energy
    scratch(vars::cons::RadEnergy) = xkp1_3; // rad energy
    scratch(vars::cons::RadFlux) = xkp1_4; // rad flux

    // --- update ---
    for (int v = 1; v < nvars; ++v) {
      scratch_nm1(v) = scratch_n(v);
      scratch_n(v) = scratch(v);
    }

    converged = convergence_checker.check_convergence(scratch_n, scratch_nm1);
    ++n;
  } // while not converged
}

template <IonizationPhysics Ionization, typename T, typename... Args>
KOKKOS_INLINE_FUNCTION void
fixed_point_radhydro_nodal(T R, double dt_a_ii, T scratch_n, T scratch_nm1,
                           T scratch, Args... args) {
  using root_finders::alpha_aa, root_finders::residual;
  // static_assert(T::rank == 2, "fixed_point_radhydro expects rank-2 views.");
  constexpr static int nvars = 5;

  auto target = [&](T u) {
    const auto [s1, s2, s3, s4] =
        compute_increment_radhydro_source_nodal<Ionization>(u, args...);

    const double omega = 1.0;
    const double g1 = R(1) + dt_a_ii * s1;
    const double g2 = R(2) + dt_a_ii * s2;
    const double g3 = R(3) + dt_a_ii * s3;
    const double g4 = R(4) + dt_a_ii * s4;

    return std::make_tuple((1.0 - omega) * u(vars::cons::Velocity) + omega * g1,
                           (1.0 - omega) * u(vars::cons::Energy) + omega * g2,
                           (1.0 - omega) * u(vars::cons::RadEnergy) +
                               omega * g3,
                           (1.0 - omega) * u(vars::cons::RadFlux) + omega * g4);
  };

  for (int v = 0; v < nvars; ++v) {
    scratch_n(v) = scratch(v);
  }

  // --- first fixed point iteration ---
  const double beta = 1.0;
  // const double beta = 1.0;
  const auto [xnp1_1_k, xnp1_2_k, xnp1_3_k, xnp1_4_k] = target(scratch_n);
  scratch(vars::cons::Velocity) =
      (1.0 - beta) * scratch(vars::cons::Velocity) + beta * xnp1_1_k;
  scratch(vars::cons::Energy) =
      (1.0 - beta) * scratch(vars::cons::Energy) + beta * xnp1_2_k;
  scratch(vars::cons::RadEnergy) =
      (1.0 - beta) * scratch(vars::cons::RadEnergy) + beta * xnp1_3_k;
  scratch(vars::cons::RadFlux) =
      (1.0 - beta) * scratch(vars::cons::RadFlux) + beta * xnp1_4_k;

  for (int v = 1; v < nvars; ++v) {
    scratch_nm1(v) = scratch_n(v);
    scratch_n(v) = scratch(v);
  }

  // Set up physical scales based on your problem
  PhysicalScales scales{};
  scales.velocity_scale = 1e7; // Typical velocity (cm/s)
  scales.energy_scale = 1e17; // Typical energy density
  scales.rad_energy_scale = 1e12; // Typical radiation energy density
  scales.rad_flux_scale = 1e20; // Typical radiation flux

  static RadHydroConvergence<T> convergence_checker(
      scales, root_finders::ABSTOL, root_finders::RELTOL, 1);

  bool converged =
      convergence_checker.check_convergence(scratch_n, scratch_nm1);

  if (converged) {
    return;
  }

  double r_1_n = 0.0;
  double r_2_n = 0.0;
  double r_3_n = 0.0;
  double r_4_n = 0.0;
  double r_1_nm1 = 0.0;
  double r_2_nm1 = 0.0;
  double r_3_nm1 = 0.0;
  double r_4_nm1 = 0.0;

  double s_1_nm1 = scratch_nm1(1);
  double s_2_nm1 = scratch_nm1(2);
  double s_3_nm1 = scratch_nm1(3);
  double s_4_nm1 = scratch_nm1(4);

  unsigned int n = 1;
  double omega = 1.0;
  while (n < root_finders::MAX_ITERS && !converged) {
    // TODO(astrobarker): we can cut down on evals.
    auto [s_1_n, s_2_n, s_3_n, s_4_n] = target(scratch_n);
    // const auto [s_1_nm1, s_2_nm1, s_3_nm1, s_4_nm1] = target(scratch_nm1, k);
    // // don't repeat

    // residuals
    r_1_n = residual(s_1_n, scratch_n(vars::cons::Velocity));
    r_2_n = residual(s_2_n, scratch_n(vars::cons::Energy));
    r_3_n = residual(s_3_n, scratch_n(vars::cons::RadEnergy));
    r_4_n = residual(s_4_n, scratch_n(vars::cons::RadFlux));
    const auto dr_1 = r_1_n - r_1_nm1;
    const auto dr_2 = r_2_n - r_2_nm1;
    const auto dr_3 = r_3_n - r_3_nm1;
    const auto dr_4 = r_4_n - r_4_nm1;
    // Anderson acceleration alpha
    double num = r_1_n * dr_1 + r_2_n * dr_2 + r_3_n * dr_3 + r_4_n * dr_4;

    double den = dr_1 * dr_1 + dr_2 * dr_2 + dr_3 * dr_3 + dr_4 * dr_4;
    double alpha = (den > 0.0) ? num / den : 0.0;

    // Anderson acceleration update

    scratch(vars::cons::Velocity) = alpha * s_1_nm1 + (1.0 - alpha) * s_1_n;
    scratch(vars::cons::Energy) = alpha * s_2_nm1 + (1.0 - alpha) * s_2_n;
    scratch(vars::cons::RadEnergy) = alpha * s_3_nm1 + (1.0 - alpha) * s_3_n;
    scratch(vars::cons::RadFlux) = alpha * s_4_nm1 + (1.0 - alpha) * s_4_n;

    // --- update ---
    for (int v = 1; v < nvars; ++v) {
      scratch_nm1(v) = scratch_n(v);
      scratch_n(v) = scratch(v);
    }

    r_1_nm1 = r_1_n;
    r_2_nm1 = r_2_n;
    r_3_nm1 = r_3_n;
    r_4_nm1 = r_4_n;

    s_1_nm1 = s_1_n;
    s_2_nm1 = s_2_n;
    s_3_nm1 = s_3_n;
    s_4_nm1 = s_4_n;

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
