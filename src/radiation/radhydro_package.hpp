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

/*
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
*/
struct RadHydroSolverIonizationContent {
  double number_density{};
  double ye{};
  double ybar{};
  double sigma1{};
  double sigma2{};
  double sigma3{};
  double e_ion_corr{};
  double X{};
  double Z{};
};

using bc::BoundaryConditions;

class RadHydroPackage {
 public:
  RadHydroPackage(const ProblemIn * /*pin*/, int n_stages, int nq,
                  BoundaryConditions *bcs, double cfl, int nx,
                  bool active = true);

  void update_explicit(const StageData &stage_data, const GridStructure &grid,
                       const TimeStepInfo &dt_info) const;
  void update_implicit(const StageData &stage_data, AthelasArray3D<double> R,
                       const GridStructure &grid, const TimeStepInfo &dt_info);

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

  // constants
  static constexpr int NUM_VARS_ = 5;
};

KOKKOS_FUNCTION
template <IonizationPhysics Ionization, typename T>
auto compute_increment_radhydro_source(
    T uCRH, AthelasArray3D<double> uaf, AthelasArray3D<double> phi_fluid,
    AthelasArray3D<double> phi_rad, AthelasArray2D<double> inv_mkk_fluid,
    AthelasArray2D<double> inv_mkk_rad, const eos::EOS &eos,
    const Opacity &opac, AthelasArray1D<double> dx,
    AthelasArray2D<double> sqrt_gm, AthelasArray1D<double> weights,
    const RadHydroSolverIonizationContent &content, const int i, const int q)
    -> std::tuple<double, double, double, double> {
  using basis::basis_eval;
  constexpr static double c = constants::c_cgs;
  constexpr static double c2 = c * c;

  const double qp1 = q + 1;
  const double dr_i = dx(i);

  auto number_density = content.number_density;
  auto ye = content.ye;
  auto ybar = content.ybar;
  auto sigma1 = content.sigma1;
  auto sigma2 = content.sigma2;
  auto sigma3 = content.sigma3;
  auto e_ion_corr = content.e_ion_corr;
  auto X = content.X;
  auto Z = content.Z;

  eos::EOSLambda lambda;
  const double wq = weights(q);
  const double wq_sqrtgm = wq * sqrt_gm(i, qp1);

  const double tau = uCRH[vars::cons::SpecificVolume];
  const double rho = 1.0 / tau;
  const double vel = uCRH[vars::cons::Velocity];
  const double em_t = uCRH[vars::cons::Energy];
  const double sie = em_t - 0.5 * vel * vel;
  const double E_r = uCRH[vars::cons::RadEnergy] * rho;
  const double F_r = uCRH[vars::cons::RadFlux] * rho;
  const double P_r = compute_closure(E_r, F_r);

  // Should I move these into a lambda?
  if constexpr (Ionization == IonizationPhysics::Active) {
    lambda.data[0] = number_density;
    lambda.data[1] = ye;
    lambda.data[2] = ybar;
    lambda.data[3] = sigma1;
    lambda.data[4] = sigma2;
    lambda.data[5] = sigma3;
    lambda.data[6] = e_ion_corr;
    lambda.data[7] = uaf(i, qp1, vars::aux::Tgas);
  }

  const double t_g =
      eos::temperature_from_density_sie(eos, rho, sie, lambda.ptr());
  uaf(i, qp1, vars::aux::Tgas) = t_g;

  const double kappa_r = opac.rosseland_mean(rho, t_g, X, Z, lambda.ptr());
  const double kappa_p = opac.planck_mean(rho, t_g, X, Z, lambda.ptr());

  // 4 force
  const auto [cG0, G] =
      radiation_four_force(rho, vel, t_g, kappa_r, kappa_p, E_r, F_r, P_r);

  const double source_e_r = -cG0 * wq_sqrtgm;
  const double source_m_r = -c2 * G * wq_sqrtgm;
  const double source_e_g = cG0 * wq_sqrtgm;
  const double source_m_g = G * wq_sqrtgm;

  // \Delta x / M_kk
  const double dx_o_mkk_fluid = dr_i * inv_mkk_fluid(i, q);
  const double dx_o_mkk_rad = dr_i * inv_mkk_rad(i, q);

  return {source_m_g * dx_o_mkk_fluid, source_e_g * dx_o_mkk_fluid,
          source_e_r * dx_o_mkk_rad, source_m_r * dx_o_mkk_rad};
}

/**
 * @brief Custom root finder for radiation-matter coulpling.
 * This should not live here forever.
 * TODO(astrobarker): port to the new root finders infra
 */
template <IonizationPhysics Ionization, typename T, typename G,
          typename... Args>
KOKKOS_INLINE_FUNCTION void fixed_point_radhydro(T R, double dt_a_ii,
                                                 double emin, G scratch,
                                                 G scratch_nm1, Args... args) {
  static_assert(T::rank == 1, "fixed_point_radhydro expects rank-1 views.");
  static constexpr int nvars = 5;

  auto target = [&](G u) {
    const auto [s1, s2, s3, s4] =
        compute_increment_radhydro_source<Ionization>(u, args...);

    const double omega = 1.0;
    const double g1 = R(1) + dt_a_ii * s1;
    const double g2 = R(2) + dt_a_ii * s2;
    const double g3 = R(3) + dt_a_ii * s3;
    const double g4 = R(4) + dt_a_ii * s4;

    return std::make_tuple((1.0 - omega) * u[vars::cons::Velocity] + omega * g1,
                           (1.0 - omega) * u[vars::cons::Energy] + omega * g2,
                           (1.0 - omega) * u[vars::cons::RadEnergy] +
                               omega * g3,
                           (1.0 - omega) * u[vars::cons::RadFlux] + omega * g4);
  };

  for (int v = 0; v < nvars; ++v) {
    scratch_nm1[v] = scratch[v]; // set to initial guess
  }

  static const PhysicalScales scales{.velocity_scale = 1.0e7,
                                     .energy_scale = 1.0e12,
                                     .rad_energy_scale = 1.0e12,
                                     .rad_flux_scale = 1.0e20};

  static RadHydroConvergence<G> convergence_checker(
      scales, root_finders::ABSTOL, root_finders::RELTOL, 1);

  unsigned int n = 0;
  bool converged = false;
  while (n <= root_finders::MAX_ITERS && !converged) {
    const auto [xkp1_1, xkp1_2, xkp1_3, xkp1_4] = target(scratch);

    double du1 = xkp1_1 - scratch[vars::cons::Velocity];
    double du2 = xkp1_2 - scratch[vars::cons::Energy];
    double du3 = xkp1_3 - scratch[vars::cons::RadEnergy];
    double du4 = xkp1_4 - scratch[vars::cons::RadFlux];

    double lam = 1.0;
    double v_trial = scratch[vars::cons::Velocity] + du1;
    double e_trial =
        scratch[vars::cons::Energy] + du2 - 0.5 * v_trial * v_trial;
    double er_trial = scratch[vars::cons::RadEnergy] + du3;

    while (er_trial < 0.0 ||
           std::abs(scratch[4] + du4) > constants::c_cgs * scratch[3] ||
           e_trial <= emin) {
      lam *= 0.5;
      du1 = lam * (xkp1_1 - scratch[vars::cons::Velocity]);
      du2 = lam * (xkp1_2 - scratch[vars::cons::Energy]);
      du3 = lam * (xkp1_3 - scratch[vars::cons::RadEnergy]);
      du4 = lam * (xkp1_4 - scratch[vars::cons::RadFlux]);
      v_trial = scratch[vars::cons::Velocity] + du1;
      e_trial = scratch[vars::cons::Energy] + du2 - 0.5 * v_trial * v_trial;
      er_trial = scratch[vars::cons::RadEnergy] + du3;
    }

    scratch[vars::cons::Velocity] += du1; // fluid vel
    scratch[vars::cons::Energy] += du2; // fluid energy
    scratch[vars::cons::RadEnergy] += du3; // rad energy
    scratch[vars::cons::RadFlux] += du4; // rad flux

    converged = convergence_checker.check_convergence(scratch, scratch_nm1);

    // --- update ---
    for (int v = 1; v < nvars; ++v) {
      scratch_nm1[v] = scratch[v];
    }

    ++n;
  } // while not converged
}

template <IonizationPhysics Ionization, typename T, typename G,
          typename... Args>
KOKKOS_INLINE_FUNCTION void
fixed_point_radhydro_nodal(T R, double dt_a_ii, double emin, G scratch,
                           G scratch_nm1, Args... args) {
  using root_finders::alpha_aa, root_finders::residual;
  // static_assert(T::rank == 2, "fixed_point_radhydro expects rank-2 views.");
  constexpr static int nvars = 5;

  auto target = [&](G u) {
    const auto [s1, s2, s3, s4] =
        compute_increment_radhydro_source<Ionization>(u, args...);

    const double omega = 1.0;
    const double g1 = R(1) + dt_a_ii * s1;
    const double g2 = R(2) + dt_a_ii * s2;
    const double g3 = R(3) + dt_a_ii * s3;
    const double g4 = R(4) + dt_a_ii * s4;

    return std::make_tuple((1.0 - omega) * u[vars::cons::Velocity] + omega * g1,
                           (1.0 - omega) * u[vars::cons::Energy] + omega * g2,
                           (1.0 - omega) * u[vars::cons::RadEnergy] +
                               omega * g3,
                           (1.0 - omega) * u[vars::cons::RadFlux] + omega * g4);
  };

  for (int v = 0; v < nvars; ++v) {
    scratch_nm1[v] = scratch[v];
  }

  // --- first fixed point iteration ---
  const double beta = 0.1;
  const auto [xnp1_1_k, xnp1_2_k, xnp1_3_k, xnp1_4_k] = target(scratch);

  double du1 = (1.0 - beta) * scratch[vars::cons::Velocity] + beta * xnp1_1_k -
               scratch[vars::cons::Velocity];
  double du2 = (1.0 - beta) * scratch[vars::cons::Energy] + beta * xnp1_2_k -
               scratch[vars::cons::Energy];
  double du3 = -du2;
  double du4 = -constants::c_cgs * constants::c_cgs * du1;
  scratch[vars::cons::Velocity] += du1;
  scratch[vars::cons::Energy] += du2;
  scratch[vars::cons::RadEnergy] += du3;
  scratch[vars::cons::RadFlux] += du4;

  // Set up physical scales based on your problem
  PhysicalScales scales{};
  scales.velocity_scale = 1e7; // Typical velocity (cm/s)
  scales.energy_scale = 1e17; // Typical energy density
  scales.rad_energy_scale = 1e12; // Typical radiation energy density
  scales.rad_flux_scale = 1e20; // Typical radiation flux

  static RadHydroConvergence<G> convergence_checker(
      scales, root_finders::ABSTOL, root_finders::RELTOL, 1);

  bool converged = convergence_checker.check_convergence(scratch, scratch_nm1);

  if (converged) {
    return;
  }

  for (int v = 1; v < nvars; ++v) {
    scratch_nm1[v] = scratch[v];
  }

  double r_1_n = 0.0;
  double r_2_n = 0.0;
  double r_3_n = 0.0;
  double r_4_n = 0.0;
  double r_1_nm1 = 0.0;
  double r_2_nm1 = 0.0;
  double r_3_nm1 = 0.0;
  double r_4_nm1 = 0.0;

  double s_1_nm1 = scratch_nm1[1];
  double s_2_nm1 = scratch_nm1[2];

  unsigned int n = 1;
  while (n < root_finders::MAX_ITERS && !converged) {
    auto [s_1_n, s_2_n, s_3_n, s_4_n] = target(scratch);

    // residuals
    r_1_n = residual(s_1_n, scratch[vars::cons::Velocity]);
    r_2_n = residual(s_2_n, scratch[vars::cons::Energy]);
    r_3_n = residual(s_3_n, scratch[vars::cons::RadEnergy]);
    r_4_n = residual(s_4_n, scratch[vars::cons::RadFlux]);
    const auto dr_1 = r_1_n - r_1_nm1;
    const auto dr_2 = r_2_n - r_2_nm1;
    const auto dr_3 = r_3_n - r_3_nm1;
    const auto dr_4 = r_4_n - r_4_nm1;

    // Anderson acceleration alpha
    double num = r_1_n * dr_1 + r_2_n * dr_2 + r_3_n * dr_3 + r_4_n * dr_4;

    double den = dr_1 * dr_1 + dr_2 * dr_2 + dr_3 * dr_3 + dr_4 * dr_4;
    double alpha = (den > 0.0) ? num / den : 0.0;

    // Anderson acceleration update

    double s1 = alpha * s_1_nm1 + (1.0 - alpha) * s_1_n;
    double s2 = alpha * s_2_nm1 + (1.0 - alpha) * s_2_n;
    double du1 = s1 - scratch[vars::cons::Velocity];
    double du2 = s2 - scratch[vars::cons::Energy];
    double du3 = -du2;
    double du4 = -constants::c_cgs * constants::c_cgs * du1;

    double lam = 1.0;
    double v_trial = scratch[vars::cons::Velocity] + du1;
    double e_trial =
        scratch[vars::cons::Energy] + du2 - 0.5 * v_trial * v_trial;
    double er_trial = scratch[vars::cons::RadEnergy] + du3;
    while (er_trial < 0.0 ||
           std::abs(scratch[4] + du4) > constants::c_cgs * er_trial ||
           e_trial <= emin) {
      lam *= 0.5;
      du1 = lam * (s1 - scratch[vars::cons::Velocity]);
      du2 = lam * (s2 - scratch[vars::cons::Energy]);
      du3 = -du2;
      du4 = -constants::c_cgs * constants::c_cgs * du1;
      v_trial = scratch[vars::cons::Velocity] + du1;
      e_trial = scratch[vars::cons::Energy] + du2 - 0.5 * v_trial * v_trial;
      er_trial = scratch[vars::cons::RadEnergy] + du3;
    }
    scratch[vars::cons::Velocity] += du1;
    scratch[vars::cons::Energy] += du2;
    scratch[vars::cons::RadEnergy] += du3;
    scratch[vars::cons::RadFlux] += du4;

    converged = convergence_checker.check_convergence(scratch, scratch_nm1);

    // --- update ---
    for (int v = 1; v < nvars; ++v) {
      scratch_nm1[v] = scratch[v];
    }

    r_1_nm1 = r_1_n;
    r_2_nm1 = r_2_n;
    r_3_nm1 = r_3_n;
    r_4_nm1 = r_4_n;

    s_1_nm1 = s_1_n;
    s_2_nm1 = s_2_n;

    ++n;
  } // while not converged
}

struct RadSourceInputs {
  double rho, e, v;
  double etot, m_tot; // conserved, fixed -- derive Er, Fr internally
  double X, Z;
  double dt_a_ii, dg_term;
  const eos::EOS *eos;
  const Opacity *opac;
};

template <OpacityType Opac>
KOKKOS_FUNCTION auto dkappa_dT(const Opacity &opac, const double rho,
                               const double T, const double X, const double Z)
    -> double {
  // Use a log-space perturbation since the table is uniform in logT
  // 1e-4 is ~0.01% of a decade,
  constexpr double h_log = 1e-4;
  const double logT = std::log10(T);
  const double T_plus = std::pow(10.0, logT + h_log);

  double k_plus = 0.0;
  double k_base = 0.0;

  if constexpr (Opac == OpacityType::Planck) {
    k_plus = opac.planck_mean(rho, T_plus, X, Z, nullptr);
    k_base = opac.planck_mean(rho, T, X, Z, nullptr);
  } else if constexpr (Opac == OpacityType::Rosseland) {
    k_plus = opac.rosseland_mean(rho, T_plus, X, Z, nullptr);
    k_base = opac.rosseland_mean(rho, T, X, Z, nullptr);
  }

  // 1. Compute the slope in log-log or linear-log space: d(kappa)/d(log10T)
  const double dk_dlogT = (k_plus - k_base) / h_log;

  // 2. d(kappa)/dT = d(kappa)/d(log10T) * d(log10T)/dT
  // d(log10T)/dT = 1 / (T * ln(10))
  constexpr double inv_ln10 = 1.0 / std::numbers::log10e;

  return dk_dlogT * (inv_ln10 / T);
}

KOKKOS_INLINE_FUNCTION
auto compute_rad_sources(const RadSourceInputs &in, double *lambda)
    -> std::tuple<double, double> {
  constexpr double c = constants::c_cgs;
  constexpr double inv_c = 1.0 / c;
  constexpr double c2 = c * c;
  const double Er = (in.etot - in.e) * in.rho;
  const double Fr = c2 * (in.m_tot - in.v) * in.rho;

  const double temperature = eos::temperature_from_density_sie(
      *in.eos, in.rho, in.e - 0.5 * in.v * in.v, lambda);

  const double at3 = constants::a * temperature * temperature * temperature;
  const double at4 = at3 * temperature;

  const double kappa_p =
      in.opac->planck_mean(in.rho, temperature, in.X, in.Z, lambda);
  const double kappa_r =
      in.opac->rosseland_mean(in.rho, temperature, in.X, in.Z, lambda);

  const double Pr = compute_closure(Er, Fr);

  const double se = in.rho *
                    (-c * kappa_p * (at4 - Er) - kappa_r * in.v * Fr * inv_c) *
                    in.dg_term;
  const double sv =
      in.rho * inv_c *
      (kappa_r * Fr - kappa_p * in.v * at4 - kappa_r * in.v * Pr) * in.dg_term;
  return {se, sv};
}

template <DiffScheme Scheme = DiffScheme::Forward>
KOKKOS_INLINE_FUNCTION auto finite_diff_source(const RadSourceInputs &in,
                                               double *lambda)
    -> std::tuple<double, double, double, double> {
  constexpr double h_base = (Scheme == DiffScheme::Central) ? 1.0e-6 : 1.0e-8;
  constexpr double tol = 1.0e-14;

  const double etot = in.etot;

  double dsede;
  double dsedv;
  double dsvde;
  double dsvdv;

  if constexpr (Scheme == DiffScheme::Forward) {
    // --- Forward / Backward Scheme ---
    // In the forward scheme we check that e + h < etot.
    // If we cross that bounds we switch to a backwards difference.
    const auto [se0, sv0] = compute_rad_sources(in, lambda);

    // Energy
    {
      const double h_e = h_base * std::abs(in.e) + tol;
      const double side = (in.e + h_e > etot) ? -1.0 : 1.0;

      auto in_p = in;
      in_p.e += side * h_e;
      const auto [sep, svp] = compute_rad_sources(in_p, lambda);

      dsede = (sep - se0) / (side * h_e);
      dsvde = (svp - sv0) / (side * h_e);
    }
    // Velocity
    {
      const double h_v = 100.0 * h_base * std::abs(in.v) + tol;
      auto in_p = in;
      in_p.v += h_v;
      const auto [sep, svp] = compute_rad_sources(in_p, lambda);

      dsedv = (sep - se0) / h_v;
      dsvdv = (svp - sv0) / h_v;
    }

  } else {
    // --- Central Difference ---
    // Energy
    {
      const double h_e = h_base * std::abs(in.e) + tol;
      auto in_p = in;
      auto in_m = in;

      in_p.e += h_e;
      in_m.e -= h_e;

      const auto [sep, svp] = compute_rad_sources(in_p, lambda);
      const auto [sem, svm] = compute_rad_sources(in_m, lambda);

      const double inv_2h = 0.5 / h_e;
      dsede = (sep - sem) * inv_2h;
      dsvde = (svp - svm) * inv_2h;
    }

    // Velocity
    {
      const double h_v = h_base * std::abs(in.v) + tol;
      auto in_p = in;
      auto in_m = in;

      in_p.v += h_v;
      in_m.v -= h_v;

      const auto [sep, svp] = compute_rad_sources(in_p, lambda);
      const auto [sem, svm] = compute_rad_sources(in_m, lambda);

      const double inv_2h = 0.5 / h_v;
      dsedv = (sep - sem) * inv_2h;
      dsvdv = (svp - svm) * inv_2h;
    }
  }

  return {dsede, dsedv, dsvde, dsvdv};
}

template <IonizationPhysics Ionization, typename T, typename G>
KOKKOS_INLINE_FUNCTION void
newton_radhydro_fd(const double dt_a_ii, const double emin, T ustar, T uaf,
                   const RadHydroSolverIonizationContent &content, G &scratch,
                   const eos::EOS &eos, const Opacity &opac,
                   eos::EOSLambda lambda, const double dg_term) {
  constexpr double c = constants::c_cgs;
  constexpr double c2 = c * c;
  constexpr double inv_c2 = 1.0 / c2;

  // line search params
  constexpr double alpha = 1.0e-4;
  constexpr int max_linesearch = 16;

  const double vstar = ustar(vars::cons::Velocity);
  const double rho = 1.0 / ustar(vars::cons::SpecificVolume);
  const double e_star = ustar(vars::cons::Energy);
  const double er_star = ustar(vars::cons::RadEnergy);
  const double fr_star = ustar(vars::cons::RadFlux);
  const double etot = e_star + er_star;
  const double m_tot = vstar + fr_star * inv_c2; // "total specific momentum"
  const double vscale = std::sqrt(2.0 * etot);

  const auto number_density = content.number_density;
  const auto ye = content.ye;
  const auto ybar = content.ybar;
  const auto sigma1 = content.sigma1;
  const auto sigma2 = content.sigma2;
  const auto sigma3 = content.sigma3;
  const auto e_ion_corr = content.e_ion_corr;
  const auto X = content.X;
  const auto Z = content.Z;

  // Initial values
  double e = e_star;
  double v = vstar;

  bool converged = false;
  std::size_t n = 0;

  RadSourceInputs src_in;
  src_in.rho = rho;
  src_in.etot = etot;
  src_in.m_tot = m_tot;
  src_in.X = X;
  src_in.Z = Z;
  src_in.dg_term = dg_term;
  src_in.eos = &eos;
  src_in.opac = &opac;
  while (n < root_finders::MAX_ITERS && !converged) {
    // Update dependent eos / opacity quantities
    if constexpr (Ionization == IonizationPhysics::Active) {
      lambda.data[0] = number_density;
      lambda.data[1] = ye;
      lambda.data[2] = ybar;
      lambda.data[3] = sigma1;
      lambda.data[4] = sigma2;
      lambda.data[5] = sigma3;
      lambda.data[6] = e_ion_corr;
      lambda.data[7] = uaf(vars::aux::Tgas);
    }

    // Build inputs struct for source evaluation
    src_in.e = e;
    src_in.v = v;

    // Sources and residuals
    const auto [se, sv] = compute_rad_sources(src_in, lambda.ptr());
    const double f_e = e - e_star - dt_a_ii * se;
    const double f_v = v - vstar - dt_a_ii * sv;

    // Jacobian via finite differences
    const auto [dsede, dsedv, dsvde, dsvdv] =
        finite_diff_source<DiffScheme::Forward>(src_in, lambda.ptr());

    const double J11 = 1.0 - dt_a_ii * dsede;
    const double J12 = -dt_a_ii * dsedv;
    const double J21 = -dt_a_ii * dsvde;
    const double J22 = 1.0 - dt_a_ii * dsvdv;

    // 1. Get Row Scales
    const double r1 = std::max({std::abs(J11), std::abs(J12), 1e-14});
    const double r2 = std::max({std::abs(J21), std::abs(J22), 1e-14});

    // 2. Scale rows (Balance the equations)
    const double a1 = J11 / r1;
    const double b1 = J12 / r1;
    const double c1 = J21 / r2;
    const double d1 = J22 / r2;

    const double rhs1 = f_e / r1;
    const double rhs2 = f_v / r2;

    // 3. Solve the balanced system
    const double det = a1 * d1 - b1 * c1;
    const double inv_det = 1.0 / det;

    const double delta_e = -(d1 * rhs1 - b1 * rhs2) * inv_det;
    const double delta_v = -(a1 * rhs2 - c1 * rhs1) * inv_det;

    // Line search
    double lam = 1.0;
    double v_trial = v + lam * delta_v;
    double e_trial = e + lam * delta_e;
    double Er_trial = etot - e_trial;
    double Fr_trial = c2 * (m_tot - v_trial);
    double sie_trial = e_trial - 0.5 * v_trial * v_trial;

    // merit function: residual norm
    const double F0 = f_e * f_e + f_v * f_v;

    for (int ls = 0; ls < max_linesearch; ++ls) {
      e_trial = e + lam * delta_e;
      v_trial = v + lam * delta_v;
      Er_trial = etot - e_trial;
      Fr_trial = c2 * (m_tot - v_trial);
      sie_trial = e_trial - 0.5 * v_trial * v_trial;

      // realizability first
      if (std::abs(Fr_trial) >= c * Er_trial || Er_trial <= 0.0 ||
          sie_trial <= emin) {
        lam *= 0.5;
        continue;
      }

      RadSourceInputs trial_in = src_in;
      trial_in.e = e_trial;
      trial_in.v = v_trial;
      const auto [se_t, sv_t] = compute_rad_sources(trial_in, lambda.ptr());
      const double fe_t = e_trial - e_star - dt_a_ii * se_t;
      const double fv_t = v_trial - vstar - dt_a_ii * sv_t;
      const double F_trial = fe_t * fe_t + fv_t * fv_t;

      // 3. Classic Armoji line search criteria
      if (F_trial < (1.0 - 2.0 * alpha * lam) * F0) {
        break;
      }

      // 4. Stagnation Check: If we are already deep in the noise floor,
      // and the line search is failing, the Jacobian is likely inaccurate.
      // We check if the update is becoming smaller than machine precision
      // relative to the variables.
      const double rel_update =
          lam * (std::abs(delta_e) / (std::abs(e) + 1e-14) +
                 std::abs(delta_v) / (std::abs(v) + 1e-14));

      if (rel_update < 1e-14) {
        // Update is too small to change the floats; further backtracking is
        // useless.
        break;
      }

      lam *= 0.5;
    }
    e += lam * delta_e;
    v += lam * delta_v;

    const bool energy_converged = (std::abs(f_e) <= 1.0e-8 * etot);
    const bool momentum_converged = (std::abs(f_v) <= 1.0e-8 * vscale);
    converged = energy_converged && momentum_converged;

    ++n;
  } // while not converged

  // Update conserved variables
  scratch[vars::cons::Velocity] = v;
  scratch[vars::cons::Energy] = std::max(e, emin);
  scratch[vars::cons::RadEnergy] = etot - std::max(e, emin);
  scratch[vars::cons::RadFlux] = c2 * (m_tot - v);
}

template <IonizationPhysics Ionization, typename T, typename G>
KOKKOS_INLINE_FUNCTION void
newton_radhydro(const double dt_a_ii, const double emin, T ustar, T uaf,
                const RadHydroSolverIonizationContent &content, G &scratch,
                const eos::EOS &eos, const Opacity &opac, eos::EOSLambda lambda,
                const double dg_term) {
  constexpr double c = constants::c_cgs;
  constexpr double inv_c = 1.0 / c;
  constexpr double c2 = c * c;
  constexpr double inv_c2 = 1.0 / c2;

  // line search params
  constexpr double alpha = 1.0e-4;
  constexpr int max_linesearch = 22;

  const double vstar = ustar(vars::cons::Velocity);
  const double rho = 1.0 / ustar(vars::cons::SpecificVolume);
  const double c_rho = constants::c_cgs * rho;
  const double e_star = ustar(vars::cons::Energy);
  const double er_star = ustar(vars::cons::RadEnergy);
  const double fr_star = ustar(vars::cons::RadFlux);
  const double etot = e_star + er_star;
  const double m_tot = vstar + fr_star * inv_c2; // "total specific momentum"
  const double vscale = std::sqrt(2.0 * etot);

  auto number_density = content.number_density;
  auto ye = content.ye;
  auto ybar = content.ybar;
  auto sigma1 = content.sigma1;
  auto sigma2 = content.sigma2;
  auto sigma3 = content.sigma3;
  auto e_ion_corr = content.e_ion_corr;
  auto X = content.X;
  auto Z = content.Z;

  // Initial values
  double e = e_star;
  double v = vstar;

  RadSourceInputs src_in;
  src_in.rho = rho;
  src_in.etot = etot;
  src_in.m_tot = m_tot;
  src_in.X = X;
  src_in.Z = Z;
  src_in.dg_term = dg_term;
  src_in.eos = &eos;
  src_in.opac = &opac;

  bool converged = false;
  std::size_t n = 0;

  while (n < root_finders::MAX_ITERS && !converged) {
    // Reconstruct radiation variables from conservation
    const double Er = (etot - e);
    const double Fr = c2 * (m_tot - v);

    src_in.e = e;
    src_in.v = v;

    // Update dependent eos / opacity quantities
    if constexpr (Ionization == IonizationPhysics::Active) {
      lambda.data[0] = number_density;
      lambda.data[1] = ye;
      lambda.data[2] = ybar;
      lambda.data[3] = sigma1;
      lambda.data[4] = sigma2;
      lambda.data[5] = sigma3;
      lambda.data[6] = e_ion_corr;
      lambda.data[7] = uaf(vars::aux::Tgas);
    }

    const double temperature = eos::temperature_from_density_sie(
        eos, rho, e - 0.5 * v * v, lambda.ptr());
    uaf(vars::aux::Tgas) = temperature;
    const double cv =
        eos::cv_from_density_temperature(eos, rho, temperature, lambda.ptr());
    const double inv_cv = 1.0 / cv;

    const double at3 = constants::a * temperature * temperature * temperature;
    const double at4 = at3 * temperature;

    const double kappa_p =
        opac.planck_mean(rho, temperature, X, Z, lambda.ptr());
    const double kappa_r =
        opac.rosseland_mean(rho, temperature, X, Z, lambda.ptr());

    const double Pr = compute_closure(rho * Er, rho * Fr);
    const double f = flux_factor(Er, Fr);
    const double chi = eddington_factor(f);
    const double chi_prime = eddington_factor_prime(f);

    // Finite difference for opacity derivatives
    const double dkappa_p_dT =
        dkappa_dT<OpacityType::Planck>(opac, rho, temperature, X, Z);
    const double dkappa_r_dT =
        dkappa_dT<OpacityType::Rosseland>(opac, rho, temperature, X, Z);

    // Sources for energy (se) and velocity (sv)
    // se = c G^0 = -c rho kappa_p (aT^4 - Er) - rho kappa_r Fr v/c
    // sv = rho kappa_r Fr / c - rho kappa_p a T^4 v / c - rho kappa_r Pr v / c;
    const double se =
        rho *
        (-c * kappa_p * (at4 - rho * Er) - kappa_r * v * (Fr * rho) * inv_c) *
        dg_term;
    const double sv =
        rho * inv_c *
        (kappa_r * (Fr * rho) - kappa_p * v * at4 - kappa_r * v * Pr) * dg_term;

    // Residuals
    const double f_e = e - e_star - dt_a_ii * se;
    const double f_v = v - vstar - dt_a_ii * sv;

    // Form various derivatives
    const double v2 = v * v;

    const double dsede =
        c_rho * inv_cv *
            (-4.0 * at3 * kappa_p - at4 * dkappa_p_dT + rho * Er * dkappa_p_dT -
             v * rho * Fr * inv_c2 * dkappa_r_dT) *
            dg_term -
        c * rho * rho * kappa_p * dg_term;
    const double dsedv =
        (c_rho * inv_cv *
             (v * at4 * dkappa_p_dT + 4.0 * at3 * v * kappa_p -
              v * (rho * Er) * dkappa_p_dT +
              (rho * Fr) * v2 * inv_c2 * dkappa_r_dT) -
         (rho * kappa_r * (rho * Fr) * inv_c) + (rho * rho * kappa_r * v * c)) *
        dg_term;

    const double dprde = rho * (-chi + f * chi_prime);
    const double dprdv = -c_rho * chi_prime;
    const double dsvde =
        (rho * inv_c * inv_cv *
             ((rho * Fr) * dkappa_r_dT - at4 * v * dkappa_p_dT -
              4.0 * at3 * kappa_p * v - v * Pr * dkappa_r_dT) -
         (rho * v * kappa_r * inv_c * dprde)) *
        dg_term;

    const double dsvdv =
        (rho * inv_c * inv_cv *
             (-v * (rho * Fr) * dkappa_r_dT + at4 * v2 * dkappa_p_dT +
              4.0 * at3 * v2 * kappa_p + Pr * v2 * dkappa_r_dT) -
         (c * rho * rho * kappa_r) - (rho * at4 * kappa_p * inv_c) -
         (rho * kappa_r * Pr * inv_c) - (rho * kappa_r * v * dprdv * inv_c)) *
        dg_term;

    const double J11 = 1.0 - dt_a_ii * dsede;
    const double J12 = -dt_a_ii * dsedv;
    const double J21 = -dt_a_ii * dsvde;
    const double J22 = 1.0 - dt_a_ii * dsvdv;

    // Get Row Scales
    const double r1 = std::max({std::abs(J11), std::abs(J12), 1e-14});
    const double r2 = std::max({std::abs(J21), std::abs(J22), 1e-14});

    // Scale rows (Balance the equations)
    const double a1 = J11 / r1;
    const double b1 = J12 / r1;
    const double c1 = J21 / r2;
    const double d1 = J22 / r2;

    const double rhs1 = f_e / r1;
    const double rhs2 = f_v / r2;

    // Solve the balanced system
    const double det = a1 * d1 - b1 * c1;

    const double inv_det = 1.0 / det;

    const double delta_e = -(d1 * rhs1 - b1 * rhs2) * inv_det;
    const double delta_v = -(a1 * rhs2 - c1 * rhs1) * inv_det;

    // Line search
    double lam = 1.0;
    double v_trial = v + lam * delta_v;
    double e_trial = e + lam * delta_e;
    double Er_trial = etot - e_trial;
    double Fr_trial = c2 * (m_tot - v_trial);
    double sie_trial = e_trial - 0.5 * v_trial * v_trial;

    // merit function: residual norm
    const double F0 = f_e * f_e + f_v * f_v;

    for (int ls = 0; ls < max_linesearch; ++ls) {
      e_trial = e + lam * delta_e;
      v_trial = v + lam * delta_v;
      Er_trial = etot - e_trial;
      Fr_trial = c2 * (m_tot - v_trial);
      sie_trial = e_trial - 0.5 * v_trial * v_trial;

      // realizability first
      if (std::abs(Fr_trial) >= c * Er_trial || Er_trial <= 0.0 ||
          sie_trial <= emin) {
        lam *= 0.5;
        continue;
      }

      RadSourceInputs trial_in = src_in;
      trial_in.e = e_trial;
      trial_in.v = v_trial;
      const auto [se_t, sv_t] = compute_rad_sources(trial_in, lambda.ptr());
      const double fe_t = e_trial - e_star - dt_a_ii * se_t;
      const double fv_t = v_trial - vstar - dt_a_ii * sv_t;
      const double F_trial = fe_t * fe_t + fv_t * fv_t;

      // 3. Classic Armoji line search criteria
      if (F_trial < (1.0 - 2.0 * alpha * lam) * F0) {
        break;
      }

      // 4. Stagnation Check: If we are already deep in the noise floor,
      // and the line search is failing, the Jacobian is likely inaccurate.
      // We check if the update is becoming smaller than machine precision
      // relative to the variables.
      const double rel_update =
          lam * (std::abs(delta_e) / (std::abs(e) + 1e-14) +
                 std::abs(delta_v) / (std::abs(v) + 1e-14));

      if (rel_update < 1e-14) {
        // Update is too small to change the floats; further backtracking is
        // useless.
        break;
      }

      lam *= 0.5;
    }
    e += lam * delta_e;
    v += lam * delta_v;

    const bool energy_converged = (std::abs(f_e) <= 1.0e-8 * etot) ||
                                  (std::abs(delta_e) <= 1.0e-12 * etot);
    const bool momentum_converged = (std::abs(f_v) <= 1.0e-8 * vscale) ||
                                    (std::abs(delta_v) <= 1.0e-12 * vscale);
    converged = energy_converged && momentum_converged;

    ++n;
  }

  // Update conserved variables
  scratch[vars::cons::Velocity] = v;
  scratch[vars::cons::Energy] = e;
  scratch[vars::cons::RadEnergy] = etot - e;
  scratch[vars::cons::RadFlux] = c2 * (m_tot - v);
}

} // namespace athelas::radiation
