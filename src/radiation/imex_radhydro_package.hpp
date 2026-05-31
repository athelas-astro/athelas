#pragma once

#include "basic_types.hpp"
#include "bc/boundary_conditions_base.hpp"
#include "geometry/mesh.hpp"
#include "interface/state.hpp"
#include "pgen/problem_in.hpp"
#include "radiation/rad_utilities.hpp"

namespace athelas::radiation {

void radiation_source_implicit(const StageData &stage_data,
                               AthelasArray3D<double> R,
                               AthelasArray4D<double> delta, const Mesh &mesh,
                               const TimeStepInfo &dt_info);

using bc::BoundaryConditions;

class RadHydroPackage {
 public:
  RadHydroPackage(const ProblemIn * /*pin*/, int n_stages, int nq,
                  BoundaryConditions *bcs, double cfl, int nx,
                  bool active = true);

  void update_explicit(const StageData &stage_data,
                       const TimeStepInfo &dt_info) const;
  void update_implicit(const StageData &stage_data, AthelasArray3D<double> R,
                       const TimeStepInfo &dt_info);

  void apply_delta(AthelasArray3D<double> lhs,
                   const TimeStepInfo &dt_info) const;

  void zero_delta() const noexcept;

  void radhydro_divergence(const StageData &stage_data, const Mesh &mesh,
                           int stage) const;

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

  double cfl_;

  BoundaryConditions *bcs_;

  // package storage
  AthelasArray2D<double> dFlux_num_; // stores Riemann solutions
  AthelasArray2D<double> u_f_l_; // left faces
  AthelasArray2D<double> u_f_r_; // right faces

  AthelasArray4D<double> delta_; // rhs delta [nstages, nx, order, nvars]
  AthelasArray4D<double> delta_im_; // rhs delta

  // constants
  static constexpr int NUM_VARS_ = 5;
};

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
    src_in.erad = etot - e;
    src_in.frad = c2 * (m_tot - v);

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
      // Conservation: trial rad vars track the trial (e, v).
      trial_in.erad = etot - e_trial;
      trial_in.frad = c2 * (m_tot - v_trial);
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

    const bool energy_converged = (std::abs(f_e) <= 1.0e-9 * etot);
    const bool momentum_converged = (std::abs(f_v) <= 1.0e-9 * vscale);
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
    // Reconstruct radiation variables from conservation (source-only solve
    // preserves etot = e + Er_spec and m_tot = v + Fr_spec/c²).
    const double Er = (etot - e);
    const double Fr = c2 * (m_tot - v);

    src_in.e = e;
    src_in.v = v;
    // Feed the reconstructed specific rad variables into the now-explicit
    // compute_rad_sources interface.
    src_in.erad = Er;
    src_in.frad = Fr;

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

    const double det = J11 * J22 - J12 * J21;
    const double inv_det = 1.0 / det;

    const double delta_e = -(J22 * f_e - J12 * f_v) * inv_det;
    const double delta_v = -(J11 * f_v - J21 * f_e) * inv_det;

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
      // Conservation: trial rad vars track the trial (e, v).
      trial_in.erad = etot - e_trial;
      trial_in.frad = c2 * (m_tot - v_trial);
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

      if (rel_update < 1e-12) {
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
