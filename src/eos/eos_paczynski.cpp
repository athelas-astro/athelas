#include <cmath>

#include "eos/eos.hpp"
#include "eos_variant.hpp"
#include "solvers/root_finders.hpp"
#include "utils/constants.hpp"
namespace athelas::eos {

using root_finders::RootFinder, root_finders::NewtonAlgorithm,
    root_finders::RelativeError, root_finders::HybridError,
    root_finders::ToleranceConfig;

/**
 * @brief Paczynski pressure from density and temperature
 * NOTE:: Lambda contents:
 * 0: N (for ion pressure)
 * 1: ye
 * 2: ybar (mean ionization state)
 */
[[nodiscard]] auto Paczynski::pressure_from_density_temperature(
    const double rho, const double temp, const double *const lambda) -> double {
  const double N = lambda[0];
  const double ye = lambda[1];
  const double ybar = lambda[2];

  const double pion = p_ion(rho, temp, N);
  const double pednr = p_ednr(rho, ye);
  const double pedr = p_edr(rho, ye);
  const double ped = p_ed(pednr, pedr);
  const double pend = p_end(rho, temp, ybar, N);
  const double pe = p_e(pend, ped);

  return pe + pion;
}

/**
 * @brief Paczynski temperature from density and specific internal energy
 * NOTE:: Lambda contents:
 * 7: temperature_guess
 */
[[nodiscard]] auto
Paczynski::temperature_from_density_sie(const double rho, const double sie,
                                        const double *const lambda) const
    -> double {
  const double temperature_guess = lambda[7];
  auto target = [&sie](const double temperature, const double rho,
                       const double *const lambda) {
    return sie_from_density_temperature(rho, temperature, lambda) - sie;
  };
  const double res =
      root_finder_.solve(target, 100.0, 1.0e12, temperature_guess, rho, lambda);
  return res;
}

/**
 * @brief Paczynski temperature from density and pressure
 * NOTE:: Lambda contents:
 * 7: temperature_guess
 */
[[nodiscard]] auto Paczynski::temperature_from_density_pressure(
    const double rho, const double pressure, const double *const lambda) const
    -> double {
  const double temperature_guess = lambda[7];
  auto target = [&pressure](const double temperature, const double rho,
                            const double *const lambda) {
    return pressure_from_density_temperature(rho, temperature, lambda) -
           pressure;
  };
  const double res =
      root_finder_.solve(target, 1.0, 5.0e12, temperature_guess, rho, lambda);
  return res;
}

[[nodiscard]] auto Paczynski::sound_speed_from_density_temperature_pressure(
    const double rho, const double temp, const double pressure,
    const double *const lambda) const -> double {
  const auto gamma_1 =
      gamma1_from_density_temperature_pressure(rho, temp, pressure, lambda);
  return std::sqrt(pressure * gamma_1 / rho);
}

/**
 * @brief Paczynski pressure from conserved variables.
 * NOTE:: Lambda contents:
 * 0: N (for ion pressure)
 * 1: ye
 * 2: ybar (mean ionization state)
 * 3: sigma1
 * 4: sigma2
 * 5: sigma3
 * 6: e_ioncorr (ionization corrcetion to internal energy)
 * 7: temperature_guess
 */
[[nodiscard]] auto
Paczynski::pressure_from_conserved(const double tau, const double V,
                                   const double EmT,
                                   const double *const lambda) const -> double {
  const double N = lambda[0];
  const double ye = lambda[1];
  const double ybar = lambda[2];
  const double rho = 1.0 / tau;

  // TODO: use an internal temperature call here instead of from_conserved
  const double temperature = temperature_from_conserved(tau, V, EmT, lambda);

  const double pion = p_ion(rho, temperature, N);

  const double pednr = p_ednr(rho, ye);
  const double pedr = p_edr(rho, ye);
  const double ped = p_ed(pednr, pedr);
  const double pend = p_end(rho, temperature, ybar, N);
  const double pe = p_e(pend, ped);

  return pe + pion;
}

[[nodiscard]] auto Paczynski::sound_speed_from_conserved(
    const double tau, const double V, const double EmT,
    const double *const lambda) const -> double {
  const auto gamma_1 = gamma1(tau, V, EmT, lambda);
  const double pressure = pressure_from_conserved(tau, V, EmT, lambda);
  return std::sqrt(pressure * gamma_1 * tau);
}

/**
 * @brief invert Pacynski eos to get temperature given rho, sie
 */
[[nodiscard]] auto Paczynski::temperature_from_conserved(
    const double tau, const double V, const double E,
    const double *const lambda) const -> double {
  const double temperature_guess = lambda[7];
  const double dT = temperature_guess / 2.0;
  const double sie = E - 0.5 * V * V;
  const double rho = 1.0 / tau;
  auto target = [sie](double temperature, double rho,
                      const double *const lambda) {
    return sie_from_density_temperature(rho, temperature, lambda) - sie;
  };
  return root_finder_.solve(target, temperature_guess - dT,
                            temperature_guess + dT, temperature_guess, rho,
                            lambda);
}

/**
 * @brief Paczynski specific internal energy from density and pressure
 *
 * Invert the eos to get the temperature from rho, p, and then compute sie
 */
[[nodiscard]] auto
Paczynski::sie_from_density_pressure(const double rho, const double pressure,
                                     const double *const lambda) const
    -> double {
  auto target = [pressure](double temperature, double rho,
                           const double *const lambda) {
    return pressure_from_rho_temperature(temperature, rho, lambda) - pressure;
  };

  const double temperature_guess = lambda[7];
  const double dT = 0.9 * temperature_guess;
  const double temperature = root_finder_.solve(target, temperature_guess - dT,
                                                temperature_guess + 5.0 * dT,
                                                temperature_guess, rho, lambda);
  return sie_from_density_temperature(rho, temperature, lambda);
}

/**
 * @brief Paczynski internal gamma1 from density temperature pressure
 */
[[nodiscard]] auto Paczynski::gamma1_from_density_temperature_pressure(
    double rho, double temp, double pressure, const double *lambda) const
    -> double {
  const double N = lambda[0];
  const double ye = lambda[1];
  const double ybar = lambda[2];
  const double sigma1 = lambda[3];

  const double pednr = p_ednr(rho, ye);
  const double pedr = p_edr(rho, ye);
  const double ped = p_ed(pednr, pedr);
  const double pend = p_end(rho, temp, ybar, N);
  const double f = degeneracy_factor(ped, pednr, pedr);

  const double chi_rho =
      (rho / pressure) * dp_drho(temp, rho, ybar, pend, ped, f, N, sigma1);
  const double chi_T = (temp / pressure) * dp_dt(temp, rho, lambda);
  const double cv = dsie_dt(temp, rho, lambda);

  return (chi_T * chi_T * pressure) / (cv * rho * temp) + chi_rho;
}

// TODO(astrobarker): make gamma1_from_T .....
// This call is slow.
[[nodiscard]] auto Paczynski::gamma1(const double tau, const double V,
                                     const double EmT,
                                     const double *const lambda) const
    -> double {
  const double N = lambda[0];
  const double ye = lambda[1];
  const double ybar = lambda[2];
  const double sigma1 = lambda[3];
  const double rho = 1.0 / tau;

  // TODO: use an internal temperature call here instead of from_conserved
  const double temperature = temperature_from_conserved(tau, V, EmT, lambda);

  const double pressure = pressure_from_conserved(tau, V, EmT, lambda);

  const double pednr = p_ednr(rho, ye);
  const double pedr = p_edr(rho, ye);
  const double ped = p_ed(pednr, pedr);
  const double pend = p_end(rho, temperature, ybar, N);
  const double f = degeneracy_factor(ped, pednr, pedr);

  const double chi_rho = (rho / pressure) * dp_drho(temperature, rho, ybar,
                                                    pend, ped, f, N, sigma1);
  const double chi_T =
      (temperature / pressure) * dp_dt(temperature, rho, lambda);
  const double cv = dsie_dt(temperature, rho, lambda);

  return (chi_T * chi_T * pressure) / (cv * rho * temperature) + chi_rho;
}

[[nodiscard]] auto Paczynski::gamma1() const -> double {
  throw_athelas_error("No arg gamma not supported for Paczynski eos!");
  return NAN;
}

[[nodiscard]] auto Paczynski::p_end(const double rho, const double T,
                                    const double ybar, const double N)
    -> double {
  return ybar * N * rho * constants::k_B * T;
}

[[nodiscard]] auto Paczynski::p_ednr(const double rho, const double ye)
    -> double {
  static constexpr double FIVE_THIRDS = 5.0 / 3.0;
  static constexpr double term = 9.991e12;
  return term * std::pow(rho * ye, FIVE_THIRDS);
}

[[nodiscard]] auto Paczynski::p_edr(const double rho, const double ye)
    -> double {
  static constexpr double FOUR_THIRDS = 4.0 / 3.0;
  static constexpr double term = 1.231e15;
  return term * std::pow(rho * ye, FOUR_THIRDS);
}

[[nodiscard]] auto Paczynski::p_ed(const double p_ednr, const double p_edr)
    -> double {
  return 1.0 / std::sqrt((1.0 / (p_ednr * p_ednr)) + (1.0 / (p_edr * p_edr)));
}

[[nodiscard]] auto Paczynski::p_e(double p_end, double p_ed) -> double {
  return std::sqrt(p_end * p_end + p_ed * p_ed);
}

[[nodiscard]] auto Paczynski::p_ion(const double rho, const double T,
                                    const double N) -> double {
  return N * rho * constants::k_B * T;
}

[[nodiscard]] auto Paczynski::degeneracy_factor(const double p_ed,
                                                const double p_ednr,
                                                const double p_edr) -> double {
  static constexpr double ONE_THIRD = 1.0 / 3.0;
  const double p_ed2 = p_ed * p_ed;
  return ONE_THIRD *
         (5.0 * p_ed2 / (p_ednr * p_ednr) + 4.0 * p_ed2 / (p_edr * p_edr));
}

/**
 * @brief internal eos function to get pressure from rho, temperature
 */
[[nodiscard]] auto
Paczynski::pressure_from_rho_temperature(const double temperature,
                                         const double rho,
                                         const double *const lambda) -> double {
  const double N = lambda[0];
  const double ye = lambda[1];
  const double ybar = lambda[2];
  const double pion = p_ion(rho, temperature, N);

  const double pednr = p_ednr(rho, ye);
  const double pedr = p_edr(rho, ye);
  const double ped = p_ed(pednr, pedr);
  const double pend = p_end(rho, temperature, ybar, N);
  const double pe = p_e(pend, ped);

  return pe + pion;
}

/**
 * @brief internal (to the eos) specific internal energy function
 */
[[nodiscard]] auto
Paczynski::sie_from_density_temperature(const double rho,
                                        const double temperature,
                                        const double *const lambda) -> double {
  static constexpr double THREE_HALVES = 1.5;
  const double N = lambda[0];
  const double ye = lambda[1];
  const double ybar = lambda[2];
  const double e_ion_corr = lambda[6];
  const double pednr = p_ednr(rho, ye);
  const double pedr = p_edr(rho, ye);
  const double ped = p_ed(pednr, pedr);
  const double pend = p_end(rho, temperature, ybar, N);
  const double pe = p_e(pend, ped);
  const double f = degeneracy_factor(ped, pednr, pedr);

  return THREE_HALVES * N * constants::k_B * temperature +
         pe / (rho * (f - 1.0)) + e_ion_corr;
}

/*
 * @brief Paczynski dsie_dT
 */
[[nodiscard]] auto Paczynski::dsie_dt(const double T, const double rho,
                                      const double *const lambda) -> double {
  static constexpr double THREE_HALVES = 3.0 / 2.0;
  static constexpr double kb = constants::k_B;
  const double inv_T = 1.0 / T;
  const double inv_kT = inv_T / (kb);
  const double N = lambda[0];
  const double ye = lambda[1];
  const double ybar = lambda[2];
  const double sigma1 = lambda[3];
  const double sigma2 = lambda[4];
  const double sigma3 = lambda[5];
  const double inv_sigma1_plus_ybar = 1.0 / (sigma1 + ybar);
  const double pednr = p_ednr(rho, ye);
  const double pedr = p_edr(rho, ye);
  const double ped = p_ed(pednr, pedr);
  const double pend = p_end(rho, T, ybar, N);
  const double pe = p_e(pend, ped);
  const double f = degeneracy_factor(ped, pednr, pedr);

  return THREE_HALVES * N * kb +
         (pend * pend * inv_T) / (pe * rho * (f - 1)) *
             (1.0 + (1.0 * inv_sigma1_plus_ybar) *
                        (THREE_HALVES * sigma1 + sigma2 * inv_kT)) +
         (N * inv_T) * ((sigma2 * inv_sigma1_plus_ybar) *
                            (THREE_HALVES * ybar - sigma2 * inv_kT) +
                        sigma3 * inv_kT);
}

/*
 * @brief Paczynski dp_dT
 */
[[nodiscard]] auto Paczynski::dp_dt(const double T, const double rho,
                                    const double *const lambda) -> double {
  static constexpr double THREE_HALVES = 3.0 / 2.0;
  static constexpr double kb = constants::k_B;

  const double N = lambda[0];
  const double ye = lambda[1];
  const double ybar = lambda[2];
  const double sigma1 = lambda[3];
  const double sigma2 = lambda[4];

  const double pednr = p_ednr(rho, ye);
  const double pedr = p_edr(rho, ye);
  const double ped = p_ed(pednr, pedr);
  const double pend = p_end(rho, T, ybar, N);
  const double pe = p_e(pend, ped);

  const double kT = kb * T;
  const double sigma1_plus_ybar = sigma1 + ybar;

  return N * kb * rho + (pend * pend) / (pe * T) *
                            (1.0 + (1.0 / sigma1_plus_ybar) *
                                       (THREE_HALVES * sigma1 + sigma2 / kT));
}

// TODO(astrobarker): pass in pe instead of computing
/*
 * @brief Paczynski dp_drho
 * TODO(astrobarker): pass in pe instead of computing
 */
[[nodiscard]] auto Paczynski::dp_drho(const double T, const double rho,
                                      const double ybar, const double pend,
                                      const double ped, const double f,
                                      const double N, const double sigma1)
    -> double {
  static constexpr double kb = constants::k_B;
  const double kT = kb * T;
  const double sigma1_plus_ybar = sigma1 + ybar;
  const double pend2 = pend * pend;
  const double pe = p_e(pend, ped);

  return N * kT - (pend2 * sigma1) / (pe * rho * sigma1_plus_ybar) +
         (1.0 / (pe * rho)) * (pend2 + f * ped * ped);
}

[[nodiscard]] auto Paczynski::eps_min(double rho, const double *lambda)
    -> double {
  const double ye = lambda[1];
  const double e_ionization = lambda[6];

  const double pednr = p_ednr(rho, ye);
  const double pedr = p_edr(rho, ye);
  const double ped = p_ed(pednr, pedr);
  const double f = degeneracy_factor(ped, pednr, pedr);
  return ped / (rho * (f - 1.0)) + e_ionization;
}

} // namespace athelas::eos
