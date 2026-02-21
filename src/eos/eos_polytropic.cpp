#include <cmath>

#include "eos/eos.hpp"
#include "utils/constants.hpp"

namespace athelas::eos {

[[nodiscard]] auto Polytropic::pressure_from_density_temperature(
    const double rho, const double /*temp*/,
    const double *const /*lambda*/) const -> double {
  return k_ * std::pow(rho, 1.0 + 1.0 / n_);
}

[[nodiscard]] auto
Polytropic::temperature_from_density_sie(const double rho, const double /*sie*/,
                                         const double *const /*lambda*/) const
    -> double {
  const double p = std::pow(rho, 1.0 + 1.0 / n_);
  const double mu =
      1.0 + constants::m_e / constants::m_p; // TODO(astrobarker) generalize
  return p * mu * constants::m_p / (rho * constants::k_B);
}

[[nodiscard]] auto Polytropic::temperature_from_density_pressure(
    const double rho, const double pressure,
    const double *const /*lambda*/) const -> double {
  const double mu =
      1.0 + constants::m_e / constants::m_p; // TODO(astrobarker) generalize
  return pressure * mu * constants::m_p / (rho * constants::k_B);
}

[[nodiscard]] auto Polytropic::sound_speed_from_density_temperature_pressure(
    const double rho, const double /*temp*/, const double /*pressure*/,
    const double *const /*lambda*/) const -> double {
  return std::sqrt((1.0 + 1.0 / n_) * k_ * std::pow(rho, 1.0 / n_));
}

[[nodiscard]] auto Polytropic::pressure_from_conserved(
    const double tau, const double /*V*/, const double /*EmT*/,
    const double *const /*lambda*/) const -> double {
  return k_ * std::pow((1.0 / tau), 1.0 + 1.0 / n_);
}

[[nodiscard]] auto Polytropic::sound_speed_from_conserved(
    const double tau, const double /*V*/, const double /*EmT*/,
    const double *const /*lambda*/) const -> double {
  return std::sqrt((1.0 + 1.0 / n_) * k_ * std::pow((1.0 / tau), 1.0 / n_));
}

// Note: For a polytropic EOS, temperature is not a thermodynamic variable.
// This returns the *diagnostic* temperature assuming an ideal-gas law.
[[nodiscard]] auto Polytropic::temperature_from_conserved(
    const double tau, const double V, const double E,
    const double *const lambda) const -> double {
  const double p = pressure_from_conserved(tau, V, E, lambda);
  const double mu =
      1.0 + constants::m_e / constants::m_p; // TODO(astrobarker) generalize
  return tau * p * mu * constants::m_p / constants::k_B;
}

[[nodiscard]] auto
Polytropic::sie_from_density_pressure(const double rho, const double pressure,
                                      const double *const /*lambda*/) const
    -> double {
  return pressure / (gamma1() - 1) / rho;
}

[[nodiscard]] auto Polytropic::sie_from_density_temperature(
    const double rho, const double temperature,
    const double *const lambda) const -> double {
  const double pressure =
      pressure_from_density_temperature(rho, temperature, lambda);
  return pressure / (gamma1() - 1) / rho;
}

[[nodiscard]] auto
Polytropic::gamma1(const double /*tau*/, const double /*V*/,
                   const double /*EmT*/,
                   const double *const /*lambda*/) const noexcept -> double {
  return 1.0 + 1.0 / n_;
}

[[nodiscard]] auto Polytropic::gamma1() const noexcept -> double {
  return 1.0 + 1.0 / n_;
}

[[nodiscard]] auto Polytropic::cv_from_density_temperature(
    const double /*rho*/, const double /*temp*/,
    const double *const /*lambda*/) const -> double {
  // Not really true to a polytrope, but hopefully this never gets called.
  static constexpr double mu =
      1.0 + constants::m_e / constants::m_p; // TODO(astrobarker) generalize
  constexpr double k_B_over_m_p = constants::k_B / constants::m_p;
  return k_B_over_m_p / ((1.0 / n_) * mu);
}

} // namespace athelas::eos
