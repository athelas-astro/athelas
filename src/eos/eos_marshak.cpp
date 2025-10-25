#include <cmath>

#include "eos/eos.hpp"
#include "utils/constants.hpp"

namespace athelas::eos {

[[nodiscard]] auto Marshak::pressure_from_conserved(
    const double tau, const double V, const double EmT,
    const double *const /*lambda*/) const -> double {
  const double Em = EmT - (0.5 * V * V);
  const double Ev = Em / tau;
  return (gamma_ - 1.0) * Ev;
}

[[nodiscard]] auto Marshak::sound_speed_from_conserved(
    const double /*tau*/, const double V, const double EmT,
    const double *const /*lambda*/) const -> double {
  const double Em = EmT - (0.5 * V * V);
  return std::sqrt(gamma_ * (gamma_ - 1.0) * Em);
}

[[nodiscard]] auto Marshak::temperature_from_conserved(
    const double tau, const double V, const double E,
    const double *const /*lambda*/) const -> double {
  const double sie = E - 0.5 * V * V;
  const double ev = sie / tau;
  return std::pow(ev / constants::a, 0.25);
}

// Is this right? Likely doesn't matter..
[[nodiscard]] auto
Marshak::sie_from_density_pressure(const double rho, const double pressure,
                                   const double *const /*lambda*/) const
    -> double {
  const double ev = pressure / (gamma_ - 1.0);
  return ev / rho;
}

[[nodiscard]] auto
Marshak::gamma1(const double /*tau*/, const double /*V*/, const double /*EmT*/,
                const double *const /*lambda*/) const noexcept -> double {
  return gamma_;
}

[[nodiscard]] auto Marshak::gamma1() const noexcept -> double { return gamma_; }

} // namespace athelas::eos
