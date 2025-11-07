#pragma once

namespace athelas::eos {

/**
 * @class EosBase
 * @brief Base class for equations of state using the Curiously Recurring
 *Template Pattern (CRTP)
 *
 * @details This header defines the EosBase template class that serves as the
 * foundation for all equation of state implementations in the codebase. It uses
 * the CRTP to provide a common interface while allowing derived classes to
 * implement specific EOS behaviors.
 *
 *          The class provides the following:
 *          - pressure_from_density_temperature
 *          - pressure_from_conserved
 *          - sound_speed_from_conserved
 *          - temperature_from_conserved
 *          - sie_from_density_pressure
 *          - gamma
 *
 *          These interfaces are implemented for all EOS
 */
template <class EOS>
class EosBase {
 public:
  auto pressure_from_density_temperature(const double rho, const double temp,
                                         const double *const lambda) const
      -> double {
    return static_cast<EOS const *>(this)->pressure_from_density_temperature(
        rho, temp, lambda);
  }
  auto temperature_from_density_sie(const double rho, const double sie,
                                    const double *const lambda) const
      -> double {
    return static_cast<EOS const *>(this)->temperature_from_density_sie(
        rho, sie, lambda);
  }
  auto sound_speed_from_density_temperature_pressure(
      const double rho, const double temp, const double pressure,
      const double *const lambda) const -> double {
    return static_cast<EOS const *>(this)
        ->sound_speed_from_density_temperature_pressure(rho, temp, pressure,
                                                        lambda);
  }
  auto pressure_from_conserved(const double tau, const double V,
                               const double EmT,
                               const double *const lambda) const -> double {
    return static_cast<EOS const *>(this)->pressure_from_conserved(tau, V, EmT,
                                                                   lambda);
  }
  auto sound_speed_from_conserved(const double tau, const double V,
                                  const double EmT,
                                  const double *const lambda) const -> double {
    return static_cast<EOS const *>(this)->sound_speed_from_conserved(
        tau, V, EmT, lambda);
  }
  auto temperature_from_conserved(const double tau, const double V,
                                  const double EmT,
                                  const double *const lambda) const -> double {
    return static_cast<EOS const *>(this)->temperature_from_conserved(
        tau, V, EmT, lambda);
  }
  auto sie_from_density_pressure(const double rho, const double pressure,
                                 const double *const lambda) const -> double {
    return static_cast<EOS const *>(this)->sie_from_density_pressure(
        rho, pressure, lambda);
  }
  auto gamma1(const double tau, const double V, const double EmT,
              const double *const lambda) const -> double {
    return static_cast<EOS const *>(this)->gamma1(tau, V, EmT, lambda);
  }
  [[nodiscard]] auto gamma1() const -> double {
    return static_cast<EOS const *>(this)->gamma();
  }
};

} // namespace athelas::eos
