#pragma once
/**
 * @brief Provides variant-based dispatch for equations of state
 *
 * @details This header implements a type-safe way to handle different EOS
 *          models at runtime using std::variant. It provides visitor functions
 *          that dispatch to the appropriate model's implementation.
 */

#include <variant>

#include "eos/eos.hpp"
#include "pgen/problem_in.hpp"
#include "utils/error.hpp"

namespace athelas::eos {

using EOS = std::variant<IdealGas, Marshak, Paczynski, Polytropic>;

KOKKOS_INLINE_FUNCTION auto
temperature_from_density_sie(const EOS *const eos, const double rho,
                             const double sie, const double *const lambda)
    -> double {
  return std::visit(
      [&rho, &sie, &lambda](auto &eos) {
        return eos.temperature_from_density_sie(rho, sie, lambda);
      },
      *eos);
}

KOKKOS_INLINE_FUNCTION auto
pressure_from_density_temperature(const EOS *const eos, const double rho,
                                  const double temp, const double *const lambda)
    -> double {
  return std::visit(
      [&rho, &temp, &lambda](auto &eos) {
        return eos.pressure_from_density_temperature(rho, temp, lambda);
      },
      *eos);
}

KOKKOS_INLINE_FUNCTION auto
pressure_from_conserved(const EOS *const eos, const double tau, const double V,
                        const double E, const double *const lambda) -> double {
  return std::visit(
      [&tau, &V, &E, &lambda](auto &eos) {
        return eos.pressure_from_conserved(tau, V, E, lambda);
      },
      *eos);
}

KOKKOS_INLINE_FUNCTION auto
sound_speed_from_conserved(const EOS *const eos, const double tau,
                           const double V, const double E,
                           const double *const lambda) -> double {
  return std::visit(
      [&tau, &V, &E, &lambda](auto &eos) {
        return eos.sound_speed_from_conserved(tau, V, E, lambda);
      },
      *eos);
}

KOKKOS_INLINE_FUNCTION auto
temperature_from_conserved(const EOS *const eos, const double tau,
                           const double V, const double E,
                           const double *const lambda) -> double {
  return std::visit(
      [&tau, &V, &E, &lambda](auto &eos) {
        return eos.temperature_from_conserved(tau, V, E, lambda);
      },
      *eos);
}

KOKKOS_INLINE_FUNCTION auto
sie_from_density_pressure(const EOS *const eos, const double rho,
                          const double pressure, const double *const lambda)
    -> double {
  return std::visit(
      [&rho, &pressure, &lambda](auto &eos) {
        return eos.sie_from_density_pressure(rho, pressure, lambda);
      },
      *eos);
}

KOKKOS_INLINE_FUNCTION auto gamma1(const EOS *const eos, const double tau,
                                   const double V, const double E,
                                   const double *const lambda) -> double {
  return std::visit([&tau, &V, &E, &lambda](
                        auto &eos) { return eos.gamma1(tau, V, E, lambda); },
                    *eos);
}

KOKKOS_INLINE_FUNCTION auto gamma1(const EOS *const eos) -> double {
  return std::visit([](auto &eos) { return eos.gamma1(); }, *eos);
}

KOKKOS_INLINE_FUNCTION auto initialize_eos(const ProblemIn *pin) -> EOS {
  EOS eos;
  const auto type = pin->param()->get<std::string>("eos.type");
  if (type == "paczynski") {
    // NOTE: This is currently where the tolerances are hard coded.
    // TODO(astrobarker): make tolerances runtime configurable.
    // Stretch goal: make algorithm configurable.
    static constexpr double abstol = 1.0e-14;
    static constexpr double reltol = 1.0e-14;
    static constexpr int max_iters = 100;
    eos = Paczynski(abstol, reltol, max_iters);
  } else if (type == "ideal") {
    eos = IdealGas(pin->param()->get<double>("eos.gamma"));
  } else if (type == "polytropic") {
    eos = Polytropic(pin->param()->get<double>("eos.k"),
                     pin->param()->get<double>("eos.n"));
  } else if (type == "marshak") {
    eos = Marshak(pin->param()->get<double>("eos.gamma"));
  } else {
    THROW_ATHELAS_ERROR("Please choose a valid eos!");
  }
  return eos;
}

} // namespace athelas::eos
