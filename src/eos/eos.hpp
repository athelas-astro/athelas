#pragma once
/**
 * @file eos.hpp
 * --------------
 *
 * @brief Declares equation of state classes that implement the EosBase
 * interface
 *
 * @details Defines specific equation of state implementations that inherit
 *          from the EosBase template class. It serves as the central
 * declaration point for all EOS classes in the codebase, with their
 * implementations provided in separate .cpp files.
 *
 *          We support the following equations of state:
 *          - Paczynski: stellar EOS adjusted for partial ionization
 *          - IdealGas (default): ideal gas EOS
 *          - Polytropic: Stellar polytrope EOS
 *          - Marshak: Weird ideal gas for Marshak test.
 *          - Stellar: currently an unused placceholder
 *
 *          Note: implementations are supplied in eos-specific cpp files.
 */

#include "eos/eos_base.hpp"
#include "solvers/root_finders.hpp"
#include "utils/error.hpp"

namespace athelas::eos {

using root_finders::RootFinder, root_finders::NewtonAlgorithm,
    root_finders::RelativeError, root_finders::HybridError,
    root_finders::ToleranceConfig, root_finders::AANewtonAlgorithm;

/**
 * @class Paczynski
 * @brief Paczynski stellar equation of state
 *
 * @details The Paczynski equation of state adjusted for partial ionization.
 * TODO(astrobarker): describe.
 */
class Paczynski : public EosBase<Paczynski> {
 public:
  Paczynski() = default;
  Paczynski(const double abstol, const double reltol, const int max_iters)
      : root_finder_(ToleranceConfig<double, RelativeError>{.abs_tol = abstol,
                                                            .rel_tol = reltol,
                                                            .max_iterations =
                                                                max_iters}) {}

  auto pressure_from_conserved(double tau, double V, double EmT,
                               const double *lambda) const -> double;
  auto sound_speed_from_conserved(double tau, double V, double EmT,
                                  const double *lambda) const -> double;
  auto temperature_from_conserved(double tau, double V, double E,
                                  const double *lambda) const -> double;
  auto sie_from_density_pressure(double rho, double pressure,
                                 const double *lambda) const -> double;
  [[nodiscard]] auto gamma1() const -> double;
  [[nodiscard]] auto gamma1(double tau, double V, double EmT,
                            const double *lambda) const -> double;
  [[nodiscard]] static auto p_end(double rho, double T, double ybar, double N)
      -> double;
  [[nodiscard]] static auto p_ednr(double rho, double ye) -> double;
  [[nodiscard]] static auto p_edr(double rho, double ye) -> double;
  [[nodiscard]] static auto p_ed(double p_ednr, double p_edr) -> double;
  [[nodiscard]] static auto p_e(double p_end, double p_ed) -> double;
  [[nodiscard]] static auto p_ion(double rho, double T, double N) -> double;
  [[nodiscard]] static auto degeneracy_factor(double p_ed, double p_ednr,
                                              double p_edr) -> double;

  // --- internal funcs ---
  [[nodiscard]] static auto pressure_from_rho_temperature(double temperature,
                                                          double rho,
                                                          const double *lambda)
      -> double;

  // TODO(astrobarker): The following 2 functions need an identical API.
  // However, I recompute some things unnecessarily. Make the arg lists bigger.
  [[nodiscard]] static auto specific_internal_energy(double T, double rho,
                                                     const double *lambda)
      -> double;
  [[nodiscard]] static auto dsie_dt(double T, double rho, const double *lambda)
      -> double;
  [[nodiscard]] static auto dp_dt(double T, double rho, const double *lambda)
      -> double;
  [[nodiscard]] static auto dp_drho(double T, double rho, double ybar,
                                    double pend, double ped, double f, double N,
                                    double sigma1) -> double;

 private:
  RootFinder<double, AANewtonAlgorithm<double>, RelativeError> root_finder_;
};

/**
 * @class IdealGas
 * @brief Ideal gas equation of state
 *
 * @details A standard ideal gas EOS P = (\gamma - 1) u
 */
class IdealGas : public EosBase<IdealGas> {
 public:
  IdealGas() = default;
  explicit IdealGas(double gm) : gamma_(gm) {
    if (gamma_ <= 0.0) {
      THROW_ATHELAS_ERROR(" ! IdealGas :: Adiabatic gamma <= 0.0!");
    }
  }

  auto pressure_from_conserved(double tau, double V, double EmT,
                               const double *lambda) const -> double;
  auto sound_speed_from_conserved(double tau, double V, double EmT,
                                  const double *lambda) const -> double;
  auto temperature_from_conserved(double tau, double V, double E,
                                  const double *lambda) const -> double;
  auto sie_from_density_pressure(double rho, double pressure,
                                 const double *lambda) const -> double;
  [[nodiscard]] auto gamma1(double tau, double V, double EmT,
                            const double *lambda) const noexcept -> double;
  [[nodiscard]] auto gamma1() const noexcept -> double;

 private:
  double gamma_{};
};

/**
 * @class Polytropic
 * @brief polytropic equation of state: P = K rho^(1 + 1/n)
 */
class Polytropic : public EosBase<Polytropic> {
 public:
  Polytropic() = default;
  explicit Polytropic(double k, double n) : k_(k), n_(n) {
    if (k_ <= 0.0) {
      THROW_ATHELAS_ERROR(" ! Polytropic :: k <= 0.0!");
    }
    if (n_ <= 0.0 || n > 5) {
      THROW_ATHELAS_ERROR(" ! Polytropic :: n must be in (0.0, 5.0]!");
    }
  }

  auto pressure_from_conserved(double tau, double V, double EmT,
                               const double *lambda) const -> double;
  auto sound_speed_from_conserved(double tau, double V, double EmT,
                                  const double *lambda) const -> double;
  auto temperature_from_conserved(double tau, double V, double E,
                                  const double *lambda) const -> double;
  auto sie_from_density_pressure(double rho, double pressure,
                                 const double *lambda) const -> double;
  [[nodiscard]] auto gamma1(double tau, double V, double EmT,
                            const double *lambda) const noexcept -> double;
  [[nodiscard]] auto gamma1() const noexcept -> double;

 private:
  double k_{};
  double n_{}; // polytropic index. technically an int -- dont want to cast
};

/**
 * @class Marshak
 * @brief Marshak equation of state
 *
 * @details A standard ideal gas EOS with a weird temperature
 *            P = (\gamma - 1) u
 *            T = (E_r/a)^(1/4)
 * TODO(astrobarker): thread alpha in
 */
class Marshak : public EosBase<Marshak> {
 public:
  Marshak() = default;
  explicit Marshak(double gm) : gamma_(gm) {
    if (gamma_ <= 0.0) {
      THROW_ATHELAS_ERROR(" ! Marshak :: Adiabatic gamma <= 0.0!");
    }
  }

  auto pressure_from_conserved(double tau, double V, double EmT,
                               const double *lambda) const -> double;
  auto sound_speed_from_conserved(double tau, double V, double EmT,
                                  const double *lambda) const -> double;
  auto temperature_from_conserved(double tau, double V, double E,
                                  const double *lambda) const -> double;
  auto sie_from_density_pressure(double rho, double pressure,
                                 const double *lambda) const -> double;
  [[nodiscard]] auto gamma1(double tau, double V, double EmT,
                            const double *lambda) const noexcept -> double;
  [[nodiscard]] auto gamma1() const noexcept -> double;

 private:
  double gamma_{};
};

} // namespace athelas::eos
