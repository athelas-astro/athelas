/**
 * @file opac_powerlaw.cpp
 * --------------
 *
 * @author Brandon L. Barker
 * @brief Density power law opacity model
 * TODO(astrobarker): generalize to rho,T powerlaw
 */

#include <cmath>

#include "opacity/opac.hpp"

namespace athelas {

auto Powerlaw::planck_mean(const double rho, const double T, const double /*X*/,
                           const double Z, double * /*lambda*/) const
    -> double {
  const double kappa =
      kP_ * std::pow(rho, rho_exp_) * std::pow(T, t_exp_) + kP_offset_;
  return std::max(floor_model_.planck(Z), kappa);
}

auto Powerlaw::rosseland_mean(const double rho, const double T,
                              const double /*X*/, const double Z,
                              double * /*lambda*/) const -> double {
  const double kappa =
      kR_ * std::pow(rho, rho_exp_) * std::pow(T, t_exp_) + kR_offset_;
  return std::max(floor_model_.rosseland(Z), kappa);
}

} // namespace athelas
