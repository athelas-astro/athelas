#include <cmath>

#include "opacity/opac.hpp"

namespace athelas {

auto TabularOpacity::planck_mean(const double rho, const double T,
                                 const double X, const double Z,
                                 double * /*lambda*/) const -> double {
  const double logT = std::clamp(std::log10(T), 2.0, 9.05);
  const double logR = std::log10(rho) - 3.0 * logT + 18.0;
  const double lkappa = table_.interpToReal(X, std::min(Z, 0.2), logT, logR);
  const double kappa = std::pow(10.0, lkappa);
  return std::max(floor_model_.planck(Z), kappa);
}

auto TabularOpacity::rosseland_mean(const double rho, const double T,
                                    const double X, const double Z,
                                    double * /*lambda*/) const -> double {
  const double logT = std::clamp(std::log10(T), 2.0, 9.05);
  const double logR = std::log10(rho) - 3.0 * logT + 18.0;
  const double lkappa = table_.interpToReal(X, std::min(0.2, Z), logT, logR);
  const double kappa = std::pow(10.0, lkappa);
  return std::max(floor_model_.rosseland(Z), kappa); // + 0.2 * (1.0 + X);
}

} // namespace athelas
