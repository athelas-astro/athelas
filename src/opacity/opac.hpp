/**
 * @file opac.hpp
 * --------------
 *
 * @brief Declares concrete opacity model classes that implement the OpacBase
 *        interface
 *
 * @details This header defines specific opacity model implementations that
 *          inherit from the OpacBase template class. It serves as the central
 *          declaration point for all opacity model classes in the codebase.
 *
 *          We provide the following opacity models:
 *          - Constant: A simple model with constant opacity value
 *          - Powerlaw: \kappa = k rho^exp
 *
 */

#pragma once

#include "opacity/opac_base.hpp"

namespace athelas {

class Constant : public OpacBase<Constant> {
 public:
  Constant() = default;
  explicit Constant(double kP, double kR) : kP_(kP), kR_(kR) {}

  auto planck_mean(double rho, double T, double X, double Y, double Z,
                   double *lambda) const -> double;

  auto rosseland_mean(double rho, double T, double X, double Y, double Z,
                      double *lambda) const -> double;

 private:
  double kP_{};
  double kR_{};
};

class Powerlaw : public OpacBase<Powerlaw> {
 public:
  Powerlaw() = default;
  Powerlaw(double kP, double kR, double rho_exp, double t_exp, double kP_floor,
           double kR_floor, double kP_offset, double kR_offset)
      : kP_(kP), kR_(kR), rho_exp_(rho_exp), t_exp_(t_exp), kP_floor_(kP_floor),
        kR_floor_(kR_floor), kP_offset_(kP_offset), kR_offset_(kR_offset) {}

  auto planck_mean(double rho, double T, double X, double Y, double Z,
                   double *lambda) const -> double;

  auto rosseland_mean(double rho, double T, double X, double Y, double Z,
                      double *lambda) const -> double;

 private:
  double kP_{};
  double kR_{};
  double rho_exp_{};
  double t_exp_{};
  double kP_floor_{};
  double kR_floor_{};
  double kP_offset_{};
  double kR_offset_{};
};

} // namespace athelas
