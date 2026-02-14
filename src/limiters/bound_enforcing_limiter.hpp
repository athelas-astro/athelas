/**
 * @file bound_enforcing_limiter.hpp
 * --------------
 *
 * @brief Implementation of bound enforcing limiters for enforcing physicality.
 *
 * @details This file implements a suite of bound enforcing limiters based on
 *          K. Schaal et al 2015 (ADS: 10.1093/mnras/stv1859). These limiters
 *          ensure physicality of the solution by preventing negative values of
 *          key physical quantities:
 *
 *          - limit_density: Prevents negative density by scaling slope
 *            coefficients
 *          - limit_internal_energy: Maintains positive internal energy using
 *            root-finding algorithms
 *          - limit_rad_momentum: Ensures physical radiation momentum values
 *
 *          Multiple root finders for the internal energy solve are implemented
 *          and an Anderson accelerated fixed point iteration is the default.
 *          point iteration being the default choice.
 */

#pragma once

#include <print>

#include "basis/polynomial_basis.hpp"
#include "limiters/slope_limiter_utilities.hpp"
#include "state/state.hpp"
#include "utils/utilities.hpp"

namespace athelas::bel {

/**
 * @brief Zhang–Shu positivity limiter
 *
 * This templated function applies the Zhang–Shu limiter to ensure that
 * a specific variable (e.g., density, energy, flux) respects a lower bound
 * or realizability condition using convex rescaling:
 *
 *   U_new(x) = U_avg + theta * (U(x) - U_avg)
 *
 * where theta is computed such that the limiting condition is satisfied.
 *
 * @tparam T Type of the variable (e.g., double, float).
 * @tparam Func Type of the target function used for solving theta.
 * @param U The 3D array of values to limit.
 * @param i The cell index.
 * @param v The variable index.
 * @param U_avg The cell average value of the variable.
 * @param target A lambda function representing the target condition.
 * @param min_value The lower bound to apply (e.g., EPSILON for positivity).
 * @param order The number of nodes (DG order).
 * @param EPSILON A small positive value used for numerical stability.
 * @return void, U is updated in-place with the limited values.
 */
template <typename T, typename Func>
void apply_zhang_shu_limiter(T U, const int i, const int v, const double U_avg, const Func& target, const double min_value, const int order, const double EPSILON = 1e-12) {
  double theta = 1.0;
  for (int q = 0; q <= order; ++q) {
    const T U_q = U(i, q, v);
    if (U_q < min_value + EPSILON) {
      // Define the target function to satisfy the constraint
      auto solve_target = [&](double t) {
        // Linear interpolation between the average and nodal values
        T U_t = U_avg + t * (U_q - U_avg);
        return target(U_t);  // Check if condition is met at theta = t
      };

      // Find theta using a simple backtrace or root-finding method
      const double theta_q = backtrace(theta, min_value, solve_target);

      // Update the node value using the computed theta
      U(i, q, v) = U_avg + theta_q * (U_q - U_avg);
      
      // Store the smallest theta across all nodes
      theta = std::min(theta, theta_q);
    }
  }

  // Apply the final scaling across all nodes
  for (int q = 0; q <= order; ++q) {
    U(i, q, v) = U_avg + theta * (U(i, q, v) - U_avg);
  }
}

void limit_density(StageData &stage_data, const GridStructure &grid);
template <IonizationPhysics Ionization>
void limit_internal_energy(StageData &stage_data, const GridStructure &grid);
void limit_rad_energy(StageData &stage_data, const GridStructure &grid);
void limit_rad_momentum(StageData &stage_data, const GridStructure &grid);
void apply_bound_enforcing_limiter(StageData &stage_data, const GridStructure &grid);
void apply_bound_enforcing_limiter_rad(StageData &stage_data, const GridStructure &grid);
auto compute_theta_state(AthelasArray3D<double> U, AthelasArray3D<double>,
                         double theta, int q, int ix, int iN) -> double;
auto target_func(double theta, double min_e, AthelasArray3D<double> U,
                 AthelasArray3D<double> phi, int ix, int iN) -> double;
auto target_func_deriv(double theta, double min_e, AthelasArray3D<double> U,
                       AthelasArray3D<double> phi, int ix, int iN) -> double;
auto target_func_rad_flux(double theta, AthelasArray3D<double> U,
                          AthelasArray3D<double> phi, int ix, int iN) -> double;
auto target_func_rad_flux_deriv(double theta, AthelasArray3D<double> U,
                                AthelasArray3D<double> phi, int ix, int iN)
    -> double;
auto target_func_rad_energy(double theta, AthelasArray3D<double> U,
                            AthelasArray3D<double> phi, int ix, int iN)
    -> double;
auto target_func_rad_energy_deriv(double theta, AthelasArray3D<double> U,
                                  AthelasArray3D<double> phi, int ix, int iN)
    -> double;

template <typename F>
auto bisection(AthelasArray3D<double> U, F target,
               const basis::NodalBasis *basis, const int ix, const int iN)
    -> double {
  constexpr static double TOL = 1e-10;
  constexpr static int MAX_ITERS = 100;
  constexpr static double delta = 1.0e-3; // reduce root by delta

  // bisection bounds on theta
  double a = 0.0;
  double b = 1.0;
  double c = 0.5;

  double fa = 0.0; // f(a) etc
  double fc = 0.0;

  int n = 0;
  while (n <= MAX_ITERS) {
    c = (a + b) / 2.0;

    fa = target(a, U, basis, ix, iN);
    fc = target(c, U, basis, ix, iN);

    if (std::abs(fc) <= TOL || (b - a) / 2.0 < TOL) {
      return c - delta;
    }

    // new interval
    if (utilities::SGN(fc) == utilities::SGN(fa)) {
      a = c;
    } else {
      b = c;
    }

    n++;
  }

  std::println("Max Iters Reach In bisection");
  return c - delta;
}

template <typename F>
auto backtrace(const double theta_guess, const double min_e,
               F target) -> double {
  constexpr static double ABS_TOL = 1.0e-10; // maybe make this smarter
  constexpr static double REL_TOL = 1.0e-6; // maybe make this smarter
  double theta = theta_guess;
  double nodal = -1.0;

  while (nodal < ABS_TOL + REL_TOL * min_e) {
    nodal = target(theta);

    theta *= 0.9;
  }

  return theta;
}

} // namespace athelas::bel
