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

/**
 * @brief Find the largest admissible theta along a monotone decreasing function.
 *
 * Specialized bisection for Zhang–Shu-style positivity limiting in DG:
 * assumes target(0) >= 0 (cell average admissible) and target(1) < 0 (violating).
 * Returns a scaled value slightly inside the admissible set to ensure strict positivity.
 *
 * @tparam F Type of the callable object (lambda, functor) representing f(theta) = e(theta) - e_min
 * @param target Callable returning f(theta)
 * @return Largest admissible theta in [0,1] scaled by SAFETY (<1) to stay strictly positive
 *
 * @note Monotone assumption allows skipping sign checks.
 */
template <typename F>
auto bisection_monotone(F target) -> double {
    constexpr static double TOL = 1e-10;       // tight tolerance
    constexpr static int MAX_ITERS = 64;
    constexpr static double SAFETY = 1.0;      // stay inside admissible set

    double a = 0.0;   // lower bound (cell average, admissible)
    double b = 1.0;   // upper bound (possibly violating)
    double c = 0.5 * (a + b);

    int n = 0;
    while (n < MAX_ITERS) {
        c = 0.5 * (a + b);
        const double fc = target(c);

        if (fc >= 0.0) {
            // still admissible: can move up
            a = c;
        } else {
            // violation: reduce upper bound
            b = c;
        }

        if ((b - a) < TOL) {
            break;
        }

        n++;
    }

    // return a scaled value slightly inside admissible set
    return SAFETY * a;
}

template <typename F>
auto bisection(F target) -> double {
    constexpr static double TOL = 1e-6;
    constexpr static int MAX_ITERS = 64;

    double a = 0.0;
    double b = 1.0;

    double fa = 0.0;// target(a);
//    double fb = 0.0;// target(b);

    double c = 0.5;
 //   double fc = target(c);

    int n = 0;
    while (n < MAX_ITERS) {
        c = 0.5 * (a + b);
        const double fc = target(c);

        // convergence check
        if (std::abs(fc) <= TOL || 0.5*(b - a) < TOL) {
            // Optional: reduce slightly for positivity
            return 0.95 * c;  
        }

        // update interval
        if (utilities::SGN(fc) == utilities::SGN(fa)) {
            a = c;
            fa = fc;  // update fa to new left endpoint
        } else {
            b = c;
            // fb = fc;  // optional
        }

        n++;
    }

    std::println("Max iterations reached in bisection");
    return 0.95 * c;
}

template <typename F>
auto backtrace(const double theta_guess, const double min_e,
               F target) -> double {
  constexpr static double ABS_TOL = 1.0e-5;
  constexpr static double REL_TOL = 1.0e-4;
  double theta = theta_guess;
  double nodal = -1.0;

  while (nodal < ABS_TOL + REL_TOL * min_e) {
    nodal = target(theta);

    theta *= 0.9;
  }

  return theta;
}

} // namespace athelas::bel
