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

#include "Kokkos_Macros.hpp"
#include "basis/polynomial_basis.hpp"
#include "limiters/slope_limiter_utilities.hpp"
#include "state/state.hpp"
#include "utils/utilities.hpp"

namespace athelas::bel {

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

KOKKOS_INLINE_FUNCTION
auto backtrace(const double ubar, const double u_q, const double u_min) -> double {
  constexpr static double ABS_TOL = 1.0e-5;
  constexpr static double REL_TOL = 1.0e-4;
  double theta = 1.0;
  double u_lim = 0.0;

  while (u_lim < ABS_TOL + REL_TOL * u_min) {
    u_lim = (1.0 - theta) * ubar + theta * u_q;
    theta *= 0.9;
  }

  return theta;
}

} // namespace athelas::bel
