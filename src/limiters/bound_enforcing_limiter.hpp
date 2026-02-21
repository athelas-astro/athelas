#pragma once

#include "Kokkos_Macros.hpp"
#include "state/state.hpp"

namespace athelas::bel {

void limit_density(StageData &stage_data, const GridStructure &grid);
template <IonizationPhysics Ionization>
void limit_internal_energy(StageData &stage_data, const GridStructure &grid);
void limit_rad_energy(StageData &stage_data, const GridStructure &grid);
void limit_rad_momentum(StageData &stage_data, const GridStructure &grid);
void apply_bound_enforcing_limiter(StageData &stage_data,
                                   const GridStructure &grid);
void apply_bound_enforcing_limiter_rad(StageData &stage_data,
                                       const GridStructure &grid);
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
 * @brief Find the largest admissible theta along a monotone decreasing
 * function.
 *
 * Specialized bisection for Zhang–Shu-style positivity limiting in DG:
 * assumes target(0) >= 0 (cell average admissible) and target(1) < 0
 * (violating). Returns a scaled value slightly inside the admissible set to
 * ensure strict positivity.
 *
 * @tparam F Type of the callable object (lambda, functor) representing f(theta)
 * = e(theta) - e_min
 * @param target Callable returning f(theta)
 * @return Largest admissible theta in [0,1] scaled by SAFETY (<1) to stay
 * strictly positive
 *
 * @note Monotone assumption allows skipping sign checks.
 */
template <typename F>
auto bisection(F target) -> double {
  constexpr static double TOL = 1e-10;
  constexpr static int MAX_ITERS = 64;
  constexpr static double SAFETY = 0.99; // stay inside admissible set

  double a = 0.0; // lower bound (cell average, admissible)
  double b = 1.0; // upper bound (violating)
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

KOKKOS_INLINE_FUNCTION
auto backtrace(const double ubar, const double u_q, const double u_min)
    -> double {
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
