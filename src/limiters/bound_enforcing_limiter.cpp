/**
 * @file bound_enforcing_limiter.cpp
 * --------------
 *
 * @author Brandon L. Barker
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
 *          and an Anderson accelerated newton iteration is the default.
 */

#include <algorithm> // std::min, std::max
#include <cmath>
#include <cstdlib> /* abs */

#include "basic_types.hpp"
#include "basis/polynomial_basis.hpp"
#include "eos/eos_variant.hpp"
#include "kokkos_abstraction.hpp"
#include "kokkos_types.hpp"
#include "limiters/bound_enforcing_limiter.hpp"
#include "loop_layout.hpp"
#include "solvers/root_finders.hpp"
#include "utils/utilities.hpp"

namespace athelas::bel {

using basis::ModalBasis, basis::basis_eval;
using utilities::ratio;

/**
 * @brief Limits density to maintain physicality following K. Schaal et al 2015
 *
 * @details This function implements the density limiter based on K. Schaal et
 * al 2015 (ADS: 10.1093/mnras/stv1859). It finds a scaling factor theta that
 * ensures density remains positive by computing: theta = min((rho_avg -
 * eps)/(rho_nodal - rho_avg), 1)
 *
 * @param U The solution array containing conserved variables
 * @param basis The modal basis used for the solution representation
 */
void limit_density(StageData &stage_data) {
  constexpr static double EPSILON = 1.0e-30; // maybe make this smarter

  const auto &basis = stage_data.fluid_basis();

  const int order = basis.order();

  auto U = stage_data.get_field("u_cf");
  auto phi = basis.phi();
  athelas::par_for(
      DEFAULT_FLAT_LOOP_PATTERN, "BEL :: Limit density", DevExecSpace(), 1,
      U.extent(0) - 2, KOKKOS_LAMBDA(const int i) {
        double theta1 = 100000.0; // big
        const double avg =
            U(i, vars::modes::CellAverage, vars::cons::SpecificVolume);

        for (int q = 0; q <= order; ++q) {
          const double nodal =
              basis_eval(phi, U, i, vars::cons::SpecificVolume, q);
          const double frac =
              std::abs(ratio(avg - EPSILON, avg - nodal + EPSILON));
          theta1 = std::min({theta1, 1.0, frac});
        }

        for (int k = 1; k < order; k++) {
          U(i, k, vars::cons::SpecificVolume) *= theta1;
        }
      });
}

/**
 * @brief Limits the solution to maintain positivity of internal energy
 *
 * @details This function implements the bound enforcing limiter for internal
 *          energy based on K. Schaal et al 2015
 *          (ADS: 10.1093/mnras/stv1859). It finds a scaling factor theta such
 *          that (1 - theta) * U_bar + theta * U_q is positive for U being the
 *          specific internal energy.
 *
 *          The function uses three possible root finding algorithms:
 *          - bisection: A robust but slower method
 *          - Anderson accelerated newton iteration: The default method
 *          - Back tracing: A simple algorithm that steps back from theta = 1
 *
 *          All methods yield the same results and are stable on difficult
 *          problems. The simulation time is insensitive to solver choice.
 *
 *          Note: In the bisection and fixed point solvers, a small delta =
 *          1.0e-3 is subtracted from the root to ensure positivity. The
 *          backtrace algorithm overshoots the root, so this adjustment is not
 *          necessary.
 *
 * @param U The solution array containing conserved variables
 * @param basis The modal basis used for the solution representation
 */
void limit_internal_energy(StageData &stage_data) {
  constexpr static double EPSILON = 1.0e-10; // maybe make this smarter

  const auto &basis = stage_data.fluid_basis();

  const int order = basis.order();

  auto U = stage_data.get_field("u_cf");
  auto phi = basis.phi();
  athelas::par_for(
      DEFAULT_FLAT_LOOP_PATTERN, "BEL :: Limit internal energy", DevExecSpace(),
      1, U.extent(0) - 2, KOKKOS_LAMBDA(const int i) {
        double theta2 = 10000000.0;
        double temp = 0.0;

        for (int q = 0; q <= order + 1; ++q) {
          const double nodal = utilities::compute_internal_energy(U, phi, i, q);

          if (nodal > EPSILON) {
            temp = 1.0;
          } else {
            const double theta_guess = 0.4;
            // temp = bisection(U, target_func, basis, i, q);
            temp = std::clamp(
                root_finders::newton_aa(target_func, target_func_deriv,
                                        theta_guess, U, phi, i, q),
                0.0, 1.0);
          }
          theta2 = std::min(theta2, temp);
        }

        for (int k = 1; k < order; ++k) {
          for (int v = 0; v < 3; ++v) {
            U(i, k, v) *= theta2;
          }
        }
      });
}

void apply_bound_enforcing_limiter(StageData &stage_data) {
  if (stage_data.fluid_basis().order() > 1) {
    limit_density(stage_data);
    limit_internal_energy(stage_data);
  }
}

// TODO(astrobarker): much more here.
void apply_bound_enforcing_limiter_rad(StageData &stage_data) {
  if (stage_data.rad_basis().order() == 1) {
    return;
  }
  limit_rad_energy(stage_data);
  // limit_rad_momentum(stage_data);
}

void limit_rad_energy(StageData &stage_data) {
  constexpr static double EPSILON = 1.0e-4; // maybe make this smarter

  const auto &basis = stage_data.rad_basis();

  const int order = basis.order();

  auto U = stage_data.get_field("u_cf");
  auto phi = basis.phi();
  athelas::par_for(
      DEFAULT_FLAT_LOOP_PATTERN, "BEL :: Limit rad energy", DevExecSpace(), 1,
      U.extent(0) - 2, KOKKOS_LAMBDA(const int i) {
        double theta2 = 10000000.0;
        double nodal = 0.0;
        double temp = 0.0;

        for (int q = 0; q <= order + 1; ++q) {
          nodal = basis_eval(phi, U, i, vars::cons::RadEnergy, q);

          if (nodal > EPSILON + 0 *
                                    std::abs(U(i, vars::modes::CellAverage,
                                               vars::cons::RadFlux)) /
                                    constants::c_cgs) {
            temp = 1.0;
          } else {
            const double theta_guess = 0.9;
            // temp = bisection(U, target_func_rad_energy, basis, ix, iN);
            temp =
                std::clamp(root_finders::newton_aa(target_func_rad_energy,
                                                   target_func_rad_energy_deriv,
                                                   theta_guess, U, phi, i, q),
                           0.0, 1.0);
          }
          theta2 = std::abs(std::min(theta2, temp));
        }

        // When we limit the radiation energy we also limit the flux
        for (int k = 1; k < order; ++k) {
          for (int v = 3; v < 5; ++v) {
            U(i, k, v) *= theta2;
          }
        }
      });
}

void limit_rad_momentum(StageData &stage_data) {
  const auto &basis = stage_data.rad_basis();
  const int order = basis.order();

  auto U = stage_data.get_field("u_cf");
  auto phi = basis.phi();
  athelas::par_for(
      DEFAULT_FLAT_LOOP_PATTERN, "BEL :: Limit rad momentum", DevExecSpace(), 1,
      U.extent(0) - 2, KOKKOS_LAMBDA(const int i) {
        double theta2 = 10000000.0;
        double nodal = 0.0;
        double temp = 0.0;

        constexpr static double c = constants::c_cgs;

        for (int q = 0; q <= order + 1; ++q) {
          nodal = basis_eval(phi, U, i, 4, q);

          if (std::abs(nodal) <= c * U(i, 0, 3)) {
            temp = 1.0;
          } else {
            const double theta_guess = 0.9;
            temp =
                std::clamp(root_finders::newton_aa(target_func_rad_flux,
                                                   target_func_rad_flux_deriv,
                                                   theta_guess, U, phi, i, q) -
                               1.0e-16,
                           0.0, 1.0);
          }
          theta2 = std::abs(std::min(theta2, temp));
        }

        for (int k = 1; k < order; k++) {
          U(i, k, vars::cons::RadFlux) *= theta2;
        }
      });
}

/* --- Utility Functions --- */

// ( 1 - theta ) U_bar + theta U_q
auto compute_theta_state(AthelasArray3D<double> U, AthelasArray3D<double> phi,
                         const double theta, const int q, const int ix,
                         const int iN) -> double {
  return theta * (basis_eval(phi, U, ix, q, iN) -
                  U(ix, vars::modes::CellAverage, q)) +
         U(ix, vars::modes::CellAverage, q);
}

auto target_func(const double theta, AthelasArray3D<double> U,
                 AthelasArray3D<double> phi, const int ix, const int iN)
    -> double {
  const double w = 1.0e-13;
  const double s1 =
      compute_theta_state(U, phi, theta, vars::cons::Velocity, ix, iN);
  const double s2 =
      compute_theta_state(U, phi, theta, vars::cons::Energy, ix, iN);

  double const e = s2 - (0.5 * s1 * s1);

  return e - w;
}
auto target_func_deriv(const double theta, AthelasArray3D<double> U,
                       AthelasArray3D<double> phi, const int ix, const int iN)
    -> double {
  const double dE = basis_eval(phi, U, ix, vars::modes::CellAverage, iN) -
                    U(ix, vars::modes::CellAverage, vars::cons::Energy);
  const double v_q = basis_eval(phi, U, ix, vars::cons::Velocity, iN);
  const double dv = v_q - U(ix, vars::modes::CellAverage, vars::cons::Velocity);
  return dE - (v_q + theta * dv) * dv;
}

// TODO(astrobarker) some redundancy below
auto target_func_rad_flux(const double theta, AthelasArray3D<double> U,
                          AthelasArray3D<double> phi, const int ix,
                          const int iN) -> double {
  const double w = 1.0e-13;
  const double s1 =
      compute_theta_state(U, phi, theta, vars::cons::RadFlux, ix, iN);
  const double s2 =
      compute_theta_state(U, phi, theta, vars::cons::RadEnergy, ix, iN);

  const double e = std::abs(s1) / (constants::c_cgs * s2);

  return e - w;
}

auto target_func_rad_flux_deriv(const double theta, AthelasArray3D<double> U,
                                AthelasArray3D<double> phi, const int ix,
                                const int iN) -> double {
  const double dE = basis_eval(phi, U, ix, vars::cons::RadEnergy, iN) -
                    U(ix, vars::modes::CellAverage, vars::cons::RadEnergy);
  const double dF = basis_eval(phi, U, ix, vars::cons::RadFlux, iN) -
                    U(ix, vars::modes::CellAverage, vars::cons::RadFlux);
  const double E_theta =
      compute_theta_state(U, phi, theta, vars::cons::RadEnergy, ix, iN);
  const double F_theta =
      compute_theta_state(U, phi, theta, vars::cons::RadFlux, ix, iN);
  const double dfdE = -F_theta / (E_theta * E_theta * constants::c_cgs);
  const double dfdF =
      F_theta / (std::abs(F_theta) * E_theta * constants::c_cgs);
  return dfdE * dE + dfdF * dF;
}

auto target_func_rad_energy_deriv(const double theta, AthelasArray3D<double> U,
                                  AthelasArray3D<double> phi, const int ix,
                                  const int iN) -> double {
  return basis_eval(phi, U, ix, vars::cons::RadEnergy, iN) -
         U(ix, vars::modes::CellAverage, vars::cons::RadEnergy);
}

auto target_func_rad_energy(const double theta, AthelasArray3D<double> U,
                            AthelasArray3D<double> phi, const int ix,
                            const int iN) -> double {
  const double w = 1.0e-13;
  const double s1 =
      compute_theta_state(U, phi, theta, vars::cons::RadEnergy, ix, iN);

  const double e = s1;

  return e - w;
}

} // namespace athelas::bel
