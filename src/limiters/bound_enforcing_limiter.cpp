/**
 * @file bound_enforcing_limiter.cpp
 * --------------
 *
 * @author Brandon L. Barker
 * @brief Implementation of bound enforcing limiters for enforcing physicality.
 *
 * @details This file implements a suite of bound enforcing limiters based on
 *          Zhang and Shu approach for nodal DG. These limiters ensure
 *          physicality of the solution by preventing negative values of key
 *          physical quantities using the formulation:
 *          u_q = (1 - theta) * u_avg + theta * u_q
 *
 *          - limit_density: Prevents negative density by limiting at each node
 *          - limit_internal_energy: Maintains positive internal energy using
 *            root-finding algorithms at each node
 *          - limit_rad_energy: Ensures physical radiation energy values
 *          - limit_rad_momentum: Ensures physical radiation momentum values
 *
 *          The limiters are expected to be applied in the above order.
 *
 * @note Refactored from modal to nodal DG representation where the second
 *       dimension of U is now a nodal index rather than a modal coefficient.
 */

#include <algorithm> // std::min, std::max
#include <cmath>
#include <cstdlib> /* abs */

#include "basic_types.hpp"
#include "basis/polynomial_basis.hpp"
#include "eos/eos_variant.hpp"
#include "grid.hpp"
#include "kokkos_abstraction.hpp"
#include "kokkos_types.hpp"
#include "limiters/bound_enforcing_limiter.hpp"
#include "loop_layout.hpp"
#include "utils/utilities.hpp"

namespace athelas::bel {

using basis::NodalBasis, basis::basis_eval;
using utilities::ratio;

/**
 * @brief Zhang–Shu positivity limiter for specific volume (tau = 1/rho).
 *
 * Enforces tau >= EPSILON at all DG nodes by rescaling the polynomial
 * toward the cell average:
 *
 *   U_new(x) = U_avg + theta (U(x) - U_avg),
 *
 * where a single theta in [0,1] is computed per cell using the
 * minimum nodal value. The cell average is preserved exactly.
 *
 * Assumes:
 *   - Cell average tau_avg > 0
 *   - Density limiter applied before dependent limiters
 */
void limit_density(StageData &stage_data, const GridStructure &grid) {
  constexpr static double EPSILON = 1.0e-30; // maybe make this smarter

  const auto &basis = stage_data.fluid_basis();
  const int order = basis.order();

  auto U = stage_data.get_field("u_cf");
  auto sqrt_gm = grid.sqrt_gm();
  auto weights = grid.weights();
  auto widths = grid.widths();

  athelas::par_for(
      DEFAULT_FLAT_LOOP_PATTERN, "BEL :: Limit density", DevExecSpace(), 1,
      U.extent(0) - 2, KOKKOS_LAMBDA(const int i) {
        // Compute cell average
        const double avg =
            cell_average(U, sqrt_gm, weights, widths(i), vars::cons::SpecificVolume, i);
        // Compute minimum in cell
        double u_min = avg;
        for (int q = 0; q < order; ++q) {
          u_min = std::min(u_min, U(i, q, vars::cons::SpecificVolume));
        }

        // Compute theta
        double theta = 1.0;
        if (u_min < EPSILON) {
          const double denom = avg - u_min;
          if (denom > EPSILON) {
            theta = std::min(1.0, (avg - EPSILON) / denom);
          }
        }

        // 4. Rescale polynomial
        if (theta < 1.0) {
          for (int q = 0; q < order; ++q) {
            const double nodal = U(i, q, vars::cons::SpecificVolume);

            U(i, q, vars::cons::SpecificVolume) =
                (1.0 - theta) * avg + theta * nodal;
          }
        }
      });
}

/**
 * @brief Nonlinear Zhang–Shu limiter for specific internal energy.
 *
 * Enforces e >= e_min(rho) at all DG nodes, where
 *
 *   e = E - 0.5 v^2,
 *
 * using a convex rescaling of the full conserved state:
 *
 *   U_new = U_avg + theta (U - U_avg).
 *
 * Because the constraint is nonlinear in theta, a bracketed
 * root solve is used per violating node to determine the
 * minimal admissible theta.
 */
template <IonizationPhysics Ionization>
void limit_internal_energy(StageData &stage_data, const GridStructure &grid) {
  const auto &basis = stage_data.fluid_basis();
  const auto &eos = stage_data.eos();
  const int order = basis.order();

  // Dummy placeholders (never used in Inactive case)
  AthelasArray2D<double> e_ion_corr;
  AthelasArray2D<double> ye;

  if constexpr (Ionization == IonizationPhysics::Active) {
    const auto *const ionization_state = stage_data.ionization_state();
    const auto *const comps = stage_data.comps();
    e_ion_corr = ionization_state->e_ion_corr();
    ye = comps->ye();
  }

  auto U = stage_data.get_field("u_cf");
  auto sqrt_gm = grid.sqrt_gm();
  auto weights = grid.weights();
  auto widths = grid.widths();

  athelas::par_for(
      DEFAULT_FLAT_LOOP_PATTERN, "BEL :: Limit internal energy", DevExecSpace(),
      1, U.extent(0) - 2, KOKKOS_LAMBDA(const int i) {
        // Compute cell-averaged conserved quantities for reconstruction
        const double tau_avg =
            cell_average(U, sqrt_gm, weights, widths(i), vars::cons::SpecificVolume, i);
        const double v_avg =
            cell_average(U, sqrt_gm, weights, widths(i), vars::cons::Velocity, i);
        const double etot_avg =
            cell_average(U, sqrt_gm, weights, widths(i), vars::cons::Energy, i);

        eos::EOSLambda lambda_avg;
        if constexpr (Ionization == IonizationPhysics::Active) {
          // Cell-centered approximations
          lambda_avg.data[1] = cell_average(ye, sqrt_gm, weights, widths(i), i);
          lambda_avg.data[6] = cell_average(e_ion_corr, sqrt_gm, weights, widths(i), i);
        }
        const double e_min_avg = min_sie(eos, 1.0 / tau_avg, lambda_avg.ptr());

        // --- Compute global theta ---
        double theta_cell = 1.0;

        for (int q = 0; q < order; ++q) {

          const double tau_q = U(i, q, vars::cons::SpecificVolume);
          const double v_q = U(i, q, vars::cons::Velocity);
          const double E_q = U(i, q, vars::cons::Energy);

          const double rho_q = 1.0 / tau_q;

          eos::EOSLambda lambda_q;
          if constexpr (Ionization == IonizationPhysics::Active) {
            lambda_q.data[1] = ye(i, q + 1);
            lambda_q.data[6] = e_ion_corr(i, q + 1);
          }

          const double e_min_q = min_sie(eos, rho_q, lambda_q.ptr());
          const double e_q = E_q - 0.5 * v_q * v_q;

          if (e_q <= e_min_q) {
            // Define nonlinear target function
            auto target = [&](double t) -> auto {
              const double v_t = v_avg + t * (v_q - v_avg);
              const double E_t = etot_avg + t * (E_q - etot_avg);
              const double e_t = E_t - 0.5 * v_t * v_t;

              return e_t - (1.0 - t) * e_min_avg - t * e_min_q;
            };

            // Solve for smallest admissible theta
            //const double theta_q = backtrace(1.0, 0.0, target);
            const double theta_q = bisection_monotone(target);

            theta_cell = std::min(theta_cell, theta_q);
          }
        }

        // --- Scale polynomials ---
        if (theta_cell < 1.0) {
          for (int q = 0; q < order; ++q) {
            U(i, q, vars::cons::SpecificVolume) =
                tau_avg +
                theta_cell * (U(i, q, vars::cons::SpecificVolume) - tau_avg);
            U(i, q, vars::cons::Velocity) =
                v_avg + theta_cell * (U(i, q, vars::cons::Velocity) - v_avg);
            U(i, q, vars::cons::Energy) =
                etot_avg + theta_cell * (U(i, q, vars::cons::Energy) - etot_avg);
          }
        }
      });
}

/**
 * @brief Apply bound enforcing limiters for fluid variables
 *
 * @param stage_data The stage data containing solution arrays
 * @param grid The grid structure with geometric information
 */
void apply_bound_enforcing_limiter(StageData &stage_data,
                                   const GridStructure &grid) {
  if (stage_data.fluid_basis().order() > 1) {
    limit_density(stage_data, grid);
    if (stage_data.ionization_enabled()) {
      limit_internal_energy<IonizationPhysics::Active>(stage_data, grid);
    } else {
      limit_internal_energy<IonizationPhysics::Inactive>(stage_data, grid);
    }
  }
}

/**
 * @brief Apply bound enforcing limiters for radiation variables
 *
 * @param stage_data The stage data containing solution arrays
 * @param grid The grid structure with geometric information
 */
void apply_bound_enforcing_limiter_rad(StageData &stage_data,
                                       const GridStructure &grid) {
  if (stage_data.rad_basis().order() == 1) {
    return;
  }
  limit_rad_energy(stage_data, grid);
  // limit_rad_momentum(stage_data, grid);
}

/**
 * @brief Zhang–Shu positivity limiter for specific radiation energy.
 *
 * Enforces E_rad/rho >= EPSILON at all DG nodes via linear scaling:
 *
 *   E_new = E_avg + theta (E - E_avg),
 *
 * where theta is computed once per cell from the minimum nodal value.
 */
void limit_rad_energy(StageData &stage_data, const GridStructure &grid) {
  constexpr static double EPSILON = 1.0e-13; // maybe make this smarter

  const auto &basis = stage_data.rad_basis();
  const int order = basis.order();

  auto U = stage_data.get_field("u_cf");
  auto sqrt_gm = grid.sqrt_gm();
  auto weights = grid.weights();
  auto widths = grid.widths();
  athelas::par_for(
      DEFAULT_FLAT_LOOP_PATTERN, "BEL :: Limit rad energy", DevExecSpace(), 1,
      U.extent(0) - 2, KOKKOS_LAMBDA(const int i) {
        const double tau_avg =
            cell_average(U, sqrt_gm, weights, widths(i), vars::cons::SpecificVolume, i);
        const double E_avg =
            cell_average(U, sqrt_gm, weights, widths(i), vars::cons::RadEnergy, i) / tau_avg;

        // --- Compute minimum over cell ---
        double theta = 1.0;
        for (int q = 0; q < order; ++q) {
          double E_q = U(i, q, vars::cons::RadEnergy) / U(i, q, vars::cons::SpecificVolume);

          // Solve for smallest admissible theta
          if (E_q < EPSILON) {
            const double theta_q = backtrace(E_avg, E_q, EPSILON);
            theta = std::min(theta, theta_q);
          }
        }

        // --- Rescale ---
        if (theta < 1.0) {
          for (int q = 0; q < order; ++q) {
            const double E_q = U(i, q, vars::cons::RadEnergy) / U(i, q, vars::cons::SpecificVolume);
            U(i, q, vars::cons::RadEnergy) = E_avg + theta * (E_q - E_avg);
          }
        }
      });
}

/**
 * @brief Zhang–Shu convex limiter enforcing |F| <= c E for radiation.
 *
 * Enforces the realizability constraint
 *
 *   |F| <= c E
 *
 * at all DG nodes using convex rescaling of the radiation state:
 *
 *   (E,F)_new = (E,F)_avg + theta [(E,F) - (E,F)_avg].
 *
 * The admissible set { E > 0, |F| <= cE } is convex; therefore
 * a single theta per cell guarantees realizability everywhere.
 * The required theta is obtained analytically from the linear
 * intersection with the boundary |F| = cE.
 *
 * Should be applied after radiation energy positivity.
 */
void limit_rad_momentum(StageData &stage_data, const GridStructure &grid) {
  constexpr static double c = constants::c_cgs;

  const auto &basis = stage_data.rad_basis();
  const int order = basis.order();

  auto U = stage_data.get_field("u_cf");
  auto sqrt_gm = grid.sqrt_gm();
  auto weights = grid.weights();
  auto widths = grid.widths();

  athelas::par_for(
      DEFAULT_FLAT_LOOP_PATTERN, "BEL :: Limit rad momentum", DevExecSpace(), 1,
      U.extent(0) - 2, KOKKOS_LAMBDA(const int i) {
        const double E_avg =
            cell_average(U, sqrt_gm, weights, widths(i), vars::cons::RadEnergy, i);

        const double F_avg =
            cell_average(U, sqrt_gm, weights, widths(i), vars::cons::RadFlux, i);

        // --- Compute theta ---
        double theta_cell = 1.0;
        for (int q = 0; q <= order; ++q) {
          const double E_q = U(i, q, vars::cons::RadEnergy);
          const double F_q = U(i, q, vars::cons::RadFlux);

          if (std::abs(F_q) > c * E_q) {
            const double s = utilities::SGN(F_q);
            const double numerator = c * E_avg - s * F_avg;
            const double denominator = s * (F_q - F_avg) - c * (E_q - E_avg);

            if (std::abs(denominator) > 1e-14) {

              const double theta_q = numerator / denominator;

              theta_cell = std::min(theta_cell, std::max(0.0, std::min(1.0, theta_q)));
            }
          }
        }

        // --- Rescale polynomial. ---
        if (theta_cell < 1.0) {
          for (int q = 0; q <= order; ++q) {
            U(i, q, vars::cons::RadEnergy) =
                E_avg + theta_cell * (U(i, q, vars::cons::RadEnergy) - E_avg);

            U(i, q, vars::cons::RadFlux) =
                F_avg + theta_cell * (U(i, q, vars::cons::RadFlux) - F_avg);
          }
        }
      });
}

} // namespace athelas::bel
