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
 *          - limit_specific_volume: Prevents negative specific volume (and so
 *            keeps density positive and finite) by limiting at each node
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
#include "geometry/mesh.hpp"
#include "kokkos_abstraction.hpp"
#include "kokkos_types.hpp"
#include "limiters/bound_enforcing_limiter.hpp"
#include "limiters/slope_limiter_utilities.hpp"
#include "loop_layout.hpp"
#include "math/utils.hpp"

namespace athelas::bel {

using basis::NodalBasis, basis::basis_eval;
using math::utils::ratio;

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
void limit_specific_volume(StageData &stage_data, const Mesh &mesh) {
  constexpr static double EPSILON = 1.0e-30; // maybe make this smarter

  const auto &basis = stage_data.basis();
  const int order = basis.order();

  auto U = stage_data.get_field("evolved");
  auto dm_deta = mesh.dm_deta();
  auto weights = mesh.weights();
  auto phi = basis.phi();
  constexpr int idx_tau = 0;

  athelas::par_for(
      DEFAULT_FLAT_LOOP_PATTERN, "BEL :: Limit specific volume", DevExecSpace(),
      1, U.extent(0) - 2, KOKKOS_LAMBDA(const int i) {
        // Compute cell average
        const double avg = cell_average_mass(U, dm_deta, weights, idx_tau, i);
        // Compute minimum over collocation and face evaluation points.
        double u_min = avg;
        for (int q = 0; q <= order + 1; ++q) {
          u_min = std::min(u_min, basis_eval(phi, U, i, idx_tau, q));
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
            const double nodal = U(i, q, idx_tau);

            U(i, q, idx_tau) = (1.0 - theta) * avg + theta * nodal;
          }
        }
      });
}

/**
 * @brief Zhang-Shu lower-bound limiter for evolved mass fractions.
 *
 * Uses one theta per cell for the whole composition vector. This preserves
 * every species' mass-weighted cell average and also preserves the pointwise
 * simplex constraint sum_k X_k = 1 whenever the incoming nodal polynomial
 * satisfies it.
 */
void limit_mass_fractions(StageData &stage_data, const Mesh &mesh) {
  constexpr static double X_FLOOR = 1.0e-99;

  if (!stage_data.enabled("composition")) {
    return;
  }

  const auto &basis = stage_data.basis();
  const int nNodes = basis.order();
  if (nNodes <= 1) {
    return;
  }

  auto mass_fractions = stage_data.mass_fractions("evolved");
  auto dm_deta = mesh.dm_deta();
  auto weights = mesh.weights();
  auto phi = basis.phi();
  const int ncomps = static_cast<int>(mass_fractions.extent(2));

  athelas::par_for(
      DEFAULT_FLAT_LOOP_PATTERN, "BEL :: Limit mass fractions", DevExecSpace(),
      1, mass_fractions.extent(0) - 2, KOKKOS_LAMBDA(const int i) {
        double theta = 1.0;

        for (int e = 0; e < ncomps; ++e) {
          const double avg =
              cell_average_mass(mass_fractions, dm_deta, weights, e, i);
          double x_min = avg;
          for (int q = 0; q < nNodes + 2; ++q) {
            x_min = std::min(x_min,
                             basis::basis_eval(phi, mass_fractions, i, e, q));
          }
          theta =
              std::min(theta, zhang_shu_theta_lower_bound(avg, x_min, X_FLOOR));
        }

        if (theta < 1.0) {
          for (int e = 0; e < ncomps; ++e) {
            const double avg =
                cell_average_mass(mass_fractions, dm_deta, weights, e, i);
            for (int q = 0; q < nNodes; ++q) {
              mass_fractions(i, q, e) =
                  avg + theta * (mass_fractions(i, q, e) - avg);
            }
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
void limit_internal_energy(StageData &stage_data, const Mesh &mesh) {
  const auto &basis = stage_data.basis();
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

  auto U = stage_data.get_field("evolved");
  auto dm_deta = mesh.dm_deta();
  auto sqrt_gm = mesh.sqrt_gm();
  auto weights = mesh.weights();
  auto widths = mesh.widths();
  constexpr int idx_tau = 0;
  constexpr int idx_vel = 1;
  constexpr int idx_ener = 2;

  auto phi = basis.phi();
  athelas::par_for(
      DEFAULT_FLAT_LOOP_PATTERN, "BEL :: Limit internal energy", DevExecSpace(),
      1, U.extent(0) - 2, KOKKOS_LAMBDA(const int i) {
        // Compute cell-averaged conserved quantities for reconstruction
        const double tau_avg =
            cell_average_mass(U, dm_deta, weights, idx_tau, i);
        const double v_avg = cell_average_mass(U, dm_deta, weights, idx_vel, i);
        const double etot_avg =
            cell_average_mass(U, dm_deta, weights, idx_ener, i);

        eos::EOSLambda lambda_avg;
        if constexpr (Ionization == IonizationPhysics::Active) {
          // Cell-centered approximations
          lambda_avg.data[1] = cell_average(ye, sqrt_gm, weights, widths(i), i);
          lambda_avg.data[6] =
              cell_average(e_ion_corr, sqrt_gm, weights, widths(i), i);
        }
        const double e_min_avg = min_sie(eos, 1.0 / tau_avg, lambda_avg.ptr());

        // --- Compute global theta ---
        double theta_cell = 1.0;
        for (int q = 0; q <= order + 1; ++q) {
          const double tau_q = basis_eval(phi, U, i, idx_tau, q);
          const double v_q = basis_eval(phi, U, i, idx_vel, q);
          const double E_q = basis_eval(phi, U, i, idx_ener, q);

          const double rho_q = 1.0 / tau_q;

          eos::EOSLambda lambda_q;
          if constexpr (Ionization == IonizationPhysics::Active) {
            lambda_q.data[1] = ye(i, q);
            lambda_q.data[6] = e_ion_corr(i, q);
          }

          const double e_min_q = min_sie(eos, rho_q, lambda_q.ptr());
          const double e_q = E_q - 0.5 * v_q * v_q;

          if (e_q <= e_min_q) {
            auto target = [&](double t) -> auto {
              const double v_t = v_avg + t * (v_q - v_avg);
              const double E_t = etot_avg + t * (E_q - etot_avg);
              const double e_t = E_t - 0.5 * v_t * v_t;

              return e_t - (1.0 - t) * e_min_avg - t * e_min_q;
            };

            // Solve for smallest admissible theta
            const double theta_q = bisection(target);

            theta_cell = std::min(theta_cell, theta_q);
          }
        }

        // --- Scale polynomials ---
        if (theta_cell < 1.0) {
          for (int q = 0; q < order; ++q) {
            U(i, q, idx_tau) =
                tau_avg + theta_cell * (U(i, q, idx_tau) - tau_avg);
            U(i, q, idx_vel) = v_avg + theta_cell * (U(i, q, idx_vel) - v_avg);
            U(i, q, idx_ener) =
                etot_avg + theta_cell * (U(i, q, idx_ener) - etot_avg);
          }
        }
      });
}

/**
 * @brief Apply bound enforcing limiters for fluid variables
 *
 * @param stage_data The stage data containing solution arrays
 * @param mesh The mesh structure with geometric information
 */
void apply_bound_enforcing_limiter(StageData &stage_data) {
  const auto &mesh = stage_data.mesh();
  if (stage_data.basis().order() > 1) {
    limit_specific_volume(stage_data, mesh);
    if (stage_data.enabled("ionization")) {
      limit_internal_energy<IonizationPhysics::Active>(stage_data, mesh);
    } else {
      limit_internal_energy<IonizationPhysics::Inactive>(stage_data, mesh);
    }
    // TODO(astrobarker): When mass fractions are evolved we will want to
    // call their BEL here.
    // limit_mass_fractions(stage_data, mesh);
  }
}

/**
 * @brief Apply bound enforcing limiters for radiation variables
 *
 * @param stage_data The stage data containing solution arrays
 * @param mesh The mesh structure with geometric information
 */
void apply_bound_enforcing_limiter_rad(StageData &stage_data) {
  const auto &mesh = stage_data.mesh();
  if (stage_data.basis().order() == 1) {
    return;
  }
  limit_rad_energy(stage_data, mesh);
  // limit_rad_momentum(stage_data, mesh);
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
void limit_rad_energy(StageData &stage_data, const Mesh &mesh) {
  constexpr static double EPSILON = 1.0e-13; // maybe make this smarter

  const auto &basis = stage_data.basis();
  const int order = basis.order();

  auto U = stage_data.get_field("evolved");
  auto dm_deta = mesh.dm_deta();
  auto weights = mesh.weights();
  constexpr int idx_rad_energy = 3;

  athelas::par_for(
      DEFAULT_FLAT_LOOP_PATTERN, "BEL :: Limit rad energy", DevExecSpace(), 1,
      U.extent(0) - 2, KOKKOS_LAMBDA(const int i) {
        const double E_avg =
            cell_average_mass(U, dm_deta, weights, idx_rad_energy, i);

        // --- Compute minimum over cell ---
        double theta = 1.0;
        for (int q = 0; q < order; ++q) {
          double E_q = U(i, q, idx_rad_energy);

          // Solve for smallest admissible theta
          if (E_q < EPSILON) {
            const double theta_q = backtrace(E_avg, E_q, EPSILON);
            theta = std::min(theta, theta_q);
          }
        }

        // --- Rescale ---
        if (theta < 1.0) {
          for (int q = 0; q < order; ++q) {
            const double E_q = U(i, q, idx_rad_energy);
            U(i, q, idx_rad_energy) = E_avg + theta * (E_q - E_avg);
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
void limit_rad_momentum(StageData &stage_data, const Mesh &mesh) {
  constexpr static double c = constants::c_cgs;

  const auto &basis = stage_data.basis();
  const int order = basis.order();

  auto U = stage_data.get_field("evolved");
  auto sqrt_gm = mesh.sqrt_gm();
  auto weights = mesh.weights();
  auto widths = mesh.widths();
  constexpr int idx_rad_energy = 3;
  constexpr int idx_rad_flux = 4;

  athelas::par_for(
      DEFAULT_FLAT_LOOP_PATTERN, "BEL :: Limit rad momentum", DevExecSpace(), 1,
      U.extent(0) - 2, KOKKOS_LAMBDA(const int i) {
        const double E_avg =
            cell_average(U, sqrt_gm, weights, widths(i), idx_rad_energy, i);

        const double F_avg =
            cell_average(U, sqrt_gm, weights, widths(i), idx_rad_flux, i);

        // --- Compute theta ---
        double theta_cell = 1.0;
        for (int q = 0; q <= order; ++q) {
          const double E_q = U(i, q, idx_rad_energy);
          const double F_q = U(i, q, idx_rad_flux);

          if (std::abs(F_q) > c * E_q) {
            const double s = math::utils::sgn(F_q);
            const double numerator = c * E_avg - s * F_avg;
            const double denominator = s * (F_q - F_avg) - c * (E_q - E_avg);

            if (std::abs(denominator) > 1e-14) {

              const double theta_q = numerator / denominator;

              theta_cell =
                  std::min(theta_cell, std::max(0.0, std::min(1.0, theta_q)));
            }
          }
        }

        // --- Rescale polynomial. ---
        if (theta_cell < 1.0) {
          for (int q = 0; q <= order; ++q) {
            U(i, q, idx_rad_energy) =
                E_avg + theta_cell * (U(i, q, idx_rad_energy) - E_avg);

            U(i, q, idx_rad_flux) =
                F_avg + theta_cell * (U(i, q, idx_rad_flux) - F_avg);
          }
        }
      });
}

} // namespace athelas::bel
