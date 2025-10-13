/**
 * @file slope_limiter_utilities.cpp
 * --------------
 *
 * @author Brandon L. Barker
 * @brief Utility functions for slope limiters.
 */

#include <algorithm> // std::min, std::max
#include <cmath>
#include <cstdlib> /* abs */
#include <limits>

#include "geometry/grid.hpp"
#include "kokkos_abstraction.hpp"
#include "kokkos_types.hpp"
#include "limiters/slope_limiter.hpp"
#include "limiters/slope_limiter_utilities.hpp"
#include "loop_layout.hpp"
#include "polynomial_basis.hpp"
#include "utils/utilities.hpp"

namespace athelas {

using basis::ModalBasis;
using eos::EOS;

auto initialize_slope_limiter(const std::string field,
                              const GridStructure *grid, const ProblemIn *pin,
                              const std::vector<int> &vars, const int nvars)
    -> SlopeLimiter {
  const auto enabled =
      pin->param()->get<bool>(field + ".limiter.enabled", false);
  const auto type =
      pin->param()->get<std::string>(field + ".limiter.type", "minmod");
  SlopeLimiter S_Limiter;
  if (enabled) {
    if (utilities::to_lower(type) == "minmod") {
      S_Limiter = TVDMinmod(
          enabled, grid, vars, nvars, pin->param()->get<int>(field + ".porder"),
          pin->param()->get<double>(field + ".limiter.b_tvd"),
          pin->param()->get<double>(field + ".limiter.m_tvb"),
          pin->param()->get<bool>(field + ".limiter.characteristic"),
          pin->param()->get<bool>(field + ".limiter.tci_enabled"),
          pin->param()->get<double>(field + ".limiter.tci_val"));
    } else {
      S_Limiter = WENO(
          enabled, grid, vars, nvars, pin->param()->get<int>(field + ".porder"),
          pin->param()->get<double>(field + ".limiter.gamma_i"),
          pin->param()->get<double>(field + ".limiter.gamma_l"),
          pin->param()->get<double>(field + ".limiter.gamma_r"),
          pin->param()->get<double>(field + ".limiter.weno_r"),
          pin->param()->get<bool>(field + ".limiter.characteristic"),
          pin->param()->get<bool>(field + ".limiter.tci_enabled"),
          pin->param()->get<double>(field + ".limiter.tci_val"));
    }
  } else {
    S_Limiter = Unlimited(); // no-op "limiter" when limiting is disabled
  }

  return S_Limiter;
}

/**
 *  UNUSED
 *  Barth-Jespersen limiter
 *  Parameters:
 *  -----------
 *  U_v_*: left/right vertex values on target cell
 *  U_c_*: cell averages of target cell + neighbors
 *    [ left, target, right ]
 *  alpha: scaling coefficient for BJ limiter.
 *    alpha=1 is classical limiter, alpha=0 enforces constant solutions
 **/
auto barth_jespersen(double U_v_L, double U_v_R, double U_c_L, double U_c_T,
                     double U_c_R, double alpha) -> double {
  // Get U_min, U_max
  double U_min_L = 10000000.0 * U_c_T;
  double U_min_R = 10000000.0 * U_c_T;
  double U_max_L = std::numeric_limits<double>::epsilon() * U_c_T * 0.00001;
  double U_max_R = std::numeric_limits<double>::epsilon() * U_c_T * 0.00001;

  U_min_L = std::min(U_min_L, std::min(U_c_T, U_c_L));
  U_max_L = std::max(U_max_L, std::max(U_c_T, U_c_L));
  U_min_R = std::min(U_min_R, std::min(U_c_T, U_c_R));
  U_max_R = std::max(U_max_R, std::max(U_c_T, U_c_R));

  // loop of cell certices
  double phi_L = 0.0;
  double phi_R = 0.0;

  // left vertex
  if (U_v_L - U_c_T + 1.0 > 1.0) {
    phi_L = std::min(1.0, alpha * (U_max_L - U_c_T) / (U_v_L - U_c_T));
  } else if (U_v_L - U_c_T + 1.0 < 1.0) {
    phi_L = std::min(1.0, alpha * (U_min_L - U_c_T) / (U_v_L - U_c_T));
  } else {
    phi_L = 1.0;
  }

  // right vertex
  if (U_v_R - U_c_T + 1.0 > 1.0) {
    phi_R = std::min(1.0, alpha * (U_max_R - U_c_T) / (U_v_R - U_c_T));
  } else if (U_v_R - U_c_T + 1.0 < 1.0) {
    phi_R = std::min(1.0, alpha * (U_min_R - U_c_T) / (U_v_R - U_c_T));
  } else {
    phi_R = 1.0;
  }

  // return min of two values
  return std::min(phi_L, phi_R);
}

/**
 * Apply the Troubled Cell Indicator of Fu & Shu (2017)
 * to flag cells for limiting
 * Detects smoothness by comparing local cell averages to extrapolated
 * neighbor projections.
 **/
void detect_troubled_cells(const AthelasArray3D<double> U,
                           AthelasArray1D<double> D, const GridStructure *grid,
                           const ModalBasis *basis,
                           const std::vector<int> &vars) {
  static const IndexRange ib(grid->domain<Domain::Interior>());
  athelas::par_for(
      DEFAULT_FLAT_LOOP_PATTERN, "SlopeLimiter :: TCI :: Zero", DevExecSpace(),
      ib.s, ib.e, KOKKOS_LAMBDA(const int i) { D(i) = 0.0; });

  // Cell averages by extrapolating L and R neighbors into current cell

  const auto phi = basis->phi();
  const auto widths = grid->widths();
  const auto weights = grid->weights();
  const auto sqrt_gm = grid->sqrt_gm();
  athelas::par_for(
      DEFAULT_FLAT_LOOP_PATTERN, "SlopeLimiter :: TCI", DevExecSpace(), ib.s,
      ib.e, KOKKOS_LAMBDA(const int i) {
        const double dr = widths(i);
        for (int v : vars) {
          if (v == 1 || v == 4) {
            continue; /* skip momenta */
          }
          const double cell_avg = U(i, 0, v);

          // Extrapolate neighboring poly representations into current cell
          // and compute the new cell averages
          const double cell_avg_L_T = cell_average(U, sqrt_gm, weights, dr, phi,
                                                   v, i + 1, -1); // from right
          const double cell_avg_R_T = cell_average(U, sqrt_gm, weights, dr, phi,
                                                   v, i - 1, +1); // from left
          const double cell_avg_L = U(i - 1, 0, v); // native left
          const double cell_avg_R = U(i + 1, 0, v); // native right

          const double result = (std::abs(cell_avg - cell_avg_L_T) +
                                 std::abs(cell_avg - cell_avg_R_T));

          const double denominator = std::max(
              {std::abs(cell_avg_L), std::abs(cell_avg_R), cell_avg, 1.0e-10});

          D(i) = std::max(D(i), result / denominator);
        } // loop v;
      }); // par_for i
}

/**
 * Modify polynomials a la
 * H. Zhu et al 2020, simple and high-order
 * compact WENO RKDG slope limiter
 **/
void modify_polynomial(const AthelasArray3D<double> U,
                       AthelasArray2D<double> modified_polynomial,
                       const double gamma_i, const double gamma_l,
                       const double gamma_r, const int ix, const int q) {
  const double Ubar_i = U(ix, 0, q);
  const double fac = 1.0;
  const int order = U.extent(2);

  const double modified_p_slope_mag =
      fac * std::min({U(ix - 1, 1, 1), U(ix, 1, 1), U(ix + 1, 1, 1)});
  const int sign_l = utilities::SGN(U(ix - 1, 1, 1));
  const int sign_r = utilities::SGN(U(ix + 1, 1, 1));

  modified_polynomial(0, 0) = Ubar_i;
  modified_polynomial(2, 0) = Ubar_i;
  modified_polynomial(0, 1) = sign_l * modified_p_slope_mag;
  modified_polynomial(2, 1) = sign_r * modified_p_slope_mag;

  for (int k = 2; k < order; k++) {
    modified_polynomial(0, k) = 0.0;
    modified_polynomial(2, k) = 0.0;
  }

  for (int k = 0; k < order; k++) {
    modified_polynomial(1, k) =
        U(ix, k, 1) / gamma_i -
        (gamma_l / gamma_i) * modified_polynomial(0, k) -
        (gamma_r / gamma_i) * modified_polynomial(2, k);
  }
}

// WENO smoothness indicator beta
// TODO(astrobarker): pass in views remove accessors
auto smoothness_indicator(const AthelasArray3D<double> U,
                          const AthelasArray2D<double> modified_polynomial,
                          const GridStructure *grid, const ModalBasis *basis,
                          const int ix, const int i, const int /*q*/)
    -> double {
  const int k = U.extent(1);

  const auto dr = grid->widths();
  const auto weights = grid->weights();
  const auto r = grid->nodal_grid();

  double beta = 0.0; // output var
  for (int s = 1; s < k; s++) { // loop over modes
    // integrate mode on cell
    double local_sum = 0.0;
    for (int q = 0; q < k; q++) {
      const auto X = r(ix, q);
      local_sum += weights(q) *
                   std::pow(modified_polynomial(s, i) *
                                ModalBasis::d_legendre_n(k, s, X),
                            2.0) *
                   std::pow(dr(ix), 2.0 * s);
    }
    beta += local_sum;
  }
  return beta;
}

auto non_linear_weight(const double gamma, const double beta, const double tau,
                       const double eps) -> double {
  return gamma * (1.0 + tau / (eps + beta));
}

// weno-z tau variable
auto weno_tau(const double beta_l, const double beta_i, const double beta_r,
              const double weno_r) -> double {
  return std::pow((std::abs(beta_i - beta_l) + std::abs(beta_i - beta_r)) / 2.0,
                  weno_r);
}

} // namespace athelas
