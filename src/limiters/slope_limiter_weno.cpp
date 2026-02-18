/**
 * @file slope_limiter_weno.cpp
 * --------------
 *
 * @author Brandon L. Barker
 * @brief Implementation of the WENO-Z slope limiter for discontinuous Galerkin
 *        methods
 *
 * @details This file implements the WENO-Z slope limiter based on H. Zhu 2020,
 *          "Simple, high-order compact WENO RKDG slope limiter". The limiter
 *          uses a compact stencil approach to maintain high-order accuracy
 *          while preventing oscillations.
 */

#include "basis/polynomial_basis.hpp"
#include "geometry/grid.hpp"
#include "kokkos_abstraction.hpp"
#include "limiters/characteristic_decomposition.hpp"
#include "limiters/slope_limiter.hpp"
#include "limiters/slope_limiter_utilities.hpp"
#include "linear_algebra.hpp"
#include "loop_layout.hpp"

namespace athelas {

using basis::NodalBasis;
using eos::EOS;
using utilities::ratio;
using namespace vars::modes;

/**
 * Apply the slope limiter. We use a compact stencil WENO-Z limiter
 * H. Zhu 2020, simple, high-order compact WENO RKDG slope limiter
 **/
void WENO::apply_slope_limiter(AthelasArray3D<double> U,
                               const GridStructure *grid,
                               const NodalBasis &basis, const EOS &eos) {

  // Do not apply for first order method or if we don't want to.
  if (order_ == 1 || !do_limiter_) {
    return;
  }

  static const IndexRange ib(grid->domain<Domain::Interior>());

  const auto nvars = nvars_;

  // --- Apply troubled cell indicator ---
  if (tci_opt_) {
    detect_troubled_cells(U, D_, grid, basis, vb_);
  }

  /* map to characteristic vars */
  if (characteristic_) {
    athelas::par_for(
        DEFAULT_FLAT_LOOP_PATTERN, "SlopeLimiter :: WENO :: ToCharacteristic",
        DevExecSpace(), ib.s, ib.e, KOKKOS_CLASS_LAMBDA(const int i) {
          // --- Characteristic Limiting Matrices ---
          // Note: using cell averages
          for (int v = 0; v < nvars; v++) {
            mult_(i, v) = U(i, CellAverage, v);
          }

          auto R_i = Kokkos::subview(R_, i, Kokkos::ALL, Kokkos::ALL);
          auto R_inv_i = Kokkos::subview(R_inv_, i, Kokkos::ALL, Kokkos::ALL);
          auto U_c_T_i = Kokkos::subview(U_c_T_, i, Kokkos::ALL);
          auto w_c_T_i = Kokkos::subview(w_c_T_, i, Kokkos::ALL);
          auto mult_i = Kokkos::subview(mult_, i, Kokkos::ALL);
          compute_characteristic_decomposition(mult_i, R_i, R_inv_i, eos);
          for (int k = 0; k < order_; k++) {
            // store w_.. = invR @ U_..
            for (int v = 0; v < nvars; v++) {
              U_c_T_i(v) = U(i, k, v);
              w_c_T_i(v) = 0.0;
            }
            MAT_MUL<3>(1.0, R_inv_i, U_c_T_i, 1.0, w_c_T_i);

            for (int v = 0; v < nvars; v++) {
              U(i, k, v) = w_c_T_i(v);
            } // end loop vars
          } // end loop k
        }); // par i
  } // end map to characteristics

  const auto dr = grid->widths();
  athelas::par_for(
      DEFAULT_FLAT_LOOP_PATTERN, "SlopeLimiter :: WENO", DevExecSpace(), ib.s,
      ib.e, KOKKOS_CLASS_LAMBDA(const int i) {
        limited_cell_(i) = 0;

        // Do nothing we don't need to limit slopes
        if (D_(i) > tci_val_ || !tci_opt_) {
          for (int v = 0; v < nvars_; ++v) {
            // get scratch modified_polynomial view for this cell's work
            auto modified_polynomial_i = Kokkos::subview(
                modified_polynomial_, i, Kokkos::ALL, Kokkos::ALL);

            // modify polynomials
            modify_polynomial(U, modified_polynomial_i, gamma_i_, gamma_l_,
                              gamma_r_, i, v);

            const double beta_l = smoothness_indicator(
                U, modified_polynomial_i, grid, basis, i, 0, v); // i - 1
            const double beta_i = smoothness_indicator(
                U, modified_polynomial_i, grid, basis, i, 1, v); // i
            const double beta_r = smoothness_indicator(
                U, modified_polynomial_i, grid, basis, i, 2, v); // i + 1
            const double tau = weno_tau(beta_l, beta_i, beta_r, weno_r_);

            // nonlinear weights w
            const double dx_i = 0.1 * dr(i);
            double w_l = non_linear_weight(gamma_l_, beta_l, tau, dx_i);
            double w_i = non_linear_weight(gamma_i_, beta_i, tau, dx_i);
            double w_r = non_linear_weight(gamma_r_, beta_r, tau, dx_i);

            const double sum_w = w_l + w_i + w_r;
            w_l /= sum_w;
            w_i /= sum_w;
            w_r /= sum_w;

            // update solution via WENO
            for (int k = 1; k < order_; k++) {
              U(i, k, v) = w_l * modified_polynomial_i(0, k) +
                           w_i * modified_polynomial_i(1, k) +
                           w_r * modified_polynomial_i(2, k);
            }

            /* Note we have limited this cell */
            limited_cell_(i) = 1;

          } // end loop v
        } // end if "limit_this_cell"
      }); // par_for i

  /* Map back to conserved variables */
  if (characteristic_) {
    athelas::par_for(
        DEFAULT_FLAT_LOOP_PATTERN, "SlopeLimiter :: WENO :: FromCharacteristic",
        DevExecSpace(), ib.s, ib.e, KOKKOS_CLASS_LAMBDA(const int i) {
          // --- Characteristic Limiting Matrices ---
          auto R_i = Kokkos::subview(R_, Kokkos::ALL, i, Kokkos::ALL);
          auto U_c_T_i = Kokkos::subview(U_c_T_, i, Kokkos::ALL);
          auto w_c_T_i = Kokkos::subview(w_c_T_, i, Kokkos::ALL);
          for (int k = 0; k < order_; k++) {
            // store w_.. = invR @ U_..
            for (int v = 0; v < nvars; v++) {
              U_c_T_i(v) = U(i, k, v);
              w_c_T_i(v) = 0.0;
            }
            MAT_MUL<3>(1.0, R_i, U_c_T_i, 1.0, w_c_T_i);

            for (int v = 0; v < nvars; v++) {
              U(i, k, v) = w_c_T_i(v);
            } // end loop vars
          } // end loop k
        }); // par_for i
  } // end map from characteristics
} // end apply slope limiter

// LimitedCell accessor
auto WENO::get_limited(const int i) const -> int {
  return (!do_limiter_) ? 0 : limited_cell_(i);
}

auto WENO::limited() const -> AthelasArray1D<int> { return limited_cell_; }
} // namespace athelas
