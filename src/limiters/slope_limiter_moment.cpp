#include <cstdlib> /* abs */

#include "basic_types.hpp"
#include "geometry/mesh.hpp"
#include "kokkos_abstraction.hpp"
#include "kokkos_types.hpp"
#include "limiters/characteristic_decomposition.hpp"
#include "limiters/slope_limiter.hpp"
#include "limiters/slope_limiter_utilities.hpp"
#include "loop_layout.hpp"
#include "math/linear_algebra.hpp"

namespace athelas {

using basis::NodalBasis;
using eos::EOS;
using namespace vars::modes;

/**
 * Hierarchical moment limiter. See Krivodonova (2007), JCP 226.
 *
 * For each cell we limit the modal coefficients from the highest mode down to
 * the slope, comparing mode j against the (scaled) differences of mode (j-1)
 * in the neighboring cells:
 *
 *   u~(i,j) = minmod( u(i,j),
 *                     a_j ( u(i+1,j-1) - u(i,j-1) ),
 *                     a_j ( u(i,j-1) - u(i-1,j-1) ) )
 *
 * The cascade stops as soon as a mode is left unchanged.
 **/
void MomentLimiter::apply_slope_limiter(AthelasArray3D<double> U,
                                        const Mesh &mesh,
                                        const NodalBasis &basis,
                                        const EOS &eos) {

  // Do not apply for first order method or if we don't want to.
  if (order_ == 1 || !enabled_) {
    return;
  }

  constexpr static double sl_threshold_ =
      1.0e-4; // TODO(astrobarker): move to input deck

  static constexpr int ilo = 1;
  const int ihi = mesh.get_ihi();

  const int nvars = nvars_;
  const int order = order_;

  athelas::par_for(
      DEFAULT_FLAT_LOOP_PATTERN, "SlopeLimiter :: Moment :: Reset indicator",
      DevExecSpace(), ilo, ihi,
      KOKKOS_CLASS_LAMBDA(const int i) { limited_cell_(i) = 0; });

  // --- Apply troubled cell indicator ---
  if (tci_opt_) {
    detect_troubled_cells(U, D_, mesh, basis, vb_);
  }

  // --- Map to modal basis ---
  basis.nodal_to_modal(u_k_, U, vb_);

  // --- map to characteristic vars ---
  // Note: unlike TVDMinmod, the moment limiter touches every mode, so the
  // characteristic transform must be applied to all modes k = 0 .. order-1.
  if (characteristic_) {
    athelas::par_for(
        DEFAULT_FLAT_LOOP_PATTERN, "SlopeLimiter :: Moment :: ToCharacteristic",
        DevExecSpace(), ilo, ihi, KOKKOS_CLASS_LAMBDA(const int i) {
          // --- Characteristic Limiting Matrices ---
          // Note: using cell averages
          for (int v = 0; v < nvars; ++v) {
            mult_(i, v) = u_k_(i, CellAverage, v);
          }

          auto R_i = Kokkos::subview(R_, i, Kokkos::ALL, Kokkos::ALL);
          auto R_inv_i = Kokkos::subview(R_inv_, i, Kokkos::ALL, Kokkos::ALL);
          auto U_c_T_i = Kokkos::subview(U_c_T_, i, Kokkos::ALL);
          auto w_c_T_i = Kokkos::subview(w_c_T_, i, Kokkos::ALL);
          auto Mult_i = Kokkos::subview(mult_, i, Kokkos::ALL);
          compute_characteristic_decomposition(Mult_i, R_i, R_inv_i, eos);
          for (int k = 0; k < order; ++k) {
            // store w_.. = invR @ U_..
            for (int v = 0; v < nvars; ++v) {
              U_c_T_i(v) = u_k_(i, k, v);
              w_c_T_i(v) = 0.0;
            }
            math::linalg::mat_mul<3>(1.0, R_inv_i, U_c_T_i, 0.0, w_c_T_i);

            for (int v = 0; v < nvars; ++v) {
              u_k_(i, k, v) = w_c_T_i(v);
            } // end loop vars
          } // end loop k
        }); // par i
  } // end map to characteristics

  // The moment stencil uses neighboring lower-order modes. Freeze the modal
  // state before limiting so cells do not read neighbors that have already
  // been partially limited by the same parallel kernel.
  Kokkos::deep_copy(u_k_unlimited_, u_k_);

  auto dr = mesh.widths();
  athelas::par_for(
      DEFAULT_FLAT_LOOP_PATTERN, "SlopeLimiter :: Moment", DevExecSpace(), ilo,
      ihi, KOKKOS_CLASS_LAMBDA(const int i) {
        // Do nothing if the TCI says this cell is fine.
        if (D_(i) > tci_val_ || !tci_opt_) {
          // Element-width factors. Neighbor differences are formed as physical
          // slopes (divide by inter-centroid distance) and scaled by the half
          // width which keeps the limiter correct on the moving
          // Lagrangian mesh.
          const double scale = 0.5 * dr(i);
          const double dist_p = 0.5 * (dr(i) + dr(i + 1));
          const double dist_m = 0.5 * (dr(i) + dr(i - 1));

          for (int v = 0; v < nvars_; ++v) {

            // --- Hierarchical moment cascade: high mode -> slope --- //
            for (int k = order_ - 1; k >= 1; --k) {
              const double a_k = u_k_(i, k, v); // mode to limit
              // differences of the next-lower mode in the neighbors
              const double am1_i = u_k_unlimited_(i, k - 1, v);
              const double am1_p = u_k_unlimited_(i + 1, k - 1, v);
              const double am1_m = u_k_unlimited_(i - 1, k - 1, v);
              const double d_p = (am1_p - am1_i) / dist_p;
              const double d_m = (am1_i - am1_m) / dist_m;

              // a_j = 1 / (2 (2j-1)); scale supplies the 1/2.
              const double inv = 1.0 / ((2.0 * k) - 1.0);
              const double new_mode =
                  MINMOD_B(a_k, b_tvd_ * inv * scale * d_p,
                           b_tvd_ * inv * scale * d_m, dr(i), m_tvb_);

              // If this mode was changed, limit it and descend to the next
              // lower mode. Otherwise terminate.
              if (std::abs(new_mode - a_k) > sl_threshold_ * std::abs(a_k)) {
                u_k_(i, k, v) = new_mode;
                limited_cell_(i) = 1;
              } else {
                break;
              }
            } // end cascade k
            // --- End moment cascade --- //

          } // end loop v
        } // end if "limit_this_cell"
      }); // par_for i

  /* Map back to conserved variables */
  if (characteristic_) {
    athelas::par_for(
        DEFAULT_FLAT_LOOP_PATTERN,
        "SlopeLimiter :: Moment :: FromCharacteristic", DevExecSpace(), ilo,
        ihi, KOKKOS_CLASS_LAMBDA(const int i) {
          // --- Characteristic Limiting Matrices ---
          auto R_i = Kokkos::subview(R_, i, Kokkos::ALL, Kokkos::ALL);
          auto U_c_T_i = Kokkos::subview(U_c_T_, i, Kokkos::ALL);
          auto w_c_T_i = Kokkos::subview(w_c_T_, i, Kokkos::ALL);
          for (int k = 0; k < order; ++k) {
            // store U.. = R @ w..
            for (int v = 0; v < nvars; ++v) {
              U_c_T_i(v) = u_k_(i, k, v);
              w_c_T_i(v) = 0.0;
            }
            math::linalg::mat_mul<3>(1.0, R_i, U_c_T_i, 0.0, w_c_T_i);

            for (int v = 0; v < nvars; ++v) {
              u_k_(i, k, v) = w_c_T_i(v);
            } // end loop vars
          } // end loop k
        }); // par_for i
  } // end map from characteristics

  // --- Project back onto nodal basis ---
  basis.modal_to_nodal(U, u_k_, vb_);
} // end apply slope limiter

// limited_cell_ accessor
auto MomentLimiter::get_limited(const int i) const -> int {
  return (!enabled_) ? 0 : limited_cell_(i);
}

auto MomentLimiter::limited() const -> AthelasArray1D<int> {
  return limited_cell_;
}
} // namespace athelas
