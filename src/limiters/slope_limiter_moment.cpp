#include "basic_types.hpp"
#include "geometry/mesh.hpp"
#include "kokkos_abstraction.hpp"
#include "kokkos_types.hpp"
#include "limiters/characteristic_decomposition.hpp"
#include "limiters/slope_limiter.hpp"
#include "limiters/slope_limiter_utilities.hpp"
#include "loop_layout.hpp"

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
                                        const NodalBasis &basis, const EOS &eos,
                                        AthelasArray2D<double> lambda_cell) {

  // Do not apply for first order method or if we don't want to.
  if (order_ == 1 || !enabled_) {
    return;
  }

  constexpr static double sl_threshold_ =
      1.0e-4; // TODO(astrobarker): move to input deck

  static constexpr int ilo = 1;
  const int ihi = mesh.get_ihi();

  const int nvars = nvars_;

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

  // The moment stencil reads neighboring lower-order modes. Freeze the modal
  // state (in conserved variables) before limiting so a cell never reads a
  // neighbor that has already been partially limited by the same parallel
  // kernel.
  Kokkos::deep_copy(u_k_unlimited_, u_k_);

  const bool characteristic =
      characteristic_ && nvars <= static_cast<int>(max_characteristic_vars) &&
      order_ <= static_cast<int>(max_characteristic_modes);

  const auto dr = mesh.widths();
  athelas::par_for(
      DEFAULT_FLAT_LOOP_PATTERN, "SlopeLimiter :: Moment", DevExecSpace(), ilo,
      ihi, KOKKOS_CLASS_LAMBDA(const int i) {
        // Do nothing if the TCI says this cell is fine.
        if (tci_opt_ && D_(i) <= tci_val_) {
          return;
        }

        // Element-width factors. Neighbor differences are formed as physical
        // slopes (divide by inter-centroid distance) and scaled by the half
        // width, which keeps the limiter correct on the moving Lagrangian mesh.
        const double scale = 0.5 * dr(i);
        const double dist_p = 0.5 * (dr(i) + dr(i + 1));
        const double dist_m = 0.5 * (dr(i) + dr(i - 1));

        // --- Component-wise cascade (no characteristic transform) --- //
        if (!characteristic) {
          for (int v = 0; v < nvars; ++v) {
            // Hierarchical moment cascade: high mode -> slope.
            for (int k = order_ - 1; k >= 1; --k) {
              const double a_k = u_k_(i, k, v); // mode to limit
              const double am1_i = u_k_unlimited_(i, k - 1, v);
              const double am1_p = u_k_unlimited_(i + 1, k - 1, v);
              const double am1_m = u_k_unlimited_(i - 1, k - 1, v);
              const double d_p = (am1_p - am1_i) / dist_p;
              const double d_m = (am1_i - am1_m) / dist_m;
              const double inv = 1.0 / ((2.0 * k) - 1.0);
              const double new_mode =
                  MINMOD_B(a_k, b_tvd_ * inv * scale * d_p,
                           b_tvd_ * inv * scale * d_m, dr(i), m_tvb_);
              if (std::abs(new_mode - a_k) > sl_threshold_ * std::abs(a_k)) {
                u_k_(i, k, v) = new_mode;
                limited_cell_(i) = 1;
              } else {
                break;
              }
            } // end cascade k
          } // end loop v
          return;
        }

        // --- Characteristic cascade in cell i's eigenbasis --- //
        // Project the frozen stencil (i-1, i, i+1) into cell i's basis so the
        // hierarchical mode differences are formed in one consistent set of
        // characteristic fields (projecting each cell with its own basis would
        // over-limit smooth flow via a spurious ~d(R)/dx term).
        auto R_i = Kokkos::subview(R_, i, Kokkos::ALL, Kokkos::ALL);
        auto R_inv_i = Kokkos::subview(R_inv_, i, Kokkos::ALL, Kokkos::ALL);
        auto avg_i = Kokkos::subview(mult_, i, Kokkos::ALL);
        for (int v = 0; v < nvars; ++v) {
          avg_i(v) = u_k_unlimited_(i, CellAverage, v);
        }
        // Per-cell EOS lambda (slot 7 = temperature; ionizing EOS also reads
        // the cell-average ionization slots).
        compute_characteristic_decomposition(avg_i, R_i, R_inv_i, eos,
                                             &lambda_cell(i, 0));

        // Full-block projection: the cascade reads neighbor mode (k-1), so all
        // modes of the neighbors are needed (unlike TVDMinmod).
        CharBlock w_i;
        CharBlock w_p;
        CharBlock w_m;
        to_characteristic(
            R_inv_i,
            Kokkos::subview(u_k_unlimited_, i, Kokkos::ALL, Kokkos::ALL), w_i,
            order_, nvars);
        to_characteristic(
            R_inv_i,
            Kokkos::subview(u_k_unlimited_, i + 1, Kokkos::ALL, Kokkos::ALL),
            w_p, order_, nvars);
        to_characteristic(
            R_inv_i,
            Kokkos::subview(u_k_unlimited_, i - 1, Kokkos::ALL, Kokkos::ALL),
            w_m, order_, nvars);

        bool any_limited = false;
        for (int m = 0; m < nvars; ++m) {
          // Hierarchical moment cascade in characteristic field m.
          for (int k = order_ - 1; k >= 1; --k) {
            const double a_k = w_i(k, m);
            const double d_p = (w_p(k - 1, m) - w_i(k - 1, m)) / dist_p;
            const double d_m = (w_i(k - 1, m) - w_m(k - 1, m)) / dist_m;
            const double inv = 1.0 / ((2.0 * k) - 1.0);
            const double new_mode =
                MINMOD_B(a_k, b_tvd_ * inv * scale * d_p,
                         b_tvd_ * inv * scale * d_m, dr(i), m_tvb_);
            if (std::abs(new_mode - a_k) > sl_threshold_ * std::abs(a_k)) {
              w_i(k, m) = new_mode;
              any_limited = true;
            } else {
              break;
            }
          } // end cascade k
        } // end loop m

        if (!any_limited) {
          return;
        }

        // Map the limited block back, preserving the cell average exactly.
        auto u_cell_i = Kokkos::subview(u_k_, i, Kokkos::ALL, Kokkos::ALL);
        from_characteristic(R_i, w_i, u_cell_i, order_, nvars);
        for (int v = 0; v < nvars; ++v) {
          u_k_(i, CellAverage, v) = avg_i(v);
        }
        limited_cell_(i) = 1;
      }); // par_for i

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
