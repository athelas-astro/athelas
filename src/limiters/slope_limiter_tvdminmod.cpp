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
 * TVD Minmod limiter. See the Cockburn & Shu papers
 **/
void TVDMinmod::apply_slope_limiter(AthelasArray3D<double> U, const Mesh &mesh,
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
      DEFAULT_FLAT_LOOP_PATTERN, "SlopeLimiter :: Minmod :: Reset indicator",
      DevExecSpace(), ilo, ihi,
      KOKKOS_CLASS_LAMBDA(const int i) { limited_cell_(i) = 0; });

  // --- Apply troubled cell indicator ---
  if (tci_opt_) {
    detect_troubled_cells(U, D_, mesh, basis, vb_);
  }

  // --- Map to modal basis ---
  basis.nodal_to_modal(u_k_, U, vb_);

  const bool characteristic =
      characteristic_ && nvars <= static_cast<int>(max_characteristic_vars) &&
      order_ <= static_cast<int>(max_characteristic_modes);

  const auto dr = mesh.widths();
  athelas::par_for(
      DEFAULT_FLAT_LOOP_PATTERN, "SlopeLimiter :: Minmod", DevExecSpace(), ilo,
      ihi, KOKKOS_CLASS_LAMBDA(const int i) {
        // Do nothing if the troubled-cell indicator says we needn't limit.
        if (tci_opt_ && D_(i) <= tci_val_) {
          return;
        }

        // --- Component-wise limiting (no characteristic transform) --- //
        if (!characteristic) {
          for (int v = 0; v < nvars; ++v) {
            bool changed = false;
            const double new_slope = limit_slope_minmod(
                u_k_(i, Slope, v), u_k_(i - 1, CellAverage, v),
                u_k_(i, CellAverage, v), u_k_(i + 1, CellAverage, v), dr(i),
                dr(i - 1), dr(i + 1), b_tvd_, m_tvb_, sl_threshold_, changed);
            if (changed) {
              u_k_(i, Slope, v) = new_slope;
              for (int k = 2; k < order_; ++k) {
                u_k_(i, k, v) = 0.0;
              }
              limited_cell_(i) = 1;
            }
          }
          return;
        }

        // --- Characteristic limiting in cell i's eigenbasis --- //
        // Build the decomposition from cell i's cell average, then project the
        // *neighbor* data with cell i's left eigenvectors (R_inv_i). Using one
        // common basis for all three cells keeps the field-by-field differences
        // meaningful; projecting each cell with its own basis (a global
        // pre-pass) injects a spurious term ~d(R)/dx that over-limits smooth
        // flow (e.g. rarefaction fans).
        auto R_i = Kokkos::subview(R_, i, Kokkos::ALL, Kokkos::ALL);
        auto R_inv_i = Kokkos::subview(R_inv_, i, Kokkos::ALL, Kokkos::ALL);
        auto avg_i = Kokkos::subview(mult_, i, Kokkos::ALL);
        for (int v = 0; v < nvars; ++v) {
          avg_i(v) = u_k_(i, CellAverage, v);
        }
        // Per-cell EOS lambda (slot 7 = temperature; ionizing EOS also reads
        // the cell-average ionization slots).
        compute_characteristic_decomposition(avg_i, R_i, R_inv_i, eos,
                                             &lambda_cell(i, 0));

        // Project cell i (all modes) and the neighbor averages (mode 0 only)
        // into characteristic variables, all in cell i's basis.
        CharBlock w_i;
        CharBlock w_p;
        CharBlock w_m;
        to_characteristic(R_inv_i,
                          Kokkos::subview(u_k_, i, Kokkos::ALL, Kokkos::ALL),
                          w_i, order_, nvars);
        to_characteristic(
            R_inv_i, Kokkos::subview(u_k_, i + 1, Kokkos::ALL, Kokkos::ALL),
            w_p, 1, nvars);
        to_characteristic(
            R_inv_i, Kokkos::subview(u_k_, i - 1, Kokkos::ALL, Kokkos::ALL),
            w_m, 1, nvars);

        // Minmod each characteristic field independently.
        Kokkos::Array<bool, max_characteristic_vars> field_limited = {false};
        bool any_limited = false;
        for (int m = 0; m < nvars; ++m) {
          bool changed = false;
          const double ns = limit_slope_minmod(
              w_i(Slope, m), w_m(CellAverage, m), w_i(CellAverage, m),
              w_p(CellAverage, m), dr(i), dr(i - 1), dr(i + 1), b_tvd_, m_tvb_,
              sl_threshold_, changed);
          if (changed) {
            w_i(Slope, m) = ns;
            field_limited[m] = true;
            any_limited = true;
          }
        }

        if (!any_limited) {
          return;
        }

        // Drop higher modes of the limited characteristic fields (go linear).
        for (int m = 0; m < nvars; ++m) {
          if (field_limited[m]) {
            for (int k = 2; k < order_; ++k) {
              w_i(k, m) = 0.0;
            }
          }
        }

        // Map the limited block back to conserved variables, preserving the
        // cell average exactly (it is never limited).
        auto u_cell_i = Kokkos::subview(u_k_, i, Kokkos::ALL, Kokkos::ALL);
        from_characteristic(R_i, w_i, u_cell_i, order_, nvars);
        for (int v = 0; v < nvars; ++v) {
          u_k_(i, CellAverage, v) = avg_i(v);
        }
        limited_cell_(i) = 1;
      }); // par_for i

  conservative_correction(u_k_, U, mesh, vb_);

  // --- Project back onto nodal basis ---
  basis.modal_to_nodal(U, u_k_, vb_);
} // end apply slope limiter

// limited_cell_ accessor
auto TVDMinmod::get_limited(const int i) const -> int {
  return (!enabled_) ? 0 : limited_cell_(i);
}

auto TVDMinmod::limited() const -> AthelasArray1D<int> { return limited_cell_; }
} // namespace athelas
