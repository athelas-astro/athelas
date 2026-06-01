/**
 * @file slope_limiter.hpp
 * --------------
 *
 * @brief Specific slope limiter classes that implement the
 *        SlopeLimiterBase interface
 *
 * @details Defines specific slope limiter implementations that
 *          inherit from the SlopeLimiterBase template class.
 *
 *          We implement the following limiters:
 *          - WENO: Weighted Essentially Non-Oscillatory limiter
 *          - TVDMinmod: Total Variation Diminishing Minmod limiter
 *
 *          Both limiters support:
 *          - Characteristic decomposition
 *          - Troubled Cell Indicator (TCI)
 *
 * TODO(astrobarker): clean up nvars / vars business
 */

#pragma once

#include <variant>

#include "eos/eos_variant.hpp"
#include "interface/state.hpp"
#include "limiters/characteristic_decomposition.hpp"
#include "limiters/slope_limiter_base.hpp"

namespace athelas {

class WENO : public SlopeLimiterBase<WENO> {
 public:
  WENO() = default;
  WENO(const bool enabled, const Mesh *mesh, IndexRange &vb, const int order,
       const double gamma_i, const double gamma_l, const double gamma_r,
       const double weno_p, const bool characteristic, const bool tci_opt,
       const double tci_val)
      : enabled_(enabled), order_(order), nvars_(vb.size()), gamma_i_(gamma_i),
        gamma_l_(gamma_l), gamma_r_(gamma_r), weno_p_(weno_p),
        characteristic_(characteristic), tci_opt_(tci_opt), tci_val_(tci_val),
        vb_(vb), modified_polynomial_("modified_polynomial",
                                      mesh->n_elements() + 2, nvars_, order),
        u_k_("modal coefficients", mesh->n_elements() + 2, order, vb.size()),
        D_("TCI", mesh->n_elements() + 2),
        limited_cell_("LimitedCell", mesh->n_elements() + 2) {
    if (characteristic) {
      R_ = AthelasArray3D<double>("R Matrix", mesh->n_elements() + 2, nvars_,
                                  nvars_);
      R_inv_ = AthelasArray3D<double>("invR Matrix", mesh->n_elements() + 2,
                                      nvars_, nvars_);
      U_c_T_ = AthelasArray2D<double>("U_c_T", mesh->n_elements() + 2, nvars_);
      w_c_T_ = AthelasArray2D<double>("w_c_T", mesh->n_elements() + 2, nvars_);
      mult_ = AthelasArray2D<double>("Mult", mesh->n_elements() + 2, nvars_);
    }
  }

  void apply_slope_limiter(AthelasArray3D<double> U, const Mesh &mesh,
                           const basis::NodalBasis &basis, const eos::EOS &eos,
                           AthelasArray2D<double> lambda_cell);
  [[nodiscard]] auto uses_characteristics() const -> bool {
    return enabled_ && characteristic_ && order_ > 1;
  }
  [[nodiscard]] auto get_limited(int ix) const -> int;
  [[nodiscard]] auto limited() const -> AthelasArray1D<int>;

 private:
  bool enabled_{};
  int order_{};
  int nvars_{};
  double gamma_i_{};
  double gamma_l_{};
  double gamma_r_{};
  double weno_p_{};
  bool characteristic_{};
  bool tci_opt_{};
  double tci_val_{};
  IndexRange vb_;

  AthelasArray3D<double> modified_polynomial_;

  AthelasArray3D<double> R_;
  AthelasArray3D<double> R_inv_;

  // --- Slope limiter quantities ---

  AthelasArray3D<double> u_k_;
  AthelasArray2D<double> U_c_T_;

  // characteristic forms
  AthelasArray2D<double> w_c_T_;

  // matrix mult scratch scape
  AthelasArray2D<double> mult_;

  AthelasArray1D<double> D_;
  AthelasArray1D<int> limited_cell_;
};

class TVDMinmod : public SlopeLimiterBase<TVDMinmod> {
 public:
  TVDMinmod() = default;
  TVDMinmod(const bool enabled, const Mesh *mesh, IndexRange &vb,
            const int order, const double b_tvd, const double m_tvb,
            const bool characteristic, const bool tci_opt, const double tci_val)
      : enabled_(enabled), order_(order), nvars_(vb.size()), b_tvd_(b_tvd),
        m_tvb_(m_tvb), characteristic_(characteristic), tci_opt_(tci_opt),
        tci_val_(tci_val), vb_(vb),
        u_k_("modal coefficients", mesh->n_elements() + 2, order, nvars_),
        D_("TCI", mesh->n_elements() + 2),
        limited_cell_("LimitedCell", mesh->n_elements() + 2) {

    if (characteristic) {
      R_ = AthelasArray3D<double>("R Matrix", mesh->n_elements() + 2, nvars_,
                                  nvars_);
      R_inv_ = AthelasArray3D<double>("invR Matrix", mesh->n_elements() + 2,
                                      nvars_, nvars_);
      mult_ = AthelasArray2D<double>("Mult", mesh->n_elements() + 2, nvars_);
    }
  }
  void apply_slope_limiter(AthelasArray3D<double> U, const Mesh &mesh,
                           const basis::NodalBasis &basis, const eos::EOS &eos,
                           AthelasArray2D<double> lambda_cell);
  [[nodiscard]] auto uses_characteristics() const -> bool {
    return enabled_ && characteristic_ && order_ > 1 &&
           nvars_ <= static_cast<int>(max_characteristic_vars) &&
           order_ <= static_cast<int>(max_characteristic_modes);
  }
  [[nodiscard]] auto get_limited(int ix) const -> int;
  [[nodiscard]] auto limited() const -> AthelasArray1D<int>;

 private:
  bool enabled_{};
  int order_{};
  int nvars_{};
  double b_tvd_{};
  double m_tvb_{};
  bool characteristic_{};
  bool tci_opt_{};
  double tci_val_{};
  IndexRange vb_;

  AthelasArray3D<double> R_;
  AthelasArray3D<double> R_inv_;

  // --- Slope limiter quantities ---

  AthelasArray3D<double> u_k_;

  // matrix mult scratch space (cell-average state for the decomposition)
  AthelasArray2D<double> mult_;

  AthelasArray1D<double> D_;
  AthelasArray1D<int> limited_cell_;
};

// Hierarchical moment limiter (Krivodonova 2007, JCP 226).
// Limits the highest mode first and descends one mode at a time, stopping the
// cascade as soon as a mode is left unchanged. The j=1 (slope) step reduces to
// the TVDMinmod slope limiter, so the same b_tvd / m_tvb controls apply.
class MomentLimiter : public SlopeLimiterBase<MomentLimiter> {
 public:
  MomentLimiter() = default;
  MomentLimiter(const bool enabled, const Mesh *mesh, IndexRange &vb,
                const int order, const double b_tvd, const double m_tvb,
                const bool characteristic, const bool tci_opt,
                const double tci_val)
      : enabled_(enabled), order_(order), nvars_(vb.size()), b_tvd_(b_tvd),
        m_tvb_(m_tvb), characteristic_(characteristic), tci_opt_(tci_opt),
        tci_val_(tci_val), vb_(vb),
        u_k_("modal coefficients", mesh->n_elements() + 2, order, nvars_),
        u_k_unlimited_("unlimited modal coefficients", mesh->n_elements() + 2,
                       order, nvars_),
        D_("TCI", mesh->n_elements() + 2),
        limited_cell_("LimitedCell", mesh->n_elements() + 2) {

    if (characteristic) {
      R_ = AthelasArray3D<double>("R Matrix", mesh->n_elements() + 2, nvars_,
                                  nvars_);
      R_inv_ = AthelasArray3D<double>("invR Matrix", mesh->n_elements() + 2,
                                      nvars_, nvars_);
      mult_ = AthelasArray2D<double>("Mult", mesh->n_elements() + 2, nvars_);
    }
  }
  void apply_slope_limiter(AthelasArray3D<double> U, const Mesh &mesh,
                           const basis::NodalBasis &basis, const eos::EOS &eos,
                           AthelasArray2D<double> lambda_cell);
  [[nodiscard]] auto uses_characteristics() const -> bool {
    return enabled_ && characteristic_ && order_ > 1 &&
           nvars_ <= static_cast<int>(max_characteristic_vars) &&
           order_ <= static_cast<int>(max_characteristic_modes);
  }
  [[nodiscard]] auto get_limited(int ix) const -> int;
  [[nodiscard]] auto limited() const -> AthelasArray1D<int>;

 private:
  bool enabled_{};
  int order_{};
  int nvars_{};
  double b_tvd_{};
  double m_tvb_{};
  bool characteristic_{};
  bool tci_opt_{};
  double tci_val_{};
  IndexRange vb_;

  AthelasArray3D<double> R_;
  AthelasArray3D<double> R_inv_;

  // --- Slope limiter quantities ---

  AthelasArray3D<double> u_k_;
  AthelasArray3D<double> u_k_unlimited_;

  // matrix mult scratch space (cell-average state for the decomposition)
  AthelasArray2D<double> mult_;

  AthelasArray1D<double> D_;
  AthelasArray1D<int> limited_cell_;
};

// A default no-op limiter used when limiting is disabled.
class Unlimited : public SlopeLimiterBase<Unlimited> {
 public:
  Unlimited() = default;
  void apply_slope_limiter(AthelasArray3D<double> U, const Mesh &mesh,
                           const basis::NodalBasis &basis, const eos::EOS &eos,
                           AthelasArray2D<double> lambda_cell);
  [[nodiscard]] auto uses_characteristics() const -> bool { return false; }
  [[nodiscard]] auto get_limited(int ix) const -> int;
  [[nodiscard]] auto limited() const -> AthelasArray1D<int>;

 private:
  AthelasArray1D<int> limited_cell_;
};

using SlopeLimiter = std::variant<WENO, TVDMinmod, MomentLimiter, Unlimited>;

// Fill a per-cell, cell-average EOS lambda for the characteristic
// decomposition. Slot EOS_LAMBDA_TEMPERATURE is always meaningful; ionizing
// runs also fill the Paczynski-specific slots.
void fill_cell_average_lambda(AthelasArray2D<double> lambda_cell,
                              const StageData &stage_data, const Mesh &mesh);

// std::visit functions
// The mesh for the limiter is taken from the stage data (canonical mesh for
// stage 0, the active stage's work buffer otherwise). The basis is passed
// explicitly since the caller selects fluid vs. radiation.
inline void apply_slope_limiter(SlopeLimiter *limiter, AthelasArray3D<double> U,
                                const StageData &stage_data,
                                const basis::NodalBasis &basis,
                                const eos::EOS &eos) {
  const auto &mesh = stage_data.mesh();
  AthelasArray2D<double> lambda_cell;
  const bool needs_lambda = std::visit(
      [](const auto &limiter) { return limiter.uses_characteristics(); },
      *limiter);
  if (needs_lambda) {
    // Cell-average EOS lambda for characteristic limiting. Slot 7 is always
    // filled with temperature; ionization runs also fill Paczynski slots 0--6.
    lambda_cell =
        stage_data.get_field<AthelasArray2D<double>>("eos_lambda_avg");
    fill_cell_average_lambda(lambda_cell, stage_data, mesh);
  }
  std::visit(
      [&U, &mesh, &basis, &eos, &lambda_cell](auto &limiter) {
        limiter.apply_slope_limiter(U, mesh, basis, eos, lambda_cell);
      },
      *limiter);
}
KOKKOS_INLINE_FUNCTION auto get_limited(SlopeLimiter *limiter, const int ix)
    -> int {
  return std::visit([&ix](auto &limiter) { return limiter.get_limited(ix); },
                    *limiter);
}
KOKKOS_INLINE_FUNCTION auto limited(SlopeLimiter *limiter)
    -> AthelasArray1D<int> {
  return std::visit([](auto &limiter) { return limiter.limited(); }, *limiter);
}

} // namespace athelas
