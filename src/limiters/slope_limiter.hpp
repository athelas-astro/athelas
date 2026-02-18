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
#include "limiters/slope_limiter_base.hpp"

namespace athelas {

class WENO : public SlopeLimiterBase<WENO> {
 public:
  WENO() = default;
  WENO(const bool enabled, const GridStructure *grid,
       const std::vector<int> &vars, const int nvars, const int order,
       const double gamma_i, const double gamma_l, const double gamma_r,
       const double weno_r, const bool characteristic, const bool tci_opt,
       const double tci_val)
      : do_limiter_(enabled), order_(order), nvars_(nvars), gamma_i_(gamma_i),
        gamma_l_(gamma_l), gamma_r_(gamma_r), weno_r_(weno_r),
        characteristic_(characteristic), tci_opt_(tci_opt), tci_val_(tci_val),
        vars_(vars), 
        modified_polynomial_("modified_polynomial",
                                          grid->n_elements() + 2, nvars, order),
        u_k_("modal coefficients", grid->n_elements() + 2, order, nvars),
          D_("TCI", grid->n_elements() + 2),
        limited_cell_("LimitedCell", grid->n_elements() + 2) {
        throw_athelas_error("The WENO slope limiter is not currently working!");
        if (characteristic) {
        R_ = AthelasArray3D<double>("R Matrix", grid->n_elements() + 2, nvars, nvars);
        R_inv_ = AthelasArray3D<double>("invR Matrix", grid->n_elements() + 2, nvars, nvars);
        U_c_T_ = AthelasArray2D<double>("U_c_T", grid->n_elements() + 2, nvars);
        w_c_T_ = AthelasArray2D<double>("w_c_T", grid->n_elements() + 2, nvars);
        mult_ = AthelasArray2D<double>("Mult", grid->n_elements() + 2, nvars);
        }
        }

  void apply_slope_limiter(AthelasArray3D<double> U, const GridStructure *grid,
                           const basis::NodalBasis &basis, const eos::EOS &eos);
  [[nodiscard]] auto get_limited(int ix) const -> int;
  [[nodiscard]] auto limited() const -> AthelasArray1D<int>;

 private:
  bool do_limiter_{};
  int order_{};
  int nvars_{};
  double gamma_i_{};
  double gamma_l_{};
  double gamma_r_{};
  double weno_r_{};
  bool characteristic_{};
  bool tci_opt_{};
  double tci_val_{};
  std::vector<int> vars_;

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
  TVDMinmod(const bool enabled, const GridStructure *grid,
            const std::vector<int> &vars, const int nvars, const int order,
            const double b_tvd, const double m_tvb, const bool characteristic,
            const bool tci_opt, const double tci_val)
      : do_limiter_(enabled), order_(order), nvars_(nvars), b_tvd_(b_tvd),
        m_tvb_(m_tvb), characteristic_(characteristic), tci_opt_(tci_opt),
        tci_val_(tci_val), vars_(vars),
        u_k_("modal coefficients", grid->n_elements() + 2, order, nvars),
          D_("TCI", grid->n_elements() + 2),
        limited_cell_("LimitedCell", grid->n_elements() + 2) {

        if (characteristic) {
        R_ = AthelasArray3D<double>("R Matrix", grid->n_elements() + 2, nvars, nvars);
        R_inv_ = AthelasArray3D<double>("invR Matrix", grid->n_elements() + 2, nvars, nvars);
        U_c_T_ = AthelasArray2D<double>("U_c_T", grid->n_elements() + 2, nvars);
        w_c_T_ = AthelasArray2D<double>("w_c_T", grid->n_elements() + 2, nvars);
        mult_ = AthelasArray2D<double>("Mult", grid->n_elements() + 2, nvars);
        }
        }
  void apply_slope_limiter(AthelasArray3D<double> U, const GridStructure *grid,
                           const basis::NodalBasis &basis, const eos::EOS &eos);
  [[nodiscard]] auto get_limited(int ix) const -> int;
  [[nodiscard]] auto limited() const -> AthelasArray1D<int>;

 private:
  bool do_limiter_{};
  int order_{};
  int nvars_{};
  double b_tvd_{};
  double m_tvb_{};
  bool characteristic_{};
  bool tci_opt_{};
  double tci_val_{};
  std::vector<int> vars_;

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

// A default no-op limiter used when limiting is disabled.
class Unlimited : public SlopeLimiterBase<Unlimited> {
 public:
  Unlimited() = default;
  void apply_slope_limiter(AthelasArray3D<double> U, const GridStructure *grid,
                           const basis::NodalBasis &basis, const eos::EOS &eos);
  [[nodiscard]] auto get_limited(int ix) const -> int;
  [[nodiscard]] auto limited() const -> AthelasArray1D<int>;

 private:
  AthelasArray1D<int> limited_cell_;
};

using SlopeLimiter = std::variant<WENO, TVDMinmod, Unlimited>;

// std::visit functions
inline void apply_slope_limiter(SlopeLimiter *limiter, AthelasArray3D<double> U,
                                const GridStructure *grid,
                                const basis::NodalBasis &basis,
                                const eos::EOS &eos) {
  std::visit(
      [&U, &grid, &basis, &eos](auto &limiter) {
        limiter.apply_slope_limiter(U, grid, basis, eos);
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
