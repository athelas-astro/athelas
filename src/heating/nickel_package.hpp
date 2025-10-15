#pragma once

#include "Kokkos_Macros.hpp"

#include "basic_types.hpp"
#include "basis/polynomial_basis.hpp"
#include "bc/boundary_conditions_base.hpp"
#include "compdata.hpp"
#include "geometry/grid.hpp"
#include "pgen/problem_in.hpp"
#include "state/state.hpp"
#include "utils/constants.hpp"

namespace athelas::nickel {

using bc::BoundaryConditions;

// FullTrapping
// Swartz
// Jeffery 1999
enum class NiHeatingModel { FullTrapping, Swartz, Jeffery };

inline auto parse_model(const std::string &model) -> NiHeatingModel {
  if (model == "full_trapping") {
    return NiHeatingModel::FullTrapping;
  }
  if (model == "jeffery") {
    return NiHeatingModel::Jeffery;
  }
  if (model == "swartz") {
    return NiHeatingModel::Swartz;
  }
  THROW_ATHELAS_ERROR("Unknown nickel heating model!");
}

class NickelHeatingPackage {
 public:
  NickelHeatingPackage(const ProblemIn *pin, basis::ModalBasis *basis,
                       bool active = true);

  void update_explicit(const State *const state, AthelasArray3D<double> dU,
                       const GridStructure &grid, const TimeStepInfo &dt_info);

  template <NiHeatingModel Model>
  void ni_update(const State *const state, atom::CompositionData *comps,
                 AthelasArray3D<double> dU, const GridStructure &grid,
                 const TimeStepInfo &dt_info) const;

  void apply_delta(AthelasArray3D<double> lhs,
                   const TimeStepInfo &dt_info) const;

  // NOTE: E_LAMBDA_NI etc are energy release per gram per second
  KOKKOS_FORCEINLINE_FUNCTION
  static auto eps_nickel_cobalt(const double x_ni, const double x_co)
      -> double {
    return eps_nickel(x_ni) + eps_cobalt(x_co);
  }

  KOKKOS_FORCEINLINE_FUNCTION
  static auto eps_nickel(const double x_ni) -> double {
    return E_LAMBDA_NI * x_ni;
  }

  KOKKOS_FORCEINLINE_FUNCTION
  static auto eps_cobalt(const double x_co) -> double {
    return E_LAMBDA_CO * x_co;
  }

  /**
   * @brief Nickel 56 heating deposition function.
   * @note The function is templated on the NiHeatingModel which selects
   * the deposition function.
   *
   * TODO(astrobarker): I should rename this.
   */
  template <NiHeatingModel Model>
  [[nodiscard]]
  KOKKOS_INLINE_FUNCTION auto
  deposition_function(const AthelasArray3D<double> ucf,
                      const atom::CompositionData *const comps,
                      const AthelasArray3D<double> phi, const int ix,
                      const int node) const -> double {
    using basis::basis_eval;
    if constexpr (Model == NiHeatingModel::FullTrapping) {
      static const auto *const species_indexer = comps->species_indexer();
      static const auto ind_ni = species_indexer->get<int>("ni56");
      static const auto ind_co = species_indexer->get<int>("co56");
      const double x_ni = basis_eval(phi, ucf, ix, ind_ni, node + 1);
      const double x_co = basis_eval(phi, ucf, ix, ind_co, node + 1);
      return eps_nickel_cobalt(x_ni, x_co);
    } else if constexpr (Model == NiHeatingModel::Swartz) {
      THROW_ATHELAS_ERROR("Swartz model not implemented!");
    } else if constexpr (Model == NiHeatingModel::Jeffery) {
      // Here we assume that the integral
      // (1/4pi) e^(-tau) dOmega is already done during fill_derived
      // and the results stored in int_etau_domega_
      const double I = 1.0 - 0.5 * int_etau_domega_(ix, node);

      static const auto *const species_indexer = comps->species_indexer();
      static const auto ind_ni = species_indexer->get<int>("ni56");
      static const auto ind_co = species_indexer->get<int>("co56");
      const double x_ni = basis_eval(phi, ucf, ix, ind_ni, node + 1);
      const double x_co = basis_eval(phi, ucf, ix, ind_co, node + 1);
      return eps_nickel(x_ni) * (F_PE_NI_ + F_GM_NI_ * I) +
             eps_cobalt(x_co) * (F_PE_CO_ + F_GM_CO_ * I);
    }
  }

  KOKKOS_FORCEINLINE_FUNCTION
  static auto dtau(const double rho, const double kappa_gamma, const double dz)
      -> double {
    return -rho * kappa_gamma * dz;
  }

  [[nodiscard]] auto min_timestep(const State * /*state*/,
                                  const GridStructure & /*grid*/,
                                  const TimeStepInfo & /*dt_info*/) const
      -> double;

  [[nodiscard]] auto name() const noexcept -> std::string_view;

  [[nodiscard]] auto is_active() const noexcept -> bool;

  void fill_derived(State *state, const GridStructure &grid,
                    const TimeStepInfo &dt_info) const;

  void set_active(bool active);

 private:
  bool active_;
  NiHeatingModel model_;
  AthelasArray3D<double> tau_gamma_; // [nx][node][angle]
  AthelasArray2D<double> int_etau_domega_; // integration of e^-tau dOmega

  basis::ModalBasis *basis_;

  AthelasArray3D<double> delta_;

  // constants
  static constexpr double TAU_NI_ =
      8.764372373400 * constants::seconds_to_days; // seconds
  static constexpr double LAMBDA_NI_ = 1.0 / TAU_NI_;
  static constexpr double TAU_CO_ =
      111.4 * constants::seconds_to_days; // seconds (113.6?)
  static constexpr double LAMBDA_CO_ = 1.0 / TAU_CO_;
  // These are eps_x * lambda_x and have units of erg/g/s
  static constexpr double E_LAMBDA_NI = 3.94e10; // erg / g / s
  static constexpr double E_LAMBDA_CO = 6.78e9; // erg / g / s

  // Jeffery 1999
  // The following are fractions of decay energy that go into gammas (F_NI_*)
  // and into positrons (F_PE_*).
  static constexpr double F_PE_NI_ = 0.004;
  static constexpr double F_GM_NI_ = 0.996;
  static constexpr double F_PE_CO_ = 0.032;
  static constexpr double F_GM_CO_ = 0.968;
};

KOKKOS_FORCEINLINE_FUNCTION
auto kappa_gamma(const double ye) -> double {
  static constexpr double KAPPA_COEF_ = 0.06; // Swartz gray opacity coef
  return KAPPA_COEF_ * ye;
}

} // namespace athelas::nickel
