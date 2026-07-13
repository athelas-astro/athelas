#pragma once

#include <algorithm>
#include <cmath>

#include "Kokkos_Macros.hpp"

#include "basic_types.hpp"
#include "basis/polynomial_basis.hpp"
#include "bc/boundary_conditions_base.hpp"
#include "compdata.hpp"
#include "geometry/mesh.hpp"
#include "interface/state.hpp"
#include "pgen/problem_in.hpp"
#include "utils/constants.hpp"

namespace athelas::nickel {

namespace pkg_vars {
constexpr int Energy = 0;
constexpr int Nickel = 1;
constexpr int Cobalt = 2;
constexpr int Iron = 3;
} // namespace pkg_vars

using bc::BoundaryConditions;

// FullTrapping
// Jeffery 1999
enum class NiHeatingModel { FullTrapping, Jeffery };

inline auto parse_model(const std::string &model) -> NiHeatingModel {
  if (model == "full_trapping") {
    return NiHeatingModel::FullTrapping;
  }
  if (model == "jeffery") {
    return NiHeatingModel::Jeffery;
  }
  throw_athelas_error("Unknown nickel heating model!");
}

class NickelHeatingPackage {
 public:
  NickelHeatingPackage(const ProblemIn *pin, const Params *indexer,
                       int n_stages, int nq, bool active = true);

  [[nodiscard]] auto update_explicit(const StageData &stage_data,
                                     const TimeStepInfo &dt_info)
      -> UpdateStatus;

  template <NiHeatingModel Model>
  void ni_update(const StageData &stage_data, atom::CompositionData *comps,
                 const Mesh &mesh, const TimeStepInfo &dt_info) const;

  void apply_delta(AthelasArray3D<double> lhs,
                   const TimeStepInfo &dt_info) const;

  void zero_delta() const noexcept;

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
  KOKKOS_INLINE_FUNCTION auto deposition_function(const int i,
                                                  const int q) const -> double {
    using basis::basis_eval;
    if constexpr (Model == NiHeatingModel::FullTrapping) {
      return 1.0;
    } else if constexpr (Model == NiHeatingModel::Jeffery) {
      // Here we assume that the integral
      // (1/4pi) e^(-tau) dOmega is already done during fill_derived
      // and the results stored in int_etau_domega_
      return jeffery_deposition_fraction(int_etau_domega_(i, q));
    }
  }

  KOKKOS_INLINE_FUNCTION
  static auto jeffery_attenuation(const double signed_tau) -> double {
    return std::isfinite(signed_tau) ? std::exp(std::min(signed_tau, 0.0))
                                     : 0.0;
  }

  KOKKOS_INLINE_FUNCTION
  static auto jeffery_deposition_fraction(const double escape_integral)
      -> double {
    const double f_dep = 1.0 - 0.5 * escape_integral;
    return std::isfinite(f_dep) ? std::clamp(f_dep, 0.0, 1.0) : 0.0;
  }

  // TODO(astrobarker): use fma?
  KOKKOS_INLINE_FUNCTION
  static auto ni_source(const double x_ni, const double x_co,
                        const double f_dep) -> double {
    return eps_nickel(x_ni) * (F_PE_NI_ + F_GM_NI_ * f_dep) +
           eps_cobalt(x_co) * (F_PE_CO_ + F_GM_CO_ * f_dep);
  }

  KOKKOS_FORCEINLINE_FUNCTION
  static auto dtau(const double rho, const double kappa_gamma, const double dz)
      -> double {
    return -rho * kappa_gamma * dz;
  }

  [[nodiscard]] auto min_timestep(const StageData & /*stage_data*/,
                                  const TimeStepInfo & /*dt_info*/) const
      -> double;

  [[nodiscard]] auto name() const noexcept -> std::string_view;

  [[nodiscard]] auto is_active() const noexcept -> bool;

  void fill_derived(StageData &stage_data, const TimeStepInfo &dt_info) const;

  void set_active(bool active);

 private:
  template <NiHeatingModel Model>
  void fill_diagnostic_heating_rate(const StageData &stage_data) const;

  bool active_;
  NiHeatingModel model_;
  AthelasArray3D<double> tau_gamma_; // [nx][node][angle]
  AthelasArray2D<double> int_etau_domega_; // integration of e^-tau dOmega

  AthelasArray4D<double> delta_; // [nstages, nx, nq, nvars]

  // We need to store both composition-local indices and full evolved-state
  // indices for our required species.
  int ind_ni_local_;
  int ind_co_local_;
  int ind_fe_local_;
  int ind_ni_;
  int ind_co_;
  int ind_fe_;

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
  // The following are fractions of decay energy that go into gammas (F_GM_*)
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
