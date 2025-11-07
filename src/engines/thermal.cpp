#include "engines/thermal.hpp"
#include "basic_types.hpp"
#include "basis/polynomial_basis.hpp"
#include "compdata.hpp"
#include "geometry/grid.hpp"
#include "kokkos_abstraction.hpp"
#include "kokkos_types.hpp"
#include "loop_layout.hpp"
#include "pgen/problem_in.hpp"
#include "utils/utilities.hpp"

namespace athelas::thermal_engine {
using atom::CompositionData;
using basis::ModalBasis;
using utilities::to_lower;

ThermalEnginePackage::ThermalEnginePackage(const ProblemIn *pin,
                                           ModalBasis *basis, const bool active)
    : active_(active), basis_(basis) {
  // set up heating deposition model
  energy_ = pin->param()->get<double>("physics.engine.thermal.energy");
  mode_ =
      to_lower(pin->param()->get<std::string>("physics.engine.thermal.mode"));
  tend_ = pin->param()->get<double>("physics.engine.thermal.tend");
  mstart_ = pin->param()->get<int>("physics.engine.thermal.mstart");
  mend_ = pin->param()->get<double>("physics.engine.thermal.mend");

  const int nx = pin->param()->get<int>("problem.nx");
  delta_ = AthelasArray3D<double>("nickel delta", nx + 2, basis->order(), 1);
}

void ThermalEnginePackage::update_explicit(const State *const state,
                                           const GridStructure &grid,
                                           const TimeStepInfo &dt_info) {
  const int &order = basis_->order();
  static const IndexRange kb(order);
  static const IndexRange ib(grid.domain<Domain::Interior>());
  const auto u_stages = state->u_cf_stages();

  const auto stage = dt_info.stage;
  const auto ucf =
      Kokkos::subview(u_stages, stage, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);

  // --- Zero out delta  ---
  athelas::par_for(
      DEFAULT_FLAT_LOOP_PATTERN, "ThermalEngine :: Zero delta", DevExecSpace(),
      ib.s, ib.e, KOKKOS_CLASS_LAMBDA(const int i) {
        for (int k = kb.s; k <= kb.e; ++k) {
          delta_(i, k, pkg_vars::Energy) = 0.0;
        }
      });

  // NOTE: We are only applying heating to the cell average currently.
  // Is there are better way to do this?
  athelas::par_for(
      DEFAULT_FLAT_LOOP_PATTERN, "ThermalEngine :: Update", DevExecSpace(),
      ib.s, ib.e, KOKKOS_CLASS_LAMBDA(const int i) {
        const int k = vars::modes::CellAverage;
        delta_(i, k, pkg_vars::Energy) = 0.0;
      });
}

/**
 * @brief apply thermal engine package delta
 */
void ThermalEnginePackage::apply_delta(AthelasArray3D<double> lhs,
                                       const TimeStepInfo &dt_info) const {
  static const int nx = static_cast<int>(lhs.extent(0));
  static const int nk = static_cast<int>(lhs.extent(1));
  static const IndexRange ib(std::make_pair(1, nx - 2));
  static const IndexRange kb(nk);

  athelas::par_for(
      DEFAULT_LOOP_PATTERN, "Thermal engine :: Apply delta", DevExecSpace(),
      ib.s, ib.e, kb.s, kb.e, KOKKOS_CLASS_LAMBDA(const int i, const int k) {
        lhs(i, k, vars::cons::Energy) +=
            dt_info.dt_coef * delta_(i, k, pkg_vars::Energy);
      });
}

/**
 * @brief Thermal engine timestep restriction
 * @note We require that the engine is spread through 1000 timesteps at least.
 **/
auto ThermalEnginePackage::min_timestep(const State *const /*state*/,
                                        const GridStructure & /*grid*/,
                                        const TimeStepInfo & /*dt_info*/) const
    -> double {
  static const double MAX_DT = tend_ / 1000.0;
  static const double dt_out = MAX_DT;
  return dt_out;
}

/**
 * @brief ThermalEngine fill derived
 */
void ThermalEnginePackage::fill_derived(State *state, const GridStructure &grid,
                                        const TimeStepInfo &dt_info) const {}

[[nodiscard]] KOKKOS_FUNCTION auto ThermalEnginePackage::name() const noexcept
    -> std::string_view {
  return "ThermalEngine";
}

[[nodiscard]] KOKKOS_FUNCTION auto
ThermalEnginePackage::is_active() const noexcept -> bool {
  return active_;
}

KOKKOS_FUNCTION
void ThermalEnginePackage::set_active(const bool active) { active_ = active; }

} // namespace athelas::thermal_engine
