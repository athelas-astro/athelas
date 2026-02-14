#include "geometry/geometry_package.hpp"
#include "basic_types.hpp"
#include "basis/polynomial_basis.hpp"
#include "geometry/grid.hpp"
#include "kokkos_abstraction.hpp"
#include "kokkos_types.hpp"
#include "loop_layout.hpp"
#include "pgen/problem_in.hpp"

namespace athelas::geometry {
using basis::NodalBasis;

GeometryPackage::GeometryPackage(const ProblemIn *pin, const int n_stages,
                                 const bool active)
    : active_(active) {
  const int nx = pin->param()->get<int>("problem.nx");
  int nvars_geom = 1; // sources velocity
  delta_ = AthelasArray4D<double>("geometry delta", n_stages, nx + 2, pin->param()->get<int>("fluid.nnodes"),
                                  nvars_geom);
}

void GeometryPackage::update_explicit(const StageData &stage_data,
                                      const GridStructure &grid,
                                      const TimeStepInfo &dt_info) {
  static const int nNodes = grid.n_nodes();
  static const IndexRange qb(nNodes);
  static const IndexRange ib(grid.domain<Domain::Interior>());

  auto uaf = stage_data.get_field("u_af");
  auto upf = stage_data.get_field("u_pf");
  auto ucf = stage_data.get_field("u_cf");
  const auto stage = dt_info.stage;

  auto r = grid.nodal_grid();
  athelas::par_for(
      DEFAULT_LOOP_PATTERN, "Geometry::Explicit", DevExecSpace(), ib.s,
      ib.e, qb.s, qb.e, KOKKOS_CLASS_LAMBDA(const int i, const int q) {
        const double P = uaf(i, q + 1, vars::aux::Pressure);
        const double tau = ucf(i, q, vars::cons::SpecificVolume);

        delta_(stage, i, q, pkg_vars::Velocity) =
            tau * (2.0 * P / r(i, q + 1));
      });
}

/**
 * @brief apply geometry package delta
 */
void GeometryPackage::apply_delta(AthelasArray3D<double> lhs,
                                  const TimeStepInfo &dt_info) const {
  static const int nx = static_cast<int>(lhs.extent(0));
  static const int nq = static_cast<int>(lhs.extent(1));
  static const IndexRange ib(std::make_pair(1, nx - 2));
  static const IndexRange qb(nq);

  const int stage = dt_info.stage;

  athelas::par_for(
      DEFAULT_LOOP_PATTERN, "Geometry :: Apply delta", DevExecSpace(), ib.s,
      ib.e, qb.s, qb.e, KOKKOS_CLASS_LAMBDA(const int i, const int q) {
        lhs(i, q, vars::cons::Velocity) +=
            dt_info.dt_coef * delta_(stage, i, q, pkg_vars::Velocity);
      });
}

/**
 * @brief zero delta field
 */
void GeometryPackage::zero_delta() const noexcept {
  static const IndexRange sb(static_cast<int>(delta_.extent(0)));
  static const IndexRange ib(static_cast<int>(delta_.extent(1)));
  static const IndexRange qb(static_cast<int>(delta_.extent(2)));
  static const IndexRange vb(static_cast<int>(delta_.extent(3)));

  athelas::par_for(
      DEFAULT_LOOP_PATTERN, "Geometry :: Zero delta", DevExecSpace(), sb.s,
      sb.e, ib.s, ib.e, qb.s, qb.e,
      KOKKOS_CLASS_LAMBDA(const int s, const int i, const int q) {
        for (int v = vb.s; v <= vb.e; ++v) {
          delta_(s, i, q, v) = 0.0;
        }
      });
}

/**
 * @brief geometry timestep restriction
 * We do not enforce a timestep restriction from geometry sources.
 **/
auto GeometryPackage::min_timestep(const StageData & /*state*/,
                                   const GridStructure & /*grid*/,
                                   const TimeStepInfo & /*dt_info*/) const
    -> double {
  static constexpr double MAX_DT = std::numeric_limits<double>::max();
  static constexpr double dt_out = MAX_DT;
  return dt_out;
}

/**
 * @brief geometry package fill derived.
 * no-op at present
 */
void GeometryPackage::fill_derived(StageData & /*state*/,
                                   const GridStructure & /*grid*/,
                                   const TimeStepInfo & /*dt_info*/) const {}

[[nodiscard]] auto GeometryPackage::name() const noexcept -> std::string_view {
  return "Geometry";
}

[[nodiscard]]
auto GeometryPackage::is_active() const noexcept -> bool {
  return active_;
}

void GeometryPackage::set_active(const bool active) { active_ = active; }

} // namespace athelas::geometry
