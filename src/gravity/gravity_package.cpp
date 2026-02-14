/**
 * @file gravity_package.cpp
 * --------------
 *
 * @brief Gravitational source package
 **/
#include <limits>

#include "basic_types.hpp"
#include "basis/polynomial_basis.hpp"
#include "utils/constants.hpp"
#include "geometry/grid.hpp"
#include "gravity/gravity_package.hpp"
#include "kokkos_abstraction.hpp"
#include "loop_layout.hpp"
#include "pgen/problem_in.hpp"

namespace athelas::gravity {

using basis::NodalBasis;

GravityPackage::GravityPackage(const ProblemIn *pin, GravityModel model,
                               const double gval, const double cfl,
                               const int n_stages, const bool active)
    : active_(active), model_(model), gval_(gval), cfl_(cfl),
      delta_("gravity delta", n_stages,
             pin->param()->get<int>("problem.nx") + 2,
             pin->param()->get<int>("fluid.porder"), 2) {}

void GravityPackage::update_explicit(const StageData &stage_data,
                                     const GridStructure &grid,
                                     const TimeStepInfo &dt_info) const {
  const auto stage = dt_info.stage;
  auto ucf = stage_data.get_field("u_cf");

  static const IndexRange ib(grid.domain<Domain::Interior>());

  if (model_ == GravityModel::Spherical) {
    gravity_update<GravityModel::Spherical>(ucf, grid, stage);
  } else [[unlikely]] {
    gravity_update<GravityModel::Constant>(ucf, grid, stage);
  }
}

template <GravityModel Model>
void GravityPackage::gravity_update(AthelasArray3D<double> ucf,
                                    const GridStructure &grid, const int stage) const {
  using basis::basis_eval;
  const int nNodes = grid.n_nodes();
  static const IndexRange ib(grid.domain<Domain::Interior>());

  auto r = grid.nodal_grid();
  auto enclosed_mass = grid.enclosed_mass();

  const double gval = gval_;
  // This can probably be simplified.
  // NOTE: the update is divided by 4pi as this factor is weirdly included
  // in enclosed mass but not in, e.g., the mass matrix.
  athelas::par_for(
      DEFAULT_FLAT_LOOP_PATTERN, "Gravity :: Update", DevExecSpace(), ib.s, ib.e,
      KOKKOS_CLASS_LAMBDA(const int i) {
        for (int q = 0; q < nNodes; ++q) {
          const double X = r(i, q + 1);
          const double denom = X * X * constants::FOURPI;
          if constexpr (Model == GravityModel::Spherical) {
            delta_(stage, i, q, pkg_vars::Velocity) = - constants::G_GRAV * enclosed_mass(i, q) / denom;
            delta_(stage, i, q, pkg_vars::Energy) = delta_(stage, i, q, pkg_vars::Velocity) * ucf(i, q, vars::cons::Velocity);
          } else {
            delta_(stage, i, q, pkg_vars::Velocity) = - constants::G_GRAV * gval;
            delta_(stage, i, q, pkg_vars::Energy) = - constants::G_GRAV * gval * ucf(i, q, vars::cons::Velocity);
          }
        }
      });
}

/**
 * @brief apply gravity package delta
 */
void GravityPackage::apply_delta(AthelasArray3D<double> lhs,
                                 const TimeStepInfo &dt_info) const {
  static const int nx = static_cast<int>(lhs.extent(0));
  static const int nk = static_cast<int>(lhs.extent(1));
  static const IndexRange ib(std::make_pair(1, nx - 2));
  static const IndexRange kb(nk);
  static const IndexRange vb(NUM_VARS_);

  const int stage = dt_info.stage;

  athelas::par_for(
      DEFAULT_LOOP_PATTERN, "Gravity :: Apply delta", DevExecSpace(), ib.s,
      ib.e, kb.s, kb.e, KOKKOS_CLASS_LAMBDA(const int i, const int k) {
        for (int v = vb.s; v <= vb.e; ++v) {
          lhs(i, k, v + 1) += dt_info.dt_coef * delta_(stage, i, k, v);
        }
      });
}

/**
 * @brief zero delta field
 */
void GravityPackage::zero_delta() const noexcept {
  static const IndexRange sb(static_cast<int>(delta_.extent(0)));
  static const IndexRange ib(static_cast<int>(delta_.extent(1)));
  static const IndexRange kb(static_cast<int>(delta_.extent(2)));
  static const IndexRange vb(static_cast<int>(delta_.extent(3)));

  athelas::par_for(
      DEFAULT_LOOP_PATTERN, "Gravity :: Zero delta", DevExecSpace(), sb.s, sb.e,
      ib.s, ib.e, kb.s, kb.e,
      KOKKOS_CLASS_LAMBDA(const int s, const int i, const int k) {
        for (int v = vb.s; v <= vb.e; ++v) {
          delta_(s, i, k, v) = 0.0;
        }
      });
}

/**
 * @brief Gravitational timestep restriction
 **/
KOKKOS_FUNCTION
auto GravityPackage::min_timestep(const StageData & /*stage_data*/,
                                  const GridStructure &grid,
                                  const TimeStepInfo & /*dt_info*/) const
    -> double {
  // static constexpr double MAX_DT = std::numeric_limits<double>::max() /
  // 100.0; static constexpr double dt_out = MAX_DT; return dt_out;
  static constexpr double MAX_DT = std::numeric_limits<double>::max();
  static constexpr double MIN_DT = 100.0 * std::numeric_limits<double>::min();

  static const IndexRange ib(grid.domain<Domain::Interior>());
  auto dr = grid.widths();
  auto r = grid.centers();
  auto m = grid.enclosed_mass();

  double dt_out = 0.0;
  athelas::par_reduce(
      DEFAULT_FLAT_LOOP_PATTERN, "Hydro :: Timestep", DevExecSpace(), ib.s,
      ib.e,
      KOKKOS_CLASS_LAMBDA(const int i, double &lmin) {
        const double dt_old =
            std::sqrt((r(i) * r(i) * dr(i)) / (constants::G_GRAV * m(i, 0)));

        lmin = std::min(dt_old, lmin);
      },
      Kokkos::Min<double>(dt_out));

  dt_out = std::max(cfl_ * dt_out, MIN_DT);
  dt_out = std::min(dt_out, MAX_DT);

  return dt_out;
}

void GravityPackage::fill_derived(StageData & /*stage_data*/,
                                  const GridStructure & /*grid*/,
                                  const TimeStepInfo & /*dt_info*/) const {}

[[nodiscard]] KOKKOS_FUNCTION auto GravityPackage::name() const noexcept
    -> std::string_view {
  return "Gravity";
}

[[nodiscard]] KOKKOS_FUNCTION auto GravityPackage::is_active() const noexcept
    -> bool {
  return active_;
}

KOKKOS_FUNCTION
void GravityPackage::set_active(const bool active) { active_ = active; }

} // namespace athelas::gravity
