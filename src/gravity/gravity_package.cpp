/**
 * @file gravity_package.cpp
 * --------------
 *
 * @brief Gravitational source package
 **/
#include <limits>

#include "basic_types.hpp"
#include "basis/polynomial_basis.hpp"
#include "geometry/mesh.hpp"
#include "gravity/gravity_package.hpp"
#include "kokkos_abstraction.hpp"
#include "loop_layout.hpp"
#include "pgen/problem_in.hpp"
#include "utils/constants.hpp"

namespace athelas::gravity {

using basis::NodalBasis;

GravityPackage::GravityPackage(const ProblemIn *pin, const std::string &model,
                               const double gval, const double cfl,
                               const int n_stages, const bool active)
    : active_(active), gval_(gval), cfl_(cfl),
      delta_("gravity delta", n_stages,
             pin->param()->get<int>("problem.nx") + 2,
             pin->param()->get<int>("basis.nnodes"), 2) {
  if (model == "constant") {
    model_ = GravityModel::Constant;
  } else if (model == "spherical") {
    model_ = GravityModel::Spherical;
  } else {
    throw_athelas_error("Bad gravity model in GRavityPackage constructor. How "
                        "did this happen?");
  }
}

void GravityPackage::update_explicit(const StageData &stage_data,
                                     const TimeStepInfo &dt_info) const {
  const auto &mesh = stage_data.mesh();
  const auto stage = dt_info.stage;
  auto evolved = stage_data.get_field("evolved");
  const int idx_tau = stage_data.var_index("evolved", "specific_volume");
  const int idx_vel = stage_data.var_index("evolved", "velocity");
  const auto &basis = stage_data.fluid_basis();
  auto inv_mkk = basis.inv_mass_matrix();

  static const IndexRange ib(mesh.domain<Domain::Interior>());
  static const IndexRange qb(mesh.n_nodes());

  if (model_ == GravityModel::Spherical) {
    gravity_update<GravityModel::Spherical>(evolved, mesh, stage, idx_tau,
                                            idx_vel);
  } else [[unlikely]] {
    gravity_update<GravityModel::Constant>(evolved, mesh, stage, idx_tau,
                                           idx_vel);
  }

  athelas::par_for(
      DEFAULT_LOOP_PATTERN, "Gravity :: Apply inverse mass matrix",
      DevExecSpace(), ib.s, ib.e, qb.s, qb.e,
      KOKKOS_CLASS_LAMBDA(const int i, const int q) {
        delta_(stage, i, q, pkg_vars::Velocity) *= inv_mkk(i, q);
        delta_(stage, i, q, pkg_vars::Energy) *= inv_mkk(i, q);
      });
}

template <GravityModel Model>
void GravityPackage::gravity_update(AthelasArray3D<double> evolved,
                                    const Mesh &mesh, const int stage,
                                    const int idx_tau,
                                    const int idx_vel) const {
  static const int nNodes = mesh.n_nodes();
  static const IndexRange ib(mesh.domain<Domain::Interior>());
  static const IndexRange qb(nNodes);

  auto r = mesh.nodal_grid();
  auto dr = mesh.widths();
  auto weights = mesh.weights();
  auto sqrt_gm = mesh.sqrt_gm();
  auto enclosed_mass = mesh.enclosed_mass();

  const double gval = gval_;
  // The spherical 4pi in <S, phi> cancels the common 4pi that would appear in
  // the spherical mass matrix. The stored nodal mass matrix omits that common
  // factor, so omit it here as well.
  athelas::par_for(
      DEFAULT_FLAT_LOOP_PATTERN, "Gravity :: Update", DevExecSpace(), ib.s,
      ib.e, KOKKOS_CLASS_LAMBDA(const int i) {
        const double width = dr(i);
        for (int q = qb.s; q <= qb.e; ++q) {
          const double X = r(i, q + 1);
          const double rho = 1.0 / evolved(i, q, idx_tau);
          const double dvol = weights(q) * sqrt_gm(i, q + 1) * width;

          double accel = 0.0;
          if constexpr (Model == GravityModel::Spherical) {
            accel = -constants::G_GRAV * enclosed_mass(i, q) / (X * X);
          } else {
            accel = -constants::G_GRAV * gval;
          }

          delta_(stage, i, q, pkg_vars::Velocity) = rho * accel * dvol;
          delta_(stage, i, q, pkg_vars::Energy) =
              delta_(stage, i, q, pkg_vars::Velocity) * evolved(i, q, idx_vel);
        }
      });
}

/**
 * @brief apply gravity package delta
 */
void GravityPackage::apply_delta(AthelasArray3D<double> lhs,
                                 const TimeStepInfo &dt_info) const {
  static const int nx = static_cast<int>(lhs.extent(0));
  static const int nq = static_cast<int>(lhs.extent(1));
  static const IndexRange ib(std::make_pair(1, nx - 2));
  static const IndexRange qb(nq);
  static const IndexRange vb(NUM_VARS_);

  const int stage = dt_info.stage;

  athelas::par_for(
      DEFAULT_LOOP_PATTERN, "Gravity :: Apply delta", DevExecSpace(), ib.s,
      ib.e, qb.s, qb.e, KOKKOS_CLASS_LAMBDA(const int i, const int q) {
        for (int v = vb.s; v <= vb.e; ++v) {
          lhs(i, q, v + 1) += dt_info.dt_coef * delta_(stage, i, q, v);
        }
      });
}

/**
 * @brief zero delta field
 */
void GravityPackage::zero_delta() const noexcept {
  static const IndexRange sb(static_cast<int>(delta_.extent(0)));
  static const IndexRange ib(static_cast<int>(delta_.extent(1)));
  static const IndexRange qb(static_cast<int>(delta_.extent(2)));
  static const IndexRange vb(static_cast<int>(delta_.extent(3)));

  athelas::par_for(
      DEFAULT_LOOP_PATTERN, "Gravity :: Zero delta", DevExecSpace(), sb.s, sb.e,
      ib.s, ib.e, qb.s, qb.e,
      KOKKOS_CLASS_LAMBDA(const int s, const int i, const int q) {
        for (int v = vb.s; v <= vb.e; ++v) {
          delta_(s, i, q, v) = 0.0;
        }
      });
}

/**
 * @brief Gravitational timestep restriction
 **/
KOKKOS_FUNCTION
auto GravityPackage::min_timestep(const StageData &stage_data,
                                  const TimeStepInfo & /*dt_info*/) const
    -> double {
  const auto &mesh = stage_data.mesh();
  // static constexpr double MAX_DT = std::numeric_limits<double>::max() /
  // 100.0; static constexpr double dt_out = MAX_DT; return dt_out;
  static constexpr double MAX_DT = std::numeric_limits<double>::max();
  static constexpr double MIN_DT = 100.0 * std::numeric_limits<double>::min();

  static const IndexRange ib(mesh.domain<Domain::Interior>());
  auto dr = mesh.widths();
  auto r = mesh.centers();
  auto m = mesh.enclosed_mass();

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
