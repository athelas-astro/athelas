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

using basis::NodalBasis, basis::geometric_weak_eta_derivative;

GravityPackage::GravityPackage(const ProblemIn *pin, const std::string &model,
                               const double gval, const double cfl,
                               const int n_stages, const bool active)
    : active_(active), gval_(gval), cfl_(cfl),
      delta_("gravity delta", n_stages,
             pin->param()->get<int>("problem.nx") + 2,
             pin->param()->get<int>("basis.nnodes"), 2),
      gravity_pressure_("gravity pressure",
                        pin->param()->get<int>("problem.nx") + 2,
                        pin->param()->get<int>("basis.nnodes") + 2) {
  if (model == "constant") {
    model_ = GravityModel::Constant;
  } else if (model == "spherical") {
    model_ = GravityModel::Spherical;
  } else {
    throw_athelas_error("Bad gravity model in GRavityPackage constructor. How "
                        "did this happen?");
  }
}

auto GravityPackage::update_explicit(const StageData &stage_data,
                                     const TimeStepInfo &dt_info) const
    -> UpdateStatus {
  const auto &mesh = stage_data.mesh();
  const auto stage = dt_info.stage;
  auto evolved = stage_data.get_field("evolved");
  const int idx_vel = stage_data.var_index("evolved", "velocity");
  const auto &basis = stage_data.basis();

  if (model_ == GravityModel::Spherical) {
    gravity_update<GravityModel::Spherical>(evolved, mesh, basis, stage,
                                            idx_vel);
  } else [[unlikely]] {
    gravity_update<GravityModel::Constant>(evolved, mesh, basis, stage,
                                           idx_vel);
  }

  return UpdateStatus::Success;
}

template <GravityModel Model>
void GravityPackage::gravity_update(AthelasArray3D<double> evolved,
                                    const Mesh &mesh, const NodalBasis &basis,
                                    const int stage, const int idx_vel) const {
  static const int nNodes = mesh.n_nodes();
  static const IndexRange ib(mesh.domain<Domain::Interior>());
  static const IndexRange qb(nNodes);

  auto r = mesh.nodal_grid();
  auto sqrt_gm = mesh.sqrt_gm();
  auto dm_deta = mesh.dm_deta();
  auto enclosed_mass = mesh.enclosed_mass();
  auto weights = mesh.weights();
  auto phi = basis.phi();
  auto dphi = basis.dphi();
  auto inv_mkk = basis.inv_mass_matrix();
  auto psi_g = gravity_pressure_;

  const double gval = gval_;
  // Build a gravity pressure psi_g satisfying
  //   d_eta psi_g = mu g / A.
  // The balanced pressure operator below approximates -A d_eta(psi_g), so the
  // gravity source is the negative of that operator. This makes hydrostatic
  // states with P = psi_g + const cancel through the same weak operator.
  athelas::par_scan(
      DEFAULT_FLAT_LOOP_PATTERN, "Gravity :: Integrated pressure",
      DevExecSpace(), ib.s, ib.e,
      KOKKOS_CLASS_LAMBDA(const int i, double &psi_left, const bool is_final) {
        // Per-node integrand of d_eta psi_g = mu * accel / A, evaluated once
        // and reused for both the full-cell increment and the within-cell node
        // reconstruction.
        Kokkos::Array<double, MAX_ORDER> g_eta;
        for (int p = 0; p < nNodes; ++p) {
          double accel = 0.0;
          if constexpr (Model == GravityModel::Spherical) {
            const double radius = r(i, p + 1);
            accel = -constants::G_GRAV * enclosed_mass(i, p + 1) /
                    (radius * radius);
          } else {
            accel = -constants::G_GRAV * gval;
          }
          g_eta[p] = dm_deta(i, p) * accel / sqrt_gm(i, p + 1);
        }

        // Full-cell increment feeds the cross-cell prefix sum.
        double dpsi = 0.0;
        for (int q = 0; q < nNodes; ++q) {
          dpsi += weights(q) * g_eta[q];
        }

        if (is_final) {
          psi_g(i, 0) = psi_left;
          for (int q = 0; q < nNodes; ++q) {
            // Partial integral from the left face up to node q.
            double psi_node = psi_left;
            for (int p = 0; p < nNodes; ++p) {
              psi_node += mesh.integration_matrix(q, p) * g_eta[p];
            }
            psi_g(i, q + 1) = psi_node;
          }
        }

        psi_left += dpsi;

        if (is_final) {
          psi_g(i, nNodes + 1) = psi_left;
        }
      });

  athelas::par_for(
      DEFAULT_LOOP_PATTERN, "Gravity :: Update", DevExecSpace(), ib.s, ib.e,
      qb.s, qb.e, KOKKOS_CLASS_LAMBDA(const int i, const int q) {
        // Weak pressure operator applied to psi_g: the flux part
        // (interior - boundary) plus the shared curvature term psi_g * <phi_q,
        // d sqrt_gm/deta>. The hydro pressure flux applies the same operator,
        // so a hydrostatic state P = psi_g + const cancels through it exactly.
        const double boundary = phi(i, nNodes + 1, q) * sqrt_gm(i, nNodes + 1) *
                                    psi_g(i, nNodes + 1) -
                                phi(i, 0, q) * sqrt_gm(i, 0) * psi_g(i, 0);
        double interior = 0.0;
        for (int p = 0; p < nNodes; ++p) {
          interior += weights(p) * dphi(i, p + 1, q) * sqrt_gm(i, p + 1) *
                      psi_g(i, p + 1);
        }
        const double curvature =
            psi_g(i, q + 1) * geometric_weak_eta_derivative(
                                  phi, dphi, sqrt_gm, weights, i, q, nNodes);

        const double pressure_operator = interior - boundary + curvature;
        const double accel = -pressure_operator * inv_mkk(i, q);
        delta_(stage, i, q, pkg_vars::Velocity) = accel;
        delta_(stage, i, q, pkg_vars::Energy) = accel * evolved(i, q, idx_vel);
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
  static constexpr double MAX_DT = std::numeric_limits<double>::max() / 10.0;
  static constexpr double MIN_DT =
      100.0 * std::numeric_limits<double>::min() * 10.0;

  if (model_ == GravityModel::Constant) {
    return MAX_DT;
  }

  const auto &mesh = stage_data.mesh();
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
            std::sqrt((r(i) * r(i) * dr(i)) / (constants::G_GRAV * m(i, 1)));

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
