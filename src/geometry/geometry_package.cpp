#include "geometry/geometry_package.hpp"
#include "basic_types.hpp"
#include "basis/polynomial_basis.hpp"
#include "geometry/mesh.hpp"
#include "kokkos_abstraction.hpp"
#include "kokkos_types.hpp"
#include "loop_layout.hpp"
#include "pgen/problem_in.hpp"
#include "radiation/rad_utilities.hpp"

namespace athelas::geometry {
using basis::NodalBasis, basis::geometric_weak_eta_derivative;

GeometryPackage::GeometryPackage(const ProblemIn *pin, const int n_stages,
                                 const bool active)
    : active_(active) {
  const int nx = pin->param()->get<int>("problem.nx");
  bool rad_active = pin->param()->get<bool>("physics.radiation.enabled");
  int nvars_geom = 1; // sources velocity
  if (rad_active) {
    nvars_geom++;
  }
  delta_ = AthelasArray4D<double>("geometry delta", n_stages, nx + 2,
                                  pin->param()->get<int>("basis.nnodes"),
                                  nvars_geom);
}

void GeometryPackage::update_explicit(const StageData &stage_data,
                                      const TimeStepInfo &dt_info) {
  const auto &mesh = stage_data.mesh();
  static const int nNodes = mesh.n_nodes();
  static const IndexRange qb(nNodes);
  static const IndexRange ib(mesh.domain<Domain::Interior>());

  auto derived = stage_data.get_field("derived");
  auto evolved = stage_data.get_field("evolved");
  const int idx_tau = stage_data.var_index("evolved", "specific_volume");
  const int idx_pressure = stage_data.var_index("derived", "pressure");
  const auto stage = dt_info.stage;

  auto sqrt_gm = mesh.sqrt_gm();
  auto weights = mesh.weights();
  const auto &basis = stage_data.fluid_basis();
  auto phi = basis.phi();
  auto dphi = basis.dphi();
  auto inv_mkk = basis.inv_mass_matrix();
  athelas::par_for(
      DEFAULT_LOOP_PATTERN, "Geometry :: Velocity", DevExecSpace(), ib.s, ib.e,
      qb.s, qb.e, KOKKOS_CLASS_LAMBDA(const int i, const int q) {
        const double P = derived(i, q + 1, idx_pressure);
        const double geom_source = geometric_weak_eta_derivative(
            phi, dphi, sqrt_gm, weights, i, q, nNodes);
        delta_(stage, i, q, pkg_vars::Velocity) =
            (P * geom_source) * inv_mkk(i, q);
      });

  const bool rad_active = stage_data.enabled("radiation");
  if (rad_active) {
    constexpr double c2 = constants::c_cgs * constants::c_cgs;
    const int idx_rad_energy =
        stage_data.var_index("evolved", "specific_radiation_energy");
    const int idx_rad_flux =
        stage_data.var_index("evolved", "specific_radiation_flux");
    athelas::par_for(
        DEFAULT_LOOP_PATTERN, "Geometry :: RadFlux", DevExecSpace(), ib.s, ib.e,
        qb.s, qb.e, KOKKOS_CLASS_LAMBDA(const int i, const int q) {
          const double rho = 1.0 / evolved(i, q, idx_tau);
          const double e_rad = evolved(i, q, idx_rad_energy) * rho;
          const double f_rad = evolved(i, q, idx_rad_flux) * rho;
          const double p_perp = radiation::p_rad_perp(e_rad, f_rad);
          const double geom_source = geometric_weak_eta_derivative(
              phi, dphi, sqrt_gm, weights, i, q, nNodes);
          delta_(stage, i, q, pkg_vars::RadFlux) =
              (c2 * p_perp * geom_source) * inv_mkk(i, q);
        });
  }
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
  static const int nvars_geom = delta_.extent(3);

  const int stage = dt_info.stage;
  constexpr int idx_vel = 1;
  constexpr int idx_rad_flux = 4;

  athelas::par_for(
      DEFAULT_LOOP_PATTERN, "Geometry :: Apply delta", DevExecSpace(), ib.s,
      ib.e, qb.s, qb.e, KOKKOS_CLASS_LAMBDA(const int i, const int q) {
        lhs(i, q, idx_vel) +=
            dt_info.dt_coef * delta_(stage, i, q, pkg_vars::Velocity);
      });

  // Clunky, but checks for radiation enabled. If more things are sources
  // by geometry later we will need to revisit.
  if (nvars_geom == 2) {
    athelas::par_for(
        DEFAULT_LOOP_PATTERN, "Geometry :: Apply delta :: Radiation",
        DevExecSpace(), ib.s, ib.e, qb.s, qb.e,
        KOKKOS_CLASS_LAMBDA(const int i, const int q) {
          lhs(i, q, idx_rad_flux) +=
              dt_info.dt_coef * delta_(stage, i, q, pkg_vars::RadFlux);
        });
  }
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
