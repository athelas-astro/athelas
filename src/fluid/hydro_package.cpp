#include "fluid/hydro_package.hpp"
#include "Kokkos_Macros.hpp"
#include "basic_types.hpp"
#include "basis/basis_utilities.hpp"
#include "bc/boundary_conditions.hpp"
#include "composition/composition.hpp"
#include "composition/saha.hpp"
#include "eos/eos_variant.hpp"
#include "fluid/fluid_utilities.hpp"
#include "geometry/mesh.hpp"
#include "kokkos_abstraction.hpp"
#include "kokkos_types.hpp"
#include "loop_layout.hpp"
#include "pgen/problem_in.hpp"
#include <limits>

namespace athelas::fluid {

using basis::NodalBasis;
using eos::EOS;

HydroPackage::HydroPackage(const ProblemIn * /*pin*/, int n_stages, int order,
                           BoundaryConditions *bcs, double cfl, int nx,
                           bool active)
    : active_(active), nx_(nx), cfl_(cfl), bcs_(bcs),
      dFlux_num_("hydro::dFlux_num_", nx + 2 + 1, 3),
      u_f_l_("hydro::u_f_l_", nx + 2, 3), u_f_r_("hydro::u_f_r_", nx + 2, 3),
      delta_("hydro :: delta", n_stages, nx_ + 2, order, 3) {}

auto HydroPackage::update_explicit(const StageData &stage_data,
                                   const TimeStepInfo &dt_info) const
    -> UpdateStatus {
  const auto &mesh = stage_data.mesh();
  const int stage = dt_info.stage;
  auto evolved = stage_data.get_field("evolved");

  auto derived = stage_data.get_field("derived");

  const auto &basis = stage_data.basis();

  static const IndexRange ib(mesh.domain<Domain::Interior>());
  static const IndexRange qb(mesh.n_nodes());
  static const IndexRange vb(NUM_VARS_);

  // --- Refresh evolved halo ---
  bc::ghost_fill(stage_data, bcs_);

  // --- Fluid Increment : Divergence ---
  fluid_divergence(stage_data, mesh, stage);

  // --- Dvbide update by mass mastrix ---
  const auto inv_mkk = basis.inv_mass_matrix();
  athelas::par_for(
      DEFAULT_LOOP_PATTERN, "Hydro :: delta / M_kk", DevExecSpace(), ib.s, ib.e,
      qb.s, qb.e, KOKKOS_CLASS_LAMBDA(const int i, const int q) {
        const double &invmkk = inv_mkk(i, q);
        for (int v = vb.s; v <= vb.e; ++v) {
          delta_(stage, i, q, v) *= invmkk;
        }
      });

  return UpdateStatus::Success;
}

// Compute the dvbergence of the flux term for the update
// TODO(astrobarker): dont pass in stage
void HydroPackage::fluid_divergence(const StageData &stage_data,
                                    const Mesh &mesh, const int stage) const {
  auto evolved = stage_data.get_field("evolved");
  auto derived = stage_data.get_field("derived");
  auto interface = stage_data.get_field<AthelasArray2D<double>>("interface");

  static const int idx_vstar =
      stage_data.var_index("interface", "interface_velocity");

  const auto &basis = stage_data.basis();

  const auto &nNodes = mesh.n_nodes();
  const auto &order = basis.order();
  static const IndexRange ib(mesh.domain<Domain::Interior>());
  static const IndexRange qb(mesh.n_nodes());
  static const IndexRange vb(NUM_VARS_);

  static const int idx_tau = stage_data.var_index("evolved", "specific_volume");
  static const int idx_vel = stage_data.var_index("evolved", "velocity");
  static const int idx_ener =
      stage_data.var_index("evolved", "specific_total_fluid_energy");
  static const int idx_pre = stage_data.var_index("derived", "pressure");
  static const int idx_cs = stage_data.var_index("derived", "sound_speed");

  auto x_l = mesh.x_l();
  auto sqrt_gm = mesh.sqrt_gm();
  auto weights = mesh.weights();

  auto phi = basis.phi();
  auto dphi = basis.dphi();
  const auto fluid_bcs = bc::fluid_bc_data(bcs_);

  // --- Interpolate Conserved Variable to Interfaces ---

  // Left/Right face states
  par_for(
      DEFAULT_FLAT_LOOP_PATTERN, "Hydro :: Interface States", DevExecSpace(),
      ib.s, ib.e + 1, KOKKOS_CLASS_LAMBDA(const int i) {
        for (int v = vb.s; v <= vb.e; ++v) {
          u_f_l_(i, v) = basis.basis_eval(evolved, i - 1, v, nNodes + 1);
          u_f_r_(i, v) = basis.basis_eval(evolved, i, v, 0);
        }
      });

  // --- Calc numerical flux at all faces ---
  par_for(
      DEFAULT_FLAT_LOOP_PATTERN, "Hydro :: Numerical Fluxes", DevExecSpace(),
      ib.s, ib.e + 1, KOKKOS_CLASS_LAMBDA(const int i) {
        const double P_L = derived(i - 1, nNodes + 1, idx_pre);
        const double Cs_L = derived(i - 1, nNodes + 1, idx_cs);

        const double P_R = derived(i, 0, idx_pre);
        const double Cs_R = derived(i, 0, idx_cs);

        // --- Numerical Fluxes ---

        // Riemann Problem
        // auto [flux_u, flux_p] = numerical_flux_gudonov( u_f_l_(ib,  1 ),
        // u_f_r_(ib,  1
        // ), P_L, P_R, lam_L, lam_R);
        const FluidRiemannState left{.tau = u_f_l_(i, idx_tau),
                                     .v = u_f_l_(i, idx_vel),
                                     .p = P_L,
                                     .cs = Cs_L};
        const FluidRiemannState right{.tau = u_f_r_(i, idx_tau),
                                      .v = u_f_r_(i, idx_vel),
                                      .p = P_R,
                                      .cs = Cs_R};
        const auto flux = numerical_flux_fluid_with_boundary(
            i, ib.s, ib.e + 1, fluid_bcs, left, right);
        const double flux_u = flux.u;
        const double flux_p = flux.p;

        if (i == ib.s) {
          interface(0, idx_vstar) = flux_u;
        } else if (i == ib.e + 1) {
          interface(ib.e + 2, idx_vstar) = flux_u;
        }

        interface(i, idx_vstar) = flux_u;

        dFlux_num_(i, idx_tau) = -flux_u;
        dFlux_num_(i, idx_vel) = flux_p;
        dFlux_num_(i, idx_ener) = flux_u * flux_p;
      });

  // --- Surface Term ---
  basis::surface_term(delta_, dFlux_num_, phi, sqrt_gm, stage, nNodes, ib, qb,
                      vb, "Hydro :: Surface Term");

  if (order > 1) [[likely]] {
    // --- Volume Term ---
    basis::volume_term<NUM_VARS_>(
        delta_, dphi, weights, sqrt_gm, stage, nNodes, ib, qb,
        KOKKOS_LAMBDA(const int i, const int q)
            ->Kokkos::Array<double, NUM_VARS_> {
              const double vel = evolved(i, q, idx_vel);
              const double P = derived(i, q + 1, idx_pre);
              const auto [flux1, flux2, flux3] = flux_fluid(vel, P);
              return {flux1, flux2, flux3};
            },
        "Hydro :: Volume Term");
  }
}

/**
 * @brief apply fluid package delta
 */
void HydroPackage::apply_delta(AthelasArray3D<double> lhs,
                               const TimeStepInfo &dt_info) const {
  static const int nx = static_cast<int>(lhs.extent(0));
  static const int nq = static_cast<int>(lhs.extent(1));
  static const IndexRange ib(std::make_pair(1, nx - 2));
  static const IndexRange qb(nq);
  static const IndexRange vb(NUM_VARS_);

  const int stage = dt_info.stage;

  athelas::par_for(
      DEFAULT_LOOP_PATTERN, "Hydro :: Apply delta", DevExecSpace(), ib.s, ib.e,
      qb.s, qb.e, KOKKOS_CLASS_LAMBDA(const int i, const int q) {
        for (int v = vb.s; v <= vb.e; ++v) {
          lhs(i, q, v) += dt_info.dt_coef * delta_(stage, i, q, v);
        }
      });
}

/**
 * @brief zero delta field
 */
void HydroPackage::zero_delta() const noexcept {
  Kokkos::deep_copy(delta_, 0.0);
}

/**
 * @brief explicit hydrodynamic timestep restriction
 **/
auto HydroPackage::min_timestep(const StageData &stage_data,
                                const TimeStepInfo &dt_info) const -> double {
  const auto &mesh = stage_data.mesh();
  static constexpr double MAX_DT = std::numeric_limits<double>::max();
  static constexpr double MIN_DT = 100.0 * std::numeric_limits<double>::min();

  auto derived = stage_data.get_field("derived");
  const int idx_cs = stage_data.var_index("derived", "sound_speed");

  static const int nnodes = mesh.n_nodes();
  static const IndexRange ib(mesh.domain<Domain::Interior>());
  auto dr = mesh.widths();

  double dt_out = 0.0;
  athelas::par_reduce(
      DEFAULT_FLAT_LOOP_PATTERN, "Hydro :: Timestep", DevExecSpace(), ib.s,
      ib.e,
      KOKKOS_CLASS_LAMBDA(const int i, double &lmin) {
        // Find the max sound speed across the element including the interfaces
        double Cs = derived(i, 0, idx_cs);
        for (int q = 1; q <= nnodes; ++q) {
          Cs = std::max(Cs, derived(i, q, idx_cs));
        }

        const double dt = dr(i) / Cs;

        lmin = std::min(dt, lmin);
      },
      Kokkos::Min<double>(dt_out));

  dt_out = std::max(cfl_ * dt_out, MIN_DT);
  dt_out = std::min(dt_out, MAX_DT);

  return dt_out;
}

/**
 * @brief fill Hydro derived quantities
 * TODO(astrobarker): The if-wrapped kernels are not so nice.
 * It would be nice to write an inner, templated on IonzationPhysics
 * function that deals with this. Has less duplicated code.
 */
void HydroPackage::fill_derived(StageData &stage_data,
                                const TimeStepInfo & /*dt_info*/) const {
  const auto &mesh = stage_data.mesh();
  using eos::EOSLambda;

  auto evolved = stage_data.get_field("evolved");
  auto derived = stage_data.get_field("derived");

  const int idx_tau = stage_data.var_index("evolved", "specific_volume");
  const int idx_vel = stage_data.var_index("evolved", "velocity");
  const int idx_ener =
      stage_data.var_index("evolved", "specific_total_fluid_energy");
  const int idx_rho = stage_data.var_index("derived", "density");
  const int idx_momentum = stage_data.var_index("derived", "momentum_density");
  const int idx_sie =
      stage_data.var_index("derived", "specific_internal_energy");
  const int idx_pressure = stage_data.var_index("derived", "pressure");
  const int idx_tgas = stage_data.var_index("derived", "gas_temperature");
  const int idx_cs = stage_data.var_index("derived", "sound_speed");

  const int nNodes = mesh.n_nodes();
  static const IndexRange ib(mesh.domain<Domain::Interior>());
  static const bool ionization_enabled = stage_data.enabled("ionization");

  const auto &basis = stage_data.basis();

  // --- Refresh evolved halo ---
  bc::ghost_fill(stage_data, bcs_);

  if (stage_data.enabled("composition")) {
    atom::fill_derived_comps(stage_data, &mesh);
  }

  const auto &eos = stage_data.eos();
  auto phi = basis.phi();

  // First we get the temperature from the density and specific internal
  // energy. The ionization case is involved and so this is all done
  // separately. In that case the temperature solve is coupled to a Saha solve.
  if (ionization_enabled) {
    auto *const ionization_state = stage_data.ionization_state();
    if (ionization_state->solver() == atom::SahaSolver::Linear) {
      atom::compute_temperature_with_saha<eos::EOSInversion::Sie,
                                          atom::SahaSolver::Linear>(stage_data,
                                                                    mesh);
    }
    if (ionization_state->solver() == atom::SahaSolver::Log) {
      atom::compute_temperature_with_saha<eos::EOSInversion::Sie,
                                          atom::SahaSolver::Log>(stage_data,
                                                                 mesh);
    }
  } else {
    athelas::par_for(
        DEFAULT_FLAT_LOOP_PATTERN, "Hydro :: Fill derived :: temperature",
        DevExecSpace(), ib.s, ib.e, KOKKOS_CLASS_LAMBDA(const int i) {
          EOSLambda lambda;
          ;
          for (int q = 0; q < nNodes + 2; ++q) {
            const double rho = 1.0 / basis.basis_eval(evolved, i, idx_tau, q);
            const double vel = basis.basis_eval(evolved, i, idx_vel, q);
            const double emt = basis.basis_eval(evolved, i, idx_ener, q);
            const double sie = emt - 0.5 * vel * vel;
            derived(i, q, idx_tgas) =
                temperature_from_density_sie(eos, rho, sie, lambda.ptr());
            derived(i, q, idx_rho) = rho;
            derived(i, q, idx_momentum) = rho * vel;
          }
        });
  }

  if (ionization_enabled) {
    const auto *const comps = stage_data.comps();
    const auto number_density = comps->number_density();
    const auto ye = comps->ye();

    const auto *const ionization_states = stage_data.ionization_state();
    const auto ybar = ionization_states->ybar();
    const auto e_ion_corr = ionization_states->e_ion_corr();
    const auto sigma1 = ionization_states->sigma1();
    const auto sigma2 = ionization_states->sigma2();
    const auto sigma3 = ionization_states->sigma3();
    athelas::par_for(
        DEFAULT_FLAT_LOOP_PATTERN, "Hydro :: Fill derived", DevExecSpace(),
        ib.s, ib.e, KOKKOS_CLASS_LAMBDA(const int i) {
          for (int q = 0; q < nNodes + 2; ++q) {
            const double rho = 1.0 / basis.basis_eval(evolved, i, idx_tau, q);
            const double vel = basis.basis_eval(evolved, i, idx_vel, q);
            const double emt = basis.basis_eval(evolved, i, idx_ener, q);

            const double momentum = rho * vel;
            const double sie = (emt - 0.5 * vel * vel);

            // This is probably not the cleanest logic, but setups with
            // ionization enabled and Paczynski disbled are an outlier.
            eos::EOSLambda lambda;
            lambda.data[0] = number_density(i, q);
            lambda.data[1] = ye(i, q);
            lambda.data[2] = ybar(i, q);
            lambda.data[3] = sigma1(i, q);
            lambda.data[4] = sigma2(i, q);
            lambda.data[5] = sigma3(i, q);
            lambda.data[6] = e_ion_corr(i, q);
            lambda.data[7] = derived(i, q, idx_tgas);
            const double t_gas = derived(i, q, idx_tgas);
            const double pressure = pressure_from_density_temperature(
                eos, rho, t_gas, lambda.ptr());
            const double cs = sound_speed_from_density_temperature_pressure(
                eos, rho, t_gas, pressure, lambda.ptr());

            derived(i, q, idx_rho) = rho;
            derived(i, q, idx_momentum) = momentum;
            derived(i, q, idx_sie) = sie;

            derived(i, q, idx_pressure) = pressure;
            derived(i, q, idx_cs) = cs;
          }
        });
  } else {
    athelas::par_for(
        DEFAULT_FLAT_LOOP_PATTERN, "Hydro :: Fill derived", DevExecSpace(),
        ib.s, ib.e, KOKKOS_CLASS_LAMBDA(const int i) {
          for (int q = 0; q < nNodes + 2; ++q) {
            const double rho = derived(i, q, idx_rho);
            const double vel = basis.basis_eval(evolved, i, idx_vel, q);
            const double emt = basis.basis_eval(evolved, i, idx_ener, q);

            const double sie = (emt - 0.5 * vel * vel);

            // This is probably not the cleanest logic, but setups with
            // ionization enabled and Paczynski disbled are an outlier.
            eos::EOSLambda lambda;
            const double t_gas = derived(i, q, idx_tgas);
            const double pressure = pressure_from_density_temperature(
                eos, rho, t_gas, lambda.ptr());
            const double cs = sound_speed_from_density_temperature_pressure(
                eos, rho, t_gas, pressure, lambda.ptr());

            derived(i, q, idx_sie) = sie;

            derived(i, q, idx_pressure) = pressure;
            derived(i, q, idx_cs) = cs;
          }
        });
  }

  // Copy derived into the ghost cells so boundary flux reads (e.g. periodic)
  // see valid pressure / sound speed there.
  bc::ghost_fill_derived(derived, bcs_);
}

[[nodiscard]] auto HydroPackage::name() const noexcept -> std::string_view {
  return "Hydro";
}

[[nodiscard]] auto HydroPackage::is_active() const noexcept -> bool {
  return active_;
}

void HydroPackage::set_active(const bool active) { active_ = active; }
} // namespace athelas::fluid
