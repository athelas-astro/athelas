/**
 * @file hydro_package.cpp
 * --------------
 *
 * @brief Pure hydrodynamics package
 */
#include <limits>

#include "Kokkos_Macros.hpp"
#include "basic_types.hpp"
#include "basis/polynomial_basis.hpp"
#include "bc/boundary_conditions.hpp"
#include "composition/composition.hpp"
#include "eos/eos_variant.hpp"
#include "fluid/fluid_utilities.hpp"
#include "fluid/hydro_package.hpp"
#include "geometry/grid.hpp"
#include "kokkos_abstraction.hpp"
#include "kokkos_types.hpp"
#include "loop_layout.hpp"
#include "pgen/problem_in.hpp"

namespace athelas::fluid {

using basis::ModalBasis, basis::basis_eval;
using eos::EOS;

HydroPackage::HydroPackage(const ProblemIn * /*pin*/, int n_stages, EOS *eos,
                           ModalBasis *basis, BoundaryConditions *bcs,
                           double cfl, int nx, bool active)
    : active_(active), nx_(nx), cfl_(cfl), eos_(eos), basis_(basis), bcs_(bcs),
      dFlux_num_("hydro::dFlux_num_", nx + 2 + 1, 3),
      u_f_l_("hydro::u_f_l_", nx + 2, 3), u_f_r_("hydro::u_f_r_", nx + 2, 3),
      flux_u_("hydro::flux_u_", n_stages + 1, nx + 2 + 1) {
} // Need long term solution for flux_u_

KOKKOS_FUNCTION
void HydroPackage::update_explicit(const State *const state,
                                   AthelasArray3D<double> dU,
                                   const GridStructure &grid,
                                   const TimeStepInfo &dt_info) const {
  const int stage = dt_info.stage;
  const auto u_stages = state->u_cf_stages();

  const auto ucf =
      Kokkos::subview(u_stages, stage, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);

  const auto uaf = state->u_af();

  const auto &order = basis_->order();
  static const IndexRange ib(grid.domain<Domain::Interior>());
  static const IndexRange kb(order);
  static const IndexRange vb(NUM_VARS_);

  // --- Apply BC ---
  bc::fill_ghost_zones<3>(ucf, &grid, basis_, bcs_, {0, 2});

  // --- Zero out dU  ---
  athelas::par_for(
      DEFAULT_LOOP_PATTERN, "Hydro :: Zero dU", DevExecSpace(), ib.s, ib.e,
      kb.s, kb.e, KOKKOS_LAMBDA(const int i, const int k) {
        for (int v = vb.s; v <= vb.e; ++v) {
          dU(i, k, v) = 0.0;
        }
      });

  // --- Fluid Increment : Dvbergence ---
  fluid_divergence(state, dU, grid, stage);

  // --- Dvbide update by mass mastrib ---
  const auto inv_mkk = basis_->inv_mass_matrix();
  athelas::par_for(
      DEFAULT_LOOP_PATTERN, "Hydro :: dU / M_kk", DevExecSpace(), ib.s, ib.e,
      kb.s, kb.e, KOKKOS_CLASS_LAMBDA(const int i, const int k) {
        for (int v = vb.s; v <= vb.e; ++v) {
          dU(i, k, v) *= inv_mkk(i, k);
        }
      });

  // --- Increment from Geometry ---
  if (grid.do_geometry()) {
    fluid_geometry(ucf, uaf, dU, grid);
  }
}

// Compute the dvbergence of the flux term for the update
// TODO(astrobarker): dont pass in stage
KOKKOS_FUNCTION
void HydroPackage::fluid_divergence(const State *const state,
                                    AthelasArray3D<double> dU,
                                    const GridStructure &grid,
                                    const int stage) const {
  const auto u_stages = state->u_cf_stages();

  const auto ucf =
      Kokkos::subview(u_stages, stage, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
  const auto uaf = state->u_af();

  const auto &nNodes = grid.n_nodes();
  const auto &order = basis_->order();
  static const IndexRange ib(grid.domain<Domain::Interior>());
  static const IndexRange kb(order);
  static const IndexRange vb(NUM_VARS_);

  const auto x_l = grid.x_l();
  const auto sqrt_gm = grid.sqrt_gm();
  const auto weights = grid.weights();

  const auto phi = basis_->phi();
  const auto dphis = basis_->dphi();

  // --- Interpolate Conserved Variable to Interfaces ---

  // Left/Right face states
  par_for(
      DEFAULT_FLAT_LOOP_PATTERN, "Hydro :: Interface States", DevExecSpace(),
      ib.s, ib.e + 1, KOKKOS_CLASS_LAMBDA(const int i) {
        for (int v = vb.s; v <= vb.e; ++v) {
          u_f_l_(i, v) = basis_eval(phi, ucf, i - 1, v, nNodes + 1);
          u_f_r_(i, v) = basis_eval(phi, ucf, i, v, 0);
        }
      });

  // --- Calc numerical flux at all faces ---
  par_for(
      DEFAULT_FLAT_LOOP_PATTERN, "Hydro :: Numerical Fluxes", DevExecSpace(),
      ib.s, ib.e + 1, KOKKOS_CLASS_LAMBDA(const int i) {
        const double P_L = uaf(i - 1, nNodes + 1, 0);
        const double Cs_L = uaf(i - 1, nNodes + 1, 2);

        const double P_R = uaf(i, 0, 0);
        const double Cs_R = uaf(i, 0, 2);

        // --- Numerical Fluxes ---

        // Riemann Problem
        // auto [flux_u, flux_p] = numerical_flux_gudonov( u_f_l_(ib,  1 ),
        // u_f_r_(ib,  1
        // ), P_L, P_R, lam_L, lam_R);
        const auto [flux_u, flux_p] = numerical_flux_gudonov_positivity(
            u_f_l_(i, 0), u_f_r_(i, 0), u_f_l_(i, 1), u_f_r_(i, 1), P_L, P_R,
            Cs_L, Cs_R);
        flux_u_(stage, i) = flux_u;

        dFlux_num_(i, 0) = -flux_u;
        dFlux_num_(i, 1) = flux_p;
        dFlux_num_(i, 2) = flux_u * flux_p;
      });

  flux_u_(stage, 0) = flux_u_(stage, 1);
  flux_u_(stage, ib.e + 2) = flux_u_(stage, ib.e + 1);

  // --- Surface Term ---
  athelas::par_for(
      DEFAULT_LOOP_PATTERN, "Hydro :: Surface Term", DevExecSpace(), ib.s, ib.e,
      kb.s, kb.e, KOKKOS_CLASS_LAMBDA(const int i, const int k) {
        for (int v = vb.s; v <= vb.e; ++v) {
          dU(i, k, v) -= (+dFlux_num_(i + 1, v) * phi(i, nNodes + 1, k) *
                              sqrt_gm(i, nNodes + 1) -
                          dFlux_num_(i + 0, v) * phi(i, 0, k) * sqrt_gm(i, 0));
        }
      });

  if (order > 1) [[likely]] {
    // --- Volume Term ---
    athelas::par_for(
        DEFAULT_LOOP_PATTERN, "Hydro :: Volume Term", DevExecSpace(), ib.s,
        ib.e, kb.s, kb.e, KOKKOS_LAMBDA(const int i, const int k) {
          double local_sum1 = 0.0;
          double local_sum2 = 0.0;
          double local_sum3 = 0.0;
          for (int q = 0; q < nNodes; ++q) {
            const double vel = basis_eval(phi, ucf, i, 1, q + 1);
            const double P = uaf(i, q + 1, 0);
            const auto [flux1, flux2, flux3] = flux_fluid(vel, P);
            const double w = weights(q);
            const double dphi = dphis(i, q + 1, k);
            const double sqrtgm = sqrt_gm(i, q + 1);

            local_sum1 += w * flux1 * dphi * sqrtgm;
            local_sum2 += w * flux2 * dphi * sqrtgm;
            local_sum3 += w * flux3 * dphi * sqrtgm;
          }

          dU(i, k, 0) += local_sum1;
          dU(i, k, 1) += local_sum2;
          dU(i, k, 2) += local_sum3;
        });
  }
}

KOKKOS_FUNCTION
void HydroPackage::fluid_geometry(const AthelasArray3D<double> ucf,
                                  const AthelasArray3D<double> uaf,
                                  AthelasArray3D<double> dU,
                                  const GridStructure &grid) const {
  const int &nNodes = grid.n_nodes();
  const int &order = basis_->order();
  static const IndexRange ib(grid.domain<Domain::Interior>());
  static const IndexRange kb(order);

  const auto sqrt_gm = grid.sqrt_gm();
  const auto dx = grid.widths();
  const auto weights = grid.weights();
  const auto position = grid.nodal_grid();
  const auto phi = basis_->phi();
  const auto inv_mkk = basis_->inv_mass_matrix();
  athelas::par_for(
      DEFAULT_LOOP_PATTERN, "Hydro :: Geometry Source", DevExecSpace(), ib.s,
      ib.e, kb.s, kb.e, KOKKOS_CLASS_LAMBDA(const int i, const int k) {
        double local_sum = 0.0;
        for (int q = 0; q < nNodes; ++q) {
          const double P = uaf(i, q + 1, 0);

          local_sum += weights(q) * P * phi(i, q + 1, k) * position(i, q);
        }

        dU(i, k, 1) += (2.0 * local_sum * dx(i)) * inv_mkk(i, k);
      });
}
/**
 * @brief explicit hydrodynamic timestep restriction
 **/
KOKKOS_FUNCTION
auto HydroPackage::min_timestep(const State *const state,
                                const GridStructure &grid,
                                const TimeStepInfo & /*dt_info*/) const
    -> double {
  const auto ucf = state->u_cf();
  static constexpr double MAX_DT = std::numeric_limits<double>::max();
  static constexpr double MIN_DT = 100.0 * std::numeric_limits<double>::min();

  static const IndexRange ib(grid.domain<Domain::Interior>());
  const auto dr = grid.widths();

  double dt_out = 0.0;
  athelas::par_reduce(
      DEFAULT_FLAT_LOOP_PATTERN, "Hydro :: Timestep", DevExecSpace(), ib.s,
      ib.e,
      KOKKOS_CLASS_LAMBDA(const int i, double &lmin) {
        // --- Using Cell Averages ---
        const double tau_x = ucf(i, 0, 0);
        const double vel_x = ucf(i, 0, 1);
        const double eint_x = ucf(i, 0, 2);

        // NOTE: This is not really correct. I'm using a nodal location for
        // getting the ionization terms but cell average quantities for the
        // sound speed. This is only an issue in pure hydro + ionization
        // which should be an edge case.
        // TODO(astrobarker): implement cell averaged Paczynski terms?
        double lambda[8];
        if (state->ionization_enabled()) {
          atom::paczynski_terms(state, i, 1, lambda);
        }
        const double Cs =
            sound_speed_from_conserved(eos_, tau_x, vel_x, eint_x, lambda);
        const double eigval = Cs + std::abs(vel_x);

        const double dt_old = std::abs(dr(i)) / std::abs(eigval);

        lmin = std::min(dt_old, lmin);
      },
      Kokkos::Min<double>(dt_out));

  dt_out = std::max(cfl_ * dt_out, MIN_DT);
  dt_out = std::min(dt_out, MAX_DT);

  return dt_out;
}

/**
 * @brief fill Hydro derived quantities
 */
void HydroPackage::fill_derived(State *const state, const GridStructure &grid,
                                const TimeStepInfo &dt_info) const {
  const int stage = dt_info.stage;

  auto u_s = state->u_cf_stages();

  auto uCF = Kokkos::subview(u_s, stage, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
  // hacky
  if (stage == -1) {
    uCF = state->u_cf();
  }
  auto uPF = state->u_pf();
  auto uAF = state->u_af();

  const int nNodes = grid.n_nodes();
  static const IndexRange ib(grid.domain<Domain::Entire>());
  static const bool ionization_enabled = state->ionization_enabled();

  // --- Apply BC ---
  bc::fill_ghost_zones<3>(uCF, &grid, basis_, bcs_, {0, 2});

  if (state->composition_enabled()) {
    atom::fill_derived_comps(state, &grid, basis_);
  }

  if (ionization_enabled) {
    atom::fill_derived_ionization(state, &grid, basis_);
  }

  const auto phi = basis_->phi();
  athelas::par_for(
      DEFAULT_FLAT_LOOP_PATTERN, "Hydro :: Fill derived", DevExecSpace(), ib.s,
      ib.e, KOKKOS_CLASS_LAMBDA(const int i) {
        for (int q = 0; q < nNodes + 2; ++q) {
          const double tau = basis_eval(phi, uCF, i, 0, q);
          const double vel = basis_eval(phi, uCF, i, 1, q);
          const double emt = basis_eval(phi, uCF, i, 2, q);

          const double rho = 1.0 / tau;
          const double momentum = rho * vel;
          const double sie = (emt - 0.5 * vel * vel);

          double lambda[8];
          // This is probably not the cleanest logic, but setups with
          // ionization enabled and Paczynski disbled are an outlier.
          // Maybe I can do this always?
          if (ionization_enabled) {
            atom::paczynski_terms(state, i, q, lambda);
          }
          const double pressure =
              pressure_from_conserved(eos_, tau, vel, emt, lambda);
          const double t_gas =
              temperature_from_conserved(eos_, tau, vel, emt, lambda);
          const double cs =
              sound_speed_from_conserved(eos_, tau, vel, emt, lambda);

          uPF(i, q, 0) = rho;
          uPF(i, q, 1) = momentum;
          uPF(i, q, 2) = sie;

          uAF(i, q, 0) = pressure;
          uAF(i, q, 1) = t_gas;
          uAF(i, q, 2) = cs;
        }
      });
}

[[nodiscard]] KOKKOS_FUNCTION auto HydroPackage::name() const noexcept
    -> std::string_view {
  return "Hydro";
}

[[nodiscard]] KOKKOS_FUNCTION auto HydroPackage::is_active() const noexcept
    -> bool {
  return active_;
}

KOKKOS_FUNCTION
void HydroPackage::set_active(const bool active) { active_ = active; }

[[nodiscard]] KOKKOS_FUNCTION auto HydroPackage::get_flux_u(const int stage,
                                                            const int i) const
    -> double {
  return flux_u_(stage, i);
}

[[nodiscard]] KOKKOS_FUNCTION auto HydroPackage::basis() const
    -> const ModalBasis * {
  return basis_;
}

} // namespace athelas::fluid
