#include <limits>

#include "basis/polynomial_basis.hpp"
#include "bc/boundary_conditions.hpp"
#include "composition/composition.hpp"
#include "eos/eos_variant.hpp"
#include "fluid/fluid_utilities.hpp"
#include "geometry/grid.hpp"
#include "kokkos_abstraction.hpp"
#include "loop_layout.hpp"
#include "opacity/opac_variant.hpp"
#include "pgen/problem_in.hpp"
#include "radiation/radhydro_package.hpp"

namespace athelas::radiation {
using basis::ModalBasis, basis::basis_eval;
using eos::EOS;
using fluid::numerical_flux_gudonov_positivity;

RadHydroPackage::RadHydroPackage(const ProblemIn *pin, int n_stages, EOS *eos,
                                 Opacity *opac, ModalBasis *fluid_basis,
                                 ModalBasis *rad_basis, BoundaryConditions *bcs,
                                 double cfl, int nx, bool active)
    : active_(active), nx_(nx), cfl_(cfl), eos_(eos), opac_(opac),
      fluid_basis_(fluid_basis), rad_basis_(rad_basis), bcs_(bcs),
      dFlux_num_("hydro::dFlux_num_", nx + 2 + 1, 5),
      u_f_l_("hydro::u_f_l_", nx + 2, 5), u_f_r_("hydro::u_f_r_", nx + 2, 5),
      flux_u_("hydro::flux_u_", n_stages + 1, nx + 2 + 1),
      scratch_k_("scratch_k_", nx + 2, fluid_basis_->order(), 5),
      scratch_km1_("scratch_km1_", nx + 2, fluid_basis_->order(), 5),
      scratch_sol_("scratch_k_", nx + 2, fluid_basis_->order(), 5) {
} // Need long term solution for flux_u_

void RadHydroPackage::update_explicit(const State *const state,
                                      AthelasArray3D<double> dU,
                                      const GridStructure &grid,
                                      const TimeStepInfo &dt_info) const {
  // TODO(astrobarker) handle separate fluid and rad orders
  const auto &order = fluid_basis_->order();
  static const IndexRange ib(grid.domain<Domain::Interior>());
  static const IndexRange kb(order);
  static const IndexRange vb(NUM_VARS_);

  const auto u_stages = state->u_cf_stages();

  const auto stage = dt_info.stage;
  const auto ucf =
      Kokkos::subview(u_stages, stage, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);

  // --- Apply BC ---
  bc::fill_ghost_zones<2>(ucf, &grid, rad_basis_, bcs_, {3, 4});
  bc::fill_ghost_zones<3>(ucf, &grid, fluid_basis_, bcs_, {0, 2});

  // --- Zero out dU  ---
  athelas::par_for(
      DEFAULT_LOOP_PATTERN, "RadHydro :: Zero dU", DevExecSpace(), ib.s, ib.e,
      kb.s, kb.e, vb.s, vb.e,
      KOKKOS_LAMBDA(const int i, const int k, const int v) {
        dU(i, k, v) = 0.0;
      });

  // --- radiation Increment : Divergence ---
  radhydro_divergence(state, dU, grid, stage);

  // --- Divide update by mass matrix ---
  const auto inv_mkk_fluid = fluid_basis_->inv_mass_matrix();
  const auto inv_mkk_rad = rad_basis_->inv_mass_matrix();
  athelas::par_for(
      DEFAULT_LOOP_PATTERN, "RadHydro :: dU / M_kk", DevExecSpace(), ib.s, ib.e,
      kb.s, kb.e, KOKKOS_CLASS_LAMBDA(const int i, const int k) {
        // Cache mass matri values to avoid repeated lookups
        const double fluid_imm = inv_mkk_fluid(i, k);
        const double rad_imm = inv_mkk_rad(i, k);

        // Process fluid variables (v=0,1,2)
        for (int v = 0; v < 3; ++v) {
          dU(i, k, v) *= fluid_imm;
        }

        // Process radiation variables (v=3,4)
        for (int v = 3; v < NUM_VARS_; ++v) {
          dU(i, k, v) *= rad_imm;
        }
      });

  // --- Increment from Geometry ---
  if (grid.do_geometry()) {
    const auto uaf = state->u_af();
    radhydro_geometry(ucf, uaf, dU, grid);
  }
} // update_explicit

/**
 * @brief radiation hydrodynamic implicit term
 * Computes dU from source terms
 **/
void RadHydroPackage::update_implicit(const State *const state,
                                      AthelasArray3D<double> dU,
                                      const GridStructure &grid,
                                      const TimeStepInfo &dt_info) const {
  // TODO(astrobarker) handle separate fluid and rad orders
  const auto &order = fluid_basis_->order();
  static const IndexRange ib(grid.domain<Domain::Interior>());
  static const IndexRange kb(order);
  static const IndexRange vb(NUM_VARS_);

  const auto u_stages = state->u_cf_stages();

  const auto stage = dt_info.stage;
  const auto ucf =
      Kokkos::subview(u_stages, stage, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);

  // --- Zero out dU  ---
  athelas::par_for(
      DEFAULT_LOOP_PATTERN, "RadHydro :: Implicit :: Zero dU", DevExecSpace(),
      ib.s, ib.e, kb.s, kb.e, KOKKOS_LAMBDA(const int i, const int k) {
        for (int v = vb.s; v <= vb.e; ++v) {
          dU(i, k, v) = 0.0;
        }
      });

  const auto phi_rad = rad_basis_->phi();
  const auto phi_fluid = fluid_basis_->phi();
  const auto inv_mkk_fluid = fluid_basis_->inv_mass_matrix();
  const auto inv_mkk_rad = rad_basis_->inv_mass_matrix();
  const auto dr = grid.widths();
  const auto weights = grid.weights();
  athelas::par_for(
      DEFAULT_LOOP_PATTERN, "RadHydro :: Implicit", DevExecSpace(), ib.s, ib.e,
      kb.s, kb.e, KOKKOS_CLASS_LAMBDA(const int i, const int k) {
        const auto ucf_i = Kokkos::subview(ucf, i, Kokkos::ALL, Kokkos::ALL);
        const auto [du1, du2, du3, du4] =
            radhydro_source(state, ucf_i, dr, weights, phi_fluid, phi_rad,
                            inv_mkk_fluid, inv_mkk_rad, i, k);
        dU(i, k, 1) = du1;
        dU(i, k, 2) = du2;
        dU(i, k, 3) = du3;
        dU(i, k, 4) = du4;
      });
} // update_implicit

void RadHydroPackage::update_implicit_iterative(const State *const state,
                                                AthelasArray3D<double> dU,
                                                const GridStructure &grid,
                                                const TimeStepInfo &dt_info) {
  // TODO(astrobarker) handle separate fluid and rad orders
  const auto &order = fluid_basis_->order();
  static const IndexRange ib(grid.domain<Domain::Interior>());

  const auto u_stages = state->u_cf_stages();

  const auto stage = dt_info.stage;
  const auto ucf =
      Kokkos::subview(u_stages, stage, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);

  const auto phi_rad = rad_basis_->phi();
  const auto phi_fluid = fluid_basis_->phi();
  const auto inv_mkk_fluid = fluid_basis_->inv_mass_matrix();
  const auto inv_mkk_rad = rad_basis_->inv_mass_matrix();
  const auto dr = grid.widths();
  const auto weights = grid.weights();

  athelas::par_for(
      DEFAULT_FLAT_LOOP_PATTERN, "RadHydro :: Implicit iterative",
      DevExecSpace(), ib.s, ib.e, KOKKOS_CLASS_LAMBDA(const int i) {
        const auto ucf_i = Kokkos::subview(ucf, i, Kokkos::ALL, Kokkos::ALL);
        auto scratch_sol_i =
            Kokkos::subview(scratch_sol_, i, Kokkos::ALL, Kokkos::ALL);
        auto scratch_sol_i_k =
            Kokkos::subview(scratch_k_, i, Kokkos::ALL, Kokkos::ALL);
        auto scratch_sol_i_km1 =
            Kokkos::subview(scratch_km1_, i, Kokkos::ALL, Kokkos::ALL);
        const auto R_i = Kokkos::subview(dU, i, Kokkos::ALL, Kokkos::ALL);

        for (int k = 0; k < order; ++k) {
          // set radhydro vars
          for (int v = 0; v < NUM_VARS_; ++v) {
            scratch_sol_i_k(k, v) = ucf_i(k, v);
            scratch_sol_i_km1(k, v) = ucf_i(k, v);
            scratch_sol_i(k, v) = ucf_i(k, v);
          }
        }

        fixed_point_radhydro(R_i, dt_info.dt_a, scratch_sol_i_k,
                             scratch_sol_i_km1, scratch_sol_i, state, dr,
                             weights, phi_fluid, phi_rad, inv_mkk_fluid,
                             inv_mkk_rad, eos_, opac_, i);

        for (int k = 0; k < order; ++k) {
          for (int v = 1; v < NUM_VARS_; ++v) {
            ucf(i, k, v) = scratch_sol_i(k, v);
          }
        }
      });

} // update_implicit_iterative

// Compute the divergence of the flux term for the update
// TODO(astrobarker): dont pass in stage
void RadHydroPackage::radhydro_divergence(const State *const state,
                                          AthelasArray3D<double> dU,
                                          const GridStructure &grid,
                                          const int stage) const {
  const auto u_stages = state->u_cf_stages();

  const auto ucf =
      Kokkos::subview(u_stages, stage, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
  const auto uaf = state->u_af();

  const auto &nNodes = grid.n_nodes();
  const auto &order = rad_basis_->order();
  static constexpr int ilo = 1;
  static const auto &ihi = grid.get_ihi();
  static const IndexRange ib(grid.domain<Domain::Interior>());
  static const IndexRange kb(order);
  static const IndexRange vb(NUM_VARS_);

  const auto sqrt_gm = grid.sqrt_gm();
  const auto weights = grid.weights();

  const auto phi_rad = rad_basis_->phi();
  const auto phi_fluid = fluid_basis_->phi();
  const auto dphi_rad = rad_basis_->dphi();
  const auto dphi_fluid = fluid_basis_->dphi();

  // --- Interpolate Conserved Variable to Interfaces ---

  // Left/Right face states

  athelas::par_for(
      DEFAULT_FLAT_LOOP_PATTERN, "RadHydro :: Interface states", DevExecSpace(),
      ib.s, ib.e + 1, KOKKOS_CLASS_LAMBDA(const int i) {
        const int nnp1 = nNodes + 1;
        for (int v = 0; v < 3; ++v) {
          u_f_l_(i, v) = basis_eval(phi_fluid, ucf, i - 1, v, nnp1);
          u_f_r_(i, v) = basis_eval(phi_fluid, ucf, i, v, 0);
        }
        for (int v = 3; v < NUM_VARS_; ++v) {
          u_f_l_(i, v) = basis_eval(phi_rad, ucf, i - 1, v, nnp1);
          u_f_r_(i, v) = basis_eval(phi_rad, ucf, i, v, 0);
        }
      });

  // --- Calc numerical flux at all faces ---
  athelas::par_for(
      DEFAULT_FLAT_LOOP_PATTERN, "RadHydro :: Numerical fluxes", DevExecSpace(),
      ib.s, ib.e + 1, KOKKOS_CLASS_LAMBDA(const int i) {
        const double Pgas_L = uaf(i - 1, nNodes + 1, 0);
        const double Cs_L = uaf(i - 1, nNodes + 1, 2);

        const double Pgas_R = uaf(i, 0, 0);
        const double Cs_R = uaf(i, 0, 2);

        const double E_L = u_f_l_(i, 3);
        const double F_L = u_f_l_(i, 4);
        const double E_R = u_f_r_(i, 3);
        const double F_R = u_f_r_(i, 4);

        const double Prad_L = compute_closure(E_L, F_L);
        const double Prad_R = compute_closure(E_R, F_R);

        // --- Numerical Fluxes ---
        static constexpr double c2 = constants::c_cgs * constants::c_cgs;

        // Riemann Problem
        // auto [flux_u, flux_p] = numerical_flux_gudonov( u_f_l_(i,  1 ),
        // u_f_r_(i,  1
        // ), P_L, P_R, lam_L, lam_R);
        const auto [flux_u, flux_p] = numerical_flux_gudonov_positivity(
            u_f_l_(i, 0), u_f_r_(i, 0), u_f_l_(i, 1), u_f_r_(i, 1), Pgas_L,
            Pgas_R, Cs_L, Cs_R);
        flux_u_(stage, i) = flux_u;

        const double vstar = flux_u;
        // auto [flux_e, flux_f] =
        //    numerical_flux_hll_rad( E_L, E_R, F_L, F_R, P_L, P_R, vstar );
        const double eddington_factor = Prad_L / E_L;
        const double alpha =
            (constants::c_cgs - vstar) * std::sqrt(eddington_factor);
        const double flux_e = llf_flux(F_R, F_L, E_R, E_L, alpha);
        const double flux_f =
            llf_flux(c2 * Prad_R, c2 * Prad_L, F_R, F_L, alpha);

        const double advective_flux_e =
            (vstar >= 0) ? vstar * E_L : vstar * E_R;
        const double advective_flux_f =
            (vstar >= 0) ? vstar * F_L : vstar * F_R;

        dFlux_num_(i, 0) = -flux_u;
        dFlux_num_(i, 1) = flux_p;
        dFlux_num_(i, 2) = +flux_u * flux_p;

        dFlux_num_(i, 3) = flux_e - advective_flux_e;
        dFlux_num_(i, 4) = flux_f - advective_flux_f;
      });

  flux_u_(stage, ilo - 1) = flux_u_(stage, ilo);
  flux_u_(stage, ihi + 2) = flux_u_(stage, ihi + 1);

  // TODO(astrobarker): Is this pattern for the surface term okay?
  // --- Surface Term ---
  athelas::par_for(
      DEFAULT_LOOP_PATTERN, "RadHydro :: Surface term", DevExecSpace(), ib.s,
      ib.e, kb.s, kb.e, KOKKOS_CLASS_LAMBDA(const int i, const int k) {
        for (int v = vb.s; v <= vb.e; ++v) {
          const auto phi_v = (v < 3) ? phi_fluid : phi_rad;

          dU(i, k, v) -=
              (+dFlux_num_(i + 1, v) * phi_v(i, nNodes + 1, k) *
                   sqrt_gm(i, nNodes + 1) -
               dFlux_num_(i + 0, v) * phi_v(i, 0, k) * sqrt_gm(i, 0));
        }
      });

  if (order > 1) [[likely]] {
    // --- Volume Term ---
    athelas::par_for(
        DEFAULT_LOOP_PATTERN, "RadHydro :: Volume term", DevExecSpace(), ib.s,
        ib.e, kb.s, kb.e, KOKKOS_CLASS_LAMBDA(const int i, const int k) {
          double local_sum1 = 0.0;
          double local_sum2 = 0.0;
          double local_sum3 = 0.0;
          double local_sum_e = 0.0;
          double local_sum_f = 0.0;
          const double vstar = flux_u_(stage, i);
          for (int q = 0; q < nNodes; ++q) {
            const int qp1 = q + 1;

            const double P = uaf(i, qp1, 0);
            const double vel = basis_eval(phi_fluid, ucf, i, 1, qp1);
            const double e_rad = basis_eval(phi_rad, ucf, i, 3, qp1);
            const double f_rad = basis_eval(phi_rad, ucf, i, 4, qp1);
            const double p_rad = compute_closure(e_rad, f_rad);
            const auto [flux1, flux2, flux3] =
                athelas::fluid::flux_fluid(vel, P);
            const auto [flux_e, flux_f] = flux_rad(e_rad, f_rad, p_rad, vstar);
            local_sum1 +=
                weights(q) * flux1 * dphi_fluid(i, qp1, k) * sqrt_gm(i, qp1);
            local_sum2 +=
                weights(q) * flux2 * dphi_fluid(i, qp1, k) * sqrt_gm(i, qp1);
            local_sum3 +=
                weights(q) * flux3 * dphi_fluid(i, qp1, k) * sqrt_gm(i, qp1);
            local_sum_e +=
                weights(q) * flux_e * dphi_rad(i, qp1, k) * sqrt_gm(i, qp1);
            local_sum_f +=
                weights(q) * flux_f * dphi_rad(i, qp1, k) * sqrt_gm(i, qp1);
          }

          dU(i, k, 0) += local_sum1;
          dU(i, k, 1) += local_sum2;
          dU(i, k, 2) += local_sum3;
          dU(i, k, 3) += local_sum_e;
          dU(i, k, 4) += local_sum_f;
        });
  }
} // radhydro_divergence

/**
 * @brief Compute source terms for radiation hydrodynamics system
 * @note Returns tuple<S_egas, S_vgas, S_erad, S_frad>
 **/
auto RadHydroPackage::radhydro_source(
    const State *const state, const AthelasArray2D<double> uCRH,
    const AthelasArray1D<double> dx, const AthelasArray1D<double> weights,
    const AthelasArray3D<double> phi_fluid,
    const AthelasArray3D<double> phi_rad,
    const AthelasArray2D<double> inv_mkk_fluid,
    const AthelasArray2D<double> inv_mkk_rad, const int i, const int k) const
    -> std::tuple<double, double, double, double> {
  return compute_increment_radhydro_source(uCRH, k, state, dx, weights,
                                           phi_fluid, phi_rad, inv_mkk_fluid,
                                           inv_mkk_rad, eos_, opac_, i);
}

/**
 * @brief geometric source terms
 *
 * NOTE: identical to fluid_geometry. Should reduce overlap.
 * TODO(astrobarker): get rid of duplicate code with Hydro
 */
void RadHydroPackage::radhydro_geometry(const AthelasArray3D<double> ucf,
                                        const AthelasArray3D<double> uaf,
                                        AthelasArray3D<double> dU,
                                        const GridStructure &grid) const {
  const int &nNodes = grid.n_nodes();
  const int &order = fluid_basis_->order();
  static const IndexRange ib(grid.domain<Domain::Interior>());
  static const IndexRange kb(order);

  const auto dr = grid.widths();
  const auto weights = grid.weights();
  const auto inv_mkk = fluid_basis_->inv_mass_matrix();
  const auto phi = fluid_basis_->phi();
  const auto r = grid.nodal_grid();

  athelas::par_for(
      DEFAULT_LOOP_PATTERN, "RadHydro :: Geometry source", DevExecSpace(), ib.s,
      ib.e, kb.s, kb.e, KOKKOS_CLASS_LAMBDA(const int i, const int k) {
        double local_sum = 0.0;
        for (int q = 0; q < nNodes; ++q) {
          // /int 2 P r phi
          local_sum +=
              weights(q) * uaf(i, q + 1, 0) * phi(i, q + 1, k) * r(i, q);
        }

        dU(i, k, 1) += 2.0 * local_sum * dr(i) * inv_mkk(i, k);
      });
}

/**
 * @brief explicit radiation hydrodynamic timestep restriction
 **/
auto RadHydroPackage::min_timestep(const State *const /*ucf*/,
                                   const GridStructure &grid,
                                   const TimeStepInfo & /*dt_info*/) const
    -> double {
  static constexpr double MAX_DT = std::numeric_limits<double>::max();
  static constexpr double MIN_DT = 100.0 * std::numeric_limits<double>::min();

  static const IndexRange ib(grid.domain<Domain::Interior>());

  const auto dr = grid.widths();

  double dt_out = 0.0;
  athelas::par_reduce(
      DEFAULT_FLAT_LOOP_PATTERN, "RadHydro :: timestep restriction",
      DevExecSpace(), ib.s, ib.e,
      KOKKOS_CLASS_LAMBDA(const int i, double &lmin) {
        static constexpr double eigval = constants::c_cgs;
        const double dt_old = dr(i) / eigval;

        lmin = std::min(dt_old, lmin);
      },
      Kokkos::Min<double>(dt_out));

  dt_out = std::max(cfl_ * dt_out, MIN_DT);
  dt_out = std::min(dt_out, MAX_DT);

  return dt_out;
}

/**
 * @brief fill RadHydro derived quantities for output
 *
 * TODO(astrobarker): extend
 */
void RadHydroPackage::fill_derived(State *state, const GridStructure &grid,
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

  const auto phi_fluid = fluid_basis_->phi();

  // --- Apply BC ---
  bc::fill_ghost_zones<2>(uCF, &grid, rad_basis_, bcs_, {3, 4});
  bc::fill_ghost_zones<3>(uCF, &grid, fluid_basis_, bcs_, {0, 2});

  if (state->composition_enabled()) {
    atom::fill_derived_comps(state, &grid, fluid_basis_);
  }

  if (ionization_enabled) {
    atom::fill_derived_ionization(state, &grid, fluid_basis_);
  }

  athelas::par_for(
      DEFAULT_FLAT_LOOP_PATTERN, "RadHydro :: fill derived", DevExecSpace(),
      ib.s, ib.e, KOKKOS_CLASS_LAMBDA(const int i) {
        for (int q = 0; q < nNodes + 2; ++q) {
          const double tau = basis_eval(phi_fluid, uCF, i, 0, q);
          const double vel = basis_eval(phi_fluid, uCF, i, 1, q);
          const double emt = basis_eval(phi_fluid, uCF, i, 2, q);

          // const double e_rad = rad_basis_->basis_eval(uCF, i, 3, q + 1);
          // const double f_rad = rad_basis_->basis_eval(uCF, i, 4, q + 1);

          // const double flux_fact = flux_factor(e_rad, f_rad);

          const double rho = 1.0 / tau;
          const double momentum = rho * vel;
          const double sie = (emt - 0.5 * vel * vel);

          // This is probably not the cleanest logic, but setups with
          // ionization enabled and Paczynski disbled are an outlier.
          double lambda[8];
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

[[nodiscard]] auto RadHydroPackage::name() const noexcept -> std::string_view {
  return "RadHydro";
}

[[nodiscard]] auto RadHydroPackage::is_active() const noexcept -> bool {
  return active_;
}

void RadHydroPackage::set_active(const bool active) { active_ = active; }

[[nodiscard]] auto RadHydroPackage::get_flux_u(const int stage,
                                               const int i) const -> double {
  return flux_u_(stage, i);
}

[[nodiscard]] auto RadHydroPackage::fluid_basis() const -> const ModalBasis * {
  return fluid_basis_;
}

[[nodiscard]] auto RadHydroPackage::rad_basis() const -> const ModalBasis * {
  return rad_basis_;
}

} // namespace athelas::radiation
