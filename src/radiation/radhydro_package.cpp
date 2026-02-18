#include <limits>

#include "basic_types.hpp"
#include "basis/polynomial_basis.hpp"
#include "bc/boundary_conditions.hpp"
#include "composition/composition.hpp"
#include "composition/saha.hpp"
#include "eos/eos_variant.hpp"
#include "fluid/fluid_utilities.hpp"
#include "geometry/grid.hpp"
#include "kokkos_abstraction.hpp"
#include "loop_layout.hpp"
#include "pgen/problem_in.hpp"
#include "radiation/radhydro_package.hpp"

namespace athelas::radiation {
using basis::NodalBasis, basis::basis_eval;
using eos::EOS;
using fluid::numerical_flux_gudonov_positivity;

RadHydroPackage::RadHydroPackage(const ProblemIn *pin, int n_stages, int nq,
                                 BoundaryConditions *bcs, double cfl, int nx,
                                 bool active)
    : active_(active), cfl_(cfl), bcs_(bcs),
      dFlux_num_("hydro::dFlux_num_", nx + 2 + 1, 5),
      u_f_l_("hydro::u_f_l_", nx + 2, 5), u_f_r_("hydro::u_f_r_", nx + 2, 5),
      flux_u_("hydro::flux_u_", n_stages, nx + 2 + 1),
      delta_("radhydro delta", n_stages, nx + 2, nq, 5),
      delta_im_("radhydro delta implicit", n_stages, nx + 2, nq, 5),
      scratch_k_("scratch_k_", nx + 2, nq, 5),
      scratch_km1_("scratch_km1_", nx + 2, nq, 5),
      scratch_sol_("scratch_k_", nx + 2, nq, 5) {
} // Need long term solution for flux_u_

void RadHydroPackage::update_explicit(const StageData &stage_data,
                                      const GridStructure &grid,
                                      const TimeStepInfo &dt_info) const {
  // TODO(astrobarker) handle separate fluid and rad orders
  const auto &rad_basis = stage_data.rad_basis();
  const auto &fluid_basis = stage_data.fluid_basis();
  static const IndexRange ib(grid.domain<Domain::Interior>());
  static const IndexRange qb(grid.n_nodes());
  static const IndexRange vb(NUM_VARS_);

  const auto stage = dt_info.stage;
  const auto ucf = stage_data.get_field("u_cf");

  // --- Apply BC ---
  bc::fill_ghost_zones<2>(ucf, &grid, rad_basis, bcs_, {3, 4});
  bc::fill_ghost_zones<3>(ucf, &grid, fluid_basis, bcs_, {0, 2});

  // --- radiation Increment : Divergence ---
  radhydro_divergence(stage_data, grid, stage);

  // --- Divide update by mass matrix ---
  const auto inv_mqq_fluid = fluid_basis.inv_mass_matrix();
  const auto inv_mqq_rad = rad_basis.inv_mass_matrix();
  athelas::par_for(
      DEFAULT_LOOP_PATTERN, "RadHydro :: delta / M_qq", DevExecSpace(), ib.s,
      ib.e, qb.s, qb.e, KOKKOS_CLASS_LAMBDA(const int i, const int q) {
        const double &fluid_imm = inv_mqq_fluid(i, q);
        const double &rad_imm = inv_mqq_rad(i, q);

        for (int v = 0; v < 3; ++v) {
          delta_(stage, i, q, v) *= fluid_imm;
        }

        for (int v = 3; v < NUM_VARS_; ++v) {
          delta_(stage, i, q, v) *= rad_imm;
        }
      });
} // update_explicit

/**
 * @brief radiation hydrodynamic implicit term
 * Computes delta from source terms
 **/
void RadHydroPackage::update_implicit(const StageData &stage_data,
                                      const GridStructure &grid,
                                      const TimeStepInfo &dt_info) const {
  // TODO(astrobarker) handle separate fluid and rad orders
  const auto &rad_basis = stage_data.rad_basis();
  const auto &fluid_basis = stage_data.fluid_basis();
  static const IndexRange ib(grid.domain<Domain::Interior>());
  static const IndexRange qb(grid.n_nodes());
  static const IndexRange vb(NUM_VARS_);

  static const bool ionization_enabled = stage_data.ionization_enabled();

  const auto stage = dt_info.stage;
  auto ucf = stage_data.get_field("u_cf");
  auto uaf = stage_data.get_field("u_af");

  const auto &eos = stage_data.eos();
  const auto &opac = stage_data.opac();

  auto phi_rad = rad_basis.phi();
  auto phi_fluid = fluid_basis.phi();
  auto inv_mkk_fluid = fluid_basis.inv_mass_matrix();
  auto inv_mkk_rad = rad_basis.inv_mass_matrix();
  auto dr = grid.widths();
  auto weights = grid.weights();
  auto sqrt_gm = grid.sqrt_gm();

  if (ionization_enabled) {
    const auto *const ionization_state = stage_data.ionization_state();
    const auto *const comps = stage_data.comps();
    auto number_density = comps->number_density();
    auto ye = comps->ye();
    auto ybar = ionization_state->ybar();
    auto sigma1 = ionization_state->sigma1();
    auto sigma2 = ionization_state->sigma2();
    auto sigma3 = ionization_state->sigma3();
    auto e_ion_corr = ionization_state->e_ion_corr();
    RadHydroSolverIonizationContent content{
        number_density, ye, ybar, sigma1, sigma2, sigma3, e_ion_corr};
    athelas::par_for(
        DEFAULT_LOOP_PATTERN, "RadHydro :: Implicit", DevExecSpace(), ib.s,
        ib.e, qb.s, qb.e, KOKKOS_CLASS_LAMBDA(const int i, const int q) {
          const auto ucf_i = Kokkos::subview(ucf, i, q, Kokkos::ALL);
          const auto [du1, du2, du3, du4] =
              compute_increment_radhydro_source_nodal<IonizationPhysics::Active>(
                  ucf_i, uaf, phi_fluid, phi_rad, inv_mkk_fluid, inv_mkk_rad,
                  eos, opac, dr, sqrt_gm, weights, content, i, q);
          delta_im_(stage, i, q, vars::cons::Velocity) = du1;
          delta_im_(stage, i, q, vars::cons::Energy) = du2;
          delta_im_(stage, i, q, vars::cons::RadEnergy) = du3;
          delta_im_(stage, i, q, vars::cons::RadFlux) = du4;
        });
  } else {
    AthelasArray2D<double> number_density;
    AthelasArray2D<double> ye;
    AthelasArray2D<double> ybar;
    AthelasArray2D<double> sigma1;
    AthelasArray2D<double> sigma2;
    AthelasArray2D<double> sigma3;
    AthelasArray2D<double> e_ion_corr;
    RadHydroSolverIonizationContent content{
        number_density, ye, ybar, sigma1, sigma2, sigma3, e_ion_corr};
    const RadHydroSolverIonizationContent test;
    athelas::par_for(
        DEFAULT_LOOP_PATTERN, "RadHydro :: Implicit", DevExecSpace(), ib.s,
        ib.e, qb.s, qb.e, KOKKOS_CLASS_LAMBDA(const int i, const int q) {
          const auto ucf_i = Kokkos::subview(ucf, i, q, Kokkos::ALL);
          const auto [du1, du2, du3, du4] =
              compute_increment_radhydro_source_nodal<IonizationPhysics::Inactive>(
                  ucf_i, uaf, phi_fluid, phi_rad, inv_mkk_fluid, inv_mkk_rad,
                  eos, opac, dr, sqrt_gm, weights, content, i, q);
          delta_im_(stage, i, q, vars::cons::Velocity) = du1;
          delta_im_(stage, i, q, vars::cons::Energy) = du2;
          delta_im_(stage, i, q, vars::cons::RadEnergy) = du3;
          delta_im_(stage, i, q, vars::cons::RadFlux) = du4;
        });
  }
} // update_implicit

void RadHydroPackage::update_implicit_iterative(const StageData &stage_data,
                                                AthelasArray3D<double> R,
                                                const GridStructure &grid,
                                                const TimeStepInfo &dt_info) {
  // TODO(astrobarker) handle separate fluid and rad orders
  const auto &rad_basis = stage_data.rad_basis();
  const auto &fluid_basis = stage_data.fluid_basis();
  static const IndexRange ib(grid.domain<Domain::Interior>());
  static const IndexRange qb(grid.n_nodes());

  static const bool ionization_enabled = stage_data.ionization_enabled();

  auto ucf = stage_data.get_field("u_cf");
  auto uaf = stage_data.get_field("u_af");

  const auto &eos = stage_data.eos();
  const auto &opac = stage_data.opac();

  auto phi_rad = rad_basis.phi();
  auto phi_fluid = fluid_basis.phi();
  auto inv_mkk_fluid = fluid_basis.inv_mass_matrix();
  auto inv_mkk_rad = rad_basis.inv_mass_matrix();
  auto dr = grid.widths();
  auto weights = grid.weights();
  auto sqrt_gm = grid.sqrt_gm();

  if (ionization_enabled) {
    const auto *const ionization_state = stage_data.ionization_state();
    const auto *const comps = stage_data.comps();
    auto number_density = comps->number_density();
    auto ye = comps->ye();
    auto ybar = ionization_state->ybar();
    auto sigma1 = ionization_state->sigma1();
    auto sigma2 = ionization_state->sigma2();
    auto sigma3 = ionization_state->sigma3();
    auto e_ion_corr = ionization_state->e_ion_corr();
    RadHydroSolverIonizationContent content{
        number_density, ye, ybar, sigma1, sigma2, sigma3, e_ion_corr};
    athelas::par_for(
        DEFAULT_LOOP_PATTERN, "RadHydro :: Implicit iterative",
        DevExecSpace(), ib.s, ib.e, qb.s, qb.e, KOKKOS_CLASS_LAMBDA(const int i, const int q) {
          const auto ucf_i = Kokkos::subview(ucf, i, q, Kokkos::ALL);
          auto scratch_sol_i =
              Kokkos::subview(scratch_sol_, i, q, Kokkos::ALL);
          auto scratch_sol_i_k =
              Kokkos::subview(scratch_k_, i, q, Kokkos::ALL);
          auto scratch_sol_i_km1 =
              Kokkos::subview(scratch_km1_, i, q, Kokkos::ALL);
          const auto R_i = Kokkos::subview(R, i, q, Kokkos::ALL);

            // set radhydro vars
            for (int v = 0; v < NUM_VARS_; ++v) {
              const double &u = ucf_i(v);
              scratch_sol_i_k(v) = u;
              scratch_sol_i_km1(v) = u;
              scratch_sol_i(v) = u;
            }

          fixed_point_radhydro<IonizationPhysics::Active>(
              R_i, dt_info.dt_coef, scratch_sol_i_k, scratch_sol_i_km1,
              scratch_sol_i, uaf, phi_fluid, phi_rad, inv_mkk_fluid,
              inv_mkk_rad, eos, opac, dr, sqrt_gm, weights, content, i, q);

            for (int v = 1; v < NUM_VARS_; ++v) {
              ucf(i, q, v) = scratch_sol_i(v);
            }
        });
  } else {
    const RadHydroSolverIonizationContent content;
    athelas::par_for(
        DEFAULT_FLAT_LOOP_PATTERN, "RadHydro :: Implicit iterative",
        DevExecSpace(), ib.s, ib.e, qb.s, qb.e, KOKKOS_CLASS_LAMBDA(const int i, const int q) {
          const auto ucf_i = Kokkos::subview(ucf, i, q, Kokkos::ALL);
          auto scratch_sol_i =
              Kokkos::subview(scratch_sol_, i, q, Kokkos::ALL);
          auto scratch_sol_i_k =
              Kokkos::subview(scratch_k_, i, q, Kokkos::ALL);
          auto scratch_sol_i_km1 =
              Kokkos::subview(scratch_km1_, i, q, Kokkos::ALL);
          const auto R_i = Kokkos::subview(R, i, q, Kokkos::ALL);

            // set radhydro vars
            for (int v = 0; v < NUM_VARS_; ++v) {
              const double &u = ucf_i(v);
              scratch_sol_i_k(v) = u;
              scratch_sol_i_km1(v) = u;
              scratch_sol_i(v) = u;
            }

          fixed_point_radhydro<IonizationPhysics::Inactive>(
              R_i, dt_info.dt_coef, scratch_sol_i_k, scratch_sol_i_km1,
              scratch_sol_i, uaf, phi_fluid, phi_rad, inv_mkk_fluid,
              inv_mkk_rad, eos, opac, dr, sqrt_gm, weights, content, i, q);

            for (int v = 1; v < NUM_VARS_; ++v) {
              ucf(i, q, v) = scratch_sol_i(v);
            }
        });
  }

} // update_implicit_iterative

/**
 * @brief apply rad hydro package delta
 */
void RadHydroPackage::apply_delta(AthelasArray3D<double> lhs,
                                  const TimeStepInfo &dt_info) const {
  static const int nx = static_cast<int>(lhs.extent(0));
  static const int nq = static_cast<int>(lhs.extent(1));
  static const IndexRange ib(std::make_pair(1, nx - 2));
  static const IndexRange qb(nq);
  static const IndexRange vb(NUM_VARS_);

  const int stage = dt_info.stage;

  athelas::par_for(
      DEFAULT_LOOP_PATTERN, "RadHydro :: Apply delta", DevExecSpace(), ib.s,
      ib.e, qb.s, qb.e, KOKKOS_CLASS_LAMBDA(const int i, const int q) {
        for (int v = vb.s; v <= vb.e; ++v) {
          lhs(i, q, v) += dt_info.dt_coef * delta_(stage, i, q, v);
          lhs(i, q, v) += dt_info.dt_coef_implicit * delta_im_(stage, i, q, v);
        }
      });
}

/**
 * @brief zero delta field
 */
void RadHydroPackage::zero_delta() const noexcept {
  static const IndexRange sb(static_cast<int>(delta_.extent(0)));
  static const IndexRange ib(static_cast<int>(delta_.extent(1)));
  static const IndexRange qb(static_cast<int>(delta_.extent(2)));
  static const IndexRange vb(static_cast<int>(delta_.extent(3)));

  athelas::par_for(
      DEFAULT_LOOP_PATTERN, "RadHydro :: Zero delta", DevExecSpace(), sb.s,
      sb.e, ib.s, ib.e, qb.s, qb.e,
      KOKKOS_CLASS_LAMBDA(const int s, const int i, const int q) {
        for (int v = vb.s; v <= vb.e; ++v) {
          delta_(s, i, q, v) = 0.0;
        }
      });
}

// Compute the divergence of the flux term for the update
// TODO(astrobarker): dont pass in stage
void RadHydroPackage::radhydro_divergence(const StageData &stage_data,
                                          const GridStructure &grid,
                                          const int stage) const {
  auto ucf = stage_data.get_field("u_cf");
  auto uaf = stage_data.get_field("u_af");

  const auto &rad_basis = stage_data.rad_basis();
  const auto &fluid_basis = stage_data.fluid_basis();

  const auto &nNodes = grid.n_nodes();
  static constexpr int ilo = 1;
  static const auto &ihi = grid.get_ihi();
  static const IndexRange ib(grid.domain<Domain::Interior>());
  static const IndexRange qb(grid.n_nodes());
  static const IndexRange vb(NUM_VARS_);

  auto sqrt_gm = grid.sqrt_gm();
  auto weights = grid.weights();

  auto phi_rad = rad_basis.phi();
  auto phi_fluid = fluid_basis.phi();
  auto dphi_rad = rad_basis.dphi();
  auto dphi_fluid = fluid_basis.dphi();

  // --- Interpolate Conserved Variable to Interfaces ---

  // Left/Right face states

  athelas::par_for(
      DEFAULT_FLAT_LOOP_PATTERN, "RadHydro :: Interface states", DevExecSpace(),
      ib.s, ib.e + 1, KOKKOS_CLASS_LAMBDA(const int i) {
        for (int v = 0; v < 3; ++v) {
          u_f_l_(i, v) = basis_eval<Interface::Right>(phi_fluid, ucf, i - 1, v);
          u_f_r_(i, v) = basis_eval<Interface::Left>(phi_fluid, ucf, i, v);
        }
        for (int v = 3; v < NUM_VARS_; ++v) {
          u_f_l_(i, v) = basis_eval<Interface::Right>(phi_rad, ucf, i - 1, v);
          u_f_r_(i, v) = basis_eval<Interface::Left>(phi_rad, ucf, i, v);
        }
      });

  // --- Calc numerical flux at all faces ---
  athelas::par_for(
      DEFAULT_FLAT_LOOP_PATTERN, "RadHydro :: Numerical fluxes", DevExecSpace(),
      ib.s, ib.e + 1, KOKKOS_CLASS_LAMBDA(const int i) {
        const double Pgas_L = uaf(i - 1, nNodes + 1, vars::aux::Pressure);
        const double Cs_L = uaf(i - 1, nNodes + 1, vars::aux::Cs);

        const double Pgas_R = uaf(i, 0, vars::aux::Pressure);
        const double Cs_R = uaf(i, 0, vars::aux::Cs);

        const double rhoL = 1.0 / u_f_l_(i, vars::cons::SpecificVolume);
        const double rhoR = 1.0 / u_f_r_(i, vars::cons::SpecificVolume);

        const double E_L = u_f_l_(i, vars::cons::RadEnergy) * rhoL;
        const double F_L = u_f_l_(i, vars::cons::RadFlux) * rhoL;
        const double E_R = u_f_r_(i, vars::cons::RadEnergy) * rhoR;
        const double F_R = u_f_r_(i, vars::cons::RadFlux) * rhoR;

        const double Prad_L = compute_closure(E_L, F_L);
        const double Prad_R = compute_closure(E_R, F_R);

        // --- Numerical Fluxes ---
        static constexpr double c2 = constants::c_cgs * constants::c_cgs;

        // Riemann Problem
        // auto [flux_u, flux_p] = numerical_flux_gudonov( u_f_l_(i,  1 ),
        // u_f_r_(i,  1
        // ), P_L, P_R, lam_L, lam_R);
        const auto [flux_u, flux_p] = numerical_flux_gudonov_positivity(
            u_f_l_(i, vars::cons::SpecificVolume),
            u_f_r_(i, vars::cons::SpecificVolume),
            u_f_l_(i, vars::cons::Velocity), u_f_r_(i, vars::cons::Velocity),
            Pgas_L, Pgas_R, Cs_L, Cs_R);
        flux_u_(stage, i) = flux_u;

        const double vstar = flux_u;
        // auto [flux_e, flux_f] =
        //    numerical_flux_hll_rad( E_L, E_R, F_L, F_R, Prad_L, Prad_R, vstar
        //    );

        const double alpha = rad_wavespeed(E_L, E_R, F_L, F_R, vstar);
        const double flux_e =
            llf_flux(F_R - vstar * E_R, F_L - vstar * E_L, E_R, E_L, alpha);
        const double flux_f =
            llf_flux(c2 * Prad_R - vstar * F_R, c2 * Prad_L - vstar * F_L, F_R,
                     F_L, alpha);

        dFlux_num_(i, vars::cons::SpecificVolume) = -flux_u;
        dFlux_num_(i, vars::cons::Velocity) = flux_p;
        dFlux_num_(i, vars::cons::Energy) = +flux_u * flux_p;

        dFlux_num_(i, vars::cons::RadEnergy) = flux_e;
        dFlux_num_(i, vars::cons::RadFlux) = flux_f;
      });

  flux_u_(stage, ilo - 1) = flux_u_(stage, ilo);
  flux_u_(stage, ihi + 2) = flux_u_(stage, ihi + 1);

  // TODO(astrobarker): Is this pattern for the surface term okay?
  // --- Surface Term ---
  athelas::par_for(
      DEFAULT_LOOP_PATTERN, "RadHydro :: Surface term", DevExecSpace(), ib.s,
      ib.e, qb.s, qb.e, KOKKOS_CLASS_LAMBDA(const int i, const int q) {
        for (int v = vb.s; v <= vb.e; ++v) {
          const auto phi_v = (v < 3) ? phi_fluid : phi_rad;

          delta_(stage, i, q, v) -=
              (+dFlux_num_(i + 1, v) * phi_v(i, nNodes + 1, q) *
                   sqrt_gm(i, nNodes + 1) -
               dFlux_num_(i + 0, v) * phi_v(i, 0, q) * sqrt_gm(i, 0));
        }
      });

  if (nNodes > 1) [[likely]] {
    // --- Volume Term ---
    auto upf = stage_data.get_field("u_pf");
    athelas::par_for(
        DEFAULT_LOOP_PATTERN, "RadHydro :: Volume term", DevExecSpace(), ib.s,
        ib.e, qb.s, qb.e, KOKKOS_CLASS_LAMBDA(const int i, const int p) {
          double local_sum1 = 0.0;
          double local_sum2 = 0.0;
          double local_sum3 = 0.0;
          double local_sum_e = 0.0;
          double local_sum_f = 0.0;
          const double vstar = flux_u_(stage, i);
          for (int q = 0; q < nNodes; ++q) {
            const int qp1 = q + 1;

            const double pressure = uaf(i, qp1, vars::aux::Pressure);
            const double rho = upf(i, q, vars::prim::Rho);
            const double vel = ucf(i, q, vars::cons::Velocity);
            const double e_rad = ucf(i, q, vars::cons::RadEnergy) * rho;
            const double f_rad = ucf(i, q, vars::cons::RadFlux) * rho;
            const double p_rad = compute_closure(e_rad, f_rad);
            const auto [flux1, flux2, flux3] =
                athelas::fluid::flux_fluid(vel, pressure);
            const auto [flux_e, flux_f] = flux_rad(e_rad, f_rad, p_rad, vstar);
	          const double w_dphi_sqrtgm = weights(q) * dphi_fluid(i, qp1, p) * sqrt_gm(i, qp1);
            local_sum1 += w_dphi_sqrtgm * flux1;
            local_sum2 += w_dphi_sqrtgm * flux2;
            local_sum3 += w_dphi_sqrtgm * flux3;
            local_sum_e += w_dphi_sqrtgm * flux_e;
            local_sum_f += w_dphi_sqrtgm * flux_f;
          }

          delta_(stage, i, p, vars::cons::SpecificVolume) += local_sum1;
          delta_(stage, i, p, vars::cons::Velocity) += local_sum2;
          delta_(stage, i, p, vars::cons::Energy) += local_sum3;
          delta_(stage, i, p, vars::cons::RadEnergy) += local_sum_e;
          delta_(stage, i, p, vars::cons::RadFlux) += local_sum_f;
        });
  }
} // radhydro_divergence

/**
 * @brief explicit radiation hydrodynamic timestep restriction
 **/
auto RadHydroPackage::min_timestep(const StageData & /*stage_data*/,
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
 * @brief fill RadHydro derived quantities
 *
 * TODO(astrobarker): extend
 * TODO(astrobarker): The if-wrapped kernels are not so nice.
 * It would be nice to write an inner, templated on IonzationPhysics
 * function that deals with this. Has less duplicated code.
 */
void RadHydroPackage::fill_derived(StageData &stage_data,
                                   const GridStructure &grid,
                                   const TimeStepInfo &dt_info) const {
  auto uCF = stage_data.get_field("u_cf");
  auto uPF = stage_data.get_field("u_pf");
  auto uAF = stage_data.get_field("u_af");

  const auto &rad_basis = stage_data.rad_basis();
  const auto &fluid_basis = stage_data.fluid_basis();

  const auto &eos = stage_data.eos();

  const int nNodes = grid.n_nodes();
  static const IndexRange ib(grid.domain<Domain::Entire>());
  static const bool ionization_enabled = stage_data.ionization_enabled();

  auto phi_fluid = fluid_basis.phi();

  // --- Apply BC ---
  bc::fill_ghost_zones<2>(uCF, &grid, rad_basis, bcs_, {3, 4});
  bc::fill_ghost_zones<3>(uCF, &grid, fluid_basis, bcs_, {0, 2});

  if (stage_data.composition_enabled()) {
    static constexpr int nvars = 5; // non-comps
    // composition boundary condition
    static const IndexRange vb_comps(
        std::make_pair(nvars, stage_data.nvars("u_cf") - 1));
    bc::fill_ghost_zones_composition(uCF, vb_comps);
    atom::fill_derived_comps<Domain::Entire>(stage_data, &grid);
  }

  // First we get the temperature from the density and specific internal
  // energy. The ionization case is involved and so this is all done
  // separately. In that case the temperature solve is coupled to a Saha solve.
  if (ionization_enabled) {
    auto *const ionization_state = stage_data.ionization_state();
    if (ionization_state->solver() == atom::SahaSolver::Linear) {
      atom::compute_temperature_with_saha<
          Domain::Entire, eos::EOSInversion::Sie, atom::SahaSolver::Linear>(
          stage_data, grid);
    }
    if (ionization_state->solver() == atom::SahaSolver::Log) {
      atom::compute_temperature_with_saha<
          Domain::Entire, eos::EOSInversion::Sie, atom::SahaSolver::Log>(
          stage_data, grid);
    }
  } else {
    athelas::par_for(
        DEFAULT_FLAT_LOOP_PATTERN, "RadHydro :: Fill derived :: temperature",
        DevExecSpace(), ib.s, ib.e, KOKKOS_LAMBDA(const int i) {
          double lambda[8];
          for (int q = 0; q < nNodes + 2; ++q) {
            const double rho = 1.0 / basis_eval(phi_fluid, uCF, i,
                                                vars::cons::SpecificVolume, q);
            const double vel =
                basis_eval(phi_fluid, uCF, i, vars::cons::Velocity, q);
            const double emt =
                basis_eval(phi_fluid, uCF, i, vars::cons::Energy, q);
            const double sie = emt - 0.5 * vel * vel;
            uAF(i, q, vars::aux::Tgas) =
                temperature_from_density_sie(eos, rho, sie, lambda);
          }
        });
  }

  if (ionization_enabled) {
    const auto *const comps = stage_data.comps();
    const auto number_density = comps->number_density();
    auto ye = comps->ye();

    const auto *const ionization_state = stage_data.ionization_state();
    auto ybar = ionization_state->ybar();
    auto e_ion_corr = ionization_state->e_ion_corr();
    auto sigma1 = ionization_state->sigma1();
    auto sigma2 = ionization_state->sigma2();
    auto sigma3 = ionization_state->sigma3();
    athelas::par_for(
        DEFAULT_FLAT_LOOP_PATTERN, "RadHydro :: fill derived", DevExecSpace(),
        ib.s, ib.e, KOKKOS_LAMBDA(const int i) {
          for (int q = 0; q < nNodes + 2; ++q) {
            const double tau =
                basis_eval(phi_fluid, uCF, i, vars::cons::SpecificVolume, q);
            const double vel =
                basis_eval(phi_fluid, uCF, i, vars::cons::Velocity, q);
            const double emt =
                basis_eval(phi_fluid, uCF, i, vars::cons::Energy, q);

            // const double e_rad = rad_basis_->basis_eval(uCF, i, 3, q + 1);
            // const double f_rad = rad_basis_->basis_eval(uCF, i, 4, q + 1);

            // const double flux_fact = flux_factor(e_rad, f_rad);

            const double rho = 1.0 / tau;
            const double momentum = rho * vel;
            const double sie = (emt - 0.5 * vel * vel);

            eos::EOSLambda lambda;
            lambda.data[0] = number_density(i, q);
            lambda.data[1] = ye(i, q);
            lambda.data[2] = ybar(i, q);
            lambda.data[3] = sigma1(i, q);
            lambda.data[4] = sigma2(i, q);
            lambda.data[5] = sigma3(i, q);
            lambda.data[6] = e_ion_corr(i, q);
            lambda.data[7] = uAF(i, q, vars::aux::Tgas);

            const double t_gas = uAF(i, q, vars::aux::Tgas);
            const double pressure = pressure_from_density_temperature(
                eos, rho, t_gas, lambda.ptr());
            const double cs = sound_speed_from_density_temperature_pressure(
                eos, rho, t_gas, pressure, lambda.ptr());

            uPF(i, q, vars::prim::Rho) = rho;
            uPF(i, q, vars::prim::Momentum) = momentum;
            uPF(i, q, vars::prim::Sie) = sie;

            uAF(i, q, vars::aux::Pressure) = pressure;
            uAF(i, q, vars::aux::Cs) = cs;
          }
        });
  } else {
    athelas::par_for(
        DEFAULT_FLAT_LOOP_PATTERN, "RadHydro :: fill derived", DevExecSpace(),
        ib.s, ib.e, KOKKOS_LAMBDA(const int i) {
          for (int q = 0; q < nNodes + 2; ++q) {
            const double tau =
                basis_eval(phi_fluid, uCF, i, vars::cons::SpecificVolume, q);
            const double vel =
                basis_eval(phi_fluid, uCF, i, vars::cons::Velocity, q);
            const double emt =
                basis_eval(phi_fluid, uCF, i, vars::cons::Energy, q);

            // const double e_rad = rad_basis_->basis_eval(uCF, i, 3, q + 1);
            // const double f_rad = rad_basis_->basis_eval(uCF, i, 4, q + 1);

            // const double flux_fact = flux_factor(e_rad, f_rad);

            const double rho = 1.0 / tau;
            const double momentum = rho * vel;
            const double sie = (emt - 0.5 * vel * vel);

            eos::EOSLambda lambda;

            const double t_gas = uAF(i, q, vars::aux::Tgas);
            const double pressure = pressure_from_density_temperature(
                eos, rho, t_gas, lambda.ptr());
            const double cs = sound_speed_from_density_temperature_pressure(
                eos, rho, t_gas, pressure, lambda.ptr());

            uPF(i, q, vars::prim::Rho) = rho;
            uPF(i, q, vars::prim::Momentum) = momentum;
            uPF(i, q, vars::prim::Sie) = sie;

            uAF(i, q, vars::aux::Pressure) = pressure;
            uAF(i, q, vars::aux::Cs) = cs;
          }
        });
  }
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

} // namespace athelas::radiation
