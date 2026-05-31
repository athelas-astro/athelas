#include "radiation/imex_radhydro_package.hpp"

#include <limits>

#include "basic_types.hpp"
#include "basis/polynomial_basis.hpp"
#include "bc/boundary_conditions.hpp"
#include "composition/composition.hpp"
#include "composition/saha.hpp"
#include "eos/eos_variant.hpp"
#include "fluid/fluid_utilities.hpp"
#include "geometry/mesh.hpp"
#include "kokkos_abstraction.hpp"
#include "loop_layout.hpp"
#include "pgen/problem_in.hpp"
#include "radiation/rad_utilities.hpp"

namespace athelas::radiation {
using basis::NodalBasis, basis::basis_eval;
using eos::EOS;
using fluid::numerical_flux_gudonov_positivity;

void radiation_source_implicit(const StageData &stage_data,
                               AthelasArray3D<double> R,
                               AthelasArray4D<double> delta, const Mesh &mesh,
                               const TimeStepInfo &dt_info) {
  // TODO(astrobarker) handle separate fluid and rad orders
  const auto &rad_basis = stage_data.rad_basis();
  const auto &fluid_basis = stage_data.fluid_basis();
  static const IndexRange ib(mesh.domain<Domain::Interior>());
  static const IndexRange qb(mesh.n_nodes());

  static const bool ionization_enabled = stage_data.enabled("ionization");

  auto ucf = stage_data.get_field("u_cf");
  auto uaf = stage_data.get_field("u_af");

  const auto &eos = stage_data.eos();
  const auto &opac = stage_data.opac();

  auto phi_rad = rad_basis.phi();
  auto phi_fluid = fluid_basis.phi();
  auto inv_mkk = fluid_basis.inv_mass_matrix();
  auto inv_mkk_rad = rad_basis.inv_mass_matrix();
  auto dr = mesh.widths();
  auto weights = mesh.weights();
  auto sqrt_gm = mesh.sqrt_gm();

  constexpr int NUM_VARS = 5;
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
    auto bulk = stage_data.get_field("bulk_composition");
    athelas::par_for(
        DEFAULT_LOOP_PATTERN, "Radiation :: Implicit sources", DevExecSpace(),
        ib.s, ib.e, qb.s, qb.e, KOKKOS_LAMBDA(const int i, const int q) {
          const auto ucf_i = Kokkos::subview(ucf, i, q, Kokkos::ALL);
          const auto uaf_i = Kokkos::subview(uaf, i, q, Kokkos::ALL);
          const auto Ustar_i = Kokkos::subview(R, i, q, Kokkos::ALL);
          const RadHydroSolverIonizationContent content{
              .number_density = number_density(i, q + 1),
              .ye = ye(i, q + 1),
              .ybar = ybar(i, q + 1),
              .sigma1 = sigma1(i, q + 1),
              .sigma2 = sigma2(i, q + 1),
              .sigma3 = sigma3(i, q + 1),
              .e_ion_corr = e_ion_corr(i, q + 1),
              .X = bulk(i, q + 1, 0),
              .Z = bulk(i, q + 1, 2)};

          Kokkos::Array<double, NUM_VARS> scratch_sol;

          // set radhydro vars
          // scratch_sol[0] = Ustar_i(0);
          for (int v = 0; v < NUM_VARS; ++v) {
            // Ustar_i(v) += delta(dt_info.stage, i, q, v - 1) *
            // dt_info.dt_coef;
            scratch_sol[v] = Ustar_i(v);
          }

          const double rho = 1.0 / ucf_i(vars::cons::SpecificVolume);
          eos::EOSLambda lambda;
          lambda.data[0] = content.number_density;
          lambda.data[1] = content.ye;
          lambda.data[2] = content.ybar;
          lambda.data[3] = content.sigma1;
          lambda.data[4] = content.sigma2;
          lambda.data[5] = content.sigma3;
          lambda.data[6] = content.e_ion_corr;
          lambda.data[7] = uaf_i(vars::aux::Tgas);
          const double emin = min_sie(eos, rho, lambda.ptr());
          const double dg_term =
              weights(q) * sqrt_gm(i, q + 1) * dr(i) * inv_mkk(i, q);

          newton_radhydro<IonizationPhysics::Active>(
              dt_info.dt_coef, emin, Ustar_i, uaf_i, content, scratch_sol, eos,
              opac, lambda, dg_term);

          for (int v = 1; v < NUM_VARS; ++v) {
            // Ustar_i(v) -= delta(dt_info.stage, i, q, v - 1) *
            // dt_info.dt_coef;
            delta(dt_info.stage, i, q, v - 1) =
                (scratch_sol[v] - Ustar_i(v)) / dt_info.dt_coef;
          }
        });
  } else {
    const RadHydroSolverIonizationContent content;
    athelas::par_for(
        DEFAULT_FLAT_LOOP_PATTERN, "Radiation :: Implicit sources",
        DevExecSpace(), ib.s, ib.e, qb.s, qb.e,
        KOKKOS_LAMBDA(const int i, const int q) {
          const auto ucf_i = Kokkos::subview(ucf, i, q, Kokkos::ALL);
          const auto uaf_i = Kokkos::subview(uaf, i, q, Kokkos::ALL);
          const auto Ustar_i = Kokkos::subview(R, i, q, Kokkos::ALL);

          Kokkos::Array<double, NUM_VARS> scratch_sol;

          // set radhydro vars
          // scratch_sol[0] = Ustar_i(0);
          for (int v = 0; v < NUM_VARS; ++v) {
            // Ustar_i(v) += delta(dt_info.stage, i, q, v - 1) *
            // dt_info.dt_coef;
            scratch_sol[v] = Ustar_i(v);
          }

          const double rho = 1.0 / ucf_i(vars::cons::SpecificVolume);
          eos::EOSLambda lambda;
          const double emin = min_sie(eos, rho, lambda.ptr());
          const double inv_mqq = inv_mkk(i, q);
          const double dr_i = dr(i);
          const double dg_term =
              weights(q) * sqrt_gm(i, q + 1) * dr_i * inv_mqq;

          newton_radhydro<IonizationPhysics::Inactive>(
              dt_info.dt_coef, emin, Ustar_i, uaf_i, content, scratch_sol, eos,
              opac, lambda, dg_term);

          for (int v = 1; v < NUM_VARS; ++v) {
            // Ustar_i(v) -= delta(dt_info.stage, i, q, v - 1) *
            // dt_info.dt_coef;
            delta(dt_info.stage, i, q, v - 1) =
                (scratch_sol[v] - Ustar_i(v)) / dt_info.dt_coef;
          }
        });
  }
}

/**
 * @brief IMEX Radiation hydrodynamics
 * Joint radiation hydrodynamics package doing hyperbolic terms explicitly
 * and sources implicitly. This is likely not used for production as
 * the explicit treatment of the hyperbolic radiation flux divergence
 * has an overly restrictive timestep restriction.
 */
RadHydroPackage::RadHydroPackage(const ProblemIn *pin, int n_stages, int nq,
                                 BoundaryConditions *bcs, double cfl, int nx,
                                 bool active)
    : active_(active), cfl_(cfl), bcs_(bcs),
      dFlux_num_("hydro::dFlux_num_", nx + 2 + 1, 5),
      u_f_l_("hydro::u_f_l_", nx + 2, 5), u_f_r_("hydro::u_f_r_", nx + 2, 5),
      delta_("radhydro delta", n_stages, nx + 2, nq, 5),
      delta_im_("radhydro delta implicit", n_stages, nx + 2, nq, 4) {}

void RadHydroPackage::update_explicit(const StageData &stage_data,
                                      const TimeStepInfo &dt_info) const {
  const auto &mesh = stage_data.mesh();
  // TODO(astrobarker) handle separate fluid and rad orders
  const auto &basis = stage_data.fluid_basis();
  static const IndexRange ib(mesh.domain<Domain::Interior>());
  static const IndexRange qb(mesh.n_nodes());
  static const IndexRange vb(NUM_VARS_);

  const auto stage = dt_info.stage;
  auto ucf = stage_data.get_field("u_cf");

  // --- Apply BC ---
  bc::fill_ghost_zones<2>(ucf, &mesh, bcs_, {3, 4});
  bc::fill_ghost_zones<3>(ucf, &mesh, bcs_, {0, 2});

  // --- radiation Increment : Divergence ---
  radhydro_divergence(stage_data, mesh, stage);

  // --- Divide update by mass matrix ---
  auto inv_mqq = basis.inv_mass_matrix();
  athelas::par_for(
      DEFAULT_LOOP_PATTERN, "RadHydro :: delta / M_qq", DevExecSpace(), ib.s,
      ib.e, qb.s, qb.e, KOKKOS_CLASS_LAMBDA(const int i, const int q) {
        const double inv_mm = inv_mqq(i, q);
        for (int v = 0; v < NUM_VARS_; ++v) {
          delta_(stage, i, q, v) *= inv_mm;
        }
      });
} // update_explicit

void RadHydroPackage::update_implicit(const StageData &stage_data,
                                      AthelasArray3D<double> R,
                                      const TimeStepInfo &dt_info) {
  const auto &mesh = stage_data.mesh();
  // compute radiation-matter coupling sources implicitly with Newton-Raphson.
  radiation_source_implicit(stage_data, R, delta_im_, mesh, dt_info);
}

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
        }
        for (int v = vb.s + 1; v <= vb.e; ++v) {
          lhs(i, q, v) +=
              dt_info.dt_coef_implicit * delta_im_(stage, i, q, v - 1);
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

  // We store the last stage source in the state = 0 slot.
  // That is, G(U^0) <- G(U^n).
  // In an ESDIRK tableau we reuse this for the first stage.
  const int ns = sb.e;
  athelas::par_for(
      DEFAULT_LOOP_PATTERN, "RadHydro :: Zero delta", DevExecSpace(), ib.s,
      ib.e, qb.s, qb.e, KOKKOS_CLASS_LAMBDA(const int i, const int q) {
        for (int v = vb.s; v <= vb.e - 1; ++v) {
          delta_im_(0, i, q, v) = delta_im_(ns, i, q, v);
        }
      });

  athelas::par_for(
      DEFAULT_LOOP_PATTERN, "RadHydro :: Zero delta_im", DevExecSpace(),
      sb.s + 1, sb.e, ib.s, ib.e, qb.s, qb.e,
      KOKKOS_CLASS_LAMBDA(const int s, const int i, const int q) {
        for (int v = vb.s; v <= vb.e - 1; ++v) {
          delta_im_(s, i, q, v) = 0.0;
        }
      });

  Kokkos::deep_copy(delta_, 0.0);
}

// Compute the divergence of the flux term for the update
// TODO(astrobarker): dont pass in stage
void RadHydroPackage::radhydro_divergence(const StageData &stage_data,
                                          const Mesh &mesh,
                                          const int stage) const {
  using fluid::FluidRiemannState;
  auto ucf = stage_data.get_field("u_cf");
  auto uaf = stage_data.get_field("u_af");
  auto facedata = stage_data.get_field<AthelasArray2D<double>>("facedata");

  const auto &rad_basis = stage_data.rad_basis();
  const auto &fluid_basis = stage_data.fluid_basis();

  const auto &nNodes = mesh.n_nodes();
  static constexpr int ilo = 1;
  static const auto &ihi = mesh.get_ihi();
  static const IndexRange ib(mesh.domain<Domain::Interior>());
  static const IndexRange qb(mesh.n_nodes());
  static const IndexRange vb(NUM_VARS_);

  static const int idx_tau = stage_data.var_index("u_cf", "tau");
  static const int idx_vel = stage_data.var_index("u_cf", "vel");
  static const int idx_ener = stage_data.var_index("u_cf", "fluid_energy");
  static const int idx_pre = stage_data.var_index("u_af", "pressure");
  static const int idx_cs = stage_data.var_index("u_af", "sound speed");
  static const int idx_radener = stage_data.var_index("u_cf", "rad_energy");
  static const int idx_radflux = stage_data.var_index("u_cf", "rad_momentum");
  static const int idx_vstar = stage_data.var_index("facedata", "vstar");

  auto sqrt_gm = mesh.sqrt_gm();
  auto weights = mesh.weights();

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
  static constexpr double c2 = constants::c_cgs * constants::c_cgs;
  athelas::par_for(
      DEFAULT_FLAT_LOOP_PATTERN, "RadHydro :: Numerical fluxes", DevExecSpace(),
      ib.s, ib.e + 1, KOKKOS_CLASS_LAMBDA(const int i) {
        const double Pgas_L = uaf(i - 1, nNodes + 1, idx_pre);
        const double Cs_L = uaf(i - 1, nNodes + 1, idx_cs);

        const double Pgas_R = uaf(i, 0, idx_pre);
        const double Cs_R = uaf(i, 0, idx_cs);

        const double rhoL = 1.0 / u_f_l_(i, idx_tau);
        const double rhoR = 1.0 / u_f_r_(i, idx_tau);

        const double E_L = u_f_l_(i, idx_radener) * rhoL;
        const double F_L = u_f_l_(i, idx_radflux) * rhoL;
        const double E_R = u_f_r_(i, idx_radener) * rhoR;
        const double F_R = u_f_r_(i, idx_radflux) * rhoR;

        const double Prad_L = compute_closure(E_L, F_L);
        const double Prad_R = compute_closure(E_R, F_R);

        // --- Numerical Fluxes ---

        // Riemann Problem
        // auto [flux_u, flux_p] = numerical_flux_gudonov( u_f_l_(i,  1 ),
        // u_f_r_(i,  1
        // ), P_L, P_R, lam_L, lam_R);
        const FluidRiemannState left{.tau = u_f_l_(i, idx_tau),
                                     .v = u_f_l_(i, idx_vel),
                                     .p = Pgas_L,
                                     .cs = Cs_L};
        const FluidRiemannState right{.tau = u_f_r_(i, idx_tau),
                                      .v = u_f_r_(i, idx_vel),
                                      .p = Pgas_R,
                                      .cs = Cs_R};
        const auto [flux_u, flux_p] =
            numerical_flux_gudonov_positivity(left, right);
        facedata(i, idx_vstar) = flux_u;

        const double vstar = flux_u;

        const double alpha = rad_wavespeed(E_L, E_R, F_L, F_R, vstar);

        const LLFRiemannState left_erad{
            .u = E_L, .f = F_L - vstar * E_L, .alpha = alpha};
        const LLFRiemannState right_erad{
            .u = E_R, .f = F_R - vstar * E_R, .alpha = alpha};
        const double flux_e = llf_flux(left_erad, right_erad);

        const LLFRiemannState left_frad{
            .u = F_L, .f = c2 * Prad_L - vstar * F_L, .alpha = alpha};
        const LLFRiemannState right_frad{
            .u = F_R, .f = c2 * Prad_R - vstar * F_R, .alpha = alpha};
        const double flux_f = llf_flux(left_frad, right_frad);

        dFlux_num_(i, idx_tau) = -flux_u;
        dFlux_num_(i, idx_vel) = flux_p;
        dFlux_num_(i, idx_ener) = +flux_u * flux_p;

        dFlux_num_(i, idx_radener) = flux_e;
        dFlux_num_(i, idx_radflux) = flux_f;
      });

  facedata(ilo - 1, idx_vstar) = facedata(ilo, idx_vstar);
  facedata(ihi + 2, idx_vstar) = facedata(ihi + 1, idx_vstar);

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
          const double vstar = facedata(i, idx_vstar);
          for (int q = 0; q < nNodes; ++q) {
            const int qp1 = q + 1;

            const double pressure = uaf(i, qp1, idx_pre);
            const double rho = upf(i, q, vars::prim::Rho);
            const double vel = ucf(i, q, idx_vel);
            const double e_rad = ucf(i, q, idx_radener) * rho;
            const double f_rad = ucf(i, q, idx_radflux) * rho;
            const double p_rad = compute_closure(e_rad, f_rad);
            const auto [flux1, flux2, flux3] =
                athelas::fluid::flux_fluid(vel, pressure);
            const auto [flux_e, flux_f] = flux_rad(e_rad, f_rad, p_rad, vstar);
            const double w_dphi_sqrtgm =
                weights(q) * dphi_fluid(i, qp1, p) * sqrt_gm(i, qp1);
            local_sum1 += w_dphi_sqrtgm * flux1;
            local_sum2 += w_dphi_sqrtgm * flux2;
            local_sum3 += w_dphi_sqrtgm * flux3;
            local_sum_e += w_dphi_sqrtgm * flux_e;
            local_sum_f += w_dphi_sqrtgm * flux_f;
          }

          delta_(stage, i, p, idx_tau) += local_sum1;
          delta_(stage, i, p, idx_vel) += local_sum2;
          delta_(stage, i, p, idx_ener) += local_sum3;
          delta_(stage, i, p, idx_radener) += local_sum_e;
          delta_(stage, i, p, idx_radflux) += local_sum_f;
        });
  }
} // radhydro_divergence

/**
 * @brief explicit radiation hydrodynamic timestep restriction
 **/
auto RadHydroPackage::min_timestep(const StageData &stage_data,
                                   const TimeStepInfo & /*dt_info*/) const
    -> double {
  const auto &mesh = stage_data.mesh();
  static constexpr double MAX_DT = std::numeric_limits<double>::max();
  static constexpr double MIN_DT = 100.0 * std::numeric_limits<double>::min();

  static const IndexRange ib(mesh.domain<Domain::Interior>());

  const auto dr = mesh.widths();

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
                                   const TimeStepInfo &dt_info) const {
  const auto &mesh = stage_data.mesh();
  auto uCF = stage_data.get_field("u_cf");
  auto uPF = stage_data.get_field("u_pf");
  auto uAF = stage_data.get_field("u_af");

  const auto &fluid_basis = stage_data.fluid_basis();

  const auto &eos = stage_data.eos();

  const int nNodes = mesh.n_nodes();
  static const IndexRange ib(mesh.domain<Domain::Entire>());
  static const bool ionization_enabled = stage_data.enabled("ionization");

  auto phi_fluid = fluid_basis.phi();

  // --- Apply BC ---
  bc::fill_ghost_zones<2>(uCF, &mesh, bcs_, {3, 4});
  bc::fill_ghost_zones<3>(uCF, &mesh, bcs_, {0, 2});

  if (stage_data.enabled("composition")) {
    static constexpr int nvars = 5; // non-comps
    // composition boundary condition
    static const IndexRange vb_comps(
        std::make_pair(nvars, stage_data.nvars("u_cf") - 1));
    bc::fill_ghost_zones_composition(uCF, vb_comps);
    atom::fill_derived_comps<Domain::Entire>(stage_data, &mesh);
  }

  // First we get the temperature from the density and specific internal
  // energy. The ionization case is involved and so this is all done
  // separately. In that case the temperature solve is coupled to a Saha
  // solve.
  if (ionization_enabled) {
    auto *const ionization_state = stage_data.ionization_state();
    if (ionization_state->solver() == atom::SahaSolver::Linear) {
      atom::compute_temperature_with_saha<
          Domain::Entire, eos::EOSInversion::Sie, atom::SahaSolver::Linear>(
          stage_data, mesh);
    }
    if (ionization_state->solver() == atom::SahaSolver::Log) {
      atom::compute_temperature_with_saha<
          Domain::Entire, eos::EOSInversion::Sie, atom::SahaSolver::Log>(
          stage_data, mesh);
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
} // namespace athelas::radiation
