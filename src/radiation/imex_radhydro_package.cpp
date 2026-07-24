#include "radiation/imex_radhydro_package.hpp"

#include <limits>

#include "basic_types.hpp"
#include "basis/basis_utilities.hpp"
#include "basis/polynomial_basis.hpp"
#include "bc/boundary_conditions.hpp"
#include "composition/composition.hpp"
#include "composition/saha.hpp"
#include "eos/eos.hpp"
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
using fluid::numerical_flux_fluid_with_boundary;

void radiation_source_implicit(const StageData &stage_data,
                               AthelasArray3D<double> R,
                               AthelasArray4D<double> delta, const Mesh &mesh,
                               const TimeStepInfo &dt_info) {
  static const IndexRange ib(mesh.domain<Domain::Interior>());
  static const IndexRange qb(mesh.n_nodes());

  static const bool ionization_enabled = stage_data.enabled("ionization");

  auto evolved = stage_data.get_field("evolved");
  auto derived = stage_data.get_field("derived");

  const int idx_tau = stage_data.var_index("evolved", "specific_volume");
  const int idx_vel = stage_data.var_index("evolved", "velocity");
  const int idx_ener =
      stage_data.var_index("evolved", "specific_total_fluid_energy");
  const int idx_rad_energy =
      stage_data.var_index("evolved", "specific_radiation_energy");
  const int idx_rad_flux =
      stage_data.var_index("evolved", "specific_radiation_flux");
  const int idx_tgas = stage_data.var_index("derived", "gas_temperature");

  const auto &eos = stage_data.eos();
  const auto &opac = stage_data.opac();

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
          const auto evolved_i = Kokkos::subview(evolved, i, q, Kokkos::ALL);
          const auto uaf_i = Kokkos::subview(derived, i, q, Kokkos::ALL);
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

          const double rho = 1.0 / evolved_i(idx_tau);
          eos::EOSLambda lambda;
          lambda.data[0] = content.number_density;
          lambda.data[1] = content.ye;
          lambda.data[2] = content.ybar;
          lambda.data[3] = content.sigma1;
          lambda.data[4] = content.sigma2;
          lambda.data[5] = content.sigma3;
          lambda.data[6] = content.e_ion_corr;
          lambda.data[7] = uaf_i(idx_tgas);
          const double emin = min_sie(eos, rho, lambda.ptr());

          newton_radhydro<IonizationPhysics::Active>(
              dt_info.dt_coef, emin, Ustar_i, uaf_i, content, scratch_sol, eos,
              opac, lambda, idx_tau, idx_vel, idx_ener, idx_rad_energy,
              idx_rad_flux, idx_tgas);

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
          const auto evolved_i = Kokkos::subview(evolved, i, q, Kokkos::ALL);
          const auto uaf_i = Kokkos::subview(derived, i, q, Kokkos::ALL);
          const auto Ustar_i = Kokkos::subview(R, i, q, Kokkos::ALL);

          Kokkos::Array<double, NUM_VARS> scratch_sol;

          // set radhydro vars
          // scratch_sol[0] = Ustar_i(0);
          for (int v = 0; v < NUM_VARS; ++v) {
            // Ustar_i(v) += delta(dt_info.stage, i, q, v - 1) *
            // dt_info.dt_coef;
            scratch_sol[v] = Ustar_i(v);
          }

          const double rho = 1.0 / evolved_i(idx_tau);
          eos::EOSLambda lambda;
          const double emin = min_sie(eos, rho, lambda.ptr());

          newton_radhydro<IonizationPhysics::Inactive>(
              dt_info.dt_coef, emin, Ustar_i, uaf_i, content, scratch_sol, eos,
              opac, lambda, idx_tau, idx_vel, idx_ener, idx_rad_energy,
              idx_rad_flux, idx_tgas);

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
    : active_(active), cfl_(cfl), ap_coefficient_(pin->param()->get<double>(
                                      "radiation.ap_coefficient", 1.0)),
      bcs_(bcs), dFlux_num_("hydro::dFlux_num_", nx + 2 + 1, 5),
      u_f_l_("hydro::u_f_l_", nx + 2, 5), u_f_r_("hydro::u_f_r_", nx + 2, 5),
      delta_("radhydro delta", n_stages, nx + 2, nq, 5),
      delta_im_("radhydro delta implicit", n_stages, nx + 2, nq, 4) {}

auto RadHydroPackage::update_explicit(const StageData &stage_data,
                                      const TimeStepInfo &dt_info) const
    -> UpdateStatus {
  const auto &mesh = stage_data.mesh();
  const auto &basis = stage_data.basis();
  static const IndexRange ib(mesh.domain<Domain::Interior>());
  static const IndexRange qb(mesh.n_nodes());
  static const IndexRange vb(NUM_VARS_);

  const auto stage = dt_info.stage;

  // --- Refresh evolved halo ---
  bc::ghost_fill(stage_data, bcs_);

  // --- radiation Increment : Divergence ---
  radhydro_divergence(stage_data, mesh, stage);

  // --- Divide update by mass matrix ---
  auto inv_mqq = basis.inv_mass_matrix();
  const auto dtau_dt = stage_data.get_field<AthelasArray2D<double>>("dtau_dt");
  athelas::par_for(
      DEFAULT_LOOP_PATTERN, "RadHydro :: delta / M_qq", DevExecSpace(), ib.s,
      ib.e, qb.s, qb.e, KOKKOS_CLASS_LAMBDA(const int i, const int q) {
        const double inv_mm = inv_mqq(i, q);
        for (int v = 0; v < NUM_VARS_; ++v) {
          delta_(stage, i, q, v) *= inv_mm;
        }
        dtau_dt(i, q) = delta_(stage, i, q, 0);
      });

  return UpdateStatus::Success;
} // update_explicit

auto RadHydroPackage::update_implicit(const StageData &stage_data,
                                      AthelasArray3D<double> R,
                                      const TimeStepInfo &dt_info)
    -> UpdateStatus {
  const auto &mesh = stage_data.mesh();
  // compute radiation-matter coupling sources implicitly with Newton-Raphson.
  radiation_source_implicit(stage_data, R, delta_im_, mesh, dt_info);

  return UpdateStatus::Success;
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
  auto evolved = stage_data.get_field("evolved");
  auto derived = stage_data.get_field("derived");
  auto interface = stage_data.get_field<AthelasArray2D<double>>("interface");

  const auto &basis = stage_data.basis();

  const auto &nNodes = mesh.n_nodes();
  static constexpr int ilo = 1;
  static const auto &ihi = mesh.get_ihi();
  static const IndexRange ib(mesh.domain<Domain::Interior>());
  static const IndexRange qb(mesh.n_nodes());
  static const IndexRange vb(NUM_VARS_);

  static const int idx_tau = stage_data.var_index("evolved", "specific_volume");
  static const int idx_vel = stage_data.var_index("evolved", "velocity");
  static const int idx_ener =
      stage_data.var_index("evolved", "specific_total_fluid_energy");
  static const int idx_pre = stage_data.var_index("derived", "pressure");
  static const int idx_cs = stage_data.var_index("derived", "sound_speed");
  static const int idx_density = stage_data.var_index("derived", "density");
  static const int idx_radener =
      stage_data.var_index("evolved", "specific_radiation_energy");
  static const int idx_radflux =
      stage_data.var_index("evolved", "specific_radiation_flux");
  static const int idx_vstar =
      stage_data.var_index("interface", "interface_velocity");
  static const int idx_tgas =
      stage_data.var_index("derived", "gas_temperature");

  auto sqrt_gm = mesh.sqrt_gm();
  auto weights = mesh.weights();
  auto dr = mesh.widths();

  auto phi = basis.phi();
  auto dphi = basis.dphi();
  const auto fluid_bcs = bc::fluid_bc_data(bcs_);
  const auto rad_bcs = bc::radiation_bc_data(bcs_);
  const auto &opac = stage_data.opac();
  const double ap_coefficient = ap_coefficient_;
  // C = 0 recovers standard LLF; skip the per-face opacity work entirely.
  const bool ap_correction = ap_coefficient > 0.0;
  const bool composition_enabled = stage_data.enabled("composition");
  AthelasArray3D<double> bulk;
  if (composition_enabled) {
    bulk = stage_data.get_field("bulk_composition");
  }

  // --- Interpolate Conserved Variable to Interfaces ---

  // Left/Right face states

  athelas::par_for(
      DEFAULT_FLAT_LOOP_PATTERN, "RadHydro :: Interface states", DevExecSpace(),
      ib.s, ib.e + 1, KOKKOS_CLASS_LAMBDA(const int i) {
        for (int v = 0; v < NUM_VARS_; ++v) {
          u_f_l_(i, v) = basis_eval<Interface::Right>(phi, evolved, i - 1, v);
          u_f_r_(i, v) = basis_eval<Interface::Left>(phi, evolved, i, v);
        }
      });

  // --- Calc numerical flux at all faces ---
  athelas::par_for(
      DEFAULT_FLAT_LOOP_PATTERN, "RadHydro :: Numerical fluxes", DevExecSpace(),
      ib.s, ib.e + 1, KOKKOS_CLASS_LAMBDA(const int i) {
        const double Pgas_L = derived(i - 1, nNodes + 1, idx_pre);
        const double Cs_L = derived(i - 1, nNodes + 1, idx_cs);

        const double Pgas_R = derived(i, 0, idx_pre);
        const double Cs_R = derived(i, 0, idx_cs);

        const double rhoL = 1.0 / u_f_l_(i, idx_tau);
        const double rhoR = 1.0 / u_f_r_(i, idx_tau);

        const double E_L = u_f_l_(i, idx_radener) * rhoL;
        const double F_L = u_f_l_(i, idx_radflux) * rhoL;
        const double E_R = u_f_r_(i, idx_radener) * rhoR;
        const double F_R = u_f_r_(i, idx_radflux) * rhoR;

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
        const auto fluid_flux = numerical_flux_fluid_with_boundary(
            i, ib.s, ib.e + 1, fluid_bcs, left, right);
        const double flux_u = fluid_flux.u;
        const double flux_p = fluid_flux.p;

        if (i == ib.s) {
          interface(ilo - 1, idx_vstar) = flux_u;
        } else if (i == ib.e + 1) {
          interface(ihi + 2, idx_vstar) = flux_u;
        }
        interface(i, idx_vstar) = flux_u;

        const double vstar = flux_u;

        // Asymptotic preserving flux correction
        // Compute beta = 1 / (1 + ap_coefficient * tau) where
        // tau is face averaged rho kappa dr (except on physical boundaries).
        double beta = 1.0;
        if (ap_correction) {
          double tau_face = 0.0;
          if (i == ib.s) {
            tau_face =
                cell_optical_depth(opac, derived, bulk, composition_enabled,
                                   idx_tgas, i, 0, rhoR, dr(i));
          } else if (i == ib.e + 1) {
            tau_face = cell_optical_depth(opac, derived, bulk,
                                          composition_enabled, idx_tgas, i - 1,
                                          nNodes + 1, rhoL, dr(i - 1));
          } else {
            const double tau_L = cell_optical_depth(
                opac, derived, bulk, composition_enabled, idx_tgas, i - 1,
                nNodes + 1, rhoL, dr(i - 1));
            const double tau_R =
                cell_optical_depth(opac, derived, bulk, composition_enabled,
                                   idx_tgas, i, 0, rhoR, dr(i));
            tau_face = face_optical_depth(tau_L, tau_R);
          }
          beta = ap_dissipation_factor(tau_face, ap_coefficient);
        }
        const auto rad_flux = numerical_flux_rad_with_boundary(
            i, ib.s, ib.e + 1, rad_bcs, RadBoundaryState{.E = E_L, .F = F_L},
            RadBoundaryState{.E = E_R, .F = F_R}, vstar, beta);

        dFlux_num_(i, idx_tau) = -flux_u;
        dFlux_num_(i, idx_vel) = flux_p;
        dFlux_num_(i, idx_ener) = +flux_u * flux_p;

        dFlux_num_(i, idx_radener) = rad_flux.e;
        dFlux_num_(i, idx_radflux) = rad_flux.f;
      });

  // --- Surface Term ---
  basis::surface_term(delta_, dFlux_num_, phi, sqrt_gm, stage, nNodes, ib, qb,
                      vb, "RadHydro :: Surface term");

  if (nNodes > 1) [[likely]] {
    // --- Volume Term ---
    auto derived = stage_data.get_field("derived");
    basis::volume_term<NUM_VARS_>(
        delta_, dphi, weights, sqrt_gm, stage, nNodes, ib, qb,
        KOKKOS_LAMBDA(const int i, const int q)
            ->Kokkos::Array<double, NUM_VARS_> {
              const double pressure = derived(i, q + 1, idx_pre);
              const double rho = derived(i, q, idx_density);
              const double vel = evolved(i, q, idx_vel);
              const double e_rad = evolved(i, q, idx_radener) * rho;
              const double f_rad = evolved(i, q, idx_radflux) * rho;
              const double p_rad = compute_closure(e_rad, f_rad);
              const double vstar = interface(i, idx_vstar);
              const auto [flux1, flux2, flux3] =
                  athelas::fluid::flux_fluid(vel, pressure);
              const auto [flux_e, flux_f] =
                  flux_rad(e_rad, f_rad, p_rad, vstar);
              return {flux1, flux2, flux3, flux_e, flux_f};
            },
        "RadHydro :: Volume term");
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
  auto evolved = stage_data.get_field("evolved");
  auto derived = stage_data.get_field("derived");

  const int idx_tau = stage_data.var_index("evolved", "specific_volume");
  const int idx_vel = stage_data.var_index("evolved", "velocity");
  const int idx_ener =
      stage_data.var_index("evolved", "specific_total_fluid_energy");
  const int idx_density = stage_data.var_index("derived", "density");
  const int idx_momentum = stage_data.var_index("derived", "momentum_density");
  const int idx_sie =
      stage_data.var_index("derived", "specific_internal_energy");
  const int idx_pressure = stage_data.var_index("derived", "pressure");
  const int idx_tgas = stage_data.var_index("derived", "gas_temperature");
  const int idx_cs = stage_data.var_index("derived", "sound_speed");

  const auto &basis = stage_data.basis();

  const auto &eos = stage_data.eos();

  const int nNodes = mesh.n_nodes();
  static const IndexRange ib(mesh.domain<Domain::Interior>());
  static const bool ionization_enabled = stage_data.enabled("ionization");

  auto phi = basis.phi();

  // --- Refresh evolved halo ---
  bc::ghost_fill(stage_data, bcs_);

  if (stage_data.enabled("composition")) {
    atom::fill_derived_comps(stage_data, &mesh);
  }

  // First we get the temperature from the density and specific internal
  // energy. The ionization case is involved and so this is all done
  // separately. In that case the temperature solve is coupled to a Saha
  // solve.
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
        DEFAULT_FLAT_LOOP_PATTERN, "RadHydro :: Fill derived :: temperature",
        DevExecSpace(), ib.s, ib.e, KOKKOS_LAMBDA(const int i) {
          double lambda[eos::EOS_LAMBDA_SIZE] = {};
          for (int q = 0; q < nNodes + 2; ++q) {
            const double rho = 1.0 / basis_eval(phi, evolved, i, idx_tau, q);
            const double vel = basis_eval(phi, evolved, i, idx_vel, q);
            const double emt = basis_eval(phi, evolved, i, idx_ener, q);
            const double sie = emt - 0.5 * vel * vel;
            derived(i, q, idx_tgas) =
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
            const double tau = basis_eval(phi, evolved, i, idx_tau, q);
            const double vel = basis_eval(phi, evolved, i, idx_vel, q);
            const double emt = basis_eval(phi, evolved, i, idx_ener, q);

            // const double e_rad = basis_->basis_eval(evolved, i, 3, q +
            // 1); const double f_rad = basis_->basis_eval(evolved, i, 4, q
            // + 1);

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
            lambda.data[7] = derived(i, q, idx_tgas);

            const double t_gas = derived(i, q, idx_tgas);
            const double pressure = pressure_from_density_temperature(
                eos, rho, t_gas, lambda.ptr());
            const double cs = sound_speed_from_density_temperature_pressure(
                eos, rho, t_gas, pressure, lambda.ptr());

            derived(i, q, idx_density) = rho;
            derived(i, q, idx_momentum) = momentum;
            derived(i, q, idx_sie) = sie;

            derived(i, q, idx_pressure) = pressure;
            derived(i, q, idx_cs) = cs;
          }
        });
  } else {
    athelas::par_for(
        DEFAULT_FLAT_LOOP_PATTERN, "RadHydro :: fill derived", DevExecSpace(),
        ib.s, ib.e, KOKKOS_LAMBDA(const int i) {
          for (int q = 0; q < nNodes + 2; ++q) {
            const double tau = basis_eval(phi, evolved, i, idx_tau, q);
            const double vel = basis_eval(phi, evolved, i, idx_vel, q);
            const double emt = basis_eval(phi, evolved, i, idx_ener, q);

            // const double e_rad = basis_->basis_eval(evolved, i, 3, q +
            // 1); const double f_rad = basis_->basis_eval(evolved, i, 4, q
            // + 1);

            // const double flux_fact = flux_factor(e_rad, f_rad);

            const double rho = 1.0 / tau;
            const double momentum = rho * vel;
            const double sie = (emt - 0.5 * vel * vel);

            eos::EOSLambda lambda;

            const double t_gas = derived(i, q, idx_tgas);
            const double pressure = pressure_from_density_temperature(
                eos, rho, t_gas, lambda.ptr());
            const double cs = sound_speed_from_density_temperature_pressure(
                eos, rho, t_gas, pressure, lambda.ptr());

            derived(i, q, idx_density) = rho;
            derived(i, q, idx_momentum) = momentum;
            derived(i, q, idx_sie) = sie;

            derived(i, q, idx_pressure) = pressure;
            derived(i, q, idx_cs) = cs;
          }
        });
  }

  // Copy derived into the ghost cells so boundary flux reads see valid
  // pressure / sound speed there (mirrors the evolved ghost fill).
  bc::ghost_fill_derived(derived, bcs_);
}

[[nodiscard]] auto RadHydroPackage::name() const noexcept -> std::string_view {
  return "RadHydro";
}

[[nodiscard]] auto RadHydroPackage::is_active() const noexcept -> bool {
  return active_;
}

void RadHydroPackage::set_active(const bool active) { active_ = active; }
} // namespace athelas::radiation
