#include "radiation/implicit_moments_package.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>

#include "basic_types.hpp"
#include "basis/polynomial_basis.hpp"
#include "bc/boundary_conditions.hpp"
#include "eos/eos_variant.hpp"
#include "geometry/mesh.hpp"
#include "kokkos_abstraction.hpp"
#include "loop_layout.hpp"
#include "math/linear_algebra.hpp"
#include "pgen/problem_in.hpp"
#include "radiation/rad_utilities.hpp"
#include "utils/error.hpp"

namespace athelas::radiation {
using basis::NodalBasis, basis::basis_eval;
using eos::EOS;

/**
 * @brief Implicit radiation moments
 * Used for fully implicit transport
 */
ImplicitRadiationMomentsPackage::ImplicitRadiationMomentsPackage(
    const ProblemIn *pin, int n_stages, int nq, BoundaryConditions *bcs, int nx,
    bool active)
    : active_(active), bcs_(bcs),
      faces_{
          .u_f_l = AthelasArray2D<double>("ImplicitMoments::u_f_l", nx + 2, 3),
          .u_f_r = AthelasArray2D<double>("ImplicitMoments::u_f_r", nx + 2, 3),
          .flux_num =
              AthelasArray2D<double>("ImplicitMoments::flux_num", nx + 2, 2),
          .beta = AthelasArray1D<double>("ImplicitMoments::beta", nx + 2),
          .A_minus =
              AthelasArray3D<double>("ImplicitMoments::A_minus", nx + 1, 2, 2),
          .A_plus =
              AthelasArray3D<double>("ImplicitMoments::A_plus", nx + 1, 2, 2),
      },
      solver_{
          .mat_diag = AthelasArray3D<double>("ImplicitMoments::solver.mat_diag",
                                             nx, 4 * nq, 4 * nq),
          .mat_upper = AthelasArray3D<double>(
              "ImplicitMoments::solver.mat_upper", nx - 1, 4 * nq, 4 * nq),
          .mat_lower = AthelasArray3D<double>(
              "ImplicitMoments::solver.mat_lower", nx - 1, 4 * nq, 4 * nq),
          .b = AthelasArray2D<double>("ImplicitMoments::solver.b", nx, 4 * nq),
          .W = AthelasArray3D<double>("ImplicitMoments::solver.W", nx - 1,
                                      4 * nq, 4 * nq),
          .Y = AthelasArray2D<double>("ImplicitMoments::solver.Y", nx - 1,
                                      4 * nq),
          .Bi_lu = AthelasArray2D<double>("ImplicitMoments::solver.Bi_lu",
                                          4 * nq, 4 * nq),
          .piv =
              math::linalg::PivotStore("ImplicitMoments::solver.piv", 4 * nq),
          .row_scale = AthelasArray2D<double>(
              "ImplicitMoments::solver.row_scale", nx, 4 * nq),
          .col_scale = AthelasArray2D<double>(
              "ImplicitMoments::solver.col_scale", nx, 4 * nq),
      },
      newton_{
          .u_rad_work = AthelasArray3D<double>(
              "ImplicitMoments::newton.u_rad_work", nx + 2, nq, 5),
          .u_rad_trial = AthelasArray3D<double>(
              "ImplicitMoments::newton.u_rad_trial", nx + 2, nq, 5),
          .ls_b_trial = AthelasArray2D<double>(
              "ImplicitMoments::newton.ls_b_trial", nx, 4 * nq),
      },
      dt_cache_{
          .e_rad_old =
              AthelasArray2D<double>("ImplicitMoments::e_rad_old", nx + 2, nq),
          .f_rad_old =
              AthelasArray2D<double>("ImplicitMoments::f_rad_old", nx + 2, nq),
      },
      delta_("ImplicitMoments::delta", n_stages, nx + 2, nq, 4) {
  // Storing package params
  params_.add<double>(
      "max_fractional_change_e",
      pin->param()->get<double>("radiation.timestep.max_fractional_change_e"));
  params_.add<double>(
      "energy_change_scale",
      pin->param()->get<double>("radiation.timestep.energy_change_scale"));
  params_.add<double>("max_change_f", pin->param()->get<double>(
                                          "radiation.timestep.max_change_f"));
  params_.add<int>("newton.max_iters",
                   pin->param()->get<int>("radiation.newton.max_iter"));
  params_.add<double>("newton.tol",
                      pin->param()->get<double>("radiation.newton.tol"));
  params_.add<double>("newton.atol",
                      pin->param()->get<double>("radiation.newton.atol", 0.0));
  params_.add<double>("ap_coefficient", pin->param()->get<double>(
                                            "radiation.ap_coefficient", 1.0));
}

void ImplicitRadiationMomentsPackage::evaluate_residual(
    AthelasArray2D<double> b_out, AthelasArray3D<double> U,
    AthelasArray3D<double> ustar, const StageData &stage_data, const Mesh &mesh,
    const double dt_aii) {
  const auto &basis = stage_data.basis();
  const int nNodes = mesh.n_nodes();
  static const IndexRange ib(mesh.domain<Domain::Interior>());

  static const int idx_tau = stage_data.var_index("evolved", "specific_volume");
  static const int idx_vel = stage_data.var_index("evolved", "velocity");
  static const int idx_ener =
      stage_data.var_index("evolved", "specific_total_fluid_energy");
  static const int idx_er =
      stage_data.var_index("evolved", "specific_radiation_energy");
  static const int idx_fr =
      stage_data.var_index("evolved", "specific_radiation_flux");
  static const int idx_vstar =
      stage_data.var_index("interface", "interface_velocity");
  static const int idx_tgas =
      stage_data.var_index("derived", "gas_temperature");
  static constexpr int u_idx_tau = 0;
  static constexpr int u_idx_vel = 1;
  static constexpr int u_idx_ener = 2;
  static constexpr int u_idx_er = 3;
  static constexpr int u_idx_fr = 4;

  auto evolved = stage_data.get_field("evolved");
  auto derived = stage_data.get_field("derived");
  auto interface = stage_data.get_field<AthelasArray2D<double>>("interface");
  const auto &eos = stage_data.eos();
  const auto &opac = stage_data.opac();
  const double ap_coefficient = params_.get<double>("ap_coefficient");
  // C = 0 recovers standard LLF; skip the per-face opacity work entirely.
  const bool ap_correction = ap_coefficient > 0.0;

  auto phi = basis.phi();
  auto dphi = basis.dphi();
  auto mkk = basis.mass_matrix();
  auto dr = mesh.widths();
  auto weights = mesh.weights();
  auto sqrt_gm = mesh.sqrt_gm();
  const auto rad_bcs = bc::radiation_bc_data(bcs_);

  static const bool ionization_enabled = stage_data.enabled("ionization");
  AthelasArray2D<double> number_density;
  AthelasArray2D<double> ye;
  AthelasArray2D<double> ybar;
  AthelasArray2D<double> sigma1;
  AthelasArray2D<double> sigma2;
  AthelasArray2D<double> sigma3;
  AthelasArray2D<double> e_ion_corr;
  AthelasArray3D<double> bulk;
  const bool composition_enabled = stage_data.enabled("composition");
  if (ionization_enabled) {
    const auto *const ionization_state = stage_data.ionization_state();
    const auto *const comps = stage_data.comps();
    number_density = comps->number_density();
    ye = comps->ye();
    ybar = ionization_state->ybar();
    sigma1 = ionization_state->sigma1();
    sigma2 = ionization_state->sigma2();
    sigma3 = ionization_state->sigma3();
    e_ion_corr = ionization_state->e_ion_corr();
  }
  if (composition_enabled || ionization_enabled) {
    bulk = stage_data.get_field("bulk_composition");
  }

  // Refresh halo cells so face states have valid stencil data.
  bc::ghost_fill(U, stage_data, bcs_);

  // Face states: faces_.u_f_l = U at right edge of cell i-1, faces_.u_f_r = U
  // at left edge of cell i. Indices on second axis: 0 = tau, 1 = E_specific, 2
  // = F_specific.
  athelas::par_for(
      DEFAULT_FLAT_LOOP_PATTERN, "ImplicitMoments :: residual :: faces",
      DevExecSpace(), ib.s, ib.e + 1, KOKKOS_CLASS_LAMBDA(const int i) {
        faces_.u_f_l(i, 0) =
            basis_eval<Interface::Right>(phi, U, i - 1, u_idx_tau);
        faces_.u_f_r(i, 0) = basis_eval<Interface::Left>(phi, U, i, u_idx_tau);
        faces_.u_f_l(i, 1) =
            basis_eval<Interface::Right>(phi, U, i - 1, u_idx_er);
        faces_.u_f_r(i, 1) = basis_eval<Interface::Left>(phi, U, i, u_idx_er);
        faces_.u_f_l(i, 2) =
            basis_eval<Interface::Right>(phi, U, i - 1, u_idx_fr);
        faces_.u_f_r(i, 2) = basis_eval<Interface::Left>(phi, U, i, u_idx_fr);
      });

  // Numerical flux at every interior face.
  athelas::par_for(
      DEFAULT_FLAT_LOOP_PATTERN, "ImplicitMoments :: residual :: flux_num",
      DevExecSpace(), ib.s, ib.e + 1, KOKKOS_CLASS_LAMBDA(const int i) {
        const double rho_L = 1.0 / faces_.u_f_l(i, 0);
        const double rho_R = 1.0 / faces_.u_f_r(i, 0);
        const double vstar = interface(i, idx_vstar);

        const double E_L = faces_.u_f_l(i, 1) * rho_L;
        const double E_R = faces_.u_f_r(i, 1) * rho_R;
        const double F_L = faces_.u_f_l(i, 2) * rho_L;
        const double F_R = faces_.u_f_r(i, 2) * rho_R;

        double beta = 1.0;
        if (ap_correction) {
          double tau_face = 0.0;
          if (i == ib.s) {
            tau_face =
                cell_optical_depth(opac, derived, bulk, composition_enabled,
                                   idx_tgas, i, 0, rho_R, dr(i));
          } else if (i == ib.e + 1) {
            tau_face = cell_optical_depth(opac, derived, bulk,
                                          composition_enabled, idx_tgas, i - 1,
                                          nNodes + 1, rho_L, dr(i - 1));
          } else {
            const double tau_L = cell_optical_depth(
                opac, derived, bulk, composition_enabled, idx_tgas, i - 1,
                nNodes + 1, rho_L, dr(i - 1));
            const double tau_R =
                cell_optical_depth(opac, derived, bulk, composition_enabled,
                                   idx_tgas, i, 0, rho_R, dr(i));
            tau_face = face_optical_depth(tau_L, tau_R);
          }
          beta = ap_dissipation_factor(tau_face, ap_coefficient);
        }
        faces_.beta(i) = beta;

        const auto flux = numerical_flux_rad_with_boundary(
            i, ib.s, ib.e + 1, rad_bcs, RadBoundaryState{.E = E_L, .F = F_L},
            RadBoundaryState{.E = E_R, .F = F_R}, vstar, beta);

        faces_.flux_num(i, 0) = flux.e;
        faces_.flux_num(i, 1) = flux.f;
      });

  // Assemble residual into b_out, with the existing sign convention
  // b_out = -R = -(M(U-U*) - dt_aii * T - dt_aii * S).
  athelas::par_for(
      DEFAULT_LOOP_PATTERN, "ImplicitMoments :: residual :: assemble",
      DevExecSpace(), ib.s, ib.e, KOKKOS_CLASS_LAMBDA(const int i) {
        const int blk = i - ib.s;
        const double vstar = interface(i, idx_vstar);

        for (int q = 0; q < nNodes; ++q) {
          // Surface contribution.
          double rhs_e =
              -(faces_.flux_num(i + 1, 0) * phi(i, nNodes + 1, q) *
                    sqrt_gm(i, nNodes + 1) -
                faces_.flux_num(i, 0) * phi(i, 0, q) * sqrt_gm(i, 0));
          double rhs_f =
              -(faces_.flux_num(i + 1, 1) * phi(i, nNodes + 1, q) *
                    sqrt_gm(i, nNodes + 1) -
                faces_.flux_num(i, 1) * phi(i, 0, q) * sqrt_gm(i, 0));

          // Volume contribution.
          for (int p = 0; p < nNodes; ++p) {
            const double rho = 1.0 / evolved(i, p, idx_tau);
            const double e_rad = U(i, p, u_idx_er) * rho;
            const double f_rad = U(i, p, u_idx_fr) * rho;
            const double p_rad = compute_closure(e_rad, f_rad);
            const auto [flux_e, flux_f] = flux_rad(e_rad, f_rad, p_rad, vstar);
            const double w_dphi_sqrtgm =
                weights(p) * dphi(i, p + 1, q) * sqrt_gm(i, p + 1);
            rhs_e += w_dphi_sqrtgm * flux_e;
            rhs_f += w_dphi_sqrtgm * flux_f;
          }

          const double m = mkk(i, q);
          eos::EOSLambda lambda;
          double X = 0.0;
          double Z = 0.0;
          if (ionization_enabled) {
            lambda.data[0] = number_density(i, q + 1);
            lambda.data[1] = ye(i, q + 1);
            lambda.data[2] = ybar(i, q + 1);
            lambda.data[3] = sigma1(i, q + 1);
            lambda.data[4] = sigma2(i, q + 1);
            lambda.data[5] = sigma3(i, q + 1);
            lambda.data[6] = e_ion_corr(i, q + 1);
            lambda.data[7] = derived(i, q + 1, idx_tgas);
            X = bulk(i, q + 1, 0);
            Z = bulk(i, q + 1, 2);
          }
          const RadSourceInputs src_in{.rho = 1.0 / evolved(i, q, idx_tau),
                                       .e = U(i, q, u_idx_ener),
                                       .v = U(i, q, u_idx_vel),
                                       .erad = U(i, q, u_idx_er),
                                       .frad = U(i, q, u_idx_fr),
                                       .etot = 0.0,
                                       .m_tot = 0.0,
                                       .X = X,
                                       .Z = Z,
                                       .dt_a_ii = dt_aii,
                                       .eos = &eos,
                                       .opac = &opac};
          const RadHydroSources src(src_in, lambda.ptr());
          const int row_e = q * 4 + 0;
          const int row_f = q * 4 + 1;
          const int row_v = q * 4 + 2;
          const int row_E = q * 4 + 3;

          b_out(blk, row_e) = -(m * (U(i, q, u_idx_er) - ustar(i, q, idx_er)) -
                                dt_aii * rhs_e - dt_aii * m * src.s_er);
          b_out(blk, row_f) = -(m * (U(i, q, u_idx_fr) - ustar(i, q, idx_fr)) -
                                dt_aii * rhs_f - dt_aii * m * src.s_fr);
          b_out(blk, row_v) =
              -(m * (U(i, q, u_idx_vel) - ustar(i, q, idx_vel)) -
                dt_aii * m * src.s_v);
          b_out(blk, row_E) =
              -(m * (U(i, q, u_idx_ener) - ustar(i, q, idx_ener)) -
                dt_aii * m * src.s_eg);
        }
      });
}

auto ImplicitRadiationMomentsPackage::update_implicit(
    const StageData &stage_data, AthelasArray3D<double> ustar,
    const TimeStepInfo &dt_info) -> UpdateStatus {
  const auto &mesh = stage_data.mesh();
  using math::linalg::ThomasScratch, math::linalg::block_thomas_solve;
  using math::utils::sgn;

  const auto &basis = stage_data.basis();
  const int nNodes = mesh.n_nodes();
  static const IndexRange ib(mesh.domain<Domain::Interior>());
  static const IndexRange qb(nNodes);

  auto evolved = stage_data.get_field("evolved");
  auto derived = stage_data.get_field("derived");
  auto interface = stage_data.get_field<AthelasArray2D<double>>("interface");

  static const int idx_tau = stage_data.var_index("evolved", "specific_volume");
  static const int idx_vel = stage_data.var_index("evolved", "velocity");
  static const int idx_ener =
      stage_data.var_index("evolved", "specific_total_fluid_energy");
  static const int idx_er =
      stage_data.var_index("evolved", "specific_radiation_energy");
  static const int idx_fr =
      stage_data.var_index("evolved", "specific_radiation_flux");
  static const int idx_vstar =
      stage_data.var_index("interface", "interface_velocity");
  static const int idx_tgas =
      stage_data.var_index("derived", "gas_temperature");
  static constexpr int u_idx_tau = 0;
  static constexpr int u_idx_vel = 1;
  static constexpr int u_idx_ener = 2;
  static constexpr int u_idx_er = 3;
  static constexpr int u_idx_fr = 4;

  const double dt_aii = dt_info.dt_coef;

  auto phi = basis.phi();
  auto dphi = basis.dphi();
  auto mkk = basis.mass_matrix();
  auto dr = mesh.widths();
  auto weights = mesh.weights();
  auto sqrt_gm = mesh.sqrt_gm();

  const auto &eos = stage_data.eos();
  const auto &opac = stage_data.opac();
  const bool paczynski_eos =
      std::holds_alternative<athelas::eos::Paczynski>(eos);
  static const bool ionization_enabled = stage_data.enabled("ionization");
  AthelasArray2D<double> number_density;
  AthelasArray2D<double> ye;
  AthelasArray2D<double> ybar;
  AthelasArray2D<double> sigma1;
  AthelasArray2D<double> sigma2;
  AthelasArray2D<double> sigma3;
  AthelasArray2D<double> e_ion_corr;
  AthelasArray3D<double> bulk;
  if (ionization_enabled) {
    const auto *const ionization_state = stage_data.ionization_state();
    const auto *const comps = stage_data.comps();
    number_density = comps->number_density();
    ye = comps->ye();
    ybar = ionization_state->ybar();
    sigma1 = ionization_state->sigma1();
    sigma2 = ionization_state->sigma2();
    sigma3 = ionization_state->sigma3();
    e_ion_corr = ionization_state->e_ion_corr();
    bulk = stage_data.get_field("bulk_composition");
  }

  // Now work through the implicit transport solve.
  // Initial guess: ustar

  ThomasScratch scratch{
      .W = solver_.W,
      .Y = solver_.Y,
      .Bi_lu = solver_.Bi_lu,
      .piv = solver_.piv,
      .row_scale = solver_.row_scale,
      .col_scale = solver_.col_scale,
  };

  athelas::par_for(
      DEFAULT_LOOP_PATTERN, "ImplicitMoments :: Newton :: Guess",
      DevExecSpace(), ib.s, ib.e, qb.s, qb.e,
      KOKKOS_CLASS_LAMBDA(const int i, const int q) {
        newton_.u_rad_work(i, q, u_idx_tau) = ustar(i, q, idx_tau);
        newton_.u_rad_work(i, q, u_idx_vel) = ustar(i, q, idx_vel);
        newton_.u_rad_work(i, q, u_idx_ener) = ustar(i, q, idx_ener);
        newton_.u_rad_work(i, q, u_idx_er) = ustar(i, q, idx_er);
        newton_.u_rad_work(i, q, u_idx_fr) = ustar(i, q, idx_fr);
      });

  const int block_size = 4 * nNodes;

  // Flat DOF index within a block (node-major: v fastest)
  auto idx = [&](const int q, const int v) { return q * 4 + v; };

  // Left/Right face states
  // Extract left/right interface states for necessary vars.
  // These refer to the left/right states on the left interface of element i.
  // v = 0: specific volume
  // v = 1: specific radiation energy density
  // v = 2: specific radiation flux

  constexpr double c = constants::c_cgs;
  constexpr double c2 = c * c;

  bc::ghost_fill(evolved, stage_data, bcs_);
  bc::ghost_fill(ustar, stage_data, bcs_);

  int iter = 0;
  double norm_resid = 1.0;

  static auto max_iters = params_.get<int>("newton.max_iters");
  static auto tol = params_.get<double>("newton.tol");
  static auto atol = params_.get<double>("newton.atol");

  // Converged is the gold standard corresponding to a converged residual.
  // Accepted is "good enough" -- perhaps a "converged" or stalled step size
  // without a converged residual.
  bool converged = false;
  bool accepted = false;

  // Stagnation constrol. After applying the line-searched
  // step, check ||lam * Δ U||_inf scaled by the per-component magnitude.
  // If the step is below this tolerance, the iterate can't move
  // meaningfully in floating point: declare "acceptable" and exit.
  constexpr double step_tol = 1.0e-11; // ~eps^(2/3) for double precision
  constexpr double U_floor = 1.0e-14;
  // Keep the line-search admissibility bound consistent with the lower end of
  // the Paczynski/Saha temperature-inversion bracket.
  constexpr double T_inversion_floor = 500.0;
  constexpr double T_inversion_ceiling = 1.0e12;

  // Project the initial guess into the realizable set. The line-search trial
  // converges to the initial iterate as lam -> 0, so an inadmissible guess
  // makes every trial inadmissible and the solve can never move: the
  // radiation field silently freezes. Nodal values only; face-edge
  // realizability for nnodes > 1 remains the line search's job.
  constexpr double f_proj = 1.0 - 1.0e-6; // keep |F| strictly inside c E
  int n_projected = 0;
  athelas::par_reduce(
      DEFAULT_LOOP_PATTERN, "ImplicitMoments :: Newton :: Project guess",
      DevExecSpace(), ib.s, ib.e, qb.s, qb.e,
      KOKKOS_CLASS_LAMBDA(const int i, const int q, int &count) {
        double er = newton_.u_rad_work(i, q, u_idx_er);
        double fr = newton_.u_rad_work(i, q, u_idx_fr);
        const double eg = newton_.u_rad_work(i, q, u_idx_ener);
        const double vg = newton_.u_rad_work(i, q, u_idx_vel);
        const double sie = eg - 0.5 * vg * vg;
        bool projected = false;

        if (!(er > 0.0)) {
          er = U_floor;
          projected = true;
        }
        if (std::abs(fr) > f_proj * c * er) {
          fr = sgn(fr) * f_proj * c * er;
          projected = true;
        }

        const double rho = 1.0 / evolved(i, q, idx_tau);
        eos::EOSLambda lam_eos;
        if (ionization_enabled) {
          lam_eos.data[0] = number_density(i, q + 1);
          lam_eos.data[1] = ye(i, q + 1);
          lam_eos.data[2] = ybar(i, q + 1);
          lam_eos.data[3] = sigma1(i, q + 1);
          lam_eos.data[4] = sigma2(i, q + 1);
          lam_eos.data[5] = sigma3(i, q + 1);
          lam_eos.data[6] = e_ion_corr(i, q + 1);
          lam_eos.data[7] = derived(i, q + 1, idx_tgas);
        }
        const double emin =
            paczynski_eos ? eos::sie_from_density_temperature(
                                eos, rho, T_inversion_floor, lam_eos.ptr())
                          : min_sie(eos, rho, lam_eos.ptr());
        const double emax =
            paczynski_eos ? eos::sie_from_density_temperature(
                                eos, rho, T_inversion_ceiling, lam_eos.ptr())
                          : std::numeric_limits<double>::infinity();
        // Twice the realizability-check margin so a projected node passes it.
        const double e_margin = 128.0 * std::numeric_limits<double>::epsilon() *
                                std::max({std::abs(eg), std::abs(emin), 1.0});
        double sie_proj = sie;
        if (sie < emin + e_margin) {
          sie_proj = emin + e_margin;
          projected = true;
        } else if (sie > emax - e_margin) {
          sie_proj = emax - e_margin;
          projected = true;
        }

        if (projected) {
          newton_.u_rad_work(i, q, u_idx_er) = er;
          newton_.u_rad_work(i, q, u_idx_fr) = fr;
          newton_.u_rad_work(i, q, u_idx_ener) = sie_proj + 0.5 * vg * vg;
          count += 1;
        }
      },
      Kokkos::Sum<int>(n_projected));
  n_guess_projections_ += n_projected;

  double resid0 = 0.0;
  bool any_step_applied = false;
  while (iter < max_iters && !converged && !accepted) {

    Kokkos::deep_copy(solver_.mat_diag, 0.0);
    Kokkos::deep_copy(solver_.mat_upper, 0.0);
    Kokkos::deep_copy(solver_.mat_lower, 0.0);

    // Compute residual (also fills faces_.u_f_l, faces_.u_f_r, faces_.flux_num
    // used below).
    evaluate_residual(solver_.b, newton_.u_rad_work, ustar, stage_data, mesh,
                      dt_aii);

    athelas::par_for(
        DEFAULT_FLAT_LOOP_PATTERN, "ImplicitMoments :: Interface Jacobians",
        DevExecSpace(), ib.s, ib.e + 1, KOKKOS_CLASS_LAMBDA(const int i) {
          // States at the interface i+1/2 (L is i, R is i+1)
          const double rho_L = 1.0 / faces_.u_f_l(i, 0);
          const double rho_R = 1.0 / faces_.u_f_r(i, 0);
          const double vstar = interface(i, idx_vstar);

          // Radiation specific variables
          const double E_L = faces_.u_f_l(i, 1) * rho_L;
          const double E_R = faces_.u_f_r(i, 1) * rho_R;
          const double F_L = faces_.u_f_l(i, 2) * rho_L;
          const double F_R = faces_.u_f_r(i, 2) * rho_R;

          const double alpha = rad_wavespeed(E_L, E_R, F_L, F_R, vstar);
          const double alpha_ap = faces_.beta(i) * alpha;
          const double f_l =
              flux_factor(faces_.u_f_l(i, 1), faces_.u_f_l(i, 2));
          const double f_r =
              flux_factor(faces_.u_f_r(i, 1), faces_.u_f_r(i, 2));
          const double chi_L = eddington_factor(f_l);
          const double chi_R = eddington_factor(f_r);
          const double chi_prime_L = eddington_factor_prime(f_l);
          const double chi_prime_R = eddington_factor_prime(f_r);

          // A_minus = d(F_hat)/d(U_local) = 0.5 * (J_local + alpha * I)
          faces_.A_minus(i - ib.s, 0, 0) = 0.5 * (-vstar + alpha_ap) * rho_L;
          faces_.A_minus(i - ib.s, 0, 1) = 0.5 * rho_L;
          faces_.A_minus(i - ib.s, 1, 0) =
              0.5 * (c2 * (chi_L - f_l * chi_prime_L)) * rho_L;
          faces_.A_minus(i - ib.s, 1, 1) =
              0.5 * (c * chi_prime_L * sgn(F_L) - vstar + alpha_ap) * rho_L;

          // A_plus = d(F_hat)/d(U_neighbor) = 0.5 * (J_neighbor - alpha * I)
          faces_.A_plus(i - ib.s, 0, 0) = 0.5 * (-vstar - alpha_ap) * rho_R;
          faces_.A_plus(i - ib.s, 0, 1) = 0.5 * rho_R;
          faces_.A_plus(i - ib.s, 1, 0) =
              0.5 * (c2 * (chi_R - f_r * chi_prime_R)) * rho_R;
          faces_.A_plus(i - ib.s, 1, 1) =
              0.5 * (c * chi_prime_R * sgn(F_R) - vstar - alpha_ap) * rho_R;
        });

    athelas::par_for(
        DEFAULT_LOOP_PATTERN, "ImplicitMoments :: Assemble solver_mat",
        DevExecSpace(), ib.s, ib.e, KOKKOS_CLASS_LAMBDA(const int i) {
          const int blk = i - ib.s;

          // Mass matrix - diagonal block only
          for (int q = 0; q < nNodes; ++q) {
            const double m = mkk(i, q);
            for (int v = 0; v < 4; ++v) {
              const int row = idx(q, v);
              solver_.mat_diag(blk, row, row) += m;
            }

            eos::EOSLambda lambda;
            double X = 0.0;
            double Z = 0.0;
            if (ionization_enabled) {
              lambda.data[0] = number_density(i, q + 1);
              lambda.data[1] = ye(i, q + 1);
              lambda.data[2] = ybar(i, q + 1);
              lambda.data[3] = sigma1(i, q + 1);
              lambda.data[4] = sigma2(i, q + 1);
              lambda.data[5] = sigma3(i, q + 1);
              lambda.data[6] = e_ion_corr(i, q + 1);
              lambda.data[7] = derived(i, q + 1, idx_tgas);
              X = bulk(i, q + 1, 0);
              Z = bulk(i, q + 1, 2);
            }
            const RadSourceInputs src_in{
                .rho = 1.0 / evolved(i, q, idx_tau),
                .e = newton_.u_rad_work(i, q, u_idx_ener),
                .v = newton_.u_rad_work(i, q, u_idx_vel),
                .erad = newton_.u_rad_work(i, q, u_idx_er),
                .frad = newton_.u_rad_work(i, q, u_idx_fr),
                .etot = 0.0,
                .m_tot = 0.0,
                .X = X,
                .Z = Z,
                .dt_a_ii = dt_aii,
                .eos = &eos,
                .opac = &opac};
            const RadHydroSourceDerivatives src_d(src_in, lambda.ptr());

            const int row_er = idx(q, 0);
            const int row_fr = idx(q, 1);
            const int row_v = idx(q, 2);
            const int row_eg = idx(q, 3);
            const int col_er = idx(q, 0);
            const int col_fr = idx(q, 1);
            const int col_v = idx(q, 2);
            const int col_eg = idx(q, 3);

            solver_.mat_diag(blk, row_er, col_er) += dt_aii * m * src_d.dseder;
            solver_.mat_diag(blk, row_er, col_fr) += dt_aii * m * src_d.dsedfr;
            solver_.mat_diag(blk, row_er, col_v) += dt_aii * m * src_d.dsedv;
            solver_.mat_diag(blk, row_er, col_eg) += dt_aii * m * src_d.dsedeg;

            solver_.mat_diag(blk, row_fr, col_er) +=
                dt_aii * m * c2 * src_d.dsvder;
            solver_.mat_diag(blk, row_fr, col_fr) +=
                dt_aii * m * c2 * src_d.dsvdfr;
            solver_.mat_diag(blk, row_fr, col_v) +=
                dt_aii * m * c2 * src_d.dsvdv;
            solver_.mat_diag(blk, row_fr, col_eg) +=
                dt_aii * m * c2 * src_d.dsvdeg;

            solver_.mat_diag(blk, row_v, col_er) -= dt_aii * m * src_d.dsvder;
            solver_.mat_diag(blk, row_v, col_fr) -= dt_aii * m * src_d.dsvdfr;
            solver_.mat_diag(blk, row_v, col_v) -= dt_aii * m * src_d.dsvdv;
            solver_.mat_diag(blk, row_v, col_eg) -= dt_aii * m * src_d.dsvdeg;

            solver_.mat_diag(blk, row_eg, col_er) -= dt_aii * m * src_d.dseder;
            solver_.mat_diag(blk, row_eg, col_fr) -= dt_aii * m * src_d.dsedfr;
            solver_.mat_diag(blk, row_eg, col_v) -= dt_aii * m * src_d.dsedv;
            solver_.mat_diag(blk, row_eg, col_eg) -= dt_aii * m * src_d.dsedeg;
          }

          // Volume term - diagonal block
          // K_vol[q*2+v, p*2+w] = [D^T W]_{qp} * J_vol[v,w](x_p)
          // J_vol = rho[[-v, 1], [s^2, -v]] with s^2 = c^2 * chi
          const double vstar = interface(i, idx_vstar);
          for (int q = 0; q < nNodes; ++q) {
            for (int v = 0; v < 2; ++v) {
              const int row = idx(q, v);
              for (int p = 0; p < nNodes; ++p) {
                const double rhop = 1.0 / evolved(i, p, idx_tau);
                const double f =
                    flux_factor(newton_.u_rad_work(i, p, u_idx_er),
                                newton_.u_rad_work(i, p, u_idx_fr));
                const double chi = eddington_factor(f);
                const double sp2 = c2 * chi;
                const double chi_prime = eddington_factor_prime(f);
                for (int w = 0; w < 2; ++w) {
                  const int col = idx(p, w);

                  double A_vw = 1.0;
                  if (v == 0 && w == 0) {
                    A_vw = -vstar;
                  } else if (v == 1 && w == 1) {
                    A_vw = c * chi_prime *
                               sgn(newton_.u_rad_work(i, p, u_idx_fr)) -
                           vstar;
                  } else if (v == 1 && w == 0) {
                    A_vw = sp2 - c2 * f * chi_prime;
                  }

                  solver_.mat_diag(blk, row, col) -=
                      dt_aii * dphi(i, p + 1, q) * weights(p) *
                      sqrt_gm(i, p + 1) * A_vw * rhop;
                }
              }
            }
          }

          const double gL = sqrt_gm(i, 0);
          const double gR = sqrt_gm(i, nNodes + 1);

          for (int q = 0; q < nNodes; ++q) {
            const double ellL_q = phi(i, 0, q);
            const double ellR_q = phi(i, nNodes + 1, q);
            for (int p = 0; p < nNodes; ++p) {
              const double ellL_p = phi(i, 0, p);
              const double ellR_p = phi(i, nNodes + 1, p);

              for (int v = 0; v < 2; ++v) {
                const int row = idx(q, v);
                for (int w = 0; w < 2; ++w) {
                  const int col = idx(p, w);

                  // --- right face ---
                  if (i < ib.e) {
                    const double ellL_p_nbr = phi(i + 1, 0, p);
                    const int ifaceR = i - ib.s + 1;

                    solver_.mat_diag(blk, row, col) +=
                        dt_aii * ellR_q * gR * faces_.A_minus(ifaceR, v, w) *
                        ellR_p;

                    solver_.mat_upper(blk, row, col) +=
                        dt_aii * ellR_q * gR * faces_.A_plus(ifaceR, v, w) *
                        ellL_p_nbr;
                  }

                  // --- left face ---
                  if (i > ib.s) {
                    const double ellR_p_nbr = phi(i - 1, nNodes + 1, p);
                    const int ifaceL = i - ib.s;

                    solver_.mat_diag(blk, row, col) -=
                        dt_aii * ellL_q * gL * faces_.A_plus(ifaceL, v, w) *
                        ellL_p;

                    solver_.mat_lower(blk - 1, row, col) -=
                        dt_aii * ellL_q * gL * faces_.A_minus(ifaceL, v, w) *
                        ellR_p_nbr;
                  }
                }
              }
            }
          }
        });

    const auto rad_bcs = bc::radiation_bc_data(bcs_);
    static const int nblocks = mesh.n_elements();
    static const int i_inner_face = 1;
    static const int i_outer_face = nblocks + 1;
    static const int i_inner_cell = ib.s; // first interior cell
    static const int i_outer_cell = ib.e; // last interior cell
    athelas::par_for(
        DEFAULT_LOOP_PATTERN,
        "ImplicitMoments :: Assemble solver_mat :: boundaries", DevExecSpace(),
        0, 0, KOKKOS_CLASS_LAMBDA(const int) {
          const double vstar_i = interface(i_inner_face, idx_vstar);
          const double vstar_o = interface(i_outer_face, idx_vstar);
          const double rho_i = 1.0 / faces_.u_f_r(i_inner_face, 0);
          const double rho_o = 1.0 / faces_.u_f_l(i_outer_face, 0);
          const double beta_i = faces_.beta(i_inner_face);
          const double beta_o = faces_.beta(i_outer_face);
          const double gL_i = sqrt_gm(i_inner_cell, 0);
          const double gR_o = sqrt_gm(i_outer_cell, nNodes + 1);

          // --- inner boundary ---
          const auto inner_bc = rad_bcs[0];
          const RadBoundaryState inner_state{
              .E = faces_.u_f_r(i_inner_face, 1) * rho_i,
              .F = faces_.u_f_r(i_inner_face, 2) * rho_i};

          // Row factor: basis at left edge of the interior cell.
          int blk = 0;
          for (int q = 0; q < nNodes; ++q) {
            const double ellL_q = phi(i_inner_cell, 0, q);
            for (int p = 0; p < nNodes; ++p) {
              const double phi_p_direct = phi(i_inner_cell, 0, p);

              for (int v = 0; v < 2; ++v) {
                const int row = idx(q, v);
                for (int w = 0; w < 2; ++w) {
                  const int col = idx(p, w);
                  const double A_vw = boundary_flux_jacobian(
                      Boundary::Interior, inner_bc.type, inner_state, vstar_i,
                      rho_i, v, w, inner_bc.marshak_incoming_energy, beta_i);
                  // Inner boundary residual has +flux_num(face)*phi_L, so
                  // dR/dU contributes with a leading minus sign from
                  // R = M(U-U*) - dt*rhs.
                  solver_.mat_diag(blk, row, col) -=
                      dt_aii * ellL_q * gL_i * A_vw * phi_p_direct;
                }
              }
            }
          }

          // --- outer boundary ---
          const auto outer_bc = rad_bcs[1];
          const RadBoundaryState outer_state{
              .E = faces_.u_f_l(i_outer_face, 1) * rho_o,
              .F = faces_.u_f_l(i_outer_face, 2) * rho_o};

          // Row factor: basis at right edge of the interior cell.
          blk = nblocks - 1;
          for (int q = 0; q < nNodes; ++q) {
            const double ellR_q = phi(i_outer_cell, nNodes + 1, q);
            for (int p = 0; p < nNodes; ++p) {
              const double phi_p_direct = phi(i_outer_cell, nNodes + 1, p);

              for (int v = 0; v < 2; ++v) {
                const int row = idx(q, v);
                for (int w = 0; w < 2; ++w) {
                  const int col = idx(p, w);
                  const double A_vw = boundary_flux_jacobian(
                      Boundary::Exterior, outer_bc.type, outer_state, vstar_o,
                      rho_o, v, w, outer_bc.marshak_incoming_energy, beta_o);
                  // Outer boundary residual has -flux_num(face)*phi_R,
                  // so dR/dU contributes with a leading plus sign.
                  solver_.mat_diag(blk, row, col) +=
                      dt_aii * ellR_q * gR_o * A_vw * phi_p_direct;
                }
              }
            }
          }
        });

    norm_resid = math::linalg::newton_norm_l2(solver_.b, sqrt_gm, dr, weights);
    if (!std::isfinite(norm_resid)) {
      throw_athelas_error("Implicit radiation Newton residual is nonfinite.");
    }

    if (iter == 0) {
      resid0 = norm_resid;
    }

    // Should we break here or compute and apply the update?
    if (norm_resid == 0.0 || norm_resid < atol || norm_resid / resid0 < tol) {
      converged = true;
      break;
    }

    // Perform the tridiagonal solve: overwrites contains of b and diag
    block_thomas_solve(nblocks, block_size, solver_.mat_lower, solver_.mat_diag,
                       solver_.mat_upper, solver_.b, scratch);

    // Fraction-to-the-boundary bound: the largest step along the Newton
    // direction keeping every node and face evaluation strictly inside the
    // realizable set, in closed form. Every constraint is linear in the step
    // length except the sie bracket, which is quadratic through the kinetic
    // energy. Starting the line search here means a strictly interior
    // iterate can never produce an empty step: halving from lam = 1 bottoms
    // out at ~4e-6 and gives up, which froze the solve whenever a cell rode
    // the causal boundary |F| ~ c E with a thinner gap than that.
    double alpha_fb = 1.0;
    athelas::par_reduce(
        DEFAULT_FLAT_LOOP_PATTERN, "ImplicitMoments :: Newton :: Step bound",
        DevExecSpace(), ib.s, ib.e,
        KOKKOS_CLASS_LAMBDA(const int i, double &lmin) {
          double amax = 1.0;
          // Largest alpha with g0 + alpha * slope > 0. Skipped when the
          // iterate is already outside (g0 <= 0); the realizability check in
          // the line search remains the safety net there.
          auto shrink_linear = [&](const double g0, const double slope) {
            if (g0 > 0.0 && slope < 0.0) {
              amax = std::min(amax, g0 / (-slope));
            }
          };

          for (int q = qb.s; q <= qb.e; ++q) {
            const double er = newton_.u_rad_work(i, q, u_idx_er);
            const double fr = newton_.u_rad_work(i, q, u_idx_fr);
            const double eg = newton_.u_rad_work(i, q, u_idx_ener);
            const double vg = newton_.u_rad_work(i, q, u_idx_vel);
            const double der = solver_.b(i - ib.s, idx(q, 0));
            const double dfr = solver_.b(i - ib.s, idx(q, 1));
            const double dvg = solver_.b(i - ib.s, idx(q, 2));
            const double deg = solver_.b(i - ib.s, idx(q, 3));

            // E > 0 and |F| < c E, all linear in alpha.
            shrink_linear(er, der);
            shrink_linear(c * er - fr, c * der - dfr);
            shrink_linear(c * er + fr, c * der + dfr);

            // sie bracket: sie(alpha) = sie0 + b_lin alpha - a_quad alpha²/2.
            const double sie0 = eg - 0.5 * vg * vg;
            const double a_quad = dvg * dvg;
            const double b_lin = deg - vg * dvg;

            const double rho = 1.0 / evolved(i, q, idx_tau);
            eos::EOSLambda lam_eos;
            if (ionization_enabled) {
              lam_eos.data[0] = number_density(i, q + 1);
              lam_eos.data[1] = ye(i, q + 1);
              lam_eos.data[2] = ybar(i, q + 1);
              lam_eos.data[3] = sigma1(i, q + 1);
              lam_eos.data[4] = sigma2(i, q + 1);
              lam_eos.data[5] = sigma3(i, q + 1);
              lam_eos.data[6] = e_ion_corr(i, q + 1);
              lam_eos.data[7] = derived(i, q + 1, idx_tgas);
            }
            const double emin =
                paczynski_eos ? eos::sie_from_density_temperature(
                                    eos, rho, T_inversion_floor, lam_eos.ptr())
                              : min_sie(eos, rho, lam_eos.ptr());
            const double e_margin =
                64.0 * std::numeric_limits<double>::epsilon() *
                std::max({std::abs(eg), std::abs(emin), 1.0});

            // Lower edge: smallest positive root of sie(alpha) = emin + m.
            const double c_lo = sie0 - (emin + e_margin);
            if (c_lo > 0.0) {
              if (a_quad > 0.0) {
                const double s = std::sqrt(b_lin * b_lin + 2.0 * a_quad * c_lo);
                amax = std::min(amax, b_lin > 0.0 ? (b_lin + s) / a_quad
                                                  : 2.0 * c_lo / (s - b_lin));
              } else if (b_lin < 0.0) {
                amax = std::min(amax, c_lo / (-b_lin));
              }
            }

            // Upper edge exists only for the Paczynski inversion bracket.
            if (paczynski_eos) {
              const double emax = eos::sie_from_density_temperature(
                  eos, rho, T_inversion_ceiling, lam_eos.ptr());
              const double c_hi = (emax - e_margin) - sie0;
              if (c_hi > 0.0 && b_lin > 0.0) {
                if (a_quad > 0.0) {
                  const double disc = b_lin * b_lin - 2.0 * a_quad * c_hi;
                  if (disc >= 0.0) {
                    amax =
                        std::min(amax, 2.0 * c_hi / (b_lin + std::sqrt(disc)));
                  }
                } else {
                  amax = std::min(amax, c_hi / b_lin);
                }
              }
            }
          }

          // Face-edge evaluations (E > 0, |F| < c E) are linear in alpha too.
          for (int side = 0; side < 2; ++side) {
            const int qf = side == 0 ? 0 : nNodes + 1;
            double er0 = 0.0;
            double fr0 = 0.0;
            double der = 0.0;
            double dfr = 0.0;
            for (int p = 0; p < nNodes; ++p) {
              const double ph = phi(i, qf, p);
              er0 += ph * newton_.u_rad_work(i, p, u_idx_er);
              fr0 += ph * newton_.u_rad_work(i, p, u_idx_fr);
              der += ph * solver_.b(i - ib.s, idx(p, 0));
              dfr += ph * solver_.b(i - ib.s, idx(p, 1));
            }
            shrink_linear(er0, der);
            shrink_linear(c * er0 - fr0, c * der - dfr);
            shrink_linear(c * er0 + fr0, c * der + dfr);
          }

          lmin = std::min(lmin, amax);
        },
        Kokkos::Min<double>(alpha_fb));

    // Realizability + Armijo line search with stagnation cutoff.
    //
    // Realizability: the M1 closure requires E > 0 and |F| < c E. The
    // fraction-to-the-boundary start makes the full first trial realizable
    // up to floating-point slop; halving remains as the safety net.
    //
    // Armijo: once realizable, the trial squared residual must drop by at
    // least 2·alpha·lam relative to F0² (sufficient decrease).
    //
    // Stagnation: If step becomes too small, call the result accepted.
    constexpr int max_ls = 18;
    constexpr double alpha_armijo = 1.0e-4;
    const double F0_sq = norm_resid * norm_resid;

    // Trial state starts from the current Newton iterate; all solved variables
    // are filled per line-search iteration.
    Kokkos::deep_copy(newton_.u_rad_trial, newton_.u_rad_work);

    // Keep a hair inside the boundary.
    constexpr double tau_fb = 0.99;
    double lam = std::min(1.0, tau_fb * alpha_fb);
    // Step length actually applied below: the realizable lam (Armijo-decreasing
    // if found, else the most-damped realizable one). Stays 0 — no update — if
    // no realizable step exists this iteration, so the iterate never leaves the
    // realizable set.
    double lam_accept = 0.0;
    for (int ls = 0; ls < max_ls; ++ls) {
      // Form trial state's solved vars at the damped step.
      athelas::par_for(
          DEFAULT_LOOP_PATTERN, "ImplicitMoments :: Newton :: Trial state",
          DevExecSpace(), ib.s, ib.e, qb.s, qb.e,
          KOKKOS_CLASS_LAMBDA(const int i, const int q) {
            newton_.u_rad_trial(i, q, u_idx_er) =
                newton_.u_rad_work(i, q, u_idx_er) +
                lam * solver_.b(i - ib.s, idx(q, 0));
            newton_.u_rad_trial(i, q, u_idx_fr) =
                newton_.u_rad_work(i, q, u_idx_fr) +
                lam * solver_.b(i - ib.s, idx(q, 1));
            newton_.u_rad_trial(i, q, u_idx_vel) =
                newton_.u_rad_work(i, q, u_idx_vel) +
                lam * solver_.b(i - ib.s, idx(q, 2));
            newton_.u_rad_trial(i, q, u_idx_ener) =
                newton_.u_rad_work(i, q, u_idx_ener) +
                lam * solver_.b(i - ib.s, idx(q, 3));
          });

      // Realizability check. We require:
      //   E > 0,  |F| < c·E, sie < emin.
      int n_bad = 0;
      athelas::par_reduce(
          DEFAULT_LOOP_PATTERN,
          "ImplicitMoments :: Newton :: Realizability check", DevExecSpace(),
          ib.s, ib.e, qb.s, qb.e,
          KOKKOS_CLASS_LAMBDA(const int i, const int q, int &count) {
            const double er = newton_.u_rad_trial(i, q, u_idx_er);
            const double fr = newton_.u_rad_trial(i, q, u_idx_fr);
            const double eg = newton_.u_rad_trial(i, q, u_idx_ener);
            const double vg = newton_.u_rad_trial(i, q, u_idx_vel);
            const double sie = eg - 0.5 * vg * vg;
            const double rho = 1.0 / evolved(i, q, idx_tau);
            eos::EOSLambda lam_eos;
            if (ionization_enabled) {
              lam_eos.data[0] = number_density(i, q + 1);
              lam_eos.data[1] = ye(i, q + 1);
              lam_eos.data[2] = ybar(i, q + 1);
              lam_eos.data[3] = sigma1(i, q + 1);
              lam_eos.data[4] = sigma2(i, q + 1);
              lam_eos.data[5] = sigma3(i, q + 1);
              lam_eos.data[6] = e_ion_corr(i, q + 1);
              lam_eos.data[7] = derived(i, q + 1, idx_tgas);
            }
            const double emin =
                paczynski_eos ? eos::sie_from_density_temperature(
                                    eos, rho, T_inversion_floor, lam_eos.ptr())
                              : min_sie(eos, rho, lam_eos.ptr());
            const double emax =
                paczynski_eos
                    ? eos::sie_from_density_temperature(
                          eos, rho, T_inversion_ceiling, lam_eos.ptr())
                    : std::numeric_limits<double>::infinity();
            const double e_margin =
                64.0 * std::numeric_limits<double>::epsilon() *
                std::max({std::abs(eg), std::abs(emin), 1.0});
            if (!std::isfinite(er) || !std::isfinite(fr) ||
                !std::isfinite(eg) || !std::isfinite(vg) ||
                !std::isfinite(sie) || er <= 0.0 || std::abs(fr) >= c * er ||
                sie <= emin + e_margin || sie >= emax - e_margin) {
              count += 1;
            }

            // Face-edge realizability, once per cell. Messy.
            if (q == qb.s) {
              const double er_L = basis_eval<Interface::Left>(
                  phi, newton_.u_rad_trial, i, u_idx_er);
              const double er_R = basis_eval<Interface::Right>(
                  phi, newton_.u_rad_trial, i, u_idx_er);
              const double fr_L = basis_eval<Interface::Left>(
                  phi, newton_.u_rad_trial, i, u_idx_fr);
              const double fr_R = basis_eval<Interface::Right>(
                  phi, newton_.u_rad_trial, i, u_idx_fr);
              if (!std::isfinite(er_L) || !std::isfinite(er_R) ||
                  !std::isfinite(fr_L) || !std::isfinite(fr_R) || er_L <= 0.0 ||
                  er_R <= 0.0 || std::abs(fr_L) >= c * er_L ||
                  std::abs(fr_R) >= c * er_R) {
                count += 1;
              }
            }
          },
          Kokkos::Sum<int>(n_bad));
      if (n_bad > 0) {
        lam *= 0.5;
        continue;
      }

      // Trial residual.
      evaluate_residual(newton_.ls_b_trial, newton_.u_rad_trial, ustar,
                        stage_data, mesh, dt_aii);
      const double F_trial = math::linalg::newton_norm_l2(newton_.ls_b_trial,
                                                          sqrt_gm, dr, weights);
      if (!std::isfinite(F_trial)) {
        lam *= 0.5;
        continue;
      }

      // Trial is realizable with finite residual: record it as the step to
      // apply if no Armijo-decreasing step is found. Never apply a
      // non-realizable or nonfinite trial.
      lam_accept = lam;
      const double F_trial_sq = F_trial * F_trial;

      // Sufficient decrease.
      if (F_trial_sq < (1.0 - 2.0 * alpha_armijo * lam) * F0_sq) {
        break;
      }

      lam *= 0.5;
    } // line search

    if (lam_accept == 0.0) {
      ++iter;
      continue;
    }

    athelas::par_for(
        DEFAULT_LOOP_PATTERN, "ImplicitMoments :: Newton :: Update",
        DevExecSpace(), ib.s, ib.e, qb.s, qb.e,
        KOKKOS_CLASS_LAMBDA(const int i, const int q) {
          newton_.u_rad_work(i, q, u_idx_er) +=
              lam_accept * solver_.b(i - ib.s, idx(q, 0));
          newton_.u_rad_work(i, q, u_idx_fr) +=
              lam_accept * solver_.b(i - ib.s, idx(q, 1));
          newton_.u_rad_work(i, q, u_idx_vel) +=
              lam_accept * solver_.b(i - ib.s, idx(q, 2));
          newton_.u_rad_work(i, q, u_idx_ener) +=
              lam_accept * solver_.b(i - ib.s, idx(q, 3));
        });
    any_step_applied = true;

    // Stagnation stopping criterion.
    // Should this be in the line search?
    double step_norm = 0.0;
    athelas::par_reduce(
        DEFAULT_LOOP_PATTERN, "ImplicitMoments :: Newton :: Step norm",
        DevExecSpace(), ib.s, ib.e, qb.s, qb.e,
        KOKKOS_CLASS_LAMBDA(const int i, const int q, double &m) {
          const double s_er =
              lam_accept * std::abs(solver_.b(i - ib.s, idx(q, 0))) /
              std::max(std::abs(newton_.u_rad_work(i, q, u_idx_er)), U_floor);
          const double s_fr =
              lam_accept * std::abs(solver_.b(i - ib.s, idx(q, 1))) /
              std::max(std::abs(newton_.u_rad_work(i, q, u_idx_fr)), U_floor);
          const double s_v =
              lam_accept * std::abs(solver_.b(i - ib.s, idx(q, 2))) /
              std::max(std::abs(newton_.u_rad_work(i, q, u_idx_vel)), U_floor);
          const double s_e =
              lam_accept * std::abs(solver_.b(i - ib.s, idx(q, 3))) /
              std::max(std::abs(newton_.u_rad_work(i, q, u_idx_ener)), U_floor);
          m = std::max({m, s_er, s_fr, s_v, s_e});
        },
        Kokkos::Max<double>(step_norm));
    if (step_norm < step_tol) {
      accepted = true;
      break;
    }

    ++iter;
  } // Newton-Raphson loop

  // Solve verdict. "accepted" (stagnation) is success-adjacent and only
  // counted; exhausting max_iters without either flag is a failure. The
  // status is returned to the caller; until a step-rejection path exists,
  // a failed solve continues with the unconverged iterate.
  auto status = UpdateStatus::Success;
  if (accepted && !converged) {
    ++n_stagnation_exits_;
  }
  if (!converged && !accepted) {
    status = UpdateStatus::Failure;
    ++n_newton_failures_;
    // With no accepted step the final iterate is just the projected initial
    // guess; applying its delta would turn the projection into a fabricated
    // source term. Restore ustar so such a solve contributes exactly nothing.
    // TODO(astrobarker): [implicit_moments] This needs to be removed.
    if (!any_step_applied) {
      athelas::par_for(
          DEFAULT_LOOP_PATTERN, "ImplicitMoments :: Newton :: Restore ustar",
          DevExecSpace(), ib.s, ib.e, qb.s, qb.e,
          KOKKOS_CLASS_LAMBDA(const int i, const int q) {
            newton_.u_rad_work(i, q, u_idx_tau) = ustar(i, q, idx_tau);
            newton_.u_rad_work(i, q, u_idx_vel) = ustar(i, q, idx_vel);
            newton_.u_rad_work(i, q, u_idx_ener) = ustar(i, q, idx_ener);
            newton_.u_rad_work(i, q, u_idx_er) = ustar(i, q, idx_er);
            newton_.u_rad_work(i, q, u_idx_fr) = ustar(i, q, idx_fr);
          });
    }
  }

  athelas::par_for(
      DEFAULT_LOOP_PATTERN, "ImplicitMoments :: Increment delta",
      DevExecSpace(), ib.s, ib.e, KOKKOS_CLASS_LAMBDA(const int i) {
        for (int q = qb.s; q <= qb.e; ++q) {
          delta_(dt_info.stage, i, q, 0) =
              (newton_.u_rad_work(i, q, u_idx_vel) - ustar(i, q, idx_vel)) /
              dt_aii;
          delta_(dt_info.stage, i, q, 1) =
              (newton_.u_rad_work(i, q, u_idx_ener) - ustar(i, q, idx_ener)) /
              dt_aii;
          delta_(dt_info.stage, i, q, 2) =
              (newton_.u_rad_work(i, q, u_idx_er) - ustar(i, q, idx_er)) /
              dt_aii;
          delta_(dt_info.stage, i, q, 3) =
              (newton_.u_rad_work(i, q, u_idx_fr) - ustar(i, q, idx_fr)) /
              dt_aii;
        }
      });

  return status;
} // update_implicit

/**
 * @brief apply rad hydro package delta
 */
void ImplicitRadiationMomentsPackage::apply_delta(
    AthelasArray3D<double> lhs, const TimeStepInfo &dt_info) const {
  static const int nx = static_cast<int>(lhs.extent(0));
  static const int nq = static_cast<int>(lhs.extent(1));
  static const IndexRange ib(std::make_pair(1, nx - 2));
  static const IndexRange qb(nq);
  static const IndexRange vb(NUM_VARS_);

  const int stage = dt_info.stage;

  athelas::par_for(
      DEFAULT_LOOP_PATTERN, "ImplicitMoments :: Apply delta", DevExecSpace(),
      ib.s, ib.e, qb.s, qb.e, KOKKOS_CLASS_LAMBDA(const int i, const int q) {
        for (int v = vb.s; v <= vb.e; ++v) {
          lhs(i, q, v + 1) += dt_info.dt_coef_implicit * delta_(stage, i, q, v);
        }
      });
}

/**
 * @brief zero delta field
 */
void ImplicitRadiationMomentsPackage::zero_delta() const noexcept {
  static const IndexRange sb(static_cast<int>(delta_.extent(0)));
  static const IndexRange ib(static_cast<int>(delta_.extent(1)));
  static const IndexRange qb(static_cast<int>(delta_.extent(2)));
  static const IndexRange vb(static_cast<int>(delta_.extent(3)));

  // We store the last stage source in the state = 0 slot.
  // That is, G(U^0) <- G(U^n).
  // In an ESDIRK tableau we reuse this for the first stage.
  const int ns = sb.e;
  athelas::par_for(
      DEFAULT_LOOP_PATTERN, "ImplicitMoments :: Cache last delta",
      DevExecSpace(), ib.s, ib.e, qb.s, qb.e,
      KOKKOS_CLASS_LAMBDA(const int i, const int q) {
        for (int v = vb.s; v <= vb.e; ++v) {
          delta_(0, i, q, v) = delta_(ns, i, q, v);
        }
      });

  athelas::par_for(
      DEFAULT_LOOP_PATTERN, "ImplicitMoments :: Zero delta", DevExecSpace(),
      sb.s + 1, sb.e, ib.s, ib.e, qb.s, qb.e,
      KOKKOS_CLASS_LAMBDA(const int s, const int i, const int q) {
        for (int v = vb.s; v <= vb.e; ++v) {
          delta_(s, i, q, v) = 0.0;
        }
      });
}

/**
 * @brief implicit radiation moments timestep restriction
 **/
auto ImplicitRadiationMomentsPackage::min_timestep(
    const StageData &stage_data, const TimeStepInfo &dt_info) const -> double {
  const auto &mesh = stage_data.mesh();
  constexpr double MAX_DT = std::numeric_limits<double>::max();
  constexpr double MIN_DT = 100.0 * std::numeric_limits<double>::min();
  constexpr double TINY = 100.0 * std::numeric_limits<double>::min();

  auto evolved = stage_data.get_field("evolved");
  static const int idx_er =
      stage_data.var_index("evolved", "specific_radiation_energy");
  static const int idx_fr =
      stage_data.var_index("evolved", "specific_radiation_flux");

  static const IndexRange ib(mesh.domain<Domain::Interior>());
  static const IndexRange qb(mesh.n_nodes());

  const auto max_frac_change_e = params_.get<double>("max_fractional_change_e");
  const auto energy_change_scale = params_.get<double>("energy_change_scale");
  const auto max_change_f = params_.get<double>("max_change_f");

  const double dt_old = dt_info.dt;
  assert(dt_old > 0.0 && "ImplicitRadiationMomentsPackage::min_timestep: dt "
                         "must be positive definite!");

  double dt_out = 0.0;
  athelas::par_reduce(
      DEFAULT_LOOP_PATTERN, "ImplicitMoments :: timestep restriction",
      DevExecSpace(), ib.s, ib.e, qb.s, qb.e,
      KOKKOS_CLASS_LAMBDA(const int i, const int q, double &lmin) {
        const double e_old = dt_cache_.e_rad_old(i, q);
        const double flux_old = dt_cache_.f_rad_old(i, q);
        const double e = evolved(i, q, idx_er);
        const double e_scale =
            std::max({std::abs(e), std::abs(e_old), energy_change_scale, TINY});
        const double f = flux_factor(std::max({e, energy_change_scale, TINY}),
                                     evolved(i, q, idx_fr));
        const double f_old =
            flux_factor(std::max({e_old, energy_change_scale, TINY}), flux_old);
        const double frac_change_e = std::abs(e - e_old) / e_scale;
        const double dt_e =
            dt_old * max_frac_change_e / std::max(frac_change_e, TINY);
        const double dt_f =
            dt_old * max_change_f / std::max(std::abs(f - f_old), TINY);
        lmin = std::min({dt_e, dt_f, lmin});
      },
      Kokkos::Min<double>(dt_out));

  dt_out = std::max(dt_out, MIN_DT);
  dt_out = std::min(dt_out, MAX_DT);

  // Store the current radiation energy and flux for use
  // in the next timestep calculation.
  athelas::par_for(
      DEFAULT_FLAT_LOOP_PATTERN, "ImplicitMoments :: cache old radiation vars",
      DevExecSpace(), ib.s, ib.e, qb.s, qb.e,
      KOKKOS_CLASS_LAMBDA(const int i, const int q) {
        dt_cache_.e_rad_old(i, q) = evolved(i, q, idx_er);
        dt_cache_.f_rad_old(i, q) = evolved(i, q, idx_fr);
      });

  if (dt_info.cycle == 1) {
    return dt_info.dt;
  }

  return dt_out;
}

/**
 * @brief fill ImplicitMoments derived quantities
 *
 * TODO(astrobarker): extend
 */
void ImplicitRadiationMomentsPackage::fill_derived(
    StageData &stage_data, const TimeStepInfo & /*dt_info*/) const {
  const auto &mesh = stage_data.mesh();
  return;
  // NOTE: When we actually use this, remove the above.
  auto evolved = stage_data.get_field("evolved");
  auto derived = stage_data.get_field("derived");

  const auto &basis = stage_data.basis();

  const int nNodes = mesh.n_nodes();
  static const IndexRange ib(mesh.domain<Domain::Interior>());

  auto phi = basis.phi();

  // --- Refresh evolved halo ---
  bc::ghost_fill(stage_data, bcs_);

  athelas::par_for(
      DEFAULT_FLAT_LOOP_PATTERN, "ImplicitMoments :: fill derived",
      DevExecSpace(), ib.s, ib.e, KOKKOS_LAMBDA(const int i) {
        for (int q = 0; q < nNodes + 2; ++q) {
          // const double rho =
          //     1.0 / basis_eval(phi, evolved, i, idx_tau, q);

          // const double e_rad =
          //     basis_eval(phi, evolved, i, idx_rad_energy, q) * rho;
          // const double flux_rad =
          //     basis_eval(phi, evolved, i, idx_rad_flux, q) * rho;

          // const double flux_fact = flux_factor(e_rad, f_rad);
        }
      });
}

[[nodiscard]] auto ImplicitRadiationMomentsPackage::name() const noexcept
    -> std::string_view {
  return "ImplicitRadiationMoments";
}

[[nodiscard]] auto ImplicitRadiationMomentsPackage::is_active() const noexcept
    -> bool {
  return active_;
}

void ImplicitRadiationMomentsPackage::set_active(const bool active) {
  active_ = active;
}
} // namespace athelas::radiation
