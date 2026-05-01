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
#include "linalg/linear_algebra.hpp"
#include "loop_layout.hpp"
#include "pgen/problem_in.hpp"
#include "rad_utilities.hpp"
#include "radiation/radhydro_package.hpp"

namespace athelas::radiation {
using basis::NodalBasis, basis::basis_eval;
using eos::EOS;
using fluid::numerical_flux_gudonov_positivity;

void radiation_source_implicit(const StageData &stage_data,
                               AthelasArray3D<double> R,
                               AthelasArray4D<double> delta,
                               const GridStructure &grid,
                               const TimeStepInfo &dt_info) {
  // TODO(astrobarker) handle separate fluid and rad orders
  const auto &rad_basis = stage_data.rad_basis();
  const auto &fluid_basis = stage_data.fluid_basis();
  static const IndexRange ib(grid.domain<Domain::Interior>());
  static const IndexRange qb(grid.n_nodes());

  static const bool ionization_enabled = stage_data.enabled("ionization");

  auto ucf = stage_data.get_field("u_cf");
  auto uaf = stage_data.get_field("u_af");

  const auto &eos = stage_data.eos();
  const auto &opac = stage_data.opac();

  auto phi_rad = rad_basis.phi();
  auto phi_fluid = fluid_basis.phi();
  auto inv_mkk = fluid_basis.inv_mass_matrix();
  auto inv_mkk_rad = rad_basis.inv_mass_matrix();
  auto dr = grid.widths();
  auto weights = grid.weights();
  auto sqrt_gm = grid.sqrt_gm();

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
          for (int v = 0; v < NUM_VARS; ++v) {
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
            ucf(i, q, v) = scratch_sol[v];
            delta(dt_info.stage, i, q, v - 1) =
                (ucf_i(v) - Ustar_i(v)) / dt_info.dt_coef;
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
          for (int v = 0; v < NUM_VARS; ++v) {
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
            ucf(i, q, v) = scratch_sol[v];
            delta(dt_info.stage, i, q, v - 1) =
                (ucf_i(v) - Ustar_i(v)) / dt_info.dt_coef;
          }
        });
  }
}

/**
 * @brief Implicit radiation moments
 * Used for fully implicit transport
 */
ImplicitRadiationMomentsPackage::ImplicitRadiationMomentsPackage(
    const ProblemIn *pin, int n_stages, int nq, BoundaryConditions *bcs,
    double cfl, int nx, bool active)
    : active_(active), cfl_(cfl), bcs_(bcs),
      u_f_l_("ImplicitMoments::u_f_l_", nx + 2, 3),
      u_f_r_("ImplicitMoments::u_f_r_", nx + 2, 3),
      solver_mat_diag_("ImplicitMoments::solver_mat_diag", nx, 4 * nq, 4 * nq),
      solver_mat_upper_("ImplicitMoments::solver_mat_upper", nx - 1, 4 * nq,
                        4 * nq),
      solver_mat_lower_("ImplicitMoments::solver_mat_lower", nx - 1, 4 * nq,
                        4 * nq),
      solver_b_("ImplicitMoments::solver_b", nx, 4 * nq),
      flux_num_("ImplicitMoments::flux_num", nx + 2, 2),
      solver_W_("ImplicitMoments::solver_W", nx - 1, 4 * nq, 4 * nq),
      solver_Y_("ImplicitMoments::solver_Y", nx - 1, 4 * nq),
      solver_Bi_lu_("ImplicitMoments::solver_Bi_lu", 4 * nq, 4 * nq),
      A_minus_("ImplicitMoments::Aplus", nx + 1, 2, 2),
      A_plus_("ImplicitMoments::Aminus", nx + 1, 2, 2),
      A_bndry_("ImplicitMoments::A_bndry", 2, 2),
      d_bndry_("ImplicitMoments::d_bndry", 2, 2),
      delta_("implicit radhydro delta implicit", n_stages, nx + 2, nq, 4),
      e_rad_old_("ImplicitMoments::e_rad_old", nx + 2, nq),
      f_rad_old_("ImplicitMoments::f_rad_old", nx + 2, nq) {
  // Storing package params
  params_.add<double>(
      "max_fractional_change_e",
      pin->param()->get<double>("radiation.timestep.max_fractional_change_e"));
  params_.add<double>("max_change_f", pin->param()->get<double>(
                                          "radiation.timestep.max_change_f"));
}

void ImplicitRadiationMomentsPackage::update_implicit(
    const StageData &stage_data, AthelasArray3D<double> ustar,
    const GridStructure &grid, const TimeStepInfo &dt_info) {
  using bc::BcType;

  const auto &basis = stage_data.fluid_basis();
  const int nNodes = grid.n_nodes();
  static const IndexRange ib(grid.domain<Domain::Interior>());
  static const IndexRange qb(nNodes);

  auto ucf = stage_data.get_field("u_cf");
  auto uaf = stage_data.get_field("u_af");
  auto facedata = stage_data.get_field<AthelasArray2D<double>>("facedata");

  static const int idx_tau = stage_data.var_index("u_cf", "tau");
  static const int idx_vel = stage_data.var_index("u_cf", "vel");
  static const int idx_ener = stage_data.var_index("u_cf", "fluid_energy");
  static const int idx_er = stage_data.var_index("u_cf", "rad_energy");
  static const int idx_fr = stage_data.var_index("u_cf", "rad_momentum");
  static const int idx_vstar = stage_data.var_index("facedata", "vstar");

  auto phi = basis.phi();
  auto dphi = basis.dphi();
  auto mkk = basis.mass_matrix();
  auto inv_mkk = basis.inv_mass_matrix();
  auto dr = grid.widths();
  auto weights = grid.weights();
  auto sqrt_gm = grid.sqrt_gm();

  // compute radiation-matter coupling sources implicitly with Newton-Raphson.
  radiation_source_implicit(stage_data, ustar, delta_, grid, dt_info);

  const int block_size = 4 * nNodes;

  // Left/Right face states
  // Extract left/right interface states for necessary vars.
  // These refer to the left/right states on the left interface of element i.
  // v = 0: specific volume
  // v = 1: specific radiation energy density
  // v = 2: specific radiation flux

  constexpr double c = constants::c_cgs;
  constexpr double c2 = c * c;

  bc::fill_ghost_zones<2>(ucf, &grid, bcs_, {3, 4});
  bc::fill_ghost_zones<3>(ucf, &grid, bcs_, {0, 2});
  bc::fill_ghost_zones<2>(ustar, &grid, bcs_, {3, 4});
  bc::fill_ghost_zones<3>(ustar, &grid, bcs_, {0, 2});

  athelas::par_for(
      DEFAULT_FLAT_LOOP_PATTERN, "ImplicitMoments :: Interface states",
      DevExecSpace(), ib.s, ib.e + 1, KOKKOS_CLASS_LAMBDA(const int i) {
        u_f_l_(i, 0) = basis_eval<Interface::Right>(phi, ustar, i - 1, idx_tau);
        u_f_r_(i, 0) = basis_eval<Interface::Left>(phi, ustar, i, idx_tau);
        for (int v = 3; v < 5; ++v) {
          u_f_l_(i, v - 2) = basis_eval<Interface::Right>(phi, ustar, i - 1, v);
          u_f_r_(i, v - 2) = basis_eval<Interface::Left>(phi, ustar, i, v);
        }
      });

  athelas::par_for(
      DEFAULT_FLAT_LOOP_PATTERN, "ImplicitMoments :: Interface Jacobians",
      DevExecSpace(), ib.s, ib.e + 1, KOKKOS_CLASS_LAMBDA(const int i) {
        // States at the interface i+1/2 (L is i, R is i+1)
        const double rho_L = 1.0 / u_f_l_(i, 0);
        const double rho_R = 1.0 / u_f_r_(i, 0);
        const double vstar = facedata(i, idx_vstar);

        // Radiation specific variables
        const double E_L = u_f_l_(i, 1) * rho_L;
        const double E_R = u_f_r_(i, 1) * rho_R;
        const double F_L = u_f_l_(i, 2) * rho_L;
        const double F_R = u_f_r_(i, 2) * rho_R;

        const double alpha = rad_wavespeed(E_L, E_R, F_L, F_R, vstar);
        const double f_l = flux_factor(u_f_l_(i, 1), u_f_l_(i, 2));
        const double f_r = flux_factor(u_f_r_(i, 1), u_f_r_(i, 2));
        const double chi_L = eddington_factor(f_l);
        const double chi_R = eddington_factor(f_r);
        const double chi_prime_L = eddington_factor_prime(f_l);
        const double chi_prime_R = eddington_factor_prime(f_r);

        // A_minus = d(F_hat)/d(U_local) = 0.5 * (J_local + alpha * I)
        A_minus_(i - ib.s, 0, 0) = 0.5 * (-vstar + alpha) * rho_L;
        A_minus_(i - ib.s, 0, 1) = 0.5 * rho_L;
        A_minus_(i - ib.s, 1, 0) =
            0.5 * (c2 * (chi_L - f_l * chi_prime_L)) * rho_L;
        A_minus_(i - ib.s, 1, 1) = 0.5 * (chi_prime_L - vstar + alpha) * rho_L;

        // A_plus = d(F_hat)/d(U_neighbor) = 0.5 * (J_neighbor - alpha * I)
        A_plus_(i - ib.s, 0, 0) = 0.5 * (-vstar - alpha) * rho_R;
        A_plus_(i - ib.s, 0, 1) = 0.5 * rho_R;
        A_plus_(i - ib.s, 1, 0) =
            0.5 * (c2 * (chi_R - f_r * chi_prime_R)) * rho_R;
        A_plus_(i - ib.s, 1, 1) = 0.5 * (chi_prime_R - vstar - alpha) * rho_R;
      });

  Kokkos::deep_copy(solver_mat_diag_, 0.0);
  Kokkos::deep_copy(solver_mat_upper_, 0.0);
  Kokkos::deep_copy(solver_mat_lower_, 0.0);
  Kokkos::deep_copy(solver_b_, 0.0);

  const double dt_aii = dt_info.dt_coef;

  // Flat DOF index within a block (node-major: v fastest)
  auto idx = [&](const int q, const int v) { return q * 4 + v; };

  // Build transport numerical fluxes from U* at all interfaces.
  athelas::par_for(
      DEFAULT_FLAT_LOOP_PATTERN,
      "ImplicitMoments :: Interface transport fluxes", DevExecSpace(), ib.s,
      ib.e + 1, KOKKOS_CLASS_LAMBDA(const int i) {
        const double rho_L = 1.0 / u_f_l_(i, 0);
        const double rho_R = 1.0 / u_f_r_(i, 0);
        const double vstar = facedata(i, idx_vstar);

        const double E_L = u_f_l_(i, 1) * rho_L;
        const double E_R = u_f_r_(i, 1) * rho_R;
        const double F_L = u_f_l_(i, 2) * rho_L;
        const double F_R = u_f_r_(i, 2) * rho_R;

        const double Prad_L = compute_closure(E_L, F_L);
        const double Prad_R = compute_closure(E_R, F_R);
        const double alpha = rad_wavespeed(E_L, E_R, F_L, F_R, vstar);

        const LLFRiemannState left_erad{
            .u = E_L, .f = F_L - vstar * E_L, .alpha = alpha};
        const LLFRiemannState right_erad{
            .u = E_R, .f = F_R - vstar * E_R, .alpha = alpha};
        flux_num_(i, 0) = llf_flux(left_erad, right_erad);
        //std::println("i fluxe {} {:.5e}", i, flux_num_(i, 0));
        //std::println("i el er v {} {:.5e} {:.5e} {:.5e} {:.5e}", i, E_L, E_R, vstar, flux_num_(i, 0));

        const LLFRiemannState left_frad{
            .u = F_L, .f = c2 * Prad_L - vstar * F_L, .alpha = alpha};
        const LLFRiemannState right_frad{
            .u = F_R, .f = c2 * Prad_R - vstar * F_R, .alpha = alpha};
        flux_num_(i, 1) = llf_flux(left_frad, right_frad);
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
            solver_mat_diag_(blk, row, row) += m;
          }
        }

        // Volume term - diagonal block
        // K_vol[q*2+v, p*2+w] = [D^T W]_{qp} * J_vol[v,w](x_p)
        // J_vol = rho[[-v, 1], [s^2, -v]] with s^2 = c^2 * chi
        for (int q = 0; q < nNodes; ++q) {
          for (int v = 0; v < 2; ++v) {
            const int row = idx(q, v);
            for (int p = 0; p < nNodes; ++p) {
              const double vp = ucf(i, p, idx_vel);
              const double rhop = 1.0 / ucf(i, p, idx_tau);
              const double f =
                  flux_factor(ucf(i, p, idx_er), ucf(i, p, idx_fr));
              const double chi = eddington_factor(f);
              const double sp2 = c2 * chi;
              const double chi_prime = eddington_factor_prime(f);
              for (int w = 0; w < 2; ++w) {
                const int col = idx(p, w);

                double A_vw = 1.0;
                if (v == 0 && w == 0) {
                  A_vw = -vp;
                } else if (v == 1 && w == 1) {
                  A_vw = c2 * chi_prime - vp;
                } else if (v == 1 && w == 0) {
                  A_vw = sp2 - f * chi_prime;
                }

                solver_mat_diag_(blk, row, col) -=
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

                  solver_mat_diag_(blk, row, col) +=
                      dt_aii * ellR_q * gR * A_minus_(ifaceR, v, w) * ellR_p;

                  solver_mat_upper_(blk, row, col) +=
                      dt_aii * ellR_q * gR * A_plus_(ifaceR, v, w) * ellL_p_nbr;
                }

                // --- left face ---
                if (i > ib.s) {
                  const double ellR_p_nbr = phi(i - 1, nNodes + 1, p);
                  const int ifaceL = i - ib.s;

                  solver_mat_diag_(blk, row, col) -=
                      dt_aii * ellL_q * gL * A_plus_(ifaceL, v, w) * ellL_p;

                  solver_mat_lower_(blk - 1, row, col) -=
                      dt_aii * ellL_q * gL * A_minus_(ifaceL, v, w) *
                      ellR_p_nbr;
                }
              }
            }
          }
        }

        // RHS - T(U*) packed into flattened block RHS b.
        const double vstar = facedata(i, idx_vstar);
        for (int q = 0; q < nNodes; ++q) {

          double rhs_e = -(flux_num_(i + 1, 0) * phi(i, nNodes + 1, q) *
                               sqrt_gm(i, nNodes + 1) -
                           flux_num_(i, 0) * phi(i, 0, q) * sqrt_gm(i, 0));
          double rhs_f = -(flux_num_(i + 1, 1) * phi(i, nNodes + 1, q) *
                               sqrt_gm(i, nNodes + 1) -
                           flux_num_(i, 1) * phi(i, 0, q) * sqrt_gm(i, 0));

          for (int p = 0; p < nNodes; ++p) {
            const double rho = 1.0 / ustar(i, p, idx_tau);
            const double e_rad = ustar(i, p, idx_er) * rho;
            const double f_rad = ustar(i, p, idx_fr) * rho;
            const double p_rad = compute_closure(e_rad, f_rad);
            const auto [flux_e, flux_f] = flux_rad(e_rad, f_rad, p_rad, vstar);
            const double w_dphi_sqrtgm =
                weights(p) * dphi(i, p + 1, q) * sqrt_gm(i, p + 1);
            rhs_e += w_dphi_sqrtgm * flux_e;
            rhs_f += w_dphi_sqrtgm * flux_f;
          }

          const double m = mkk(i, q);
          // rhs = - M(U^i - U*) - dt a_ii T
          solver_b_(blk, idx(q, 0)) = -(m * (ucf(i, q, idx_er) - ustar(i, q, idx_er)) - dt_aii * rhs_e);
          solver_b_(blk, idx(q, 1)) = -(m * (ucf(i, q, idx_fr) - ustar(i, q, idx_fr)) - dt_aii * rhs_f);
          //solver_b_(blk, idx(q, 0)) = m * ustar(i, q, idx_er);
          //solver_b_(blk, idx(q, 1)) = m * ustar(i, q, idx_fr);
          solver_b_(blk, idx(q, 2)) = m * ustar(i, q, idx_vel) - 0 * dt_aii * rhs_e;
          solver_b_(blk, idx(q, 3)) = m * ustar(i, q, idx_ener) - 0 * dt_aii * rhs_f;
        }
      });

  const auto rad_bcs = get_bc_data<2>(bcs_);
  static const int nblocks = grid.n_elements();
  static const int i_inner = 1;
  static const int i_outer = nblocks + 1;
  athelas::par_for(
      DEFAULT_LOOP_PATTERN,
      "ImplicitMoments :: Assemble solver_mat :: boundaries", DevExecSpace(), 0,
      0, KOKKOS_CLASS_LAMBDA(const int) {
        const double vstar_i = facedata(i_inner, idx_vstar);
        const double vstar_o = facedata(i_outer, idx_vstar);
        const double rho_i = 1.0 / u_f_r_(i_inner, 0);
        const double rho_o = 1.0 / u_f_l_(i_outer, 0);
        const double gL_i = sqrt_gm(i_inner, 0);
        const double gR_o = sqrt_gm(i_outer, nNodes + 1);

        // inner boundary
        int blk = 0;
        const auto inner_bc = rad_bcs[0];
        switch (inner_bc.type) {

        case BcType::Outflow:
          for (int v = 0; v < 2; ++v) {
            d_bndry_(v, v) = 1.0;
          }
          break;
        case BcType::Reflecting:
          d_bndry_(0, 0) = 1.0;
          d_bndry_(1, 1) = -1.0;
          break;
        case BcType::Marshak:
          d_bndry_(0, 0) = 1.0;
          d_bndry_(0, 1) = 0.0;
          d_bndry_(1, 0) = -0.5 * c;
          d_bndry_(1, 1) = -1.0;
          break;
        default:
          break;
        }
        boundary_jacobian<Boundary::Interior>(A_bndry_, u_f_l_, u_f_r_,
                                              d_bndry_, vstar_i);
        for (int q = 0; q < nNodes; ++q) {
          for (int p = 0; p < nNodes; ++p) {
            const double ellL_q = phi(i_inner, 0, q);
            const double ellL_p = phi(i_inner, 0, p);

            for (int v = 0; v < 2; ++v) {
              const int row = idx(q, v);
              for (int w = 0; w < 2; ++w) {
                const int col = idx(p, w);

                solver_mat_diag_(blk, row, col) -=
                    dt_aii * ellL_q * gL_i * A_bndry_(v, w) * ellL_p * rho_i;
              }
            }
          }
        }

        blk = nblocks - 1;
        const auto outer_bc = rad_bcs[1];
        switch (outer_bc.type) {

        case BcType::Outflow:
          for (int v = 0; v < 2; ++v) {
            d_bndry_(v, v) = 1.0;
          }
          break;
        case BcType::Reflecting:
          d_bndry_(0, 0) = 1.0;
          d_bndry_(1, 1) = -1.0;
          break;
        case BcType::Marshak:
          d_bndry_(0, 0) = 1.0;
          d_bndry_(0, 1) = 0.0;
          d_bndry_(1, 0) = -0.5 * c;
          d_bndry_(1, 1) = -1.0;
          break;
        default:
          break;
        }
        boundary_jacobian<Boundary::Exterior>(A_bndry_, u_f_l_, u_f_r_,
                                              d_bndry_, vstar_o);
        for (int q = 0; q < nNodes; ++q) {
          for (int p = 0; p < nNodes; ++p) {
            const double ellR_q = phi(i_outer, nNodes + 1, q);
            const double ellR_p = phi(i_outer, nNodes + 1, p);

            for (int v = 0; v < 2; ++v) {
              const int row = idx(q, v);
              for (int w = 0; w < 2; ++w) {
                const int col = idx(p, w);
                solver_mat_diag_(blk, row, col) +=
                    dt_aii * ellR_q * gR_o * A_bndry_(v, w) * ellR_p * rho_o;
              }
            }
          }
        }
      });

  athelas::ThomasScratch scratch{
      .W = solver_W_,
      .Y = solver_Y_,
      .Bi_lu = solver_Bi_lu_,
  };

  block_thomas_solve(nblocks, block_size, solver_mat_lower_, solver_mat_diag_,
                     solver_mat_upper_, solver_b_, scratch);

  athelas::par_for(
      DEFAULT_LOOP_PATTERN, "ImplicitMoments :: Increment delta",
      DevExecSpace(), ib.s, ib.e, KOKKOS_CLASS_LAMBDA(const int i) {
        const int blk = i - ib.s;
        for (int q = qb.s; q <= qb.e; ++q) {
          // E (v=0): row 2*q_local
        std::println("blk deltae (old, change) {} {:.5e} {:.5e}", blk, delta_(dt_info.stage, i, q, 2), solver_b_(blk, idx(q, 0)/dt_aii));
          delta_(dt_info.stage, i, q, 2) += solver_b_(blk, idx(q, 0)) / dt_aii;
          //delta_(dt_info.stage, i, q, 2) += (solver_b_(blk, idx(q, 0)) - ustar(i, q, idx_er)) / dt_aii;
          //std::println("blk delta e {} {:.5e}", blk, delta_(dt_info.stage, i, q, 2));

          // F (v=1): row 2*q_local + 1
          delta_(dt_info.stage, i, q, 3) += solver_b_(blk, idx(q, 1)) / dt_aii;
          //delta_(dt_info.stage, i, q, 3) += (solver_b_(blk, idx(q, 1)) - ustar(i, q, idx_fr)) / dt_aii;
        }
      });
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
      DEFAULT_LOOP_PATTERN, "ImplicitMoments :: Store last stage source",
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
    const StageData &stage_data, const GridStructure &grid,
    const TimeStepInfo &dt_info) const -> double {
  constexpr double MAX_DT = std::numeric_limits<double>::max();
  constexpr double MIN_DT = 100.0 * std::numeric_limits<double>::min();
  constexpr double EPS = 1.0e-10;

  auto ucf = stage_data.get_field("u_cf");
  static const int idx_er = stage_data.var_index("u_cf", "rad_energy");
  static const int idx_fr = stage_data.var_index("u_cf", "rad_momentum");

  static const IndexRange ib(grid.domain<Domain::Interior>());
  static const IndexRange qb(grid.n_nodes());

  const auto max_frac_change_e = params_.get<double>("max_fractional_change_e");
  const auto max_change_f = params_.get<double>("max_change_f");

  const double dt_old = dt_info.dt;

  double dt_out = 0.0;
  athelas::par_reduce(
      DEFAULT_LOOP_PATTERN, "ImplicitMoments :: timestep restriction",
      DevExecSpace(), ib.s, ib.e, qb.s, qb.e,
      KOKKOS_CLASS_LAMBDA(const int i, const int q, double &lmin) {
        const double e_old = e_rad_old_(i, q);
        const double flux_old = f_rad_old_(i, q);
        const double f =
            flux_factor(ucf(i, q, idx_er), ucf(i, q, idx_fr) + EPS);
        const double f_old = flux_factor(e_old + EPS, flux_old + EPS);
        const double dt_e = dt_old * max_frac_change_e * (e_old + EPS) /
                            (std::abs(ucf(i, q, idx_er) - e_old) + EPS);
        const double dt_f = dt_old * max_change_f / (std::abs(f - f_old) + EPS);
        lmin = std::min({dt_e, dt_f, lmin});
      },
      Kokkos::Min<double>(dt_out));

  dt_out = std::max(cfl_ * dt_out, MIN_DT);
  dt_out = std::min(dt_out, MAX_DT);

  // If we are at the start of a run and don't have previous timestep data,
  // then ignore the implicit timestep control, as it interferes
  // with the ramp up.
  // A bit hacky and maybe not appropriate for restarts.
  if (dt_info.t == 0.0) {
    dt_out = MAX_DT;
  }

  // Store the current radiation energy and flux for use
  // in the next timestep calculation.
  athelas::par_for(
      DEFAULT_FLAT_LOOP_PATTERN, "ImplicitMoments :: cache old radiation vars",
      DevExecSpace(), ib.s, ib.e, qb.s, qb.e,
      KOKKOS_CLASS_LAMBDA(const int i, const int q) {
        e_rad_old_(i, q) = ucf(i, q, idx_er);
        f_rad_old_(i, q) = ucf(i, q, idx_fr);
      });

  return dt_out;
}

/**
 * @brief fill ImplicitMoments derived quantities
 *
 * TODO(astrobarker): extend
 */
void ImplicitRadiationMomentsPackage::fill_derived(
    StageData &stage_data, const GridStructure &grid,
    const TimeStepInfo & /*dt_info*/) const {
  return;
  // NOTE: When we actually use this, remove the above.
  auto ucf = stage_data.get_field("u_cf");
  auto upf = stage_data.get_field("u_pf");
  auto uaf = stage_data.get_field("u_af");

  const auto &fluid_basis = stage_data.fluid_basis();

  const int nNodes = grid.n_nodes();
  static const IndexRange ib(grid.domain<Domain::Entire>());

  auto phi = fluid_basis.phi();

  // --- Apply BC ---
  bc::fill_ghost_zones<2>(ucf, &grid, bcs_, {3, 4});

  athelas::par_for(
      DEFAULT_FLAT_LOOP_PATTERN, "ImplicitMoments :: fill derived",
      DevExecSpace(), ib.s, ib.e, KOKKOS_LAMBDA(const int i) {
        for (int q = 0; q < nNodes + 2; ++q) {
          // const double rho =
          //     1.0 / basis_eval(phi, ucf, i, vars::cons::SpecificVolume, q);

          // const double e_rad =
          //     basis_eval(phi, ucf, i, vars::cons::RadEnergy, q) * rho;
          // const double flux_rad =
          //     basis_eval(phi, ucf, i, vars::cons::RadFlux, q) * rho;

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

// ----------------------------------------------------------------------------

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
                                      const GridStructure &grid,
                                      const TimeStepInfo &dt_info) const {
  // TODO(astrobarker) handle separate fluid and rad orders
  const auto &basis = stage_data.fluid_basis();
  static const IndexRange ib(grid.domain<Domain::Interior>());
  static const IndexRange qb(grid.n_nodes());
  static const IndexRange vb(NUM_VARS_);

  const auto stage = dt_info.stage;
  auto ucf = stage_data.get_field("u_cf");

  // --- Apply BC ---
  bc::fill_ghost_zones<2>(ucf, &grid, bcs_, {3, 4});
  bc::fill_ghost_zones<3>(ucf, &grid, bcs_, {0, 2});

  // --- radiation Increment : Divergence ---
  radhydro_divergence(stage_data, grid, stage);

  // --- Divide update by mass matrix ---
  auto inv_mqq = basis.inv_mass_matrix();
  athelas::par_for(
      DEFAULT_LOOP_PATTERN, "RadHydro :: delta / M_qq", DevExecSpace(), ib.s,
      ib.e, qb.s, qb.e, KOKKOS_CLASS_LAMBDA(const int i, const int q) {
        const double inv_mm = inv_mqq(i, q);
        for (int v = 0; v < NUM_VARS_; ++v) {
          delta_(stage, i, q, v) *= inv_mm;
          std::println("i de {} {:.5e}", i, delta_(dt_info.stage, i, q, 2));
        }
      });
} // update_explicit

void RadHydroPackage::update_implicit(const StageData &stage_data,
                                      AthelasArray3D<double> R,
                                      const GridStructure &grid,
                                      const TimeStepInfo &dt_info) {
  // compute radiation-matter coupling sources implicitly with Newton-Raphson.
  radiation_source_implicit(stage_data, R, delta_im_, grid, dt_info);
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
                                          const GridStructure &grid,
                                          const int stage) const {
  using fluid::FluidRiemannState;
  auto ucf = stage_data.get_field("u_cf");
  auto uaf = stage_data.get_field("u_af");
  auto facedata = stage_data.get_field<AthelasArray2D<double>>("facedata");

  const auto &rad_basis = stage_data.rad_basis();
  const auto &fluid_basis = stage_data.fluid_basis();

  const auto &nNodes = grid.n_nodes();
  static constexpr int ilo = 1;
  static const auto &ihi = grid.get_ihi();
  static const IndexRange ib(grid.domain<Domain::Interior>());
  static const IndexRange qb(grid.n_nodes());
  static const IndexRange vb(NUM_VARS_);

  static const int idx_tau = stage_data.var_index("u_cf", "tau");
  static const int idx_vel = stage_data.var_index("u_cf", "vel");
  static const int idx_ener = stage_data.var_index("u_cf", "fluid_energy");
  static const int idx_pre = stage_data.var_index("u_af", "pressure");
  static const int idx_cs = stage_data.var_index("u_af", "sound speed");
  static const int idx_radener = stage_data.var_index("u_cf", "rad_energy");
  static const int idx_radflux = stage_data.var_index("u_cf", "rad_momentum");
  static const int idx_vstar = stage_data.var_index("facedata", "vstar");

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
        std::println("i el er v {} {:.5e} {:.5e} {:.5e} {:.5e}", i, E_L, E_R, vstar, flux_e);

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

  const auto &fluid_basis = stage_data.fluid_basis();

  const auto &eos = stage_data.eos();

  const int nNodes = grid.n_nodes();
  static const IndexRange ib(grid.domain<Domain::Entire>());
  static const bool ionization_enabled = stage_data.enabled("ionization");

  auto phi_fluid = fluid_basis.phi();

  // --- Apply BC ---
  bc::fill_ghost_zones<2>(uCF, &grid, bcs_, {3, 4});
  bc::fill_ghost_zones<3>(uCF, &grid, bcs_, {0, 2});

  if (stage_data.enabled("composition")) {
    static constexpr int nvars = 5; // non-comps
    // composition boundary condition
    static const IndexRange vb_comps(
        std::make_pair(nvars, stage_data.nvars("u_cf") - 1));
    bc::fill_ghost_zones_composition(uCF, vb_comps);
    atom::fill_derived_comps<Domain::Entire>(stage_data, &grid);
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
} // namespace athelas::radiation
