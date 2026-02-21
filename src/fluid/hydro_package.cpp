/**
 * @file hydro_package.cpp
 * --------------
 *
 * @brief Pure hydrodynamics package
 */
#include <limits>

#include "Kokkos_Macros.hpp"
#include "basic_types.hpp"
#include "bc/boundary_conditions.hpp"
#include "composition/composition.hpp"
#include "composition/saha.hpp"
#include "eos/eos_variant.hpp"
#include "fluid/fluid_utilities.hpp"
#include "fluid/hydro_package.hpp"
#include "geometry/grid.hpp"
#include "kokkos_abstraction.hpp"
#include "kokkos_types.hpp"
#include "loop_layout.hpp"
#include "pgen/problem_in.hpp"

namespace athelas::fluid {

using basis::NodalBasis;
using eos::EOS;

HydroPackage::HydroPackage(const ProblemIn * /*pin*/, int n_stages, int order,
                           BoundaryConditions *bcs, double cfl, int nx,
                           bool active)
    : active_(active), nx_(nx), cfl_(cfl), bcs_(bcs),
      dFlux_num_("hydro::dFlux_num_", nx + 2 + 1, 3),
      u_f_l_("hydro::u_f_l_", nx + 2, 3), u_f_r_("hydro::u_f_r_", nx + 2, 3),
      flux_u_("hydro::flux_u_", n_stages, nx + 2 + 1),
      delta_("hydro :: delta", n_stages, nx_ + 2, order, 3) {
} // Need long term solution for flux_u_

void HydroPackage::update_explicit(const StageData &stage_data,
                                   const GridStructure &grid,
                                   const TimeStepInfo &dt_info) const {
  const int stage = dt_info.stage;
  auto ucf = stage_data.get_field("u_cf");

  auto uaf = stage_data.get_field("u_af");

  const auto &basis = stage_data.fluid_basis();

  static const IndexRange ib(grid.domain<Domain::Interior>());
  static const IndexRange qb(grid.n_nodes());
  static const IndexRange vb(NUM_VARS_);

  // --- Apply BC ---
  bc::fill_ghost_zones<3>(ucf, &grid, bcs_, {0, 2});
  if (stage_data.composition_enabled()) {
    static const IndexRange vb_comps(
        std::make_pair(NUM_VARS_, stage_data.nvars("u_cf") - 1));
    bc::fill_ghost_zones_composition(ucf, vb_comps);
  }

  // --- Fluid Increment : Divergence ---
  fluid_divergence(stage_data, grid, stage);

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
}

// Compute the dvbergence of the flux term for the update
// TODO(astrobarker): dont pass in stage
void HydroPackage::fluid_divergence(const StageData &stage_data,
                                    const GridStructure &grid,
                                    const int stage) const {
  auto ucf = stage_data.get_field("u_cf");

  auto uaf = stage_data.get_field("u_af");

  const auto &basis = stage_data.fluid_basis();

  const auto &nNodes = grid.n_nodes();
  const auto &order = basis.order();
  static const IndexRange ib(grid.domain<Domain::Interior>());
  static const IndexRange qb(grid.n_nodes());
  static const IndexRange vb(NUM_VARS_);

  auto x_l = grid.x_l();
  auto sqrt_gm = grid.sqrt_gm();
  auto weights = grid.weights();

  auto phi = basis.phi();
  auto dphis = basis.dphi();

  // --- Interpolate Conserved Variable to Interfaces ---

  // Left/Right face states
  par_for(
      DEFAULT_FLAT_LOOP_PATTERN, "Hydro :: Interface States", DevExecSpace(),
      ib.s, ib.e + 1, KOKKOS_CLASS_LAMBDA(const int i) {
        for (int v = vb.s; v <= vb.e; ++v) {
          u_f_l_(i, v) = basis.basis_eval(ucf, i - 1, v, nNodes + 1);
          u_f_r_(i, v) = basis.basis_eval(ucf, i, v, 0);
        }
      });

  // --- Calc numerical flux at all faces ---
  par_for(
      DEFAULT_FLAT_LOOP_PATTERN, "Hydro :: Numerical Fluxes", DevExecSpace(),
      ib.s, ib.e + 1, KOKKOS_CLASS_LAMBDA(const int i) {
        const double P_L = uaf(i - 1, nNodes + 1, vars::aux::Pressure);
        const double Cs_L = uaf(i - 1, nNodes + 1, vars::aux::Cs);

        const double P_R = uaf(i, 0, vars::aux::Pressure);
        const double Cs_R = uaf(i, 0, vars::aux::Cs);

        // --- Numerical Fluxes ---

        // Riemann Problem
        // auto [flux_u, flux_p] = numerical_flux_gudonov( u_f_l_(ib,  1 ),
        // u_f_r_(ib,  1
        // ), P_L, P_R, lam_L, lam_R);
        const auto [flux_u, flux_p] = numerical_flux_gudonov_positivity(
            u_f_l_(i, vars::cons::SpecificVolume),
            u_f_r_(i, vars::cons::SpecificVolume),
            u_f_l_(i, vars::cons::Velocity), u_f_r_(i, vars::cons::Velocity),
            P_L, P_R, Cs_L, Cs_R);
        flux_u_(stage, i) = flux_u;

        dFlux_num_(i, vars::cons::SpecificVolume) = -flux_u;
        dFlux_num_(i, vars::cons::Velocity) = flux_p;
        dFlux_num_(i, vars::cons::Energy) = flux_u * flux_p;
      });

  flux_u_(stage, 0) = flux_u_(stage, 1);
  flux_u_(stage, ib.e + 2) = flux_u_(stage, ib.e + 1);

  // --- Surface Term ---
  athelas::par_for(
      DEFAULT_LOOP_PATTERN, "Hydro :: Surface Term", DevExecSpace(), ib.s, ib.e,
      qb.s, qb.e, KOKKOS_CLASS_LAMBDA(const int i, const int q) {
        for (int v = vb.s; v <= vb.e; ++v) {
          delta_(stage, i, q, v) -=
              (+dFlux_num_(i + 1, v) * phi(i, nNodes + 1, q) *
                   sqrt_gm(i, nNodes + 1) -
               dFlux_num_(i + 0, v) * phi(i, 0, q) * sqrt_gm(i, 0));
        }
      });

  if (order > 1) [[likely]] {
    // --- Volume Term ---
    athelas::par_for(
        DEFAULT_LOOP_PATTERN, "Hydro :: Volume Term", DevExecSpace(), ib.s,
        ib.e, qb.s, qb.e, KOKKOS_CLASS_LAMBDA(const int i, const int p) {
          double local_sum1 = 0.0;
          double local_sum2 = 0.0;
          double local_sum3 = 0.0;
          for (int q = 0; q < nNodes; ++q) {
            const double vel = ucf(i, q, vars::cons::Velocity);
            const double P = uaf(i, q + 1, vars::aux::Pressure);
            const auto [flux1, flux2, flux3] = flux_fluid(vel, P);
            const double w = weights(q);
            const double dphi = dphis(i, q + 1, p);
            const double sqrtgm = sqrt_gm(i, q + 1);

            local_sum1 += w * flux1 * dphi * sqrtgm;
            local_sum2 += w * flux2 * dphi * sqrtgm;
            local_sum3 += w * flux3 * dphi * sqrtgm;
          }

          delta_(stage, i, p, vars::cons::SpecificVolume) += local_sum1;
          delta_(stage, i, p, vars::cons::Velocity) += local_sum2;
          delta_(stage, i, p, vars::cons::Energy) += local_sum3;
        });
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
  static const IndexRange sb(static_cast<int>(delta_.extent(0)));
  static const IndexRange ib(static_cast<int>(delta_.extent(1)));
  static const IndexRange qb(static_cast<int>(delta_.extent(2)));
  static const IndexRange vb(static_cast<int>(delta_.extent(3)));

  athelas::par_for(
      DEFAULT_LOOP_PATTERN, "Hydro :: Zero delta", DevExecSpace(), sb.s, sb.e,
      ib.s, ib.e, qb.s, qb.e,
      KOKKOS_CLASS_LAMBDA(const int s, const int i, const int q) {
        for (int v = vb.s; v <= vb.e; ++v) {
          delta_(s, i, q, v) = 0.0;
        }
      });
}

/**
 * @brief explicit hydrodynamic timestep restriction
 **/
auto HydroPackage::min_timestep(const StageData &stage_data,
                                const GridStructure &grid,
                                const TimeStepInfo & /*dt_info*/) const
    -> double {
  static constexpr double MAX_DT = std::numeric_limits<double>::max();
  static constexpr double MIN_DT = 100.0 * std::numeric_limits<double>::min();

  auto uaf = stage_data.get_field("u_af");

  static const int nnodes = grid.n_nodes();
  static const IndexRange ib(grid.domain<Domain::Interior>());
  auto dr = grid.widths();

  double dt_out = 0.0;
  athelas::par_reduce(
      DEFAULT_FLAT_LOOP_PATTERN, "Hydro :: Timestep", DevExecSpace(), ib.s,
      ib.e,
      KOKKOS_CLASS_LAMBDA(const int i, double &lmin) {
        // Find the max sound speed across the element including the interfaces
        double Cs = uaf(i, 0, vars::aux::Cs);
        for (int q = 1; q <= nnodes; ++q) {
          Cs = std::max(Cs, uaf(i, q, vars::aux::Cs));
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
                                const GridStructure &grid,
                                const TimeStepInfo & /*dt_info*/) const {
  auto uCF = stage_data.get_field("u_cf");
  auto uPF = stage_data.get_field("u_pf");
  auto uAF = stage_data.get_field("u_af");

  const int nNodes = grid.n_nodes();
  static const IndexRange ib(grid.domain<Domain::Entire>());
  static const bool ionization_enabled = stage_data.ionization_enabled();

  const auto &basis = stage_data.fluid_basis();

  // --- Apply BC ---
  bc::fill_ghost_zones<3>(uCF, &grid, bcs_, {0, 2});

  if (stage_data.composition_enabled()) {
    // composition boundary condition
    static const IndexRange vb_comps(
        std::make_pair(NUM_VARS_, stage_data.nvars("u_cf") - 1));
    bc::fill_ghost_zones_composition(uCF, vb_comps);
    atom::fill_derived_comps<Domain::Entire>(stage_data, &grid);
  }

  const auto &eos = stage_data.eos();
  auto phi = basis.phi();

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
        DEFAULT_FLAT_LOOP_PATTERN, "Hydro :: Fill derived :: temperature",
        DevExecSpace(), ib.s, ib.e, KOKKOS_CLASS_LAMBDA(const int i) {
          double lambda[8];
          for (int q = 0; q < nNodes + 2; ++q) {
            const double rho =
                1.0 / basis.basis_eval(uCF, i, vars::cons::SpecificVolume, q);
            const double vel =
                basis.basis_eval(uCF, i, vars::cons::Velocity, q);
            const double emt = basis.basis_eval(uCF, i, vars::cons::Energy, q);
            const double sie = emt - 0.5 * vel * vel;
            uAF(i, q, vars::aux::Tgas) =
                temperature_from_density_sie(eos, rho, sie, lambda);
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
            const double rho =
                1.0 / basis.basis_eval(uCF, i, vars::cons::SpecificVolume, q);
            const double vel =
                basis.basis_eval(uCF, i, vars::cons::Velocity, q);
            const double emt = basis.basis_eval(uCF, i, vars::cons::Energy, q);

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
        DEFAULT_FLAT_LOOP_PATTERN, "Hydro :: Fill derived", DevExecSpace(),
        ib.s, ib.e, KOKKOS_CLASS_LAMBDA(const int i) {
          for (int q = 0; q < nNodes + 2; ++q) {
            const double rho =
                1.0 / basis.basis_eval(uCF, i, vars::cons::SpecificVolume, q);
            const double vel =
                basis.basis_eval(uCF, i, vars::cons::Velocity, q);
            const double emt = basis.basis_eval(uCF, i, vars::cons::Energy, q);

            const double momentum = rho * vel;
            const double sie = (emt - 0.5 * vel * vel);

            // This is probably not the cleanest logic, but setups with
            // ionization enabled and Paczynski disbled are an outlier.
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

[[nodiscard]] auto HydroPackage::name() const noexcept -> std::string_view {
  return "Hydro";
}

[[nodiscard]] auto HydroPackage::is_active() const noexcept -> bool {
  return active_;
}

void HydroPackage::set_active(const bool active) { active_ = active; }

[[nodiscard]] auto HydroPackage::get_flux_u(const int stage, const int i) const
    -> double {
  return flux_u_(stage, i);
}

} // namespace athelas::fluid
