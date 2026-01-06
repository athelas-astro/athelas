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

using basis::ModalBasis, basis::basis_eval;
using eos::EOS;

HydroPackage::HydroPackage(const ProblemIn * /*pin*/, int n_stages, EOS *eos,
                           ModalBasis *basis, BoundaryConditions *bcs,
                           double cfl, int nx, bool active)
    : active_(active), nx_(nx), cfl_(cfl), eos_(eos), basis_(basis), bcs_(bcs),
      dFlux_num_("hydro::dFlux_num_", nx + 2 + 1, 3),
      u_f_l_("hydro::u_f_l_", nx + 2, 3), u_f_r_("hydro::u_f_r_", nx + 2, 3),
      flux_u_("hydro::flux_u_", n_stages, nx + 2 + 1),
      delta_("hydro :: delta", n_stages, nx_ + 2, basis->order(), 3) {
} // Need long term solution for flux_u_

void HydroPackage::update_explicit(const StageData &stage_data,
                                   const GridStructure &grid,
                                   const TimeStepInfo &dt_info) const {
  const int stage = dt_info.stage;
  auto ucf = stage_data.get_field("u_cf");

  auto uaf = stage_data.get_field("u_af");

  const auto &order = basis_->order();
  static const IndexRange ib(grid.domain<Domain::Interior>());
  static const IndexRange kb(order);
  static const IndexRange vb(NUM_VARS_);

  // --- Apply BC ---
  bc::fill_ghost_zones<3>(ucf, &grid, basis_, bcs_, {0, 2});
  if (stage_data.composition_enabled()) {
    static const IndexRange vb_comps(
        std::make_pair(NUM_VARS_, stage_data.nvars("u_cf") - 1));
    bc::fill_ghost_zones_composition(ucf, vb_comps);
  }

  // --- Fluid Increment : Divergence ---
  fluid_divergence(stage_data, grid, stage);

  // --- Dvbide update by mass mastrix ---
  const auto inv_mkk = basis_->inv_mass_matrix();
  athelas::par_for(
      DEFAULT_LOOP_PATTERN, "Hydro :: delta / M_kk", DevExecSpace(), ib.s, ib.e,
      kb.s, kb.e, KOKKOS_CLASS_LAMBDA(const int i, const int k) {
        const double &invmkk = inv_mkk(i, k);
        for (int v = vb.s; v <= vb.e; ++v) {
          delta_(stage, i, k, v) *= invmkk;
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

  const auto &nNodes = grid.n_nodes();
  const auto &order = basis_->order();
  static const IndexRange ib(grid.domain<Domain::Interior>());
  static const IndexRange kb(order);
  static const IndexRange vb(NUM_VARS_);

  auto x_l = grid.x_l();
  auto sqrt_gm = grid.sqrt_gm();
  auto weights = grid.weights();

  auto phi = basis_->phi();
  auto dphis = basis_->dphi();

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
      kb.s, kb.e, KOKKOS_CLASS_LAMBDA(const int i, const int k) {
        for (int v = vb.s; v <= vb.e; ++v) {
          delta_(stage, i, k, v) -=
              (+dFlux_num_(i + 1, v) * phi(i, nNodes + 1, k) *
                   sqrt_gm(i, nNodes + 1) -
               dFlux_num_(i + 0, v) * phi(i, 0, k) * sqrt_gm(i, 0));
        }
      });

  if (order > 1) [[likely]] {
    // --- Volume Term ---
    athelas::par_for(
        DEFAULT_LOOP_PATTERN, "Hydro :: Volume Term", DevExecSpace(), ib.s,
        ib.e, kb.s, kb.e, KOKKOS_CLASS_LAMBDA(const int i, const int k) {
          double local_sum1 = 0.0;
          double local_sum2 = 0.0;
          double local_sum3 = 0.0;
          for (int q = 0; q < nNodes; ++q) {
            const double vel =
                basis_eval(phi, ucf, i, vars::cons::Velocity, q + 1);
            const double P = uaf(i, q + 1, vars::aux::Pressure);
            const auto [flux1, flux2, flux3] = flux_fluid(vel, P);
            const double w = weights(q);
            const double dphi = dphis(i, q + 1, k);
            const double sqrtgm = sqrt_gm(i, q + 1);

            local_sum1 += w * flux1 * dphi * sqrtgm;
            local_sum2 += w * flux2 * dphi * sqrtgm;
            local_sum3 += w * flux3 * dphi * sqrtgm;
          }

          delta_(stage, i, k, vars::cons::SpecificVolume) += local_sum1;
          delta_(stage, i, k, vars::cons::Velocity) += local_sum2;
          delta_(stage, i, k, vars::cons::Energy) += local_sum3;
        });
  }
}

/**
 * @brief apply fluid package delta
 */
void HydroPackage::apply_delta(AthelasArray3D<double> lhs,
                               const TimeStepInfo &dt_info) const {
  static const int nx = static_cast<int>(lhs.extent(0));
  static const int nk = static_cast<int>(lhs.extent(1));
  static const IndexRange ib(std::make_pair(1, nx - 2));
  static const IndexRange kb(nk);
  static const IndexRange vb(NUM_VARS_);

  const int stage = dt_info.stage;

  athelas::par_for(
      DEFAULT_LOOP_PATTERN, "Hydro :: Apply delta", DevExecSpace(), ib.s, ib.e,
      kb.s, kb.e, KOKKOS_CLASS_LAMBDA(const int i, const int k) {
        for (int v = vb.s; v <= vb.e; ++v) {
          lhs(i, k, v) += dt_info.dt_coef * delta_(stage, i, k, v);
        }
      });
}

/**
 * @brief zero delta field
 */
void HydroPackage::zero_delta() const noexcept {
  static const IndexRange sb(static_cast<int>(delta_.extent(0)));
  static const IndexRange ib(static_cast<int>(delta_.extent(1)));
  static const IndexRange kb(static_cast<int>(delta_.extent(2)));
  static const IndexRange vb(static_cast<int>(delta_.extent(3)));

  athelas::par_for(
      DEFAULT_LOOP_PATTERN, "Hydro :: Zero delta", DevExecSpace(), sb.s, sb.e,
      ib.s, ib.e, kb.s, kb.e,
      KOKKOS_CLASS_LAMBDA(const int s, const int i, const int k) {
        for (int v = vb.s; v <= vb.e; ++v) {
          delta_(s, i, k, v) = 0.0;
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

  // --- Apply BC ---
  bc::fill_ghost_zones<3>(uCF, &grid, basis_, bcs_, {0, 2});

  if (stage_data.composition_enabled()) {
    // composition boundary condition
    static const IndexRange vb_comps(
        std::make_pair(NUM_VARS_, stage_data.nvars("u_cf") - 1));
    bc::fill_ghost_zones_composition(uCF, vb_comps);
    atom::fill_derived_comps<Domain::Entire>(stage_data, uCF, &grid, basis_);
  }

  auto phi = basis_->phi();

  // First we get the temperature from the density and specific internal
  // energy. The ionization case is involved and so this is all done
  // separately. In that case the temperature solve is coupled to a Saha solve.
  if (ionization_enabled) {
    atom::compute_temperature_with_saha<Domain::Entire, eos::EOSInversion::Sie>(
        eos_, stage_data, uCF, grid, *basis_);
  } else {
    athelas::par_for(
        DEFAULT_FLAT_LOOP_PATTERN, "Hydro :: Fill derived :: temperature",
        DevExecSpace(), ib.s, ib.e, KOKKOS_CLASS_LAMBDA(const int i) {
          double lambda[8];
          for (int q = 0; q < nNodes + 2; ++q) {
            const double rho =
                1.0 / basis_eval(phi, uCF, i, vars::cons::SpecificVolume, q);
            const double vel = basis_eval(phi, uCF, i, vars::cons::Velocity, q);
            const double emt = basis_eval(phi, uCF, i, vars::cons::Energy, q);
            const double sie = emt - 0.5 * vel * vel;
            uAF(i, q, vars::aux::Tgas) =
                temperature_from_density_sie(eos_, rho, sie, lambda);
          }
        });
  }

  athelas::par_for(
      DEFAULT_FLAT_LOOP_PATTERN, "Hydro :: Fill derived", DevExecSpace(), ib.s,
      ib.e, KOKKOS_CLASS_LAMBDA(const int i) {
        for (int q = 0; q < nNodes + 2; ++q) {
          const double rho =
              1.0 / basis_eval(phi, uCF, i, vars::cons::SpecificVolume, q);
          const double vel = basis_eval(phi, uCF, i, vars::cons::Velocity, q);
          const double emt = basis_eval(phi, uCF, i, vars::cons::Energy, q);

          const double momentum = rho * vel;
          const double sie = (emt - 0.5 * vel * vel);

          // This is probably not the cleanest logic, but setups with
          // ionization enabled and Paczynski disbled are an outlier.
          double lambda[8];
          if (ionization_enabled) {
            atom::paczynski_terms(stage_data, i, q, lambda);
          }
          const double t_gas = uAF(i, q, vars::aux::Tgas);
          const double pressure =
              pressure_from_density_temperature(eos_, rho, t_gas, lambda);
          const double cs = sound_speed_from_density_temperature_pressure(
              eos_, rho, t_gas, pressure, lambda);

          uPF(i, q, vars::prim::Rho) = rho;
          uPF(i, q, vars::prim::Momentum) = momentum;
          uPF(i, q, vars::prim::Sie) = sie;

          uAF(i, q, vars::aux::Pressure) = pressure;
          uAF(i, q, vars::aux::Cs) = cs;
        }
      });
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

[[nodiscard]] auto HydroPackage::basis() const -> const ModalBasis * {
  return basis_;
}

} // namespace athelas::fluid
