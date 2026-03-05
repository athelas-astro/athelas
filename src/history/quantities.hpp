/**
 * @file quantities.hpp
 * --------------
 *
 * @brief Select quantity calcuations for history
 *
 * TODO(astrobarker): track boundary fluxes
 * TODO(astrobarker): Loop is 4 pi
 */

#include "basic_types.hpp"
#include "geometry/grid.hpp"
#include "kokkos_abstraction.hpp"
#include "loop_layout.hpp"
#include "polynomial_basis.hpp"
#include "state/state.hpp"
#include "utils/constants.hpp"

namespace athelas::analysis {

inline auto total_gravitational_energy(const MeshState &mesh_state,
                                       const GridStructure &grid) -> double {
  using basis::basis_eval;
  const auto &nNodes = grid.n_nodes();
  static const IndexRange ib(grid.domain<Domain::Interior>());
  auto weights = grid.weights();
  auto enclosed_mass = grid.enclosed_mass();
  auto mass_cell = grid.mass();
  auto r = grid.nodal_grid();
  auto u = mesh_state(0).get_field("u_cf");

  const bool do_geometry = grid.do_geometry();

  double output = 0.0;
  athelas::par_reduce(
      DEFAULT_FLAT_LOOP_PATTERN, "History :: TotalGravitationalEnergy",
      DevExecSpace(), ib.s, ib.e,
      KOKKOS_LAMBDA(const int i, double &lsum) {
        double local_sum = 0.0;
        for (int q = 0; q < nNodes; ++q) {
          const double &X = r(i, q + 1);
          local_sum += (enclosed_mass(i, q) / X) * weights(q);
        }
        double mcell = mass_cell(i);
        if (do_geometry) {
          mcell *= constants::FOURPI;
        }
        lsum += local_sum * mcell;
      },
      Kokkos::Sum<double>(output));

  return -constants::G_GRAV * output;
}

// Perhaps the below will be more optimal by calculating
// with cell mass
inline auto total_fluid_energy(const MeshState &mesh_state,
                               const GridStructure &grid) -> double {
  using basis::basis_eval;
  const auto &nNodes = grid.n_nodes();
  static const IndexRange ib(grid.domain<Domain::Interior>());
  const auto dr = grid.widths();
  auto weights = grid.weights();
  auto mcell = grid.mass();

  auto u = mesh_state(0).get_field("u_cf");

  double output = 0.0;
  athelas::par_reduce(
      DEFAULT_FLAT_LOOP_PATTERN, "History :: TotalEnergyFluid", DevExecSpace(),
      ib.s, ib.e,
      KOKKOS_LAMBDA(const int i, double &lsum) {
        double local_sum = 0.0;
        for (int q = 0; q < nNodes; ++q) {
          local_sum += weights(q) * u(i, q, vars::cons::Energy);
        }
        lsum += local_sum * mcell(i);
      },
      Kokkos::Sum<double>(output));

  if (grid.do_geometry()) {
    output *= constants::FOURPI;
  }
  return output;
}

inline auto total_fluid_momentum(const MeshState &mesh_state,
                                 const GridStructure &grid) -> double {
  using basis::basis_eval;
  const auto &nNodes = grid.n_nodes();
  static const IndexRange ib(grid.domain<Domain::Interior>());
  auto dr = grid.widths();
  auto sqrt_gm = grid.sqrt_gm();
  auto weights = grid.weights();

  auto u = mesh_state(0).get_field("u_cf");

  double output = 0.0;
  athelas::par_reduce(
      DEFAULT_FLAT_LOOP_PATTERN, "History :: TotalMomentumFluid",
      DevExecSpace(), ib.s, ib.e,
      KOKKOS_LAMBDA(const int i, double &lsum) {
        double local_sum = 0.0;
        for (int q = 0; q < nNodes; ++q) {
          local_sum += weights(q) * u(i, q, vars::cons::Velocity) /
                       u(i, q, vars::cons::SpecificVolume) * sqrt_gm(i, q + 1) *
                       weights(q);
        }
        lsum += local_sum * dr(i);
      },
      Kokkos::Sum<double>(output));

  if (grid.do_geometry()) {
    output *= constants::FOURPI;
  }
  return output;
}

inline auto total_internal_energy(const MeshState &mesh_state,
                                  const GridStructure &grid) -> double {
  using basis::basis_eval;
  const auto &nNodes = grid.n_nodes();
  static const IndexRange ib(grid.domain<Domain::Interior>());
  auto dr = grid.widths();
  auto sqrt_gm = grid.sqrt_gm();
  auto weights = grid.weights();

  auto u = mesh_state(0).get_field("u_cf");

  double output = 0.0;
  athelas::par_reduce(
      DEFAULT_FLAT_LOOP_PATTERN, "History :: TotalInternalEnergy",
      DevExecSpace(), ib.s, ib.e,
      KOKKOS_LAMBDA(const int i, double &lsum) {
        double local_sum = 0.0;
        for (int q = 0; q < nNodes; ++q) {
          const double vel = u(i, q, vars::cons::Velocity);
          local_sum += (u(i, q, vars::cons::Energy) - 0.5 * vel * vel) /
                       u(i, q, vars::cons::SpecificVolume) * sqrt_gm(i, q + 1) *
                       weights(q);
        }
        lsum += local_sum * dr(i);
      },
      Kokkos::Sum<double>(output));

  if (grid.do_geometry()) {
    output *= constants::FOURPI;
  }
  return output;
}

inline auto total_kinetic_energy(const MeshState &mesh_state,
                                 const GridStructure &grid) -> double {
  using basis::basis_eval;
  const auto &nNodes = grid.n_nodes();
  static const IndexRange ib(grid.domain<Domain::Interior>());
  auto dr = grid.widths();
  auto mass = grid.mass();
  auto sqrt_gm = grid.sqrt_gm();
  auto weights = grid.weights();

  auto phi = mesh_state.fluid_basis().phi();
  auto u = mesh_state(0).get_field("u_cf");

  double output = 0.0;
  athelas::par_reduce(
      DEFAULT_FLAT_LOOP_PATTERN, "History :: TotalKineticEnergy",
      DevExecSpace(), ib.s, ib.e,
      KOKKOS_LAMBDA(const int i, double &lsum) {
        double local_sum = 0.0;
        for (int q = 0; q < nNodes; ++q) {
          const double vel = u(i, q, vars::cons::Velocity);
          local_sum += weights(q) * (0.5 * vel * vel);
        }
        lsum += local_sum * mass(i);
      },
      Kokkos::Sum<double>(output));

  if (grid.do_geometry()) {
    output *= constants::FOURPI;
  }
  return output;
}

// This total_energy is only radiation
inline auto total_rad_energy(const MeshState &mesh_state,
                             const GridStructure &grid) -> double {
  using basis::basis_eval;
  const auto &nNodes = grid.n_nodes();
  static const IndexRange ib(grid.domain<Domain::Interior>());
  auto dm = grid.widths();
  auto weights = grid.weights();

  auto u = mesh_state(0).get_field("u_cf");

  double output = 0.0;
  athelas::par_reduce(
      DEFAULT_FLAT_LOOP_PATTERN, "History :: TotalEnergyRad", DevExecSpace(),
      ib.s, ib.e,
      KOKKOS_LAMBDA(const int i, double &lsum) {
        double local_sum = 0.0;
        for (int q = 0; q < nNodes; ++q) {
          local_sum += u(i, q, vars::cons::RadEnergy) * weights(q);
        }
        lsum += local_sum * dm(i);
      },
      Kokkos::Sum<double>(output));

  if (grid.do_geometry()) {
    output *= constants::FOURPI;
  }
  return output;
}

inline auto total_rad_momentum(const MeshState &mesh_state,
                               const GridStructure &grid) -> double {
  using basis::basis_eval;
  const auto &nNodes = grid.n_nodes();
  static const IndexRange ib(grid.domain<Domain::Interior>());
  auto dr = grid.widths();
  auto sqrt_gm = grid.sqrt_gm();
  auto weights = grid.weights();

  auto phi_rad = mesh_state.rad_basis().phi();
  auto u = mesh_state(0).get_field("u_cf");

  double output = 0.0;
  athelas::par_reduce(
      DEFAULT_FLAT_LOOP_PATTERN, "History :: TotalRadMomentum", DevExecSpace(),
      ib.s, ib.e,
      KOKKOS_LAMBDA(const int i, double &lsum) {
        double local_sum = 0.0;
        for (int q = 0; q < nNodes; ++q) {
          local_sum +=
              u(i, q, vars::cons::RadFlux) * sqrt_gm(i, q + 1) * weights(q);
        }
        lsum += local_sum * dr(i);
      },
      Kokkos::Sum<double>(output));

  if (grid.do_geometry()) {
    output *= constants::FOURPI;
  }
  return output / (constants::c_cgs * constants::c_cgs);
}

// This total_energy is all sources
// NOTE: Pattern is somewhat suboptimal
inline auto total_energy(const MeshState &mesh_state, const GridStructure &grid)
    -> double {
  // Probably could be optimized, but it is history..
  double output = total_fluid_energy(mesh_state, grid);

  const bool radiation_enabled = mesh_state.enabled("radiation");
  if (radiation_enabled) {
    output += total_rad_energy(mesh_state, grid);
  }

  const bool gravity_enabled = mesh_state.enabled("gravity");
  if (gravity_enabled) {
    output += total_gravitational_energy(mesh_state, grid);
  }
  return output;
}

inline auto total_momentum(const MeshState &mesh_state,
                           const GridStructure &grid) -> double {
  double output = total_fluid_momentum(mesh_state, grid);

  // Probably could be optimized, but it is history..
  const bool radiation_enabled = mesh_state.enabled("radiation");
  if (radiation_enabled) {
    output += total_rad_momentum(mesh_state, grid);
  }
  return output;
}

inline auto total_mass(const MeshState &mesh_state, const GridStructure &grid)
    -> double {
  using basis::basis_eval;
  const auto &nNodes = grid.n_nodes();
  static const IndexRange ib(grid.domain<Domain::Interior>());
  auto dr = grid.widths();
  auto sqrt_gm = grid.sqrt_gm();
  auto weights = grid.weights();

  auto phi = mesh_state.fluid_basis().phi();
  auto prims = mesh_state(0).get_field("u_pf");

  double output = 0.0;
  athelas::par_reduce(
      DEFAULT_FLAT_LOOP_PATTERN, "History :: TotalMass", DevExecSpace(), ib.s,
      ib.e,
      KOKKOS_LAMBDA(const int i, double &lsum) {
        double local_sum = 0.0;
        for (int q = 0; q < nNodes; ++q) {
          local_sum +=
              prims(i, q + 1, vars::prim::Rho) * sqrt_gm(i, q + 1) * weights(q);
        }
        lsum += local_sum * dr(i);
      },
      Kokkos::Sum<double>(output));

  if (grid.do_geometry()) {
    output *= constants::FOURPI;
  }
  return output;
}

// TODO(astrobarker): surely there is a non-invasive way to combine
// these total_mass_x. Would need to pass in either index or string for indexer
inline auto total_mass_ni56(const MeshState &mesh_state,
                            const GridStructure &grid) {
  using basis::basis_eval;
  const auto &nNodes = grid.n_nodes();
  static const IndexRange ib(grid.domain<Domain::Interior>());
  auto dr = grid.widths();
  auto sqrt_gm = grid.sqrt_gm();
  auto weights = grid.weights();

  auto phi = mesh_state.fluid_basis().phi();
  auto u = mesh_state(0).get_field("u_cf");
  auto prims = mesh_state(0).get_field("u_pf");

  const auto *const species_indexer = mesh_state.comps()->species_indexer();
  const auto ind_x = species_indexer->get<int>("ni56");

  double output = 0.0;
  athelas::par_reduce(
      DEFAULT_FLAT_LOOP_PATTERN, "History :: TotalMassNi56", DevExecSpace(),
      ib.s, ib.e,
      KOKKOS_LAMBDA(const int i, double &lsum) {
        double local_sum = 0.0;
        for (int q = 0; q < nNodes; ++q) {
          const double x_ni = u(i, q, ind_x);
          local_sum += x_ni * prims(i, q + 1, vars::prim::Rho) *
                       sqrt_gm(i, q + 1) * weights(q);
        }
        lsum += local_sum * dr(i);
      },
      Kokkos::Sum<double>(output));

  if (grid.do_geometry()) {
    output *= constants::FOURPI;
  }
  return output;
}

inline auto total_mass_co56(const MeshState &mesh_state,
                            const GridStructure &grid) {
  using basis::basis_eval;
  const auto &nNodes = grid.n_nodes();
  static const IndexRange ib(grid.domain<Domain::Interior>());
  auto dr = grid.widths();
  auto sqrt_gm = grid.sqrt_gm();
  auto weights = grid.weights();

  auto phi = mesh_state.fluid_basis().phi();
  auto u = mesh_state(0).get_field("u_cf");
  auto prims = mesh_state(0).get_field("u_pf");

  const auto *const species_indexer = mesh_state.comps()->species_indexer();
  const auto ind_x = species_indexer->get<int>("co56");

  double output = 0.0;
  athelas::par_reduce(
      DEFAULT_FLAT_LOOP_PATTERN, "History :: TotalMassCo56", DevExecSpace(),
      ib.s, ib.e,
      KOKKOS_LAMBDA(const int i, double &lsum) {
        double local_sum = 0.0;
        for (int q = 0; q < nNodes; ++q) {
          const double x_co = u(i, q, ind_x);
          local_sum += x_co * prims(i, q + 1, vars::prim::Rho) *
                       sqrt_gm(i, q + 1) * weights(q);
        }
        lsum += local_sum * dr(i);
      },
      Kokkos::Sum<double>(output));

  if (grid.do_geometry()) {
    output *= constants::FOURPI;
  }
  return output;
}

inline auto total_mass_fe56(const MeshState &mesh_state,
                            const GridStructure &grid) {
  using basis::basis_eval;
  const auto &nNodes = grid.n_nodes();
  static const IndexRange ib(grid.domain<Domain::Interior>());
  auto dr = grid.widths();
  auto sqrt_gm = grid.sqrt_gm();
  auto weights = grid.weights();

  auto phi = mesh_state.fluid_basis().phi();
  auto u = mesh_state(0).get_field("u_cf");
  auto prims = mesh_state(0).get_field("u_pf");

  const auto *const species_indexer = mesh_state.comps()->species_indexer();
  const auto ind_x = species_indexer->get<int>("fe56");

  double output = 0.0;
  athelas::par_reduce(
      DEFAULT_FLAT_LOOP_PATTERN, "History :: TotalMassFe56", DevExecSpace(),
      ib.s, ib.e,
      KOKKOS_LAMBDA(const int i, double &lsum) {
        double local_sum = 0.0;
        for (int q = 0; q < nNodes; ++q) {
          const double x_fe = u(i, q, ind_x);
          local_sum += x_fe * prims(i, q + 1, vars::prim::Rho) *
                       sqrt_gm(i, q + 1) * weights(q);
        }
        lsum += local_sum * dr(i);
      },
      Kokkos::Sum<double>(output));

  if (grid.do_geometry()) {
    output *= constants::FOURPI;
  }
  return output;
}

} // namespace athelas::analysis
