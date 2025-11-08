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

// Perhaps the below will be more optimal by calculating
// with cell mass
inline auto total_fluid_energy(const State &state, const GridStructure &grid,
                               const basis::ModalBasis *fluid_basis,
                               const basis::ModalBasis * /*rad_basis*/)
    -> double {
  using basis::basis_eval;
  const auto &nNodes = grid.n_nodes();
  static const IndexRange ib(grid.domain<Domain::Interior>());
  const auto dr = grid.widths();
  const auto sqrt_gm = grid.sqrt_gm();
  const auto weights = grid.weights();

  const auto phi = fluid_basis->phi();

  const auto u = state.u_cf();

  double output = 0.0;
  athelas::par_reduce(
      DEFAULT_FLAT_LOOP_PATTERN, "History :: TotalEnergyFluid", DevExecSpace(),
      ib.s, ib.e,
      KOKKOS_LAMBDA(const int i, double &lsum) {
        double local_sum = 0.0;
        for (int q = 0; q < nNodes; ++q) {
          local_sum +=
              basis_eval(phi, u, i, vars::cons::Energy, q + 1) /
              basis_eval(phi, u, i, vars::cons::SpecificVolume, q + 1) *
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

inline auto total_fluid_momentum(const State &state, const GridStructure &grid,
                                 const basis::ModalBasis *fluid_basis,
                                 const basis::ModalBasis * /*rad_basis*/)
    -> double {
  using basis::basis_eval;
  const auto &nNodes = grid.n_nodes();
  static const IndexRange ib(grid.domain<Domain::Interior>());
  const auto dr = grid.widths();
  const auto sqrt_gm = grid.sqrt_gm();
  const auto weights = grid.weights();

  const auto phi = fluid_basis->phi();
  const auto u = state.u_cf();

  double output = 0.0;
  athelas::par_reduce(
      DEFAULT_FLAT_LOOP_PATTERN, "History :: TotalMomentumFluid",
      DevExecSpace(), ib.s, ib.e,
      KOKKOS_LAMBDA(const int i, double &lsum) {
        double local_sum = 0.0;
        for (int q = 0; q < nNodes; ++q) {
          local_sum +=
              basis_eval(phi, u, i, vars::cons::Velocity, q + 1) /
              basis_eval(phi, u, i, vars::cons::SpecificVolume, q + 1) *
              basis_eval(phi, u, i, vars::cons::SpecificVolume, q + 1) *
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

inline auto total_internal_energy(const State &state, const GridStructure &grid,
                                  const basis::ModalBasis *fluid_basis,
                                  const basis::ModalBasis * /*rad_basis*/)
    -> double {
  using basis::basis_eval;
  const auto &nNodes = grid.n_nodes();
  static const IndexRange ib(grid.domain<Domain::Interior>());
  const auto dr = grid.widths();
  const auto sqrt_gm = grid.sqrt_gm();
  const auto weights = grid.weights();

  const auto phi = fluid_basis->phi();
  const auto u = state.u_cf();

  double output = 0.0;
  athelas::par_reduce(
      DEFAULT_FLAT_LOOP_PATTERN, "History :: TotalInternalEnergy",
      DevExecSpace(), ib.s, ib.e,
      KOKKOS_LAMBDA(const int i, double &lsum) {
        double local_sum = 0.0;
        for (int q = 0; q < nNodes; ++q) {
          const double vel = basis_eval(phi, u, i, vars::cons::Velocity, q + 1);
          local_sum +=
              (basis_eval(phi, u, i, vars::cons::Energy, q + 1) -
               0.5 * vel * vel) /
              basis_eval(phi, u, i, vars::cons::SpecificVolume, q + 1) *
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

inline auto total_gravitational_energy(const State &state,
                                       const GridStructure &grid,
                                       const basis::ModalBasis *fluid_basis,
                                       const basis::ModalBasis * /*rad_basis*/)
    -> double {
  using basis::basis_eval;
  const auto &nNodes = grid.n_nodes();
  static const IndexRange ib(grid.domain<Domain::Interior>());
  const auto weights = grid.weights();
  const auto enclosed_mass = grid.enclosed_mass();
  const auto mass_cell = grid.mass();
  const auto r = grid.nodal_grid();

  const auto phi = fluid_basis->phi();
  const auto u = state.u_cf();

  double output = 0.0;
  athelas::par_reduce(
      DEFAULT_FLAT_LOOP_PATTERN, "History :: TotalGravitationalEnergy",
      DevExecSpace(), ib.s, ib.e,
      KOKKOS_LAMBDA(const int i, double &lsum) {
        double local_sum = 0.0;
        for (int q = 0; q < nNodes; ++q) {
          const double X = r(i, q);
          local_sum += (enclosed_mass(i, q) * mass_cell(i) / X) * weights(q);
        }
        lsum += local_sum;
      },
      Kokkos::Sum<double>(output));

  if (grid.do_geometry()) {
    output *= constants::FOURPI;
  }
  return -constants::G_GRAV * output;
}

inline auto total_kinetic_energy(const State &state, const GridStructure &grid,
                                 const basis::ModalBasis *fluid_basis,
                                 const basis::ModalBasis * /*rad_basis*/)
    -> double {
  using basis::basis_eval;
  const auto &nNodes = grid.n_nodes();
  static const IndexRange ib(grid.domain<Domain::Interior>());
  const auto dr = grid.widths();
  const auto sqrt_gm = grid.sqrt_gm();
  const auto weights = grid.weights();

  const auto phi = fluid_basis->phi();
  const auto u = state.u_cf();

  double output = 0.0;
  athelas::par_reduce(
      DEFAULT_FLAT_LOOP_PATTERN, "History :: TotalKineticEnergy",
      DevExecSpace(), ib.s, ib.e,
      KOKKOS_LAMBDA(const int i, double &lsum) {
        double local_sum = 0.0;
        for (int q = 0; q < nNodes; ++q) {
          const double vel = basis_eval(phi, u, i, vars::cons::Velocity, q + 1);
          local_sum +=
              (0.5 * vel * vel) /
              basis_eval(phi, u, i, vars::cons::SpecificVolume, q + 1) *
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

// This total_energy is only radiation
inline auto total_rad_energy(const State &state, const GridStructure &grid,
                             const basis::ModalBasis * /*fluid_basis*/,
                             const basis::ModalBasis *rad_basis) -> double {
  using basis::basis_eval;
  const auto &nNodes = grid.n_nodes();
  static const IndexRange ib(grid.domain<Domain::Interior>());
  const auto dr = grid.widths();
  const auto sqrt_gm = grid.sqrt_gm();
  const auto weights = grid.weights();

  const auto phi_rad = rad_basis->phi();
  const auto u = state.u_cf();

  double output = 0.0;
  athelas::par_reduce(
      DEFAULT_FLAT_LOOP_PATTERN, "History :: TotalEnergyRad", DevExecSpace(),
      ib.s, ib.e,
      KOKKOS_LAMBDA(const int i, double &lsum) {
        double local_sum = 0.0;
        for (int q = 0; q < nNodes; ++q) {
          local_sum += basis_eval(phi_rad, u, i, vars::cons::RadEnergy, q + 1) *
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

// TODO(astrobarker): confirm
inline auto total_rad_momentum(const State &state, const GridStructure &grid,
                               const basis::ModalBasis * /*fluid_basis*/,
                               const basis::ModalBasis *rad_basis) -> double {
  using basis::basis_eval;
  const auto &nNodes = grid.n_nodes();
  static const IndexRange ib(grid.domain<Domain::Interior>());
  const auto dr = grid.widths();
  const auto sqrt_gm = grid.sqrt_gm();
  const auto weights = grid.weights();

  const auto phi_rad = rad_basis->phi();
  const auto u = state.u_cf();

  double output = 0.0;
  athelas::par_reduce(
      DEFAULT_FLAT_LOOP_PATTERN, "History :: TotalRadMomentum", DevExecSpace(),
      ib.s, ib.e,
      KOKKOS_LAMBDA(const int i, double &lsum) {
        double local_sum = 0.0;
        for (int q = 0; q < nNodes; ++q) {
          local_sum += basis_eval(phi_rad, u, i, vars::cons::RadFlux, q + 1) *
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

// This total_energy is matter and radiation
inline auto total_energy(const State &state, const GridStructure &grid,
                         const basis::ModalBasis *fluid_basis,
                         const basis::ModalBasis *rad_basis) -> double {
  using basis::basis_eval;
  const auto &nNodes = grid.n_nodes();
  static const IndexRange ib(grid.domain<Domain::Interior>());
  const auto dr = grid.widths();
  const auto sqrt_gm = grid.sqrt_gm();
  const auto weights = grid.weights();

  const auto phi_fluid = fluid_basis->phi();
  const auto phi_rad = rad_basis->phi();
  const auto u = state.u_cf();

  double output = 0.0;
  athelas::par_reduce(
      DEFAULT_FLAT_LOOP_PATTERN, "History :: TotalEnergy", DevExecSpace(), ib.s,
      ib.e,
      KOKKOS_LAMBDA(const int i, double &lsum) {
        double local_sum = 0.0;
        for (int q = 0; q < nNodes; ++q) {
          local_sum +=
              ((basis_eval(phi_fluid, u, i, vars::cons::Energy, q + 1) /
                basis_eval(phi_fluid, u, i, vars::cons::SpecificVolume,
                           q + 1)) +
               basis_eval(phi_rad, u, i, vars::cons::RadEnergy, q + 1)) *
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

// This total_energy is matter and radiation
inline auto total_momentum(const State &state, const GridStructure &grid,
                           const basis::ModalBasis *fluid_basis,
                           const basis::ModalBasis *rad_basis) -> double {
  using basis::basis_eval;
  const auto &nNodes = grid.n_nodes();
  static const IndexRange ib(grid.domain<Domain::Interior>());
  const auto dr = grid.widths();
  const auto sqrt_gm = grid.sqrt_gm();
  const auto weights = grid.weights();

  const auto phi_fluid = fluid_basis->phi();
  const auto phi_rad = rad_basis->phi();
  const auto u = state.u_cf();

  double output = 0.0;
  athelas::par_reduce(
      DEFAULT_FLAT_LOOP_PATTERN, "History :: TotalMomentum", DevExecSpace(),
      ib.s, ib.e,
      KOKKOS_LAMBDA(const int i, double &lsum) {
        double local_sum = 0.0;
        for (int q = 0; q < nNodes; ++q) {
          local_sum +=
              ((basis_eval(phi_fluid, u, i, vars::cons::Velocity, q + 1) /
                basis_eval(phi_fluid, u, i, vars::cons::SpecificVolume,
                           q + 1)) +
               basis_eval(phi_rad, u, i, vars::cons::RadFlux, q + 1)) *
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

inline auto total_mass(const State &state, const GridStructure &grid,
                       const basis::ModalBasis *fluid_basis,
                       const basis::ModalBasis * /*rad_basis*/) -> double {
  using basis::basis_eval;
  const auto &nNodes = grid.n_nodes();
  static const IndexRange ib(grid.domain<Domain::Interior>());
  const auto dr = grid.widths();
  const auto sqrt_gm = grid.sqrt_gm();
  const auto weights = grid.weights();

  const auto phi = fluid_basis->phi();
  const auto u = state.u_cf();

  double output = 0.0;
  athelas::par_reduce(
      DEFAULT_FLAT_LOOP_PATTERN, "History :: TotalMass", DevExecSpace(), ib.s,
      ib.e,
      KOKKOS_LAMBDA(const int i, double &lsum) {
        double local_sum = 0.0;
        for (int q = 0; q < nNodes; ++q) {
          local_sum +=
              (1.0 / basis_eval(phi, u, i, vars::cons::SpecificVolume, q + 1)) *
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

// TODO(astrobarker): surely there is a non-invasive way to combine
// these total_mass_x. Would need to pass in either index or string for indexer
inline auto total_mass_ni56(const State &state, const GridStructure &grid,
                            const basis::ModalBasis *fluid_basis,
                            const basis::ModalBasis * /*rad_basis*/) {
  using basis::basis_eval;
  const auto &nNodes = grid.n_nodes();
  static const IndexRange ib(grid.domain<Domain::Interior>());
  const auto dr = grid.widths();
  const auto sqrt_gm = grid.sqrt_gm();
  const auto weights = grid.weights();

  const auto phi = fluid_basis->phi();
  const auto u = state.u_cf();
  const auto *const species_indexer = state.comps()->species_indexer();
  const auto ind_x = species_indexer->get<int>("ni56");

  double output = 0.0;
  athelas::par_reduce(
      DEFAULT_FLAT_LOOP_PATTERN, "History :: TotalMassNi56", DevExecSpace(),
      ib.s, ib.e,
      KOKKOS_LAMBDA(const int i, double &lsum) {
        double local_sum = 0.0;
        for (int q = 0; q < nNodes; ++q) {
          const double x_ni = basis_eval(phi, u, i, ind_x, q + 1);
          local_sum +=
              x_ni *
              (1.0 / basis_eval(phi, u, i, vars::cons::SpecificVolume, q + 1)) *
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

inline auto total_mass_co56(const State &state, const GridStructure &grid,
                            const basis::ModalBasis *fluid_basis,
                            const basis::ModalBasis * /*rad_basis*/) {
  using basis::basis_eval;
  const auto &nNodes = grid.n_nodes();
  static const IndexRange ib(grid.domain<Domain::Interior>());
  const auto dr = grid.widths();
  const auto sqrt_gm = grid.sqrt_gm();
  const auto weights = grid.weights();

  const auto phi = fluid_basis->phi();
  const auto u = state.u_cf();
  const auto *const species_indexer = state.comps()->species_indexer();
  const auto ind_x = species_indexer->get<int>("co56");

  double output = 0.0;
  athelas::par_reduce(
      DEFAULT_FLAT_LOOP_PATTERN, "History :: TotalMassCo56", DevExecSpace(),
      ib.s, ib.e,
      KOKKOS_LAMBDA(const int i, double &lsum) {
        double local_sum = 0.0;
        for (int q = 0; q < nNodes; ++q) {
          const double x_co = basis_eval(phi, u, i, ind_x, q + 1);
          local_sum +=
              x_co *
              (1.0 / basis_eval(phi, u, i, vars::cons::SpecificVolume, q + 1)) *
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

inline auto total_mass_fe56(const State &state, const GridStructure &grid,
                            const basis::ModalBasis *fluid_basis,
                            const basis::ModalBasis * /*rad_basis*/) {
  using basis::basis_eval;
  const auto &nNodes = grid.n_nodes();
  static const IndexRange ib(grid.domain<Domain::Interior>());
  const auto dr = grid.widths();
  const auto sqrt_gm = grid.sqrt_gm();
  const auto weights = grid.weights();

  const auto phi = fluid_basis->phi();
  const auto u = state.u_cf();
  const auto *const species_indexer = state.comps()->species_indexer();
  const auto ind_x = species_indexer->get<int>("fe56");

  double output = 0.0;
  athelas::par_reduce(
      DEFAULT_FLAT_LOOP_PATTERN, "History :: TotalMassFe56", DevExecSpace(),
      ib.s, ib.e,
      KOKKOS_LAMBDA(const int i, double &lsum) {
        double local_sum = 0.0;
        for (int q = 0; q < nNodes; ++q) {
          const double x_fe = basis_eval(phi, u, i, ind_x, q + 1);
          local_sum +=
              x_fe *
              (1.0 / basis_eval(phi, u, i, vars::cons::SpecificVolume, q + 1)) *
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
