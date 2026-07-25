/**
 * @file quantities.hpp
 * --------------
 *
 * @brief Select quantity calcuations for history
 *
 * @details All integrals are mass integrals in the reference-mass DG
 *          formulation: a conserved/extensive quantity is
 *            Q = \int U dm = \sum_i \sum_q w_q dm_deta(i,q) U(i,q),
 *          optionally times 4pi in spherical symmetry. dm_deta = mu =
 *          sqrt(gamma) * rho * J is the fixed reference-mass measure and
 *          already carries the geometry (sqrt(gamma)) and the (reconstructed)
 *          Jacobian J, so no separate sqrt_gm / dr / tau factors are needed.
 *
 * TODO(astrobarker): track boundary fluxes
 * TODO(astrobarker): Loop is 4 pi
 */

#include "basic_types.hpp"
#include "bc/boundary_conditions.hpp"
#include "fluid/fluid_utilities.hpp"
#include "geometry/mesh.hpp"
#include "gravity/gravity_potential.hpp"
#include "interface/state.hpp"
#include "kokkos_abstraction.hpp"
#include "loop_layout.hpp"
#include "polynomial_basis.hpp"
#include "radiation/rad_utilities.hpp"
#include "utils/constants.hpp"

namespace athelas::analysis {

inline auto geometry_factor(const Mesh &mesh) -> double {
  return mesh.do_geometry() ? constants::FOURPI : 1.0;
}

inline auto total_gravitational_energy(const MeshState & /*mesh_state*/,
                                       const Mesh &mesh,
                                       const GravityModel model,
                                       const double gval) -> double {
  const auto &nNodes = mesh.n_nodes();
  static const IndexRange ib(mesh.domain<Domain::Interior>());
  auto weights = mesh.weights();
  auto enclosed_mass = mesh.enclosed_mass();
  auto dm_deta = mesh.dm_deta();
  auto r = mesh.nodal_grid();

  double output = 0.0;
  athelas::par_reduce(
      DEFAULT_FLAT_LOOP_PATTERN, "History :: TotalGravitationalEnergy",
      DevExecSpace(), ib.s, ib.e,
      KOKKOS_LAMBDA(const int i, double &lsum) {
        double local_sum = 0.0;
        for (int q = 0; q < nNodes; ++q) {
          local_sum += weights(q) * dm_deta(i, q) *
                       gravity::gravitational_potential(
                           model, gval, enclosed_mass(i, q + 1), r(i, q + 1));
        }
        lsum += local_sum;
      },
      Kokkos::Sum<double>(output));

  if (mesh.do_geometry()) {
    output *= constants::FOURPI;
  }
  return output;
}

inline auto total_fluid_energy(const MeshState &mesh_state, const Mesh &mesh)
    -> double {
  const auto &nNodes = mesh.n_nodes();
  static const IndexRange ib(mesh.domain<Domain::Interior>());
  auto weights = mesh.weights();
  auto dm_deta = mesh.dm_deta();

  auto u = mesh_state(0).get_field("evolved");
  const int idx_ener =
      mesh_state.var_index("evolved", "specific_total_fluid_energy");

  double output = 0.0;
  athelas::par_reduce(
      DEFAULT_FLAT_LOOP_PATTERN, "History :: TotalEnergyFluid", DevExecSpace(),
      ib.s, ib.e,
      KOKKOS_LAMBDA(const int i, double &lsum) {
        double local_sum = 0.0;
        for (int q = 0; q < nNodes; ++q) {
          local_sum += weights(q) * dm_deta(i, q) * u(i, q, idx_ener);
        }
        lsum += local_sum;
      },
      Kokkos::Sum<double>(output));

  if (mesh.do_geometry()) {
    output *= constants::FOURPI;
  }
  return output;
}

inline auto total_fluid_momentum(const MeshState &mesh_state, const Mesh &mesh)
    -> double {
  const auto &nNodes = mesh.n_nodes();
  static const IndexRange ib(mesh.domain<Domain::Interior>());
  auto weights = mesh.weights();
  auto dm_deta = mesh.dm_deta();

  auto u = mesh_state(0).get_field("evolved");
  const int idx_vel = mesh_state.var_index("evolved", "velocity");

  double output = 0.0;
  athelas::par_reduce(
      DEFAULT_FLAT_LOOP_PATTERN, "History :: TotalMomentumFluid",
      DevExecSpace(), ib.s, ib.e,
      KOKKOS_LAMBDA(const int i, double &lsum) {
        double local_sum = 0.0;
        // momentum = \int rho v dV = \int v dm
        for (int q = 0; q < nNodes; ++q) {
          local_sum += weights(q) * dm_deta(i, q) * u(i, q, idx_vel);
        }
        lsum += local_sum;
      },
      Kokkos::Sum<double>(output));

  if (mesh.do_geometry()) {
    output *= constants::FOURPI;
  }
  return output;
}

inline auto total_internal_energy(const MeshState &mesh_state, const Mesh &mesh)
    -> double {
  const auto &nNodes = mesh.n_nodes();
  static const IndexRange ib(mesh.domain<Domain::Interior>());
  auto weights = mesh.weights();
  auto dm_deta = mesh.dm_deta();

  auto u = mesh_state(0).get_field("evolved");
  const int idx_vel = mesh_state.var_index("evolved", "velocity");
  const int idx_ener =
      mesh_state.var_index("evolved", "specific_total_fluid_energy");

  double output = 0.0;
  athelas::par_reduce(
      DEFAULT_FLAT_LOOP_PATTERN, "History :: TotalInternalEnergy",
      DevExecSpace(), ib.s, ib.e,
      KOKKOS_LAMBDA(const int i, double &lsum) {
        double local_sum = 0.0;
        for (int q = 0; q < nNodes; ++q) {
          const double vel = u(i, q, idx_vel);
          local_sum += weights(q) * dm_deta(i, q) *
                       (u(i, q, idx_ener) - 0.5 * vel * vel);
        }
        lsum += local_sum;
      },
      Kokkos::Sum<double>(output));

  if (mesh.do_geometry()) {
    output *= constants::FOURPI;
  }
  return output;
}

inline auto total_kinetic_energy(const MeshState &mesh_state, const Mesh &mesh)
    -> double {
  const auto &nNodes = mesh.n_nodes();
  static const IndexRange ib(mesh.domain<Domain::Interior>());
  auto weights = mesh.weights();
  auto dm_deta = mesh.dm_deta();

  auto u = mesh_state(0).get_field("evolved");
  const int idx_vel = mesh_state.var_index("evolved", "velocity");

  double output = 0.0;
  athelas::par_reduce(
      DEFAULT_FLAT_LOOP_PATTERN, "History :: TotalKineticEnergy",
      DevExecSpace(), ib.s, ib.e,
      KOKKOS_LAMBDA(const int i, double &lsum) {
        double local_sum = 0.0;
        for (int q = 0; q < nNodes; ++q) {
          const double vel = u(i, q, idx_vel);
          local_sum += weights(q) * dm_deta(i, q) * (0.5 * vel * vel);
        }
        lsum += local_sum;
      },
      Kokkos::Sum<double>(output));

  if (mesh.do_geometry()) {
    output *= constants::FOURPI;
  }
  return output;
}

// This total_energy is only radiation
inline auto total_rad_energy(const MeshState &mesh_state, const Mesh &mesh)
    -> double {
  const auto &nNodes = mesh.n_nodes();
  static const IndexRange ib(mesh.domain<Domain::Interior>());
  auto weights = mesh.weights();
  auto dm_deta = mesh.dm_deta();

  auto u = mesh_state(0).get_field("evolved");
  const int idx_rad_energy =
      mesh_state.var_index("evolved", "specific_radiation_energy");

  double output = 0.0;
  athelas::par_reduce(
      DEFAULT_FLAT_LOOP_PATTERN, "History :: TotalEnergyRad", DevExecSpace(),
      ib.s, ib.e,
      KOKKOS_LAMBDA(const int i, double &lsum) {
        double local_sum = 0.0;
        for (int q = 0; q < nNodes; ++q) {
          local_sum += weights(q) * dm_deta(i, q) * u(i, q, idx_rad_energy);
        }
        lsum += local_sum;
      },
      Kokkos::Sum<double>(output));

  if (mesh.do_geometry()) {
    output *= constants::FOURPI;
  }
  return output;
}

inline auto total_rad_momentum(const MeshState &mesh_state, const Mesh &mesh)
    -> double {
  const auto &nNodes = mesh.n_nodes();
  static const IndexRange ib(mesh.domain<Domain::Interior>());
  auto weights = mesh.weights();
  auto dm_deta = mesh.dm_deta();

  auto u = mesh_state(0).get_field("evolved");
  const int idx_rad_flux =
      mesh_state.var_index("evolved", "specific_radiation_flux");

  double output = 0.0;
  athelas::par_reduce(
      DEFAULT_FLAT_LOOP_PATTERN, "History :: TotalRadMomentum", DevExecSpace(),
      ib.s, ib.e,
      KOKKOS_LAMBDA(const int i, double &lsum) {
        double local_sum = 0.0;
        // rad momentum = \int F_r/c^2 dV = (1/c^2) \int frad dm
        for (int q = 0; q < nNodes; ++q) {
          local_sum += weights(q) * dm_deta(i, q) * u(i, q, idx_rad_flux);
        }
        lsum += local_sum;
      },
      Kokkos::Sum<double>(output));

  if (mesh.do_geometry()) {
    output *= constants::FOURPI;
  }
  return output / (constants::c_cgs * constants::c_cgs);
}

inline auto fluid_boundary_energy_rate(const MeshState &mesh_state,
                                       const Mesh &mesh,
                                       bc::BoundaryConditions *bcs) -> double {
  using basis::basis_eval;
  using fluid::FluidRiemannState;
  using fluid::numerical_flux_fluid_with_boundary;

  const auto &nNodes = mesh.n_nodes();
  static const IndexRange ib(mesh.domain<Domain::Interior>());

  auto evolved = mesh_state(0).get_field("evolved");
  auto derived = mesh_state.get_field<AthelasArray3D<double>>("derived");
  auto sqrt_gm = mesh.sqrt_gm();
  auto phi = mesh_state.basis().phi();
  const auto fluid_bcs = bc::fluid_bc_data(bcs);

  const int idx_tau = mesh_state.var_index("evolved", "specific_volume");
  const int idx_vel = mesh_state.var_index("evolved", "velocity");
  const int idx_pre = mesh_state.var_index("derived", "pressure");
  const int idx_cs = mesh_state.var_index("derived", "sound_speed");

  double output = 0.0;
  athelas::par_reduce(
      DEFAULT_FLAT_LOOP_PATTERN, "History :: FluidBoundaryEnergyRate",
      DevExecSpace(), 0, 1,
      KOKKOS_LAMBDA(const int side, double &lsum) {
        const bool inner = side == 0;
        const int face = inner ? ib.s : ib.e + 1;
        const int cell = inner ? ib.s : ib.e;
        const int q_face = inner ? 0 : nNodes + 1;

        const FluidRiemannState left{
            .tau =
                basis_eval<Interface::Right>(phi, evolved, face - 1, idx_tau),
            .v = basis_eval<Interface::Right>(phi, evolved, face - 1, idx_vel),
            .p = derived(face - 1, nNodes + 1, idx_pre),
            .cs = derived(face - 1, nNodes + 1, idx_cs)};
        const FluidRiemannState right{
            .tau = basis_eval<Interface::Left>(phi, evolved, face, idx_tau),
            .v = basis_eval<Interface::Left>(phi, evolved, face, idx_vel),
            .p = derived(face, 0, idx_pre),
            .cs = derived(face, 0, idx_cs)};

        const auto flux = numerical_flux_fluid_with_boundary(
            face, ib.s, ib.e + 1, fluid_bcs, left, right);
        const double sign = inner ? 1.0 : -1.0;
        lsum += sign * flux.u * flux.p * sqrt_gm(cell, q_face);
      },
      Kokkos::Sum<double>(output));

  return geometry_factor(mesh) * output;
}

inline auto radiation_boundary_energy_rate(const MeshState &mesh_state,
                                           const Mesh &mesh,
                                           bc::BoundaryConditions *bcs,
                                           const double ap_coefficient = 0.0)
    -> double {
  using basis::basis_eval;
  using fluid::FluidRiemannState;
  using fluid::numerical_flux_fluid_with_boundary;
  using radiation::numerical_flux_rad_with_boundary;
  using radiation::RadBoundaryState;

  if (!mesh_state.enabled("radiation")) {
    return 0.0;
  }

  const auto &nNodes = mesh.n_nodes();
  static const IndexRange ib(mesh.domain<Domain::Interior>());

  auto evolved = mesh_state(0).get_field("evolved");
  auto derived = mesh_state.get_field<AthelasArray3D<double>>("derived");
  auto sqrt_gm = mesh.sqrt_gm();
  auto dr = mesh.widths();
  auto phi = mesh_state.basis().phi();
  const auto fluid_bcs = bc::fluid_bc_data(bcs);
  const auto rad_bcs = bc::radiation_bc_data(bcs);

  const int idx_tau = mesh_state.var_index("evolved", "specific_volume");
  const int idx_vel = mesh_state.var_index("evolved", "velocity");
  const int idx_pre = mesh_state.var_index("derived", "pressure");
  const int idx_cs = mesh_state.var_index("derived", "sound_speed");
  const int idx_rad_energy =
      mesh_state.var_index("evolved", "specific_radiation_energy");
  const int idx_rad_flux =
      mesh_state.var_index("evolved", "specific_radiation_flux");
  const int idx_tgas = mesh_state.var_index("derived", "gas_temperature");
  const auto &opac = mesh_state(0).opac();
  const bool composition_enabled = mesh_state.enabled("composition");
  AthelasArray3D<double> bulk;
  if (composition_enabled) {
    bulk = mesh_state(0).get_field("bulk_composition");
  }

  double output = 0.0;
  athelas::par_reduce(
      DEFAULT_FLAT_LOOP_PATTERN, "History :: RadBoundaryEnergyRate",
      DevExecSpace(), 0, 1,
      KOKKOS_LAMBDA(const int side, double &lsum) {
        const bool inner = side == 0;
        const int face = inner ? ib.s : ib.e + 1;
        const int cell = inner ? ib.s : ib.e;
        const int q_face = inner ? 0 : nNodes + 1;

        const FluidRiemannState fluid_left{
            .tau =
                basis_eval<Interface::Right>(phi, evolved, face - 1, idx_tau),
            .v = basis_eval<Interface::Right>(phi, evolved, face - 1, idx_vel),
            .p = derived(face - 1, nNodes + 1, idx_pre),
            .cs = derived(face - 1, nNodes + 1, idx_cs)};
        const FluidRiemannState fluid_right{
            .tau = basis_eval<Interface::Left>(phi, evolved, face, idx_tau),
            .v = basis_eval<Interface::Left>(phi, evolved, face, idx_vel),
            .p = derived(face, 0, idx_pre),
            .cs = derived(face, 0, idx_cs)};
        const auto fluid_flux = numerical_flux_fluid_with_boundary(
            face, ib.s, ib.e + 1, fluid_bcs, fluid_left, fluid_right);

        const double rho_l =
            1.0 / basis_eval<Interface::Right>(phi, evolved, face - 1, idx_tau);
        const double rho_r =
            1.0 / basis_eval<Interface::Left>(phi, evolved, face, idx_tau);
        const RadBoundaryState rad_left{
            .E = basis_eval<Interface::Right>(phi, evolved, face - 1,
                                              idx_rad_energy) *
                 rho_l,
            .F = basis_eval<Interface::Right>(phi, evolved, face - 1,
                                              idx_rad_flux) *
                 rho_l};
        const RadBoundaryState rad_right{
            .E = basis_eval<Interface::Left>(phi, evolved, face,
                                             idx_rad_energy) *
                 rho_r,
            .F = basis_eval<Interface::Left>(phi, evolved, face, idx_rad_flux) *
                 rho_r};

        double beta = 1.0;
        if (ap_coefficient > 0.0) {
          const int state_cell = inner ? face : face - 1;
          const int state_node = inner ? 0 : nNodes + 1;
          const double rho = inner ? rho_r : rho_l;
          const double tau = radiation::cell_optical_depth(
              opac, derived, bulk, composition_enabled, idx_tgas, state_cell,
              state_node, rho, dr(state_cell));
          beta = radiation::ap_dissipation_factor(tau, ap_coefficient);
        }

        const auto rad_flux = numerical_flux_rad_with_boundary(
            face, ib.s, ib.e + 1, rad_bcs, rad_left, rad_right, fluid_flux.u,
            beta);
        const double sign = inner ? 1.0 : -1.0;
        lsum += sign * rad_flux.e * sqrt_gm(cell, q_face);
      },
      Kokkos::Sum<double>(output));

  return geometry_factor(mesh) * output;
}

// This total_energy is all sources. `model` and `gval` are only consulted when
// gravity is enabled; callers that never enable gravity may pass any value.
// NOTE: Pattern is somewhat suboptimal
inline auto total_energy(const MeshState &mesh_state, const Mesh &mesh,
                         const GravityModel model, const double gval)
    -> double {
  // Probably could be optimized, but it is history..
  double output = total_fluid_energy(mesh_state, mesh);

  const bool radiation_enabled = mesh_state.enabled("radiation");
  if (radiation_enabled) {
    output += total_rad_energy(mesh_state, mesh);
  }

  const bool gravity_enabled = mesh_state.enabled("gravity");
  if (gravity_enabled) {
    output += total_gravitational_energy(mesh_state, mesh, model, gval);
  }
  return output;
}

inline auto total_momentum(const MeshState &mesh_state, const Mesh &mesh)
    -> double {
  double output = total_fluid_momentum(mesh_state, mesh);

  // Probably could be optimized, but it is history..
  const bool radiation_enabled = mesh_state.enabled("radiation");
  if (radiation_enabled) {
    output += total_rad_momentum(mesh_state, mesh);
  }
  return output;
}

inline auto total_mass(const MeshState &mesh_state, const Mesh &mesh)
    -> double {
  const auto &nNodes = mesh.n_nodes();
  static const IndexRange ib(mesh.domain<Domain::Interior>());
  auto weights = mesh.weights();
  auto dm_deta = mesh.dm_deta();

  // total mass = \int rho dV = \int dm = sum w_q dm_deta
  double output = 0.0;
  athelas::par_reduce(
      DEFAULT_FLAT_LOOP_PATTERN, "History :: TotalMass", DevExecSpace(), ib.s,
      ib.e,
      KOKKOS_LAMBDA(const int i, double &lsum) {
        double local_sum = 0.0;
        for (int q = 0; q < nNodes; ++q) {
          local_sum += weights(q) * dm_deta(i, q);
        }
        lsum += local_sum;
      },
      Kokkos::Sum<double>(output));

  if (mesh.do_geometry()) {
    output *= constants::FOURPI;
  }
  return output;
}

// TODO(astrobarker): surely there is a non-invasive way to combine
// these total_mass_x. Would need to pass in either index or string for indexer
inline auto total_mass_ni56(const MeshState &mesh_state, const Mesh &mesh) {
  const auto &nNodes = mesh.n_nodes();
  static const IndexRange ib(mesh.domain<Domain::Interior>());
  auto weights = mesh.weights();
  auto dm_deta = mesh.dm_deta();

  auto mass_fractions = mesh_state.mass_fractions("evolved");

  const auto *const species_indexer = mesh_state.comps()->species_indexer();
  const auto ind_x = species_indexer->get<int>("ni56");

  double output = 0.0;
  athelas::par_reduce(
      DEFAULT_FLAT_LOOP_PATTERN, "History :: TotalMassNi56", DevExecSpace(),
      ib.s, ib.e,
      KOKKOS_LAMBDA(const int i, double &lsum) {
        double local_sum = 0.0;
        for (int q = 0; q < nNodes; ++q) {
          local_sum += weights(q) * dm_deta(i, q) * mass_fractions(i, q, ind_x);
        }
        lsum += local_sum;
      },
      Kokkos::Sum<double>(output));

  if (mesh.do_geometry()) {
    output *= constants::FOURPI;
  }
  return output;
}

inline auto total_mass_co56(const MeshState &mesh_state, const Mesh &mesh) {
  const auto &nNodes = mesh.n_nodes();
  static const IndexRange ib(mesh.domain<Domain::Interior>());
  auto weights = mesh.weights();
  auto dm_deta = mesh.dm_deta();

  auto mass_fractions = mesh_state.mass_fractions("evolved");

  const auto *const species_indexer = mesh_state.comps()->species_indexer();
  const auto ind_x = species_indexer->get<int>("co56");

  double output = 0.0;
  athelas::par_reduce(
      DEFAULT_FLAT_LOOP_PATTERN, "History :: TotalMassCo56", DevExecSpace(),
      ib.s, ib.e,
      KOKKOS_LAMBDA(const int i, double &lsum) {
        double local_sum = 0.0;
        for (int q = 0; q < nNodes; ++q) {
          local_sum += weights(q) * dm_deta(i, q) * mass_fractions(i, q, ind_x);
        }
        lsum += local_sum;
      },
      Kokkos::Sum<double>(output));

  if (mesh.do_geometry()) {
    output *= constants::FOURPI;
  }
  return output;
}

inline auto total_mass_fe56(const MeshState &mesh_state, const Mesh &mesh) {
  const auto &nNodes = mesh.n_nodes();
  static const IndexRange ib(mesh.domain<Domain::Interior>());
  auto weights = mesh.weights();
  auto dm_deta = mesh.dm_deta();

  auto mass_fractions = mesh_state.mass_fractions("evolved");

  const auto *const species_indexer = mesh_state.comps()->species_indexer();
  const auto ind_x = species_indexer->get<int>("fe56");

  double output = 0.0;
  athelas::par_reduce(
      DEFAULT_FLAT_LOOP_PATTERN, "History :: TotalMassFe56", DevExecSpace(),
      ib.s, ib.e,
      KOKKOS_LAMBDA(const int i, double &lsum) {
        double local_sum = 0.0;
        for (int q = 0; q < nNodes; ++q) {
          local_sum += weights(q) * dm_deta(i, q) * mass_fractions(i, q, ind_x);
        }
        lsum += local_sum;
      },
      Kokkos::Sum<double>(output));

  if (mesh.do_geometry()) {
    output *= constants::FOURPI;
  }
  return output;
}

} // namespace athelas::analysis
