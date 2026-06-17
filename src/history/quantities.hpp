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
#include "geometry/mesh.hpp"
#include "interface/state.hpp"
#include "kokkos_abstraction.hpp"
#include "loop_layout.hpp"
#include "polynomial_basis.hpp"
#include "utils/constants.hpp"

namespace athelas::analysis {

inline auto total_gravitational_energy(const MeshState &mesh_state,
                                       const Mesh &mesh) -> double {
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
                       (enclosed_mass(i, q + 1) / r(i, q + 1));
        }
        lsum += local_sum;
      },
      Kokkos::Sum<double>(output));

  if (mesh.do_geometry()) {
    output *= constants::FOURPI;
  }
  return -constants::G_GRAV * output;
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

// This total_energy is all sources
// NOTE: Pattern is somewhat suboptimal
inline auto total_energy(const MeshState &mesh_state, const Mesh &mesh)
    -> double {
  // Probably could be optimized, but it is history..
  double output = total_fluid_energy(mesh_state, mesh);

  const bool radiation_enabled = mesh_state.enabled("radiation");
  if (radiation_enabled) {
    output += total_rad_energy(mesh_state, mesh);
  }

  const bool gravity_enabled = mesh_state.enabled("gravity");
  if (gravity_enabled) {
    output += total_gravitational_energy(mesh_state, mesh);
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

  auto evolved = mesh_state(0).get_field("evolved");

  const auto *const species_indexer = mesh_state.comps()->species_indexer();
  const auto ind_x = species_indexer->get<int>("ni56");

  double output = 0.0;
  athelas::par_reduce(
      DEFAULT_FLAT_LOOP_PATTERN, "History :: TotalMassNi56", DevExecSpace(),
      ib.s, ib.e,
      KOKKOS_LAMBDA(const int i, double &lsum) {
        double local_sum = 0.0;
        for (int q = 0; q < nNodes; ++q) {
          local_sum += weights(q) * dm_deta(i, q) * evolved(i, q, ind_x);
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

  auto evolved = mesh_state(0).get_field("evolved");

  const auto *const species_indexer = mesh_state.comps()->species_indexer();
  const auto ind_x = species_indexer->get<int>("co56");

  double output = 0.0;
  athelas::par_reduce(
      DEFAULT_FLAT_LOOP_PATTERN, "History :: TotalMassCo56", DevExecSpace(),
      ib.s, ib.e,
      KOKKOS_LAMBDA(const int i, double &lsum) {
        double local_sum = 0.0;
        for (int q = 0; q < nNodes; ++q) {
          local_sum += weights(q) * dm_deta(i, q) * evolved(i, q, ind_x);
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

  auto evolved = mesh_state(0).get_field("evolved");

  const auto *const species_indexer = mesh_state.comps()->species_indexer();
  const auto ind_x = species_indexer->get<int>("fe56");

  double output = 0.0;
  athelas::par_reduce(
      DEFAULT_FLAT_LOOP_PATTERN, "History :: TotalMassFe56", DevExecSpace(),
      ib.s, ib.e,
      KOKKOS_LAMBDA(const int i, double &lsum) {
        double local_sum = 0.0;
        for (int q = 0; q < nNodes; ++q) {
          local_sum += weights(q) * dm_deta(i, q) * evolved(i, q, ind_x);
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
