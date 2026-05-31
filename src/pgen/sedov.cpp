#include "pgen/sedov.hpp"

#include <cmath>

#include "eos/eos_variant.hpp"
#include "geometry/mesh.hpp"
#include "interface/state.hpp"
#include "kokkos_abstraction.hpp"

namespace athelas::pgen::sedov {

/**
 * @brief Initialize sedov blast wave
 **/
void init(MeshState &mesh_state, Mesh *mesh, ProblemIn *pin) {
  athelas_requires(pin->param()->get<std::string>("eos.type") == "ideal",
                   "Sedov requires ideal gas eos!");

  auto evolved = mesh_state(0).get_field("evolved");
  auto derived = mesh_state(0).get_field("derived");

  const int idx_tau = mesh_state(0).var_index("evolved", "specific_volume");
  const int idx_vel = mesh_state(0).var_index("evolved", "velocity");
  const int idx_ener =
      mesh_state(0).var_index("evolved", "specific_total_fluid_energy");
  const int idx_density = mesh_state(0).var_index("derived", "density");

  static const int nNodes = mesh->n_nodes();
  static const IndexRange ib(mesh->domain<Domain::Interior>());
  static const IndexRange qb(nNodes);
  auto left_interface = mesh->x_l();

  const auto D0 = pin->param()->get<double>("problem.params.rho0", 1.0);
  const auto V0 = pin->param()->get<double>("problem.params.v0", 0.0);
  const auto E0 = pin->param()->get<double>("problem.params.E0", 0.3);

  const int origin = 1;

  // TODO(astrobarker): geometry aware volume for energy
  auto &eos = mesh_state.eos();
  const double gamma = gamma1(eos);
  const double gm1 = gamma - 1.0;

  athelas::par_for(
      DEFAULT_LOOP_PATTERN, "Pgen :: Sedov", DevExecSpace(), ib.s, ib.e, qb.s,
      qb.e, KOKKOS_LAMBDA(const int i, const int q) {
        const double volume =
            (4.0 * M_PI / 3.0) * std::pow(left_interface(origin + 1), 3.0);
        const double P0 = gm1 * E0 / volume;

        evolved(i, q, idx_tau) = 1.0 / D0;
        evolved(i, q, idx_vel) = V0;
        if (i == origin - 1 || i == origin) {
          evolved(i, q, idx_ener) =
              (P0 / gm1) * evolved(i, q, idx_tau) + 0.5 * V0 * V0;
        } else {
          evolved(i, q, idx_ener) =
              (1.0e-6 / gm1) * evolved(i, q, idx_tau) + 0.5 * V0 * V0;
        }

        derived(i, q + 1, idx_density) = D0;
      });

  // Fill density in guard cells
  athelas::par_for(
      DEFAULT_FLAT_LOOP_PATTERN, "Pgen :: Sedov (ghost)", DevExecSpace(), 0,
      ib.s - 1, KOKKOS_LAMBDA(const int i) {
        for (int iN = 0; iN < nNodes + 2; iN++) {
          derived(ib.s - 1 - i, iN, 0) =
              derived(ib.s + i, (nNodes + 2) - iN - 1, 0);
          derived(ib.e + 1 + i, iN, 0) =
              derived(ib.e - i, (nNodes + 2) - iN - 1, 0);
        }
      });
}

} // namespace athelas::pgen::sedov
