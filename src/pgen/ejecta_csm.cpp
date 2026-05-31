#include "pgen/ejecta_csm.hpp"

#include <cmath>

#include "eos/eos_variant.hpp"
#include "geometry/mesh.hpp"
#include "interface/state.hpp"
#include "kokkos_abstraction.hpp"

namespace athelas::pgen::ejecta_csm {

void init(MeshState &mesh_state, Mesh *mesh, ProblemIn *pin) {
  athelas_requires(pin->param()->get<std::string>("eos.type") == "ideal",
                   "Ejecta CSM requires ideal gas eos!");

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

  const auto rstar = pin->param()->get<double>("problem.params.rstar", 0.01);
  const auto vmax =
      pin->param()->get<double>("problem.params.vmax", std::sqrt(10.0 / 3.0));

  const double rstar3 = rstar * rstar * rstar;

  const auto &eos = mesh_state.eos();
  const double gamma = gamma1(eos);
  const double gm1 = gamma - 1.0;

  // We use cell centers for setting up the profile to avoid intense gradients.
  auto r = mesh->centers();
  athelas::par_for(
      DEFAULT_FLAT_LOOP_PATTERN, "Pgen :: EjectaCSM (1)", DevExecSpace(), ib.s,
      ib.e, KOKKOS_LAMBDA(const int i) {
        const double x = r(i);
        for (int q = 0; q < nNodes + 2; q++) {
          if (x <= rstar) {
            derived(i, q, idx_density) =
                1.0 / (constants::FOURPI * rstar3 / 3.0);
          } else {
            derived(i, q, idx_density) = 1.0;
          }
        }
      });

  athelas::par_for(
      DEFAULT_LOOP_PATTERN, "Pgen :: EjectaCSM (2)", DevExecSpace(), ib.s, ib.e,
      qb.s, qb.e, KOKKOS_LAMBDA(const int i, const int q) {
        const double X1 = r(i);

        if (X1 <= rstar) {
          const double rho = 1.0 / (constants::FOURPI * rstar3 / 3.0);
          const double pressure = (1.0e-5) * rho * vmax * vmax;
          const double vel = vmax * (X1 / rstar);
          evolved(i, q, idx_tau) = 1.0 / rho;
          evolved(i, q, idx_vel) = vel;
          evolved(i, q, idx_ener) = (pressure / gm1 / rho) + 0.5 * vel * vel;
        } else {
          const double rho = 1.0;
          const double pressure = (1.0e-5) * rho * vmax * vmax;
          evolved(i, q, idx_tau) = 1.0 / rho;
          evolved(i, q, idx_vel) = 0.0;
          evolved(i, q, idx_ener) = (pressure / gm1 / rho);
        }
      });
}

} // namespace athelas::pgen::ejecta_csm
