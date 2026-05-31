#include "pgen/moving_contact.hpp"

#include "eos/eos_variant.hpp"
#include "geometry/mesh.hpp"
#include "interface/state.hpp"
#include "kokkos_abstraction.hpp"

namespace athelas::pgen::moving_contact {

/**
 * @brief Initialize moving contact discontinuity test
 **/
void init(MeshState &mesh_state, Mesh *mesh, ProblemIn *pin) {
  athelas_requires(pin->param()->get<std::string>("eos.type") == "ideal",
                   "Moving contact requires ideal gas eos!");

  auto evolved = mesh_state(0).get_field("evolved");
  auto derived = mesh_state(0).get_field("derived");

  const int idx_tau = mesh_state(0).var_index("evolved", "specific_volume");
  const int idx_vel = mesh_state(0).var_index("evolved", "velocity");
  const int idx_ener =
      mesh_state(0).var_index("evolved", "specific_total_fluid_energy");
  const int idx_density = mesh_state(0).var_index("derived", "density");

  const int nNodes = mesh->n_nodes();
  static const IndexRange ib(mesh->domain<Domain::Interior>());
  const IndexRange qb(nNodes);

  const auto V0 = pin->param()->get<double>("problem.params.v0", 0.1);
  const auto D_L = pin->param()->get<double>("problem.params.rhoL", 1.4);
  const auto D_R = pin->param()->get<double>("problem.params.rhoR", 1.0);
  const auto P_L = pin->param()->get<double>("problem.params.pL", 1.0);
  const auto P_R = pin->param()->get<double>("problem.params.pR", 1.0);

  const auto &eos = mesh_state.eos();
  const double gamma = gamma1(eos);
  const double gm1 = gamma - 1.0;

  athelas::par_for(
      DEFAULT_LOOP_PATTERN, "Pgen :: MovingContact (1)", DevExecSpace(), ib.s,
      ib.e, qb.s, qb.e, KOKKOS_LAMBDA(const int i, const int q) {
        const double X1 = mesh->centers(i);

        if (X1 <= 0.5) {
          evolved(i, q, idx_tau) = 1.0 / D_L;
          evolved(i, q, idx_vel) = V0;
          evolved(i, q, idx_ener) =
              (P_L / gm1) * evolved(i, q, idx_tau) + 0.5 * V0 * V0;

          derived(i, q, idx_density) = D_L;
        } else {
          evolved(i, q, idx_tau) = 1.0 / D_R;
          evolved(i, q, idx_vel) = V0;
          evolved(i, q, idx_ener) =
              (P_R / gm1) * evolved(i, q, idx_tau) + 0.5 * V0 * V0;

          derived(i, q, idx_density) = D_R;
        }
      });
}

} // namespace athelas::pgen::moving_contact
