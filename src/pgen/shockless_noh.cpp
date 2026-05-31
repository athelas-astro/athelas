#include "pgen/shockless_noh.hpp"

#include "geometry/mesh.hpp"
#include "interface/state.hpp"
#include "kokkos_abstraction.hpp"

namespace athelas::pgen::shockless_noh {

/**
 * @brief Initialize shockless Noh problem
 **/
void init(MeshState &mesh_state, Mesh *mesh, ProblemIn *pin) {
  athelas_requires(pin->param()->get<std::string>("eos.type") == "ideal",
                   "Shockless Noh requires ideal gas eos!");

  auto evolved = mesh_state(0).get_field("evolved");
  auto derived = mesh_state(0).get_field("derived");

  const int idx_tau = mesh_state(0).var_index("evolved", "specific_volume");
  const int idx_vel = mesh_state(0).var_index("evolved", "velocity");
  const int idx_ener =
      mesh_state(0).var_index("evolved", "specific_total_fluid_energy");
  const int idx_density = mesh_state(0).var_index("derived", "density");

  static const int nNodes = mesh->n_nodes();
  static const IndexRange ib(mesh->domain<Domain::Interior>());
  const IndexRange qb(nNodes);

  const auto D = pin->param()->get<double>("problem.params.rho0", 1.0);
  const auto E_M =
      pin->param()->get<double>("problem.params.specific_energy", 1.0);

  athelas::par_for(
      DEFAULT_LOOP_PATTERN, "Pgen :: ShocklessNoh", DevExecSpace(), ib.s, ib.e,
      qb.s, qb.e, KOKKOS_LAMBDA(const int i, const int q) {
        const double X1 = mesh->centers(i);

        evolved(i, q, idx_tau) = 1.0 / D;
        evolved(i, q, idx_vel) = -X1;
        evolved(i, q, idx_ener) =
            E_M + 0.5 * evolved(i, q, idx_vel) * evolved(i, q, idx_vel);

        derived(i, q, idx_density) = D;
      });
}

} // namespace athelas::pgen::shockless_noh
