#include "pgen/shocktube.hpp"

#include "eos/eos_variant.hpp"
#include "geometry/mesh.hpp"
#include "interface/state.hpp"
#include "kokkos_abstraction.hpp"
#include "loop_layout.hpp"

namespace athelas::pgen::shocktube {

/**
 * @brief Initialize shock tube
 **/
void init(MeshState &mesh_state, Mesh *mesh, ProblemIn *pin) {
  athelas_requires(pin->param()->get<std::string>("eos.type") == "ideal",
                   "Shock tube requires ideal gas eos!");

  auto evolved = mesh_state(0).get_field("evolved");
  auto derived = mesh_state(0).get_field("derived");

  const int idx_tau = mesh_state(0).var_index("evolved", "specific_volume");
  const int idx_vel = mesh_state(0).var_index("evolved", "velocity");
  const int idx_ener =
      mesh_state(0).var_index("evolved", "specific_total_fluid_energy");
  const int idx_density = mesh_state(0).var_index("derived", "density");

  static const IndexRange ib(mesh->domain<Domain::Interior>());
  static const int nNodes = mesh->n_nodes();

  const auto V_L = pin->param()->get<double>("problem.params.vL", 0.0);
  const auto V_R = pin->param()->get<double>("problem.params.vR", 0.0);
  const auto D_L = pin->param()->get<double>("problem.params.rhoL", 1.0);
  const auto D_R = pin->param()->get<double>("problem.params.rhoR", 0.125);
  const auto P_L = pin->param()->get<double>("problem.params.pL", 1.0);
  const auto P_R = pin->param()->get<double>("problem.params.pR", 0.1);
  const auto x_d = pin->param()->get<double>("problem.params.x_d", 0.5);

  auto &eos = mesh_state.eos();
  const double gamma = gamma1(eos);
  const double gm1 = gamma - 1.0;

  athelas::par_for(
      DEFAULT_FLAT_LOOP_PATTERN, "Pgen :: ShockTube (1)", DevExecSpace(), ib.s,
      ib.e, KOKKOS_LAMBDA(const int i) {
        const double X1 = mesh->centers(i);

        if (X1 <= x_d) {
          for (int iNodeX = 0; iNodeX < nNodes + 2; iNodeX++) {
            derived(i, iNodeX, idx_density) = D_L;
          }
        } else {
          for (int iNodeX = 0; iNodeX < nNodes + 2; iNodeX++) {
            derived(i, iNodeX, idx_density) = D_R;
          }
        }
      });

  static const IndexRange nb(nNodes);
  auto r = mesh->nodal_grid();
  athelas::par_for(
      DEFAULT_LOOP_PATTERN, "Pgen :: ShockTube", DevExecSpace(), ib.s, ib.e,
      nb.s, nb.e, KOKKOS_LAMBDA(const int i, const int q) {
        const double x = r(i, q + 1);
        if (x <= x_d) {
          evolved(i, q, idx_tau) = 1.0 / D_L;
          evolved(i, q, idx_vel) = V_L;
          evolved(i, q, idx_ener) =
              (P_L / gm1) * evolved(i, q, idx_tau) + 0.5 * V_L * V_L;
        } else {
          evolved(i, q, idx_tau) = 1.0 / D_R;
          evolved(i, q, idx_vel) = V_R;
          evolved(i, q, idx_ener) =
              (P_R / gm1) * evolved(i, q, idx_tau) + 0.5 * V_R * V_R;
        }
      });
}

} // namespace athelas::pgen::shocktube
