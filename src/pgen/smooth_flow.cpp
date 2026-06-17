#include "pgen/smooth_flow.hpp"

#include <cmath>

#include "geometry/mesh.hpp"
#include "interface/state.hpp"
#include "kokkos_abstraction.hpp"
#include "utils/constants.hpp"

namespace athelas::pgen::smooth_flow {

/**
 * @brief Initialize smooth flow test problem
 * https://doi.org/10.1016/j.camwa.2018.03.040
 **/
void init(MeshState &mesh_state, Mesh *mesh, ProblemIn *pin) {
  athelas_requires(pin->param()->get<std::string>("eos.type") == "ideal",
                   "Smooth flow requires ideal gas eos!");

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

  const auto amp =
      pin->param()->get<double>("problem.params.amp", 0.9999999999999999);

  athelas::par_for(
      DEFAULT_FLAT_LOOP_PATTERN, "Pgen :: SmoothFlow (1)", DevExecSpace(), ib.s,
      ib.e, KOKKOS_LAMBDA(const int i) {
        for (int iNodeX = 0; iNodeX < nNodes; iNodeX++) {
          const double x = mesh->node_coordinate(i, iNodeX);
          derived(i, iNodeX + 1, idx_density) =
              (1.0 + amp * sin(constants::PI * x));
        }
      });

  auto density_func = [&amp](double x, int /*ix*/, int /*iN*/) -> double {
    return 1.0 + amp * sin(constants::PI * x);
  };
  auto velocity_func = [](double /*x*/, int /*ix*/, int /*iN*/) -> double {
    return 0.0;
  };
  auto energy_func = [&amp](double x, int /*ix*/, int /*iN*/) -> double {
    const double D = 1.0 + amp * sin(constants::PI * x);
    return (D * D * D / 2.0) / D;
  };

  static const IndexRange nb(nNodes);
  athelas::par_for(
      DEFAULT_LOOP_PATTERN, "Pgen :: SmoothFlow (2)", DevExecSpace(), ib.s,
      ib.e, nb.s, nb.e, KOKKOS_LAMBDA(const int i, const int node) {
        const double x = mesh->node_coordinate(i, node);
        evolved(i, node, idx_tau) = 1.0 / density_func(x, i, node);
        evolved(i, node, idx_vel) = velocity_func(x, i, node);
        evolved(i, node, idx_ener) = energy_func(x, i, node);
      });
}

} // namespace athelas::pgen::smooth_flow
