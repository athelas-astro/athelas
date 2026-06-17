#include "pgen/advection.hpp"

#include <cmath> /* sin */

#include "eos/eos_variant.hpp"
#include "geometry/mesh.hpp"
#include "interface/state.hpp"
#include "kokkos_abstraction.hpp"
#include "utils/constants.hpp"

namespace athelas::pgen::advection {

/**
 * @brief Initialize advection test
 **/
void init(MeshState &mesh_state, Mesh *mesh, ProblemIn *pin) {
  athelas_requires(pin->param()->get<std::string>("eos.type") == "ideal",
                   "Advection requires ideal gas eos!");

  // Smooth advection problem
  auto evolved = mesh_state(0).get_field("evolved");
  auto derived = mesh_state(0).get_field("derived");

  const int idx_tau = mesh_state(0).var_index("evolved", "specific_volume");
  const int idx_vel = mesh_state(0).var_index("evolved", "velocity");
  const int idx_ener =
      mesh_state(0).var_index("evolved", "specific_total_fluid_energy");
  const int idx_density = mesh_state(0).var_index("derived", "density");

  static const IndexRange ib(mesh->domain<Domain::Interior>());
  static const int nNodes = mesh->n_nodes();

  const auto V0 = pin->param()->get<double>("problem.params.v0", -1.0);
  const auto P0 = pin->param()->get<double>("problem.params.p0", 0.01);
  const auto Amp = pin->param()->get<double>("problem.params.amp", 1.0);

  const auto &eos = mesh_state.eos();
  const double gamma = gamma1(eos);
  const double gm1 = gamma - 1.0;

  athelas::par_for(
      DEFAULT_FLAT_LOOP_PATTERN, "Pgen :: Advection (1)", DevExecSpace(), ib.s,
      ib.e, KOKKOS_LAMBDA(const int i) {
        for (int iNodeX = 0; iNodeX < nNodes; iNodeX++) {
          const double x = mesh->node_coordinate(i, iNodeX);
          derived(i, iNodeX + 1, idx_density) =
              (2.0 + Amp * sin(2.0 * constants::PI * x));
        }
      });

  auto density_func = [&Amp](double x, int /*ix*/, int /*iN*/) -> double {
    return 2.0 + Amp * sin(2.0 * constants::PI * x);
  };
  auto velocity_func = [&V0](double /*x*/, int /*ix*/, int /*iN*/) -> double {
    return V0;
  };
  auto energy_func = [&P0, &V0, &Amp, &gm1](double x, int /*ix*/,
                                            int /*iN*/) -> double {
    const double rho = 2.0 + Amp * sin(2.0 * constants::PI * x);
    return (P0 / gm1) / rho + 0.5 * V0 * V0;
  };

  static const IndexRange nb(nNodes);
  athelas::par_for(
      DEFAULT_LOOP_PATTERN, "Pgen :: Advection (nodal)", DevExecSpace(), ib.s,
      ib.e, nb.s, nb.e, KOKKOS_LAMBDA(const int i, const int node) {
        const double x = mesh->node_coordinate(i, node);
        evolved(i, node, idx_tau) = 1.0 / density_func(x, i, node);
        evolved(i, node, idx_vel) = velocity_func(x, i, node);
        evolved(i, node, idx_ener) = energy_func(x, i, node);
      });
}

} // namespace athelas::pgen::advection
