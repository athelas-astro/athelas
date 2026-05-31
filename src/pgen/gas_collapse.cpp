#include "pgen/gas_collapse.hpp"

#include "basis/polynomial_basis.hpp"
#include "eos/eos_variant.hpp"
#include "geometry/mesh.hpp"
#include "interface/state.hpp"
#include "kokkos_abstraction.hpp"

namespace athelas::pgen::gas_collapse {

/**
 * @brief Initialize gas collapse
 **/
void init(MeshState &mesh_state, Mesh *mesh, ProblemIn *pin,
          const eos::EOS *eos, basis::ModalBasis * /*fluid_basis = nullptr*/) {
  athelas_requires(pin->param()->get<std::string>("eos.type") == "ideal",
                   "Gas collapse requires ideal gas eos!");

  auto evolved = mesh_state(0).get_field("evolved");
  auto derived = mesh_state(0).get_field("derived");

  const int idx_tau = mesh_state(0).var_index("evolved", "specific_volume");
  const int idx_vel = mesh_state(0).var_index("evolved", "velocity");
  const int idx_ener =
      mesh_state(0).var_index("evolved", "specific_total_fluid_energy");
  const int idx_density = mesh_state(0).var_index("derived", "density");

  static const IndexRange ib(mesh->domain<Domain::Interior>());
  const int nNodes = mesh->n_nodes();

  const auto V0 = pin->param()->get<double>("problem.params.v0", 0.0);
  const auto rho0 = pin->param()->get<double>("problem.params.rho0", 1.0);
  const auto p0 = pin->param()->get<double>("problem.params.p0", 10.0);

  const double gamma = gamma1(*eos);
  const double gm1 = gamma - 1.0;

  athelas::par_for(
      DEFAULT_FLAT_LOOP_PATTERN, "Pgen :: GasCollapse (1)", DevExecSpace(),
      ib.s, ib.e, KOKKOS_LAMBDA(const int i) {
        const int k = 0;

        evolved(i, k, idx_tau) = rho0; // / rho0 * (1.0 / std::cosh(x / H));
        evolved(i, k, idx_vel) = V0;
        evolved(i, k, idx_ener) =
            (p0 / gm1) * evolved(i, k, idx_tau) + 0.5 * V0 * V0;

        for (int iNodeX = 0; iNodeX < nNodes + 2; iNodeX++) {
          derived(i, iNodeX, idx_density) = rho0;
        }
      });

  // Fill density in guard cells
  athelas::par_for(
      DEFAULT_FLAT_LOOP_PATTERN, "Pgen :: GasCollapse (ghost)", DevExecSpace(),
      0, ib.s - 1, KOKKOS_LAMBDA(const int i) {
        for (int iN = 0; iN < nNodes + 2; iN++) {
          derived(ib.s - 1 - i, iN, 0) =
              derived(ib.s + i, (nNodes + 2) - iN - 1, 0);
          derived(ib.s + 1 + i, iN, 0) =
              derived(ib.s - i, (nNodes + 2) - iN - 1, 0);
        }
      });
}

} // namespace athelas::pgen::gas_collapse
