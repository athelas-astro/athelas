#include "pgen/rad_wave.hpp"

#include "basis/polynomial_basis.hpp"
#include "eos/eos_variant.hpp"
#include "geometry/mesh.hpp"
#include "interface/state.hpp"
#include "kokkos_abstraction.hpp"

namespace athelas::pgen::rad_wave {

/**
 * @brief Initialize radiation wave test
 **/
void init(MeshState &mesh_state, Mesh *mesh, ProblemIn *pin,
          const eos::EOS *eos, basis::ModalBasis * /*fluid_basis = nullptr*/,
          basis::ModalBasis * /*radiation_basis = nullptr*/) {
  const bool rad_active = pin->param()->get<bool>("physics.radiation.enabled");
  athelas_requires(rad_active, "Radiation wave requires radiation enabled!");
  athelas_requires(pin->param()->get<std::string>("eos.type") == "ideal",
                   "Radiation wave requires ideal gas eos!");

  auto evolved = mesh_state(0).get_field("evolved");
  auto derived = mesh_state(0).get_field("derived");

  const int idx_tau = mesh_state(0).var_index("evolved", "specific_volume");
  const int idx_vel = mesh_state(0).var_index("evolved", "velocity");
  const int idx_ener =
      mesh_state(0).var_index("evolved", "specific_total_fluid_energy");
  const int idx_rad_energy =
      mesh_state(0).var_index("evolved", "specific_radiation_energy");
  const int idx_density = mesh_state(0).var_index("derived", "density");

  static const int nNodes = mesh->n_nodes();
  static const IndexRange ib(mesh->domain<Domain::Interior>());
  const IndexRange qb(nNodes);

  [[maybe_unused]] const auto lambda =
      pin->param()->get<double>("problem.params.lambda", 0.1);
  [[maybe_unused]] const auto kappa =
      pin->param()->get<double>("problem.params.kappa", 1.0);
  const auto epsilon =
      pin->param()->get<double>("problem.params.epsilon", 1.0e-6);
  const auto rho0 = pin->param()->get<double>("problem.params.rho0", 1.0);
  const auto P0 = pin->param()->get<double>("problem.params.p0", 1.0e-6);

  // TODO(astrobarker): thread through
  const double gamma = gamma1(*eos);
  const double gm1 = gamma - 1.0;

  athelas::par_for(
      DEFAULT_FLAT_LOOP_PATTERN, "Pgen :: RadWave (1)", DevExecSpace(), ib.s,
      ib.e, KOKKOS_LAMBDA(const int i) {
        const int k = 0;
        [[maybe_unused]] const double X1 = mesh->centers(i);

        evolved(i, k, idx_tau) = 1.0 / rho0;
        evolved(i, k, idx_vel) = 0.0;
        evolved(i, k, idx_ener) = (P0 / gm1) / rho0;
        evolved(i, k, idx_rad_energy) = epsilon;

        for (int iNodeX = 0; iNodeX < nNodes + 2; iNodeX++) {
          derived(i, iNodeX, idx_density) = rho0;
        }
      });

  // Fill density in guard cells
  athelas::par_for(
      DEFAULT_FLAT_LOOP_PATTERN, "Pgen :: RadWave (ghost)", DevExecSpace(), 0,
      ib.s - 1, KOKKOS_LAMBDA(const int i) {
        for (int iN = 0; iN < nNodes + 2; iN++) {
          derived(ib.s - 1 - i, iN, 0) =
              derived(ib.s + i, (nNodes + 2) - iN - 1, 0);
          derived(ib.s + 1 + i, iN, 0) =
              derived(ib.s - i, (nNodes + 2) - iN - 1, 0);
        }
      });
}

} // namespace athelas::pgen::rad_wave
