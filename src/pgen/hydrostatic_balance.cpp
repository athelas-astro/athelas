#include "pgen/hydrostatic_balance.hpp"

#include <cmath>

#include "eos/eos_variant.hpp"
#include "geometry/mesh.hpp"
#include "interface/state.hpp"
#include "kokkos_abstraction.hpp"
#include "solvers/hydrostatic_equilibrium.hpp"

namespace athelas::pgen::hydrostatic_balance {

/**
 * @brief Initialize hydrostatic balance self gravity test
 **/
void init(MeshState &mesh_state, Mesh *mesh, ProblemIn *pin) {
  athelas_requires(pin->param()->get<std::string>("eos.type") == "polytropic",
                   "Hydrostatic balance requires polytropic eos!");

  auto evolved = mesh_state(0).get_field("evolved");
  auto derived = mesh_state(0).get_field("derived");

  const int idx_tau = mesh_state(0).var_index("evolved", "specific_volume");
  const int idx_vel = mesh_state(0).var_index("evolved", "velocity");
  const int idx_ener =
      mesh_state(0).var_index("evolved", "specific_total_fluid_energy");
  const int idx_density = mesh_state(0).var_index("derived", "density");

  static const IndexRange ib(mesh->domain<Domain::Interior>());
  const int nNodes = mesh->n_nodes();

  const auto rho_c = pin->param()->get<double>("problem.params.rho_c", 1.0e8);
  const auto p_thresh =
      pin->param()->get<double>("problem.params.p_threshold", 1.0e-10);

  const auto polytropic_k = pin->param()->get<double>("eos.k");
  const auto polytropic_n = pin->param()->get<double>("eos.n");

  const auto &eos = mesh_state.eos();
  const double gamma = gamma1(eos);
  const double gm1 = gamma - 1.0;

  auto rho_from_p = [&polytropic_k, &polytropic_n](const double p) -> double {
    return std::pow(p / polytropic_k, polytropic_n / (polytropic_n + 1.0));
  };

  auto solver = HydrostaticEquilibrium(rho_c, p_thresh,
                                       pin->param()->get<double>("eos.k"),
                                       pin->param()->get<double>("eos.n"));
  solver.solve(mesh_state, mesh, pin);

  athelas::par_for(
      DEFAULT_FLAT_LOOP_PATTERN, "Pgen :: HydrostaticBalance (1)",
      DevExecSpace(), ib.s, ib.e, KOKKOS_LAMBDA(const int i) {
        for (int iNodeX = 0; iNodeX < nNodes + 2; iNodeX++) {
          derived(i, iNodeX, idx_density) =
              rho_from_p(derived(i, iNodeX, idx_density));
        }
      });

  auto pressure_from_rho = [&](const double rho) -> double {
    return polytropic_k * std::pow(rho, 1.0 + 1.0 / polytropic_n);
  };

  auto tau_func = [&](double /*x*/, int ix, int iN) -> double {
    return 1.0 / derived(ix, iN, 0);
  };
  auto energy_func = [&](double /*x*/, int ix, int iN) -> double {
    const double rho = derived(ix, iN, 0);
    return (pressure_from_rho(rho) / gm1) / rho;
  };

  static const IndexRange qb(nNodes);
  athelas::par_for(
      DEFAULT_LOOP_PATTERN, "Pgen :: HydrostaticBalance (2)", DevExecSpace(),
      ib.s, ib.e, qb.s, qb.e, KOKKOS_LAMBDA(const int i, const int q) {
        const int iN = q + 1; // derived interior index
        evolved(i, q, idx_tau) = tau_func(0.0, i, iN);
        evolved(i, q, idx_vel) = 0.0;
        evolved(i, q, idx_ener) = energy_func(0.0, i, iN);
      });
}

} // namespace athelas::pgen::hydrostatic_balance
