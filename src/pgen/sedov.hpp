#pragma once

#include <cmath>

#include "eos/eos_variant.hpp"
#include "geometry/grid.hpp"
#include "kokkos_abstraction.hpp"
#include "state/state.hpp"

namespace athelas {

/**
 * @brief Initialize sedov blast wave
 **/
void sedov_init(MeshState &mesh_state, GridStructure *grid, ProblemIn *pin) {
  athelas_requires(pin->param()->get<std::string>("eos.type") == "ideal",
                   "Sedov requires ideal gas eos!");

  auto uCF = mesh_state(0).get_field("u_cf");
  auto uPF = mesh_state(0).get_field("u_pf");

  static const int nNodes = grid->n_nodes();
  static const IndexRange ib(grid->domain<Domain::Interior>());
  static const IndexRange qb(nNodes);
  auto left_interface = grid->x_l();

  const auto D0 = pin->param()->get<double>("problem.params.rho0", 1.0);
  const auto V0 = pin->param()->get<double>("problem.params.v0", 0.0);
  const auto E0 = pin->param()->get<double>("problem.params.E0", 0.3);

  const int origin = 1;

  // TODO(astrobarker): geometry aware volume for energy
  auto &eos = mesh_state.eos();
  const double gamma = gamma1(eos);
  const double gm1 = gamma - 1.0;

  athelas::par_for(
      DEFAULT_LOOP_PATTERN, "Pgen :: Sedov", DevExecSpace(), ib.s,
      ib.e, qb.s, qb.e, KOKKOS_LAMBDA(const int i, const int q) {
        const double volume =
            (4.0 * M_PI / 3.0) * std::pow(left_interface(origin + 1), 3.0);
        const double P0 = gm1 * E0 / volume;

        uCF(i, q, vars::cons::SpecificVolume) = 1.0 / D0;
        uCF(i, q, vars::cons::Velocity) = V0;
        if (i == origin - 1 || i == origin) {
          uCF(i, q, vars::cons::Energy) = (P0 / gm1) * uCF(i, q, vars::cons::SpecificVolume) + 0.5 * V0 * V0;
        } else {
          uCF(i, q, vars::cons::Energy) = (1.0e-6 / gm1) * uCF(i, q, vars::cons::SpecificVolume) + 0.5 * V0 * V0;
        }

          uPF(i, q, vars::prim::Rho) = D0;
      });

  // Fill density in guard cells
  athelas::par_for(
      DEFAULT_FLAT_LOOP_PATTERN, "Pgen :: Sedov (ghost)", DevExecSpace(), 0,
      ib.s - 1, KOKKOS_LAMBDA(const int i) {
        for (int iN = 0; iN < nNodes + 2; iN++) {
          uPF(ib.s - 1 - i, iN, 0) = uPF(ib.s + i, (nNodes + 2) - iN - 1, 0);
          uPF(ib.e + 1 + i, iN, 0) = uPF(ib.e - i, (nNodes + 2) - iN - 1, 0);
        }
      });
}

} // namespace athelas
