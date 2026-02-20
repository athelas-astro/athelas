/**
 * @file boundary_conditions.hpp
 * --------------
 *
 * @brief Boundary conditions
 *
 * @details Implemented BCs
 *            - outflow
 *            - reflecting
 *            - periodic
 *            - Dirichlet
 *            - Marshak
 */

#pragma once

#include "basic_types.hpp"
#include "basis/nodal_basis.hpp"
#include "bc/boundary_conditions_base.hpp"
#include "geometry/grid.hpp"
#include "kokkos_abstraction.hpp"
#include "loop_layout.hpp"

namespace athelas::bc {

void fill_ghost_zones_composition(AthelasArray3D<double> U,
                                  const IndexRange &vb);

/**
 * @brief Apply Boundary Conditions to fluid fields
 *
 * @note Templated on number of variables, probably should change.
 * As it stands, N = 3 for fluid and N = 2 for radiation boundaries.
 *
 * Supported Options:
 *  outflow
 *  reflecting
 *  periodic
 *  dirichlet
 *  marshak
 *
 * TODO(astrobarker): Some generalizing
 * between rad and fluid bcs is needed.
 **/
template <int N> // N = 3 for fluid, N = 2 for rad...
void fill_ghost_zones(AthelasArray3D<double> U, const GridStructure *grid,
                      BoundaryConditions *bcs,
                      const std::tuple<int, int> &vars) {

  const int nX = grid->n_elements();

  auto this_bc = get_bc_data<N>(bcs);

  auto [start, stop] = vars;

  const int num_modes = U.extent(1);
  athelas::par_for(
      DEFAULT_FLAT_LOOP_PATTERN, "Fill ghosts", DevExecSpace(), start, stop,
      KOKKOS_LAMBDA(const int v) {
        const int ghost_L = 0;
        const int interior_L = (this_bc[0].type != BcType::Periodic) ? 1 : nX;
        const int ghost_R = nX + 1;
        const int interior_R = (this_bc[1].type != BcType::Periodic) ? nX : 1;

        apply_bc<N>(this_bc[0], U, v, ghost_L, interior_L, num_modes);
        apply_bc<N>(this_bc[1], U, v, ghost_R, interior_R, num_modes);
      });
}

template <int N>
KOKKOS_INLINE_FUNCTION void
apply_bc(const BoundaryConditionsData<N> &bc,
               AthelasArray3D<double> U,
               const int v,
               const int ghost_cell,
               const int interior_cell,
               const int n_nodes)
{
  switch (bc.type) {

  // --------------------------------------------------
  // OUTFLOW: copy nodal values directly
  // --------------------------------------------------
  case BcType::Outflow:
    for (int i = 0; i < n_nodes; ++i) {
      U(ghost_cell, i, v) = U(interior_cell, i, v);
    }
    break;

  // --------------------------------------------------
  // PERIODIC: copy from mapped interior cell
  // (interior_cell already chosen correctly)
  // --------------------------------------------------
  case BcType::Periodic:
    for (int i = 0; i < n_nodes; ++i) {
      U(ghost_cell, i, v) = U(interior_cell, i, v);
    }
    break;

  // --------------------------------------------------
  // REFLECTING
  //
  // 1) reverse node ordering
  // 2) flip sign of normal momentum component
  // --------------------------------------------------
  case BcType::Reflecting:
    for (int i = 0; i < n_nodes; ++i) {

      const int i_ref = n_nodes - 1 - i;

      if (v == 1 || v == 4) {
        // normal momentum / radiation flux
        U(ghost_cell, i, v) =
            -U(interior_cell, i_ref, v);
      } else {
        // scalar quantities
        U(ghost_cell, i, v) =
            U(interior_cell, i_ref, v);
      }
    }
    break;

  // --------------------------------------------------
  // DIRICHLET
  //
  // Strong nodal enforcement
  // --------------------------------------------------
  case BcType::Dirichlet: {
    const double g = bc.dirichlet_values[v];

    for (int i = 0; i < n_nodes; ++i) {
      U(ghost_cell, i, v) = g;
    }
  } break;

  // --------------------------------------------------
  // MARSHak (radiation)
  //
  // Incoming half-range enforcement
  // --------------------------------------------------
  case BcType::Marshak: {

    constexpr double c = constants::c_cgs;
    const double Einc = bc.dirichlet_values[0] * U(interior_cell, 0, vars::cons::SpecificVolume);

    for (int i = 0; i < n_nodes; ++i) {

      const int i_ref = n_nodes - 1 - i;

      if (v == vars::cons::RadEnergy) {

        // Set incoming radiation energy to Einc
        U(ghost_cell, i, v) = Einc;

      }
      else if (v == vars::cons::RadFlux) {

        const double E0 = U(interior_cell, i_ref,
                            vars::cons::RadEnergy);

        const double F0 = U(interior_cell, i_ref,
                            vars::cons::RadFlux);

        // Marshak incoming flux
        U(ghost_cell, i, v) =
            0.5 * c * Einc
          - 0.5 * (c * E0 + 2.0 * F0);
      }
      else {
        // other vars: simple reflection
        U(ghost_cell, i, v) =
            U(interior_cell, i_ref, v);
      }
    }

  } break;

  // --------------------------------------------------
  case BcType::Null:
    throw_athelas_error("Null BC is not for use!");
    break;
  }
}
} // namespace athelas::bc
