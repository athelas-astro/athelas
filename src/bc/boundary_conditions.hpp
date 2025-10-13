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

#include "basis/polynomial_basis.hpp"
#include "bc/boundary_conditions_base.hpp"
#include "geometry/grid.hpp"
#include "kokkos_abstraction.hpp"
#include "loop_layout.hpp"

namespace athelas::bc {
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
                      const basis::ModalBasis *basis, BoundaryConditions *bcs,
                      const std::tuple<int, int> &vars) {

  const int nX = grid->n_elements();

  auto this_bc = get_bc_data<N>(bcs);

  auto [start, stop] = vars;

  athelas::par_for(
      DEFAULT_FLAT_LOOP_PATTERN, "Fill ghosts", DevExecSpace(), start, stop,
      KOKKOS_LAMBDA(const int v) {
        const int ghost_L = 0;
        const int interior_L = (this_bc[0].type != BcType::Periodic) ? 1 : nX;
        const int ghost_R = nX + 1;
        const int interior_R = (this_bc[1].type != BcType::Periodic) ? nX : 1;

        apply_bc<N>(this_bc[0], U, v, ghost_L, interior_L, basis);
        apply_bc<N>(this_bc[1], U, v, ghost_R, interior_R, basis);
      });
}

// Applies boundary condition for one variable `v`
template <int N>
KOKKOS_INLINE_FUNCTION void
apply_bc(const BoundaryConditionsData<N> &bc, AthelasArray3D<double> U,
         const int v, const int ghost_cell, const int interior_cell,
         const basis::ModalBasis *basis) {
  const int num_modes = basis->order();
  switch (bc.type) {
  case BcType::Outflow:
    for (int k = 0; k < num_modes; k++) {
      U(ghost_cell, k, v) = U(interior_cell, k, v);
    }
    break;

  // NOTE: Literally the same as the above, but the required
  // use is different. interior_cell should be of the opposite
  // side as ghost_cell. Not ideal, but works.
  case BcType::Periodic:
    // assert( interior_cell != ghost_cell + 1 && "Bad use of periodic BC!\n" );
    // assert( interior_cell != ghost_cell - 1 && "Bad use of periodic BC!\n" );
    for (int k = 0; k < num_modes; k++) {
      U(ghost_cell, k, v) = U(interior_cell, k, v);
    }
    break;

  case BcType::Reflecting:
    for (int k = 0; k < num_modes; k++) {
      if (v == 1) { // Momentum (v == 1)
        // Reflect momentum in the cell average (k == 0) and leave higher modes
        // unchanged (k > 0)
        U(ghost_cell, k, v) =
            (k == 0) ? -U(interior_cell, k, v) : U(interior_cell, k, v);
      } else { // Non-momentum variables
        // Reflect cell averages (k == 0) and invert higher modes (k > 0)
        U(ghost_cell, k, v) =
            (k == 0) ? U(interior_cell, k, v) : -U(interior_cell, k, v);
      }
    }
    break;

  // TODO(astrobarker): could need extending. FIX
  case BcType::Dirichlet:
    U(ghost_cell, 0, v) = 2.0 * bc.dirichlet_values[v] - U(interior_cell, 0, v);
    // U(v, ghost_cell, 0) = bc.dirichlet_values[v];
    for (int k = 1; k < num_modes; k++) {
      U(ghost_cell, k, v) = 0.0; // slopes++ set to 0
    }
    break;

  // FIX: check if slope++ are correct.
  case BcType::Marshak: {
    // Marshak uses dirichlet_values
    const double Einc = bc.dirichlet_values[0]; // aT^4
    for (int k = 0; k < 1; k++) {
      if (v == 3) {
        if (k == 0) {
          U(ghost_cell, k, v) = (k == 0) ? Einc : 0;
        }
      } else if (v == 4) {
        constexpr static double c = constants::c_cgs;
        const double E0 = U(interior_cell, k, 3);
        const double F0 = U(interior_cell, k, 4);
        U(ghost_cell, k, v) =
            (k == 0) ? 0.5 * c * Einc - 0.5 * (c * E0 + 2.0 * F0) : 0.0;
      }
    }
  } // case Marshak
  break;

  // formality
  case BcType::Null:
    // TODO(astrobarker) not okay for device
    THROW_ATHELAS_ERROR("Null BC is not for use!");
    break;
  }
}
} // namespace athelas::bc
