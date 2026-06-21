/**
 * @file boundary_conditions.hpp
 * --------------
 *
 * @brief Boundary conditions
 */

#pragma once

#include "bc/boundary_conditions_base.hpp"
#include "kokkos_types.hpp"

namespace athelas {
class MeshState;
class StageData;
} // namespace athelas

namespace athelas::bc {

void ghost_fill(const MeshState &mesh_state, BoundaryConditions *bcs);
void ghost_fill(const StageData &stage_data, BoundaryConditions *bcs);
void ghost_fill(AthelasArray3D<double> U, const StageData &stage_data,
                BoundaryConditions *bcs);

// Copy the derived field into the ghost cells (periodic wrap, else neighbor
// copy), mirroring ghost_fill for the evolved field. fill_derived only computes
// interior cells, so boundary flux reads of derived (pressure, sound speed) at
// the ghosts would otherwise be uninitialized -- harmless for BCs that ignore
// the exterior state, but fatal for periodic, which solves a real Riemann
// problem against it. This is a cheap copy, not an EOS recompute.
void ghost_fill_derived(AthelasArray3D<double> derived,
                        BoundaryConditions *bcs);

} // namespace athelas::bc
