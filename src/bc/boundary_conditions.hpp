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

} // namespace athelas::bc
