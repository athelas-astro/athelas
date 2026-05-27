#pragma once

namespace athelas {
class MeshState;
class GridStructure;
class ProblemIn;
} // namespace athelas

namespace athelas::pgen::marshak {

void init(MeshState &mesh_state, GridStructure *grid, ProblemIn *pin);

} // namespace athelas::pgen::marshak
