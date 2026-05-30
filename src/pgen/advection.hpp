#pragma once

namespace athelas {
class MeshState;
class Mesh;
class ProblemIn;
} // namespace athelas

namespace athelas::pgen::advection {

void init(MeshState &mesh_state, Mesh *mesh, ProblemIn *pin);

} // namespace athelas::pgen::advection
