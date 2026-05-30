#pragma once

namespace athelas {
class MeshState;
class Mesh;
class ProblemIn;
} // namespace athelas

namespace athelas::pgen::sedov {

void init(MeshState &mesh_state, Mesh *mesh, ProblemIn *pin);

} // namespace athelas::pgen::sedov
