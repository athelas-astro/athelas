#pragma once

#include "eos/eos_variant.hpp"

namespace athelas {
class MeshState;
class Mesh;
class ProblemIn;
} // namespace athelas

namespace athelas::basis {
class ModalBasis;
} // namespace athelas::basis

namespace athelas::pgen::gas_collapse {

void init(MeshState &mesh_state, Mesh *mesh, ProblemIn *pin);

} // namespace athelas::pgen::gas_collapse
