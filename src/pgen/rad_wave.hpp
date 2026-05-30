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

namespace athelas::pgen::rad_wave {

void init(MeshState &mesh_state, Mesh *mesh, ProblemIn *pin,
          const eos::EOS *eos, basis::ModalBasis *fluid_basis = nullptr,
          basis::ModalBasis *radiation_basis = nullptr);

} // namespace athelas::pgen::rad_wave
