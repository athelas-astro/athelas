#pragma once

#include "geometry/mesh.hpp"
#include "interface/state.hpp"

namespace athelas::diagnostics {

// Result of the photosphere search. Members are doubles so they can be exposed
// directly as scalar history quantities (cell index / validity included).
struct PhotosphereResult {
  double radius; // NaN if no crossing found
  double cell; // -1 if no crossing found
  double valid; // 1.0 if found, else 0.0
};

// Result of the shock search (strongest velocity compression).
struct ShockResult {
  double radius; // NaN if no compression found
  double cell; // -1 if no compression found
  double compression; // 0 if no compression found
};

// Fill the registered "diagnostics" optical_depth field with the Rosseland
// optical depth integrated inward from the outer boundary. No-op if the field
// is absent. Host-side sequential integral — call only at output cadence.
void compute_optical_depth(const MeshState &mesh_state, const Mesh &mesh);

// Locate the photosphere (tau == tau_target) from the optical_depth field,
// which must already be up to date (see compute_optical_depth).
[[nodiscard]] auto detect_photosphere(const MeshState &mesh_state,
                                    const Mesh &mesh, double tau_target)
    -> PhotosphereResult;

// Locate the strongest velocity compression (shock) from the current state.
[[nodiscard]] auto detect_shock(const MeshState &mesh_state, const Mesh &mesh)
    -> ShockResult;

} // namespace athelas::diagnostics
