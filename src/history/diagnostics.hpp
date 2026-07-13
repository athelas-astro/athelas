#pragma once

#include "geometry/mesh.hpp"
#include "interface/state.hpp"

namespace athelas::diagnostics {

// Photosphere location and photospheric light-curve components. When no tau
// crossing is found, found is false, cell is -1, and the doubles are NaN.
struct PhotosphereDiagnostics {
  double radius;
  int cell;
  bool found;
  double photospheric_luminosity;
  double exterior_radioactive_luminosity;
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
// which must already be up to date (see compute_optical_depth), and estimate
// the light curve there:
//   L = 4 pi R_ph^2 rho F
// plus any radioactive heating deposited exterior to the photosphere when the
// nickel package has published specific_nickel_heating_rate.
[[nodiscard]] auto photosphere_diagnostics(const MeshState &mesh_state,
                                           const Mesh &mesh, double tau_target)
    -> PhotosphereDiagnostics;

// Locate the strongest velocity compression (shock) from the current state.
[[nodiscard]] auto detect_shock(const MeshState &mesh_state, const Mesh &mesh)
    -> ShockResult;

} // namespace athelas::diagnostics
