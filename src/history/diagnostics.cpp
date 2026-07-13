#include "history/diagnostics.hpp"

#include <algorithm>
#include <limits>
#include <vector>

#include "basis/polynomial_basis.hpp"
#include "eos/eos.hpp"
#include "kokkos_abstraction.hpp"
#include "kokkos_types.hpp"
#include "utils/constants.hpp"

namespace athelas::diagnostics {

namespace {

struct PhotosphereBracket {
  double radius = std::numeric_limits<double>::quiet_NaN();
  int cell = -1;
  int q_outer = -1;
  int q_inner = -1;
  double theta = 0.0;
  bool found = false;
};

auto find_photosphere_bracket(const MeshState &mesh_state, const Mesh &mesh,
                              const double tau_target) -> PhotosphereBracket {
  PhotosphereBracket bracket;
  if (!mesh_state.has_field("diagnostics")) {
    return bracket;
  }

  const int nnodes = mesh.n_nodes();
  const auto field =
      mesh_state.get_field<AthelasArray3D<double>>("diagnostics");
  const auto field_h =
      Kokkos::create_mirror_view_and_copy(HostMemSpace(), field);
  const auto grid_h =
      Kokkos::create_mirror_view_and_copy(HostMemSpace(), mesh.nodal_grid());
  const int idx_tau = mesh_state.var_index("diagnostics", "optical_depth");

  const int ilo = Mesh::get_ilo();
  const int ihi = mesh.get_ihi();
  for (int i = ihi; i >= ilo; --i) {
    for (int q_outer = nnodes + 1; q_outer > 0; --q_outer) {
      const int q_inner = q_outer - 1;
      const double tau_outer = field_h(i, q_outer, idx_tau);
      const double tau_inner = field_h(i, q_inner, idx_tau);
      if (tau_outer <= tau_target && tau_target <= tau_inner) {
        const double denom = tau_inner - tau_outer;
        const double theta =
            denom > 0.0 ? (tau_target - tau_outer) / denom : 0.0;
        bracket.radius = grid_h(i, q_outer) +
                         theta * (grid_h(i, q_inner) - grid_h(i, q_outer));
        bracket.cell = i;
        bracket.q_outer = q_outer;
        bracket.q_inner = q_inner;
        bracket.theta = theta;
        bracket.found = true;
        return bracket;
      }
    }
  }
  return bracket;
}

} // namespace

void compute_optical_depth(const MeshState &mesh_state, const Mesh &mesh) {
  if (!mesh_state.has_field("diagnostics")) {
    return;
  }
  const int nnodes = mesh.n_nodes();

  const auto derived = mesh_state.get_field<AthelasArray3D<double>>("derived");
  const auto h_derived =
      Kokkos::create_mirror_view_and_copy(HostMemSpace(), derived);
  const auto h_dr =
      Kokkos::create_mirror_view_and_copy(HostMemSpace(), mesh.widths());
  const auto weights_h =
      Kokkos::create_mirror_view_and_copy(HostMemSpace(), mesh.weights());

  const bool composition_enabled = mesh_state.enabled("composition") &&
                                   mesh_state.has_field("bulk_composition");
  AthelasArray3D<double>::HostMirror h_bulk;
  if (composition_enabled) {
    const auto bulk =
        mesh_state.get_field<AthelasArray3D<double>>("bulk_composition");
    h_bulk = Kokkos::create_mirror_view_and_copy(HostMemSpace(), bulk);
  }

  const int idx_rho = mesh_state.var_index("derived", "density");
  const int idx_tgas = mesh_state.var_index("derived", "gas_temperature");
  const auto &opac = mesh_state.opac();

  auto field = mesh_state.get_field<AthelasArray3D<double>>("diagnostics");
  auto field_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), field);
  const int idx_tau = mesh_state.var_index("diagnostics", "optical_depth");
  // Zero only the optical_depth column: the diagnostics field may hold other
  // variables (e.g., specific_nickel_heating_rate) that must be preserved.
  for (std::size_t i = 0; i < field_h.extent(0); ++i) {
    for (std::size_t q = 0; q < field_h.extent(1); ++q) {
      field_h(i, q, idx_tau) = 0.0;
    }
  }

  const int ilo = Mesh::get_ilo();
  const int ihi = mesh.get_ihi();
  double cumulative_tau = 0.0;

  std::vector<double> alpha(static_cast<std::size_t>(nnodes), 0.0);
  for (int i = ihi; i >= ilo; --i) {
    double cell_tau = 0.0;
    for (int p = 0; p < nnodes; ++p) {
      const int node = p + 1;
      const double rho = h_derived(i, node, idx_rho);
      const double T = h_derived(i, node, idx_tgas);
      double X = 0.0;
      double Z = 0.0;
      if (composition_enabled) {
        X = h_bulk(i, node, 0);
        Z = h_bulk(i, node, 2);
      }
      eos::EOSLambda lambda;
      alpha[static_cast<std::size_t>(p)] =
          rho * opac.rosseland_mean(rho, T, X, Z, lambda.ptr());
      cell_tau += weights_h(p) * alpha[static_cast<std::size_t>(p)] * h_dr(i);
    }

    field_h(i, nnodes + 1, idx_tau) = cumulative_tau; // outer face
    field_h(i, 0, idx_tau) = cumulative_tau + cell_tau; // inner face

    for (int q = 0; q < nnodes; ++q) {
      double partial_left = 0.0;
      for (int p = 0; p < nnodes; ++p) {
        partial_left +=
            mesh.integration_matrix(q, p) * alpha[static_cast<std::size_t>(p)];
      }
      field_h(i, q + 1, idx_tau) =
          cumulative_tau + cell_tau - h_dr(i) * partial_left;
    }

    cumulative_tau += cell_tau;
  }

  Kokkos::deep_copy(field, field_h);
}

auto photosphere_diagnostics(const MeshState &mesh_state, const Mesh &mesh,
                             const double tau_target)
    -> PhotosphereDiagnostics {
  const auto bracket = find_photosphere_bracket(mesh_state, mesh, tau_target);
  PhotosphereDiagnostics result{.radius = bracket.radius,
                                .cell = bracket.cell,
                                .found = bracket.found,
                                .photospheric_luminosity =
                                    std::numeric_limits<double>::quiet_NaN(),
                                .exterior_radioactive_luminosity =
                                    std::numeric_limits<double>::quiet_NaN()};

  if (!bracket.found) {
    return result;
  }

  const int idx_tau = mesh_state.var_index("evolved", "specific_volume");
  const int idx_rad_flux =
      mesh_state.var_index("evolved", "specific_radiation_flux");
  const auto evolved = mesh_state(0).get_field("evolved");
  const auto h_evolved =
      Kokkos::create_mirror_view_and_copy(HostMemSpace(), evolved);
  const auto h_phi = Kokkos::create_mirror_view_and_copy(
      HostMemSpace(), mesh_state.basis().phi());

  const int nnodes = mesh.n_nodes();
  const int cell = bracket.cell;
  auto evolved_at = [&](const int q_diag, const int var) -> double {
    return basis::basis_eval(h_phi, h_evolved, cell, var, q_diag);
  };

  const double specific_volume_outer = evolved_at(bracket.q_outer, idx_tau);
  const double specific_volume_inner = evolved_at(bracket.q_inner, idx_tau);
  const double specific_flux_outer = evolved_at(bracket.q_outer, idx_rad_flux);
  const double specific_flux_inner = evolved_at(bracket.q_inner, idx_rad_flux);
  const double theta = bracket.theta;
  const double specific_volume =
      specific_volume_outer +
      theta * (specific_volume_inner - specific_volume_outer);
  const double specific_flux =
      specific_flux_outer + theta * (specific_flux_inner - specific_flux_outer);
  const double rho = 1.0 / specific_volume;
  const double radius = bracket.radius;
  const double area =
      mesh.do_geometry() ? constants::FOURPI * radius * radius : 1.0;
  result.photospheric_luminosity = area * rho * specific_flux;

  result.exterior_radioactive_luminosity = 0.0;
  if (mesh_state.has_variable("diagnostics", "specific_nickel_heating_rate")) {
    const int idx_heating =
        mesh_state.var_index("diagnostics", "specific_nickel_heating_rate");
    const auto diag =
        mesh_state.get_field<AthelasArray3D<double>>("diagnostics");
    const auto diag_h =
        Kokkos::create_mirror_view_and_copy(HostMemSpace(), diag);
    const auto weights_h =
        Kokkos::create_mirror_view_and_copy(HostMemSpace(), mesh.weights());
    const auto dm_deta_h =
        Kokkos::create_mirror_view_and_copy(HostMemSpace(), mesh.dm_deta());
    const auto grid_h =
        Kokkos::create_mirror_view_and_copy(HostMemSpace(), mesh.nodal_grid());

    const int ihi = mesh.get_ihi();
    double exterior_power = 0.0;
    for (int i = cell + 1; i <= ihi; ++i) {
      for (int q = 0; q < nnodes; ++q) {
        exterior_power +=
            weights_h(q) * dm_deta_h(i, q) * diag_h(i, q + 1, idx_heating);
      }
    }

    double photosphere_cell_power = 0.0;
    for (int q = 0; q < nnodes; ++q) {
      photosphere_cell_power +=
          weights_h(q) * dm_deta_h(cell, q) * diag_h(cell, q + 1, idx_heating);
    }
    const double r_inner = grid_h(cell, 0);
    const double r_outer = grid_h(cell, nnodes + 1);
    const double exterior_fraction =
        std::clamp((r_outer - radius) / (r_outer - r_inner), 0.0, 1.0);
    exterior_power += exterior_fraction * photosphere_cell_power;

    result.exterior_radioactive_luminosity =
        (mesh.do_geometry() ? constants::FOURPI : 1.0) * exterior_power;
  }

  return result;
}

auto detect_shock(const MeshState &mesh_state, const Mesh &mesh)
    -> ShockResult {
  ShockResult result{.radius = std::numeric_limits<double>::quiet_NaN(),
                     .cell = -1.0,
                     .compression = 0.0};

  const auto evolved = mesh_state(0).get_field("evolved");
  const auto dr = mesh.widths();
  const auto grid = mesh.nodal_grid();
  const auto diff = mesh_state.basis().differentiation_matrix();
  const auto phi = mesh_state.basis().phi();

  const int idx_vel = mesh_state.var_index("evolved", "velocity");
  const int nnodes = mesh.n_nodes();
  const int ilo = Mesh::get_ilo();
  const int ihi = mesh.get_ihi();

  using custom_reductions::MaxValLoc;

  // Strongest compression at an interior nodal point, -dv/dr = -(dv/deta)/dr.
  MaxValLoc node_best;
  athelas::par_reduce(
      DEFAULT_FLAT_LOOP_PATTERN, "Diagnostics :: Shock nodes", DevExecSpace(),
      ilo, ihi, 0, nnodes - 1,
      KOKKOS_LAMBDA(const int i, const int q, MaxValLoc &lmax) {
        double dv_deta = 0.0;
        for (int p = 0; p < nnodes; ++p) {
          dv_deta += diff(q, p) * evolved(i, p, idx_vel);
        }
        const MaxValLoc cand(-dv_deta / dr(i), grid(i, q + 1), i);
        if (cand > lmax) {
          lmax = cand;
        }
      },
      Kokkos::Max<MaxValLoc>(node_best));

  // Strongest compression across an interior face, from the velocity jump.
  MaxValLoc face_best;
  athelas::par_reduce(
      DEFAULT_FLAT_LOOP_PATTERN, "Diagnostics :: Shock faces", DevExecSpace(),
      ilo + 1, ihi,
      KOKKOS_LAMBDA(const int face, MaxValLoc &lmax) {
        double v_left = 0.0;
        double v_right = 0.0;
        for (int p = 0; p < nnodes; ++p) {
          v_left +=
              phi(face - 1, nnodes + 1, p) * evolved(face - 1, p, idx_vel);
          v_right += phi(face, 0, p) * evolved(face, p, idx_vel);
        }
        const double dx = 0.5 * (dr(face - 1) + dr(face));
        const MaxValLoc cand(-(v_right - v_left) / dx, grid(face, 0), face);
        if (cand > lmax) {
          lmax = cand;
        }
      },
      Kokkos::Max<MaxValLoc>(face_best));

  const MaxValLoc best = face_best > node_best ? face_best : node_best;
  if (best.value > 0.0) {
    result.radius = best.position;
    result.cell = static_cast<double>(best.index);
    result.compression = best.value;
  }
  return result;
}

} // namespace athelas::diagnostics
