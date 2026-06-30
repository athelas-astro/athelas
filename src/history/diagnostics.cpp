#include "history/diagnostics.hpp"

#include <limits>
#include <vector>

#include "eos/eos.hpp"
#include "kokkos_abstraction.hpp"
#include "kokkos_types.hpp"

namespace athelas::diagnostics {

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
  const auto h_weights =
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
  auto h_field = Kokkos::create_mirror_view(field);
  Kokkos::deep_copy(h_field, 0.0); // ghost cells stay zero
  const int idx_tau = mesh_state.var_index("diagnostics", "optical_depth");

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
      lambda.data[eos::EOS_LAMBDA_TEMPERATURE] = T;
      alpha[static_cast<std::size_t>(p)] =
          rho * opac.rosseland_mean(rho, T, X, Z, lambda.ptr());
      cell_tau += h_weights(p) * alpha[static_cast<std::size_t>(p)] * h_dr(i);
    }

    h_field(i, nnodes + 1, idx_tau) = cumulative_tau; // outer face
    h_field(i, 0, idx_tau) = cumulative_tau + cell_tau; // inner face

    for (int q = 0; q < nnodes; ++q) {
      double partial_left = 0.0;
      for (int p = 0; p < nnodes; ++p) {
        partial_left +=
            mesh.integration_matrix(q, p) * alpha[static_cast<std::size_t>(p)];
      }
      h_field(i, q + 1, idx_tau) =
          cumulative_tau + cell_tau - h_dr(i) * partial_left;
    }

    cumulative_tau += cell_tau;
  }

  Kokkos::deep_copy(field, h_field);
}

auto detect_photosphere(const MeshState &mesh_state, const Mesh &mesh,
                      const double tau_target) -> PhotosphereResult {
  PhotosphereResult result{.radius = std::numeric_limits<double>::quiet_NaN(),
                           .cell = -1.0,
                           .valid = 0.0};
  if (!mesh_state.has_field("diagnostics")) {
    return result;
  }

  const int nnodes = mesh.n_nodes();
  const auto field = mesh_state.get_field<AthelasArray3D<double>>("diagnostics");
  const auto h_field =
      Kokkos::create_mirror_view_and_copy(HostMemSpace(), field);
  const auto h_grid =
      Kokkos::create_mirror_view_and_copy(HostMemSpace(), mesh.nodal_grid());
  const int idx_tau = mesh_state.var_index("diagnostics", "optical_depth");

  const int ilo = Mesh::get_ilo();
  const int ihi = mesh.get_ihi();
  for (int i = ihi; i >= ilo; --i) {
    for (int q_outer = nnodes + 1; q_outer > 0; --q_outer) {
      const int q_inner = q_outer - 1;
      const double tau_outer = h_field(i, q_outer, idx_tau);
      const double tau_inner = h_field(i, q_inner, idx_tau);
      if (tau_outer <= tau_target && tau_target <= tau_inner) {
        const double denom = tau_inner - tau_outer;
        const double theta =
            denom > 0.0 ? (tau_target - tau_outer) / denom : 0.0;
        result.radius = h_grid(i, q_outer) +
                        theta * (h_grid(i, q_inner) - h_grid(i, q_outer));
        result.cell = static_cast<double>(i);
        result.valid = 1.0;
        return result;
      }
    }
  }
  return result;
}

auto detect_shock(const MeshState &mesh_state, const Mesh &mesh) -> ShockResult {
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
          v_left += phi(face - 1, nnodes + 1, p) * evolved(face - 1, p, idx_vel);
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
