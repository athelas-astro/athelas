/**
 * @file gravity_package.cpp
 * --------------
 *
 * @brief Gravitational source package
 **/
#include <algorithm>
#include <limits>

#include "basic_types.hpp"
#include "basis/polynomial_basis.hpp"
#include "eos/eos.hpp"
#include "eos/eos_variant.hpp"
#include "geometry/mesh.hpp"
#include "gravity/gravity_package.hpp"
#include "gravity/gravity_potential.hpp"
#include "kokkos_abstraction.hpp"
#include "loop_layout.hpp"
#include "pgen/problem_in.hpp"
#include "utils/constants.hpp"

namespace athelas::gravity {

using basis::NodalBasis, basis::geometric_weak_eta_derivative;

GravityPackage::GravityPackage(const ProblemIn *pin, const std::string &model,
                               const double gval, const double cfl,
                               const int n_stages, const bool active)
    : active_(active), gval_(gval), cfl_(cfl),
      correct_limiter_energy_(
          pin->param()->get<bool>("gravity.limiter_energy_correction", false)),
      delta_("gravity delta", n_stages, pin->param()->get<int>("mesh.nx") + 2,
             pin->param()->get<int>("basis.nnodes"), 2),
      gravity_pressure_("gravity pressure",
                        pin->param()->get<int>("mesh.nx") + 2,
                        pin->param()->get<int>("basis.nnodes") + 2),
      tau_dot_mesh_velocity_("gravity tau-dot mesh velocity",
                             pin->param()->get<int>("mesh.nx") + 2,
                             pin->param()->get<int>("basis.nnodes")),
      limiter_r_old_("gravity limiter r_old",
                     pin->param()->get<int>("mesh.nx") + 2,
                     pin->param()->get<int>("basis.nnodes")) {
  if (model == "constant") {
    model_ = GravityModel::Constant;
  } else if (model == "spherical") {
    model_ = GravityModel::Spherical;
  } else {
    throw_athelas_error("Bad gravity model in GRavityPackage constructor. How "
                        "did this happen?");
  }

  // The weak gravity-pressure energy companion reads the specific-volume RHS
  // that hydro publishes into `dtau_dt` earlier in the same stage. An
  // operator-split gravity package runs outside that stage and would consume a
  // stale RHS.
  if (pin->param()->get<bool>("physics.gravity.split")) {
    throw_athelas_error("Gravity does not support physics.gravity.split=true: "
                        "the weak energy source requires the specific-volume "
                        "RHS published within the same stage.");
  }
}

auto GravityPackage::update_explicit(const StageData &stage_data,
                                     const TimeStepInfo &dt_info) const
    -> UpdateStatus {
  const auto &mesh = stage_data.mesh();
  const auto stage = dt_info.stage;
  auto interface = stage_data.get_field<AthelasArray2D<double>>("interface");
  const int idx_vstar = stage_data.var_index("interface", "interface_velocity");
  auto dtau_dt = stage_data.get_field<AthelasArray2D<double>>("dtau_dt");
  const auto &basis = stage_data.basis();

  // Hydro/RadHydro publish the mass-matrix-scaled tau RHS before gravity runs.
  // Differentiate the same branch of Mesh::reconstruct_mesh to obtain the
  // interior mesh velocity used by the gravity-pressure energy companion.
  compute_mesh_velocity(mesh, interface, dtau_dt, idx_vstar);

  if (model_ == GravityModel::Spherical) {
    gravity_update<GravityModel::Spherical>(mesh, basis, interface, dtau_dt,
                                            stage, idx_vstar);
  } else [[unlikely]] {
    gravity_update<GravityModel::Constant>(mesh, basis, interface, dtau_dt,
                                           stage, idx_vstar);
  }

  return UpdateStatus::Success;
}

template <GravityModel Model>
void GravityPackage::gravity_update(const Mesh &mesh, const NodalBasis &basis,
                                    AthelasArray2D<double> interface,
                                    AthelasArray2D<double> dtau_dt,
                                    const int stage,
                                    const int idx_vstar) const {
  static const int nNodes = mesh.n_nodes();
  static const IndexRange ib(mesh.domain<Domain::Interior>());
  static const IndexRange qb(nNodes);

  auto r = mesh.nodal_grid();
  auto sqrt_gm = mesh.sqrt_gm();
  auto dm_deta = mesh.dm_deta();
  auto enclosed_mass = mesh.enclosed_mass();
  auto weights = mesh.weights();
  auto phi = basis.phi();
  auto dphi = basis.dphi();
  auto inv_mkk = basis.inv_mass_matrix();
  auto psi_g = gravity_pressure_;
  auto velocity = tau_dot_mesh_velocity_;

  const double gval = gval_;
  // Gravity pressure Pi: the discrete variational derivative of the
  // gravitational potential energy W_h with respect to the specific volume,
  //   H Pi = J^T H (-g/A),
  // where J is the tangent of Mesh::reconstruct_mesh (tau -> spherical volume
  // coordinate). It is built as the reverse-mode sweep of that reconstruction:
  // an inward (outer -> inner) prefix scan whose face accumulator lambda_right
  // is the exclusive potential of all exterior cells, plus the transpose of the
  // within-cell placement (high-order integration matrix, or the frozen
  // mass-fraction fallback). Anchoring lambda = 0 at the surface gives the
  // well-conditioned gauge P = Pi + p_surface for a star. Making Pi the exact
  // adjoint of the reconstruction is what lets the weak energy source conserve
  // hydrodynamic + gravitational energy at P1; see docs/src/gravity.rst.
  {
    const auto high_order = mesh.node_placement_high_order();
    athelas::par_scan(
        DEFAULT_FLAT_LOOP_PATTERN, "Gravity :: Adjoint pressure",
        DevExecSpace(), ib.s, ib.e,
        KOKKOS_CLASS_LAMBDA(const int k, double &lambda_right,
                            const bool is_final) {
          const int i = ib.e - (k - ib.s);
          Kokkos::Array<double, MAX_ORDER> a;
          double cell_adjoint = 0.0;
          for (int q = 0; q < nNodes; ++q) {
            double accel = -constants::G_GRAV * gval;
            if constexpr (Model == GravityModel::Spherical) {
              const double radius = r(i, q + 1);
              accel = -constants::G_GRAV * enclosed_mass(i, q + 1) /
                      (radius * radius);
            }
            a[q] = -weights(q) * dm_deta(i, q) * accel / sqrt_gm(i, q + 1);
            cell_adjoint += a[q];
          }

          if (is_final) {
            const bool ho = high_order(i) != 0;
            const double mass_left = enclosed_mass(i, 0);
            const double mass_width =
                enclosed_mass(i, nNodes + 1) - mass_left;
            psi_g(i, nNodes + 1) = lambda_right;
            for (int p = 0; p < nNodes; ++p) {
              double local_adjoint = 0.0;
              if (ho) {
                for (int q = 0; q < nNodes; ++q) {
                  local_adjoint += mesh.integration_matrix(q, p) * a[q];
                }
                psi_g(i, p + 1) =
                    lambda_right + local_adjoint / weights(p);
              } else {
                for (int q = 0; q < nNodes; ++q) {
                  const double theta =
                      (mass_width > 0.0)
                          ? Kokkos::clamp(
                                (enclosed_mass(i, q + 1) - mass_left) /
                                    mass_width,
                                0.0, 1.0)
                          : 0.5;
                  local_adjoint += theta * a[q];
                }
                psi_g(i, p + 1) = lambda_right + local_adjoint;
              }
            }
            psi_g(i, 0) = lambda_right + cell_adjoint;
          }
          lambda_right += cell_adjoint;
        });
  }

  athelas::par_for(
      DEFAULT_LOOP_PATTERN, "Gravity :: Update", DevExecSpace(), ib.s, ib.e,
      qb.s, qb.e, KOKKOS_CLASS_LAMBDA(const int i, const int q) {
        // Weak pressure operator applied to psi_g: the flux part
        // (interior - boundary) plus the shared curvature term psi_g * <phi_q,
        // d sqrt_gm/deta>. The hydro pressure flux applies the same operator,
        // so a hydrostatic state P = psi_g + const cancels through it exactly.
        const double boundary = phi(i, nNodes + 1, q) * sqrt_gm(i, nNodes + 1) *
                                    psi_g(i, nNodes + 1) -
                                phi(i, 0, q) * sqrt_gm(i, 0) * psi_g(i, 0);
        double interior = 0.0;
        for (int p = 0; p < nNodes; ++p) {
          interior += weights(p) * dphi(i, p + 1, q) * sqrt_gm(i, p + 1) *
                      psi_g(i, p + 1);
        }
        const double curvature =
            psi_g(i, q + 1) * geometric_weak_eta_derivative(
                                  phi, dphi, sqrt_gm, weights, i, q, nNodes);

        const double pressure_operator = interior - boundary + curvature;
        const double accel = -pressure_operator * inv_mkk(i, q);
        delta_(stage, i, q, pkg_vars::Velocity) = accel;

        // Filled below by the weak gravity-pressure energy companion.
        delta_(stage, i, q, pkg_vars::Energy) = 0.0;
      });

  // Weak gravity-pressure energy companion: the discrete product rule
  //   g w = d_m(A psi_g w) - psi_g dtau_dt,
  // assembled as (face - volume - dilatation). Summed over the nodal test
  // functions of a cell the face term telescopes (sum_q phi_q = 1 at each
  // face), the volume term vanishes identically (sum_q dphi_q = 0), and the
  // dilatation term reproduces Pi^T H taudot -- so the cell total is exactly
  // -dW_h/dt up to boundary work, and is invariant under a constant shift of
  // psi_g.
  athelas::par_for(
      DEFAULT_LOOP_PATTERN, "Gravity :: Energy companion", DevExecSpace(), ib.s,
      ib.e, qb.s, qb.e, KOKKOS_CLASS_LAMBDA(const int i, const int q) {
        const double w_left = interface(i, idx_vstar);
        const double w_right = interface(i + 1, idx_vstar);
        const double face = phi(i, nNodes + 1, q) * sqrt_gm(i, nNodes + 1) *
                                psi_g(i, nNodes + 1) * w_right -
                            phi(i, 0, q) * sqrt_gm(i, 0) * psi_g(i, 0) * w_left;

        double volume = 0.0;
        for (int p = 0; p < nNodes; ++p) {
          volume += weights(p) * dphi(i, p + 1, q) * sqrt_gm(i, p + 1) *
                    psi_g(i, p + 1) * velocity(i, p);
        }
        const double dilatation =
            weights(q) * dm_deta(i, q) * psi_g(i, q + 1) * dtau_dt(i, q);
        // The product-rule expression is a weak residual for test function q.
        // Convert it to the nodal energy RHS with the same inverse lumped mass
        // used by the gravity momentum operator above.  The legacy g*w branch
        // is already a pointwise (mass-inverted) RHS and must not be scaled.
        delta_(stage, i, q, pkg_vars::Energy) =
            (face - volume - dilatation) * inv_mkk(i, q);
      });
}

void GravityPackage::compute_mesh_velocity(const Mesh &mesh,
                                           AthelasArray2D<double> interface,
                                           AthelasArray2D<double> dtau_dt,
                                           const int idx_vstar) const {
  const int n_nodes = mesh.n_nodes();
  const IndexRange ib(mesh.domain<Domain::Interior>());
  const auto weights = mesh.weights();
  const auto nodes = mesh.nodes();
  const auto dm_deta = mesh.dm_deta();
  const auto enclosed_mass = mesh.enclosed_mass();
  const auto sqrt_gm = mesh.sqrt_gm();
  const auto r = mesh.nodal_grid();
  const auto high_order = mesh.node_placement_high_order();
  const bool spherical = model_ == GravityModel::Spherical;
  const auto velocity = tau_dot_mesh_velocity_;

  // The mesh reconstruction anchors the whole grid at the inner face, which is
  // advanced Lagrangianly by v*(ilo). Its volume-coordinate velocity is
  // sqrt_gm(ilo, 0) * v*(ilo) (r_inner^2 * v* spherical, v* planar).
  const int ilo = Mesh::get_ilo();
  const double x_dot_inner = sqrt_gm(ilo, 0) * interface(ilo, idx_vstar);

  athelas::par_scan(
      DEFAULT_FLAT_LOOP_PATTERN, "Gravity :: Mesh velocity", DevExecSpace(),
      ib.s, ib.e,
      KOKKOS_CLASS_LAMBDA(const int i, double &partial_x_dot,
                          const bool is_final) {
        // This is d/dt of the cell volume increment used by
        // Mesh::reconstruct_mesh.  Both reconstruction branches must use it:
        // it is not interchangeable with the jump in face volume velocity
        // until the discrete continuity identity happens to hold.
        double cell_x_dot = 0.0;
        for (int q = 0; q < n_nodes; ++q) {
          cell_x_dot += weights(q) * dm_deta(i, q) * dtau_dt(i, q);
        }

        if (is_final) {
          const bool ho = high_order(i) != 0;
          const double m_left = enclosed_mass(i, 0);
          const double m_width = enclosed_mass(i, n_nodes + 1) - m_left;
          for (int q = 0; q < n_nodes; ++q) {
            double x_dot = 0.0;
            if (ho) {
              // Monotone placement X_q = X_left + sum_p I(q,p) mu_p tau_p.
              // Differentiate in time (mu frozen): X_left contributes the inner
              // anchor plus the cross-cell prefix sum of cell dX rates.
              x_dot = x_dot_inner + partial_x_dot;
              for (int p = 0; p < n_nodes; ++p) {
                x_dot += mesh.integration_matrix(q, p) * dm_deta(i, p) *
                         dtau_dt(i, p);
              }
            } else {
              // Fallback placement X_q = X_left + theta_m (X_right - X_left)
              // with theta_m the clamped enclosed-mass fraction. Differentiate
              // that exact placement: both faces are reconstructed from tau,
              // so the local contribution is theta_m * dX_cell/dt.
              const double theta_m =
                  (m_width > 0.0)
                      ? std::clamp((enclosed_mass(i, q + 1) - m_left) / m_width,
                                   0.0, 1.0)
                      : nodes(q) + 0.5;
              x_dot = x_dot_inner + partial_x_dot + theta_m * cell_x_dot;
            }
            const double r_q = r(i, q + 1);
            velocity(i, q) = spherical ? x_dot / (r_q * r_q) : x_dot;
          }
        }

        // Cell increment of the reference-mass volume rate feeds the prefix sum
        // (telescopes to the left-face volume velocity of the next cell).
        partial_x_dot += cell_x_dot;
      });
}

/**
 * @brief apply gravity package delta
 */
void GravityPackage::apply_delta(AthelasArray3D<double> lhs,
                                 const TimeStepInfo &dt_info) const {
  static const int nx = static_cast<int>(lhs.extent(0));
  static const int nq = static_cast<int>(lhs.extent(1));
  static const IndexRange ib(std::make_pair(1, nx - 2));
  static const IndexRange qb(nq);
  static const IndexRange vb(NUM_VARS_);

  const int stage = dt_info.stage;

  athelas::par_for(
      DEFAULT_LOOP_PATTERN, "Gravity :: Apply delta", DevExecSpace(), ib.s,
      ib.e, qb.s, qb.e, KOKKOS_CLASS_LAMBDA(const int i, const int q) {
        for (int v = vb.s; v <= vb.e; ++v) {
          lhs(i, q, v + 1) += dt_info.dt_coef * delta_(stage, i, q, v);
        }
      });
}

auto GravityPackage::corrects_limiter_energy() const noexcept -> bool {
  return correct_limiter_energy_;
}

void GravityPackage::snapshot_limiter_radii(const Mesh &mesh) const {
  const int n_nodes = mesh.n_nodes();
  const IndexRange ib(mesh.domain<Domain::Interior>());
  const auto r = mesh.nodal_grid();
  const auto r_old = limiter_r_old_;
  athelas::par_for(
      DEFAULT_LOOP_PATTERN, "Gravity :: Snapshot limiter radii", DevExecSpace(),
      ib.s, ib.e, 0, n_nodes - 1,
      KOKKOS_CLASS_LAMBDA(const int i, const int q) {
        r_old(i, q) = r(i, q + 1);
      });
}

auto GravityPackage::apply_limiter_energy_correction(
    const StageData &stage_data, const Mesh &mesh,
    AthelasArray3D<double> evolved, const int idx_vel,
    const int idx_energy) const -> LimiterCorrection {
  const int n_nodes = mesh.n_nodes();
  const IndexRange ib(mesh.domain<Domain::Interior>());
  const auto weights = mesh.weights();
  const auto dm_deta = mesh.dm_deta();
  const auto enclosed_mass = mesh.enclosed_mass();
  const auto r = mesh.nodal_grid();
  const auto r_old = limiter_r_old_;
  const GravityModel model = model_;
  const double gval = gval_;
  constexpr int idx_tau = 0;

  // EOS floor for the clamp: min_sie(eos, rho, lambda). It is a value-typed
  // variant (safe to capture by value into a device kernel) and returns 0 for
  // ideal/polytropic EOS, or Paczynski::eps_min for the ionizing EOS -- which
  // needs the cell's electron fraction and ionization-energy correction, the
  // same lambda slots limit_internal_energy fills.
  const eos::EOS eos = stage_data.eos();
  const bool ionization = stage_data.enabled("ionization");
  AthelasArray2D<double> ye;
  AthelasArray2D<double> e_ion_corr;
  if (ionization) {
    ye = stage_data.comps()->ye();
    e_ion_corr = stage_data.ionization_state()->e_ion_corr();
  }

  double applied_total = 0.0;
  double clamp_total = 0.0;
  athelas::par_reduce(
      DEFAULT_FLAT_LOOP_PATTERN, "Gravity :: Limiter energy correction",
      DevExecSpace(), ib.s, ib.e,
      KOKKOS_CLASS_LAMBDA(const int i, double &applied, double &clamped) {
        // Potential energy the limiter moved (per cell, without 4pi), and the
        // cell reference mass.
        double dW = 0.0;
        double m_cell = 0.0;
        for (int q = 0; q < n_nodes; ++q) {
          const double r_new = r(i, q + 1);
          const double ro = r_old(i, q);
          const double m_enc = enclosed_mass(i, q + 1);
          // dphi_q = phi(r_new) - phi(r_old), with the same potential the
          // W_h diagnostic uses so the correction cancels the reported change.
          const double dphi =
              gravitational_potential(model, gval, m_enc, r_new) -
              gravitational_potential(model, gval, m_enc, ro);
          const double h = weights(q) * dm_deta(i, q);
          dW += h * dphi;
          m_cell += h;
        }
        if (m_cell <= 0.0) {
          return;
        }

        // Uniform specific-energy shift that returns -dW to the fluid. Only tau
        // moves the mesh, so this has no geometric feedback.
        const double dE = -dW / m_cell;

        // Clamp so the specific internal energy e = E - v^2/2 stays at or above
        // the EOS floor e_min at every node. Only limit a *removal* (dE < 0):
        // the headroom is the smallest nodal (e - e_min), and is zero if some
        // node is already at or below its floor -- never inject energy to lift a
        // pre-existing violation, which would run away.
        double headroom = std::numeric_limits<double>::max();
        for (int q = 0; q < n_nodes; ++q) {
          const double v = evolved(i, q, idx_vel);
          const double e = evolved(i, q, idx_energy) - 0.5 * v * v;
          const double rho_q = 1.0 / evolved(i, q, idx_tau);
          eos::EOSLambda lambda_q;
          if (ionization) {
            lambda_q.data[1] = ye(i, q);
            lambda_q.data[6] = e_ion_corr(i, q);
          }
          const double e_min_q = eos::min_sie(eos, rho_q, lambda_q.ptr());
          headroom = std::min(headroom, e - e_min_q);
        }
        headroom = std::max(headroom, 0.0);
        const double dE_applied = std::max(dE, -headroom);

        for (int q = 0; q < n_nodes; ++q) {
          evolved(i, q, idx_energy) += dE_applied;
        }
        applied += m_cell * dE_applied;
        clamped += m_cell * (dE - dE_applied);
      },
      Kokkos::Sum<double>(applied_total), Kokkos::Sum<double>(clamp_total));

  const double geom = mesh.do_geometry() ? constants::FOURPI : 1.0;
  return {geom * applied_total, geom * clamp_total};
}

/**
 * @brief zero delta field
 */
void GravityPackage::zero_delta() const noexcept {
  static const IndexRange sb(static_cast<int>(delta_.extent(0)));
  static const IndexRange ib(static_cast<int>(delta_.extent(1)));
  static const IndexRange qb(static_cast<int>(delta_.extent(2)));
  static const IndexRange vb(static_cast<int>(delta_.extent(3)));

  athelas::par_for(
      DEFAULT_LOOP_PATTERN, "Gravity :: Zero delta", DevExecSpace(), sb.s, sb.e,
      ib.s, ib.e, qb.s, qb.e,
      KOKKOS_CLASS_LAMBDA(const int s, const int i, const int q) {
        for (int v = vb.s; v <= vb.e; ++v) {
          delta_(s, i, q, v) = 0.0;
        }
      });
}

/**
 * @brief Gravitational timestep restriction
 **/
KOKKOS_FUNCTION
auto GravityPackage::min_timestep(const StageData &stage_data,
                                  const TimeStepInfo & /*dt_info*/) const
    -> double {
  static constexpr double MAX_DT = std::numeric_limits<double>::max() / 10.0;
  static constexpr double MIN_DT =
      100.0 * std::numeric_limits<double>::min() * 10.0;

  if (model_ == GravityModel::Constant) {
    return MAX_DT;
  }

  const auto &mesh = stage_data.mesh();
  static const IndexRange ib(mesh.domain<Domain::Interior>());
  auto dr = mesh.widths();
  auto r = mesh.centers();
  auto m = mesh.enclosed_mass();

  double dt_out = 0.0;
  athelas::par_reduce(
      DEFAULT_FLAT_LOOP_PATTERN, "Hydro :: Timestep", DevExecSpace(), ib.s,
      ib.e,
      KOKKOS_CLASS_LAMBDA(const int i, double &lmin) {
        const double dt_old =
            std::sqrt((r(i) * r(i) * dr(i)) / (constants::G_GRAV * m(i, 1)));

        lmin = std::min(dt_old, lmin);
      },
      Kokkos::Min<double>(dt_out));

  dt_out = std::max(cfl_ * dt_out, MIN_DT);
  dt_out = std::min(dt_out, MAX_DT);

  return dt_out;
}

void GravityPackage::fill_derived(StageData & /*stage_data*/,
                                  const TimeStepInfo & /*dt_info*/) const {}

[[nodiscard]] KOKKOS_FUNCTION auto GravityPackage::name() const noexcept
    -> std::string_view {
  return "Gravity";
}

[[nodiscard]] KOKKOS_FUNCTION auto GravityPackage::is_active() const noexcept
    -> bool {
  return active_;
}

[[nodiscard]] auto GravityPackage::model() const noexcept -> GravityModel {
  return model_;
}

[[nodiscard]] auto GravityPackage::gval() const noexcept -> double {
  return gval_;
}

KOKKOS_FUNCTION
void GravityPackage::set_active(const bool active) { active_ = active; }

} // namespace athelas::gravity
